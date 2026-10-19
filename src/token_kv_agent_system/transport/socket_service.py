"""Bounded TCP services: opaque full-duplex relay and remote byte verifier."""
from __future__ import annotations

from dataclasses import asdict
import os
import select
import socket
import socketserver
import threading
import time
from typing import Callable

from .socket_wire import (REQUEST, VERIFIED, SocketConfig, SocketCounters, WireError,
                          connect_tcp, encode_frame, recv_frame, send_all)


class ByteService(socketserver.ThreadingTCPServer):
    """Finite concurrent connections and bounded per-connection buffers.

    Endpoint callback receives opaque bytes. It may reject before ACK. It must
    enforce its own application deadline/transaction if it mutates model state.
    Default callback is a verifier only; no model/packet parser is imported here.
    """
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False

    def __init__(self, address: tuple[str, int], *, config: SocketConfig = SocketConfig(),
                 upstream: tuple[str, int] | None = None, target: str | None = None,
                 on_delivery: Callable[[bytes, str, str, str], None] | None = None,
                 log: Callable[[dict], None] | None = None, max_connections: int = 8):
        if (upstream is None) == (target is None):
            raise ValueError('choose either relay upstream or endpoint target')
        if type(max_connections) is not int or max_connections <= 0:
            raise ValueError('max_connections must be positive')
        self.config, self.upstream, self.target = config, upstream, target
        self.on_delivery, self.log = on_delivery, log
        self._slots = threading.BoundedSemaphore(max_connections)
        self._active: set[socket.socket] = set()
        self._active_lock = threading.Lock()
        super().__init__(address, _Handler)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            request.close()
            self.emit(dict(event='connection_rejected', error='concurrency_limit'))
            return
        with self._active_lock:
            self._active.add(request)
        try:
            super().process_request(request, client_address)
        except Exception:
            self.release(request)
            raise

    def release(self, request):
        with self._active_lock:
            self._active.discard(request)
        self._slots.release()

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.release(request)

    def server_close(self):
        # Closing the listener must also release blocked downstream clients.
        with self._active_lock:
            for connection in self._active:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        super().server_close()

    def emit(self, row):
        if self.log:
            self.log(dict(pid=os.getpid(), role='relay' if self.upstream else 'byte_endpoint',
                          listen=self.server_address, upstream=self.upstream, **row))


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        server = self.server
        started = time.monotonic()
        counters = SocketCounters()
        error, transfer_id = None, None
        try:
            if server.upstream:
                _relay(self.request, server.upstream, server.config, counters)
            else:
                deadline = started + server.config.timeout_seconds
                kind, identity, source, target, data = recv_frame(self.request, server.config, deadline, counters)
                transfer_id = identity.hex()
                if kind != REQUEST or target != server.target:
                    raise WireError('incorrect endpoint target or frame kind')
                if server.on_delivery:
                    server.on_delivery(data, source, target, transfer_id)
                # The response carries exactly the bytes received, never regenerated model messages.
                reply = encode_frame(data, source, target, identity, VERIFIED, server.config)
                send_all(self.request, reply, server.config, deadline, counters)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            server.emit(dict(event='connection_finished', transfer_id=transfer_id,
                peer=self.client_address,
                **asdict(counters), elapsed_ms=(time.monotonic() - started) * 1000,
                success=error is None, error=error,
                success_scope='stream_forwarding_only' if server.upstream else 'byte_verification_only',
                retries=0, measurement_kind='measured_socket_application_io'))


def _relay(client: socket.socket, upstream: tuple[str, int], config: SocketConfig,
           counters: SocketCounters):
    """Forward raw stream chunks in both directions; no frame/model inspection.

    Nonblocking partial reads/writes allow both directions to progress under
    backpressure. Buffering is bounded to one chunk per direction. EOF drains its
    buffer before half-closing the peer; idle deadlines terminate stalled links.
    """
    with connect_tcp(upstream, config.timeout_seconds) as peer:
        destinations = {client: peer, peer: client}
        readers, half_closed = {client, peer}, set()
        buffers = {client: bytearray(), peer: bytearray()}
        for stream in destinations:
            stream.setblocking(False)
        last_progress = time.monotonic()
        while readers or any(buffers.values()):
            for source, destination in destinations.items():
                if source not in readers and not buffers[destination] and destination not in half_closed:
                    destination.shutdown(socket.SHUT_WR)
                    half_closed.add(destination)
            timeout = config.timeout_seconds - (time.monotonic() - last_progress)
            if timeout <= 0:
                raise TimeoutError('relay idle timeout')
            ready, writable, _ = select.select(
                [s for s in readers if not buffers[destinations[s]]],
                [s for s in buffers if buffers[s]], [], timeout)
            if not ready and not writable:
                raise TimeoutError('relay idle timeout')
            for source in ready:
                try:
                    part = source.recv(config.chunk_bytes)
                except (BlockingIOError, InterruptedError):
                    continue
                counters.recv_calls += 1
                last_progress = time.monotonic()
                if not part:
                    readers.remove(source)
                    continue
                counters.received_bytes += len(part)
                buffers[destinations[source]].extend(part)
            for destination in writable:
                try:
                    count = destination.send(buffers[destination])
                except (BlockingIOError, InterruptedError):
                    continue
                counters.send_calls += 1
                if count <= 0:
                    raise WireError('relay stream closed during send')
                counters.sent_bytes += count
                del buffers[destination][:count]
                last_progress = time.monotonic()
        for destination in destinations:
            if destination not in half_closed:
                destination.shutdown(socket.SHUT_WR)
