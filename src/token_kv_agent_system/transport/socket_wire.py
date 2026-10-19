"""Bounded transport framing, independent of TOKEN/KV model protocols."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import socket
import struct
import time
from typing import Protocol


class ByteStream(Protocol):
    """Future SDK/serial adapters implement partial I/O and finite timeouts.

    IP devices can use socket.socket directly. No device MTU or driver assumed.
    A connector owns opening/closing the stream; framing owns message integrity.
    """
    def send(self, data: memoryview) -> int: ...
    def recv(self, size: int) -> bytes: ...
    def settimeout(self, seconds: float) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class SocketConfig:
    max_payload_bytes: int = 64 * 1024**2
    chunk_bytes: int = 64 * 1024
    timeout_seconds: float = 30.0

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.max_payload_bytes, self.chunk_bytes)):
            raise ValueError('payload and chunk bounds must be positive integers')
        if self.max_payload_bytes > (1 << 63) or self.chunk_bytes > 1024**2:
            raise ValueError('payload/chunk bound too large')
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError('timeout must be finite and positive')


@dataclass
class SocketCounters:
    sent_bytes: int = 0
    received_bytes: int = 0
    send_calls: int = 0
    recv_calls: int = 0


class WireError(OSError):
    pass


PREFIX = struct.Struct('!4sBB16sQ')
HEADER_BYTES = PREFIX.size + 32
NAMES = struct.Struct('!HH')
REQUEST, VERIFIED = 1, 2
MAX_NAME_BYTES = 128


def remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError('socket operation deadline exceeded')
    return value


def send_all(stream: ByteStream, data: bytes, config: SocketConfig,
             deadline: float, counters: SocketCounters) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        stream.settimeout(remaining(deadline))
        count = stream.send(view[offset:offset + config.chunk_bytes])
        counters.send_calls += 1
        if count <= 0:
            raise WireError('stream closed during send')
        counters.sent_bytes += count
        offset += count


def recv_exact(stream: ByteStream, size: int, config: SocketConfig,
               deadline: float, counters: SocketCounters) -> bytes:
    result = bytearray()
    while len(result) < size:
        stream.settimeout(remaining(deadline))
        part = stream.recv(min(size - len(result), config.chunk_bytes))
        counters.recv_calls += 1
        if not part:
            raise WireError('stream closed during receive')
        counters.received_bytes += len(part)
        result.extend(part)
    return bytes(result)


def validate_names(source: str, target: str) -> tuple[bytes, bytes]:
    if not isinstance(source, str) or not isinstance(target, str):
        raise TypeError('endpoint names must be strings')
    names = source.encode('utf-8'), target.encode('utf-8')
    if source == target or any(not n.strip() or len(n) > MAX_NAME_BYTES for n in names):
        raise ValueError('distinct nonempty endpoint names of at most 128 UTF-8 bytes required')
    return names


def encode_frame(data: bytes, source: str, target: str, transfer_id: bytes,
                 kind: int, config: SocketConfig) -> bytes:
    if not isinstance(data, bytes):
        raise TypeError('byte transport data must be bytes')
    if len(data) > config.max_payload_bytes:
        raise ValueError('payload exceeds configured limit')
    a, b = validate_names(source, target)
    body = NAMES.pack(len(a), len(b)) + a + b + data
    prefix = PREFIX.pack(b'TKVS', 1, kind, transfer_id, len(body))
    return prefix + hashlib.sha256(prefix + body).digest() + body


def recv_frame(stream: ByteStream, config: SocketConfig, deadline: float,
               counters: SocketCounters) -> tuple[int, bytes, str, str, bytes]:
    header = recv_exact(stream, HEADER_BYTES, config, deadline, counters)
    magic, version, kind, transfer_id, size = PREFIX.unpack(header[:PREFIX.size])
    if magic != b'TKVS' or version != 1 or kind not in (REQUEST, VERIFIED):
        raise WireError('invalid transport header')
    if not NAMES.size + 2 <= size <= config.max_payload_bytes + NAMES.size + 2 * MAX_NAME_BYTES:
        raise WireError('transport frame length exceeds bounds')
    body = recv_exact(stream, size, config, deadline, counters)
    if hashlib.sha256(header[:PREFIX.size] + body).digest() != header[PREFIX.size:]:
        raise WireError('transport checksum mismatch')
    na, nb = NAMES.unpack_from(body)
    if not 0 < na <= MAX_NAME_BYTES or not 0 < nb <= MAX_NAME_BYTES or NAMES.size + na + nb > len(body):
        raise WireError('invalid endpoint lengths')
    try:
        source = body[NAMES.size:NAMES.size + na].decode('utf-8')
        target = body[NAMES.size + na:NAMES.size + na + nb].decode('utf-8')
        validate_names(source, target)
    except (ValueError, UnicodeError) as exc:
        raise WireError('invalid endpoint names') from exc
    data = body[NAMES.size + na + nb:]
    if len(data) > config.max_payload_bytes:
        raise WireError('payload exceeds configured limit')
    return kind, transfer_id, source, target, data


def connect_tcp(address: tuple[str, int], timeout: float) -> socket.socket:
    return socket.create_connection(address, timeout=timeout)
