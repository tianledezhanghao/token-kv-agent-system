"""Explicit socket bridge for the existing synchronous shared-weight Sessions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Mapping
from uuid import uuid4

from .byte_transport import BaseByteTransport, ByteTransferRecord
from .socket_wire import (ByteStream, SocketConfig, SocketCounters, REQUEST, VERIFIED,
                          WireError, connect_tcp, encode_frame, recv_frame, remaining, send_all,
                          validate_names)


@dataclass(frozen=True)
class SocketMeasurement:
    transfer_id: str
    source: str
    target: str
    payload_bytes: int
    sent_bytes: int
    received_bytes: int
    send_calls: int
    recv_calls: int
    elapsed_ms: float
    success: bool
    outcome: str
    error: str | None
    retries: int = 0
    measurement_kind: str = 'measured_socket_application_io'
    acknowledgement: str = 'remote_bytes_verified_only'


class SocketTransferError(OSError):
    def __init__(self, measurement: SocketMeasurement):
        super().__init__(f'{measurement.outcome}: {measurement.error}')
        self.measurement = measurement


class SocketByteTransport(BaseByteTransport):
    """Send to a target endpoint through explicitly supplied routes.

    Returns bytes actually read back from the remote verifier. This compatibility
    bridge lets current Mac Sessions keep both caches and their local transaction.
    The remote ACK does NOT mean model import/Decode or distributed commit.
    No automatic retry: a lost ACK can leave delivery outcome unknown.
    """
    def __init__(self, routes: Mapping[str, tuple[str, int]], *,
                 config: SocketConfig = SocketConfig(),
                 connector: Callable[[tuple[str, int], float], ByteStream] = connect_tcp,
                 measurement_sink: Callable[[dict], None] | None = None):
        self.routes = dict(routes)
        for host, port in self.routes.values():
            if not host or type(port) is not int or not 0 < port < 65536:
                raise ValueError('invalid socket route')
        self.config, self.connector = config, connector
        self.measurement_sink = measurement_sink
        self._records: list[ByteTransferRecord] = []
        self._measurements: list[SocketMeasurement] = []

    @classmethod
    def from_config(cls, path: str | Path, **kwargs) -> 'SocketByteTransport':
        """Explicit opt-in file; malformed/missing files fail without fallback."""
        settings = json.loads(Path(path).read_text(encoding='utf-8'))
        if set(settings) != {'routes', 'socket'}:
            raise ValueError('socket configuration requires exactly routes and socket')
        return cls(settings['routes'], config=SocketConfig(**settings['socket']), **kwargs)

    @property
    def records(self) -> tuple[ByteTransferRecord, ...]:
        return tuple(self._records)

    @property
    def measurements(self) -> tuple[SocketMeasurement, ...]:
        return tuple(self._measurements)

    def transfer(self, data: bytes, *, source: str, target: str) -> bytes:
        validate_names(source, target)
        transfer_id = uuid4().bytes
        frame = encode_frame(data, source, target, transfer_id, REQUEST, self.config)
        # Missing route is a configuration error, never an in-memory fallback.
        address = self.routes[target]
        counters, started = SocketCounters(), time.monotonic()
        deadline = started + self.config.timeout_seconds
        stream, error, delivered = None, None, None
        try:
            stream = self.connector(address, remaining(deadline))
            send_all(stream, frame, self.config, deadline, counters)
            kind, echoed_id, a, b, delivered = recv_frame(stream, self.config, deadline, counters)
            if (kind, echoed_id, a, b) != (VERIFIED, transfer_id, source, target):
                raise WireError('remote acknowledgement identity mismatch')
            if hashlib.sha256(delivered).digest() != hashlib.sha256(data).digest():
                raise WireError('remote payload identity mismatch')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            if stream is not None:
                try:
                    stream.close()
                except OSError as exc:
                    error = error or f'close failed: {exc}'
        success = error is None
        measurement = SocketMeasurement(transfer_id.hex(), source, target, len(data),
            **asdict(counters), elapsed_ms=(time.monotonic() - started) * 1000,
            success=success, outcome=('bytes_verified' if success else
                'delivery_unknown' if counters.sent_bytes else 'not_sent'), error=error)
        self._measurements.append(measurement)
        if self.measurement_sink:
            self.measurement_sink(asdict(measurement))
        if not success:
            raise SocketTransferError(measurement)
        self._records.append(ByteTransferRecord(source, target, len(delivered)))
        return delivered
