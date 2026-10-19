"""Application delivery over the 8.6A bounded TCP frames and opaque relays.

VERIFIED here contains an explicit application receipt, never an echoed payload.
The compatibility SocketByteTransport and byte endpoints retain their semantics.
"""
from dataclasses import asdict
import hashlib
import json
import struct
import time
from uuid import uuid4

from .incremental_kv_protocol import canonical_json
from .transport.socket_transport import SocketMeasurement
from .transport.socket_wire import (SocketConfig, SocketCounters, REQUEST, VERIFIED,
    connect_tcp, encode_frame, recv_frame, send_all, remaining, WireError)

MAGIC = b'TKWA\x01'
MAX_METADATA = 64 * 1024
RECEIPT_SCHEMA = 'remote_agent_commit_v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pack_delivery(metadata, protocol_bytes):
    header = canonical_json(metadata)
    if len(header) > MAX_METADATA:
        raise ValueError('application metadata exceeds limit')
    return MAGIC + struct.pack('!I', len(header)) + header + protocol_bytes


def unpack_delivery(data):
    if len(data) < 9 or data[:5] != MAGIC:
        raise ValueError('invalid remote Agent envelope')
    size = struct.unpack_from('!I', data, 5)[0]
    if not 0 < size <= MAX_METADATA or 9 + size >= len(data):
        raise ValueError('invalid application metadata size')
    metadata = json.loads(data[9:9+size])
    if not isinstance(metadata, dict):
        raise ValueError('invalid application metadata')
    return metadata, data[9+size:]


class RemoteAgentTransport:
    """One attempt, no echo, retry, or fallback. Unknown results require query."""
    def __init__(self, routes, *, config=SocketConfig(), connector=connect_tcp):
        self.routes, self.config, self.connector = dict(routes), config, connector
        self.measurements = []

    def deliver(self, data, *, source, target, expected):
        identity = uuid4().bytes
        frame = encode_frame(data, source, target, identity, REQUEST, self.config)
        address = self.routes[target]  # Missing route fails before sending.
        counters = SocketCounters()
        started = time.monotonic()
        deadline = started + self.config.timeout_seconds
        stream, receipt, error = None, None, None
        outcome = 'not_sent'
        try:
            stream = self.connector(address, remaining(deadline))
            send_all(stream, frame, self.config, deadline, counters)
            kind, echoed, a, b, response = recv_frame(stream, self.config, deadline, counters)
            if (kind, echoed, a, b) != (VERIFIED, identity, source, target):
                raise WireError('application acknowledgement transport identity mismatch')
            receipt = json.loads(response)
            if (not isinstance(receipt, dict) or receipt.get('schema') != RECEIPT_SCHEMA
                    or any(receipt.get(k) != v for k, v in expected.items())
                    or receipt.get('status') not in ('committed', 'rejected')):
                raise WireError('application acknowledgement identity/status mismatch')
            if receipt['status'] == 'committed' and not (
                    receipt.get('cache_imported') is True and receipt.get('state_committed') is True):
                raise WireError('missing Receiver model commit confirmation')
            outcome = 'receiver_committed' if receipt['status'] == 'committed' else 'receiver_rejected'
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            outcome = 'delivery_unknown' if counters.sent_bytes else 'not_sent'
            receipt = None
        finally:
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass  # A validated application receipt remains valid after close.
        measured = SocketMeasurement(identity.hex(), source, target, len(data), **asdict(counters),
            elapsed_ms=(time.monotonic()-started)*1000, success=outcome == 'receiver_committed',
            outcome=outcome, error=error, acknowledgement=RECEIPT_SCHEMA)
        self.measurements.append(measured)
        return dict(outcome=outcome, receipt=receipt, measurement=asdict(measured), error=error)
