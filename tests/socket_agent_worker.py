"""Separate sending/receiving Agent fixture processes for socket acceptance."""
import json
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys

from token_kv_agent_system.transport import SocketByteTransport, SocketConfig
from token_kv_agent_system.transport.socket_service import ByteService
from token_kv_agent_system.token_protocol import deserialize_token_packet


if __name__ == '__main__':
    if sys.argv[1] == '--serve':
        target, output = sys.argv[2:]
        def receive(data, source, target, transfer_id):
            ids = deserialize_token_packet(data).token_ids
            Path(output).write_text(json.dumps(dict(source=source, target=target,
                received_ids=ids, packet_sha256=hashlib.sha256(data).hexdigest(), transfer_id=transfer_id)))
        with ByteService(('127.0.0.1', 0), target=target, on_delivery=receive) as server:
            print(json.dumps(dict(listen=server.server_address)), flush=True)
            server.serve_forever(poll_interval=.1)
        raise SystemExit(0)
    packet, source, target, host, port, output = sys.argv[1:]
    transport = SocketByteTransport({target: (host, int(port))}, config=SocketConfig(chunk_bytes=8191))
    received = transport.transfer(Path(packet).read_bytes(), source=source, target=target)
    Path(output).write_bytes(received)
    print(json.dumps([asdict(m) for m in transport.measurements]))
