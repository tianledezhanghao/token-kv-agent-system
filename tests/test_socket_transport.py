"""Socket frames, independent-process two-relay paths, faults and Session gates."""
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from uuid import uuid4

import pytest

from token_kv_agent_system.socket_lab import SocketLab
from token_kv_agent_system.transport import SocketByteTransport, SocketConfig, SocketTransferError
from token_kv_agent_system.transport.socket_service import ByteService
from token_kv_agent_system.transport.socket_wire import (
    HEADER_BYTES, PREFIX, REQUEST, VERIFIED, SocketCounters, WireError,
    encode_frame, recv_exact, recv_frame, send_all)
from token_kv_agent_system.token_protocol import serialize_token_ids, deserialize_token_packet


@contextmanager
def service(**kwargs):
    server = ByteService(('127.0.0.1', 0), **kwargs)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class PartialStream:
    """Force both short writes and short reads, independent of OS buffering."""
    def __init__(self, data=b'', limit=3):
        self.input = bytearray(data)
        self.output = bytearray()
        self.limit = limit
    def settimeout(self, seconds):
        assert seconds > 0
    def send(self, data):
        count = min(self.limit, len(data))
        self.output.extend(data[:count])
        return count
    def recv(self, size):
        count = min(self.limit, size, len(self.input))
        data = bytes(self.input[:count]); del self.input[:count]
        return data
    def close(self): pass


def test_forced_partial_io_frame_and_empty_payload():
    config = SocketConfig(chunk_bytes=7)
    identity = uuid4().bytes
    for payload in (b'', b'\x00\xff' * 1000):
        wire = encode_frame(payload, 'A', 'B', identity, REQUEST, config)
        stream, counts = PartialStream(wire), SocketCounters()
        send_all(stream, wire, config, time.monotonic()+3, counts)
        assert bytes(stream.output) == wire
        assert recv_frame(stream, config, time.monotonic()+3, counts) == (REQUEST, identity, 'A', 'B', payload)
        assert counts.sent_bytes == counts.received_bytes == len(wire)
        assert counts.send_calls == counts.recv_calls or abs(counts.send_calls - counts.recv_calls) <= 2


@pytest.mark.parametrize('fault', ['short_header', 'short_body', 'checksum', 'version', 'length', 'kind', 'header_bit'])
def test_corrupt_truncated_and_oversize_frames(fault):
    config = SocketConfig(max_payload_bytes=1024)
    wire = bytearray(encode_frame(b'hello', 'A', 'B', uuid4().bytes, REQUEST, config))
    if fault == 'short_header': wire = wire[:4]
    elif fault == 'short_body': wire = wire[:-1]
    elif fault == 'checksum': wire[-1] ^= 1
    elif fault == 'version': wire[4] = 99
    elif fault == 'kind': wire[5] = 99
    elif fault == 'header_bit': wire[6] ^= 1
    elif fault == 'length': wire[PREFIX.size-8:PREFIX.size] = (1<<48).to_bytes(8, 'big')
    with pytest.raises(WireError):
        recv_frame(PartialStream(wire), config, time.monotonic()+2, SocketCounters())


def test_endpoint_callback_observes_exact_bytes_before_ack_and_target_validation():
    deliveries = []
    with service(target='B', on_delivery=lambda *args: deliveries.append(args)) as endpoint:
        transport = SocketByteTransport({'B': endpoint.server_address, 'C': endpoint.server_address})
        assert transport.transfer(b'\x00opaque', source='A', target='B') == b'\x00opaque'
        assert deliveries[0][:3] == (b'\x00opaque', 'A', 'B')
        with pytest.raises(SocketTransferError):
            transport.transfer(b'wrong', source='A', target='C')
        assert len(deliveries) == len(transport.records) == 1
        assert transport.measurements[-1].outcome == 'delivery_unknown'


def test_no_route_refusal_and_invalid_input_never_fall_back():
    transport = SocketByteTransport({})
    with pytest.raises(KeyError): transport.transfer(b'x', source='A', target='B')
    with pytest.raises(TypeError): transport.transfer(bytearray(b'x'), source='A', target='B')
    with pytest.raises(ValueError): transport.transfer(b'x', source='A', target='A')
    with service(target='B') as endpoint:
        address = endpoint.server_address
    transport = SocketByteTransport({'B': address})
    with pytest.raises(SocketTransferError) as exc:
        transport.transfer(b'x', source='A', target='B')
    assert exc.value.measurement.outcome == 'not_sent'
    assert transport.records == ()
    assert len(transport.measurements) == 1 and transport.measurements[0].retries == 0


@contextmanager
def faulty_peer(action):
    listener = socket.socket(); listener.bind(('127.0.0.1', 0)); listener.listen()
    errors = []
    def run():
        try:
            connection, _ = listener.accept()
            with connection:
                action(connection)
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run, daemon=True); thread.start()
    try: yield listener.getsockname()
    finally:
        listener.close(); thread.join(timeout=2)
        assert all(isinstance(error, OSError) for error in errors), errors


@pytest.mark.parametrize('fault', ['disconnect', 'corrupt_response', 'timeout', 'wrong_id', 'wrong_payload', 'lost_ack'])
def test_failures_do_not_create_success_records(fault):
    observed = []
    def action(connection):
        if fault == 'disconnect': return
        kind, identity, a, b, data = recv_frame(connection, SocketConfig(), time.monotonic()+2, SocketCounters())
        observed.append(data)
        if fault == 'timeout': time.sleep(.15); return
        if fault == 'lost_ack': return
        response = encode_frame(b'other' if fault == 'wrong_payload' else data,
                                a, b, uuid4().bytes if fault == 'wrong_id' else identity,
                                VERIFIED, SocketConfig())
        if fault == 'corrupt_response': response = response[:-1] + bytes([response[-1] ^ 1])
        connection.sendall(response)
    with faulty_peer(action) as address:
        transport = SocketByteTransport({'B': address}, config=SocketConfig(timeout_seconds=.08))
        with pytest.raises(SocketTransferError): transport.transfer(b'test', source='A', target='B')
    assert transport.records == ()
    row = transport.measurements[-1]
    assert row.success is False and row.retries == 0
    assert row.outcome == 'delivery_unknown' and row.elapsed_ms > 0
    if fault == 'lost_ack': assert observed == [b'test']


def test_absolute_deadline_and_zero_send():
    with pytest.raises(TimeoutError):
        recv_exact(PartialStream(b'x'), 1, SocketConfig(), time.monotonic()-1, SocketCounters())
    stream = PartialStream(limit=0)
    with pytest.raises(WireError):
        send_all(stream, b'x', SocketConfig(), time.monotonic()+1, SocketCounters())


def test_explicit_config_file_and_payload_limit(tmp_path):
    path = tmp_path/'socket.json'
    path.write_text(json.dumps(dict(routes={'B': ['127.0.0.1', 1]}, socket={'max_payload_bytes': 2})))
    transport = SocketByteTransport.from_config(path)
    with pytest.raises(ValueError, match='limit'):
        transport.transfer(b'123', source='A', target='B')
    assert transport.records == () and transport.measurements == ()
    path.write_text('{"mode":"inmemory"}')
    with pytest.raises(ValueError): SocketByteTransport.from_config(path)


def test_relay_is_opaque_full_duplex_and_preserves_half_close():
    # Raw, non-TOKEN/non-KV/non-framed traffic; peer speaks BEFORE client sends.
    payload = bytes(range(256)) * 99
    def upstream(peer):
        peer.sendall(b'peer-first')
        received = bytearray()
        while part := peer.recv(79): received.extend(part)
        assert received == payload
        peer.sendall(b'after-half-close')
    with faulty_peer(upstream) as address, service(upstream=address, config=SocketConfig(chunk_bytes=73)) as relay:
        with socket.create_connection(relay.server_address, timeout=2) as client:
            assert recv_exact(client, 10, SocketConfig(), time.monotonic()+2, SocketCounters()) == b'peer-first'
            client.sendall(payload)
            client.shutdown(socket.SHUT_WR)
            assert recv_exact(client, 16, SocketConfig(), time.monotonic()+2, SocketCounters()) == b'after-half-close'
            assert client.recv(1) == b''


@pytest.mark.parametrize('fault', ['corrupt', 'oversize', 'timeout', 'reject'])
def test_endpoint_rejects_bad_requests_without_ack(fault):
    rows, seen = [], []
    def callback(*args):
        seen.append(args)
        if fault == 'reject': raise ValueError('application rejected bytes')
    config = SocketConfig(max_payload_bytes=16, timeout_seconds=.08)
    with service(target='B', config=config, on_delivery=callback, log=rows.append) as endpoint:
        with socket.create_connection(endpoint.server_address, timeout=2) as client:
            wire = encode_frame(b'test', 'A', 'B', uuid4().bytes, REQUEST, SocketConfig())
            if fault == 'corrupt': wire = wire[:-1] + b'!'
            if fault == 'oversize':
                wire = PREFIX.pack(b'TKVS', 1, REQUEST, uuid4().bytes, 2**40) + b'0'*32
            if fault == 'timeout': wire = wire[:1]
            client.sendall(wire)
            try:
                assert client.recv(1) == b''
            except ConnectionResetError:
                pass
    assert len(seen) == int(fault == 'reject')
    assert rows and rows[-1]['success'] is False


def test_relay_reports_idle_timeout():
    rows = []
    with faulty_peer(lambda peer: time.sleep(.15)) as address:
        with service(upstream=address, config=SocketConfig(timeout_seconds=.05), log=rows.append) as relay:
            with socket.create_connection(relay.server_address, timeout=2) as client:
                assert client.recv(1) == b''
    assert rows[-1]['success'] is False and 'timeout' in rows[-1]['error']


def test_relay_simultaneous_large_streams_with_backpressure():
    payload = bytes(range(256)) * 16384
    def exchange(connection):
        errors = []
        def send():
            try:
                connection.sendall(payload)
                connection.shutdown(socket.SHUT_WR)
            except OSError as exc:
                errors.append(exc)
        sender = threading.Thread(target=send, daemon=True)
        sender.start()
        connection.settimeout(5)
        received = bytearray()
        while part := connection.recv(1597): received.extend(part)
        sender.join(timeout=5)
        assert not sender.is_alive() and not errors
        assert received == payload
    with faulty_peer(exchange) as address, service(upstream=address, config=SocketConfig(chunk_bytes=4093)) as relay:
        with socket.create_connection(relay.server_address, timeout=5) as client:
            exchange(client)


@pytest.mark.parametrize('kwargs', [{'chunk_bytes': 0}, {'chunk_bytes': 2**30}, {'max_payload_bytes': -1},
                                   {'timeout_seconds': float('nan')}, {'timeout_seconds': 0}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError): SocketConfig(**kwargs)


def test_two_relays_bidirectional_independent_sender_processes(tmp_path):
    payloads = [serialize_token_ids((0, 42, 151935)), bytes(range(256)) * 32768]
    with SocketLab(tmp_path / 'logs', config=SocketConfig(chunk_bytes=997)) as lab:
        assert len({p.pid for p in lab.processes}) == 4
        assert all(p.poll() is None for p in lab.processes)
        for source, target in [('A', 'B'), ('B', 'A')]:
            for index, packet in enumerate(payloads):
                path, output = tmp_path / 'packet', tmp_path / 'received'
                path.write_bytes(packet)
                host, port = lab.routes[target]
                result = subprocess.run([sys.executable, str(Path(__file__).with_name('socket_agent_worker.py')),
                    str(path), source, target, host, str(port), str(output)],
                    capture_output=True, text=True, timeout=30, check=True)
                assert output.read_bytes() == packet
                if index == 0: assert deserialize_token_packet(output.read_bytes()).token_ids == (0, 42, 151935)
                measurement = json.loads(result.stdout)[0]
                assert measurement['success'] and measurement['sent_bytes'] > len(packet)
                assert measurement['received_bytes'] == measurement['sent_bytes']
                assert measurement['send_calls'] >= 1
                if index: assert measurement['send_calls'] > 1
    for name in ('relay-a', 'relay-b'):
        rows = [json.loads(line) for line in (tmp_path/'logs'/f'{name}.jsonl').read_text().splitlines()]
        done = [r for r in rows if r['event'] == 'connection_finished']
        assert len(done) == 4 and all(r['success'] for r in done)
        assert sum(r['sent_bytes'] for r in done) > 2 * sum(map(len, payloads))


def test_break_relay_no_hidden_fallback(tmp_path):
    with SocketLab(tmp_path) as lab:
        lab.processes[-1].terminate(); lab.processes[-1].wait(timeout=3)
        with pytest.raises(SocketTransferError): lab.transport.transfer(b'packet', source='A', target='B')
        assert lab.transport.records == ()


@pytest.mark.parametrize('source,target', [('A', 'B'), ('B', 'A')])
def test_independent_agent_receiver_consumes_protocol_ids(tmp_path, source, target):
    # Both source and consuming receiver are separate processes. The receiver
    # actually deserializes original TOKEN bytes and persists the delivered IDs;
    # correctness is checked independently of the returned verification payload.
    import select
    worker = str(Path(__file__).with_name('socket_agent_worker.py'))
    captured = tmp_path/'receiver-state.json'
    receiver = subprocess.Popen([sys.executable, worker, '--serve', target, str(captured)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lab = SocketLab(tmp_path)
    try:
        assert select.select([receiver.stdout], [], [], 5)[0], 'Agent receiver startup timeout'
        endpoint = json.loads(receiver.stdout.readline())['listen']
        rb = lab.start('relay-b', '--listen', '127.0.0.1:0', '--forward', f'{endpoint[0]}:{endpoint[1]}')
        hop_b = rb['listeners'][0]['listen']
        ra = lab.start('relay-a', '--listen', '127.0.0.1:0', '--forward', f'{hop_b[0]}:{hop_b[1]}')
        ingress = ra['listeners'][0]['listen']
        wire = serialize_token_ids((22, 151645))
        packet, returned = tmp_path/'packet', tmp_path/'returned'
        packet.write_bytes(wire)
        result = subprocess.run([sys.executable, worker, str(packet), source, target,
                                 ingress[0], str(ingress[1]), str(returned)],
                                capture_output=True, text=True, timeout=10, check=True)
        state = json.loads(captured.read_text())
        assert state['source'] == source and state['target'] == target
        assert state['received_ids'] == [22, 151645]
        assert state['packet_sha256'] == hashlib.sha256(wire).hexdigest()
        assert state['transfer_id'] == json.loads(result.stdout)[0]['transfer_id']
        assert returned.read_bytes() == wire
    finally:
        lab.__exit__(None, None, None)
        receiver.terminate()
        receiver.communicate(timeout=5)


def test_large_kv_protocol_packets_all_four_strategies_bidirectional(tmp_path):
    from token_kv_agent_system.incremental_kv_protocol import (
        ModelContract, IncrementalKVMessage, encode_incremental_message, decode_incremental_message)
    from token_kv_agent_system.kv_protocol import (
        KVModelCompatibility, NeutralKVPayload, NeutralKVTensor, NeutralKVLayer)
    # V0–V3 deliberately share the exact existing packet schema; strategy is
    # receiver policy, not a rewritten message. The small Session test executes
    # each policy; here each path carries a 2+ MiB standards-compliant envelope.
    count = 2048
    tensor = NeutralKVTensor('bfloat16', (1, 2, count, 64), bytes(range(256)) * 2048)
    delta = NeutralKVPayload(KVModelCompatibility('fixture/large', 2, 2, 64), count, 'cpu',
                            tuple(NeutralKVLayer(i, tensor, tensor) for i in range(2)))
    contract = ModelContract('fixture/large', 'a'*64, 'b'*64, 'c'*64,
                            256, 2, 2, 64, 'bfloat16', (2,), (1, 2, 3))
    with SocketLab(tmp_path, config=SocketConfig(chunk_bytes=16381)) as lab:
        for mode in ('kv:v0', 'kv:v1', 'kv:v2', 'kv:v3'):
            for source, target in (('A', 'B'), ('B', 'A')):
                message = IncrementalKVMessage(f'{source}:1', 1, source, target, (59,)*count,
                                              count, 'max_new_tokens', None, 71, count, contract, delta)
                wire = encode_incremental_message(message)
                assert len(wire) > 2 * 1024**2
                delivered = lab.transport.transfer(wire, source=source, target=target)
                assert delivered == wire, mode
                assert decode_incremental_message(delivered, expected_contract=contract,
                    source=source, target=target, sequence=1) == message
        assert len(lab.transport.records) == 8


def test_byte_ack_does_not_fake_model_commit():
    from test_incremental_kv_transport import make_backend
    from token_kv_agent_system.agent import LLMAgent
    from token_kv_agent_system.stateful_token import StatefulTokenSession
    backend = make_backend()
    with service(target='B') as endpoint:
        transport = SocketByteTransport({'B': endpoint.server_address})
        session = StatefulTokenSession([
            LLMAgent('A', backend, system_prompt='You are A.'),
            LLMAgent('B', backend, system_prompt='You are B.')], transport)
        backend.outputs = [100, 101, 2]
        session.generate_message('A', 'B', max_new_tokens=3)
        old = dict(session._states)
        def fail_import(*args, **kwargs): raise RuntimeError('receiver import failed after bytes ACK')
        backend.continue_from_cache = fail_import
        with pytest.raises(RuntimeError, match='after bytes ACK'):
            session.handoff('A', 'B')
        assert transport.measurements[-1].outcome == 'bytes_verified'
        assert transport.measurements[-1].acknowledgement == 'remote_bytes_verified_only'
        assert all(session.state(n) is old[n] for n in old)
        assert not session.state('B').received_messages


@pytest.mark.parametrize('mode', ['token', 'kv:v0', 'kv:v1', 'kv:v2', 'kv:v3'])
def test_socket_session_bidirectional_import_and_continued_decode(tmp_path, mode):
    from test_incremental_kv_transport import make_backend
    from token_kv_agent_system.agent import LLMAgent
    from token_kv_agent_system.stateful_token import StatefulTokenSession
    from token_kv_agent_system.communication import HandoffChoice
    backend = make_backend()
    choice = HandoffChoice.parse(mode)
    with SocketLab(tmp_path, config=SocketConfig(chunk_bytes=31)) as lab:
        session = StatefulTokenSession([LLMAgent('A', backend, system_prompt='You are A.'), LLMAgent('B', backend, system_prompt='You are B.')], lab.transport)
        for source, target in [('A', 'B'), ('B', 'A')]:
            backend.outputs = [59, 100, 2]  # TinyTokenizer ASCII '1' at 59; applicable V3 slot.
            message = session.generate_message(source, target, max_new_tokens=3)
            receipt = (session.handoff(source, target) if mode == 'token' else
                       session.handoff_kv(source, target, strategy=choice.kv_strategy))
            assert receipt.received_token_ids == message.message_token_ids
            assert session.state(target).received_messages[-1].message_id == message.message_id
            for name in ('A', 'B'):
                session.state(name).validate()
                assert session.state(name).cached_token_count == len(session.state(name).token_ledger)
        # Both previous receiver caches support another actual tiny-model Decode.
        assert session.generate_local_message('A', local_prompt='Continue', max_new_tokens=2).token_count == 2
        assert len(lab.transport.records) == 2


@pytest.mark.parametrize('mode', ['token', 'kv:v0'])
def test_transport_failure_preserves_session_transaction_and_draft(mode):
    from test_incremental_kv_transport import make_backend
    from token_kv_agent_system.agent import LLMAgent
    from token_kv_agent_system.stateful_token import StatefulTokenSession
    from token_kv_agent_system.communication import HandoffChoice
    backend = make_backend()
    with faulty_peer(lambda c: None) as address:
        transport = SocketByteTransport({'B': address})
        session = StatefulTokenSession([LLMAgent('A', backend, system_prompt='You are A.'), LLMAgent('B', backend, system_prompt='You are B.')], transport)
        backend.outputs = [100, 101, 2]
        session.generate_message('A', 'B', max_new_tokens=3)
        old = {name: session.state(name) for name in ('A', 'B')}
        with pytest.raises(SocketTransferError):
            if mode == 'token': session.handoff('A', 'B')
            else: session.handoff_kv('A', 'B', strategy=HandoffChoice.parse(mode).kv_strategy)
        assert all(session.state(n) is old[n] for n in old)
        assert session.state('A').draft_message is not None
        assert not session.state('B').received_messages and not transport.records
