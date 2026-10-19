"""Dependency-free envelope, corruption, compatibility and bounded parsing tests."""
from dataclasses import replace
import hashlib
import json
from struct import Struct

import pytest

from token_kv_agent_system.incremental_kv import IncrementalKVReceiver
from token_kv_agent_system.incremental_kv_protocol import (
    IncrementalKVMessage, IncrementalLimits, ModelContract, canonical_json,
    decode_incremental_message, encode_incremental_message)
from token_kv_agent_system.kv_protocol import (
    KVModelCompatibility, NeutralKVLayer, NeutralKVPayload, NeutralKVTensor,
    deserialize_kv_packet, serialize_kv_payload)


@pytest.fixture
def message():
    contract = ModelContract('test/checkpoint', 'a'*64, 'b'*64, 'c'*64,
                             256, 2, 1, 2, 'bfloat16', (2, 3), (1, 2, 3))
    # Preserve arbitrary BF16 bits, including -0 and NaN (allclose is unsuitable).
    k = NeutralKVTensor('bfloat16', (1, 1, 2, 2), bytes.fromhex('0080c17f01000200'))
    v = replace(k, data=bytes.fromhex('017f0000008000ff'))
    delta = NeutralKVPayload(KVModelCompatibility('test/checkpoint', 2, 1, 2), 2, 'cpu',
                             (NeutralKVLayer(0, k, v), NeutralKVLayer(1, v, k)))
    return IncrementalKVMessage('A:1', 1, 'A', 'B', (100, 101), 2, 'eos', 2, 71, 2, contract, delta)


def decode(packet, message):
    return decode_incremental_message(packet, expected_contract=message.model_contract,
                                      source='A', target='B', sequence=1)


def repack_metadata(packet, mutate):
    header = Struct('!4sBIII')
    magic, version, nm, nt, nk = header.unpack_from(packet)
    meta = json.loads(packet[header.size:header.size+nm])
    mutate(meta)
    encoded = canonical_json(meta)
    body = header.pack(magic, version, len(encoded), nt, nk) + encoded + packet[header.size+nm:-32]
    return body + hashlib.sha256(body).digest()


def test_exact_roundtrip_nested_unmodified_v1_and_raw_ids(message):
    wire = encode_incremental_message(message)
    assert decode(wire, message) == message
    assert message.raw_generated_token_ids == (100, 101, 2)
    assert b'TKVT' in wire and b'TKVK' in wire
    kv = serialize_kv_payload(message.delta)
    assert deserialize_kv_packet(kv.packet).payload == message.delta
    assert all(a.key.data == b.key.data and a.value.data == b.value.data
               for a, b in zip(message.delta.layers, decode(wire, message).delta.layers))


@pytest.mark.parametrize('change', [
    {'message_token_count': 3}, {'sender_span_length': 3}, {'sender_span_start': -1},
    {'sender_span_length': 0}, {'sequence': 0}, {'sequence': True}, {'message_id': 'fake'},
    {'message_token_ids': (100, 256)}, {'message_token_ids': (100, 2)},
    {'termination_reason': 'unknown'}, {'termination_token_id': 100},
    {'payload_format_version': 2}, {'payload_mode': 'canonical_kv'},
])
def test_encoder_rejects_invalid_content_metadata(message, change):
    with pytest.raises(ValueError):
        encode_incremental_message(replace(message, **change))


@pytest.mark.parametrize('field,value', [
    ('message_token_count', 3), ('sender_span_length', -1), ('sender_span_start', -5),
    ('source_agent', 'B'), ('target_agent', 'A'), ('sequence', 2), ('sequence', True),
    ('message_id', 'A:bad'), ('payload_format_version', 5), ('termination_token_id', 100),
])
def test_decoder_rejects_forged_metadata_without_sequence_commit(message, field, value):
    receiver = IncrementalKVReceiver(message.model_contract, 'A', 'B')
    wire = repack_metadata(encode_incremental_message(message), lambda m: m.update({field: value}))
    with pytest.raises(ValueError):
        receiver.unpack(wire)
    assert receiver.next_sequence == 1
    assert receiver.unpack(encode_incremental_message(message)) == message


@pytest.mark.parametrize('field', ['weights_sha256', 'config_sha256', 'identity_sha256', 'model_id', 'dtype'])
def test_same_shape_cannot_override_checkpoint_or_rope_contract(message, field):
    changed = replace(message.model_contract, **{field: 'different'})
    receiver = IncrementalKVReceiver(changed, 'A', 'B')
    with pytest.raises(ValueError, match='incompatible'):
        receiver.unpack(encode_incremental_message(message))
    assert receiver.next_sequence == 1


@pytest.mark.parametrize('bad', ['truncated', 'trailing', 'tensor_bit', 'version', 'metadata_size'])
def test_corrupt_packet_and_bit_damage_fail_without_sequence_commit(message, bad):
    wire = encode_incremental_message(message)
    if bad == 'truncated':
        wire = wire[:-1]
    elif bad == 'trailing':
        wire += b'x'
    elif bad == 'tensor_bit':
        altered = bytearray(wire); altered[-33] ^= 1; wire = bytes(altered)
    elif bad == 'version':
        wire = wire[:4] + b'\x02' + wire[5:]
    else:
        wire = wire[:5] + (65537).to_bytes(4, 'big') + wire[9:]
    receiver = IncrementalKVReceiver(message.model_contract, 'A', 'B')
    with pytest.raises(ValueError):
        receiver.unpack(wire)
    assert receiver.next_sequence == 1


def test_duplicate_order_and_wrong_direction_are_rejected(message):
    receiver = IncrementalKVReceiver(message.model_contract, 'A', 'B')
    wire = encode_incremental_message(message)
    receiver.unpack(wire)
    with pytest.raises(ValueError, match='sequence'):
        receiver.unpack(wire)
    assert receiver.next_sequence == 2
    with pytest.raises(ValueError):
        IncrementalKVReceiver(message.model_contract, 'B', 'A').unpack(wire)


@pytest.mark.parametrize('change', ['missing_layer', 'short_value', 'wrong_dtype', 'heads', 'seq', 'body', 'model', 'device'])
def test_all_layers_shapes_dtypes_and_raw_byte_sizes_are_checked(message, change):
    delta = message.delta
    layer = delta.layers[1]
    if change == 'missing_layer': delta = replace(delta, layers=delta.layers[:1])
    elif change == 'short_value': layer = replace(layer, value=replace(layer.value, shape=(1, 1, 1, 2)))
    elif change == 'wrong_dtype': layer = replace(layer, key=replace(layer.key, dtype='float16'))
    elif change == 'heads': layer = replace(layer, key=replace(layer.key, shape=(1, 2, 2, 2)))
    elif change == 'seq': delta = replace(delta, logical_cached_token_count=3)
    elif change == 'body': layer = replace(layer, value=replace(layer.value, data=b'x'))
    elif change == 'model': delta = replace(delta, model=replace(delta.model, model_id='other'))
    else: delta = replace(delta, source_device='cuda:7')
    if layer != message.delta.layers[1]: delta = replace(delta, layers=(delta.layers[0], layer))
    with pytest.raises(ValueError): encode_incremental_message(replace(message, delta=delta))


def test_oversized_packet_tokens_positions_and_invalid_limits(message):
    for limits in (IncrementalLimits(max_packet_bytes=100), IncrementalLimits(max_message_tokens=1),
                   IncrementalLimits(max_sender_positions=72)):
        with pytest.raises(ValueError): encode_incremental_message(message, limits=limits)
        with pytest.raises(ValueError):
            decode_incremental_message(encode_incremental_message(message), expected_contract=message.model_contract,
                                       source='A', target='B', sequence=1, limits=limits)
    with pytest.raises(ValueError): IncrementalLimits(max_packet_bytes=-1)


def test_binary_tensor_descriptors_rejected_even_with_recomputed_checksum(message):
    wire = encode_incremental_message(message)
    _, _, nm, nt, _ = Struct('!4sBIII').unpack_from(wire)
    # KV header layer count is at offset 8; forge it without changing size.
    offset = 17 + nm + nt
    body = bytearray(wire[:-32]); body[offset+8:offset+12] = (3).to_bytes(4, 'big')
    with pytest.raises(ValueError): decode(bytes(body) + hashlib.sha256(body).digest(), message)


@pytest.mark.parametrize('dtype,width', [('bfloat16', 2), ('float16', 2), ('float32', 4)])
def test_all_supported_wire_dtypes_are_lossless(message, dtype, width):
    layers = tuple(replace(l, key=replace(l.key, dtype=dtype, data=bytes(range(4*width))),
                             value=replace(l.value, dtype=dtype, data=bytes(reversed(range(4*width)))))
                   for l in message.delta.layers)
    changed = replace(message, model_contract=replace(message.model_contract, dtype=dtype),
                      delta=replace(message.delta, layers=layers))
    assert decode(encode_incremental_message(changed), changed) == changed


@pytest.mark.parametrize('field', ['payload_format_version', 'termination_token_id', 'message_id'])
def test_required_envelope_metadata_cannot_be_defaulted_on_receive(message, field):
    wire = repack_metadata(encode_incremental_message(message), lambda m: m.pop(field))
    with pytest.raises(ValueError, match='missing'):
        decode(wire, message)


def test_bitwise_expectation_failure_does_not_advance_sequence(message):
    receiver = IncrementalKVReceiver(message.model_contract, 'A', 'B')
    wrong = replace(message, delta=replace(message.delta, layers=tuple(reversed(message.delta.layers))))
    with pytest.raises(ValueError, match='bitwise'):
        receiver.unpack(encode_incremental_message(message), expected_message=wrong)
    assert receiver.next_sequence == 1
