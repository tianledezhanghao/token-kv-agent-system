from dataclasses import replace
from struct import pack_into, unpack_from

import pytest

from token_kv_agent_system.kv_protocol import (
    DTYPE_BFLOAT16,
    KV_PROTOCOL_MAGIC,
    KV_PROTOCOL_VERSION,
    KV_WIRE_FIXED_HEADER_BYTES,
    MAX_LAYER_COUNT,
    KVDeserializationError,
    KVModelCompatibility,
    KVSerializationError,
    NeutralKVLayer,
    NeutralKVPayload,
    NeutralKVTensor,
    UnsupportedKVProtocolVersion,
    deserialize_kv_packet,
    serialize_kv_payload,
)


def _tensor(dtype="bfloat16", shape=(1, 2, 3, 4), seed=0):
    element_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}[dtype]
    size = element_bytes
    for dimension in shape:
        size *= dimension
    return NeutralKVTensor(
        dtype=dtype,
        shape=shape,
        data=bytes((seed + index) % 256 for index in range(size)),
    )


def _payload(dtype="bfloat16"):
    return NeutralKVPayload(
        model=KVModelCompatibility("test/model", 2, 2, 4),
        logical_cached_token_count=3,
        source_device="mps:0",
        layers=(
            NeutralKVLayer(0, _tensor(dtype, seed=1), _tensor(dtype, seed=2)),
            NeutralKVLayer(1, _tensor(dtype, seed=3), _tensor(dtype, seed=4)),
        ),
    )


def _first_tensor_offset(packet):
    model_length = unpack_from("!H", packet, 40)[0]
    device_length = unpack_from("!H", packet, 42)[0]
    return KV_WIRE_FIXED_HEADER_BYTES + model_length + device_length + 4


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_multilayer_round_trip_is_bitwise_exact(dtype):
    source = _payload(dtype)
    encoded = serialize_kv_payload(source)
    decoded = deserialize_kv_packet(encoded.packet)

    assert encoded.packet[:4] == KV_PROTOCOL_MAGIC
    assert encoded.packet[4] == KV_PROTOCOL_VERSION
    assert decoded.payload == source
    assert [layer.layer_index for layer in decoded.payload.layers] == [0, 1]
    assert decoded.payload.logical_cached_token_count == 3
    assert decoded.payload.layers[0].key.data == source.layers[0].key.data
    assert decoded.payload.layers[0].value.data == source.layers[0].value.data
    assert encoded.raw_kv_tensor_bytes == source.raw_kv_tensor_bytes
    assert encoded.wire_tensor_body_bytes == source.raw_kv_tensor_bytes
    assert encoded.total_wire_bytes == len(encoded.packet)
    assert encoded.total_wire_bytes == (
        encoded.wire_metadata_bytes + encoded.wire_tensor_body_bytes
    )
    assert decoded.wire_tensor_body_bytes == encoded.wire_tensor_body_bytes
    assert decoded.wire_metadata_bytes == encoded.wire_metadata_bytes


def test_encoding_is_deterministic():
    source = _payload()

    first = serialize_kv_payload(source).packet
    second = serialize_kv_payload(source).packet

    assert first == second


def test_metadata_uses_network_byte_order_and_tensor_body_is_unchanged():
    source = _payload()
    encoded = serialize_kv_payload(source).packet

    assert encoded[8:12] == b"\x00\x00\x00\x02"
    assert encoded[12:20] == b"\x00\x00\x00\x00\x00\x00\x00\x03"
    first_tensor = _first_tensor_offset(encoded)
    assert encoded[first_tensor] == DTYPE_BFLOAT16
    rank = encoded[first_tensor + 1]
    body_length = unpack_from("!Q", encoded, first_tensor + 4)[0]
    body_offset = first_tensor + 12 + rank * 8
    assert encoded[body_offset : body_offset + body_length] == source.layers[0].key.data


def test_device_is_diagnostic_metadata_only():
    source = _payload()
    cuda_source = replace(source, source_device="cuda:7")

    restored = deserialize_kv_packet(serialize_kv_payload(cuda_source).packet).payload

    assert restored.source_device == "cuda:7"
    assert restored.layers == source.layers


def test_unknown_version_is_rejected():
    packet = bytearray(serialize_kv_payload(_payload()).packet)
    packet[4] = 2
    with pytest.raises(UnsupportedKVProtocolVersion):
        deserialize_kv_packet(packet)


def test_invalid_magic_is_rejected():
    packet = bytearray(serialize_kv_payload(_payload()).packet)
    packet[:4] = b"NOPE"
    with pytest.raises(KVDeserializationError, match="magic"):
        deserialize_kv_packet(packet)


@pytest.mark.parametrize("cut", [0, 4, KV_WIRE_FIXED_HEADER_BYTES - 1])
def test_truncated_header_is_rejected(cut):
    packet = serialize_kv_payload(_payload()).packet
    with pytest.raises(KVDeserializationError, match="header"):
        deserialize_kv_packet(packet[:cut])


def test_truncated_metadata_is_rejected():
    packet = bytearray(serialize_kv_payload(_payload()).packet)
    pack_into("!Q", packet, 20, KV_WIRE_FIXED_HEADER_BYTES + 2)
    del packet[KV_WIRE_FIXED_HEADER_BYTES + 2 :]
    with pytest.raises(KVDeserializationError, match="model metadata"):
        deserialize_kv_packet(packet)


def test_truncated_tensor_body_is_rejected():
    packet = bytearray(serialize_kv_payload(_payload()).packet)
    del packet[-1]
    pack_into("!Q", packet, 20, len(packet))
    with pytest.raises(KVDeserializationError, match="truncated"):
        deserialize_kv_packet(packet)


def test_trailing_garbage_is_rejected():
    packet = serialize_kv_payload(_payload()).packet + b"x"
    with pytest.raises(KVDeserializationError, match="trailing"):
        deserialize_kv_packet(packet)


def test_unknown_dtype_is_rejected():
    packet = bytearray(serialize_kv_payload(_payload()).packet)
    packet[_first_tensor_offset(packet)] = 255
    with pytest.raises(KVDeserializationError, match="dtype"):
        deserialize_kv_packet(packet)


def test_invalid_rank_is_rejected():
    packet = bytearray(serialize_kv_payload(_payload()).packet)
    packet[_first_tensor_offset(packet) + 1] = 0
    with pytest.raises(KVDeserializationError, match="rank"):
        deserialize_kv_packet(packet)


def test_rank_without_sequence_dimension_is_rejected():
    packet = bytearray(serialize_kv_payload(_payload()).packet)
    packet[_first_tensor_offset(packet) + 1] = 1
    with pytest.raises(KVDeserializationError, match="rank"):
        deserialize_kv_packet(packet)


def test_invalid_dimension_is_rejected():
    packet = bytearray(serialize_kv_payload(_payload()).packet)
    pack_into("!Q", packet, _first_tensor_offset(packet) + 12, 0)
    with pytest.raises(KVDeserializationError, match="dimension"):
        deserialize_kv_packet(packet)


def test_shape_and_declared_length_mismatch_is_rejected():
    packet = bytearray(serialize_kv_payload(_payload()).packet)
    tensor_offset = _first_tensor_offset(packet)
    pack_into("!Q", packet, tensor_offset + 4, 2)
    with pytest.raises(KVDeserializationError, match="requires"):
        deserialize_kv_packet(packet)


def test_duplicate_or_missing_layer_order_is_rejected():
    packet = bytearray(serialize_kv_payload(_payload()).packet)
    first_tensor = _first_tensor_offset(packet)
    tensor_body_length = unpack_from("!Q", packet, first_tensor + 4)[0]
    one_tensor_size = 12 + 4 * 8 + tensor_body_length
    second_layer_offset = first_tensor + 2 * one_tensor_size
    pack_into("!I", packet, second_layer_offset, 0)
    with pytest.raises(KVDeserializationError, match="layer ordering"):
        deserialize_kv_packet(packet)


def test_out_of_range_layer_is_rejected():
    packet = bytearray(serialize_kv_payload(_payload()).packet)
    first_layer_offset = _first_tensor_offset(packet) - 4
    pack_into("!I", packet, first_layer_offset, 2)
    with pytest.raises(KVDeserializationError, match="layer ordering"):
        deserialize_kv_packet(packet)


def test_absurd_layer_count_is_rejected_before_records():
    packet = bytearray(serialize_kv_payload(_payload()).packet)
    pack_into("!I", packet, 8, MAX_LAYER_COUNT + 1)
    with pytest.raises(KVDeserializationError, match="layer count"):
        deserialize_kv_packet(packet)


def test_absurd_dimension_is_rejected_before_body_read():
    packet = bytearray(serialize_kv_payload(_payload()).packet)
    pack_into("!Q", packet, _first_tensor_offset(packet) + 12, 1 << 40)
    with pytest.raises(KVDeserializationError, match="dimension"):
        deserialize_kv_packet(packet)


def test_absurd_tensor_body_length_is_rejected_before_body_read():
    packet = bytearray(serialize_kv_payload(_payload()).packet)
    pack_into("!Q", packet, _first_tensor_offset(packet) + 4, 9 * 1024**3)
    with pytest.raises(KVDeserializationError, match="maximum tensor size"):
        deserialize_kv_packet(packet)


def test_serializer_rejects_missing_and_duplicate_layers():
    source = _payload()
    missing = replace(source, layers=(source.layers[0],))
    duplicate = replace(source, layers=(source.layers[0], source.layers[0]))

    with pytest.raises(KVSerializationError, match="layer count"):
        serialize_kv_payload(missing)
    with pytest.raises(KVSerializationError, match="ordered"):
        serialize_kv_payload(duplicate)


def test_serializer_rejects_unknown_dtype_and_invalid_shape():
    source = _payload()
    bad_dtype = replace(source.layers[0].key, dtype="float64")
    bad_shape = replace(source.layers[0].key, shape=(1, 2, 0, 4))

    with pytest.raises(KVSerializationError, match="dtype"):
        serialize_kv_payload(
            replace(source, layers=(replace(source.layers[0], key=bad_dtype), source.layers[1]))
        )
    with pytest.raises(KVSerializationError, match="dimension"):
        serialize_kv_payload(
            replace(source, layers=(replace(source.layers[0], key=bad_shape), source.layers[1]))
        )


def test_serializer_requires_immutable_tensor_bytes():
    source = _payload()
    mutable = replace(source.layers[0].key, data=bytearray(source.layers[0].key.data))

    with pytest.raises(KVSerializationError, match="immutable bytes"):
        serialize_kv_payload(
            replace(source, layers=(replace(source.layers[0], key=mutable), source.layers[1]))
        )


def test_serializer_rejects_logical_length_mismatch():
    with pytest.raises(KVSerializationError, match="logical_cached_token_count"):
        serialize_kv_payload(replace(_payload(), logical_cached_token_count=4))
