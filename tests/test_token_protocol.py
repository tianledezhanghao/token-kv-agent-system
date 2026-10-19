from struct import pack

import pytest

from token_kv_agent_system.models import TokenPayload
from token_kv_agent_system.token_protocol import (
    MAX_TOKEN_COUNT,
    MAX_UINT32,
    TOKEN_BYTES,
    TOKEN_PROTOCOL_MAGIC,
    TOKEN_PROTOCOL_VERSION,
    TOKEN_WIDTH_BITS,
    TOKEN_WIRE_HEADER_BYTES,
    TokenDeserializationError,
    TokenSerializationError,
    UnsupportedTokenProtocolVersion,
    UnsupportedTokenWidth,
    deserialize_token_packet,
    measure_token_wire_size,
    serialize_token_ids,
    serialize_token_payload,
)


@pytest.mark.parametrize(
    "token_ids",
    [(), (7,), (1, 2, 3), (65_536,), (MAX_UINT32,)],
)
def test_token_ids_round_trip(token_ids):
    encoded = serialize_token_ids(token_ids)
    packet = deserialize_token_packet(encoded)

    assert packet.token_ids == token_ids
    assert packet.token_count == len(token_ids)
    assert packet.protocol_version == TOKEN_PROTOCOL_VERSION
    assert packet.token_width_bits == TOKEN_WIDTH_BITS
    assert packet.wire_bytes == encoded


def test_serialize_token_payload_keeps_domain_and_wire_separate():
    payload = TokenPayload.from_token_ids([1, 65_536], vocab_size=151_936)

    encoded = serialize_token_payload(payload)

    assert deserialize_token_packet(encoded).token_ids == payload.token_ids
    assert payload.estimated_size_bytes == 5
    assert len(encoded) == TOKEN_WIRE_HEADER_BYTES + 2 * TOKEN_BYTES


@pytest.mark.parametrize("token_id", [-1, MAX_UINT32 + 1])
def test_unrepresentable_token_id_is_rejected(token_id):
    with pytest.raises(TokenSerializationError):
        serialize_token_ids((token_id,))


def test_non_integer_token_id_is_rejected():
    with pytest.raises(TokenSerializationError, match="must be an integer"):
        serialize_token_ids((1.5,))


def test_serialize_rejects_excessive_token_count():
    with pytest.raises(TokenSerializationError, match="exceeds maximum"):
        serialize_token_ids(range(MAX_TOKEN_COUNT + 1))


def test_invalid_magic_is_rejected():
    encoded = bytearray(serialize_token_ids((1,)))
    encoded[0:4] = b"NOPE"

    with pytest.raises(TokenDeserializationError, match="magic"):
        deserialize_token_packet(encoded)


def test_unknown_version_is_rejected():
    encoded = bytearray(serialize_token_ids((1,)))
    encoded[4] = TOKEN_PROTOCOL_VERSION + 1

    with pytest.raises(UnsupportedTokenProtocolVersion, match="version"):
        deserialize_token_packet(encoded)


def test_unknown_token_width_is_rejected():
    encoded = bytearray(serialize_token_ids((1,)))
    encoded[5] = 16

    with pytest.raises(UnsupportedTokenWidth, match="width"):
        deserialize_token_packet(encoded)


def test_truncated_header_is_rejected():
    with pytest.raises(TokenDeserializationError, match="header"):
        deserialize_token_packet(TOKEN_PROTOCOL_MAGIC)


def test_truncated_payload_is_rejected():
    with pytest.raises(TokenDeserializationError, match="truncated"):
        deserialize_token_packet(serialize_token_ids((1, 2))[:-1])


def test_count_and_payload_length_mismatch_is_rejected():
    encoded = bytearray(serialize_token_ids((1,)))
    encoded[6:10] = pack("!I", 2)

    with pytest.raises(TokenDeserializationError, match="truncated"):
        deserialize_token_packet(encoded)


def test_trailing_garbage_is_rejected():
    with pytest.raises(TokenDeserializationError, match="trailing"):
        deserialize_token_packet(serialize_token_ids((1,)) + b"\x00")


def test_excessive_declared_token_count_is_rejected_before_body_parsing():
    header = pack(
        "!4sBBI",
        TOKEN_PROTOCOL_MAGIC,
        TOKEN_PROTOCOL_VERSION,
        TOKEN_WIDTH_BITS,
        MAX_TOKEN_COUNT + 1,
    )

    with pytest.raises(TokenDeserializationError, match="exceeds maximum"):
        deserialize_token_packet(header)


def test_network_byte_order_is_big_endian():
    encoded = serialize_token_ids((0x01020304,))

    assert encoded[:4] == TOKEN_PROTOCOL_MAGIC
    assert encoded[4] == TOKEN_PROTOCOL_VERSION
    assert encoded[5] == TOKEN_WIDTH_BITS
    assert encoded[6:10] == b"\x00\x00\x00\x01"
    assert encoded[10:] == b"\x01\x02\x03\x04"


def test_uint32_representation_covers_qwen_model_vocab_domain():
    qwen_vocab_size = 151_936
    highest_qwen_token_id = qwen_vocab_size - 1

    assert highest_qwen_token_id <= MAX_UINT32
    assert deserialize_token_packet(
        serialize_token_ids((highest_qwen_token_id,))
    ).token_ids == (highest_qwen_token_id,)


def test_serialized_size_and_wire_comparison_are_derived_from_packet():
    payload = TokenPayload.from_token_ids([1, 2, 3], vocab_size=5)
    packet = deserialize_token_packet(serialize_token_payload(payload))
    size = measure_token_wire_size(payload, packet)

    assert TOKEN_WIRE_HEADER_BYTES == 10
    assert size.token_count == 3
    assert size.theoretical_minimum_bytes == 2
    assert size.wire_header_bytes == 10
    assert size.wire_token_data_bytes == 12
    assert size.total_wire_bytes == len(packet.wire_bytes) == 22
    assert size.wire_overhead_bytes == 20
    assert size.wire_to_theoretical_ratio == 11.0


def test_empty_wire_ratio_is_explicitly_undefined():
    payload = TokenPayload.from_token_ids([], vocab_size=151_936)
    packet = deserialize_token_packet(serialize_token_payload(payload))

    assert measure_token_wire_size(payload, packet).wire_to_theoretical_ratio is None
