"""Dependency-free binary codec for the Token Protocol v1 data plane."""

from __future__ import annotations

from dataclasses import dataclass
from struct import Struct, iter_unpack
from typing import Iterable

from .models import TokenPayload


TOKEN_PROTOCOL_MAGIC = b"TKVT"
TOKEN_PROTOCOL_VERSION = 1
TOKEN_WIDTH_BITS = 32
TOKEN_BYTES = TOKEN_WIDTH_BITS // 8
MAX_UINT32 = (1 << TOKEN_WIDTH_BITS) - 1
MAX_TOKEN_COUNT = 1_000_000

# Network byte order: magic[4], version[1], token width bits[1], count[4].
_HEADER = Struct("!4sBBI")
TOKEN_WIRE_HEADER_BYTES = _HEADER.size
_TOKEN = Struct("!I")


class TokenProtocolError(ValueError):
    """Base class for invalid Token Protocol inputs or packets."""


class TokenSerializationError(TokenProtocolError):
    """Raised when logical Token IDs cannot be represented by protocol v1."""


class TokenDeserializationError(TokenProtocolError):
    """Raised when a byte stream is not one complete valid token packet."""


class UnsupportedTokenProtocolVersion(TokenDeserializationError):
    """Raised when a packet uses a protocol version this codec cannot read."""


class UnsupportedTokenWidth(TokenDeserializationError):
    """Raised when a packet uses a token representation this version cannot read."""


@dataclass(frozen=True)
class TokenWirePacket:
    """A parsed Token Protocol packet, separate from the domain TokenPayload."""

    protocol_version: int
    token_width_bits: int
    token_ids: tuple[int, ...]
    wire_bytes: bytes

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    @property
    def header_bytes(self) -> int:
        return TOKEN_WIRE_HEADER_BYTES

    @property
    def token_data_bytes(self) -> int:
        return self.token_count * TOKEN_BYTES

    @property
    def total_wire_bytes(self) -> int:
        return len(self.wire_bytes)


@dataclass(frozen=True)
class TokenWireSize:
    """Comparison of theoretical logical capacity with an actual wire packet."""

    token_count: int
    theoretical_minimum_bytes: int
    wire_header_bytes: int
    wire_token_data_bytes: int
    total_wire_bytes: int
    wire_overhead_bytes: int
    wire_to_theoretical_ratio: float | None


def estimate_token_wire_bytes(token_count: int) -> int:
    """Exact v1 packet length without constructing Token IDs or a packet."""
    if isinstance(token_count, bool) or not isinstance(token_count, int) or not 0 <= token_count <= MAX_TOKEN_COUNT:
        raise ValueError("invalid Token Protocol v1 token count")
    return TOKEN_WIRE_HEADER_BYTES + token_count * TOKEN_BYTES


def serialize_token_ids(token_ids: Iterable[int]) -> bytes:
    """Serialize Token IDs as one complete Token Protocol v1 packet."""

    normalized = tuple(token_ids)
    if len(normalized) > MAX_TOKEN_COUNT:
        raise TokenSerializationError(
            f"token count {len(normalized)} exceeds maximum {MAX_TOKEN_COUNT}"
        )
    for index, token_id in enumerate(normalized):
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise TokenSerializationError(f"token ID at index {index} must be an integer")
        if token_id < 0:
            raise TokenSerializationError(
                f"token ID at index {index} cannot be negative"
            )
        if token_id > MAX_UINT32:
            raise TokenSerializationError(
                f"token ID at index {index} exceeds uint32 maximum {MAX_UINT32}"
            )

    header = _HEADER.pack(
        TOKEN_PROTOCOL_MAGIC,
        TOKEN_PROTOCOL_VERSION,
        TOKEN_WIDTH_BITS,
        len(normalized),
    )
    body = b"".join(_TOKEN.pack(token_id) for token_id in normalized)
    return header + body


def serialize_token_payload(payload: TokenPayload) -> bytes:
    """Serialize the logical IDs in a TokenPayload without changing the domain object."""

    if not isinstance(payload, TokenPayload):
        raise TypeError("serialize_token_payload requires TokenPayload")
    return serialize_token_ids(payload.token_ids)


def deserialize_token_packet(data: bytes | bytearray | memoryview) -> TokenWirePacket:
    """Parse exactly one Token Protocol v1 packet and reject malformed framing."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    wire_bytes = bytes(data)
    if len(wire_bytes) < TOKEN_WIRE_HEADER_BYTES:
        raise TokenDeserializationError(
            f"packet is too short for {TOKEN_WIRE_HEADER_BYTES}-byte header"
        )

    magic, version, token_width_bits, token_count = _HEADER.unpack_from(wire_bytes)
    if magic != TOKEN_PROTOCOL_MAGIC:
        raise TokenDeserializationError("invalid token protocol magic")
    if version != TOKEN_PROTOCOL_VERSION:
        raise UnsupportedTokenProtocolVersion(
            f"unsupported token protocol version {version}"
        )
    if token_width_bits != TOKEN_WIDTH_BITS:
        raise UnsupportedTokenWidth(
            f"unsupported token width {token_width_bits} bits"
        )
    if token_count > MAX_TOKEN_COUNT:
        raise TokenDeserializationError(
            f"token count {token_count} exceeds maximum {MAX_TOKEN_COUNT}"
        )

    expected_size = estimate_token_wire_bytes(token_count)
    if len(wire_bytes) < expected_size:
        raise TokenDeserializationError(
            f"truncated token payload: expected {expected_size} bytes, "
            f"received {len(wire_bytes)}"
        )
    if len(wire_bytes) > expected_size:
        raise TokenDeserializationError(
            f"trailing bytes: expected {expected_size} bytes, received {len(wire_bytes)}"
        )

    body = wire_bytes[TOKEN_WIRE_HEADER_BYTES:]
    token_ids = tuple(values[0] for values in iter_unpack("!I", body))
    return TokenWirePacket(
        protocol_version=version,
        token_width_bits=token_width_bits,
        token_ids=token_ids,
        wire_bytes=wire_bytes,
    )


def measure_token_wire_size(
    payload: TokenPayload, packet: TokenWirePacket
) -> TokenWireSize:
    """Measure actual v1 bytes and compare them with logical packing metadata."""

    if payload.token_ids != packet.token_ids:
        raise ValueError("TokenPayload and TokenWirePacket token IDs must match")
    theoretical_bytes = payload.theoretical_payload_bytes
    if theoretical_bytes is None:
        raise ValueError("TokenPayload must include theoretical logical-size metadata")
    ratio = (
        packet.total_wire_bytes / theoretical_bytes
        if theoretical_bytes > 0
        else None
    )
    return TokenWireSize(
        token_count=payload.token_count,
        theoretical_minimum_bytes=theoretical_bytes,
        wire_header_bytes=packet.header_bytes,
        wire_token_data_bytes=packet.token_data_bytes,
        total_wire_bytes=packet.total_wire_bytes,
        wire_overhead_bytes=packet.total_wire_bytes - theoretical_bytes,
        wire_to_theoretical_ratio=ratio,
    )
