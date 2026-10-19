"""Core value objects shared across component interfaces.

Sizes are bytes and all latency values are milliseconds. Network bandwidth is
decimal megabits per second (1 Mbps = 1,000,000 bits/second).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CommunicationMode(str, Enum):
    TOKEN = "TOKEN"
    KV = "KV"


@dataclass(frozen=True)
class NetworkState:
    bandwidth_mbps: float
    base_latency_ms: float

    def __post_init__(self) -> None:
        if self.bandwidth_mbps <= 0:
            raise ValueError("bandwidth_mbps must be greater than zero")
        if self.base_latency_ms < 0:
            raise ValueError("base_latency_ms cannot be negative")


@dataclass(frozen=True)
class TokenPayload:
    """Discrete token IDs and optional theoretical logical-size metadata.

    ``estimated_size_bytes`` remains the byte value consumed by the existing
    selector and transports. Real tokenizer payloads set it to the
    bit-packed theoretical logical minimum. This is not a measured wire size.
    Mock payloads may continue to model an
    explicitly fixed-width protocol and leave the theoretical fields unset.
    """

    token_ids: tuple[int, ...]
    estimated_size_bytes: int
    theoretical_bits_per_token: int | None = None
    theoretical_payload_bits: int | None = None

    def __post_init__(self) -> None:
        if self.estimated_size_bytes < 0:
            raise ValueError("estimated_size_bytes cannot be negative")
        if any(not isinstance(token_id, int) or token_id < 0 for token_id in self.token_ids):
            raise ValueError("token_ids must contain non-negative integers")
        if self.theoretical_bits_per_token is not None:
            if self.theoretical_bits_per_token < 0:
                raise ValueError("theoretical_bits_per_token cannot be negative")
            expected_bits = self.token_count * self.theoretical_bits_per_token
            if self.theoretical_payload_bits != expected_bits:
                raise ValueError(
                    "theoretical_payload_bits must equal token_count times "
                    "theoretical_bits_per_token"
                )
        elif self.theoretical_payload_bits is not None:
            raise ValueError(
                "theoretical_payload_bits requires theoretical_bits_per_token"
            )

    @classmethod
    def from_token_ids(
        cls, token_ids: tuple[int, ...] | list[int], *, vocab_size: int
    ) -> TokenPayload:
        normalized_ids = tuple(int(token_id) for token_id in token_ids)
        bits_per_token = theoretical_bits_per_token(vocab_size)
        payload_bits = len(normalized_ids) * bits_per_token
        return cls(
            token_ids=normalized_ids,
            estimated_size_bytes=bits_to_bytes(payload_bits),
            theoretical_bits_per_token=bits_per_token,
            theoretical_payload_bits=payload_bits,
        )

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    @property
    def theoretical_payload_bytes(self) -> int | None:
        if self.theoretical_payload_bits is None:
            return None
        return bits_to_bytes(self.theoretical_payload_bits)


def theoretical_bits_per_token(vocab_size: int) -> int:
    """Return the ideal logical bit width without assuming byte alignment."""

    if vocab_size <= 0:
        raise ValueError("vocab_size must be greater than zero")
    return (vocab_size - 1).bit_length()


def bits_to_bytes(bit_count: int) -> int:
    """Round a total bit count up once, after aggregating all tokens."""

    if bit_count < 0:
        raise ValueError("bit_count cannot be negative")
    return (bit_count + 7) // 8


@dataclass(frozen=True)
class KVPayload:
    """Metadata-only stand-in for a future tensor-backed KV cache."""

    cache_key: str
    estimated_size_bytes: int

    def __post_init__(self) -> None:
        if self.estimated_size_bytes < 0:
            raise ValueError("estimated_size_bytes cannot be negative")


@dataclass(frozen=True)
class CommunicationContext:
    source_agent_id: str
    target_agent_id: str
    network_state: NetworkState
    token_payload: TokenPayload
    kv_payload: KVPayload
    receiver_prefill_latency_ms: float

    def __post_init__(self) -> None:
        if self.receiver_prefill_latency_ms < 0:
            raise ValueError("receiver_prefill_latency_ms cannot be negative")


@dataclass(frozen=True)
class SelectionDecision:
    mode: CommunicationMode
    estimated_token_latency_ms: float
    estimated_kv_latency_ms: float
