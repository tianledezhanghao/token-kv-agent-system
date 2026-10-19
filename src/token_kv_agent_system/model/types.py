"""Model-layer result and metadata value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ModelMetadata:
    model_name: str
    device: str
    vocab_size: int
    eos_token_id: int | None
    bos_token_id: int | None
    pad_token_id: int | None
    load_wall_time_seconds: float
    device_fallback_reason: str | None = None
    dtype: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    input_token_ids: tuple[int, ...]
    generated_text: str
    output_token_ids: tuple[int, ...]
    generation_wall_time_seconds: float
    full_token_ids: tuple[int, ...]

    @property
    def input_token_count(self) -> int:
        return len(self.input_token_ids)

    @property
    def output_token_count(self) -> int:
        return len(self.output_token_ids)


@dataclass(frozen=True)
class PrefillResult:
    """Output of one full-prompt forward; cache is intentionally runtime-only."""

    input_token_ids: tuple[int, ...]
    first_token_id: int
    cache: Any = field(repr=False, compare=False)
    prefill_forward_ms: float
    logical_cached_token_count: int
    logits_shape: tuple[int, ...]
    last_token_logits_shape: tuple[int, ...]
    last_token_logits: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class DecodeStepResult:
    """Output of one cached forward whose model input is exactly one token."""

    input_token_id: int
    predicted_next_token_id: int
    cache: Any = field(repr=False, compare=False)
    decode_forward_ms: float
    logical_cached_token_count: int


@dataclass(frozen=True)
class CachedContinuationResult:
    """One cached forward over Receiver-owned Token IDs."""

    input_token_ids: tuple[int, ...]
    predicted_next_token_id: int
    cache: Any = field(repr=False, compare=False)
    forward_ms: float
    logical_cached_token_count: int
    last_token_logits: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class TokenMaterializationResult:
    """Evidence that exactly one pending prediction was consumed into a cache."""

    materialized_token_id: int
    predicted_next_token_id: int
    cache: Any = field(repr=False, compare=False)
    logical_cached_token_count: int
    materialization_forward_ms: float
    last_token_logits: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class ManualGenerationResult:
    prompt: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    generated_text: str
    prefill: PrefillResult
    decode_steps: tuple[DecodeStepResult, ...]
    stop_reason: Literal["eos", "max_new_tokens"]
    warmup_runs: int

    @property
    def prefill_forward_ms(self) -> float:
        return self.prefill.prefill_forward_ms

    @property
    def decode_forward_latencies_ms(self) -> tuple[float, ...]:
        return tuple(step.decode_forward_ms for step in self.decode_steps)

    @property
    def total_decode_forward_ms(self) -> float:
        return sum(self.decode_forward_latencies_ms)

    @property
    def input_token_count(self) -> int:
        return len(self.input_token_ids)

    @property
    def output_token_count(self) -> int:
        return len(self.output_token_ids)


@dataclass(frozen=True)
class TokenIdGenerationResult:
    """Generation that begins with already-tokenized IDs, never source text."""

    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    generated_text: str
    prefill: PrefillResult
    decode_steps: tuple[DecodeStepResult, ...]
    stop_reason: Literal["eos", "max_new_tokens"]

    @property
    def prefill_forward_ms(self) -> float:
        return self.prefill.prefill_forward_ms

    @property
    def decode_forward_latencies_ms(self) -> tuple[float, ...]:
        return tuple(step.decode_forward_ms for step in self.decode_steps)

    @property
    def input_token_count(self) -> int:
        return len(self.input_token_ids)

    @property
    def output_token_count(self) -> int:
        return len(self.output_token_ids)
