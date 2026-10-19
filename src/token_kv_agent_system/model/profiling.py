"""Serializable observation profiles for real local inference costs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from statistics import mean, median
from typing import Literal

from token_kv_agent_system.models import TokenPayload

from .base import IncrementalCausalLMBackend
from .cache_inspection import inspect_kv_cache


PROFILE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProfileConfig:
    """Small benchmark configuration; every formal run starts with an empty cache."""

    max_new_tokens: int = 8
    warmup_runs: int = 1
    measurement_runs: int = 3

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative")
        if self.measurement_runs <= 0:
            raise ValueError("measurement_runs must be greater than zero")


@dataclass(frozen=True)
class ProfileModelMeasurement:
    model_name: str
    device: str
    dtype: str | None
    vocab_size: int


@dataclass(frozen=True)
class ProfileInputMeasurement:
    prompt: str
    input_token_ids: tuple[int, ...]
    input_token_count: int


@dataclass(frozen=True)
class PrefillMeasurement:
    prefill_forward_ms: float
    cached_token_count_after_prefill: int


@dataclass(frozen=True)
class KVMeasurement:
    cache_type: str
    dtype: str
    kv_layer_count: int
    key_bytes: int
    value_bytes: int
    raw_kv_tensor_bytes: int
    raw_kv_tensor_bytes_after_decode_steps: tuple[int, ...]


@dataclass(frozen=True)
class DecodeMeasurement:
    generated_token_count: int
    decode_forward_count: int
    per_step_decode_forward_ms: tuple[float, ...]


@dataclass(frozen=True)
class OutputMeasurement:
    output_token_ids: tuple[int, ...]
    generated_text: str
    stop_reason: Literal["eos", "max_new_tokens"]


@dataclass(frozen=True)
class InferenceMeasurementSample:
    """One independent run containing only captured values, never runtime objects."""

    prefill: PrefillMeasurement
    kv: KVMeasurement
    decode: DecodeMeasurement
    output: OutputMeasurement


@dataclass(frozen=True)
class MeasuredInferenceProfile:
    """Values observed directly from tokenizer, forwards, cache, and decoding."""

    model: ProfileModelMeasurement
    input: ProfileInputMeasurement
    samples: tuple[InferenceMeasurementSample, ...]


@dataclass(frozen=True)
class DerivedInferenceMetrics:
    """Values calculated from measured metadata and raw timing samples."""

    theoretical_bits_per_token: int
    theoretical_token_payload_bits: int
    theoretical_token_payload_bytes: int
    prefill_forward_samples_ms: tuple[float, ...]
    median_prefill_forward_ms: float
    decode_forward_samples_ms: tuple[float, ...]
    mean_decode_forward_ms: float | None
    median_decode_forward_ms: float | None
    per_run_total_decode_forward_ms: tuple[float, ...]
    median_total_decode_forward_ms: float | None
    raw_kv_bytes_per_cached_token: int
    raw_kv_growth_bytes_per_decode_step: tuple[int, ...]
    raw_kv_to_theoretical_token_payload_ratio: float | None


@dataclass(frozen=True)
class InferenceCostProfile:
    """Immutable, JSON-safe measurement profile with explicit provenance."""

    schema_version: int
    config: ProfileConfig
    measured: MeasuredInferenceProfile
    derived: DerivedInferenceMetrics

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass(frozen=True)
class ScalingProfileEntry:
    requested_token_count: int
    profile: InferenceCostProfile


@dataclass(frozen=True)
class InferenceScalingProfile:
    schema_version: int
    entries: tuple[ScalingProfileEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def measure_inference_profile(
    backend: IncrementalCausalLMBackend,
    prompt: str,
    config: ProfileConfig | None = None,
) -> InferenceCostProfile:
    """Measure one prompt without retaining any model, tensor, or cache reference."""

    config = config or ProfileConfig()
    if not prompt:
        raise ValueError("prompt cannot be empty")

    input_token_ids = backend.encode(prompt)
    if not input_token_ids:
        raise ValueError("prompt must encode to at least one token")

    for _ in range(config.warmup_runs):
        _measure_once(backend, input_token_ids, config.max_new_tokens)

    samples = tuple(
        _measure_once(backend, input_token_ids, config.max_new_tokens)
        for _ in range(config.measurement_runs)
    )
    return build_inference_profile(backend, prompt, input_token_ids, config, samples)


def build_inference_profile(
    backend: IncrementalCausalLMBackend,
    prompt: str,
    input_token_ids: tuple[int, ...],
    config: ProfileConfig,
    samples: tuple[InferenceMeasurementSample, ...],
) -> InferenceCostProfile:
    """Aggregate already captured samples; exposed for dependency-free tests."""

    if not samples:
        raise ValueError("samples cannot be empty")
    if len(samples) != config.measurement_runs:
        raise ValueError("sample count must equal config.measurement_runs")

    token_payload = TokenPayload.from_token_ids(
        input_token_ids, vocab_size=backend.vocab_size
    )
    token_bits = token_payload.theoretical_payload_bits
    token_bytes = token_payload.theoretical_payload_bytes
    bits_per_token = token_payload.theoretical_bits_per_token
    assert token_bits is not None and token_bytes is not None
    assert bits_per_token is not None

    first_kv = samples[0].kv
    cached_tokens = samples[0].prefill.cached_token_count_after_prefill
    _validate_consistent_samples(samples, cached_tokens, first_kv)
    if first_kv.raw_kv_tensor_bytes % cached_tokens:
        raise ValueError("raw KV tensor bytes are not integral per cached token")
    raw_kv_per_token = first_kv.raw_kv_tensor_bytes // cached_tokens

    growth_samples = tuple(
        current - previous
        for sample in samples
        for previous, current in zip(
            (sample.kv.raw_kv_tensor_bytes,)
            + sample.kv.raw_kv_tensor_bytes_after_decode_steps,
            sample.kv.raw_kv_tensor_bytes_after_decode_steps,
        )
    )
    if any(growth != raw_kv_per_token for growth in growth_samples):
        raise ValueError(
            "raw KV tensor growth per Decode does not match bytes per cached token"
        )

    prefill_samples = tuple(sample.prefill.prefill_forward_ms for sample in samples)
    decode_samples = tuple(
        latency
        for sample in samples
        for latency in sample.decode.per_step_decode_forward_ms
    )
    per_run_decode_totals = tuple(
        sum(sample.decode.per_step_decode_forward_ms) for sample in samples
    )
    dtype = backend.metadata.dtype or samples[0].kv.dtype
    return InferenceCostProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        config=config,
        measured=MeasuredInferenceProfile(
            model=ProfileModelMeasurement(
                model_name=backend.model_name,
                device=backend.device,
                dtype=dtype,
                vocab_size=backend.vocab_size,
            ),
            input=ProfileInputMeasurement(
                prompt=prompt,
                input_token_ids=input_token_ids,
                input_token_count=len(input_token_ids),
            ),
            samples=samples,
        ),
        derived=DerivedInferenceMetrics(
            theoretical_bits_per_token=bits_per_token,
            theoretical_token_payload_bits=token_bits,
            theoretical_token_payload_bytes=token_bytes,
            prefill_forward_samples_ms=prefill_samples,
            median_prefill_forward_ms=median(prefill_samples),
            decode_forward_samples_ms=decode_samples,
            mean_decode_forward_ms=mean(decode_samples) if decode_samples else None,
            median_decode_forward_ms=median(decode_samples) if decode_samples else None,
            per_run_total_decode_forward_ms=per_run_decode_totals,
            median_total_decode_forward_ms=median(per_run_decode_totals),
            raw_kv_bytes_per_cached_token=raw_kv_per_token,
            raw_kv_growth_bytes_per_decode_step=growth_samples,
            raw_kv_to_theoretical_token_payload_ratio=(
                first_kv.raw_kv_tensor_bytes / token_bytes if token_bytes else None
            ),
        ),
    )


def measure_context_scaling(
    backend: IncrementalCausalLMBackend,
    *,
    target_token_counts: tuple[int, ...] = (16, 64, 256),
    config: ProfileConfig | None = None,
    seed_text: str = "KV cache stores reusable attention state for incremental decoding. ",
) -> InferenceScalingProfile:
    """Measure deterministic repeated-text prompts near requested token lengths."""

    config = config or ProfileConfig()
    if not target_token_counts:
        raise ValueError("target_token_counts cannot be empty")
    if any(value <= 0 for value in target_token_counts):
        raise ValueError("target_token_counts must contain positive integers")
    if any(
        after <= before
        for before, after in zip(target_token_counts, target_token_counts[1:])
    ):
        raise ValueError("target_token_counts must be strictly increasing")
    if not seed_text:
        raise ValueError("seed_text cannot be empty")
    if not backend.encode(seed_text):
        raise ValueError("seed_text must encode to at least one token")

    entries = []
    for target in target_token_counts:
        prompt = _repeat_to_token_target(backend, seed_text, target)
        entries.append(
            ScalingProfileEntry(
                requested_token_count=target,
                profile=measure_inference_profile(backend, prompt, config),
            )
        )
    frozen_entries = tuple(entries)
    _validate_scaling_entries(frozen_entries)
    return InferenceScalingProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        entries=frozen_entries,
    )


def _measure_once(
    backend: IncrementalCausalLMBackend,
    input_token_ids: tuple[int, ...],
    max_new_tokens: int,
) -> InferenceMeasurementSample:
    # This call creates a fresh cache. Inspection happens before any decode can
    # mutate a Transformers DynamicCache in place.
    prefill = backend.prefill(input_token_ids=input_token_ids)
    snapshot = inspect_kv_cache(
        prefill.cache, prefill.logical_cached_token_count
    )
    first_layer = snapshot.layers[0]
    dtype = (
        first_layer.key_dtype
        if first_layer.key_dtype == first_layer.value_dtype
        else f"key={first_layer.key_dtype},value={first_layer.value_dtype}"
    )
    kv = KVMeasurement(
        cache_type=snapshot.cache_type,
        dtype=dtype,
        kv_layer_count=snapshot.layer_count,
        key_bytes=snapshot.total_key_bytes,
        value_bytes=snapshot.total_value_bytes,
        raw_kv_tensor_bytes=snapshot.raw_kv_tensor_bytes,
        raw_kv_tensor_bytes_after_decode_steps=(),
    )
    output_ids = [prefill.first_token_id]
    decode_latencies: list[float] = []
    raw_kv_after_decode_steps: list[int] = []
    cache = prefill.cache
    cached_count = prefill.logical_cached_token_count
    stop_reason: Literal["eos", "max_new_tokens"] = (
        "eos" if prefill.first_token_id in backend.eos_token_ids else "max_new_tokens"
    )
    while stop_reason != "eos" and len(output_ids) < max_new_tokens:
        step = backend.decode_step(
            output_ids[-1],
            previous_cache=cache,
            logical_cached_token_count=cached_count,
        )
        decode_latencies.append(step.decode_forward_ms)
        cache = step.cache
        cached_count = step.logical_cached_token_count
        decode_snapshot = inspect_kv_cache(cache, cached_count)
        raw_kv_after_decode_steps.append(decode_snapshot.raw_kv_tensor_bytes)
        output_ids.append(step.predicted_next_token_id)
        if step.predicted_next_token_id in backend.eos_token_ids:
            stop_reason = "eos"

    frozen_output_ids = tuple(output_ids)
    kv = KVMeasurement(
        cache_type=kv.cache_type,
        dtype=kv.dtype,
        kv_layer_count=kv.kv_layer_count,
        key_bytes=kv.key_bytes,
        value_bytes=kv.value_bytes,
        raw_kv_tensor_bytes=kv.raw_kv_tensor_bytes,
        raw_kv_tensor_bytes_after_decode_steps=tuple(raw_kv_after_decode_steps),
    )
    return InferenceMeasurementSample(
        prefill=PrefillMeasurement(
            prefill_forward_ms=prefill.prefill_forward_ms,
            cached_token_count_after_prefill=prefill.logical_cached_token_count,
        ),
        kv=kv,
        decode=DecodeMeasurement(
            generated_token_count=len(frozen_output_ids),
            decode_forward_count=len(decode_latencies),
            per_step_decode_forward_ms=tuple(decode_latencies),
        ),
        output=OutputMeasurement(
            output_token_ids=frozen_output_ids,
            generated_text=backend.decode(frozen_output_ids, skip_special_tokens=True),
            stop_reason=stop_reason,
        ),
    )


def _validate_consistent_samples(
    samples: tuple[InferenceMeasurementSample, ...],
    cached_tokens: int,
    first_kv: KVMeasurement,
) -> None:
    if cached_tokens <= 0:
        raise ValueError("cached token count must be greater than zero")
    for sample in samples:
        if sample.prefill.prefill_forward_ms <= 0:
            raise ValueError("prefill timing samples must be positive")
        if sample.prefill.cached_token_count_after_prefill != cached_tokens:
            raise ValueError("prefill cached token count differs across samples")
        if sample.kv != first_kv:
            raise ValueError("raw KV metadata differs across measurement samples")
        if any(value <= 0 for value in sample.decode.per_step_decode_forward_ms):
            raise ValueError("decode timing samples must be positive")
        if sample.decode.decode_forward_count != len(
            sample.decode.per_step_decode_forward_ms
        ):
            raise ValueError("decode forward count does not match timing samples")
        if sample.decode.decode_forward_count != len(
            sample.kv.raw_kv_tensor_bytes_after_decode_steps
        ):
            raise ValueError("decode forward count does not match KV snapshots")
        if sample.decode.generated_token_count != len(sample.output.output_token_ids):
            raise ValueError("generated token count does not match output Token IDs")


def _repeat_to_token_target(
    backend: IncrementalCausalLMBackend,
    seed_text: str,
    target_token_count: int,
) -> str:
    prompt = seed_text
    token_count = len(backend.encode(prompt))
    while token_count < target_token_count:
        prompt += seed_text
        next_token_count = len(backend.encode(prompt))
        if next_token_count <= token_count:
            raise ValueError("repeating seed_text did not increase token count")
        token_count = next_token_count
    return prompt


def _validate_scaling_entries(
    entries: tuple[ScalingProfileEntry, ...],
) -> None:
    actual_counts = tuple(
        entry.profile.measured.input.input_token_count for entry in entries
    )
    if any(after <= before for before, after in zip(actual_counts, actual_counts[1:])):
        raise ValueError("scaling prompts must have strictly increasing token counts")

    reference_per_token = entries[0].profile.derived.raw_kv_bytes_per_cached_token
    for entry in entries:
        profile = entry.profile
        count = profile.measured.input.input_token_count
        derived = profile.derived
        raw_kv_bytes = profile.measured.samples[0].kv.raw_kv_tensor_bytes
        if derived.theoretical_token_payload_bits != (
            count * derived.theoretical_bits_per_token
        ):
            raise ValueError("Token logical payload bits do not scale with token count")
        if raw_kv_bytes != count * derived.raw_kv_bytes_per_cached_token:
            raise ValueError("raw KV tensor bytes do not scale with cached token count")
        if derived.raw_kv_bytes_per_cached_token != reference_per_token:
            raise ValueError("raw KV bytes per cached token differ across contexts")
