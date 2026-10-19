"""Real Receiver KV reconstruction and causal continuation demonstration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from .kv_protocol import deserialize_kv_packet
from .model import (
    DEFAULT_MODEL_NAME,
    HuggingFaceModelBackend,
    KVCacheRestoreResult,
    serialize_huggingface_kv_cache,
)


DEFAULT_PREFIX = "KV cache stores previous attention states."
DEFAULT_SUFFIX = " Now explain why this saves computation."


@dataclass(frozen=True)
class LogitsDifference:
    max_abs_diff: float
    mean_abs_diff: float
    exact: bool
    close: bool


@dataclass(frozen=True)
class ContinuationEvidence:
    receiver_input_token_ids: tuple[int, ...]
    original_next_token_id: int
    restored_next_token_id: int
    full_recompute_next_token_id: int
    original_vs_restored: LogitsDifference
    restored_vs_full_recompute: LogitsDifference
    resulting_cache_length: int

    @property
    def all_next_token_ids_equal(self) -> bool:
        return (
            self.original_next_token_id
            == self.restored_next_token_id
            == self.full_recompute_next_token_id
        )


@dataclass(frozen=True)
class MultiStepEvidence:
    original_token_ids: tuple[int, ...]
    restored_token_ids: tuple[int, ...]
    full_recompute_token_ids: tuple[int, ...]
    first_divergence_step: int | None
    first_divergence_max_abs_diff: float | None
    restored_top1_margin_at_divergence: float | None
    full_recompute_top1_margin_at_divergence: float | None

    @property
    def original_restored_exact(self) -> bool:
        return self.original_token_ids == self.restored_token_ids

    @property
    def all_paths_exact(self) -> bool:
        return (
            self.original_token_ids
            == self.restored_token_ids
            == self.full_recompute_token_ids
        )


@dataclass(frozen=True)
class MaterializationEvidence:
    generated_token_ids: tuple[int, ...]
    pending_token_id: int
    cache_length_before: int
    cache_length_after: int
    materialization_forward_ms: float
    materialized_next_token_id: int
    full_recompute_next_token_id: int
    continuation_matches_full_recompute: bool
    logits_difference: LogitsDifference


@dataclass(frozen=True)
class KVRestoreDemoResult:
    prefix: str
    prefix_token_ids: tuple[int, ...]
    receiver_suffix: str
    receiver_suffix_token_ids: tuple[int, ...]
    cache_type: str
    wire_bytes: int
    deserialize_ms: float
    restore: KVCacheRestoreResult
    one_token: ContinuationEvidence
    multi_token: ContinuationEvidence
    multi_step: MultiStepEvidence
    materialization: MaterializationEvidence


def run_kv_restore_demo(
    backend: HuggingFaceModelBackend,
    *,
    prefix: str = DEFAULT_PREFIX,
    receiver_suffix: str = DEFAULT_SUFFIX,
    decode_tokens: int = 6,
) -> KVRestoreDemoResult:
    if decode_tokens < 4:
        raise ValueError("decode_tokens must be at least 4")
    prefix_ids = backend.encode(prefix)
    suffix_ids = backend.encode(receiver_suffix)
    if not prefix_ids or not suffix_ids:
        raise ValueError("prefix and Receiver suffix must each encode to Token IDs")

    sender = backend.prefill(input_token_ids=prefix_ids)
    wire = serialize_huggingface_kv_cache(
        sender.cache,
        logical_cached_token_count=len(prefix_ids),
        model_id=backend.model_name,
        model_config=backend.kv_model_config,
    )
    decoded = deserialize_kv_packet(wire.packet)
    primary_restore = backend.restore_kv_cache(decoded.payload)
    cache_type = type(primary_restore.cache).__name__

    one_token = _continuation_evidence(backend, prefix_ids, suffix_ids[:1], decoded.payload)
    multi_token = _continuation_evidence(backend, prefix_ids, suffix_ids, decoded.payload)
    multi_step = _multi_step_evidence(
        backend, prefix_ids, suffix_ids, decoded.payload, decode_tokens
    )
    materialization = _materialization_evidence(backend, prefix_ids, decode_tokens)
    result = KVRestoreDemoResult(
        prefix=prefix,
        prefix_token_ids=prefix_ids,
        receiver_suffix=receiver_suffix,
        receiver_suffix_token_ids=suffix_ids,
        cache_type=cache_type,
        wire_bytes=wire.total_wire_bytes,
        deserialize_ms=decoded.deserialize_ms,
        restore=primary_restore,
        one_token=one_token,
        multi_token=multi_token,
        multi_step=multi_step,
        materialization=materialization,
    )
    _assert_correctness(result)
    return result


def _assert_correctness(result: KVRestoreDemoResult) -> None:
    for label, scenario in (
        ("one-token", result.one_token),
        ("multi-token", result.multi_token),
    ):
        if not scenario.all_next_token_ids_equal:
            raise RuntimeError(f"{label} continuation next-token oracle failed")
        if not scenario.original_vs_restored.exact:
            raise RuntimeError(f"{label} Original/Restored logits are not exact")
        if not scenario.restored_vs_full_recompute.close:
            raise RuntimeError(f"{label} Restored/Full logits exceed tolerance")
    if not result.multi_step.original_restored_exact:
        raise RuntimeError("Original/Restored multi-step greedy sequences diverged")
    materialized = result.materialization
    if materialized.cache_length_after != materialized.cache_length_before + 1:
        raise RuntimeError("pending-token materialization did not grow cache by one")
    if not materialized.continuation_matches_full_recompute:
        raise RuntimeError("materialized continuation disagrees with Full Recompute")
    if not materialized.logits_difference.close:
        raise RuntimeError("materialized/Full Recompute logits exceed tolerance")


def _continuation_evidence(
    backend: HuggingFaceModelBackend,
    prefix_ids: tuple[int, ...],
    suffix_ids: tuple[int, ...],
    payload: Any,
) -> ContinuationEvidence:
    original = backend.prefill(input_token_ids=prefix_ids)
    restored = backend.restore_kv_cache(payload)
    original_result = backend.continue_from_cache(
        suffix_ids,
        previous_cache=original.cache,
        logical_cached_token_count=len(prefix_ids),
    )
    restored_result = backend.continue_from_cache(
        suffix_ids,
        previous_cache=restored.cache,
        logical_cached_token_count=len(prefix_ids),
    )
    full = backend.prefill(input_token_ids=prefix_ids + suffix_ids)
    return ContinuationEvidence(
        receiver_input_token_ids=suffix_ids,
        original_next_token_id=original_result.predicted_next_token_id,
        restored_next_token_id=restored_result.predicted_next_token_id,
        full_recompute_next_token_id=full.first_token_id,
        original_vs_restored=_compare_logits(
            backend, original_result.last_token_logits, restored_result.last_token_logits
        ),
        restored_vs_full_recompute=_compare_logits(
            backend, restored_result.last_token_logits, full.last_token_logits
        ),
        resulting_cache_length=restored_result.logical_cached_token_count,
    )


def _multi_step_evidence(
    backend: HuggingFaceModelBackend,
    prefix_ids: tuple[int, ...],
    suffix_ids: tuple[int, ...],
    payload: Any,
    decode_tokens: int,
) -> MultiStepEvidence:
    original_prefix = backend.prefill(input_token_ids=prefix_ids)
    restored_prefix = backend.restore_kv_cache(payload)
    original = backend.continue_from_cache(
        suffix_ids,
        previous_cache=original_prefix.cache,
        logical_cached_token_count=len(prefix_ids),
    )
    restored = backend.continue_from_cache(
        suffix_ids,
        previous_cache=restored_prefix.cache,
        logical_cached_token_count=len(prefix_ids),
    )
    full = backend.prefill(input_token_ids=prefix_ids + suffix_ids)
    original_ids = [original.predicted_next_token_id]
    restored_ids = [restored.predicted_next_token_id]
    full_ids = [full.first_token_id]
    original_cache, original_length = original.cache, original.logical_cached_token_count
    restored_cache, restored_length = restored.cache, restored.logical_cached_token_count
    full_cache, full_length = full.cache, full.logical_cached_token_count
    first_divergence = None
    divergence_diff = None
    restored_margin = None
    full_margin = None
    restored_logits = restored.last_token_logits
    full_logits = full.last_token_logits
    while True:
        step_index = len(original_ids) - 1
        if first_divergence is None and len(
            {original_ids[-1], restored_ids[-1], full_ids[-1]}
        ) != 1:
            first_divergence = step_index
            divergence_diff = _compare_logits(
                backend, restored_logits, full_logits
            ).max_abs_diff
            restored_margin = _top1_margin(backend, restored_logits)
            full_margin = _top1_margin(backend, full_logits)
        if len(original_ids) >= decode_tokens:
            break
        original_step = backend.continue_from_cache(
            (original_ids[-1],),
            previous_cache=original_cache,
            logical_cached_token_count=original_length,
        )
        restored_step = backend.continue_from_cache(
            (restored_ids[-1],),
            previous_cache=restored_cache,
            logical_cached_token_count=restored_length,
        )
        full_step = backend.continue_from_cache(
            (full_ids[-1],),
            previous_cache=full_cache,
            logical_cached_token_count=full_length,
        )
        original_ids.append(original_step.predicted_next_token_id)
        restored_ids.append(restored_step.predicted_next_token_id)
        full_ids.append(full_step.predicted_next_token_id)
        original_cache, original_length = (
            original_step.cache,
            original_step.logical_cached_token_count,
        )
        restored_cache, restored_length = (
            restored_step.cache,
            restored_step.logical_cached_token_count,
        )
        full_cache, full_length = full_step.cache, full_step.logical_cached_token_count
        restored_logits, full_logits = (
            restored_step.last_token_logits,
            full_step.last_token_logits,
        )
    return MultiStepEvidence(
        tuple(original_ids),
        tuple(restored_ids),
        tuple(full_ids),
        first_divergence,
        divergence_diff,
        restored_margin,
        full_margin,
    )


def _materialization_evidence(
    backend: HuggingFaceModelBackend,
    prefix_ids: tuple[int, ...],
    generated_count: int,
) -> MaterializationEvidence:
    prefill = backend.prefill(input_token_ids=prefix_ids)
    generated = [prefill.first_token_id]
    cache = prefill.cache
    length = prefill.logical_cached_token_count
    while len(generated) < generated_count:
        step = backend.continue_from_cache(
            (generated[-1],),
            previous_cache=cache,
            logical_cached_token_count=length,
        )
        generated.append(step.predicted_next_token_id)
        cache, length = step.cache, step.logical_cached_token_count
    pending = generated[-1]
    before = length
    materialized = backend.materialize_token_into_cache(
        pending,
        previous_cache=cache,
        logical_cached_token_count=before,
    )
    full = backend.prefill(input_token_ids=prefix_ids + tuple(generated))
    difference = _compare_logits(
        backend, materialized.last_token_logits, full.last_token_logits
    )
    return MaterializationEvidence(
        generated_token_ids=tuple(generated),
        pending_token_id=pending,
        cache_length_before=before,
        cache_length_after=materialized.logical_cached_token_count,
        materialization_forward_ms=materialized.materialization_forward_ms,
        materialized_next_token_id=materialized.predicted_next_token_id,
        full_recompute_next_token_id=full.first_token_id,
        continuation_matches_full_recompute=(
            materialized.predicted_next_token_id == full.first_token_id
        ),
        logits_difference=difference,
    )


def _compare_logits(
    backend: HuggingFaceModelBackend, left: Any, right: Any
) -> LogitsDifference:
    exact = bool(backend._torch.equal(left, right))
    absolute = backend._torch.abs(left.float() - right.float())
    max_abs_diff = float(backend._torch.max(absolute).item())
    mean_abs_diff = float(backend._torch.mean(absolute).item())
    # BF16 cached and clean-prefill MPS kernels may quantize along different
    # execution paths. This tolerance is diagnostic; greedy equality remains
    # independently mandatory for the fixed continuation oracle.
    close = bool(backend._torch.allclose(left.float(), right.float(), rtol=2e-2, atol=5e-1))
    return LogitsDifference(
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        exact=exact,
        close=close,
    )


def _top1_margin(backend: HuggingFaceModelBackend, logits: Any) -> float:
    values = backend._torch.topk(logits.float(), k=2).values
    return float((values[0] - values[1]).item())


def _print_scenario(label: str, value: ContinuationEvidence) -> None:
    print(f"\n=== {label} ===")
    print(f"Receiver input Token IDs: {list(value.receiver_input_token_ids)}")
    print(f"Original-cache next Token: {value.original_next_token_id}")
    print(f"Restored-cache next Token: {value.restored_next_token_id}")
    print(f"Full-recompute next Token: {value.full_recompute_next_token_id}")
    print(f"All next-token IDs equal: {value.all_next_token_ids_equal}")
    print(
        "Original vs Restored logits max abs diff: "
        f"{value.original_vs_restored.max_abs_diff:.8f} "
        f"(mean={value.original_vs_restored.mean_abs_diff:.8f}, "
        f"exact={value.original_vs_restored.exact})"
    )
    print(
        "Restored vs Full logits max abs diff: "
        f"{value.restored_vs_full_recompute.max_abs_diff:.8f} "
        f"(mean={value.restored_vs_full_recompute.mean_abs_diff:.8f}, "
        f"close={value.restored_vs_full_recompute.close})"
    )
    print(f"Resulting cache length: {value.resulting_cache_length}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="mps", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--decode-tokens", type=int, default=6)
    args = parser.parse_args()
    backend = HuggingFaceModelBackend(args.model, device=args.device)
    result = run_kv_restore_demo(
        backend,
        prefix=args.prefix,
        receiver_suffix=args.suffix,
        decode_tokens=args.decode_tokens,
    )
    restore = result.restore
    print("=== SOURCE / WIRE / RESTORE ===")
    print(f"Model: {backend.model_name}")
    print(f"Device: {backend.device}")
    print(f"Prefix: {result.prefix}")
    print(f"Prefix Token IDs: {list(result.prefix_token_ids)}")
    print(f"Prefix tokens: {len(result.prefix_token_ids)}")
    print(f"KV wire: {result.wire_bytes} B")
    print(f"Deserialize: {result.deserialize_ms:.3f} ms")
    print(f"Compatibility: {'Passed' if restore.compatibility.passed else 'Failed'}")
    print(f"CPU tensor reconstruction: {restore.cpu_tensor_reconstruction_ms:.3f} ms")
    print(f"Host -> {restore.target_device}: {restore.host_to_device_ms:.3f} ms")
    print(f"Cache construction: {restore.cache_object_construction_ms:.3f} ms")
    print(f"Total restore: {restore.total_restore_ms:.3f} ms")
    print(f"Restored cache type: {result.cache_type}")
    print(f"Restored cached tokens: {restore.logical_cached_token_count}")
    _print_scenario("ONE-TOKEN CONTINUATION", result.one_token)
    _print_scenario("MULTI-TOKEN RECEIVER SUFFIX", result.multi_token)
    print("\n=== MULTI-STEP DECODE ===")
    print(f"Original: {list(result.multi_step.original_token_ids)}")
    print(f"Restored: {list(result.multi_step.restored_token_ids)}")
    print(f"Full recompute: {list(result.multi_step.full_recompute_token_ids)}")
    print(f"Original vs Restored exact: {result.multi_step.original_restored_exact}")
    print(f"All three paths exact: {result.multi_step.all_paths_exact}")
    print(f"First divergence step: {result.multi_step.first_divergence_step}")
    if result.multi_step.first_divergence_step is not None:
        print(
            "Restored vs Full max abs diff at divergence: "
            f"{result.multi_step.first_divergence_max_abs_diff:.8f}"
        )
        print(
            "Top-1 margins at divergence (restored/full): "
            f"{result.multi_step.restored_top1_margin_at_divergence:.8f} / "
            f"{result.multi_step.full_recompute_top1_margin_at_divergence:.8f}"
        )
    materialized = result.materialization
    print("\n=== MATERIALIZATION CHECK ===")
    print(f"Generated Token IDs: {list(materialized.generated_token_ids)}")
    print(f"Pending token: {materialized.pending_token_id}")
    print(f"Before: {materialized.cache_length_before}")
    print(f"After: {materialized.cache_length_after}")
    print(f"Materialization forward: {materialized.materialization_forward_ms:.3f} ms")
    print(f"Full-recompute next Token: {materialized.full_recompute_next_token_id}")
    print(
        "Continuation matches full recompute: "
        f"{materialized.continuation_matches_full_recompute}"
    )
    print(f"Logits max abs diff: {materialized.logits_difference.max_abs_diff:.8f}")


if __name__ == "__main__":
    main()
