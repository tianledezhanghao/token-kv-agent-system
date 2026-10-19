"""CLI for real inference compute and data-volume profiling."""

from __future__ import annotations

import argparse
from pathlib import Path

from .inference_demo import DEFAULT_PROMPT
from .model import (
    DEFAULT_MODEL_NAME,
    HuggingFaceModelBackend,
    InferenceCostProfile,
    InferenceScalingProfile,
    ProfileConfig,
    format_byte_count,
    measure_context_scaling,
    measure_inference_profile,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile real inference compute and logical/raw data volumes"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measurement-runs", type=int, default=3)
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="profile deterministic prompts near 16, 64, and 256 tokens",
    )
    parser.add_argument(
        "--scaling-targets",
        type=int,
        nargs="+",
        default=(16, 64, 256),
        metavar="TOKENS",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the complete profile as JSON"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON only when this path is explicitly provided",
    )
    return parser


def run_profile_demo(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 8,
    warmup_runs: int = 1,
    measurement_runs: int = 3,
    scaling: bool = False,
    scaling_targets: tuple[int, ...] = (16, 64, 256),
    json_output: bool = False,
    output: Path | None = None,
) -> InferenceCostProfile | InferenceScalingProfile:
    config = ProfileConfig(
        max_new_tokens=max_new_tokens,
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
    )
    backend = HuggingFaceModelBackend(model_name, device=device)
    result: InferenceCostProfile | InferenceScalingProfile
    if scaling:
        result = measure_context_scaling(
            backend,
            target_token_counts=scaling_targets,
            config=config,
        )
    else:
        result = measure_inference_profile(backend, prompt, config)

    if json_output:
        print(result.to_json())
    elif isinstance(result, InferenceScalingProfile):
        _print_scaling(result)
    else:
        _print_profile(result)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.to_json() + "\n", encoding="utf-8")
        if not json_output:
            print(f"\nJSON profile written to: {output}")
    return result


def _print_profile(profile: InferenceCostProfile) -> None:
    measured = profile.measured
    derived = profile.derived
    kv = measured.samples[0].kv
    output = measured.samples[0].output

    print("=== MODEL ===")
    print(f"Model: {measured.model.model_name}")
    print(f"Device: {measured.model.device.upper()}")
    print(f"Dtype: {measured.model.dtype or 'n/a'}")
    print(f"Vocabulary: {measured.model.vocab_size}")
    print(f"Measurement runs: {profile.config.measurement_runs}")
    print(f"Warmup runs: {profile.config.warmup_runs}")
    print()
    print("=== INPUT ===")
    print(f"Prompt: {measured.input.prompt}")
    print(f"Tokens: {measured.input.input_token_count}")
    print(f"Token IDs: {list(measured.input.input_token_ids)}")
    print(
        "Theoretical Token logical payload: "
        f"{derived.theoretical_token_payload_bits} bits / "
        f"{derived.theoretical_token_payload_bytes} B"
    )
    print()
    print("=== COMPUTE ===")
    print(
        "Prefill synchronized forward samples: "
        f"{_format_ms_samples(derived.prefill_forward_samples_ms)}"
    )
    print(f"Prefill median: {derived.median_prefill_forward_ms:.3f} ms")
    print(f"Decode forwards sampled: {len(derived.decode_forward_samples_ms)}")
    print(
        "Per-step Decode synchronized forward samples: "
        f"{_format_ms_samples(derived.decode_forward_samples_ms)}"
    )
    print(f"Decode mean: {_format_optional_ms(derived.mean_decode_forward_ms)}")
    print(f"Decode median: {_format_optional_ms(derived.median_decode_forward_ms)}")
    print(
        "Per-run Decode totals: "
        f"{_format_ms_samples(derived.per_run_total_decode_forward_ms)}"
    )
    print(
        "Median per-run Decode total: "
        f"{_format_optional_ms(derived.median_total_decode_forward_ms)}"
    )
    print()
    print("=== KV CACHE ===")
    print(f"Cache type: {kv.cache_type}")
    print(f"Layers: {kv.kv_layer_count}")
    print(f"Cached tokens after Prefill: {measured.input.input_token_count}")
    print(f"Raw K tensors: {format_byte_count(kv.key_bytes)}")
    print(f"Raw V tensors: {format_byte_count(kv.value_bytes)}")
    print(f"Raw KV tensors: {format_byte_count(kv.raw_kv_tensor_bytes)}")
    print(
        "Raw KV tensors/cached token: "
        f"{format_byte_count(derived.raw_kv_bytes_per_cached_token)}"
    )
    print(
        "Measured raw KV growth per Decode-cached token: "
        f"{list(derived.raw_kv_growth_bytes_per_decode_step)} B"
    )
    print()
    print("=== OUTPUT ===")
    print(f"Generated Token IDs: {list(output.output_token_ids)}")
    print(f"Generated text: {output.generated_text}")
    print(f"Stop reason: {output.stop_reason}")
    print()
    print("=== SIZE COMPARISON ===")
    print(
        "Theoretical Token logical payload: "
        f"{derived.theoretical_token_payload_bytes} B"
    )
    print(f"Raw in-memory KV tensors: {kv.raw_kv_tensor_bytes} B")
    ratio = derived.raw_kv_to_theoretical_token_payload_ratio
    print(
        f"Raw-KV / theoretical-Token ratio: {ratio:.2f}x"
        if ratio is not None
        else "Raw-KV / theoretical-Token ratio: n/a"
    )
    print("Raw KV tensor bytes are not serialized or wire payload bytes.")
    print("Token logical bytes are an ideal packing lower bound, not wire bytes.")
    print("No network or end-to-end communication latency is measured here.")


def _print_scaling(scaling: InferenceScalingProfile) -> None:
    print("=== CONTEXT LENGTH SCALING ===")
    print(
        "Tokens | Token logical B | Prefill median ms | Raw KV B | "
        "KV B/token | Decode mean ms | Decode median ms | Raw-KV/Token"
    )
    print("-" * 118)
    for entry in scaling.entries:
        profile = entry.profile
        measured = profile.measured
        derived = profile.derived
        kv = measured.samples[0].kv
        print(
            f"{measured.input.input_token_count:>6} | "
            f"{derived.theoretical_token_payload_bytes:>15} | "
            f"{derived.median_prefill_forward_ms:>17.3f} | "
            f"{kv.raw_kv_tensor_bytes:>8} | "
            f"{derived.raw_kv_bytes_per_cached_token:>10} | "
            f"{_format_optional_number(derived.mean_decode_forward_ms):>14} | "
            f"{_format_optional_number(derived.median_decode_forward_ms):>16} | "
            f"{_format_optional_ratio(derived.raw_kv_to_theoretical_token_payload_ratio):>13}"
        )
    print()
    print("All values above come from this run; timings are synchronized forward timings.")
    print("Raw KV is in-memory tensor capacity, not a serialized/wire payload.")
    print("The Token value is a theoretical logical packing lower bound, not wire bytes.")


def _format_ms_samples(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in values) + "] ms"


def _format_optional_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f} ms"


def _format_optional_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_optional_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}x"


def main() -> None:
    args = build_parser().parse_args()
    run_profile_demo(
        model_name=args.model,
        device=args.device,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        measurement_runs=args.measurement_runs,
        scaling=args.scaling,
        scaling_targets=tuple(args.scaling_targets),
        json_output=args.json,
        output=args.output,
    )


if __name__ == "__main__":
    main()
