"""Observable real-model prefill and cached decode demo."""

from __future__ import annotations

import argparse
from statistics import mean, median

from .model import DEFAULT_MODEL_NAME, GenerationConfig, HuggingFaceModelBackend


DEFAULT_PROMPT = "Explain KV cache in one short sentence."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run explicit HuggingFace prefill and cached decode"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser


def run_inference_demo(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 16,
    warmup_runs: int = 1,
) -> None:
    backend = HuggingFaceModelBackend(model_name, device=device)
    # Qwen's shipped generation config has repetition_penalty=1.1. Setting the
    # neutral value makes HF generate a fair oracle for raw-logit argmax.
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
    )
    manual = backend.manual_generate(
        prompt, generation_config, warmup_runs=warmup_runs
    )
    hf = backend.generate(prompt, generation_config)
    prefill = manual.prefill
    latencies = manual.decode_forward_latencies_ms

    print(f"Model: {backend.model_name}")
    print(f"Device: {backend.device.upper()}")
    print(f"Prompt: {prompt}")
    print(f"Warmup runs: {manual.warmup_runs}")
    print(f"Input Token IDs: {list(manual.input_token_ids)}")
    print()
    print("=== PREFILL ===")
    print(f"Input tokens: {manual.input_token_count}")
    print(f"Logits shape: {list(prefill.logits_shape)}")
    print(f"Last-position logits shape: {list(prefill.last_token_logits_shape)}")
    print(
        "First predicted token: "
        f"ID {prefill.first_token_id} / {backend.decode([prefill.first_token_id])!r}"
    )
    print(f"Cache: <opaque {type(prefill.cache).__module__}.{type(prefill.cache).__qualname__}>")
    print(f"Logical cached tokens after prefill: {prefill.logical_cached_token_count}")
    print(f"Prefill forward: {prefill.prefill_forward_ms:.3f} ms")
    print()
    print("=== DECODE ===")
    for index, step in enumerate(manual.decode_steps, start=1):
        print(
            f"Step {index}: input token {step.input_token_id} -> "
            f"predicted token {step.predicted_next_token_id}; "
            f"cache tokens {step.logical_cached_token_count}; "
            f"forward {step.decode_forward_ms:.3f} ms"
        )
    print()
    print("=== RESULT ===")
    print(f"Generated tokens: {manual.output_token_count}")
    print(f"Generated Token IDs: {list(manual.output_token_ids)}")
    print(f"Generated text: {manual.generated_text}")
    print(f"Stop reason: {manual.stop_reason}")
    print(f"Prefill forward: {manual.prefill_forward_ms:.3f} ms")
    print(f"Decode forwards: {len(latencies)}")
    print(f"Per-step decode forward ms: {[round(value, 3) for value in latencies]}")
    print(f"Total decode forward: {manual.total_decode_forward_ms:.3f} ms")
    if latencies:
        print(f"Mean decode forward: {mean(latencies):.3f} ms")
        print(f"Median decode forward: {median(latencies):.3f} ms")
    else:
        print("Mean decode forward: n/a")
        print("Median decode forward: n/a")
    print(f"HF generate Token IDs: {list(hf.output_token_ids)}")
    print(f"Manual == HF generate: {manual.output_token_ids == hf.output_token_ids}")


def main() -> None:
    args = build_parser().parse_args()
    run_inference_demo(
        model_name=args.model,
        device=args.device,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
    )


if __name__ == "__main__":
    main()
