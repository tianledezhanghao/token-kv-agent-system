"""CLI demonstration of real runtime KV cache structure and raw capacity."""

from __future__ import annotations

import argparse

from .inference_demo import DEFAULT_PROMPT
from .model import (
    DEFAULT_MODEL_NAME,
    HuggingFaceModelBackend,
    calculate_raw_kv_tensor_bytes,
    config_derived_raw_kv_tensor_bytes,
    format_byte_count,
    inspect_kv_cache,
)
from .models import TokenPayload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect real HuggingFace KV cache tensors and raw bytes"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--decode-steps", type=int, default=3)
    return parser


def run_kv_inspection_demo(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    prompt: str = DEFAULT_PROMPT,
    decode_steps: int = 3,
) -> None:
    if decode_steps < 1:
        raise ValueError("decode_steps must be greater than zero")

    backend = HuggingFaceModelBackend(model_name, device=device)
    config = backend.kv_model_config

    prefill = backend.prefill(prompt)
    snapshots = [
        inspect_kv_cache(prefill.cache, prefill.logical_cached_token_count)
    ]
    next_input_token = prefill.first_token_id
    cache = prefill.cache
    logical_count = prefill.logical_cached_token_count
    for _ in range(decode_steps):
        step = backend.decode_step(
            next_input_token,
            previous_cache=cache,
            logical_cached_token_count=logical_count,
        )
        cache = step.cache
        logical_count = step.logical_cached_token_count
        next_input_token = step.predicted_next_token_id
        snapshots.append(inspect_kv_cache(cache, logical_count))

    prefill_snapshot = snapshots[0]
    first_layer = prefill_snapshot.layers[0]
    sample_indices = tuple(dict.fromkeys((0, 1, prefill_snapshot.layer_count - 1)))
    theoretical_prefill = config_derived_raw_kv_tensor_bytes(
        config, prefill_snapshot
    )
    growth = tuple(
        current.raw_kv_tensor_bytes - previous.raw_kv_tensor_bytes
        for previous, current in zip(snapshots, snapshots[1:])
    )
    per_token_formula = calculate_raw_kv_tensor_bytes(
        num_layers=config.num_hidden_layers,
        batch_size=first_layer.key_shape[0],
        num_key_value_heads=config.num_key_value_heads,
        cached_tokens=1,
        head_dim=config.head_dim,
        key_element_size=first_layer.key_element_size,
        value_element_size=first_layer.value_element_size,
    )
    token_payload = TokenPayload.from_token_ids(
        prefill.input_token_ids, vocab_size=backend.vocab_size
    )
    token_bytes = token_payload.theoretical_payload_bytes
    assert token_bytes is not None

    print("=== MODEL ===")
    print(f"Model: {backend.model_name}")
    print(f"Device: {backend.device.upper()}")
    print(f"Hidden layers: {config.num_hidden_layers}")
    print(f"Attention heads: {config.num_attention_heads}")
    print(f"KV heads: {config.num_key_value_heads}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Head dim: {config.head_dim} ({config.head_dim_source})")
    print()
    print("=== PREFILL ===")
    print(f"Prompt: {prompt}")
    print(f"Prompt tokens: {prefill.logical_cached_token_count}")
    print(f"Input Token IDs: {list(prefill.input_token_ids)}")
    print(f"Cache type: {prefill_snapshot.cache_type}")
    print(f"Layer count: {prefill_snapshot.layer_count}")
    print(f"Runtime cached tokens: {prefill_snapshot.runtime_cached_token_count}")
    print(f"All layers structurally uniform: {prefill_snapshot.layers_are_uniform}")
    for layer_index in sample_indices:
        layer = prefill_snapshot.layers[layer_index]
        print(f"Layer {layer.layer_index}:")
        print(f"  K shape: {list(layer.key_shape)}")
        print(f"  V shape: {list(layer.value_shape)}")
        print(
            f"  K dtype/device: {layer.key_dtype} / {layer.key_device}; "
            f"element size: {layer.key_element_size} B"
        )
        print(
            f"  V dtype/device: {layer.value_dtype} / {layer.value_device}; "
            f"element size: {layer.value_element_size} B"
        )
        print(f"  K numel: {layer.key_numel}")
        print(f"  V numel: {layer.value_numel}")
        print(f"  K bytes: {format_byte_count(layer.key_bytes)}")
        print(f"  V bytes: {format_byte_count(layer.value_bytes)}")
    print(f"Total K: {format_byte_count(prefill_snapshot.total_key_bytes)}")
    print(f"Total V: {format_byte_count(prefill_snapshot.total_value_bytes)}")
    print(
        "Total raw KV tensors: "
        f"{format_byte_count(prefill_snapshot.raw_kv_tensor_bytes)}"
    )
    print(f"Raw KV per cached token: {format_byte_count(per_token_formula)}")
    print()
    print("Shape semantics: [Batch, KV Heads, Cached Tokens, Head Dim]")
    print(
        "Layer 0 values: "
        f"[{first_layer.key_shape[0]}, {first_layer.key_shape[1]}, "
        f"{first_layer.key_shape[2]}, {first_layer.key_shape[3]}]"
    )
    print()
    print("=== CACHE GROWTH ===")
    print("tokens     raw KV bytes")
    for snapshot in snapshots:
        print(
            f"{snapshot.logical_cached_token_count:<10} "
            f"{snapshot.raw_kv_tensor_bytes}"
        )
    print(f"Growth deltas: {list(growth)} B")
    print(f"Expected growth: {per_token_formula} B/token")
    linear_growth = bool(growth) and all(
        value == per_token_formula for value in growth
    )
    print(f"Linear growth: {linear_growth}")
    print()
    print("=== FORMULA CHECK ===")
    print(
        "Formula: layers * batch * KV heads * cached tokens * head dim * "
        "(K element bytes + V element bytes)"
    )
    print(f"Runtime measured: {prefill_snapshot.raw_kv_tensor_bytes} B")
    print(f"Config-derived: {theoretical_prefill} B")
    print(
        "Match: "
        f"{prefill_snapshot.raw_kv_tensor_bytes == theoretical_prefill}"
    )
    print()
    print("=== TOKEN / KV SIZE CONTEXT ===")
    print(
        "Token IDs theoretical minimum logical payload: "
        f"{token_payload.theoretical_payload_bits} bits / {token_bytes} B"
    )
    print(
        "Raw in-memory KV tensors: "
        f"{prefill_snapshot.raw_kv_tensor_bytes} B"
    )
    ratio = prefill_snapshot.raw_kv_tensor_bytes / token_bytes
    print(f"Raw-KV / theoretical-Token ratio: {ratio:.2f}x")
    print(
        "This is an educational magnitude comparison, not a measured network "
        "payload ratio; this inspection demo reports raw capacity and does not run "
        "the separate KV Wire Protocol v1 serializer."
    )
    print()
    print(
        "KV grows linearly because every cached token adds one K and one V "
        "vector per KV head in every layer."
    )
    print(
        "GQA stores 2 KV heads rather than 14 query/attention heads here, "
        "reducing cache capacity while query heads share K/V groups."
    )


def main() -> None:
    args = build_parser().parse_args()
    run_kv_inspection_demo(
        model_name=args.model,
        device=args.device,
        prompt=args.prompt,
        decode_steps=args.decode_steps,
    )


if __name__ == "__main__":
    main()
