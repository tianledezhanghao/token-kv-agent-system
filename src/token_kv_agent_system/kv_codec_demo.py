"""Real Qwen/MPS KV Wire Protocol v1 round-trip demonstration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .kv_protocol import (
    KV_PROTOCOL_MAGIC,
    KV_PROTOCOL_VERSION,
    KVDeserializationResult,
    KVSerializationResult,
    NeutralKVPayload,
    deserialize_kv_packet,
)
from .model import (
    DEFAULT_MODEL_NAME,
    HuggingFaceModelBackend,
    KVCacheSnapshot,
    extract_huggingface_kv_payload,
    inspect_kv_cache,
    serialize_huggingface_kv_cache,
)


DEFAULT_PROMPT = "Explain KV cache in one short sentence."
DEFAULT_LONG_PROMPT = (
    "Explain in a few concise sentences how a transformer KV cache avoids repeated "
    "attention computation during autoregressive decoding and why its memory grows "
    "with context length."
)


@dataclass(frozen=True)
class RoundTripEvidence:
    prompt: str
    prompt_tokens: int
    snapshot: KVCacheSnapshot
    serialization: KVSerializationResult
    deserialization: KVDeserializationResult
    key_shapes_exact: bool
    value_shapes_exact: bool
    dtypes_exact: bool
    key_bits_exact: bool
    value_bits_exact: bool
    exact_layer_count: int

    @property
    def round_trip_exact(self) -> bool:
        return all(
            (
                self.key_shapes_exact,
                self.value_shapes_exact,
                self.dtypes_exact,
                self.key_bits_exact,
                self.value_bits_exact,
                self.exact_layer_count == self.snapshot.layer_count,
                self.deserialization.payload.logical_cached_token_count
                == self.snapshot.logical_cached_token_count,
            )
        )


def run_round_trip(
    backend: HuggingFaceModelBackend, prompt: str
) -> RoundTripEvidence:
    """Prefill once, serialize that exact cache, then compare every restored layer."""

    prefill = backend.prefill(prompt)
    snapshot = inspect_kv_cache(prefill.cache, prefill.logical_cached_token_count)
    serialization = serialize_huggingface_kv_cache(
        prefill.cache,
        logical_cached_token_count=prefill.logical_cached_token_count,
        model_id=backend.metadata.model_name,
        model_config=backend.kv_model_config,
    )
    deserialization = deserialize_kv_packet(serialization.packet)
    # This second adapter read is verification only and is outside serialization timing.
    source, _, _ = extract_huggingface_kv_payload(
        prefill.cache,
        logical_cached_token_count=prefill.logical_cached_token_count,
        model_id=backend.metadata.model_name,
        model_config=backend.kv_model_config,
    )
    restored = deserialization.payload
    comparisons = [
        _compare_layer(source, restored, index) for index in range(len(source.layers))
    ]
    return RoundTripEvidence(
        prompt=prompt,
        prompt_tokens=len(prefill.input_token_ids),
        snapshot=snapshot,
        serialization=serialization,
        deserialization=deserialization,
        key_shapes_exact=all(item[0] for item in comparisons),
        value_shapes_exact=all(item[1] for item in comparisons),
        dtypes_exact=all(item[2] for item in comparisons),
        key_bits_exact=all(item[3] for item in comparisons),
        value_bits_exact=all(item[4] for item in comparisons),
        exact_layer_count=sum(all(item) for item in comparisons),
    )


def _compare_layer(
    source: NeutralKVPayload, restored: NeutralKVPayload, index: int
) -> tuple[bool, bool, bool, bool, bool]:
    source_layer = source.layers[index]
    restored_layer = restored.layers[index]
    return (
        source_layer.key.shape == restored_layer.key.shape,
        source_layer.value.shape == restored_layer.value.shape,
        source_layer.key.dtype == restored_layer.key.dtype
        and source_layer.value.dtype == restored_layer.value.dtype,
        source_layer.key.data == restored_layer.key.data,
        source_layer.value.data == restored_layer.value.data,
    )


def _print_primary(backend: HuggingFaceModelBackend, result: RoundTripEvidence) -> None:
    first = result.snapshot.layers[0]
    wire = result.serialization
    decoded = result.deserialization
    config = backend.kv_model_config
    print("=== MODEL ===")
    print(f"Model: {backend.metadata.model_name}")
    print(f"Device: {backend.metadata.device}")
    print(f"Cache: {result.snapshot.cache_type}")
    print(f"Layers: {result.snapshot.layer_count}")
    print(f"KV Heads: {config.num_key_value_heads}")
    print(f"Head Dim: {config.head_dim}")
    print("\n=== SOURCE CACHE ===")
    print(f"Prompt: {result.prompt}")
    print(f"Prompt tokens: {result.prompt_tokens}")
    print(f"Logical cached tokens: {result.snapshot.logical_cached_token_count}")
    print(f"Layer 0 K: {list(first.key_shape)}")
    print(f"Layer 0 V: {list(first.value_shape)}")
    print(f"dtype: {first.key_dtype}")
    print(f"Raw KV tensor bytes: {wire.raw_kv_tensor_bytes}")
    print("\n=== KV PROTOCOL V1 ===")
    print(f"Magic: {KV_PROTOCOL_MAGIC.decode('ascii')}")
    print(f"Version: {KV_PROTOCOL_VERSION}")
    print(f"Model compatibility: {decoded.payload.model}")
    print(f"Tensor body: {wire.wire_tensor_body_bytes} B")
    print(f"Metadata: {wire.wire_metadata_bytes} B")
    print(f"Total wire: {wire.total_wire_bytes} B")
    print(f"Wire overhead: {wire.wire_overhead_ratio:.6%}")
    print("\n=== SERIALIZATION ===")
    print(f"Device -> Host: {wire.device_to_host_ms:.3f} ms")
    print(f"Protocol encode: {wire.protocol_encode_ms:.3f} ms")
    print(f"Total serialize: {wire.total_serialize_ms:.3f} ms")
    print(f"Deserialize: {decoded.deserialize_ms:.3f} ms")
    print("\n=== ROUND TRIP ===")
    print(f"Layers: {result.exact_layer_count} / {result.snapshot.layer_count}")
    print(f"K shapes exact: {result.key_shapes_exact}")
    print(f"V shapes exact: {result.value_shapes_exact}")
    print(f"Dtypes exact: {result.dtypes_exact}")
    print(f"K bit patterns exact: {result.key_bits_exact}")
    print(f"V bit patterns exact: {result.value_bits_exact}")
    print(
        "Logical cache length preserved: "
        f"{decoded.payload.logical_cached_token_count == result.prompt_tokens}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="mps", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--long-prompt", default=DEFAULT_LONG_PROMPT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backend = HuggingFaceModelBackend(args.model, device=args.device)
    primary = run_round_trip(backend, args.prompt)
    _print_primary(backend, primary)
    second = run_round_trip(backend, args.long_prompt)
    print("\n=== SECOND CONTEXT ===")
    print(f"Tokens: {second.prompt_tokens}")
    print(f"Logical cached tokens: {second.snapshot.logical_cached_token_count}")
    print(f"Tensor body: {second.serialization.wire_tensor_body_bytes} B")
    print(f"Metadata: {second.serialization.wire_metadata_bytes} B")
    print(f"Total wire: {second.serialization.total_wire_bytes} B")
    print(f"Device -> Host: {second.serialization.device_to_host_ms:.3f} ms")
    print(f"Protocol encode: {second.serialization.protocol_encode_ms:.3f} ms")
    print(f"Total serialize: {second.serialization.total_serialize_ms:.3f} ms")
    print(f"Deserialize: {second.deserialization.deserialize_ms:.3f} ms")
    print(f"K bit patterns exact: {second.key_bits_exact}")
    print(f"V bit patterns exact: {second.value_bits_exact}")
    print(f"Round-trip: {second.round_trip_exact}")
    print("\n=== IMPORTANT ===")
    print("This demo does NOT restore the packet into a Receiver model.")
    print("Receiver cache reconstruction and inference continuation belong to Phase 5.2.")


if __name__ == "__main__":
    main()
