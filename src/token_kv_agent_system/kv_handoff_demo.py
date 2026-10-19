"""Real single-process bidirectional KV handoff and same-span cost demo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .agent import LLMAgent
from .kv_handoff import (
    BidirectionalKVHandoffSession,
    CanonicalEquivalenceResult,
    CanonicalHandoffSpan,
    KVHandoffResult,
    KVScalingMeasurement,
    compare_canonical_token_kv,
    measure_kv_scaling,
)
from .model import DEFAULT_MODEL_NAME, GenerationConfig, HuggingFaceModelBackend
from .network import SimulatedNetwork
from .token_cost import (
    TokenCostMeasurementConfig,
    TokenHandoffMeasurement,
    measure_token_handoff_cost,
)
from .transport import InMemoryByteTransport


DEFAULT_PROMPT = "Explain KV cache communication in one short sentence."
DEFAULT_SUFFIX = " Continue briefly."
SCALING_SEED = "Canonical handoff preserves the same logical token sequence. "


@dataclass(frozen=True)
class KVHandoffDemoResult:
    a_to_b: KVHandoffResult
    b_to_a: KVHandoffResult
    equivalence: CanonicalEquivalenceResult
    token_same_span: TokenHandoffMeasurement
    token_scaling: tuple[TokenHandoffMeasurement, ...]
    kv_scaling: tuple[KVScalingMeasurement, ...]


def _networks(base_latency_ms: float) -> tuple[SimulatedNetwork, ...]:
    return (
        SimulatedNetwork(20.0, base_latency_ms=base_latency_ms),
        SimulatedNetwork(1_000.0, base_latency_ms=base_latency_ms),
    )


def _exact_scaling_ids(
    backend: HuggingFaceModelBackend, count: int
) -> tuple[int, ...]:
    seed = backend.encode(SCALING_SEED)
    if not seed:
        raise RuntimeError("scaling seed encoded to no Token IDs")
    repeats = (count + len(seed) - 1) // len(seed)
    return (seed * repeats)[:count]


def run_kv_handoff_demo(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "mps",
    prompt: str = DEFAULT_PROMPT,
    receiver_suffix: str = DEFAULT_SUFFIX,
    max_new_tokens: int = 8,
    measurement_runs: int = 3,
    base_latency_ms: float = 5.0,
) -> KVHandoffDemoResult:
    backend = HuggingFaceModelBackend(model_name, device=device)
    agent_a = LLMAgent("agent_a", backend)
    agent_b = LLMAgent("agent_b", backend)
    networks = _networks(base_latency_ms)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
    )
    initial = agent_a.generate_reply(prompt, generation_config)
    suffix_ids = backend.encode(receiver_suffix)
    if not initial.output_token_ids or not suffix_ids:
        raise RuntimeError("prompt generation and Receiver suffix must be non-empty")

    session = BidirectionalKVHandoffSession(
        agent_a, agent_b, InMemoryByteTransport(), networks
    )
    a_to_b = session.handoff_next(
        initial.output_token_ids,
        receiver_suffix_token_ids=suffix_ids,
        receiver_max_new_tokens=max_new_tokens,
    )
    b_to_a = session.handoff_next(
        a_to_b.receiver_output_token_ids,
        receiver_suffix_token_ids=suffix_ids,
        receiver_max_new_tokens=max_new_tokens,
    )
    equivalence = compare_canonical_token_kv(
        source=agent_a,
        target=agent_b,
        span=CanonicalHandoffSpan(initial.output_token_ids),
        receiver_suffix_token_ids=suffix_ids,
        transport=InMemoryByteTransport(),
    )

    token_config = TokenCostMeasurementConfig(
        warmup_runs=1,
        measurement_runs=measurement_runs,
        receiver_max_new_tokens=1,
    )
    token_same_span = measure_token_handoff_cost(
        source=agent_a,
        target=agent_b,
        token_ids=initial.output_token_ids,
        transport=InMemoryByteTransport(),
        networks=networks,
        config=token_config,
    )
    scaling_spans = tuple(
        CanonicalHandoffSpan(_exact_scaling_ids(backend, count))
        for count in (8, 32, 128)
    )
    token_scaling = tuple(
        measure_token_handoff_cost(
            source=agent_a,
            target=agent_b,
            token_ids=span.handoff_token_ids,
            transport=InMemoryByteTransport(),
            networks=networks,
            config=token_config,
        )
        for span in scaling_spans
    )
    kv_scaling = measure_kv_scaling(
        source=agent_a,
        target=agent_b,
        spans=scaling_spans,
        receiver_suffix_token_ids=suffix_ids,
        transport=InMemoryByteTransport(),
        networks=networks,
        measurement_runs=measurement_runs,
    )
    return KVHandoffDemoResult(
        a_to_b=a_to_b,
        b_to_a=b_to_a,
        equivalence=equivalence,
        token_same_span=token_same_span,
        token_scaling=token_scaling,
        kv_scaling=kv_scaling,
    )


def _print_handoff(label: str, result: KVHandoffResult) -> None:
    print(f"\n=== {label} ===")
    print(f"Direction: {result.source_agent_id} -> {result.target_agent_id}")
    print(f"Tokens: {result.token_count}")
    print(f"Token IDs: {list(result.handoff_token_ids)}")
    print(f"Sender clean Prefill: {result.sender_kv_prepare_ms:.3f} ms")
    print(f"Cached tokens: {result.logical_cached_tokens}")
    print(f"Materialization forward: {result.materialization_forward_ms}")
    print(f"Raw KV: {result.raw_kv_tensor_bytes} B")
    print(
        f"Wire: {result.wire_total_bytes} B "
        f"(body={result.wire_tensor_body_bytes}, metadata={result.wire_metadata_bytes})"
    )
    print(
        "Serialize: "
        f"device->host={result.device_to_host_ms:.3f} ms, "
        f"encode={result.protocol_encode_ms:.3f} ms, "
        f"total={result.total_serialize_ms:.3f} ms"
    )
    print(f"Local transfer: {result.local_transfer_ms:.6f} ms (diagnostic only)")
    print(f"Deserialize: {result.deserialize_ms:.3f} ms")
    print(
        "Restore: "
        f"CPU={result.cpu_tensor_reconstruction_ms:.3f} ms, "
        f"host->device={result.host_to_device_ms:.3f} ms, "
        f"cache={result.cache_construction_ms:.3f} ms, "
        f"total={result.total_restore_ms:.3f} ms"
    )
    print("Receiver prefix Prefill: NOT EXECUTED")
    for link in result.simulated_links:
        print(
            f"{link.bandwidth_mbps:g} Mbps: link={link.simulated_link_latency_ms:.3f} ms, "
            f"KV ready={link.kv_handoff_cost_ms:.3f} ms, "
            f"KV message-ready E2E={link.kv_end_to_end_representation_cost_ms:.3f} ms"
        )
    print(f"Receiver output IDs: {list(result.receiver_output_token_ids)}")
    print(f"Receiver text: {result.receiver_generated_text}")


def _print_same_span(result: KVHandoffDemoResult) -> None:
    print("\n=== TOKEN VS KV SAME-SPAN COMPARISON ===")
    token = result.token_same_span
    kv = result.a_to_b
    print(f"Tokens: {token.token_count}")
    print(f"TOKEN wire: {token.wire_total_bytes} B")
    print(f"KV wire: {kv.wire_total_bytes} B")
    print(f"KV/TOKEN wire-size ratio: {kv.wire_total_bytes / token.wire_total_bytes:.3f}x")
    for token_link, kv_link in zip(token.simulated_links, kv.simulated_links):
        print(
            f"{token_link.bandwidth_mbps:g} Mbps | "
            f"TOKEN message/ready={token_link.token_handoff_cost_ms:.3f} ms | "
            f"KV ready={kv_link.kv_handoff_cost_ms:.3f} ms | "
            f"KV message-ready E2E={kv_link.kv_end_to_end_representation_cost_ms:.3f} ms"
        )


def _print_scaling(result: KVHandoffDemoResult) -> None:
    print("\n=== CONTEXT SCALING ===")
    print(
        "tokens | TOKEN wire/prefill/ser/deser | TOKEN total 20/1000 | "
        "KV wire/prepare/ser/deser/restore | KV ready 20/1000 | "
        "KV E2E 20/1000 | wire ratio"
    )
    for token, kv in zip(result.token_scaling, result.kv_scaling):
        token_20, token_1000 = token.simulated_links
        kv_20, kv_1000 = kv.links
        print(
            f"{token.token_count:>6} | {token.wire_total_bytes} B/"
            f"{token.receiver_prefill_median_ms:.3f}/{token.serialization_median_ms:.3f}/"
            f"{token.deserialization_median_ms:.3f} ms | "
            f"{token_20.token_handoff_cost_ms:.3f}/{token_1000.token_handoff_cost_ms:.3f} ms | "
            f"{kv.wire_total_bytes} B/{kv.sender_kv_prepare_median_ms:.3f}/"
            f"{kv.total_serialize_median_ms:.3f}/{kv.deserialize_median_ms:.3f}/"
            f"{kv.total_restore_median_ms:.3f} ms | "
            f"{kv_20.kv_handoff_cost_ms:.3f}/{kv_1000.kv_handoff_cost_ms:.3f} ms | "
            f"{kv_20.kv_end_to_end_representation_cost_ms:.3f}/"
            f"{kv_1000.kv_end_to_end_representation_cost_ms:.3f} ms | "
            f"{kv.wire_total_bytes / token.wire_total_bytes:.3f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="mps"
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--receiver-suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--measurement-runs", type=int, default=3)
    parser.add_argument("--base-latency-ms", type=float, default=5.0)
    args = parser.parse_args()
    result = run_kv_handoff_demo(
        model_name=args.model,
        device=args.device,
        prompt=args.prompt,
        receiver_suffix=args.receiver_suffix,
        max_new_tokens=args.max_new_tokens,
        measurement_runs=args.measurement_runs,
        base_latency_ms=args.base_latency_ms,
    )
    print("=== CANONICAL HANDOFF ===")
    print(f"Message Token IDs: {list(result.a_to_b.handoff_token_ids)}")
    print(f"Tokens: {result.a_to_b.token_count}")
    _print_handoff("A -> B", result.a_to_b)
    _print_handoff("B -> A", result.b_to_a)
    oracle = result.equivalence
    print("\n=== CANONICAL EQUIVALENCE ORACLE ===")
    print(f"TOKEN path next Token: {oracle.token_path_next_token_id}")
    print(f"KV path next Token: {oracle.kv_path_next_token_id}")
    print(f"Full recompute next Token: {oracle.full_recompute_next_token_id}")
    print(f"All IDs matched: {oracle.all_next_token_ids_equal}")
    print(
        f"TOKEN vs KV logits max/mean: {oracle.token_vs_kv_max_abs_diff:.8f}/"
        f"{oracle.token_vs_kv_mean_abs_diff:.8f}"
    )
    print(
        f"KV vs Full logits max/mean: {oracle.kv_vs_full_max_abs_diff:.8f}/"
        f"{oracle.kv_vs_full_mean_abs_diff:.8f}"
    )
    _print_same_span(result)
    _print_scaling(result)


if __name__ == "__main__":
    main()
