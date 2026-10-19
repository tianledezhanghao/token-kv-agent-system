"""CLI for real Token communication cost and payload scaling measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import LLMAgent
from .model import DEFAULT_MODEL_NAME, GenerationConfig, HuggingFaceModelBackend
from .network import SimulatedNetwork
from .token_cost import (
    TokenCostMeasurementConfig,
    TokenHandoffMeasurement,
    measure_token_handoff_cost,
)
from .transport import InMemoryByteTransport


DEFAULT_PROMPT = "Explain Token communication in one short sentence."
SCALING_SEED = "Token communication preserves exact model vocabulary identifiers. "


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure real Token handoff costs with virtual network links"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measurement-runs", type=int, default=5)
    parser.add_argument("--base-latency-ms", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    return parser


def _networks(base_latency_ms: float) -> tuple[SimulatedNetwork, ...]:
    return (
        SimulatedNetwork(20, base_latency_ms=base_latency_ms),
        SimulatedNetwork(1_000, base_latency_ms=base_latency_ms),
    )


def _measure(
    agent_a: LLMAgent,
    agent_b: LLMAgent,
    token_ids: tuple[int, ...],
    config: TokenCostMeasurementConfig,
    base_latency_ms: float,
) -> TokenHandoffMeasurement:
    return measure_token_handoff_cost(
        source=agent_a,
        target=agent_b,
        token_ids=token_ids,
        transport=InMemoryByteTransport(),
        networks=_networks(base_latency_ms),
        config=config,
    )


def _exact_scaling_ids(backend: HuggingFaceModelBackend, count: int) -> tuple[int, ...]:
    seed_ids = backend.encode(SCALING_SEED)
    if not seed_ids:
        raise RuntimeError("scaling seed encoded to no Token IDs")
    repeats = (count + len(seed_ids) - 1) // len(seed_ids)
    return (seed_ids * repeats)[:count]


def _print_measurement(result: TokenHandoffMeasurement) -> None:
    print("=== TOKEN PAYLOAD ===")
    print(f"Tokens: {result.token_count}")
    print(f"Theoretical logical: {result.theoretical_token_payload_bytes} B")
    print(f"Protocol header: {result.wire_header_bytes} B")
    print(f"Protocol body: {result.wire_body_bytes} B")
    print(f"Wire: {result.wire_total_bytes} B")
    print()
    print("=== LOCAL MEASUREMENTS ===")
    print(f"Serialize median: {result.serialization_median_ms:.6f} ms")
    print(f"In-memory transfer: {result.local_byte_transfer_ms:.6f} ms")
    print("(diagnostic only; not network latency or formal handoff cost)")
    print(f"Deserialize median: {result.deserialization_median_ms:.6f} ms")
    if result.first_warmup_prefill_ms is not None:
        print(f"First warmup Prefill: {result.first_warmup_prefill_ms:.3f} ms")
    print(f"Receiver Prefill median: {result.receiver_prefill_median_ms:.3f} ms")
    print(f"Receiver Decode forwards/run: {result.receiver_decode_forward_count}")
    if result.receiver_decode_mean_ms is not None:
        print(f"Receiver Decode mean: {result.receiver_decode_mean_ms:.3f} ms")
        print(f"Receiver Decode median: {result.receiver_decode_median_ms:.3f} ms")
    print()
    print("=== SIMULATED LINK ===")
    for link in result.simulated_links:
        print(
            f"{link.bandwidth_mbps:g} Mbps: "
            f"{link.simulated_link_latency_ms:.6f} ms "
            f"(base {link.base_latency_ms:g} ms, no sleep)"
        )
    print()
    print("=== TOKEN HANDOFF COST ===")
    for link in result.simulated_links:
        print(
            f"{link.bandwidth_mbps:g} Mbps: serialize + simulated link + "
            f"deserialize + Prefill = {link.token_handoff_cost_ms:.6f} ms"
        )
    print(f"Sender/restored IDs exact match: {result.sender_ids_match_received}")
    print(
        "Restored/Receiver Prefill IDs exact match: "
        f"{result.received_ids_match_prefill_input}"
    )


def run_token_cost_demo(
    *,
    model_name: str,
    device: str,
    prompt: str,
    max_new_tokens: int,
    warmup_runs: int,
    measurement_runs: int,
    base_latency_ms: float,
    output: Path | None = None,
) -> tuple[TokenHandoffMeasurement, ...]:
    backend = HuggingFaceModelBackend(model_name, device=device)
    agent_a = LLMAgent("agent_a", backend)
    agent_b = LLMAgent("agent_b", backend)
    generation = agent_a.generate_reply(
        prompt,
        GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
        ),
    )
    config = TokenCostMeasurementConfig(
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        receiver_max_new_tokens=max_new_tokens,
    )

    payloads = [generation.output_token_ids]
    payloads.extend(_exact_scaling_ids(backend, count) for count in (32, 128))
    results = tuple(
        _measure(agent_a, agent_b, token_ids, config, base_latency_ms)
        for token_ids in payloads
    )

    print("=== BENCHMARK CONTEXT ===")
    print(f"Model: {backend.model_name}")
    print(f"Device: {backend.device}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Measurement runs: {measurement_runs}")
    print("Sender generation is excluded from Token handoff cost.")
    print()
    _print_measurement(results[0])
    print()
    print("=== PAYLOAD SCALING (constructed legal tokenizer IDs) ===")
    print("tokens | wire B | 20 Mbps link/cost ms | 1000 Mbps link/cost ms | prefill ms")
    for result in results:
        slow, fast = result.simulated_links
        print(
            f"{result.token_count:>6} | {result.wire_total_bytes:>6} | "
            f"{slow.simulated_link_latency_ms:.6f}/{slow.token_handoff_cost_ms:.3f} | "
            f"{fast.simulated_link_latency_ms:.6f}/{fast.token_handoff_cost_ms:.3f} | "
            f"{result.receiver_prefill_median_ms:.3f}"
        )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps([result.to_dict() for result in results], indent=2),
            encoding="utf-8",
        )
        print(f"Wrote JSON: {output}")
    return results


def main() -> None:
    args = build_parser().parse_args()
    run_token_cost_demo(
        model_name=args.model,
        device=args.device,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        measurement_runs=args.measurement_runs,
        base_latency_ms=args.base_latency_ms,
        output=args.output,
    )


if __name__ == "__main__":
    main()
