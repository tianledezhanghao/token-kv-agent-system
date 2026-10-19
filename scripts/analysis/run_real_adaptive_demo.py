"""Run predict-first, selected-path-only real adaptive handoffs."""

import argparse
from pathlib import Path

from token_kv_agent_system.agent import LLMAgent
from token_kv_agent_system.model import DEFAULT_MODEL_NAME, HuggingFaceModelBackend
from token_kv_agent_system.network import SimulatedNetwork
from token_kv_agent_system.real_adaptive_runtime import RealAdaptiveRuntime
from token_kv_agent_system.selection_stability import SelectionStabilityConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="mps", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--selector-profile", type=Path, required=True)
    parser.add_argument("--bandwidth-mbps", type=float, default=20)
    parser.add_argument("--base-latency-ms", type=float, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--prompt", default="Explain KV cache in one short sentence.")
    parser.add_argument("--enable-stability", action="store_true")
    parser.add_argument("--min-absolute-saving-ms", type=float, default=0)
    parser.add_argument("--min-relative-saving", type=float, default=0)
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    backend = HuggingFaceModelBackend(args.model, device=args.device)
    runtime = RealAdaptiveRuntime.from_profile_path(
        LLMAgent("Agent A", backend), LLMAgent("Agent B", backend),
        args.selector_profile,
        SimulatedNetwork(args.bandwidth_mbps, args.base_latency_ms),
        initial_prompt=args.prompt, max_new_tokens=args.max_new_tokens,
        stability_config=SelectionStabilityConfig(args.enable_stability,
            args.min_absolute_saving_ms, args.min_relative_saving))
    for index in range(args.rounds):
        result = runtime.communicate_next()
        print(f"=== REAL ADAPTIVE COMMUNICATION #{index + 1} ===")
        print(f"{result.source} -> {result.target}; canonical tokens: {result.canonical_token_count}")
        print(f"Sender: {result.sender_text}")
        print(f"Network: {result.bandwidth_mbps:g} Mbps / {result.base_latency_ms:g} ms base")
        print(f"Estimated before execution: TOKEN {result.token_estimated_ms:.3f} ms; "
              f"KV {result.kv_estimated_ms:.3f} ms; selected {result.selected_mode}")
        print(f"Estimated saving: {result.estimated_saving_ms:.3f} ms")
        if result.stability:
            print(f"Stability: {result.stability}")
        print(f"Executed path: {result.actually_executed_mode}; "
              f"unselected path executed: False")
        print(f"Selected measurement: {result.selected_measurement}")
        print(f"Actual MESSAGE_READY: {result.actual_selected_cost_ms:.3f} ms; "
              f"absolute prediction error: {result.absolute_error_ms:.3f} ms "
              f"({result.relative_error:.2%})")
        print(f"Receiver: {result.receiver_text}")


if __name__ == "__main__":
    main()
