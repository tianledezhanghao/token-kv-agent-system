"""Read a calibration profile and predict without loading a model."""

import argparse
from pathlib import Path

from token_kv_agent_system.real_cost import (
    MESSAGE_READY, REPRESENTATION_READY, RealCommunicationCostModel,
    RealCommunicationCostProfile, RealCostBasedSelector, RealSelectionContext,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--token-count", type=int, default=32)
    parser.add_argument("--bandwidths", type=float, nargs="+", default=(20, 100, 1000))
    parser.add_argument("--base-latency-ms", type=float, default=5.0)
    args = parser.parse_args()
    profile = RealCommunicationCostProfile.from_json(args.profile.read_text(encoding="utf-8"))
    model = RealCommunicationCostModel(profile)
    selector = RealCostBasedSelector(model)
    print(f"Profile: {profile.profile_name} / {profile.device_type} / {profile.model_id}")
    for boundary in (MESSAGE_READY, REPRESENTATION_READY):
        print(f"\n{boundary}: {args.token_count} tokens")
        print("Bandwidth Mbps | TOKEN est ms | KV est ms | Selected")
        for bandwidth in args.bandwidths:
            context = RealSelectionContext.from_profile(profile, args.token_count, bandwidth, args.base_latency_ms)
            decision = selector.select(context, boundary=boundary)
            kv_total = (decision.kv_estimate.message_ready_total_ms if boundary == MESSAGE_READY
                        else decision.kv_estimate.representation_ready_total_ms)
            print(f"{bandwidth:14g} | {decision.token_estimate.total_ms:12.3f} | {kv_total:9.3f} | {decision.selected_mode.value}")
        crossover = model.crossover_bandwidth_mbps(context, boundary=boundary)
        print(f"Crossover Mbps: {crossover:.6g}" if crossover is not None else "Crossover: No crossover")


if __name__ == "__main__":
    main()
