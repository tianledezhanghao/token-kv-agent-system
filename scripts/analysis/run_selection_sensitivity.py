"""Sweep a measured profile without executing a model or mutating calibration."""

import argparse
import json
from pathlib import Path

from token_kv_agent_system.real_cost import MESSAGE_READY, REPRESENTATION_READY, RealCommunicationCostProfile
from token_kv_agent_system.selection_analysis import sensitivity_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--token-counts", type=int, nargs="+", default=(8, 32, 128, 256, 512))
    parser.add_argument("--bandwidths", type=float, nargs="+", default=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000))
    parser.add_argument("--base-latencies", type=float, nargs="+", default=(0.5, 5, 20, 50))
    parser.add_argument("--near-tie-ms", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    profile = RealCommunicationCostProfile.from_json(args.profile.read_text(encoding="utf-8"))
    rows = sensitivity_rows(profile, boundaries=(MESSAGE_READY, REPRESENTATION_READY),
                            token_counts=tuple(args.token_counts), bandwidths=tuple(args.bandwidths),
                            base_latencies=tuple(args.base_latencies), near_tie_ms=args.near_tie_ms)
    if args.json:
        print(json.dumps([row.to_dict() for row in rows], indent=2))
        return
    print(f"Profile: {profile.profile_name} / {profile.device_type}. Exact/interpolated/extrapolated are profile positions, not new measurements.")
    print("Boundary | Tokens | Position | Base ms | Mbps | TOKEN ms | KV ms | Raw | Gap ms | Near tie | Crossover Mbps")
    for row in rows:
        cross = f"{row.crossover_bandwidth_mbps:.2f}" if row.crossover_bandwidth_mbps else "none"
        print(f"{row.boundary} | {row.token_count} | {row.calibration_position} | {row.base_latency_ms:g} | {row.bandwidth_mbps:g} | {row.token_estimated_ms:.3f} | {row.kv_estimated_ms:.3f} | {row.raw_selected_mode} | {row.decision_margin_ms:.3f} | {row.near_tie} | {cross}")


if __name__ == "__main__":
    main()
