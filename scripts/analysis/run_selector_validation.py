"""Offline predict-first validation against both real paths."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from token_kv_agent_system.model import HuggingFaceModelBackend
from token_kv_agent_system.real_cost import RealCommunicationCostModel, RealCommunicationCostProfile
from token_kv_agent_system.selector_calibration import validate_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--measurement-runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    profile = RealCommunicationCostProfile.from_json(args.profile.read_text(encoding="utf-8"))
    backend = HuggingFaceModelBackend(profile.model_id, device=profile.device_type)
    model = RealCommunicationCostModel(profile)
    results = [validate_prediction(backend, model, count=count, bandwidth_mbps=bandwidth,
        measurement_runs=args.measurement_runs) for count, bandwidth in
        ((8, 20), (8, 1000), (128, 20), (128, 1000))]
    output = json.dumps([asdict(r) for r in results], indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
