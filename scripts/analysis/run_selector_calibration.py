"""Offline real device calibration; writes only when --output is supplied."""

import argparse
from pathlib import Path

from token_kv_agent_system.model import DEFAULT_MODEL_NAME, HuggingFaceModelBackend
from token_kv_agent_system.selector_calibration import calibrate_backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--profile-name")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measurement-runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    backend = HuggingFaceModelBackend(args.model, device=args.device)
    profile = calibrate_backend(backend, profile_name=args.profile_name or
        ("macbook-development" if args.device == "mps" else f"{args.device}-calibration"),
        warmup_runs=args.warmup_runs, measurement_runs=args.measurement_runs)
    print(profile.to_json())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(profile.to_json() + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
