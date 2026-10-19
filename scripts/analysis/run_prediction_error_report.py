"""Summarize saved real adaptive results; does not alter a profile."""

import argparse
import json
from pathlib import Path

from token_kv_agent_system.selection_analysis import prediction_error_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="JSON list of AdaptiveHandoffResult objects")
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    print(json.dumps(prediction_error_summary(data), indent=2))


if __name__ == "__main__":
    main()
