"""Compatibility entry point; implementation: scripts/analysis/run_prediction_error_report.py."""

if __name__ == "__main__":
    from pathlib import Path
    import runpy

    runpy.run_path(str(Path(__file__).resolve().parent / "scripts/analysis/run_prediction_error_report.py"),
                   run_name="__main__")
else:
    from scripts.analysis.run_prediction_error_report import *  # noqa: F401,F403
