"""Compatibility entry point; implementation: scripts/analysis/run_selector_validation.py."""

if __name__ == "__main__":
    from pathlib import Path
    import runpy

    runpy.run_path(str(Path(__file__).resolve().parent / "scripts/analysis/run_selector_validation.py"),
                   run_name="__main__")
else:
    from scripts.analysis.run_selector_validation import *  # noqa: F401,F403
