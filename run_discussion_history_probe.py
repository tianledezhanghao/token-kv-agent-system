"""Compatibility entry point; implementation: scripts/legacy/run_discussion_history_probe.py."""

if __name__ == "__main__":
    from pathlib import Path
    import runpy

    runpy.run_path(str(Path(__file__).resolve().parent / "scripts/legacy/run_discussion_history_probe.py"),
                   run_name="__main__")
else:
    from scripts.legacy.run_discussion_history_probe import *  # noqa: F401,F403
