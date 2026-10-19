"""Compatibility entry point; implementation: scripts/exports/export_kv_v3_diagnostics.py."""

if __name__ == "__main__":
    from pathlib import Path
    import runpy

    runpy.run_path(str(Path(__file__).resolve().parent / "scripts/exports/export_kv_v3_diagnostics.py"),
                   run_name="__main__")
else:
    from scripts.exports.export_kv_v3_diagnostics import *  # noqa: F401,F403
