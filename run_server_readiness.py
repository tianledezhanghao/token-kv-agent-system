"""Inspect local prerequisites for single-server validation without claiming CUDA success."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from token_kv_agent_system.real_cost import RealCommunicationCostProfile


ENTRY_POINTS = ("run_token_handoff_demo.py", "run_kv_codec_demo.py", "run_kv_restore_demo.py",
                "run_kv_handoff_demo.py", "run_selector_calibration.py", "run_selector_validation.py",
                "run_real_adaptive_demo.py", "run_dashboard.py", "run_selection_sensitivity.py")


def assess_device(requested: str, torch_module) -> tuple[bool, str]:
    if torch_module is None:
        return False, "PyTorch unavailable"
    if requested == "cuda":
        available = bool(torch_module.cuda.is_available())
        return available, "CUDA available" if available else "CUDA unavailable on this machine"
    if requested == "mps":
        available = bool(getattr(torch_module.backends.mps, "is_available", lambda: False)())
        return available, "MPS available" if available else "MPS unavailable on this machine"
    return True, "CPU available"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--load-model", action="store_true", help="actually load model on requested device")
    parser.add_argument("--protocol-tests", action="store_true", help="run offline protocol pytest subset")
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    checks = {}
    for module in ("torch", "transformers", "pytest", "fastapi", "uvicorn"):
        checks[f"dependency_{module}"] = {"ok": importlib.util.find_spec(module) is not None}
    torch_module = None
    if checks["dependency_torch"]["ok"]:
        import torch
        torch_module = torch
    ok, detail = assess_device(args.device, torch_module)
    checks["requested_device"] = {"ok": ok, "detail": detail}
    checks["entry_points"] = {"ok": all((root / name).is_file() for name in ENTRY_POINTS),
                              "missing": [name for name in ENTRY_POINTS if not (root / name).is_file()]}
    ignored = subprocess.run(["git", "check-ignore", "-q", "benchmark-results/readiness-probe.json"], cwd=root).returncode == 0
    checks["benchmark_results_ignored"] = {"ok": ignored}
    core_sources = (root / "src/token_kv_agent_system/real_cost.py",
                    root / "src/token_kv_agent_system/real_adaptive_runtime.py",
                    root / "src/token_kv_agent_system/selection_stability.py",
                    root / "src/token_kv_agent_system/kv_handoff.py",
                    root / "src/token_kv_agent_system/token_handoff.py")
    local_paths = [str(path.relative_to(root)) for path in core_sources
                   if "/Users/" in path.read_text(encoding="utf-8")]
    checks["core_local_path_audit"] = {"ok": not local_paths, "files": local_paths}
    cost_selector_text = (root / "src/token_kv_agent_system/real_cost.py").read_text(encoding="utf-8")
    device_formula_branch = any(fragment in cost_selector_text for fragment in (
        'if self.device_type == "mps"', 'if self.device_type == "cuda"',
        'if context.device_type == "mps"', 'if context.device_type == "cuda"'))
    checks["shared_cost_formula"] = {"ok": not device_formula_branch}
    try:
        from token_kv_agent_system.real_cost import RealCommunicationCostModel, RealCostBasedSelector
        from token_kv_agent_system.real_adaptive_runtime import RealAdaptiveRuntime
        from token_kv_agent_system.token_protocol import estimate_token_wire_bytes
        from token_kv_agent_system.kv_protocol import estimate_kv_wire_size
        from token_kv_agent_system.synthetic_selection import synthetic_crossover_profile
        fixture = synthetic_crossover_profile()
        token_bytes = estimate_token_wire_bytes(8)
        kv_bytes = estimate_kv_wire_size(8, fixture.model, dtype=fixture.kv_dtype,
                                         source_device=fixture.source_device, batch_size=fixture.batch_size)
        checks["runtime_and_estimators"] = {"ok": bool(RealCommunicationCostModel and RealCostBasedSelector
            and RealAdaptiveRuntime and token_bytes == 42 and kv_bytes.total_wire_bytes > token_bytes),
            "token_wire_bytes_for_8": token_bytes, "synthetic_kv_wire_bytes_for_8": kv_bytes.total_wire_bytes}
    except ImportError as exc:
        checks["runtime_and_estimators"] = {"ok": False, "detail": str(exc)}
    loaded_profile = None
    if args.profile:
        try:
            profile = RealCommunicationCostProfile.from_json(args.profile.read_text(encoding="utf-8"))
            loaded_profile = profile
            checks["profile_compatibility"] = {"ok": profile.device_type == args.device and profile.model_id == args.model,
                                                "device": profile.device_type, "model": profile.model_id,
                                                "schema_version": profile.schema_version}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            checks["profile_compatibility"] = {"ok": False, "detail": str(exc)}
    if args.load_model and ok:
        try:
            from token_kv_agent_system.model import HuggingFaceModelBackend
            if args.offline:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            backend = HuggingFaceModelBackend(args.model, device=args.device)
            checks["model_load"] = {"ok": True, "device": backend.metadata.device}
            if loaded_profile is not None:
                config = backend.kv_model_config
                metadata = backend.metadata
                expected_source = f"{metadata.device}:0" if metadata.device in {"mps", "cuda"} else metadata.device
                compatible = (loaded_profile.source_device == expected_source and
                              loaded_profile.num_hidden_layers == config.num_hidden_layers and
                              loaded_profile.num_key_value_heads == config.num_key_value_heads and
                              loaded_profile.head_dim == config.head_dim and
                              loaded_profile.kv_dtype == metadata.dtype.removeprefix("torch."))
                checks["profile_compatibility"]["ok"] &= compatible
                checks["profile_compatibility"]["runtime_metadata_checked"] = True
        except Exception as exc:
            checks["model_load"] = {"ok": False, "detail": str(exc)}
    elif args.load_model:
        checks["model_load"] = {"ok": False, "detail": "requested device unavailable; model load skipped"}
    else:
        checks["model_load"] = {"ok": None, "detail": "not requested; use --load-model"}
    if args.protocol_tests:
        command = [sys.executable, "-m", "pytest", "-q", "tests/test_token_protocol.py", "tests/test_kv_protocol.py"]
        result = subprocess.run(command, cwd=root, capture_output=True, text=True)
        checks["protocol_tests"] = {"ok": result.returncode == 0, "detail": result.stdout[-1000:] or result.stderr[-1000:]}
    else:
        checks["protocol_tests"] = {"ok": None, "detail": "not requested; use --protocol-tests"}
    print(json.dumps({"requested_device": args.device, "offline": args.offline, "checks": checks}, indent=2, default=bool))
    return 0 if all(value["ok"] is not False for value in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
