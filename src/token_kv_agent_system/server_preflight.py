"""Read-only environment/config inspection and opt-in staged model smoke tests.

This is a migration preflight, not a CUDA benchmark or full 7.1 acceptance report.
No checkpoint is selected implicitly and no model loads in the default stage.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import inspect
import json
import os
import platform
import sys


def inspect_environment(torch_module=None, transformers_module=None) -> dict:
    if torch_module is None:
        import torch as torch_module
    if transformers_module is None:
        import transformers as transformers_module
    cuda = torch_module.cuda
    available = bool(cuda.is_available())
    gpus = []
    for index in range(cuda.device_count() if available else 0):
        props = cuda.get_device_properties(index)
        free, total = cuda.mem_get_info(index)
        gpus.append({"visible_device": f"cuda:{index}", "name": props.name,
                     "total_memory_bytes": total, "free_memory_bytes": free,
                     "allocated_memory_bytes": cuda.memory_allocated(index)})
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch_module.__version__, "transformers": transformers_module.__version__,
        "torch_cuda_build": torch_module.version.cuda, "cuda_available": available,
        "visible_gpu_count": len(gpus), "gpus": gpus,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_id_semantics": "process-visible logical CUDA IDs, possibly remapped by CUDA_VISIBLE_DEVICES",
        "dynamic_cache_constructor": str(inspect.signature(transformers_module.DynamicCache)),
        "dynamic_cache_update": str(inspect.signature(transformers_module.DynamicCache.update)),
    }


def inspect_checkpoint(model_name: str, *, local_files_only: bool = True) -> dict:
    from transformers import AutoConfig, AutoTokenizer
    from .model.cache_inspection import inspect_kv_model_config

    config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    structure = inspect_kv_model_config(config)
    return {
        "model": model_name, "model_type": config.model_type,
        "declared_architectures": config.architectures,
        "configured_dtype": str(getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)),
        "structure": asdict(structure), "vocab_size": config.vocab_size,
        "attention_kind": ("MQA" if structure.num_key_value_heads == 1 and structure.num_attention_heads > 1
                           else "GQA" if structure.num_key_value_heads < structure.num_attention_heads else "MHA"),
        "tokenizer_class": type(tokenizer).__name__, "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_length": len(tokenizer), "special_token_ids": tokenizer.all_special_ids,
        "loading": "AutoTokenizer + AutoModelForCausalLM.from_pretrained; model.to(single_device); no device_map",
        "runtime_dtype_and_cache": "pending actual model load/forward",
    }


def single_model_smoke(backend, prompt: str, max_new_tokens: int) -> dict:
    from .kv_codec_demo import run_round_trip
    from .model import GenerationConfig, inspect_kv_cache, config_derived_raw_kv_tensor_bytes
    from .kv_protocol import deserialize_kv_packet

    generation = GenerationConfig(max_new_tokens=max_new_tokens, repetition_penalty=1.0)
    hf = backend.generate(prompt, generation)
    manual = backend.manual_generate(prompt, generation, warmup_runs=0)
    if manual.input_token_ids != hf.input_token_ids or manual.output_token_ids != hf.output_token_ids:
        raise RuntimeError("manual/HF greedy Token IDs differ; inspect masks, EOS and generation processors")
    prefill = backend.prefill(input_token_ids=hf.input_token_ids)
    snapshots = [inspect_kv_cache(prefill.cache, len(hf.input_token_ids))]
    cache, count, token_id = prefill.cache, len(hf.input_token_ids), prefill.first_token_id
    for _ in range(3):
        step = backend.decode_step(token_id, previous_cache=cache, logical_cached_token_count=count)
        cache, count, token_id = step.cache, step.logical_cached_token_count, step.predicted_next_token_id
        snapshots.append(inspect_kv_cache(cache, count))
    for index, snapshot in enumerate(snapshots):
        if snapshot.runtime_cached_token_count != len(hf.input_token_ids) + index:
            raise RuntimeError("unexpected cache growth/pending-token semantics")
        if snapshot.raw_kv_tensor_bytes != config_derived_raw_kv_tensor_bytes(backend.kv_model_config, snapshot):
            raise RuntimeError("runtime KV byte count differs from config-derived formula")
    codec = run_round_trip(backend, prompt)
    if not codec.round_trip_exact:
        raise RuntimeError("KV Protocol v1 round-trip is not bitwise exact")
    # Restore comes from parsed wire bytes, never a Sender tensor/cache reference.
    restored = backend.restore_kv_cache(deserialize_kv_packet(codec.serialization.packet).payload)
    original = backend.prefill(input_token_ids=hf.input_token_ids)
    suffix = (original.first_token_id,)
    original_next = backend.continue_from_cache(suffix, previous_cache=original.cache,
                                                logical_cached_token_count=len(hf.input_token_ids))
    restored_next = backend.continue_from_cache(suffix, previous_cache=restored.cache,
                                                logical_cached_token_count=len(hf.input_token_ids))
    if original_next.predicted_next_token_id != restored_next.predicted_next_token_id:
        raise RuntimeError("original/restored continuation Token ID mismatch")
    return {
        "prompt": prompt, "input_token_ids": hf.input_token_ids,
        "generated_token_ids": hf.output_token_ids, "manual_hf_ids_exact": True,
        "logits_shape": prefill.logits_shape,
        "prefill_ms": prefill.prefill_forward_ms,
        "cache_snapshots": [asdict(snapshot) | {"runtime_cached_token_count": snapshot.runtime_cached_token_count}
                            for snapshot in snapshots],
        "kv_bytes_per_cached_token": snapshots[0].raw_kv_tensor_bytes / len(hf.input_token_ids),
        "raw_kv_bytes": codec.serialization.raw_kv_tensor_bytes,
        "body_bytes": codec.serialization.wire_tensor_body_bytes,
        "metadata_bytes": codec.serialization.wire_metadata_bytes,
        "wire_bytes": codec.serialization.total_wire_bytes,
        "all_layers_bitwise_exact": codec.round_trip_exact,
        "restore_api": "DynamicCache(ddp_cache_data=..., config=...)",
        "restored_one_token_id_exact": True,
        "remaining_server_acceptance": ["multi-token suffix and multi-step original/restored/full-recompute oracle",
                                        "actual checkpoint CUDA placement and cross-device evidence"],
    }


def dual_handoff_smoke(pair, prompt: str, max_new_tokens: int) -> dict:
    from .agent import LLMAgent
    from .model import GenerationConfig
    from .token_handoff import TokenHandoffSession
    from .kv_handoff import CanonicalHandoffSpan, perform_kv_handoff
    from .network import SimulatedNetwork
    from .transport import InMemoryByteTransport

    agents = (LLMAgent("Agent A", pair.agent_a_backend), LLMAgent("Agent B", pair.agent_b_backend))
    transport = InMemoryByteTransport()
    token_session = TokenHandoffSession(agents, transport)
    config = GenerationConfig(max_new_tokens=max_new_tokens, repetition_penalty=1.0)
    results = []
    for source, target in (agents, tuple(reversed(agents))):
        token = token_session.generate_and_handoff(source.agent_id, target.agent_id, prompt,
                                                   sender_config=config, receiver_config=config)
        # Receiver-owned suffix is encoded once locally; the handoff IDs are never retokenized.
        suffix = target.backend.encode(" Continue.")
        kv = perform_kv_handoff(source=source, target=target,
            span=CanonicalHandoffSpan(token.sender_output_token_ids), receiver_suffix_token_ids=suffix,
            transport=transport, networks=(SimulatedNetwork(1000, base_latency_ms=5),),
            receiver_max_new_tokens=max_new_tokens)
        results.append({"source_device": source.backend.device, "target_device": target.backend.device,
                        "token_handoff": asdict(token), "canonical_kv_baseline": kv.to_dict()})
    return {"backends": pair.to_dict(), "directions": results,
            "transport": type(transport).__name__, "transfer_records": [asdict(r) for r in transport.records],
            "status": "smoke only; full real-server acceptance remains required"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="7.1A staged server/Llama migration preflight (no weights by default)")
    parser.add_argument("--stage", choices=("environment", "config", "single", "dual"), default="environment")
    parser.add_argument("--model", help="actual model ID or local checkpoint path; no assumed Llama default")
    parser.add_argument("--agent-a-device", default="cuda:0")
    parser.add_argument("--agent-b-device", default="cuda:1")
    parser.add_argument("--allow-download", action="store_true", help="allow HF downloads; default is local cache only")
    parser.add_argument("--prompt", default="Explain KV cache in one short sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    def show(label, value):
        print(f"=== {label} ===", flush=True)
        print(json.dumps(value, indent=2, ensure_ascii=False), flush=True)
    try:
        from .model import HuggingFaceModelBackend, IndependentBackendConfig, IndependentBackends
        from .model.huggingface import resolve_device
        import torch

        config = IndependentBackendConfig(args.model or "environment-only", args.agent_a_device, args.agent_b_device)
        if args.max_new_tokens <= 0:
            raise ValueError("max-new-tokens must be positive")
        show("SERVER", inspect_environment())
        if args.stage == "environment":
            show("PLAN", {"agent_a_device": config.agent_a_device, "agent_b_device": config.agent_b_device,
                          "model_loaded": False, "device_allocation_validated": False})
        else:
            if not args.model:
                raise ValueError("--model is required for config/single/dual stages")
            show("MODEL CONFIG", inspect_checkpoint(args.model, local_files_only=not args.allow_download))
            if args.stage in {"single", "dual"}:
                devices = (args.agent_a_device, args.agent_b_device) if args.stage == "dual" else (args.agent_a_device,)
                for device in devices:
                    resolve_device(device, torch)
                show("BEFORE MODEL LOAD", inspect_environment())
                a = HuggingFaceModelBackend(args.model, device=args.agent_a_device)
                show("AGENT A", {"device": a.device, "identity": asdict(a.compatibility_metadata)})
                show("SINGLE MODEL PREFILL / DECODE / KV CODEC / RESTORE", single_model_smoke(a, args.prompt, args.max_new_tokens))
                if args.stage == "dual":
                    # The second model is constructed only after the single-model gate passes.
                    b = HuggingFaceModelBackend(args.model, device=args.agent_b_device)
                    pair = IndependentBackends(a, b)
                    show("AGENT B", {"device": b.device, "identity": asdict(b.compatibility_metadata)})
                    show("DUAL TOKEN / CANONICAL KV BASELINE", dual_handoff_smoke(pair, args.prompt, args.max_new_tokens))
                show("AFTER MODEL LOAD / SMOKE", inspect_environment())
        show("IMPORTANT", {
            "physical_network": False, "transport": "InMemoryByteTransport",
            "kv_semantics": "Canonical clean-prefill KV is a baseline only",
            "working_memory_latentmas": "not implemented; research pending",
            "selector_calibration": "not performed", "full_7_1_acceptance": "not established by this preflight",
        })
        return 0
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as error:
        print(f"Preflight failed: {error}", file=sys.stderr)
        return 1
