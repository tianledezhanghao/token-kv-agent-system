"""7.2A: decode-derived KV reuse with independent receiver histories.

This in-process probe deliberately does not call canonical handoff or selectors.
Only TOKEN_BASELINE forwards message IDs at the receiver. Cache tensors here are
local experiment data, not a new wire protocol or production KV capability.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Any, Callable

from ..model.config import DEFAULT_MODEL_NAME
from ..model.huggingface import HuggingFaceModelBackend, synchronize_device

from ..model.decode_kv import cache_layers, extract_span, complete_message

MODES = ("TOKEN_BASELINE", "RAW_DELTA_KV", "ROPE_ALIGNED_DELTA_KV")


@dataclass(frozen=True)
class ProbeTask:
    name: str
    report: str
    question: str
    expected: str


# Freeze before collecting any model outputs. No per-mode prompt tuning.
TASKS = (
    ProbeTask("identifier", "The access code is 7392.",
              "What is the access code in the report? Answer with the code only.", "7392"),
    ProbeTask("fact", "The parcel is stored in locker 17.",
              "Which locker contains the parcel? Answer with the number only.", "17"),
    ProbeTask("color", "The cup is blue. The plate is red.",
              "What color is the cup in the report? Answer with one color only.", "blue"),
    ProbeTask("constraint", "The allowed labels are oak and pine. The forbidden label is elm.",
              "Which label is forbidden in the report? Answer with the label only.", "elm"),
)


from ..model.delta_repair import (reconstruct, clone_cache, append_delta,
                                  rotate_half, rephase_key, rotary_tables, inspect_rope)


def timed(backend: Any, action: Callable[[], Any]) -> tuple[Any, float]:
    synchronize_device(backend._torch, backend.device)
    start = perf_counter()
    value = action()
    synchronize_device(backend._torch, backend.device)
    return value, (perf_counter() - start) * 1000


def greedy(backend: Any, initial: Any, limit: int) -> tuple[tuple[int, ...], Any, dict[str, Any]]:
    if limit <= 0:
        raise ValueError("token limit must be positive")
    import torch
    logits = initial.last_token_logits
    if not torch.isfinite(logits).all().item():
        raise ValueError("nonfinite initial logits")
    token = int(logits.argmax().item())
    cache = initial.cache
    ids = [token]
    lengths = [int(cache.get_seq_length())]
    ms = []
    while ids[-1] not in backend.eos_token_ids and len(ids) < limit:
        out = backend.continue_from_cache((ids[-1],), previous_cache=cache,
                                          logical_cached_token_count=int(cache.get_seq_length()))
        if not torch.isfinite(out.last_token_logits).all().item():
            raise ValueError("nonfinite decode logits")
        ids.append(out.predicted_next_token_id)
        cache = out.cache
        lengths.append(int(cache.get_seq_length()))
        ms.append(out.forward_ms)
    return tuple(ids), cache, {"stop_reason": "eos" if ids[-1] in backend.eos_token_ids else "max_new_tokens",
                              "cache_lengths": lengths, "decode_forward_ms": ms,
                              "final_prediction_cached": False}


def exact_answer(text: str, expected: str) -> bool:
    # Strict normalized short answer; don't count fluent prose with wrong facts.
    answer = text.strip().lower().strip(' .!\n\t"\'`')
    return answer == expected.lower()


def logit_difference(reference: Any, actual: Any) -> dict[str, Any]:
    import torch
    a, b = reference.detach().float().cpu(), actual.detach().float().cpu()
    if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
        raise ValueError("nonfinite comparison logits")
    delta = (a - b).abs()
    la, lb = a.log_softmax(-1), b.log_softmax(-1)
    pa, pb = la.exp(), lb.exp()
    return {"max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
            "kl_baseline_to_path": (pa * (la - lb)).sum().clamp_min(0).item(),
            "total_variation": (pa - pb).abs().sum().item() / 2,
            "top1_equal": int(a.argmax()) == int(b.argmax())}



def render_sender(tokenizer: Any, task: ProbeTask) -> tuple[str, tuple[int, ...]]:
    messages = [
        {"role": "system", "content": "You are Agent A, a precise report sender. Follow the user's instructions."},
        {"role": "user", "content": "Earlier we organized a weather notebook. This conversation belongs to Agent A."},
        {"role": "assistant", "content": "I will keep my own notebook and send a short report when asked."},
        {"role": "user", "content": "Repeat exactly the following report and nothing else:\n" + task.report},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text, tuple(tokenizer.encode(text, add_special_tokens=False))


def render_receiver(tokenizer: Any, task: ProbeTask, message_ids: tuple[int, ...],
                    sender_history: str, shared: bool) -> tuple[str, tuple[int, ...], str, tuple[int, ...]]:
    marker = "__INCREMENTAL_KV_REPORT_PLACEHOLDER__"
    if shared:
        # The exact sender prefix, then its message, then a receiver-owned question.
        prefix = sender_history
        full = tokenizer.apply_chat_template([
            {"role": "user", "content": marker},
            {"role": "assistant", "content": marker + "_ANSWER"},
            {"role": "user", "content": task.question},
        ], tokenize=False, add_generation_prompt=True)
        suffix = full.split(marker + "_ANSWER", 1)[1]
    else:
        prior = [
            {"role": "system", "content": "You are Agent B. Read the incoming report and answer the next question concisely."},
            {"role": "user", "content": "Our earlier discussion was about music."},
            {"role": "assistant", "content": "I keep this separate conversation about music."},
            {"role": "user", "content": "An independent agent is about to provide a report."},
        ]
        full = tokenizer.apply_chat_template(prior + [
            {"role": "assistant", "content": marker},
            {"role": "user", "content": task.question},
        ], tokenize=False, add_generation_prompt=True)
        prefix, suffix = full.split(marker)
    # Sender EOS is part of M and was explicitly materialized. It already closes
    # the report's assistant turn; don't append the same boundary token twice.
    suffix_ids = tuple(tokenizer.encode(suffix, add_special_tokens=False))
    if message_ids[-1] in tokenizer.all_special_ids:
        if not suffix_ids or suffix_ids[0] != message_ids[-1]:
            raise ValueError("sender stop token does not match receiver assistant boundary")
        suffix_ids = suffix_ids[1:]
    if not suffix_ids:
        raise ValueError("empty receiver suffix")
    return prefix, tuple(tokenizer.encode(prefix, add_special_tokens=False)), tokenizer.decode(suffix_ids), suffix_ids


def run_receiver(backend: Any, history: Any, delta: tuple[tuple[Any, Any], ...],
                 message_ids: tuple[int, ...], suffix_ids: tuple[int, ...], mode: str,
                 old_tables: tuple[Any, Any], new_tables: tuple[Any, Any], limit: int,
                 expected: str) -> tuple[dict[str, Any], Any | None]:
    config = backend._model.config
    result: dict[str, Any] = {"mode": mode, "completed": False, "correct": False,
                             "receiver_message_prefill_tokens": 0, "forward_inputs": []}
    try:
        cache, result["history_clone_ms"] = timed(backend, lambda: clone_cache(history, config))
        result["cache_start"] = int(cache.get_seq_length())
        result["rope_correction_ms"] = 0.0
        result["receiver_append_reconstruction_ms"] = 0.0
        result["token_baseline_prefill_ms"] = 0.0
        if mode == "TOKEN_BASELINE":
            out = backend.continue_from_cache(message_ids, previous_cache=cache,
                                              logical_cached_token_count=int(cache.get_seq_length()))
            result["forward_inputs"].append({"stage": "message_prefill", "token_ids": list(message_ids)})
            cache = out.cache
            result["token_baseline_prefill_ms"] = out.forward_ms
            result["receiver_message_prefill_tokens"] = len(message_ids)
        elif mode in MODES[1:]:
            use_delta = delta
            if mode == "ROPE_ALIGNED_DELTA_KV":
                use_delta, result["rope_correction_ms"] = timed(backend, lambda: tuple(
                    (rephase_key(k, *old_tables, *new_tables), v.clone()) for k, v in delta))
            cache, result["receiver_append_reconstruction_ms"] = timed(
                backend, lambda: append_delta(cache, use_delta, config))
        else:
            raise ValueError("unknown probe mode")
        result["cache_after_message"] = int(cache.get_seq_length())
        if result["cache_after_message"] != result["cache_start"] + len(message_ids):
            raise ValueError("duplicate or missing message positions")
        out = backend.continue_from_cache(suffix_ids, previous_cache=cache,
                                          logical_cached_token_count=int(cache.get_seq_length()))
        result["forward_inputs"].append({"stage": "receiver_prompt_prefill", "token_ids": list(suffix_ids)})
        result["receiver_prompt_prefill_ms"] = out.forward_ms
        result["cache_after_prompt"] = int(out.cache.get_seq_length())
        logits = out.last_token_logits.detach().clone()
        ids, cache, trace = greedy(backend, out, limit)
        text = backend.decode(ids, skip_special_tokens=True)
        result.update({"completed": True, "text": text, "generated_token_ids": list(ids),
                       "correct": exact_answer(text, expected), "decode": trace,
                       "cache_final": int(cache.get_seq_length()),
                       "repetition_detected": len(ids) >= 6 and len(set(ids[-6:])) <= 2})
        import torch
        probs = logits.float().softmax(-1)
        values, indices = torch.topk(probs, 5)
        result["next_token_top5"] = [{"id": int(i), "probability": float(p)}
                                      for i, p in zip(indices.cpu(), values.cpu())]
        return result, logits
    except Exception as error:
        result["exception"] = {"type": type(error).__name__, "message": str(error),
                               "cause": str(error.__cause__) if error.__cause__ else None}
        return result, None


def run_task(backend: Any, task: ProbeTask, sender_limit: int, receiver_limit: int) -> dict[str, Any]:
    import torch
    history_text, history_ids = render_sender(backend._tokenizer, task)
    initial = backend.prefill(input_token_ids=history_ids)
    message_ids, cache, trace = greedy(backend, initial, sender_limit)
    cache, completeness = complete_message(backend, cache, len(history_ids), message_ids)
    delta, extraction_ms = timed(backend, lambda: extract_span(cache, len(history_ids), len(history_ids) + len(message_ids)))
    source_snapshot = extract_span(cache, 0, int(cache.get_seq_length()))
    raw_bytes = sum(t.numel() * t.element_size() for pair in delta for t in pair)
    message_text = backend.decode(message_ids, skip_special_tokens=True)
    result = {"task": asdict(task), "sender_history_text": history_text, "sender_history_ids": list(history_ids),
              "sender_history_length": len(history_ids), "message_token_ids": list(message_ids),
              "message_text": message_text, "message_text_with_specials": backend.decode(message_ids),
              "message_token_count": len(message_ids), "sender_decode": trace,
              "sender_report_matches_requested": message_text.strip() == task.report,
              "sender_prefill_ms": initial.prefill_forward_ms, **completeness,
              "sender_position_interval": [len(history_ids), len(history_ids) + len(message_ids)],
              "raw_delta_kv_bytes": raw_bytes, "kv_extraction_ms": extraction_ms,
              "extra_data_bytes": {"optional_token_ids_uint32": 4 * len(message_ids),
                                   "positions_two_uint64_starts_plus_count": 24, "hidden_states": 0},
              "receivers": []}
    old_tables = rotary_tables(backend, list(range(len(history_ids), len(history_ids) + len(message_ids))))
    for shared in (False, True):
        b_text, b_ids, suffix_text, suffix_ids = render_receiver(
            backend._tokenizer, task, message_ids, history_text, shared)
        b_initial = backend.prefill(input_token_ids=b_ids)
        b_snapshot = extract_span(b_initial.cache, 0, len(b_ids))
        new_tables = rotary_tables(backend, list(range(len(b_ids), len(b_ids) + len(message_ids))))
        paths, logits = [], []
        for mode in MODES:
            path, logit = run_receiver(backend, b_initial.cache, delta, message_ids, suffix_ids,
                                       mode, old_tables, new_tables, receiver_limit, task.expected)
            paths.append(path)
            logits.append(logit)
        for path, logit in zip(paths, logits):
            if logit is not None and logits[0] is not None:
                path["logits_vs_token_baseline"] = logit_difference(logits[0], logit)
        b_unchanged = all(torch.equal(k, bk) and torch.equal(v, bv)
                          for (k, v), (bk, bv) in zip(cache_layers(b_initial.cache), b_snapshot))
        if not b_unchanged:
            raise RuntimeError("receiver paths mutated the history snapshot")
        # Actual K round-trip with nonzero original->target->original positions.
        aligned = tuple(rephase_key(k, *old_tables, *new_tables) for k, _ in delta)
        back = tuple(rephase_key(k, *new_tables, *old_tables) for k in aligned)
        result["receivers"].append({"shared_prefix_control": shared, "history_text": b_text,
                                   "history_token_ids": list(b_ids), "history_length": len(b_ids),
                                   "history_snapshot_unchanged": b_unchanged,
                                   "receiver_prompt_text": suffix_text, "receiver_prompt_ids": list(suffix_ids),
                                   "target_position_interval": [len(b_ids), len(b_ids) + len(message_ids)],
                                   "position_shift": len(b_ids) - len(history_ids),
                                   "key_roundtrip_max_abs": max((a.float() - k.float()).abs().max().item()
                                                                for a, (k, _) in zip(back, delta)),
                                   "paths": paths})
    result["sender_cache_unchanged"] = all(torch.equal(k, sk) and torch.equal(v, sv)
                                           for (k, v), (sk, sv) in zip(cache_layers(cache), source_snapshot))
    if not result["sender_cache_unchanged"]:
        raise RuntimeError("receiver paths mutated sender cache")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--sender-max-new-tokens", type=int, default=64)
    parser.add_argument("--receiver-max-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/incremental-kv-probe.json"))
    args = parser.parse_args(argv)
    if args.sender_max_new_tokens <= 0 or args.receiver_max_new_tokens <= 0:
        parser.error("token limits must be positive")
    import torch
    import transformers
    # Local-only model load: no accidental downloads or remote server activity.
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True)
    backend = HuggingFaceModelBackend(args.model, device=args.device, tokenizer=tokenizer, model=model)
    rope = inspect_rope(backend)
    report = {"experiment": "7.2A", "experimental_only": True,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "torch_version": torch.__version__, "transformers_version": transformers.__version__,
              "model": asdict(backend.metadata), "model_config": model.config.to_dict(),
              "checkpoint_revision": getattr(model.config, "_commit_hash", None),
              "eos_token_ids": list(backend.eos_token_ids),
              "rope": rope, "deterministic_generation": {"do_sample": False, "repetition_penalty": 1.0,
              "sender_max_new_tokens": args.sender_max_new_tokens, "receiver_max_new_tokens": args.receiver_max_new_tokens},
              "task_suite_frozen": [asdict(t) for t in TASKS], "results": [], "errors": []}
    for task in TASKS:
        try:
            report["results"].append(run_task(backend, task, args.sender_max_new_tokens, args.receiver_max_new_tokens))
            print(f"Finished {task.name}", flush=True)
        except Exception as error:
            report["errors"].append({"task": task.name, "type": type(error).__name__, "message": str(error)})
            print(f"Failed {task.name}: {error}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    report["summary"] = []
    for shared in (False, True):
        for mode in MODES:
            paths = [p for r in report["results"] for b in r["receivers"]
                     if b["shared_prefix_control"] == shared for p in b["paths"] if p["mode"] == mode]
            report["summary"].append({"shared_prefix_control": shared, "mode": mode,
                                      "correct": sum(p["correct"] for p in paths),
                                      "completed": sum(p["completed"] for p in paths),
                                      "total": len(TASKS)})
            print(f"shared={shared} {mode}: {sum(p['correct'] for p in paths)}/{len(TASKS)} correct, "
                  f"{sum(p['completed'] for p in paths)}/{len(TASKS)} completed")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Results: {args.output.resolve()}")
    return int(bool(report["errors"]) or any(not p["completed"] for r in report["results"]
               for b in r["receivers"] for p in b["paths"]))
