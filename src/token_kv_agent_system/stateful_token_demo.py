"""Bounded 7.5A state/bytes smoke, without question or final-answer orchestration."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform

from .agent import LLMAgent
from .model import HuggingFaceModelBackend
from .model.config import DEFAULT_MODEL_NAME
from .model.state_cache import require_disjoint_caches
from .stateful_token import StatefulTokenSession
from .transport import InMemoryByteTransport


def smoke(backend, *, rounds: int = 3, max_new_tokens: int = 8) -> dict:
    if not 2 <= rounds <= 3 or not 1 <= max_new_tokens <= 16:
        raise ValueError("smoke allows 2–3 rounds and 1–16 tokens per message")
    transport = InMemoryByteTransport()
    session = StatefulTokenSession([
        LLMAgent("A", backend, system_prompt="You are Agent A. Solve the arithmetic problem and explain your calculation briefly."),
        LLMAgent("B", backend, system_prompt="You are Agent B. Check Agent A's calculation. Point out any error and explain briefly."),
    ], transport)
    initial = {name: session.state(name).cached_token_count for name in ("A", "B")}
    session.append_local_user("A", "Send Agent B a short greeting to test communication.")
    results = []
    for round_index in range(rounds):
        for source, target in (("A", "B"), ("B", "A")):
            before = {name: session.state(name) for name in ("A", "B")}
            message = session.generate_message(source, target, max_new_tokens=max_new_tokens)
            pending = session.state(source)
            assert pending.pending_token_id == message.message_token_ids[-1]
            assert len(pending.token_ledger) == pending.cached_token_count + 1
            receipt = session.handoff(source, target)
            for name in ("A", "B"):
                state = session.state(name)
                state.validate()
                assert state.pending_token_id is None
                assert state.cached_token_count > before[name].cached_token_count
                assert state.token_ledger[:len(before[name].token_ledger)] == before[name].token_ledger
                assert before[name].cache.get_seq_length() == before[name].cached_token_count
                for old_layer, new_layer in zip(before[name].cache.layers, state.cache.layers):
                    for old_tensor, new_tensor in ((old_layer.keys, new_layer.keys),
                                                   (old_layer.values, new_layer.values)):
                        assert backend._torch.equal(
                            old_tensor, new_tensor[..., :before[name].cached_token_count, :])
            require_disjoint_caches(session.state("A").cache, session.state("B").cache)
            start = receipt.message.receiver_message_start
            count = receipt.message.receiver_content_token_count
            assert session.state(target).token_ledger[start:start + count] == message.message_token_ids[:count]
            results.append({"round": round_index + 1, "source": source, "target": target,
                            "token_ids": list(message.message_token_ids),
                            "text": backend.decode(message.message_token_ids, skip_special_tokens=True),
                            "termination_reason": message.termination_reason,
                            "wire_bytes": receipt.wire_byte_count,
                            "receiver_message_start": start,
                            "receiver_appended_count": len(receipt.receiver_appended_ids),
                            "cached_counts": {name: session.state(name).cached_token_count for name in ("A", "B")}})
    return {"scope": "7.5A state/bytes smoke; not discussion or arithmetic correctness",
            "backend_mode": "shared model instance, independent KV caches",
            "model": backend.model_name, "device": backend.device,
            "dtype": backend.metadata.dtype, "platform": platform.platform(),
            "torch_version": backend._torch.__version__,
            "initial_cached_counts": initial, "messages": results,
            "transfers": [asdict(record) for record in transport.records],
            "independent_cache_storage": True, "ledger_cache_lengths_verified": True,
            "old_cache_prefixes_verified": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--rounds", type=int, choices=(2, 3), default=3)
    parser.add_argument("--max-new-tokens", type=int, choices=range(1, 17), default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    # Cache-only smoke: never implicitly download weights.
    os.environ["HF_HUB_OFFLINE"] = "1"
    backend = HuggingFaceModelBackend(args.model, device=args.device)
    result = smoke(backend, rounds=args.rounds, max_new_tokens=args.max_new_tokens)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0
