"""Run a bounded general two-Agent discussion and model-generated final answer."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from .communication import HandoffChoice
from .generalized_cost import load_cost_profile
from .incremental_auto import IncrementalAutoConfig
from .network.handoff_network import HandoffNetworkConfig
from .discussion import DiscussionConfig, DiscussionResult, DiscussionRuntime, error_text
from .model import HuggingFaceModelBackend
from .model.config import DEFAULT_MODEL_NAME


def print_result(result: DiscussionResult) -> None:
    print(f"\nQuestion:\n{result.config.question}")
    print(f"Initial cache lengths: {result.initial_cache_lengths}")
    for turn in result.turns:
        m, r = turn.message, turn.receipt
        print(f"\nAgent {m.source_agent} / Round {turn.round}:\n{turn.text}")
        print(f"Generation stop: {m.termination_reason}")
        print(f"Generation budget: configured={m.configured_max_new_tokens}; effective={m.effective_max_new_tokens}; limit={m.budget_limit_reason or 'none'}")
        print(f"\nTransport {m.source_agent} -> {m.target_agent}:\nMode: {m.payload_mode}")
        print(f"Message tokens: {m.token_count}; Content tokens: {r.message.receiver_content_token_count}")
        print(f"Wire bytes: {r.wire_byte_count}")
        print(f"Receiver cache before: {r.receiver_cached_count_before}")
        print(f"Receiver cache after: {r.receiver_cached_count_after}")
        print(f"Receiver ledger appended tokens (framing included): {len(r.receiver_appended_ids)}")
        log = result.handoffs[result.turns.index(turn)]
        print(f"Requested/executed: {log.requested_mode}/{log.executed_mode}; "
              f"KV strategy: {log.requested_kv_strategy}/{log.executed_kv_strategy}")
        if log.auto_decision:
            print(f"AUTO {log.auto_decision['auto_policy']} at {log.network['bandwidth_mbps']} Mbps: {log.auto_decision['selected_candidate']}; profile={log.auto_decision['profile_provenance']}")
            print(f"Predicted local/total: {log.predicted_local_ms}/{log.predicted_total_ms}; observed local: {log.observed_local_ms}; simulated link: {log.simulated_transfer_ms}")
        print(f"Sender span: {log.sender_kv_span}; Receiver span: {log.receiver_kv_target_span}")
        print(f"Receiver M Prefill: {log.receiver_message_prefill_count}; "
              f"Repair: {log.kv_repair_method}; RoPE delta: {log.rope_position_delta}")
        print(f"Agent cache lengths after handoff: {turn.cache_after}")
    if result.status == "completed":
        print(f"\nFinal Answer / Agent {result.config.summarizer}:\n{result.final_text}")
        print(f"Generation stop: {result.final_message.termination_reason}")
        print(f"Generation budget: configured={result.final_message.configured_max_new_tokens}; effective={result.final_message.effective_max_new_tokens}; limit={result.final_message.budget_limit_reason or 'none'}")
    else:
        print(f"\nDiscussion FAILED at {result.failure_stage}:\n{result.error}")
    print("\nDiscussion Summary:")
    for name in ("A", "B"):
        print(f"{name} messages: {sum(t.message.source_agent == name for t in result.turns)} (delivered)")
    print(f"TOKEN handoffs: {sum(t.message.payload_mode == 'TOKEN' for t in result.turns)}")
    print(f"KV handoffs: {sum(t.message.payload_mode == 'KV' for t in result.turns)}")
    print(f"Total wire bytes (including failed attempts): {sum(h.token_protocol_bytes + h.incremental_kv_envelope_bytes for h in result.handoffs)}")
    for h in result.handoffs:
        if h.error:
            print(f"Failed handoff {h.source}->{h.target}: {h.error}; wire bytes: "
                  f"{h.token_protocol_bytes + h.incremental_kv_envelope_bytes}; no fallback")
    for name, count in result.final_cache_lengths.items():
        print(f"Agent {name} final cache length: {count}")
    print(f"Context budget per Agent: {result.context_budget}")
    print(f"Status: {result.status}; Termination reason: {result.termination_reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="23 + 48 = ?")
    parser.add_argument("--rounds", type=int, choices=range(1,21), default=4)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="mps", help="auto, cpu, mps, cuda or cuda:N")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--summarizer", choices=("A", "B"), default="A")
    parser.add_argument('--agent-a-system-prompt', help='Blank uses the backend default')
    parser.add_argument('--agent-b-system-prompt', help='Blank uses the backend default')
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--allow-download", action="store_true", help="Allow HF model download; default is cache-only")
    parser.add_argument('--mode', choices=('token', 'kv', 'auto'), default='token')
    parser.add_argument('--kv-strategy', choices=('v0', 'v1', 'v2', 'v3'))
    parser.add_argument('--handoff-plan', help='Exactly rounds*2 comma-separated token,kv:v0,kv:v1,kv:v2,kv:v3 entries')
    parser.add_argument("--output", type=Path, help="Optional JSON with exact generated/received IDs and counts")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument('--kv-recompute-tokens', type=int, help='V2 tail count, clipped to N; default 1')
    policy.add_argument('--kv-recompute-ratio', type=float, help='V2 ceil(N*ratio), ratio in [0,1]')
    parser.add_argument('--auto-policy', choices=('quality_first','performance_experiment'), default='quality_first')
    parser.add_argument('--auto-candidates', default=None, help='Compatibility: TOKEN and one KV only')
    parser.add_argument('--auto-kv-version', choices=('v0','v1','v2','v3'), default=None, help='AUTO KV candidate (default v0)')
    parser.add_argument('--idealized-simulation', action='store_true', help='Synthetic teaching assumptions; requires performance_experiment')
    parser.add_argument('--allow-experimental-v3', action='store_true')
    parser.add_argument('--cost-profile', type=Path, help='Incremental profile JSON; never a Canonical profile')
    parser.add_argument('--network-mode', choices=('manual','trace','shannon'), default='manual')
    parser.add_argument('--channel-json', help='ShannonChannelConfig JSON; MHz and dB, never Mbps bandwidth')
    parser.add_argument('--bandwidth-mbps', type=float, default=20)
    parser.add_argument('--base-latency-ms', type=float, default=0, help='One-way per handoff; no sleeping')
    parser.add_argument('--bandwidth-trace', help='Comma-separated Mbps, hold last after exhaustion')
    args = parser.parse_args(argv)
    try:
        from .network.shannon import ShannonChannelConfig
        network = HandoffNetworkConfig(args.network_mode, args.bandwidth_mbps, args.base_latency_ms,
            tuple(float(x.strip()) for x in args.bandwidth_trace.split(',')) if args.bandwidth_trace is not None else (), 'cli_session_config',
            channel=ShannonChannelConfig.from_dict(json.loads(args.channel_json or '{}')) if args.network_mode=='shannon' else None)
        auto = None
        if args.mode == 'auto':
            if args.handoff_plan or args.kv_strategy:
                raise ValueError('AUTO cannot coexist with manual KV or mixed plan')
            auto = IncrementalAutoConfig(args.auto_policy, tuple(x.strip() for x in args.auto_candidates.split(',')) if args.auto_candidates else ('token','kv:'+(args.auto_kv_version or 'v0')),
                args.allow_experimental_v3, load_cost_profile(json.loads(args.cost_profile.read_text())) if args.cost_profile else None,
                args.kv_recompute_tokens, args.kv_recompute_ratio, args.idealized_simulation)
            if args.auto_kv_version and auto.selected_kv != 'kv:'+args.auto_kv_version:
                raise ValueError('auto-candidates must match selected KV version')
        elif args.cost_profile or args.allow_experimental_v3 or args.auto_policy != 'quality_first' or args.auto_candidates is not None or args.auto_kv_version is not None or args.idealized_simulation:
            raise ValueError('AUTO configuration requires --mode auto')
        if args.handoff_plan is not None:
            plan = tuple(HandoffChoice.parse(x.strip()) for x in args.handoff_plan.split(','))
            choice = HandoffChoice()
        else:
            plan = ()
            if args.mode in ('token', 'auto') and args.kv_strategy is not None:
                raise ValueError('TOKEN cannot specify --kv-strategy')
            if args.mode == 'kv' and args.kv_strategy is None:
                raise ValueError('--mode kv requires --kv-strategy v0/v1/v2/v3')
            choice = HandoffChoice.parse('token' if args.mode in ('token', 'auto') else 'kv:' + args.kv_strategy)
        modes = plan or (choice,)
        if not auto and (args.kv_recompute_tokens is not None or args.kv_recompute_ratio is not None):
            from dataclasses import replace
            from .communication import KVStrategy
            if not any(c.kv_strategy == KVStrategy.SUFFIX_RECOMPUTE_DELTA for c in modes):
                raise ValueError('recompute configuration requires kv:v2')
            modes = tuple(replace(c, recompute_tokens=args.kv_recompute_tokens,
                recompute_ratio=args.kv_recompute_ratio) if c.kv_strategy == KVStrategy.SUFFIX_RECOMPUTE_DELTA else c for c in modes)
            if plan: plan = modes
            else: choice = modes[0]
        config = DiscussionConfig(args.question, args.rounds, args.max_new_tokens,
                                  args.summarizer, args.max_context_tokens, choice, plan,
                                  args.agent_a_system_prompt, args.agent_b_system_prompt, auto=auto, network=network)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
    stage = "model load"
    try:
        backend = HuggingFaceModelBackend(args.model, device=args.device)
        print(f"Model: {backend.metadata.model_name}; Device: {backend.metadata.device}; Dtype: {backend.metadata.dtype}")
        print("Backend mode: shared model weights, independent Agent KV caches")
        stage = "initialize Agent roles"
        runtime = DiscussionRuntime(backend, config=config)
        result = runtime.run()
        print_result(result)
        if args.output:
            stage = "write JSON result"
            record = {"model": asdict(backend.metadata), "discussion": asdict(result)}
            # Preserve a failed generation draft for diagnosis, without claiming delivery.
            record["undelivered_drafts"] = {
                name: asdict(runtime.session.state(name).draft_message)
                for name in ("A", "B") if runtime.session.state(name).draft_message is not None}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0 if result.status == "completed" else 1
    except Exception as exc:
        print(f"Discussion FAILED at {stage}: {error_text(exc)}")
        return 1
