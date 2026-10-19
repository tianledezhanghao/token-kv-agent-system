"""Small manually routed turn policy over the existing persistent state session."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Callable
from threading import Lock
from time import perf_counter
from typing import Any

from .communication import CommunicationMode, HandoffChoice
from .network.handoff_network import HandoffNetwork, HandoffNetworkConfig
from .incremental_auto import IncrementalAutoConfig, IncrementalSelector
from .agent import LLMAgent
from .agent.state import AgentMessage
from .model.state_cache import require_disjoint_caches
from .stateful_token import IncrementalTokenReceipt, StatefulTokenSession
from .transport import BaseByteTransport, InMemoryByteTransport
from .discussion_prompts import (
    AGENT_A_SYSTEM_PROMPT, AGENT_B_SYSTEM_PROMPT, STAGE_INSTRUCTIONS,
    PromptConfiguration, resolve_prompts)


# Compatibility names for existing CLI/tests; definitions live in one module.
SOLVER_PROMPT = AGENT_A_SYSTEM_PROMPT
REVIEWER_PROMPT = AGENT_B_SYSTEM_PROMPT
FINAL_PROMPT = STAGE_INSTRUCTIONS.final


def error_text(exc: Exception) -> str:
    """Keep backend cause information in a bounded, readable diagnostic."""
    parts = [f"{type(exc).__name__}: {exc}"]
    cause = exc.__cause__
    for _ in range(3):
        if cause is None:
            break
        parts.append(f"{type(cause).__name__}: {cause}")
        cause = cause.__cause__
    return " caused by ".join(parts)

@dataclass(frozen=True)
class DiscussionConfig:
    question: str = "23 + 48 = ?"
    rounds: int = 2
    max_new_tokens: int = 512
    summarizer: str = "A"
    max_context_tokens: int = 4096  # Per Agent, including all local framing.

    handoff_choice: HandoffChoice = HandoffChoice()
    handoff_plan: tuple[HandoffChoice, ...] = ()
    agent_a_system_prompt: str | None = None
    agent_b_system_prompt: str | None = None

    auto: IncrementalAutoConfig | None = None
    network: HandoffNetworkConfig = HandoffNetworkConfig()

    def choice_for(self, index: int) -> HandoffChoice:
        return self.handoff_plan[index] if self.handoff_plan else self.handoff_choice

    def __post_init__(self) -> None:
        if not isinstance(self.network, HandoffNetworkConfig):
            raise ValueError('handoff network configuration required')
        if self.auto is not None:
            if not isinstance(self.auto, IncrementalAutoConfig):
                raise ValueError('incremental auto config required')
            if self.handoff_plan or self.handoff_choice != HandoffChoice():
                raise ValueError('AUTO cannot coexist with a manual choice or mixed plan')
        resolve_prompts(self.agent_a_system_prompt, self.agent_b_system_prompt)
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be nonempty")
        if type(self.rounds) is not int or not 1 <= self.rounds <= 20:
            raise ValueError("rounds must be 1–20 complete A/B rounds")
        if not isinstance(self.handoff_choice, HandoffChoice) or any(
                not isinstance(c, HandoffChoice) for c in self.handoff_plan):
            raise ValueError('handoff choices must be HandoffChoice values')
        if self.handoff_plan and len(self.handoff_plan) != self.rounds * 2:
            raise ValueError('handoff plan length must equal rounds * 2')
        if self.summarizer not in ("A", "B"):
            raise ValueError("summarizer must be A or B")
        for name in ("max_new_tokens", "max_context_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")

def estimate_discussion_context(backend, config):
    """Tokenizer-only conservative full-budget check before starting long sessions.

    Accounts for both local generation and received messages, all exact framing,
    prompts and final summary. Does not truncate history or load a second model.
    """
    from .agent.chat_framing import ChatFraming
    from .stateful_token import ContextBudgetError
    prompts = resolve_prompts(config.agent_a_system_prompt,config.agent_b_system_prompt,
                              control_tokens=backend.chat_tokenizer.all_special_tokens)
    frames = {n:ChatFraming(backend.chat_tokenizer,getattr(prompts,'agent_'+n.lower()).text) for n in ('A','B')}
    counts = {n:len(f.initial_ids)+len(f.local_user_ids(config.question)) for n,f in frames.items()}
    for r in range(1,config.rounds+1):
        for sender,receiver in (('A','B'),('B','A')):
            f = frames[sender]; g = frames[receiver]; incoming = g.incoming(sender,receiver)
            counts[sender] += len(f.local_user_ids(prompts.stages.for_turn(sender,r))) + len(f.assistant.prefix_ids) + len(f.assistant.closing_ids) + config.max_new_tokens
            counts[receiver] += len(incoming.prefix_ids) + len(incoming.closing_ids) + config.max_new_tokens
    f = frames[config.summarizer]
    counts[config.summarizer] += len(f.local_user_ids(prompts.stages.final)) + len(f.assistant.prefix_ids) + len(f.assistant.closing_ids) + config.max_new_tokens
    limit = min(config.max_context_tokens,getattr(backend,'context_window_tokens',config.max_context_tokens))
    if max(counts.values()) > limit:
        raise ContextBudgetError(f'Long discussion worst-case context {counts} exceeds {limit}; reduce rounds/output budget or increase context within model/resource limits')
    result = dict(estimated_tokens=counts,context_limit=limit,method='conservative_full_generation_budget')
    # Existing model-worker guard keeps protecting forwards. Check incremental
    # cache/transaction headroom without changing any OS or MPS memory limits.
    if getattr(getattr(backend,'metadata',None),'device','') == 'mps':
        import re
        from .local_qwen_comparison import memory
        from .local_qwen_resources import ResourceStop
        row = memory(backend._torch)
        if row['pressure_level'] >= 4: raise ResourceStop('critical resource pressure before long discussion')
        result['resource_pressure'] = row['pressure_level']
        structure = backend.kv_model_config
        element_bytes = 4 if backend.metadata.dtype == 'torch.float32' else 2
        result['estimated_peak_cache_bytes'] = (5*sum(counts.values())*2*structure.num_hidden_layers
            *structure.num_key_value_heads*structure.head_dim*element_bytes)
        vm = row['vm_stat']; page = int(re.search(r'page size of (\d+) bytes',vm)[1])
        available = page*sum(int(re.search(label+r':\s+(\d+)',vm)[1])
                            for label in ('Pages free','Pages inactive','Pages purgeable'))
        result['available_cache_budget_bytes'] = available
        if result['estimated_peak_cache_bytes'] > available:
            raise ResourceStop('insufficient memory for long discussion cache/transaction reservation')
    return result


@dataclass(frozen=True)
class HandoffLog:
    source: str
    target: str
    discussion_round: int
    requested_mode: str
    requested_kv_strategy: str | None
    executed_mode: str | None
    executed_kv_strategy: str | None
    message_content_token_count: int
    token_protocol_bytes: int
    incremental_kv_envelope_bytes: int
    sender_kv_span: tuple[int, int]
    receiver_kv_target_span: tuple[int, int] | None
    receiver_cache_length_before: int
    receiver_cache_length_after: int
    receiver_message_prefill_count: int | None
    kv_repair_method: str | None
    rope_position_delta: int | None
    error: str | None
    fallback_reason: str | None
    generation_termination_reason: str
    unpack_succeeded: bool | None
    append_succeeded: bool
    cache_ledger_consistent: bool | None
    receiver_recomputed_token_count: int | None = None
    recompute_forward_count: int | None = None
    repaired_token_indices: tuple[int, ...] = ()
    repaired_layer_indices: tuple[int, ...] = ()
    repair_layer_token_count: int = 0
    handoff_index: int = 0
    network: dict | None = None
    auto_decision: dict | None = None
    selected_mode: str | None = None
    selected_kv_strategy: str | None = None
    actual_wire_bytes: int = 0
    predicted_wire_bytes: int | None = None
    predicted_local_ms: float | None = None
    observed_local_ms: float | None = None
    observed_stage_ms: dict | None = None
    simulated_transfer_ms: float | None = None
    predicted_total_ms: float | None = None
    actual_execution_status: str = 'not_started'
    observation_scope: str = 'not_measured'
    message_id: str | None = None
    wire_prediction_error_bytes: int | None = None


@dataclass(frozen=True)
class DiscussionTurn:
    round: int
    message: AgentMessage
    text: str
    receipt: IncrementalTokenReceipt
    cache_before: dict[str, int]
    cache_after: dict[str, int]

@dataclass(frozen=True)
class DiscussionResult:
    config: DiscussionConfig
    status: str
    termination_reason: str
    failure_stage: str | None
    error: str | None
    initial_cache_lengths: dict[str, int]
    turns: tuple[DiscussionTurn, ...]
    final_message: AgentMessage | None
    final_text: str | None
    final_cache_lengths: dict[str, int]
    context_budget: int
    prompt_config: PromptConfiguration
    models: dict[str, Any]
    handoffs: tuple[HandoffLog, ...] = ()


class DiscussionRuntime:
    """One bounded run. A failure is terminal; last valid session states survive.

    Backends may share model weights or be compatible independent endpoints.
    Mutable state remains entirely owned by StatefulTokenSession.
    """

    def __init__(self, backend_a: Any, backend_b: Any | None = None, *,
                 config: DiscussionConfig | None = None,
                 transport: BaseByteTransport | None = None,
                 event_sink: Callable[[str, dict[str, Any]], None] | None = None,
                 network: HandoffNetwork | None = None) -> None:
        self.config = config or DiscussionConfig()
        self.event_sink = event_sink
        backend_b = backend_b if backend_b is not None else backend_a
        controls = tuple(set(backend_a.chat_tokenizer.all_special_tokens)
                         | set(backend_b.chat_tokenizer.all_special_tokens))
        self._prompt_config = resolve_prompts(self.config.agent_a_system_prompt,
                                             self.config.agent_b_system_prompt,
                                             control_tokens=controls)
        self.models = {name: {k: v for k, v in asdict(b.metadata).items()
                              if k != 'load_wall_time_seconds'}
                       if getattr(b, 'metadata', None) else None
                       for name, b in (('A', backend_a), ('B', backend_b))}
        limits = [self.config.max_context_tokens]
        for backend in (backend_a, backend_b):
            limit = getattr(backend, "context_window_tokens", None)
            if type(limit) is int and limit > 0:
                limits.append(limit)
        self.context_budget = min(limits)
        self._resource_backends = []
        if self.config.rounds > 3:
            estimate_discussion_context(backend_a,self.config)
            if backend_b is not backend_a: estimate_discussion_context(backend_b,self.config)
            from .local_qwen_comparison import memory
            for b in ((backend_a,) if backend_b is backend_a else (backend_a,backend_b)):
                if getattr(getattr(b,'metadata',None),'device','') == 'mps':
                    self._resource_backends.append((b,memory(b._torch)))
        self.session = StatefulTokenSession([
            LLMAgent("A", backend_a, system_prompt=self.prompt_config.agent_a.text),
            LLMAgent("B", backend_b, system_prompt=self.prompt_config.agent_b.text),
        ], transport if transport is not None else InMemoryByteTransport(),
            max_context_tokens=self.context_budget)
        self.status = "ready"
        self.failure_stage: str | None = None
        self._lock = Lock()
        self.network = network if network is not None else HandoffNetwork(self.config.network)

    @property
    def prompt_config(self) -> PromptConfiguration:
        return self._prompt_config

    @property
    def config(self) -> DiscussionConfig:
        return self._config

    @config.setter
    def config(self, value: DiscussionConfig) -> None:
        if hasattr(self, '_config'):
            if self.status != 'ready' or (
                value.agent_a_system_prompt, value.agent_b_system_prompt
            ) != (self._config.agent_a_system_prompt, self._config.agent_b_system_prompt):
                raise ValueError('Session configuration is fixed; create a new Session to change prompts')
        self._config = value

    def _lengths(self) -> dict[str, int]:
        return {name: self.session.state(name).cached_token_count for name in ("A", "B")}

    def _emit(self, kind: str, **metadata: Any) -> None:
        if self.event_sink is not None:
            self.event_sink(kind, metadata)

    def _verify(self) -> None:
        for name in ("A", "B"):
            state = self.session.state(name)
            state.validate()
            if state.pending_token_id is not None or state.cached_token_count != len(state.token_ledger):
                raise ValueError("completed stage must have a closed, fully cached ledger")
            if state.cached_token_count > self.context_budget:
                raise ValueError("context budget exceeded")
        require_disjoint_caches(self.session.state("A").cache, self.session.state("B").cache)

    def _generation_budget(self, sequence: int) -> dict:
        """Reserve actual framing and one output for every later stage on both Agents.

        Account for non-EOS content (worst case) and receiver framing before
        accepting IDs. EOS may release space; no configured budget is rewritten.
        """
        from .stateful_token import ContextBudgetError
        # Long-session guards run at safe generation boundaries; no other
        # process/OS settings are touched and completed turns remain in result.
        from .local_qwen_resources import ResourceStop
        for backend, baseline in self._resource_backends:
            from .local_qwen_comparison import memory
            row = memory(backend._torch)
            if row['pressure_level'] >= 4 or row['swap_used_mib']-baseline['swap_used_mib'] > 256:
                raise ResourceStop('long discussion resource pressure or >256 MiB swap growth')
        stages = [(s, t, self.prompt_config.stages.for_turn(s, r))
                  for r in range(1, self.config.rounds + 1) for s, t in (('A', 'B'), ('B', 'A'))]
        stages.append((self.config.summarizer, None, self.prompt_config.stages.final))
        source, target, _ = stages[sequence - 1]
        reserves = {'A': 0, 'B': 0}
        for offset, (sender, receiver, prompt) in enumerate(stages[sequence - 1:]):
            state = self.session.state(sender)
            reserves[sender] += (len(state.framing.local_user_ids(prompt))
                                 + len(state.framing.assistant.prefix_ids)
                                 + len(state.framing.assistant.closing_ids))
            if offset: reserves[sender] += 1  # minimum safe future generation
            if receiver:
                frame = self.session.state(receiver).framing.incoming(sender, receiver)
                reserves[receiver] += len(frame.prefix_ids) + len(frame.closing_ids) + int(bool(offset))
        rooms = {n: self.context_budget - len(self.session.state(n).token_ledger) - reserves[n]
                 for n in ('A', 'B')}
        if any(value < 0 for value in rooms.values()):
            raise ContextBudgetError('Remaining stages/framing exceed context budget before generation')
        remaining = min(rooms[source], rooms[target]) if target else rooms[source]
        effective = min(self.config.max_new_tokens, remaining)
        if effective < 1:
            raise ContextBudgetError('No safe generation remains in context budget')
        # Manual KV must fit its unchanged TKVI/TKVK limits before Decode starts.
        # AUTO filters the actual message capacity per candidate after generation.
        if target and self.config.auto is None and self.config.choice_for(sequence - 1).mode == CommunicationMode.KV:
            from .incremental_kv_protocol import (IncrementalLimits, MAX_METADATA_BYTES,
                                                   INCREMENTAL_WIRE_HEADER_BYTES, DIGEST_BYTES)
            from .token_protocol import TOKEN_WIRE_HEADER_BYTES, TOKEN_BYTES
            from .kv_protocol import estimate_kv_wire_size, KVModelCompatibility
            b = self.session.state(source).backend
            k = b.kv_model_config
            limits = IncrementalLimits()
            estimate = estimate_kv_wire_size(effective, KVModelCompatibility(b.model_name,
                k.num_hidden_layers, k.num_key_value_heads, k.head_dim),
                dtype=b.metadata.dtype.removeprefix('torch.'), source_device='cpu')
            state = self.session.state(source)
            sender_start = (len(state.token_ledger) + len(state.framing.local_user_ids(stages[sequence - 1][2]))
                            + len(state.framing.assistant.prefix_ids))
            if (effective > limits.max_message_tokens or sender_start + effective > limits.max_sender_positions or
                    estimate.total_wire_bytes + MAX_METADATA_BYTES + INCREMENTAL_WIRE_HEADER_BYTES
                    + DIGEST_BYTES + TOKEN_WIRE_HEADER_BYTES + TOKEN_BYTES * effective > limits.max_packet_bytes):
                raise ValueError('Configured safe generation exceeds incremental KV protocol capacity')
        return dict(configured_max_new_tokens=self.config.max_new_tokens,
                    effective_max_new_tokens=effective, context_remaining_tokens=remaining,
                    budget_limit_reason='context_limit' if effective < self.config.max_new_tokens else None)

    def run(self) -> DiscussionResult:
        with self._lock:
            if self.status != "ready":
                raise RuntimeError(f"discussion is terminal or already running: {self.status}")
            self.status = "running"
            turns: list[DiscussionTurn] = []
            handoffs: list[HandoffLog] = []
            incoming_message_ids: dict[str, str] = {}
            initial: dict[str, int] = {}
            final_message = None
            final_text = None
            error = None
            stage = "initialize question A"
            try:
                for name in ("A", "B"):
                    stage = f"initialize question {name}"
                    self.session.append_local_user(name, self.config.question)
                self._verify()
                initial = self._lengths()
                for round_number in range(1, self.config.rounds + 1):
                    for source, target in (("A", "B"), ("B", "A")):
                        before = self._lengths()
                        stage = f"round {round_number} / Agent {source} generation"
                        sequence = len(turns) + 1
                        context = dict(round=round_number, agent=source, target=target,
                                       sequence=sequence)
                        self._emit("AGENT_TURN_STARTED", **context)
                        prompt = self.prompt_config.stages.for_turn(source, round_number)
                        budget = self._generation_budget(sequence)
                        message = self.session.generate_message(
                            source, target, max_new_tokens=budget['effective_max_new_tokens'],
                            local_prompt=prompt, generation_budget=budget)
                        stage = f"round {round_number} / Agent {source} display"
                        text = self.session.state(source).backend.decode(
                            message.message_token_ids, skip_special_tokens=True)
                        self._emit("AGENT_TURN_COMPLETED", **context,
                                   turn=dict(round=round_number, agent=source, target=target,
                                             sequence=sequence, message_id=message.message_id,
                                             text=text, token_count=message.token_count,
                                             received_message_id=incoming_message_ids.get(source),
                                             termination_reason=message.termination_reason, **budget))
                        context['message_id'] = message.message_id
                        choice = self.config.choice_for(len(turns))
                        stage = f"round {round_number} / handoff decision {source} -> {target}"
                        snapshot = self.network.snapshot(sequence)
                        decision = None
                        requested_mode = 'auto' if self.config.auto else choice.mode.value
                        if self.config.auto:
                            try:
                                choice, decision = IncrementalSelector(self.config.auto).select(
                                    self.session.state(source), self.session.state(target), message, snapshot,
                                    kv_contract_ready=hasattr(self.session, '_kv_contract'),
                                    require_workload_domain=True)
                                self._emit('AUTO_DECISION', **context, decision=decision)
                            except Exception as exc:
                                self._emit('AUTO_DECISION_FAILED', **context, network=asdict(snapshot),
                                    decision_status='failed', error=error_text(exc), requested_mode='auto')
                                raise
                        stage = f"round {round_number} / {choice.mode.value.upper()} {source} -> {target}"
                        content_count = message.token_count - int(message.termination_reason == 'eos')
                        receiver_before = self.session.state(target).cached_token_count
                        records_before = len(getattr(self.session.transport, 'records', ()))
                        receipt = None
                        handoff_error = None
                        self._emit("HANDOFF_STARTED", **context,
                                   requested_mode=requested_mode, network=asdict(snapshot),
                                   selected_mode=choice.mode.value, auto_decision=decision,
                                   selected_kv_strategy=choice.kv_strategy.value if choice.kv_strategy else None,
                                   requested_kv_strategy=choice.kv_strategy.value if choice.kv_strategy and not self.config.auto else None)
                        execution_started = perf_counter()
                        try:
                            receipt = (self.session.handoff(source, target, **({'measure_timing': True} if self.config.auto else {}))
                                       if choice.mode == CommunicationMode.TOKEN else
                                       self.session.handoff_kv(source, target, strategy=choice.kv_strategy,
                                           recompute_tokens=choice.recompute_tokens, recompute_ratio=choice.recompute_ratio,
                                           measure_timing=bool(self.config.auto)))
                        except Exception as exc:
                            handoff_error = error_text(exc)
                            raise
                        finally:
                            records = getattr(self.session.transport, 'records', ())
                            wire = receipt.wire_byte_count if receipt else sum(
                                r.byte_count for r in records[records_before:])
                            target_start = receipt.message.receiver_message_start if receipt else None
                            handoffs.append(HandoffLog(
                                source, target, round_number, requested_mode,
                                choice.kv_strategy.value if choice.kv_strategy and not self.config.auto else None,
                                receipt.message.payload_mode.lower() if receipt else None,
                                getattr(receipt, 'kv_strategy', None),
                                content_count, wire if choice.mode == CommunicationMode.TOKEN else 0,
                                wire if choice.mode == CommunicationMode.KV else 0,
                                (message.sender_message_span[0], message.sender_message_span[0] + content_count),
                                (target_start, target_start + content_count) if receipt else None,
                                receiver_before, self.session.state(target).cached_token_count,
                                getattr(receipt, 'receiver_message_prefill_count', content_count) if receipt else None,
                                getattr(receipt, 'repair_method', None),
                                getattr(receipt, 'rope_position_delta', None), handoff_error, None,
                                message.termination_reason,
                                True if receipt and choice.mode == CommunicationMode.KV else None,
                                receipt is not None, True if receipt else None,
                                getattr(receipt, 'receiver_message_prefill_count', None),
                                getattr(receipt, 'recompute_forward_count', None),
                                getattr(receipt, 'repaired_token_indices', ()),
                                getattr(receipt, 'repaired_layer_indices', ()),
                                getattr(receipt, 'repair_layer_token_count', 0),
                                message_id=message.message_id,
                                handoff_index=sequence, network=asdict(snapshot),
                                auto_decision={**decision, 'decision_status': 'executed' if receipt else 'execution_failed'} if decision else None,
                                selected_mode=choice.mode.value,
                                selected_kv_strategy=choice.kv_strategy.value if choice.kv_strategy else None,
                                actual_wire_bytes=wire,
                                wire_prediction_error_bytes=wire-decision['predicted_wire_bytes'] if decision and receipt else None,
                                predicted_wire_bytes=decision['predicted_wire_bytes'] if decision else None,
                                predicted_local_ms=decision['predicted_local_ms'] if decision else None,
                                observed_local_ms=(perf_counter()-execution_started)*1000,
                                observed_stage_ms=getattr(receipt, 'timings_ms', None),
                                simulated_transfer_ms=snapshot.transfer_ms(wire),
                                predicted_total_ms=decision['predicted_total_ms'] if decision else None,
                                actual_execution_status='succeeded' if receipt else 'failed',
                                observation_scope='synchronized_instrumented_handoff' if self.config.auto and receipt else 'local_wall_handoff'))
                            self._emit("HANDOFF_COMPLETED" if receipt else "HANDOFF_FAILED",
                                       **context, handoff={**asdict(handoffs[-1]), "sequence": sequence,
                                                          "wire_bytes": wire},
                                       cache_lengths=self._lengths())
                        incoming_message_ids[target] = message.message_id
                        self._verify()
                        turns.append(DiscussionTurn(round_number, receipt.message, text,
                                                    receipt, before, self._lengths()))
                stage = f"final / Agent {self.config.summarizer} generation and close"
                self._emit("DISCUSSION_FINALIZING", round=self.config.rounds,
                           agent=self.config.summarizer, sequence=len(turns) + 1)
                budget = self._generation_budget(len(turns) + 1)
                final_message = self.session.generate_local_message(
                    self.config.summarizer, max_new_tokens=budget['effective_max_new_tokens'],
                    local_prompt=self.prompt_config.stages.final, generation_budget=budget)
                self._verify()
                stage = f"final / Agent {self.config.summarizer} display"
                final_text = self.session.state(self.config.summarizer).backend.decode(
                    final_message.message_token_ids, skip_special_tokens=True)
                self.status = "completed"
            except Exception as exc:
                from .stateful_token import ContextBudgetError
                from .local_qwen_resources import classify_failure
                failure_reason = ('context_limit' if isinstance(exc, ContextBudgetError) else
                                  'resource_abort' if classify_failure(exc) in ('oom', 'resource_policy') else 'error')
                self.status = "failed"
                self.failure_stage = stage
                error = error_text(exc)
            return DiscussionResult(
                self.config, self.status,
                "rounds_complete" if self.status == "completed" else failure_reason,
                self.failure_stage, error, initial, tuple(turns), final_message, final_text,
                self._lengths(), self.context_budget, self.prompt_config, self.models, tuple(handoffs))
