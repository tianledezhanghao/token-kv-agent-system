"""Bounded Control Plane projection; all inference and KV stay in DiscussionRuntime."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..incremental_auto import IncrementalAutoConfig, backend_cost_identity
from ..generalized_cost import load_cost_profile
from ..synthetic_cost import SyntheticCostAssumptions
from ..network.handoff_network import HandoffNetwork, HandoffNetworkConfig
from ..network.shannon import ShannonChannelConfig
from ..communication import HandoffChoice, validate_recompute_policy
from ..discussion import DiscussionConfig, DiscussionRuntime, error_text
from ..secret_sum import PUBLIC_TASK, SecretSumConfig, SecretSumRuntime, score_result
from ..metrics import CommunicationEvent, EventType, MetricCollector
from ..local_qwen_resources import classify_failure, ResourceStop
from ..stateful_token import ContextBudgetError
from ..discussion_prompts import MAX_SYSTEM_PROMPT_CHARS, resolve_prompts

Mode = Literal['token', 'kv:v0', 'kv:v1', 'kv:v2', 'kv:v3']


class PlannedHandoff(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    round: int = Field(ge=1, le=20)
    source: Literal['A', 'B']
    target: Literal['A', 'B']
    mode: Mode


class PromptPreviewRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    agent_a_system_prompt: str | None = Field(default=None, max_length=MAX_SYSTEM_PROMPT_CHARS)
    agent_b_system_prompt: str | None = Field(default=None, max_length=MAX_SYSTEM_PROMPT_CHARS)

    @model_validator(mode='after')
    def validate_prompts(self):
        resolve_prompts(self.agent_a_system_prompt, self.agent_b_system_prompt)
        return self


class DiscussionStartRequest(PromptPreviewRequest):
    model_config = ConfigDict(extra='forbid', strict=True)
    session_id: str | None = Field(default=None, min_length=8, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')
    architecture: Literal['local', 'distributed'] = 'local'
    max_context_tokens: int = Field(default=4096, ge=128, le=32768)
    scenario: Literal['discussion', 'secret_sum'] = 'discussion'
    secret_a: int = Field(default=17, ge=1, le=99)
    secret_b: int = Field(default=26, ge=1, le=99)
    question: str = Field(min_length=1, max_length=512)
    rounds: int = Field(default=4, ge=1, le=20)
    advanced_rounds: bool = False
    max_new_tokens: int = Field(default=512, ge=1)
    summarizer: Literal['A', 'B'] = 'A'
    mode: Literal['token', 'kv:v0', 'kv:v1', 'kv:v2', 'kv:v3', 'mixed', 'auto'] = 'token'
    kv_recompute_tokens: int | None = Field(default=None, ge=0)
    kv_recompute_ratio: float | None = Field(default=None, ge=0, le=1)
    handoff_plan: list[PlannedHandoff] = Field(default_factory=list, max_length=40)

    auto_policy: Literal['quality_first', 'performance_experiment'] = 'quality_first'
    auto_candidates: list[Mode] | None = Field(default=None, min_length=1, max_length=2)
    auto_kv_version: Literal['v0','v1','v2','v3'] = 'v0'
    idealized_simulation: bool = False
    synthetic_assumptions: dict[str, Any] | None = None
    allow_experimental_v3: bool = False
    cost_profile: dict[str, Any] | None = None
    network_mode: Literal['manual', 'trace', 'shannon'] = 'manual'
    channel: dict[str, Any] | None = None
    bandwidth_mbps: float = Field(default=20, gt=0, allow_inf_nan=False)
    base_latency_ms: float = Field(default=0, ge=0, allow_inf_nan=False)
    bandwidth_trace: list[float] = Field(default_factory=list, max_length=1000)

    def network_config(self):
        return HandoffNetworkConfig(self.network_mode, self.bandwidth_mbps,
            self.base_latency_ms, tuple(self.bandwidth_trace), 'rest_session_config',
            channel=ShannonChannelConfig.from_dict(self.channel or {}) if self.network_mode == 'shannon' else None)

    def auto_config(self):
        return IncrementalAutoConfig(self.auto_policy, tuple(self.auto_candidates or ['token', 'kv:'+self.auto_kv_version]),
            self.allow_experimental_v3, load_cost_profile(self.cost_profile) if self.cost_profile is not None else None,
            self.kv_recompute_tokens, self.kv_recompute_ratio, self.idealized_simulation,
            SyntheticCostAssumptions.from_dict(self.synthetic_assumptions) if self.synthetic_assumptions is not None else None)

    @model_validator(mode='after')
    def validate_plan(self):
        if self.network_mode != 'shannon' and 'rounds' not in self.model_fields_set: self.rounds = 2
        if self.channel is not None and self.network_mode != 'shannon':
            raise ValueError('channel parameters require shannon mode')
        if self.rounds > 10 and not self.advanced_rounds:
            raise ValueError('11–20 rounds require advanced_rounds=true')
        if self.architecture == 'distributed':
            if 'rounds' not in self.model_fields_set: self.rounds = 2
            if self.rounds > 3: raise ValueError('legacy distributed mode supports at most 3 rounds')
        if self.scenario == 'secret_sum' and 'rounds' not in self.model_fields_set: self.rounds = 1
        self.network_config()
        if self.mode == 'auto':
            config = self.auto_config()
            if 'auto_kv_version' in self.model_fields_set and config.selected_kv != 'kv:'+self.auto_kv_version:
                raise ValueError('auto_candidates must match selected KV version')
            self.synthetic_assumptions = config.synthetic_assumptions.to_dict() if config.idealized else None
            self.auto_candidates = list(config.candidates)
            self.auto_kv_version = config.selected_kv[3:]
            if config.selected_kv == 'kv:v2':
                self.kv_recompute_tokens = config.recompute_tokens
        elif self.cost_profile is not None or self.allow_experimental_v3 or self.auto_policy != 'quality_first' or self.auto_candidates is not None or self.idealized_simulation or self.auto_kv_version != 'v0' or self.synthetic_assumptions is not None:
            raise ValueError('AUTO configuration requires mode=auto')
        validate_recompute_policy(self.kv_recompute_tokens, self.kv_recompute_ratio)
        has_v2 = (self.mode == 'auto' and 'kv:v2' in self.auto_candidates) or self.mode == 'kv:v2' or (self.mode == 'mixed' and any(h.mode == 'kv:v2' for h in self.handoff_plan))
        if not has_v2 and (self.kv_recompute_tokens is not None or self.kv_recompute_ratio is not None):
            raise ValueError('recompute configuration requires kv:v2')
        if self.scenario == 'secret_sum':
            if self.rounds != 1 or self.summarizer != 'A':
                raise ValueError('secret_sum requires one exchange and Agent A final')
            if self.agent_a_system_prompt or self.agent_b_system_prompt:
                raise ValueError('secret_sum owns its isolated private prompts')
            self.question = PUBLIC_TASK
        if not self.question.strip():
            raise ValueError('question must be nonempty')
        if self.mode == 'mixed':
            if len(self.handoff_plan) != self.rounds * 2:
                raise ValueError('Mixed handoff plan length must equal rounds * 2')
            for i, item in enumerate(self.handoff_plan):
                source, target = ('A', 'B') if i % 2 == 0 else ('B', 'A')
                if (item.round, item.source, item.target) != (i // 2 + 1, source, target):
                    raise ValueError('handoff plan must follow round order A→B, B→A')
        elif self.handoff_plan:
            raise ValueError('uniform mode cannot specify a handoff plan')
        return self

    def choice(self, mode):
        return HandoffChoice.parse(mode, **(dict(recompute_tokens=self.kv_recompute_tokens,
            recompute_ratio=self.kv_recompute_ratio) if mode == 'kv:v2' else {}))

    def runtime_config(self) -> DiscussionConfig | SecretSumConfig:
        if self.scenario == 'secret_sum':
            return SecretSumConfig(secret_a=self.secret_a, secret_b=self.secret_b,
                max_new_tokens=self.max_new_tokens,
                handoff_choice=self.choice(self.mode if self.mode not in ('mixed', 'auto') else 'token'),
                handoff_plan=tuple(self.choice(h.mode) for h in self.handoff_plan),
                network=self.network_config(), auto=self.auto_config() if self.mode == 'auto' else None)
        return DiscussionConfig(
            question=self.question, rounds=self.rounds, max_new_tokens=self.max_new_tokens,
            summarizer=self.summarizer, max_context_tokens=self.max_context_tokens,
            agent_a_system_prompt=self.agent_a_system_prompt,
            agent_b_system_prompt=self.agent_b_system_prompt,
            handoff_choice=self.choice(self.mode if self.mode not in ('mixed', 'auto') else 'token'),
            handoff_plan=tuple(self.choice(h.mode) for h in self.handoff_plan),
                network=self.network_config(), auto=self.auto_config() if self.mode == 'auto' else None)


class DiscussionNetworkRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    bandwidth_mbps: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    base_latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    manual_override: bool = False
    channel: dict[str, Any] | None = None

    @model_validator(mode='after')
    def validate_update(self):
        if not self.model_fields_set or not ({'bandwidth_mbps', 'base_latency_ms', 'channel'} & self.model_fields_set):
            raise ValueError('at least one network value is required')
        if any(getattr(self, key) is None for key in self.model_fields_set):
            raise ValueError('network values cannot be null')
        return self


class DiscussionBusyError(RuntimeError):
    pass


class DiscussionUnavailableError(RuntimeError):
    pass


def message_view(message, text: str | None) -> dict[str, Any] | None:
    if message is None:
        return None
    return dict(agent=message.source_agent, text=text, token_count=message.token_count,
                termination_reason=message.termination_reason, message_id=message.message_id,
                configured_max_new_tokens=message.configured_max_new_tokens,
                effective_max_new_tokens=message.effective_max_new_tokens,
                context_remaining_tokens=message.context_remaining_tokens,
                budget_limit_reason=message.budget_limit_reason)


class DiscussionService:
    """One active run, one loaded backend, three metadata-only recent sessions.

    Construction of the existing Runtime (including role Prefill) and run() both
    happen on a single controlled worker. No session cache survives its worker.
    """
    def __init__(self, backend: Any = None, *, collector: MetricCollector | None = None):
        self.backend = backend
        self.collector = collector or MetricCollector()
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='discussion')
        self._sessions: dict[str, dict[str, Any]] = {}
        self._networks: dict[str, HandoffNetwork] = {}
        self._started_ids: set[str] = set()  # Small tombstones; never restart an evicted Session ID.
        self._latest: str | None = None
        self._active: str | None = None
        self._closed = False

    def latest(self) -> dict[str, Any]:
        with self._lock:
            metadata = getattr(self.backend, 'metadata', None)
            try:
                identity = backend_cost_identity(self.backend) if self.backend is not None else None
            except (AttributeError, TypeError, ValueError):
                identity = None  # Legacy/scripted backends cannot expose a cost identity.
            recent = [dict(session_id=s['session_id'], status=s['status'], question=s['config']['question'],
                           mode=s['config']['mode'], created_at=s['created_at']) for s in reversed(list(self._sessions.values()))]
            return {**({'recent_sessions': recent} if recent else {}),
                    **({'cost_model_identity': {'sender': identity, 'receiver': identity}} if identity else {}),
                    'available': self.backend is not None and not self._closed,
                    'model': asdict(metadata) if metadata is not None else None,
                    'status': self._sessions[self._latest]['status'] if self._latest else 'idle',
                    'session': deepcopy(self._sessions.get(self._latest)),
                    'active_session_id': self._active}

    def get(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._sessions[session_id])

    def network(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._sessions[session_id]['network'])

    def update_network(self, session_id: str, request: DiscussionNetworkRequest) -> dict[str, Any]:
        # Same RLock as snapshot(): accept+replace+event is one linearized operation.
        # This never acquires a Session inference/cache lock.
        with self._lock:
            record = self._sessions[session_id]
            if record['status'] != 'running' or self._active != session_id:
                raise DiscussionBusyError('Only an active Discussion Session can update its network')
            network = self._networks[session_id]
            before = network.current()
            if before['next_unfrozen_handoff_index'] > record['config']['rounds'] * 2:
                raise DiscussionBusyError('All configured handoffs are already frozen; no future network update can take effect')
            c = before['config']
            if c['mode'] == 'trace' and not request.manual_override:
                raise ValueError('trace requires explicit manual_override=true')
            if c['mode'] == 'shannon':
                if request.bandwidth_mbps is not None:
                    raise ValueError('Shannon effective Mbps is derived; update channel.bandwidth_mhz instead')
                channel = ShannonChannelConfig.from_dict({**c['channel'], **(request.channel or {})})
                config = HandoffNetworkConfig('shannon', c['bandwidth_mbps'], channel=channel,
                    base_latency_ms=request.base_latency_ms if request.base_latency_ms is not None else c['base_latency_ms'],
                    source='rest_shannon_update')
            else:
                if request.channel is not None: raise ValueError('channel PATCH requires an existing shannon Session')
                config = HandoffNetworkConfig('manual',
                    request.bandwidth_mbps if request.bandwidth_mbps is not None else c['bandwidth_mbps'],
                    request.base_latency_ms if request.base_latency_ms is not None else c['base_latency_ms'],
                    source='rest_manual_override' if request.manual_override else 'rest_live_update')
            network.update(config)
            after = network.current()
            accepted_at = datetime.now(timezone.utc).isoformat()
            update = dict(session_id=session_id, before=before, after=after,
                          config_revision=after['config_revision'], source=config.source,
                          effective_from_handoff_index=after['next_unfrozen_handoff_index'],
                          effective_rule=after['effective_rule'], is_simulated=True,
                          accepted_at=accepted_at, mode_before=c['mode'], mode_after=config.mode,
                          channel_bandwidth_changed=bool(config.channel and config.channel.bandwidth_mhz != c['channel']['bandwidth_mhz']))
            record['network'] = after
            record['network_updates'].append(update)
            self._event(session_id, 'DISCUSSION_NETWORK_UPDATED', dict(network=after, network_update=update))
            return dict(updated=True, **update, revision=record['revision'])

    def preview_prompts(self, request: PromptPreviewRequest) -> dict[str, Any]:
        tokenizer = getattr(self.backend, 'chat_tokenizer', None)
        controls = tokenizer.all_special_tokens if tokenizer else ()
        return resolve_prompts(request.agent_a_system_prompt, request.agent_b_system_prompt,
                               control_tokens=controls).to_dict()

    def _event(self, session_id: str, kind: str, data: dict[str, Any]) -> None:
        with self._lock:
            record = self._sessions[session_id]
            record['revision'] += 1
            if kind != 'DISCUSSION_NETWORK_UPDATED':
                record['phase'] = kind
            for field in ('round', 'agent', 'sequence', 'cache_lengths'):
                if field in data:
                    record[field] = data[field]
            if 'decision' in data:
                record['auto_decisions'].append({**data['decision'], 'handoff_index': data['sequence'],
                    'message_id': data.get('message_id'), 'source': data['agent'], 'target': data.get('target')})
            if kind == 'AUTO_DECISION_FAILED':
                record['auto_decisions'].append(deepcopy(data))
            if kind in ('AUTO_DECISION', 'AUTO_DECISION_FAILED', 'HANDOFF_STARTED'):
                record['active_handoff'] = deepcopy(data)
            if kind in ('HANDOFF_COMPLETED', 'HANDOFF_FAILED', 'HANDOFF_UNCONFIRMED', 'DISCUSSION_COMPLETED', 'DISCUSSION_FAILED'):
                record['active_handoff'] = None
            if 'turn' in data:
                record['turns'].append(data['turn'])
            if 'handoff' in data:
                record['handoffs'].append(data['handoff'])
            if session_id in self._networks and kind in ('HANDOFF_STARTED', 'HANDOFF_COMPLETED', 'HANDOFF_FAILED', 'HANDOFF_UNCONFIRMED'):
                record['network'] = self._networks[session_id].current()
                data = {**data, 'network_state': record['network']}
            timestamp = datetime.now(timezone.utc)
            record['updated_at'] = timestamp.isoformat()
            metadata = deepcopy({**data, 'session_id': session_id, 'revision': record['revision'],
                                 'architecture': record.get('architecture', 'local'),
                                 'status': record['status'], 'round': record['round'],
                                 'agent': record['agent'], 'sequence': record['sequence']})
            self.collector.record(CommunicationEvent(
                EventType(kind), timestamp=timestamp, source_agent_id=record['agent'],
                target_agent_id=data.get('target'), metadata=metadata))

    def start(self, request: DiscussionStartRequest) -> dict[str, Any]:
        if request.architecture != 'local':
            raise ValueError('Local endpoint requires architecture=local')
        # Validate runtime plan before reserving the only execution slot.
        config = request.runtime_config()
        prompt_config = (config.prompts(control_tokens=self.backend.chat_tokenizer.all_special_tokens
                         if self.backend is not None else ()).to_dict()
                         if isinstance(config, SecretSumConfig) else self.preview_prompts(request))
        with self._lock:
            if self.backend is None or self._closed:
                raise DiscussionUnavailableError('Discussion model is not loaded; start backend with --discussion')
            session_id = request.session_id or uuid4().hex
            if session_id in self._started_ids:
                raise DiscussionBusyError('Session already started; a Session cannot be restarted')
            if self._active is not None:
                raise DiscussionBusyError('One Discussion Session is already running')
            context_preflight = None
            if isinstance(config, DiscussionConfig) and config.rounds > 3:
                from ..discussion import estimate_discussion_context
                try:
                    context_preflight = estimate_discussion_context(self.backend,config)
                except ResourceStop as exc:
                    raise DiscussionUnavailableError(str(exc)) from exc
            self._started_ids.add(session_id)
            self._active = self._latest = session_id
            self._networks[session_id] = HandoffNetwork(config.network, lock=self._lock)
            timestamp = datetime.now(timezone.utc).isoformat()
            self._sessions[session_id] = dict(
                session_id=session_id, architecture='local', status='running', config=request.model_dump(exclude={'session_id'}),
                prompt_config=prompt_config,
                context_preflight=context_preflight,
                model=asdict(self.backend.metadata) if getattr(self.backend, 'metadata', None) else None,
                network=self._networks[session_id].current(), network_updates=[],
                revision=0, phase='initializing', round=0, agent=None, sequence=0,
                turns=[], handoffs=[], auto_decisions=[], active_handoff=None, final_answer=None, summary=None, score=None, error=None,
                failure_stage=None, termination_reason=None, initial_cache_lengths={}, cache_lengths={}, cache_released=False,
                transport='InMemoryByteTransport', created_at=timestamp, updated_at=timestamp)
            while len(self._sessions) > 3:
                evicted = next(iter(self._sessions))
                del self._sessions[evicted]
                self._networks.pop(evicted, None)
            self._event(session_id, 'DISCUSSION_STARTED', {
                'config': self._sessions[session_id]['config'], 'prompt_config': prompt_config,
                'model': self._sessions[session_id]['model']})
            try:
                self._executor.submit(self._run, session_id, config)
            except Exception as exc:
                self._finish(session_id, dict(status='failed', error=error_text(exc),
                                             failure_stage='worker dispatch', termination_reason='error'))
            return deepcopy(self._sessions[session_id])

    def _run(self, session_id: str, config: DiscussionConfig | SecretSumConfig) -> None:
        runtime = None
        outcome: dict[str, Any]
        try:
            runtime_type = SecretSumRuntime if isinstance(config, SecretSumConfig) else DiscussionRuntime
            runtime = runtime_type(self.backend, config=config, network=self._networks[session_id],
                event_sink=lambda kind, data: self._event(session_id, kind, data))
            result = runtime.run()
            logs = result.handoffs
            outcome = dict(status=result.status, error=result.error, failure_stage=result.failure_stage,
                           termination_reason=result.termination_reason,
                           score=score_result(config, result) if isinstance(config, SecretSumConfig) else None,
                           prompt_config=result.prompt_config.to_dict(),
                           final_answer=message_view(result.final_message, result.final_text),
                           initial_cache_lengths=result.initial_cache_lengths,
                           cache_lengths=result.final_cache_lengths,
                           summary=dict(rounds=config.rounds, completed_turns=len(result.turns),
                               handoff_count=sum(h.append_succeeded for h in logs),
                               attempted_handoff_count=len(logs),
                               token_handoffs=sum(h.executed_mode == 'token' for h in logs),
                               kv_v0_handoffs=sum(h.executed_kv_strategy == 'raw_delta' for h in logs),
                               kv_v1_handoffs=sum(h.executed_kv_strategy == 'rope_aligned_delta' for h in logs),
                               kv_v2_handoffs=sum(h.executed_kv_strategy == 'suffix_recompute_delta' for h in logs),
                               kv_v3_handoffs=sum(h.executed_kv_strategy == 'numeric_slot_delta' for h in logs),
                               repair_layer_token_count=sum(h.repair_layer_token_count for h in logs),
                               recomputed_tokens=sum(h.receiver_recomputed_token_count or 0 for h in logs),
                               total_wire_bytes=sum(h.token_protocol_bytes + h.incremental_kv_envelope_bytes for h in logs),
                               final_cache_lengths=result.final_cache_lengths))
        except Exception as exc:
            outcome = dict(status='failed', error=error_text(exc), failure_stage='initialize Agent roles',
                           termination_reason=('context_limit' if isinstance(exc, ContextBudgetError) or 'context budget' in str(exc) else
                               'resource_abort' if classify_failure(exc) in ('oom', 'resource_policy') else 'error'))
        finally:
            # Drop the sole Runtime/StatefulTokenSession reference before releasing the slot.
            # Result receipts contain IDs/metadata only, never caches or tensors.
            runtime = None
        self._finish(session_id, outcome)

    def _finish(self, session_id: str, outcome: dict[str, Any]) -> None:
        with self._lock:
            self._sessions[session_id].update(outcome, cache_released=True, network=self._networks[session_id].current())
            del self._networks[session_id]
            self._active = None
            self._event(session_id, 'DISCUSSION_COMPLETED' if outcome['status'] == 'completed' else 'DISCUSSION_FAILED',
                        {**outcome, 'cache_released': True})

    def close(self) -> None:
        with self._lock:
            self._closed = True
        # During ASGI shutdown, wait on a thread, never on the event loop.
        self._executor.shutdown(wait=True)
