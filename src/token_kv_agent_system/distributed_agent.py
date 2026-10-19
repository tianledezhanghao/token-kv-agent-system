"""Single-state Agent executor. No peer Agent, tensor, or cache can be supplied."""
from dataclasses import asdict, replace
from threading import RLock, Lock
from types import SimpleNamespace
from uuid import uuid4
import gc
import time

from .agent import LLMAgent
from .agent.state import (AgentMessage, initialize_agent_state, working_state,
                         close_working_draft, append_working_ids)
from .communication import HandoffChoice, CommunicationMode
from .discussion_prompts import resolve_prompts
from .distributed_wire import pack_delivery, unpack_delivery, sha, RECEIPT_SCHEMA
from .incremental_kv import prepare_incremental_message
from .incremental_kv_protocol import (inspect_model_contract, canonical_json,
    encode_incremental_message, decode_incremental_message)
from .incremental_kv_receive import receive_incremental_kv
from .stateful_token import StatefulTokenSession
from .token_protocol import serialize_token_ids, deserialize_token_packet


class LocalAgentSession(StatefulTokenSession):
    """Reuse local generation/budget logic, with exactly ONE state and no transport.

    Pair handoff APIs are unavailable. Only the distributed executor can commit
    incoming data or close an outgoing draft after an application receipt.
    """
    def __init__(self, agent, max_context_tokens):
        self.max_context_tokens = max_context_tokens
        self._lock = Lock()
        self._states = {agent.agent_id: initialize_agent_state(agent,
            max_context_tokens=max_context_tokens)}

    def generate_message(self, source, target, **kwargs):
        with self._lock:
            work = self._generate_working(self.state(source), target, **kwargs)
            self._states[source] = work
            return work.draft_message

    def _pair(self, *args):
        raise RuntimeError('distributed worker never owns a peer state')


def receive_token(state, protocol, message, session):
    """Same TOKEN framing/prefill transaction as the shared-process Session."""
    ids = deserialize_token_packet(protocol).token_ids
    message = replace(message, message_token_ids=ids)
    StatefulTokenSession._validate_message(message, state)
    content = ids[:-1] if message.termination_reason == 'eos' else ids
    frame = state.framing.incoming(message.source_agent, state.agent_id)
    session._reserve(state, session._closing_extra(state) + len(frame.prefix_ids)
                     + len(content) + len(frame.closing_ids))
    work = close_working_draft(working_state(state))
    start = work.cached_token_count + len(frame.prefix_ids)
    work = append_working_ids(work, frame.prefix_ids + content + frame.closing_ids)
    return work, message, dict(target_span=(start, start+len(content)),
                              message_prefill_count=len(content), repair_method='token_prefill')


class AgentExecutor:
    def __init__(self, agent_id, backend, *, max_context_tokens=4096, max_sessions=32,
                 max_handoffs=6, evidence=None, packet_sink=None):
        if agent_id not in ('A', 'B') or not 1 <= max_handoffs <= 100:
            raise ValueError('invalid Agent identity/handoff limit')
        if type(max_context_tokens) is not int or max_context_tokens <= 0 or type(max_sessions) is not int or max_sessions <= 0:
            raise ValueError('invalid context/session resource limit')
        self.agent_id, self.peer = agent_id, 'B' if agent_id == 'A' else 'A'
        self.backend, self.contract = backend, inspect_model_contract(backend)
        from .incremental_auto import backend_cost_identity
        from .generalized_cost import environment_identity
        # Old Mac single-process profiles MUST NOT be reused across processes.
        backend._incremental_cost_identity = dict(backend_cost_identity(backend),
            execution_scope='distributed_agent_commit_v1')
        environment_identity(backend)
        backend._generalized_cost_environment = dict(backend._generalized_cost_environment,
            execution_scope='distributed_agent_commit_v1')
        self.max_context_tokens = min(max_context_tokens, backend.context_window_tokens)
        self.max_sessions, self.max_handoffs = max_sessions, max_handoffs
        self.boot_id = uuid4().hex
        self.lock = RLock()
        self.session = None
        self.session_id = None
        self.used_sessions = set()
        self.receipts = {}
        self.outgoing = None
        self.last_handoff = 0
        self.evidence = evidence or (lambda event, row: None)
        self.packet_sink = packet_sink or (lambda direction, metadata, packet:None)
        self.transport = None
        self.peer_query = None
        self.peer_boot_id = None

    def configure(self, transport, peer_query, peer_status):
        with self.lock:
            if self.session or canonical_json(peer_status['contract']) != canonical_json(asdict(self.contract)):
                raise ValueError('active session or incompatible remote ModelContract')
            if peer_status['agent_id'] != self.peer or peer_status['boot_id'] == self.boot_id:
                raise ValueError('incorrect peer identity')
            self.transport, self.peer_query, self.peer_boot_id = transport, peer_query, peer_status['boot_id']
            return self.status()

    def start(self, session_id, config):
        with self.lock:
            if (self.session or session_id in self.used_sessions or not isinstance(session_id, str)
                    or __import__('re').fullmatch(r'[A-Za-z0-9_-]{1,128}', session_id) is None
                    or len(self.used_sessions) >= self.max_sessions):
                raise ValueError('active/reused/invalid Session or session resource limit')
            if self.transport is None:
                raise ValueError('configure compatible remote peer and relay route first')
            from .discussion import DiscussionConfig
            cfg = DiscussionConfig(**config)
            if cfg.auto is not None or cfg.handoff_plan:
                raise ValueError('selection is configured at send, not initialization')
            prompts = resolve_prompts(cfg.agent_a_system_prompt, cfg.agent_b_system_prompt,
                control_tokens=self.backend.chat_tokenizer.all_special_tokens)
            self.prompts, self.config = prompts, cfg
            prompt = prompts.agent_a.text if self.agent_id == 'A' else prompts.agent_b.text
            budget = min(self.max_context_tokens, cfg.max_context_tokens)
            local = LocalAgentSession(LLMAgent(self.agent_id, self.backend, system_prompt=prompt), budget)
            local.append_local_user(self.agent_id, cfg.question)
            self.session, self.session_id = local, session_id
            self.used_sessions.add(session_id)
            self.last_handoff, self.receipts, self.outgoing = 0, {}, None
            self.final_done = False
            self.evidence('session_started', self.status())
            return self.status()

    def _require(self, session_id):
        if self.session is None or session_id != self.session_id:
            raise ValueError('unknown/inactive Session')
        self.session.state(self.agent_id).validate()

    def status(self):
        import os
        import platform
        row = dict(host_id=platform.node(), agent_id=self.agent_id, pid=os.getpid(), boot_id=self.boot_id,
            session_id=self.session_id, contract=asdict(self.contract),
            model=asdict(self.backend.metadata), last_handoff=self.last_handoff,
            outgoing_status=self.outgoing['status'] if self.outgoing else None)
        if self.session:
            state = self.session.state(self.agent_id)
            row.update(cached_tokens=state.cached_token_count, ledger_tokens=len(state.token_ledger),
                ledger_sha256=sha(canonical_json(state.token_ledger)), version=state.version,
                pending=state.pending_token_id, sent=[m.message_id for m in state.sent_messages],
                received=[m.message_id for m in state.received_messages],
                draft_id=state.draft_message.message_id if state.draft_message else None)
        return row

    def audit(self, session_id):
        from .model.decode_kv import cache_layers
        import hashlib
        with self.lock:
            self._require(session_id)
            state = self.session.state(self.agent_id)
            digest = hashlib.sha256()
            shapes = []
            for key, value in cache_layers(state.cache):
                shapes.append(list(key.shape))
                for tensor in (key, value):
                    digest.update(tensor.detach().contiguous().cpu().view(self.backend._torch.uint8).numpy().tobytes())
            return dict(**self.status(), cache_sha256=digest.hexdigest(), cache_layer_shapes=shapes)

    def selection_summary(self, session_id):
        from .incremental_auto import backend_cost_identity
        from .generalized_cost import environment_identity
        with self.lock:
            self._require(session_id)
            s = self.session.state(self.agent_id)
            if s.assistant_open or s.draft_message or self.outgoing:
                raise ValueError('Receiver is not ready for a handoff')
            return dict(agent_id=self.agent_id, boot_id=self.boot_id, session_id=session_id,
                cached_tokens=s.cached_token_count, ledger_sha256=sha(canonical_json(s.token_ledger)),
                frame=asdict(s.framing.incoming(self.peer, self.agent_id)),
                cost_identity=backend_cost_identity(self.backend),
                environment=environment_identity(self.backend), last_handoff=self.last_handoff)

    def generate(self, session_id, stage, final=False):
        with self.lock:
            self._require(session_id)
            if self.outgoing or self.final_done:
                raise ValueError('outgoing transaction unresolved or final already generated; query/reconcile before advancing')
            expected = self.config.rounds*2 if final else stage-1
            if (type(stage) is not int or type(final) is not bool or self.last_handoff != expected
                    or (final and (self.agent_id != self.config.summarizer or stage != expected+1))
                    or (not final and (stage > self.config.rounds*2 or stage < 1
                        or self.agent_id != ('A' if stage % 2 else 'B')))):
                raise ValueError('illegal discussion stage')
            prompt = self.prompts.stages.final if final else self.prompts.stages.for_turn(self.agent_id, (stage+1)//2)
            # Each local operation still reserves framing/output space. Distributed
            # mode safely stops if a later stage does not fit; it never drops history.
            if final:
                message = self.session.generate_local_message(self.agent_id,
                    max_new_tokens=self.config.max_new_tokens, local_prompt=prompt)
            else:
                message = self.session.generate_message(self.agent_id, self.peer,
                    max_new_tokens=self.config.max_new_tokens, local_prompt=prompt)
            if final:
                self.final_done = True
            result = dict(message=asdict(message), text=self.backend.decode(message.message_token_ids,
                          skip_special_tokens=True), state=self.status())
            self.evidence('generation', dict(stage=stage, final=final, **result))
            return result

    def delivery_metadata(self, session_id, handoff, message, choice):
        return dict(session_id=session_id, handoff=handoff, sender_boot_id=self.boot_id,
            receiver_boot_id=self.peer_boot_id, contract=asdict(self.contract),
            message={k:v for k,v in asdict(message).items() if k != 'message_token_ids'},
            choice=asdict(choice))

    def prepare(self, session_id, handoff, choice):
        self._require(session_id)
        if self.outgoing or handoff != self.last_handoff+1 or handoff > self.max_handoffs:
            raise ValueError('outgoing exists or illegal handoff index')
        state = self.session.state(self.agent_id)
        message = state.draft_message
        if message is None:
            raise ValueError('no generated draft')
        StatefulTokenSession._validate_message(message, state)
        self.session._reserve(state, self.session._closing_extra(state))
        # Prepare local close BEFORE delivery. Once Receiver commits, Sender only
        # swaps a validated pointer; no model forward can fail after the ACK.
        closed = close_working_draft(working_state(state))
        closed.validate()
        if choice.mode == CommunicationMode.TOKEN:
            protocol = serialize_token_ids(message.message_token_ids)
        else:
            envelope, _ = prepare_incremental_message(state, message, contract=self.contract)
            protocol = encode_incremental_message(envelope)
        metadata = self.delivery_metadata(session_id, handoff, message, choice)
        packet = pack_delivery(metadata, protocol)
        self.packet_sink('outgoing', metadata, packet)
        expected = dict(session_id=session_id, message_id=message.message_id, handoff=handoff,
            packet_sha256=sha(packet), receiver_boot_id=self.peer_boot_id,
            source=self.agent_id, target=self.peer, mode=choice.mode.value,
            kv_strategy=choice.kv_strategy.value if choice.kv_strategy else None)
        self.outgoing = dict(status='prepared', packet=packet, expected=expected,
                             closed=closed, message=message, protocol_bytes=len(protocol))
        self.evidence('outgoing_prepared', dict(**expected, protocol_bytes=len(protocol),
            protocol_sha256=sha(protocol), application_bytes=len(packet)))
        return self.outgoing

    def _finish(self, receipt):
        out = self.outgoing
        if any(receipt.get(k) != v for k,v in out['expected'].items()):
            raise ValueError('reconciliation receipt identity mismatch')
        if receipt.get('schema') != RECEIPT_SCHEMA or receipt.get('status') != 'committed' or not (
                receipt.get('state_committed') is True and receipt.get('cache_imported') is True):
            raise ValueError('Receiver has no confirmed commit')
        old = self.session.state(self.agent_id)
        record = replace(out['message'], payload_mode=receipt['mode'].upper(),
            receiver_message_start=receipt['target_span'][0],
            receiver_content_token_count=receipt['target_span'][1]-receipt['target_span'][0])
        work = replace(out['closed'], draft_message=None,
            sent_messages=old.sent_messages+(record,), version=old.version+1)
        work.validate()
        self.session._states[self.agent_id] = work
        self.last_handoff = receipt['handoff']
        self.outgoing = None
        self.evidence('sender_committed', dict(receipt=receipt, state=self.status()))

    def decide(self, session_id, handoff, choice, auto=None, network=None):
        from .incremental_auto import IncrementalAutoConfig, IncrementalSelector
        from .network.handoff_network import HandoffNetworkSnapshot
        with self.lock:
            self._require(session_id)
            decision = None
            if auto is not None:
                settings = dict(auto)
                if settings.get('profile'):
                    from .generalized_cost import load_cost_profile
                    settings['profile'] = load_cost_profile(settings['profile'])
                if 'candidates' in settings:
                    settings['candidates'] = tuple(settings['candidates'])
                if settings.get('synthetic_assumptions') is not None:
                    from .synthetic_cost import SyntheticCostAssumptions
                    settings['synthetic_assumptions'] = SyntheticCostAssumptions.from_dict(settings['synthetic_assumptions'])
                c = IncrementalAutoConfig(**settings)
                summary = self.peer_query('selection', session_id=session_id)
                if summary['boot_id'] != self.peer_boot_id or summary['last_handoff'] != self.last_handoff:
                    raise ValueError('Receiver incarnation/handoff mismatch')
                frame = SimpleNamespace(**{k:tuple(v) for k,v in summary['frame'].items()})
                proxy = SimpleNamespace(cached_token_count=summary['cached_tokens'], pending_token_id=None,
                    assistant_open=False, selection_ledger_sha256=summary['ledger_sha256'],
                    selection_framing_sha256=sha(canonical_json(summary['frame'])),
                    framing=SimpleNamespace(incoming=lambda *args: frame),
                    backend=SimpleNamespace(_incremental_cost_identity=summary['cost_identity'],
                                            _generalized_cost_environment=summary['environment']))
                sender = self.session.state(self.agent_id)
                def overhead(candidate):
                    selected = HandoffChoice.parse(candidate, **(dict(recompute_tokens=c.recompute_tokens,
                        recompute_ratio=c.recompute_ratio) if candidate == 'kv:v2' else {}))
                    metadata = self.delivery_metadata(session_id, handoff, sender.draft_message, selected)
                    return len(pack_delivery(metadata, b''))
                choice, decision = IncrementalSelector(c).select(sender, proxy, sender.draft_message,
                    HandoffNetworkSnapshot(**network), kv_contract_ready=True, require_workload_domain=True, wire_overhead=overhead)
            else:
                choice = HandoffChoice(**choice) if isinstance(choice, dict) else HandoffChoice.parse(choice)
            return dict(choice=asdict(choice), decision=decision)

    def send(self, session_id, handoff, choice, auto=None, network=None):
        with self.lock:
            selected = self.decide(session_id, handoff, choice, auto, network)
            choice, decision = HandoffChoice(**selected['choice']), selected['decision']
            out = self.prepare(session_id, handoff, choice)
            out['status'] = 'in_flight'
            result = self.transport.deliver(out['packet'], source=self.agent_id, target=self.peer,
                                            expected=out['expected'])
            out['status'] = result['outcome']
            if result['outcome'] == 'receiver_committed':
                self._finish(result['receipt'])
            # All other outcomes block progression. Even not_sent is never retried
            # implicitly; release/diagnosis is an explicit caller decision.
            result.update(decision=decision, protocol_bytes=out['protocol_bytes'],
                          application_bytes=len(out['packet']), identity=out['expected'])
            self.evidence('send_result', result)
            return result

    def reconcile(self, session_id):
        with self.lock:
            self._require(session_id)
            if not self.outgoing:
                raise ValueError('no unresolved outgoing transaction')
            out = self.outgoing
            result = self.peer_query('receipt', session_id=session_id,
                                     message_id=out['expected']['message_id'])
            # Absence does not prove failure: original delivery may still execute.
            if result.get('status') == 'committed':
                self._finish(result)
                return dict(outcome='receiver_committed', receipt=result)
            return dict(outcome='delivery_unknown', receiver_status=result)

    def receipt(self, session_id, message_id):
        with self.lock:
            self._require(session_id)
            return self.receipts.get(message_id, dict(status='unknown', receiver_boot_id=self.boot_id,
                                                      session_id=session_id, message_id=message_id))

    def receive(self, packet, source, target):
        started = time.monotonic()
        with self.lock:
            meta, protocol = unpack_delivery(packet)
            if set(meta) != {'session_id','handoff','sender_boot_id','receiver_boot_id','contract','message','choice'}:
                raise ValueError('invalid remote metadata fields')
            self._require(meta['session_id'])
            choice = HandoffChoice(**meta['choice'])
            info = meta['message']
            identity = dict(session_id=meta['session_id'], message_id=info['message_id'],
                handoff=meta['handoff'], packet_sha256=sha(packet), receiver_boot_id=self.boot_id,
                source=source, target=target, mode=choice.mode.value,
                kv_strategy=choice.kv_strategy.value if choice.kv_strategy else None)
            if source != self.peer or target != self.agent_id or meta['receiver_boot_id'] != self.boot_id or meta['sender_boot_id'] != self.peer_boot_id:
                raise ValueError('remote endpoint/incarnation mismatch')
            if (type(meta['handoff']) is not int or type(info['sequence']) is not int
                    or not 1 <= info['sequence'] <= self.max_handoffs
                    or info['message_id'] != f'{source}:{info["sequence"]}'):
                raise ValueError('invalid bounded message identity/sequence')
            previous = self.receipts.get(info['message_id'])
            if previous:
                if previous['packet_sha256'] != identity['packet_sha256']:
                    raise ValueError('duplicate message identity with conflicting bytes')
                return previous  # Same receipt, zero import/forward/commit.
            stages = ['request_received']
            self.evidence('request_received', identity)
            self.packet_sink('incoming', meta, packet)
            try:
                if canonical_json(meta['contract']) != canonical_json(asdict(self.contract)):
                    raise ValueError('remote ModelContract mismatch')
                if (self.outgoing or self.final_done or meta['handoff'] != self.last_handoff+1
                        or meta['handoff'] > self.max_handoffs
                        or source != ('A' if meta['handoff'] % 2 else 'B')):
                    raise ValueError('outgoing unresolved or handoff out of order/limit')
                old = self.session.state(self.agent_id)
                sequence = len(old.received_messages)+1
                if (info['source_agent'] != source or info['target_agent'] != target
                        or info['sequence'] != sequence or info['message_id'] != f'{source}:{sequence}'):
                    raise ValueError('message direction/sequence/identity mismatch')
                message = AgentMessage(**{**info, 'sender_message_span':tuple(info['sender_message_span']),
                                          'message_token_ids':()})
                if choice.mode == CommunicationMode.TOKEN:
                    ids = deserialize_token_packet(protocol).token_ids
                else:
                    decoded = decode_incremental_message(protocol, expected_contract=self.contract,
                        source=source, target=target, sequence=sequence)
                    ids = decoded.raw_generated_token_ids
                    if (decoded.message_id != message.message_id or decoded.termination_reason != message.termination_reason
                            or decoded.sender_span_start != message.sender_message_span[0]):
                        raise ValueError('inner/outer KV identity mismatch')
                message = replace(message, message_token_ids=ids)
                StatefulTokenSession._validate_message(message, old)
                start, end = message.sender_message_span
                if type(start) is not int or start <= 0 or end-start != len(ids):
                    raise ValueError('invalid source span')
                stages.append('protocol_verified')
                if choice.mode == CommunicationMode.TOKEN:
                    work, message, details = receive_token(old, protocol, message, self.session)
                else:
                    work, decoded, detail = receive_incremental_kv(old, protocol, contract=self.contract,
                        source=source, sequence=sequence, strategy=choice.kv_strategy,
                        recompute_tokens=choice.recompute_tokens, recompute_ratio=choice.recompute_ratio,
                        max_context_tokens=self.session.max_context_tokens)
                    details = asdict(detail)
                stages.append('cache_imported_and_corrected')
                record = replace(message, payload_mode=choice.mode.value.upper(),
                    receiver_message_start=details['target_span'][0],
                    receiver_content_token_count=details['target_span'][1]-details['target_span'][0])
                work = replace(work, received_messages=old.received_messages+(record,), version=old.version+1)
                work.validate()
                result = dict(schema=RECEIPT_SCHEMA, architecture='distributed', **identity, status='committed',
                    cache_imported=True, state_committed=True, stages=stages+['state_committed'],
                    protocol_bytes=len(protocol), protocol_sha256=sha(protocol),
                    receiver_cache_before=old.cached_token_count, receiver_cache_after=work.cached_token_count,
                    receiver_ledger_sha256=sha(canonical_json(work.token_ledger)),
                    receive_ms=(time.monotonic()-started)*1000, **details)
                # In-memory linearization boundary under this process's single
                # state lock. No disk/cache durability or cross-process atomicity.
                self.session._states[self.agent_id] = work
                self.last_handoff = meta['handoff']
            except Exception as exc:
                result = dict(schema=RECEIPT_SCHEMA, architecture='distributed', **identity, status='rejected',
                    cache_imported=False, state_committed=False, stages=stages,
                    error=f'{type(exc).__name__}: {exc}')
            self.receipts[info['message_id']] = result
            self.evidence('receiver_result', result)
            return result

    def release(self, session_id):
        with self.lock:
            self._require(session_id)
            result = self.status()
            result['abandoned_outgoing'] = self.outgoing['expected'] if self.outgoing else None
            self.session = self.outgoing = None
            self.session_id = None
            self.receipts = {}
            gc.collect()
            self.evidence('session_released', result)
            return result
