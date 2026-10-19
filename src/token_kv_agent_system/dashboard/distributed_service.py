"""Dashboard metadata adapter over independent Workers; never receives cache bytes."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
from uuid import uuid4

from ..distributed_discussion import DistributedDiscussion
from ..distributed_worker import control_call
from ..network.handoff_network import HandoffNetwork
from .discussion_service import (DiscussionService, DiscussionBusyError, DiscussionUnavailableError)

ARCHITECTURE = 'distributed'
TRANSPORT = 'RemoteAgentTransport / dual Socket Relay'


def turn_view(generated, sequence, round_number, target=None):
    m = generated['message']
    return dict(agent=m['source_agent'], text=generated['text'], token_count=len(m['message_token_ids']),
        message_id=m['message_id'], sequence=sequence, round=round_number, target=target,
        **{k:m.get(k) for k in ('termination_reason','configured_max_new_tokens',
          'effective_max_new_tokens','context_remaining_tokens','budget_limit_reason')})


def handoff_view(row, session_id):
    """Only a matching application receipt can produce append_succeeded=true."""
    d = row.get('delivery') or {}
    r, identity = d.get('receipt') or {}, d.get('identity') or {}
    m = row['generated']['message']
    index = row['network']['handoff_index']
    required = {'session_id','message_id','handoff','packet_sha256','receiver_boot_id','source','target','mode','kv_strategy'}
    matches = (r.get('schema') == 'remote_agent_commit_v1' and required <= identity.keys() and r.get('session_id') == session_id and r.get('message_id') == m['message_id']
        and r.get('handoff') == index and r.get('source') == row['source'] and r.get('target') == row['target']
        and all(r.get(k) == v for k,v in identity.items()) and bool(identity))
    committed = (matches and d.get('outcome') == 'receiver_committed' and r.get('status') == 'committed'
                 and r.get('state_committed') is True and r.get('cache_imported') is True)
    outcome = d.get('outcome', 'delivery_unknown')
    if outcome == 'receiver_committed' and not committed:
        outcome = 'delivery_unknown'
    choice = row.get('selection', {}).get('choice', {})
    decision = d.get('decision') or row.get('selection', {}).get('decision')
    selected = next((c for c in (decision or {}).get('candidate_estimates', [])
                     if c['candidate'] == decision.get('selected_candidate')), {})
    return dict(architecture=ARCHITECTURE, session_id=session_id, sequence=index,
        source=row['source'], target=row['target'], discussion_round=row['round'], message_id=m['message_id'],
        requested_mode='auto' if decision else choice.get('mode'), requested_kv_strategy=choice.get('kv_strategy'),
        selected_mode=choice.get('mode'), selected_kv_strategy=choice.get('kv_strategy'),
        executed_mode=r.get('mode') if committed else None, executed_kv_strategy=r.get('kv_strategy') if committed else None,
        append_succeeded=True if committed else False if outcome in ('receiver_rejected','not_sent') else None,
        actual_execution_status=outcome, receiver_receipt=r or None, delivery_identity=identity or None,
        socket_measurement=d.get('measurement'), observed_handoff_wall_ms=row.get('observed_handoff_wall_ms'),
        socket_application_delivery_ms=(d.get('measurement') or {}).get('elapsed_ms'),
        receiver_processing_ms=r.get('receive_ms'), measurement_source='Mac TCP / loopback' if row.get('loopback', False) else 'TCP / configured hosts',
        protocol_bytes=d.get('protocol_bytes'), application_bytes=d.get('application_bytes'),
        wire_bytes=d.get('protocol_bytes'), observed_local_ms=None,
        observation_scope='distributed_agent_commit_v1', network=row['network'], auto_decision=decision,
        predicted_wire_bytes=selected.get('predicted_wire_bytes'), predicted_local_ms=selected.get('predicted_local_ms'),
        predicted_total_ms=selected.get('predicted_total_ms'), simulated_transfer_ms=row.get('simulated_application_transfer_ms'),
        message_content_token_count=len(m['message_token_ids'])-(m['termination_reason']=='eos'), sender_kv_span=m.get('sender_message_span'),
        receiver_kv_target_span=r.get('target_span'), receiver_cache_length_before=r.get('receiver_cache_before'),
        receiver_cache_length_after=r.get('receiver_cache_after'), receiver_message_prefill_count=r.get('message_prefill_count'),
        kv_repair_method=r.get('repair_method'), receiver_recomputed_token_count=r.get('recomputed_tokens'),
        repaired_token_indices=r.get('repaired_token_indices'), repaired_layer_indices=r.get('repaired_layer_indices'),
        repair_layer_token_count=r.get('repair_layer_token_count'), rope_position_delta=r.get('rope_position_delta'),
        unpack_succeeded=True if committed else None, cache_ledger_consistent=True if committed else None,
        error=d.get('error') or r.get('error'), fallback_reason=None)


class DistributedDashboardService(DiscussionService):
    def __init__(self, settings=None, *, collector=None, history_path=None):
        super().__init__(collector=collector)
        self.settings = settings
        self.history_path = Path(history_path) if history_path else None
        self._runtime = None
        self._health = None
        self._health_at = 0
        self._health_lock = __import__('threading').Lock()
        if self.history_path and self.history_path.exists():
            for record in json.loads(self.history_path.read_text())[-3:]:
                if record.get('architecture') != ARCHITECTURE:
                    continue
                if record['status'] == 'running':
                    record.update(status='delivery_unknown', error='Coordinator restarted; cache continuation unavailable',
                                  failure_stage='coordinator_restart', active_handoff=None)
                record['resumable'] = False
                self._sessions[record['session_id']] = record
                self._started_ids.add(record['session_id'])
                self._latest = record['session_id']

    def health(self, force=False):
        # Never hold the projection lock across Worker HTTP or Relay TCP probes.
        with self._health_lock:
            if not force and self._health and time.monotonic()-self._health_at < 2:
                return deepcopy(self._health)
            state = dict(architecture=ARCHITECTURE, transport=TRANSPORT, coordinator='ready', workers={}, relays={},
                         available=False, status='unavailable', model_contract_consistent=None,
                         measurement_source=(self.settings or {}).get('measurement_source', 'TCP / configured hosts'),
                         startup='启动：.venv/bin/python run_distributed_dashboard.py --output-dir benchmark-results/distributed-dashboard-new')
            settings = self.settings
            if settings:
                def worker(item):
                    name,url = item
                    try:
                        row = control_call(url, 'status', timeout=2)
                        model_path = row['model']['model_name']
                        row['model']['revision'] = model_path.split('/snapshots/',1)[1].split('/')[0] if '/snapshots/' in model_path else None
                        return name, dict(status='running' if row.get('session_id') else 'ready', **row)
                    except Exception as exc:
                        # Inference serializes Worker control. Cached identity is explicitly stale.
                        old = (self._health or {}).get('workers', {}).get(name, {})
                        return name, dict(last_known=old.get('last_known', old), status='running' if self._active else 'unavailable', error=str(exc))
                with ThreadPoolExecutor(max_workers=2) as pool:
                    state['workers'] = dict(pool.map(worker, settings['workers'].items()))
                for name, relay in settings.get('relays', {}).items():
                    errors = []
                    for endpoint in relay['listeners']:
                        for address in (endpoint['listen'], endpoint.get('upstream')):
                            if address is None:
                                continue
                            try:
                                with socket.create_connection(tuple(address), timeout=.5):
                                    pass
                            except OSError as exc:
                                errors.append(str(exc))
                    state['relays'][name] = dict(**relay, status='unavailable' if errors else 'ready',
                        check='TCP listeners and configured upstreams; end-to-end commit is verified per receipt', errors=errors)
                w = state['workers']
                if all(w.get(n,{}).get('status') in ('ready','running') for n in ('A','B')):
                    state['model_contract_consistent'] = (w['A']['contract'] == w['B']['contract']
                        and w['A']['agent_id'] == 'A' and w['B']['agent_id'] == 'B'
                        and (w['A']['host_id'], w['A']['pid']) != (w['B']['host_id'], w['B']['pid']))
                relays_ready = len(state['relays']) == 2 and all(r['status'] == 'ready' for r in state['relays'].values())
                state['available'] = bool(state['model_contract_consistent'] and relays_ready and not self._closed)
                state['status'] = 'running' if self._active else 'ready' if state['available'] else 'unavailable'
            self._health, self._health_at = deepcopy(state), time.monotonic()
            return state

    def latest(self):
        health = self.health()
        with self._lock:
            record = self._sessions.get(self._latest)
            last_handoff = next((h for s in reversed(list(self._sessions.values())) for h in reversed(s['handoffs'])
                                 if h['append_succeeded'] is True), None)
            failure = next((s for s in reversed(list(self._sessions.values())) if s.get('error')), None)
            return dict(available=health['available'], model=None, architecture=ARCHITECTURE, health=health,
                profile_status='unavailable: distributed calibration required; Local profiles rejected',
                session=deepcopy(record), active_session_id=self._active,
                last_successful_handoff=deepcopy(last_handoff), recent_failure={k:failure.get(k) for k in ('session_id','architecture','status','failure_stage','error')} if failure else None,
                recent_sessions=[dict(session_id=s['session_id'], architecture=ARCHITECTURE, status=s['status'],
                    question=s['config']['question'], mode=s['config']['mode'], created_at=s['created_at'])
                    for s in reversed(list(self._sessions.values()))])

    def _save(self):
        if self.history_path:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.history_path.with_suffix('.tmp')
            temp.write_text(json.dumps(list(self._sessions.values()), ensure_ascii=False, indent=2)+'\n')
            temp.replace(self.history_path)

    def _event(self, session_id, kind, data):
        with self._lock:
            # Dedupe stage projection by stable message/handoff identity.
            s = self._sessions[session_id]
            if 'turn' in data and any(t['message_id'] == data['turn']['message_id'] for t in s['turns']):
                return
            if 'handoff' in data:
                s['handoffs'] = [h for h in s['handoffs'] if h['sequence'] != data['handoff']['sequence']]
            super()._event(session_id, kind, dict(data, architecture=ARCHITECTURE, transport=TRANSPORT))
            self._save()

    def start(self, request):
        if request.architecture != ARCHITECTURE:
            raise ValueError('Distributed endpoint requires architecture=distributed')
        if request.scenario != 'discussion':
            raise ValueError('Distributed supports ordinary discussion only')
        if request.cost_profile is not None:
            identities = request.cost_profile.get('model_identity', {})
            if any(identities.get(n, {}).get('execution_scope') != 'distributed_agent_commit_v1' for n in ('sender','receiver')):
                raise ValueError('Local Profile cannot calibrate distributed execution; distributed Profile unavailable')
        config = request.runtime_config()
        health = self.health(force=True)
        if not health['available']:
            raise DiscussionUnavailableError('分布式服务未就绪；检查 Worker、双 Relay 与 ModelContract')
        with self._lock:
            if self._closed or self._active:
                raise DiscussionBusyError('A distributed discussion is active or service closed')
            if self._runtime and self._runtime.status in ('failed','delivery_unknown'):
                raise DiscussionBusyError('Inspect unresolved session and explicitly release before a new run')
            if any(w.get('session_id') for w in health['workers'].values()):
                raise DiscussionBusyError('Workers already hold a Session; inspect/release; no cache recovery')
            sid = request.session_id or uuid4().hex
            if sid in self._started_ids:
                raise DiscussionBusyError('Session already used; cannot restart cached history')
            self._started_ids.add(sid)
            self._active = self._latest = sid
            network = HandoffNetwork(config.network, lock=self._lock)
            self._networks[sid] = network
            stamp = datetime.now(timezone.utc).isoformat()
            self._sessions[sid] = dict(session_id=sid, architecture=ARCHITECTURE, resumable=False,
                status='running', revision=0, phase='initializing', round=0, sequence=0, agent=None,
                config=request.model_dump(exclude={'session_id'}), prompt_config=self.preview_prompts(request),
                model=None, workers=health['workers'], transport=TRANSPORT, network=network.current(), network_updates=[],
                turns=[], handoffs=[], auto_decisions=[], active_handoff=None, final_answer=None, summary=None, score=None,
                cache_lengths={}, cache_released=False, error=None, failure_stage=None, termination_reason=None,
                created_at=stamp, updated_at=stamp)
            while len(self._sessions)>3:
                old = next(iter(self._sessions)); del self._sessions[old]; self._networks.pop(old, None)
            self._runtime = DistributedDiscussion(self.settings['workers'], routes=self.settings['routes'],
                config=config, timeout=self.settings.get('timeout',120), session_id=sid, network=network,
                event_sink=lambda kind,row:self._project(sid,kind,row))
            self._event(sid,'DISCUSSION_STARTED',{})
            self._executor.submit(self._runtime.run)
            return deepcopy(self._sessions[sid])

    def _project(self, sid, kind, row):
        with self._lock:
            s = self._sessions[sid]
            if kind == 'control_command':
                return
            if kind == 'worker_prepared':
                self._event(sid,'DISCUSSION_WORKER_READY',dict(agent=row['agent'], worker=row['state']))
            elif kind in ('generation_started','generation_completed'):
                data = dict(agent=row['source'], target=row['target'], sequence=row['index'], round=row['round'])
                if kind == 'generation_completed':
                    data['turn'] = turn_view(row['generated'], row['index'], row['round'], row['target'])
                self._event(sid,'AGENT_TURN_STARTED' if kind.endswith('started') else 'AGENT_TURN_COMPLETED',data)
            elif kind in ('decision_completed','delivery_started','delivery_completed'):
                data = dict(agent=row['source'], target=row['target'], round=row['round'], sequence=row['network']['handoff_index'],
                    message_id=row['generated']['message']['message_id'], network=row['network'],
                    requested_mode=s['config']['mode'], selected_mode=row['selection']['choice']['mode'],
                    selected_kv_strategy=row['selection']['choice']['kv_strategy'])
                if kind == 'decision_completed':
                    if row['selection']['decision']:
                        self._event(sid,'AUTO_DECISION',dict(data,decision=row['selection']['decision']))
                elif kind == 'delivery_started':
                    self._event(sid,'HANDOFF_STARTED',data)
                else:
                    row = dict(row, loopback=self.settings.get('measurement_source') == 'Mac TCP / loopback')
                    h = handoff_view(row,sid)
                    self._event(sid,'HANDOFF_COMPLETED' if h['append_succeeded'] is True else 'HANDOFF_UNCONFIRMED' if h['actual_execution_status']=='delivery_unknown' else 'HANDOFF_FAILED',dict(data,handoff=h))
            elif kind == 'final_started':
                self._event(sid,'DISCUSSION_FINALIZING',dict(agent=row['source']))
            elif kind == 'final_completed':
                s['final_answer'] = turn_view(row,s['config']['rounds']*2+1, s['config']['rounds'])
            elif kind == 'discussion_finished':
                # A control timeout cannot tell whether the receiver committed.
                if row.get('failure_stage','').startswith('delivery:') and not any(h['sequence']==s['sequence'] for h in s['handoffs']):
                    pending = dict(row['turns'][-1], loopback=self.settings.get('measurement_source')=='Mac TCP / loopback')
                    h = handoff_view(pending,sid)
                    self._event(sid,'HANDOFF_UNCONFIRMED',dict(handoff=h))
                if row.get('failure_stage','').startswith('decision:'):
                    pending=row['turns'][-1]
                    self._event(sid,'AUTO_DECISION_FAILED',dict(sequence=pending['network']['handoff_index'], agent=pending['source'], target=pending['target'], message_id=pending['generated']['message']['message_id'], error=row.get('error'), decision_status='failed'))
                s.update(status=row['status'], error=row.get('error'), failure_stage=row.get('failure_stage'),
                    termination_reason='completed' if row['status']=='completed' else row['status'],
                    cache_lengths={n:v['cached_tokens'] for n,v in row.get('final_states',{}).items()},
                    cache_released=len(row['released'])==2, raw_result=row)
                hs = s['handoffs']
                s['summary'] = dict(rounds=s['config']['rounds'], completed_turns=len(s['turns']),
                    handoff_count=sum(h['append_succeeded'] is True for h in hs), attempted_handoff_count=len(hs),
                    total_wire_bytes=sum(h.get('protocol_bytes') or 0 for h in hs), final_cache_lengths=s['cache_lengths'],
                    token_handoffs=sum(h['executed_mode']=='token' for h in hs),
                    **{f'kv_v{i}_handoffs':sum(h['executed_kv_strategy']==strategy for h in hs) for i,strategy in enumerate(
                        ('raw_delta','rope_aligned_delta','suffix_recompute_delta','numeric_slot_delta'))})
                self._active = None
                self._health_at = 0
                self._event(sid,'DISCUSSION_COMPLETED' if row['status']=='completed' else 'DISCUSSION_FAILED',
                    {k:s[k] for k in ('final_answer','summary','error','failure_stage','termination_reason','cache_lengths','cache_released')})

    def release(self, sid):
        with self._lock:
            record = self._sessions[sid]
            if self._active:
                raise DiscussionBusyError('Cannot release while background execution is active')
        released = {}
        for name,url in (self.settings or {}).get('workers',{}).items():
            status = control_call(url,'status',timeout=2)
            if status.get('session_id') == sid:
                # Incarnation must match; metadata never restores a cache.
                if status['boot_id'] != record['workers'][name]['boot_id']:
                    raise DiscussionBusyError('Worker restarted; old cache cannot continue')
                released[name] = control_call(url,'release',session_id=sid,timeout=5)
        with self._lock:
            record.update(cache_released=True, release_note='Explicit abandon; does not roll back a remote commit')
            if self._runtime and self._runtime.session_id == sid:
                self._runtime = None
            self._health_at=0; self._save()
        return dict(session_id=sid, architecture=ARCHITECTURE, released=list(released), rollback=False)

    def close(self):
        # No thread interruption during a transaction. Control/socket deadlines
        # bound completion; unknown caches remain available for explicit diagnosis.
        if self._runtime:
            self._runtime.request_stop()
        super().close()
