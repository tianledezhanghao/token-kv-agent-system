"""8.3A opt-in, single-process research instrumentation and bandwidth analysis.

Production algorithms are called unchanged. Patches exist only inside an explicit
context in this exclusive CLI; never use this recorder in a serving process.
"""
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from functools import wraps
import hashlib
import math
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

from .communication import HandoffChoice
from .model.huggingface import synchronize_device


def statistics(values):
    values = list(values)
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('nonempty finite nonnegative samples required')
    center = median(values)
    return dict(sample_count=len(values), median=center, min=min(values), max=max(values),
                mad=median(abs(x-center) for x in values))


def bandwidth_analysis(token_ms, kv_ms, token_bytes, kv_bytes,
                       bandwidths=(1,20,100,1000,10000), base_latency_ms=2):
    for x in (token_ms, kv_ms, base_latency_ms):
        if type(x) not in (int,float) or not math.isfinite(x) or x < 0:
            raise ValueError('finite nonnegative local costs/latency required')
    if any(type(x) is not int or x < 0 for x in (token_bytes, kv_bytes)):
        raise ValueError('nonnegative integer wire sizes required')
    curves = []
    for b in bandwidths:
        if type(b) not in (int,float) or not math.isfinite(b) or b <= 0:
            raise ValueError('positive finite bandwidth required')
        tn = token_bytes*8/(b*1000)+base_latency_ms
        kn = kv_bytes*8/(b*1000)+base_latency_ms
        curves.append(dict(bandwidth_mbps=b, token_network_ms=tn, kv_network_ms=kn,
                           token_total_ms=token_ms+tn, kv_total_ms=kv_ms+kn,
                           nature='measured_local_plus_simulated_network'))
    delta_local = kv_ms-token_ms
    delta_bytes = kv_bytes-token_bytes
    crossing = (-delta_bytes*8/(1000*delta_local)) if delta_local else None
    crossing = crossing if crossing is not None and crossing > 0 else None
    return dict(curves=curves, break_even_mbps=crossing,
                reason='positive_crossing' if crossing else 'equal_all_bandwidths' if not delta_local and not delta_bytes else 'no_positive_bandwidth_crossing',
                delta_local_ms=delta_local, delta_wire_bytes=delta_bytes,
                base_latency_ms=base_latency_ms, physical_network='not_measured')


class StageRecorder:
    """Nested inclusive/exclusive times. Only exclusive totals may be added.

    Synchronization is included in the span (pre and post). Child times are
    subtracted exactly once. Original adapter submetrics are informational only.
    """
    def __init__(self, sync=lambda: None):
        self.sync = sync
        self.stack = []
        self.events = []
        self.adapter_metrics = defaultdict(list)

    @contextmanager
    def span(self, name):
        started = perf_counter()
        entry = dict(name=name, child_ms=0)
        parent = self.stack[-1]['name'] if self.stack else None
        self.stack.append(entry)
        try:
            self.sync()
            yield
        finally:
            self.sync()
            elapsed = (perf_counter()-started)*1000
            self.stack.pop()
            self.events.append(dict(name=name, parent=parent, inclusive_ms=elapsed,
                                    exclusive_ms=max(0,elapsed-entry['child_ms'])))
            if self.stack:
                self.stack[-1]['child_ms'] += elapsed

    def wrap(self, original, name, after=None):
        @wraps(original)
        def wrapped(*args, **kwargs):
            label = name(*args, **kwargs) if callable(name) else name
            with self.span(label):
                result = original(*args, **kwargs)
                if after is not None:
                    after(result)
                return result
        return wrapped

    def aggregate(self):
        result = {}
        for event in self.events:
            row = result.setdefault(event['name'],dict(calls=0,inclusive_ms=0,exclusive_ms=0))
            row['calls'] += 1
            for field in ('inclusive_ms','exclusive_ms'):
                row[field] += event[field]
        return result


@contextmanager
def instrument(session, recorder):
    """Scope wrappers around actual operation boundaries, no algorithm rewrite."""
    from . import stateful_token as st, incremental_kv as export
    from . import incremental_kv_receive as rx, incremental_kv_protocol as protocol
    from .model import kv_adapter as adapter, decode_kv, token_slot_repair as slot
    backend = session.state('A').backend
    def restored(result):
        for key in ('cpu_tensor_reconstruction_ms','host_to_device_ms',
                    'cache_object_construction_ms','total_restore_ms'):
            recorder.adapter_metrics[key].append(getattr(result,key))
    def extracted(result):
        recorder.adapter_metrics['device_to_host_and_host_bytes_ms'].append(result[2])
    def state_label(kind):
        return lambda state,*a,**kw: kind+'_'+state.agent_id
    def append_label(state, ids):
        if state.agent_id == 'A':
            return 'sender_close_forward'
        frame = state.framing.incoming('A','B')
        if tuple(ids) == frame.prefix_ids:
            return 'receiver_prefix'
        if tuple(ids) == frame.closing_ids:
            return 'receiver_closing'
        return 'suffix_recompute'
    def forward_label(*a,**kw):
        if any(e['name']=='slot_repair' for e in recorder.stack):
            return 'slot_forward'
        if any(e['name']=='export' for e in recorder.stack):
            return 'sender_pending_forward'
        # All other forwards stay in their framing/content parent spans.
        return 'model_forward'
    def clone_label(*a,**kw):
        return 'sender_pending_cache_clone' if any(e['name']=='export' for e in recorder.stack) else 'cache_clone'
    targets = [
        (session,'_pair','entry_state_validation',None),
        (st,'serialize_token_ids','token_encode',None),
        (st,'deserialize_token_packet','token_decode',None),
        (session.transport,'transfer','memory_transport',None),
        (export,'prepare_incremental_message','export',None),
        (protocol,'encode_incremental_message_with_metrics','kv_encode_checksum',None),
        (rx,'decode_incremental_message','kv_decode_checksum',None),
        (rx,'restore_huggingface_kv_cache','restore',restored),
        (decode_kv,'extract_span','span_tensor_clone',None),
        (adapter,'extract_huggingface_kv_payload','host_export',extracted),
        (rx,'append_delta','cache_append_cat_reconstruct',None),
        (rx,'inspect_rope','rope_contract_inspection',None),
        (rx,'rotary_tables','rope_tables',None),
        (rx,'rephase_key','rope_keys',None),
        (rx,'first_numeric','token_selection',None),
        (rx,'repair_slot','slot_repair',None),
        (rx,'delta_prefix','suffix_prefix_view',None),
        (slot,'reconstruct','slot_cache_reconstruct',None),
        (st,'working_state',state_label('work_clone'),None),
        (rx,'working_state',state_label('work_clone'),None),
        (st,'close_working_draft',state_label('close_draft'),None),
        (rx,'close_working_draft',state_label('close_draft'),None),
        (st,'append_working_ids','receiver_framing_message_prefill',None),
        (rx,'append_working_ids',append_label,None),
        (backend,'continue_from_cache',forward_label,None),
        (backend,'clone_cache',clone_label,None),
    ]
    with ExitStack() as stack:
        for obj,attr,label,after in targets:
            stack.enter_context(patch.object(obj,attr,recorder.wrap(getattr(obj,attr),label,after)))
        yield


def cache_sha256(cache):
    """Evidence only: never called inside the measured handoff."""
    h = hashlib.sha256()
    for layer in cache.layers:
        for t in (layer.keys,layer.values):
            h.update(t.detach().contiguous().cpu().view(dtype=__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def real_prefix(state, n):
    """Truncate a real Decode draft at N; preserve its original absolute K/V.

    Leave accepted token N pending exactly as if Decode had stopped there. Never
    re-Prefill the content. Prefix crop/clone is setup, outside the handoff timer.
    """
    from .model.decode_kv import cache_layers
    from .model.delta_repair import reconstruct
    message = state.draft_message
    content = message.message_token_ids[:-1] if message.termination_reason=='eos' else message.message_token_ids
    if type(n) is not int or not 0 < n <= len(content):
        raise ValueError('prefix length outside actual generated content')
    start = message.sender_message_span[0]
    count = start+n-1
    message = replace(message,message_token_ids=content[:n],sender_message_span=(start,start+n),
                      termination_reason='max_new_tokens',configured_max_new_tokens=n,
                      effective_max_new_tokens=n,context_remaining_tokens=None,budget_limit_reason=None)
    cache = reconstruct(tuple((k[...,:count,:],v[...,:count,:]) for k,v in cache_layers(state.cache)),state.backend._model.config)
    work = replace(state, cache=cache,cached_token_count=count,token_ledger=state.token_ledger[:start+n],
                   pending_token_id=content[n-1],draft_message=message)
    work.validate()
    return work


def measured_handoff(session, saved, candidate, *, mode='steady', recompute_tokens=None,
                     recompute_ratio=None):
    """External synchronized boundary includes lock, validations and atomic commit."""
    if mode not in ('steady','instrumented'):
        raise ValueError('unknown timing mode')
    session._states = dict(saved)
    backend = saved['A'].backend
    sync = lambda: synchronize_device(backend._torch,backend.device)
    recorder = StageRecorder(sync)
    choice = HandoffChoice.parse(candidate,recompute_tokens=recompute_tokens,recompute_ratio=recompute_ratio)
    def run():
        if candidate=='token':
            return session.handoff('A','B')
        return session.handoff_kv('A','B',strategy=choice.kv_strategy,
                                  recompute_tokens=recompute_tokens,recompute_ratio=recompute_ratio)
    with ExitStack() as stack:
        if mode=='instrumented':
            stack.enter_context(instrument(session,recorder))
        sync()
        started = perf_counter()
        if mode=='instrumented':
            with recorder.span('handoff_residual_validation_commit'):
                receipt = run()
        else:
            receipt = run()
        sync()
        elapsed = (perf_counter()-started)*1000
    # Checks/export are outside measured interval. Receipt identity is mandatory.
    assert receipt.received_token_ids == saved['A'].draft_message.message_token_ids
    assert receipt.wire_byte_count == session.transport.records[-1].byte_count
    for name in ('A','B'):
        session.state(name).validate()
        assert session.state(name) is not saved[name]
    if candidate!='token':
        assert receipt.message.payload_mode=='KV'
        if candidate in ('kv:v0','kv:v1'):
            assert receipt.receiver_message_prefill_count==0
    return dict(local_ms=elapsed, wire_bytes=receipt.wire_byte_count,receipt=asdict(receipt),
                stages=recorder.aggregate(),stage_events=recorder.events,
                adapter_submetrics=dict(recorder.adapter_metrics),mode=mode)


def research_profile(summary, identity, *, origin, n, cache_count):
    """Point-domain real Decode profile; do not fit sparse length/cache data."""
    from .incremental_auto import IncrementalCostProfile, StageCost
    if origin not in ('real_decode','real_decode_prefix'):
        raise ValueError('fixed-input calibration cannot masquerade as real Decode')
    return IncrementalCostProfile('83a-point-domain-v1','measured_incremental',identity,
        {'token_handoff':StageCost(summary['token']['median']),
         'kv_base':StageCost(summary['kv:v0']['median'])},
        summary['token']['sample_count']+summary['kv:v0']['sample_count'],
        (n,n),(cache_count,cache_count),
        f'{origin}; synchronized full committed handoff; steady samples; exact point domain; '
        'only TOKEN/V0 calibrated; no startup estimate; no extrapolation; source samples linked in experiment evidence.',
        recompute_token_range=(0,0))


def offline_analysis(groups, settings):
    """Reproduce network/selector predictions using metadata only, no model load."""
    from types import SimpleNamespace
    from .incremental_auto import IncrementalSelector, IncrementalAutoConfig
    from .network.handoff_network import HandoffNetwork, HandoffNetworkConfig
    identity=settings['model_identity']
    contract=settings['contract']
    backend=SimpleNamespace(_incremental_cost_identity=identity['sender'],
        model_name=contract['model_id'],vocab_size=contract['vocab_size'],metadata=SimpleNamespace(dtype=settings['metadata']['dtype']),
        kv_model_config=SimpleNamespace(**{k:contract[k] for k in ('num_hidden_layers','num_key_value_heads','head_dim')}),
        eos_token_ids=tuple(contract['eos_token_ids']),
        chat_tokenizer=SimpleNamespace(all_special_ids=contract['special_token_ids']))
    receiver_backend=SimpleNamespace(**vars(backend))
    receiver_backend._incremental_cost_identity=identity['receiver']
    networks=[];profiles=[];decisions=[]
    for row in groups:
        c=row['config'];summary=row['summary']['steady']
        if 'token' not in summary:
            continue
        token=summary['token'];kv=summary['kv:v0']
        analysis=bandwidth_analysis(token['median'],kv['median'],token['wire_bytes'],kv['wire_bytes'])
        analysis['crossing_range_from_observed_extremes']=[
            bandwidth_analysis(token[tk],kv[vk],token['wire_bytes'],kv['wire_bytes'])['break_even_mbps']
            for tk,vk in [('min','max'),('max','min')]]
        networks.append(dict(label=c['label'],**analysis))
        p=research_profile(summary,identity,origin=c['origin'],n=c['message_tokens'],cache_count=c['total_cache_tokens'])
        p=replace(p,notes=p.notes+f" Frozen Sender/Receiver caches: {c['sender_cache_tokens']}/{c['receiver_cache_tokens']}; "
            'schema guards sum only, exact per-endpoint conditions must be checked by caller; research/offline use.')
        profiles.append(dict(label=c['label'],profile=asdict(p),source=c['label']+'-samples.jsonl'))
        msg=SimpleNamespace(token_count=len(c['message_ids']),message_token_ids=tuple(c['message_ids']),
            termination_reason=c['termination'],sender_message_span=tuple(c['sender_span']),message_id='A:1',sequence=1,
            source_agent='A',target_agent='B')
        sender=SimpleNamespace(backend=backend,cached_token_count=c['sender_cache_tokens'])
        receiver=SimpleNamespace(backend=receiver_backend,cached_token_count=c['receiver_cache_tokens'])
        selector=IncrementalSelector(IncrementalAutoConfig('performance_experiment',('token','kv:v0'),profile=p))
        for bandwidth in (1,20,100,1000,10000):
            net=HandoffNetwork(HandoffNetworkConfig(bandwidth_mbps=bandwidth,base_latency_ms=2)).snapshot(1)
            _,decision=selector.select(sender,receiver,msg,net,kv_contract_ready=True)
            assert {e['candidate']:e['predicted_wire_bytes'] for e in decision['candidate_estimates']}=={
                'token':token['wire_bytes'],'kv:v0':kv['wire_bytes']}
            decisions.append(dict(label=c['label'],decision=decision,nature='offline_prediction'))
    return networks,profiles,decisions
