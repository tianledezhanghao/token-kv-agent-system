"""Bounded offline 7.9C diagnostics; no production strategy or oracle selector."""
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from .agent.state import working_state, close_working_draft, append_working_ids
from .communication import KVStrategy
from .discussion import DiscussionConfig, DiscussionRuntime
from .incremental_kv import prepare_incremental_message
from .incremental_kv_protocol import inspect_model_contract, encode_incremental_message
from .kv_v2_validation import PacketEvidence
from .local_qwen_comparison import REVISIONS, snapshot_path, file_hash, memory
from .local_qwen_resources import ResourceConfig, ResourceMonitor, classify_failure
from .message_diagnostics import ForwardTrace, cache_fingerprint, logits_difference, cache_shapes
from .message_understanding_probe import initialize_question, RECALL_QUESTION, RECALL_STAGE, SENDER_STAGE
from .model.decode_kv import cache_layers
from .model.delta_repair import reconstruct
from .model.token_slot_repair import first_numeric
from .secret_sum import SecretSumConfig, SecretSumRuntime, STAGES

VARIANTS = ('token', 'v0', 'v1', 'v2-k1', 'full-message', 'sparse-first-numeric', 'middle-layer-oracle')


def span_metrics(reference, actual):
    """One aligned message layer at a time on CPU; K and V remain separate."""
    import torch
    if len(reference) != len(actual) or not reference:
        raise ValueError('layer count mismatch')
    rows = []
    for layer, (ref_pair, pair) in enumerate(zip(reference, actual)):
        for kind, ref, value in zip(('K', 'V'), ref_pair, pair):
            if ref.shape != value.shape or ref.ndim != 4 or ref.shape[0] != 1:
                raise ValueError('unaligned message shape')
            r, a = ref.detach().float().cpu(), value.detach().float().cpu()
            if not torch.isfinite(r).all() or not torch.isfinite(a).all():
                raise ValueError('nonfinite diagnostic tensor')
            # sequence-major, flatten heads/dim, same ID and Receiver absolute position
            r = r[0].permute(1, 0, 2).flatten(1)
            a = a[0].permute(1, 0, 2).flatten(1)
            d = a - r
            for token in range(r.shape[0]):
                rn, an = r[token].norm(), a[token].norm()
                rows.append(dict(layer=layer, token=token, kind=kind,
                    mean_abs=float(d[token].abs().mean()), max_abs=float(d[token].abs().max()),
                    relative_l2=float(d[token].norm()/rn.clamp_min(1e-12)),
                    cosine=float((r[token]*a[token]).sum()/(rn*an).clamp_min(1e-12))))
    return rows


def message_span(cache, start, n, cpu=False):
    return tuple(tuple(t[..., start:start+n, :].detach().cpu().clone() if cpu else
                       t[..., start:start+n, :] for t in pair) for pair in cache_layers(cache))


def sparse_slot(state, start, ids, index):
    """Offline executable candidate: prefix crop, real all-layer single-token forward,
    scatter its new slot, retain all other slots. No oracle or extra hidden states.
    Future message slots intentionally remain stale approximations.
    """
    from .model.token_slot_repair import repair_slot
    work = replace(state, cache=repair_slot(state.backend,state.cache,start,ids,index))
    work.validate()
    return work


def oracle_layer_intervention(state, reference, start, n, layers=range(8,16)):
    """Counterfactual upper-bound probe; explicitly pays for full-prefill oracle."""
    pairs = []
    for layer, original in enumerate(cache_layers(state.cache)):
        changed = []
        for old, ref in zip(original, reference[layer]):
            t = old.clone()
            if layer in layers:
                t[..., start:start+n, :] = ref.to(device=t.device, dtype=t.dtype)
            changed.append(t)
        pairs.append(tuple(changed))
    work = replace(state, cache=reconstruct(tuple(pairs), state.backend._model.config))
    work.validate()
    return work


def controlled_case(backend, *, secret_b=None, emit=lambda name,row:None):
    config = SecretSumConfig(secret_b=secret_b) if secret_b is not None else DiscussionConfig(
        question=RECALL_QUESTION, rounds=1, max_new_tokens=96)
    runtime = SecretSumRuntime(backend, config=config) if secret_b is not None else DiscussionRuntime(backend, config=config)
    initialize_question(runtime)
    stage_a = STAGES.a_first if secret_b is not None else SENDER_STAGE
    stage_b = STAGES.b_first if secret_b is not None else RECALL_STAGE
    if secret_b is None:
        runtime.session.append_local_user('A','仅 Agent A 可见的私有信息：测试暗号是：紫色袋鼠5729')
    s = runtime.session
    with ForwardTrace(backend) as trace:
        message = s.generate_message('A','B',max_new_tokens=96,local_prompt=stage_a)
    snapshots = {n:s.state(n) for n in ('A','B')}
    hashes = {n:cache_fingerprint(state.cache) for n,state in snapshots.items()}
    contract = inspect_model_contract(backend); s._kv_contract = contract
    envelope, export = prepare_incremental_message(snapshots['A'], message, contract=contract)
    packet = encode_incremental_message(envelope)
    ids = envelope.message_token_ids; n = len(ids)
    index = first_numeric(ids, lambda x:backend.decode(x,skip_special_tokens=False))
    # Explicit independent full Prefill, same Receiver history/framing/IDs. Offline only.
    frame = snapshots['B'].framing.incoming('A','B')
    prefix = close_working_draft(working_state(snapshots['B']))
    prefix = append_working_ids(prefix, frame.prefix_ids)
    start = prefix.cached_token_count
    oracle_ids = prefix.token_ledger + ids
    began = perf_counter()
    with ForwardTrace(backend) as oracle_trace:
        oracle = backend.prefill(input_token_ids=oracle_ids)
    oracle_ms = (perf_counter()-began)*1000
    reference = message_span(oracle.cache,start,n,cpu=True)
    oracle_state = replace(prefix, cache=oracle.cache, cached_token_count=start+n, token_ledger=oracle_ids)
    oracle_state = append_working_ids(oracle_state, frame.closing_ids)
    source = dict(message=asdict(message), sender_text=backend.decode(ids,skip_special_tokens=True),
        content_ids=ids, token_pieces=[backend.decode((t,),skip_special_tokens=False) for t in ids],
        source_span=[envelope.sender_span_start,envelope.sender_span_start+n],target_span=[start,start+n],
        sender_trace=trace.calls,export=asdict(export),packet_sha256=hashlib.sha256(packet).hexdigest(),
        kv_wire_bytes=len(packet),snapshot_hashes=hashes,online_selected_index=index,
        oracle=dict(input_ids=oracle_ids,forward_calls=oracle_trace.calls,prefill_ms=oracle_ms,
                    input_tokens=len(oracle_ids),layer_token_work=len(oracle_ids)*24),
        storage_estimate=dict(message_reference_bytes=n*24*2*2*64*2,
            full_cache_bytes_per_256_tokens=256*24*2*2*64*2,
            policy='one CPU BF16 message oracle + one branch; no persistent tensor files'))
    emit('source',source)
    replies = {}; logits_by_variant = {}; rows = []
    for variant in VARIANTS + ('independent-full-oracle',):
        s._states = snapshots.copy(); s.transport = PacketEvidence()
        began = perf_counter()
        with ForwardTrace(backend) as ht:
            if variant == 'independent-full-oracle':
                # Computation-only, no handoff claim; cache clone protects oracle.
                state = working_state(oracle_state); receipt = None
            else:
                if variant == 'token': receipt = s.handoff('A','B')
                else:
                    strategy = {'v0':KVStrategy.RAW_DELTA,'v2-k1':KVStrategy.SUFFIX_RECOMPUTE_DELTA,
                                'full-message':KVStrategy.SUFFIX_RECOMPUTE_DELTA}.get(variant,KVStrategy.ROPE_ALIGNED_DELTA)
                    receipt = s.handoff_kv('A','B',strategy=strategy,
                        recompute_tokens=(n if variant=='full-message' else 1) if strategy==KVStrategy.SUFFIX_RECOMPUTE_DELTA else None,
                        measure_timing=True)
                    assert s.transport.packets[-1]['sha256']==source['packet_sha256']
                state = s.state('B')
                if variant == 'sparse-first-numeric':
                    if index is None: raise ValueError('frozen candidate has no numeric token')
                    state = sparse_slot(state,start,ids,index)
                elif variant == 'middle-layer-oracle':
                    state = oracle_layer_intervention(state,reference,start,n)
                if variant in ('sparse-first-numeric','middle-layer-oracle'):
                    # Closing was previously computed with V1; rerun it after repair.
                    end = start+n
                    crop = reconstruct(tuple((k[...,:end,:],v[...,:end,:]) for k,v in cache_layers(state.cache)),backend._model.config)
                    state = replace(state,cache=crop,cached_token_count=end,token_ledger=state.token_ledger[:end])
                    state = append_working_ids(state,frame.closing_ids)
            state.validate()
        ms = (perf_counter()-began)*1000
        metrics = span_metrics(reference,message_span(state.cache,start,n))
        # Same exact Receiver ledger across every branch, including independent oracle.
        if rows: assert state.token_ledger == rows[0]['receiver_ledger']
        s._states['B'] = state
        with ForwardTrace(backend,state.framing.assistant.prefix_ids) as gt:
            reply = s.generate_message('B','A',max_new_tokens=96,local_prompt=stage_b)
        logits = gt.prefix_result[0].detach().float().cpu(); gt.prefix_result=None
        logits_by_variant[variant] = logits; replies[variant] = reply.message_token_ids
        row = dict(variant=variant,receipt=asdict(receipt) if receipt else None,
            receiver_ledger=state.token_ledger,shapes=cache_shapes(state.cache),metrics=metrics,
            handoff_and_intervention_ms=ms,handoff_forward_calls=ht.calls,decode_forward_calls=gt.calls,
            reply=asdict(reply),reply_text=backend.decode(reply.message_token_ids,skip_special_tokens=True),
            packets=s.transport.packets,oracle_used=variant in ('middle-layer-oracle','independent-full-oracle'),
            extra_repair_tokens=int(variant=='sparse-first-numeric'),
            extra_repair_layer_token_work=24 if variant=='sparse-first-numeric' else 0,
            oracle_cost_excluded_from_branch_ms=variant in ('middle-layer-oracle','independent-full-oracle'),
            selected_tokens=[index] if variant=='sparse-first-numeric' else [],
            selected_layers=list(range(8,16)) if variant=='middle-layer-oracle' else [],fallback=None)
        for name,state0 in snapshots.items(): assert cache_fingerprint(state0.cache)==hashes[name]
        row['snapshots_unchanged']=True
        rows.append(row); emit(variant,row)
        print(variant,repr(row['reply_text']),flush=True)
        del state; gc.collect()
    for row in rows:
        v=row['variant']; ref=replies['independent-full-oracle']; out=replies[v]
        common=next((i for i,(a,b) in enumerate(zip(ref,out)) if a!=b),min(len(ref),len(out)))
        row['first_output_divergence_vs_oracle']=None if ref==out else common
        row['first_logits_vs_oracle']=logits_difference(logits_by_variant['independent-full-oracle'],logits_by_variant[v])
        row['first_logits_vs_token']=logits_difference(logits_by_variant['token'],logits_by_variant[v])
        emit(v,row)
    return dict(source=source,branches=rows)


def production_case(backend, *, secret_b=None, emit=lambda name,row:None):
    """Frozen same inputs; formal V3 A→B and, for secrets, B→A + A final."""
    config = SecretSumConfig(secret_b=secret_b) if secret_b is not None else DiscussionConfig(
        question=RECALL_QUESTION,rounds=1,max_new_tokens=96)
    runtime = SecretSumRuntime(backend,config=config) if secret_b is not None else DiscussionRuntime(backend,config=config)
    initialize_question(runtime)
    if secret_b is None:
        runtime.session.append_local_user('A','仅 Agent A 可见的私有信息：测试暗号是：紫色袋鼠5729')
    s=runtime.session;s.transport=PacketEvidence();s._kv_contract=inspect_model_contract(backend)
    with ForwardTrace(backend) as st:
        sender=s.generate_message('A','B',max_new_tokens=96,local_prompt=STAGES.a_first if secret_b else SENDER_STAGE)
    original={name:s.state(name) for name in ('A','B')}
    hashes={name:cache_fingerprint(state.cache) for name,state in original.items()}
    with ForwardTrace(backend) as ht:
        receipt=s.handoff_kv('A','B',strategy=KVStrategy.NUMERIC_SLOT_DELTA,measure_timing=True)
    before=s.state('B')
    with ForwardTrace(backend,before.framing.assistant.prefix_ids) as gt:
        reply=s.generate_message('B','A',max_new_tokens=96,local_prompt=STAGES.b_first if secret_b else RECALL_STAGE)
    first_logits=gt.prefix_result[0].detach().float().cpu();gt.prefix_result=None
    top=first_logits.topk(5)
    row=dict(sender=asdict(sender),sender_text=backend.decode(sender.message_token_ids,skip_special_tokens=True),
        receipt=asdict(receipt),shapes=cache_shapes(before.cache),sender_trace=st.calls,handoff_trace=ht.calls,
        reply_trace=gt.calls,reply=asdict(reply),reply_text=backend.decode(reply.message_token_ids,skip_special_tokens=True),
        first_logits_top5=dict(ids=top.indices.tolist(),values=top.values.tolist()),snapshot_hashes=hashes,
        oracle_used=False,fallback=None)
    for name,state in original.items():assert cache_fingerprint(state.cache)==hashes[name]
    row['snapshots_unchanged']=True
    if secret_b is not None:
        with ForwardTrace(backend) as rt:
            reverse=s.handoff_kv('B','A',strategy=KVStrategy.NUMERIC_SLOT_DELTA,measure_timing=True)
        with ForwardTrace(backend) as ft:
            final=s.generate_local_message('A',max_new_tokens=96,local_prompt=STAGES.final)
        row.update(reverse_receipt=asdict(reverse),reverse_trace=rt.calls,final_trace=ft.calls,
            final=asdict(final),final_text=backend.decode(final.message_token_ids,skip_special_tokens=True))
    for name in ('A','B'):s.state(name).validate()
    from .model.state_cache import require_disjoint_caches
    require_disjoint_caches(s.state('A').cache,s.state('B').cache)
    row.update(packets=s.transport.packets,engineering_status='verified')
    emit('production',row);print('formal V3',repr(row['reply_text']),repr(row.get('final_text')),flush=True)
    return row


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('benchmark-results/kv-v3-79c'))
    parser.add_argument('--phase',choices=('diagnostic','production'),default='diagnostic')
    parser.add_argument('--case',choices=('all','secret-17-26','remaining'),default='all')
    args=parser.parse_args(argv); args.output_dir.mkdir(parents=True,exist_ok=False)
    def emit(name,row):
        (args.output_dir/(name+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
    spec=dict(task='7.9C',model=REVISIONS['05b'],device='mps',dtype='bfloat16',greedy=True,batch=1,max_new_tokens=96,
        variants=VARIANTS,cases=['secret-17-26','secret-17-31','long-message'],
        candidates=['first ASCII numeric token, all-layer slot repair, later slots retained',
                    'layers [8,16) full-message oracle intervention; NOT online algorithm'],
        prompts='exact existing kv_v2_validation controlled_case prompts; no search',
        resource_config=asdict(ResourceConfig()),
        source_sha256={str(p):file_hash(p) for p in sorted(Path('src').rglob('*.py'))},
        no_tensor_snapshots='Historical raw IDs/config reused; all tensors and generations are new runs',
        estimated_cache_memory='256 tokens ~3 MiB/cache, message oracle <=84 KiB; single 0.5B model')
    cases=[('secret-17-26',26),('secret-17-31',31),('long-message',None)]
    if args.case=='secret-17-26':cases=cases[:1]
    elif args.case=='remaining':cases=cases[1:]
    spec['cases']=[name for name,_ in cases]
    spec['phase']=args.phase
    emit('frozen-spec',spec)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OTEL_SDK_DISABLED='true')
    import torch, transformers
    from .model import HuggingFaceModelBackend
    baseline=memory(torch);emit('preload',baseline)
    monitor=ResourceMonitor(lambda:memory(torch),baseline,args.output_dir)
    backend=None;original=None
    try:
        monitor.start();monitor.check(); path=snapshot_path('05b')
        if not all((path/p).is_file() for p in ('config.json','tokenizer.json','tokenizer_config.json','model.safetensors')):
            raise FileNotFoundError('incomplete frozen checkpoint')
        torch.manual_seed(77)
        tokenizer=transformers.AutoTokenizer.from_pretrained(path,local_files_only=True)
        model=transformers.AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,local_files_only=True)
        backend=HuggingFaceModelBackend(str(path),device='mps',tokenizer=tokenizer,model=model)
        del tokenizer,model
        assert backend.device=='mps' and backend.metadata.dtype=='torch.bfloat16'
        original=backend._timed_forward;backend._timed_forward=lambda **kw:monitor.forward(original,**kw)
        emit('settings',dict(metadata=asdict(backend.metadata),torch=torch.__version__,transformers=transformers.__version__,
            contract=asdict(inspect_model_contract(backend))))
        for name,b in cases:
            monitor.check(); row=(controlled_case if args.phase=='diagnostic' else production_case)(backend,secret_b=b,emit=lambda key,row:emit(name+'-'+key,row))
            emit(name,row);del row;gc.collect();torch.mps.empty_cache()
        emit('completion',dict(status='verified',cases=len(cases),
            real_generations=sum(9 if args.phase=='diagnostic' else 3 if b is not None else 2 for _,b in cases),
            bytes_handoffs=sum(7 if args.phase=='diagnostic' else 2 if b is not None else 1 for _,b in cases),
            forward_count=len(monitor.forwards),resource_stop=monitor.stop_reason,memory=monitor.sample()))
        return 0
    except Exception as exc:
        emit('failure',dict(status='failed',error=f'{type(exc).__name__}: {exc}',category=classify_failure(exc),
            forward_count=len(monitor.forwards),resource_stop=monitor.stop_reason))
        print(type(exc).__name__,str(exc),flush=True);return 1
    finally:
        if original is not None:backend._timed_forward=original
        monitor.close()
