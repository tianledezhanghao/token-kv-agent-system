"""8.3B bounded paired MPS replay of actual 8.3A frozen Decode caches.

Research-only exclusive process: original Session and 8.3A timing boundaries.
No generation, download, alternate inference device or serving-process patches.
"""
import argparse
from dataclasses import asdict, replace
from functools import partial
import gc
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
from time import monotonic
from unittest.mock import patch

from .performance_characterization import (measured_handoff, cache_sha256, real_prefix,
                                           statistics, bandwidth_analysis, research_profile)
from .performance_validation import dump, summarize_group
from .local_qwen_comparison import REVISIONS, snapshot_path, file_hash, memory
from .local_qwen_resources import ResourceMonitor, ResourceConfig, classify_failure


def signed_statistics(values):
    values=list(values)
    if not values:
        raise ValueError('nonempty paired differences required')
    import math
    if not all(math.isfinite(x) for x in values):
        raise ValueError('finite differences required')
    center=median(values)
    return dict(sample_count=len(values),median=center,min=min(values),max=max(values),
                mad=median(abs(x-center) for x in values),values=values)


def verify_index(directory):
    rows=json.loads((directory/'sha256-index.json').read_text())
    for row in rows:
        p=directory/row['path']
        if not p.resolve().is_relative_to(directory.resolve()):
            raise ValueError('unsafe input index path')
        if file_hash(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:
            raise ValueError(f'input SHA mismatch: {p}')
    return rows


def frozen_state(directory,label,template):
    """Restore actual preexisting tensor bytes, never re-Prefill message IDs."""
    from .agent.state import AgentMessage
    from .kv_protocol import deserialize_kv_packet
    from .model.kv_adapter import restore_huggingface_kv_cache
    row=json.loads((directory/(label+'.json')).read_text())
    p=directory/row['cache_file']
    if file_hash(p)!=row['cache_file_sha256']:
        raise ValueError('frozen Cache file SHA mismatch')
    b=template.backend
    payload=deserialize_kv_packet(p.read_bytes()).payload
    result=restore_huggingface_kv_cache(payload,model_id=b.model_name,model_config=b.kv_model_config,
        model_dtype=b.metadata.dtype,target_device='mps',transformers_config=b._model.config,
        torch_module=b._torch,transfer_path='legacy')
    draft=row['draft_message']
    if draft:
        draft=dict(draft)
        for key in ('message_token_ids','sender_message_span'):draft[key]=tuple(draft[key])
        draft=AgentMessage(**draft)
    if row['system_prompt']!=template.system_prompt or row['agent_id']!=template.agent_id:
        raise ValueError('frozen role identity differs')
    state=replace(template,cache=result.cache,token_ledger=tuple(row['token_ledger']),
        cached_token_count=row['cached_token_count'],pending_token_id=row['pending_token_id'],
        assistant_open=row['assistant_open'],draft_message=draft)
    state.validate()
    if cache_sha256(state.cache)!=row['tensor_sha256']:
        raise ValueError('frozen actual Tensor bits differ')
    return state


def group(session,saved,label,options,output,monitor,forward_rows,*,warmup=2,runs=5,modes=('steady','instrumented')):
    from .model import kv_adapter as adapter
    from .model.state_cache import require_disjoint_caches
    from .incremental_auto import estimate_incremental_wire_bytes
    b=saved['A'].backend
    before={n:cache_sha256(s.cache) for n,s in saved.items()}
    m=saved['A'].draft_message
    n=m.token_count-int(m.termination_reason=='eos')
    c=dict(label=label,origin='real_decode' if m.termination_reason=='eos' else 'real_decode_prefix',
        message_tokens=n,message_ids=m.message_token_ids,termination=m.termination_reason,
        sender_span=m.sender_message_span,sender_cache_tokens=saved['A'].cached_token_count,
        receiver_cache_tokens=saved['B'].cached_token_count,
        total_cache_tokens=sum(s.cached_token_count for s in saved.values()),snapshot_sha256=before,
        framing=asdict(saved['B'].framing.incoming('A','B')),options=options,warmup=warmup,runs=runs,
        snapshot_reset='same immutable 8.3A actual Tensor/ledger; original per-handoff private work clones',
        ledger_sha256={a:hashlib.sha256(json.dumps(s.token_ledger).encode()).hexdigest() for a,s in saved.items()},
        input_layout=[dict(agent=a,layer=i,kind=k,shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),
                           device=str(t.device),contiguous=t.is_contiguous()) for a,s in saved.items()
                      for i,l in enumerate(s.cache.layers) for k,t in [('K',l.keys),('V',l.values)]])
    dump(output/(label+'-config.json'),c)
    samples=[];oracles={};packet_sha=None
    original=session.transport.transfer
    packets=[]
    def transfer(data,**kw):
        packets.append(data) # SHA deliberately computed AFTER the timed handoff.
        return original(data,**kw)
    with patch.object(session.transport,'transfer',transfer):
        for mode in modes:
            for i in range(warmup+runs):
                for option in (options if i%2==0 else list(reversed(options))):
                    monitor.check();start=len(forward_rows);packets.clear()
                    with patch.object(adapter,'DEFAULT_TRANSFER_PATH',option['path']):
                        sample=measured_handoff(session,saved,option['candidate'],mode=mode,
                            recompute_tokens=option.get('recompute_tokens'))
                    assert len(packets)==1
                    sha=hashlib.sha256(packets[0]).hexdigest()
                    if option['candidate']!='token':
                        if packet_sha is None:packet_sha=sha
                        assert sha==packet_sha,'KV protocol bytes changed across paths/versions'
                        assert sample['wire_bytes']==estimate_incremental_wire_bytes(saved['A'],m)
                    hashes={a:cache_sha256(session.state(a).cache) for a in ('A','B')}
                    semantic=(hashes,session.state('A').token_ledger,session.state('B').token_ledger)
                    candidate=option['candidate']
                    if candidate in oracles:assert semantic==oracles[candidate],'committed Cache bits/ledger differ'
                    else:oracles[candidate]=semantic
                    require_disjoint_caches(session.state('A').cache,session.state('B').cache)
                    calls=forward_rows[start:]
                    assert all(x['device'].startswith('mps') for x in calls)
                    if candidate in ('kv:v0','kv:v1'):
                        target=saved['B'].cached_token_count+len(c['framing']['prefix_ids'])
                        assert not any(x['cache_tokens']==target for x in calls)
                        assert any(x['cache_tokens']==target+n for x in calls)
                    sample.update(candidate=candidate,candidate_label=option['label'],path=option['path'],
                        index=i,warmup=i<warmup,packet_sha256=sha,committed_tensor_sha256=hashes,
                        forward_calls=calls,base_snapshot_unchanged=True,
                        mps_allocated_after_bytes=b._torch.mps.current_allocated_memory(),
                        mps_driver_after_bytes=b._torch.mps.driver_allocated_memory())
                    samples.append(sample)
                    with (output/(label+'-samples.jsonl')).open('a') as f:
                        f.write(json.dumps(sample,ensure_ascii=False,allow_nan=False)+'\n')
            print(label,mode,'complete',flush=True)
    assert before=={a:cache_sha256(s.cache) for a,s in saved.items()}
    # summarize_group expects both modes; pilot and smoke retain raw only.
    summary=summarize_group(samples) if len(modes)==2 else {}
    row=dict(config=c,summary=summary,successful_handoffs=len(samples),failed_handoffs=0,
             snapshot_unchanged=True,packet_sha256=packet_sha,resource=monitor.sample())
    dump(output/(label+'-summary.json'),row)
    session._states=dict(saved);packets.clear()
    if summary:print(label,{k:round(v['median'],3) for k,v in summary['steady'].items()},flush=True)
    return row,samples


def correctness_probe(session,saved,output,monitor,forward_rows):
    """Whole real MPS V0–V3, real stride, payload ownership and continued Decode."""
    from .model import kv_adapter as adapter
    from .incremental_kv import prepare_incremental_message
    from .model.state_cache import require_disjoint_caches
    b=saved['A'].backend;torch=b._torch
    results=[]
    before={n:cache_sha256(s.cache) for n,s in saved.items()}
    for candidate in ('kv:v0','kv:v1','kv:v2','kv:v3'):
        outputs=[]
        for path in ('legacy','packed'):
            monitor.check()
            with patch.object(adapter,'DEFAULT_TRANSFER_PATH',path):
                sample=measured_handoff(session,saved,candidate,recompute_tokens=1 if candidate=='kv:v2' else None)
            received={n:cache_sha256(session.state(n).cache) for n in ('A','B')}
            # Continue actual model generation beyond handoff, same cached input.
            message=session.generate_message('B','A',max_new_tokens=3)
            outputs.append(dict(path=path,wire_bytes=sample['wire_bytes'],committed=received,
                generated=asdict(message),after_decode_sha256=cache_sha256(session.state('B').cache)))
            require_disjoint_caches(session.state('A').cache,session.state('B').cache)
        assert {k:v for k,v in outputs[0].items() if k!='path'}=={k:v for k,v in outputs[1].items() if k!='path'}
        results.append(dict(candidate=candidate,paths=outputs,exact=True))
    assert before=={n:cache_sha256(s.cache) for n,s in saved.items()}
    session._states=dict(saved)
    # Span already materialized once outside any benchmark, immutable neutral bytes.
    envelope,_=prepare_incremental_message(saved['A'],saved['A'].draft_message,contract=session._kv_contract)
    kw=dict(model_id=b.model_name,model_config=b.kv_model_config,model_dtype=b.metadata.dtype,
        target_device='mps',transformers_config=b._model.config,torch_module=torch,transfer_path='packed')
    first=adapter.restore_huggingface_kv_cache(envelope.delta,**kw).cache
    second=adapter.restore_huggingface_kv_cache(envelope.delta,**kw).cache
    require_disjoint_caches(first,second);require_disjoint_caches(first,saved['A'].cache)
    original=cache_sha256(second)
    del envelope;gc.collect();first.layers[0].keys.zero_();torch.mps.synchronize()
    assert cache_sha256(second)==original
    assert before=={n:cache_sha256(s.cache) for n,s in saved.items()}
    # Genuine MPS stride change, no random replacement of actual KV.
    from types import SimpleNamespace
    original_cache=second
    strided=SimpleNamespace(get_seq_length=second.get_seq_length,layers=[])
    for l in second.layers:
        strided.layers.append(SimpleNamespace(**{name:torch.stack((t,t),-1)[...,0]
            for name,t in [('keys',l.keys),('values',l.values)]}))
    from .kv_protocol import serialize_kv_payload
    packets=[]
    for path in ('legacy','packed'):
        p,_,_=adapter.extract_huggingface_kv_payload(strided,logical_cached_token_count=second.get_seq_length(),
            model_id=b.model_name,model_config=b.kv_model_config,torch_module=torch,transfer_path=path)
        packets.append(serialize_kv_payload(p).packet)
    assert packets[0]==packets[1]
    dump(output/'correctness.json',dict(versions=results,base_snapshot_unchanged=True,
        independent_restores=True,buffer_lifetime=True,mps_strided_packet_exact=True,
        actual_strides=list(strided.layers[0].keys.stride()),source='8.3A real Decode Tensor bytes',
        no_cpu_forward=True,no_token_fallback=True))
    del original_cache,first,second,strided,packets;gc.collect()



def buffer_probe(session,saved,label,output):
    """Untimed transfer API counts, exact real bits and live allocation boundaries.

    These are observed allocation boundaries, not a claim of allocator highwater.
    """
    from .model import kv_adapter as adapter
    from .incremental_kv import prepare_incremental_message
    from .kv_protocol import serialize_kv_payload
    from types import SimpleNamespace
    b=saved['A'].backend;torch=b._torch
    envelope,_=prepare_incremental_message(saved['A'],saved['A'].draft_message,contract=session._kv_contract)
    restored=adapter.restore_huggingface_kv_cache(envelope.delta,model_id=b.model_name,
        model_config=b.kv_model_config,model_dtype=b.metadata.dtype,target_device='mps',
        transformers_config=b._model.config,torch_module=torch,transfer_path='legacy').cache
    expected=hashlib.sha256(b''.join(t.data for l in envelope.delta.layers for t in (l.key,l.value))).hexdigest()
    assert cache_sha256(restored)==expected
    rows=[]
    for path in ('legacy','packed'):
        gc.collect();torch.mps.synchronize()
        events=[dict(event='before',**memory(torch))];counts=dict(stack=0,tensor_to_neutral=0,neutral_to_cpu=0)
        stack=torch.stack;to_neutral=adapter._tensor_to_neutral;to_cpu=adapter._neutral_to_cpu_tensor
        def observe_stack(tensors,*a,**kw):
            result=stack(tensors,*a,**kw);counts['stack']+=1
            events.append(dict(event='packed_stack_live',**memory(torch)))
            return result
        def observe_export(*a,**kw):
            counts['tensor_to_neutral']+=1
            return to_neutral(*a,**kw)
        def observe_cpu(*a,**kw):
            counts['neutral_to_cpu']+=1
            return to_cpu(*a,**kw)
        with patch.object(torch,'stack',observe_stack),patch.object(adapter,'_tensor_to_neutral',observe_export):
            payload,raw_bytes,_=adapter.extract_huggingface_kv_payload(restored,
                logical_cached_token_count=restored.get_seq_length(),model_id=b.model_name,
                model_config=b.kv_model_config,torch_module=torch,transfer_path=path)
        assert payload.layers==envelope.delta.layers
        events.append(dict(event='host_payload_returned',**memory(torch)))
        def cache_class(**kw):
            pairs=kw['ddp_cache_data']
            tensors=[t for pair in pairs for t in pair]
            shared=len({t.untyped_storage().data_ptr() for t in tensors})
            assert shared==(1 if path=='packed' else 48)
            events.append(dict(event='upload_buffer_live_before_cache',input_storages=shared,**memory(torch)))
            result=__import__('transformers').DynamicCache(**kw)
            events.append(dict(event='cache_plus_input_buffers_live',**memory(torch)))
            return result
        with patch.object(adapter,'_neutral_to_cpu_tensor',observe_cpu):
            result=adapter.restore_huggingface_kv_cache(payload,model_id=b.model_name,model_config=b.kv_model_config,
                model_dtype=b.metadata.dtype,target_device='mps',transformers_config=b._model.config,
                torch_module=torch,dynamic_cache_class=cache_class,transfer_path=path)
        assert cache_sha256(result.cache)==expected
        assert counts==dict(stack=1 if path=='packed' else 0,tensor_to_neutral=0 if path=='packed' else 48,
                            neutral_to_cpu=0 if path=='packed' else 48)
        del result,payload;gc.collect();torch.mps.synchronize()
        events.append(dict(event='released',**memory(torch)))
        rows.append(dict(path=path,raw_bytes=raw_bytes,counts=counts,events=events,
                         bitwise_exact=True,device_crossings_from_executed_branch=1 if path=='packed' else 48))
    dump(output/(label+'-buffer-probe.json'),rows)

def analyze(groups,settings):
    rows=[];networks=[];profiles=[]
    for g in groups:
        c=g['config'];ss=g['summary'];samples=g['_samples']
        for mode in ('steady','instrumented'):
            for version in ('v0','v1'):
                old=f'legacy:{version}';new=f'packed:{version}'
                if old not in ss[mode]:continue
                paired=[]
                for i in range(c['warmup'],c['warmup']+c['runs']):
                    pair={s['candidate_label']:s for s in samples if s['index']==i and s['mode']==mode}
                    paired.append(pair[new]['local_ms']-pair[old]['local_ms'])
                rows.append(dict(label=c['label'],mode=mode,version=version,
                    optimized_minus_baseline_ms=signed_statistics(paired),
                    difference_of_medians_ms=ss[mode][new]['median']-ss[mode][old]['median']))
        token=ss['steady']['token'];kv=ss['steady']['packed:v0']
        network=bandwidth_analysis(token['median'],kv['median'],token['wire_bytes'],kv['wire_bytes'],
                                  bandwidths=(20,100,1000,10000))
        pair_deltas=[]
        for i in range(c['warmup'],c['warmup']+c['runs']):
            pair={s['candidate_label']:s for s in samples if s['index']==i and s['mode']=='steady'}
            pair_deltas.append(pair['packed:v0']['local_ms']-pair['token']['local_ms'])
        networks.append(dict(label=c['label'],**network,paired_local_kv_minus_token_ms=signed_statistics(pair_deltas)))
        profile=research_profile({'token':token,'kv:v0':kv},settings['model_identity'],origin=c['origin'],
                                 n=c['message_tokens'],cache_count=c['total_cache_tokens'])
        profile=replace(profile,version='83b-packed-point-domain-v1',notes=profile.notes+
            f" packed transfer v1; exact A/B caches {c['sender_cache_tokens']}/{c['receiver_cache_tokens']}; "
            f"termination {c['termination']}; source8.3A actual snapshot; research only, not installed in AUTO")
        profiles.append(dict(label=c['label'],profile=asdict(profile)))
    return rows,networks,profiles


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path('benchmark-results/performance-83a'))
    parser.add_argument('--output-dir',type=Path,default=Path('benchmark-results/kv-transfer-83b'))
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--runs',type=int,default=5)
    args=parser.parse_args(argv)
    if args.warmup<2 or args.runs<5:parser.error('at least 2 warmup and 5 valid samples required')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OTEL_SDK_DISABLED='true')
    inputs=verify_index(args.input_dir)
    source_paths=sorted(Path('src').rglob('*.py'))
    spec=dict(task='8.3B',model=REVISIONS['05b'],device='mps',dtype='bfloat16',batch=1,
        offline=True,real_decode='actual 8.3A Cache bytes replay, zero new Sender generations',
        input_dir=str(args.input_dir),input_index_sha256=file_hash(args.input_dir/'sha256-index.json'),
        input_index=inputs,warmup=args.warmup,runs=args.runs,resource_config=asdict(ResourceConfig()),
        boundary='8.3A measured_handoff unchanged: complete Sender draft to both states committed plus MPS sync',
        modes=['steady','instrumented'],source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):file_hash(p) for p in source_paths},
        source_settings_sha256=file_hash(args.input_dir/'settings.json'),
        protected_user_assets={str(p):file_hash(p) for p in Path('docs').rglob('*') if p.is_file() and (
            'dashboard-design-exploration' in str(p) or 'dashboard-redesign-preview' in p.name or '副本' in p.name)})
    dump(args.output_dir/'frozen-spec.json',spec)
    (args.output_dir/'sources').mkdir()
    for name in ('model/kv_adapter.py','kv_transfer_validation.py','performance_characterization.py'):
        p=Path('src/token_kv_agent_system')/name
        (args.output_dir/'sources'/p.name).write_bytes(p.read_bytes())
    import torch,transformers
    from .model import HuggingFaceModelBackend
    from .incremental_kv_protocol import inspect_model_contract
    from .incremental_auto import backend_cost_identity
    from .secret_sum import SecretSumRuntime
    from .discussion import DiscussionRuntime,DiscussionConfig
    monitor=ResourceMonitor(lambda:memory(torch),memory(torch),args.output_dir)
    forward_rows=[];groups=[]
    try:
        monitor.start();monitor.check()
        path=snapshot_path('05b');torch.manual_seed(77)
        # Fixed local snapshot; no network API and no auto device selection.
        tokenizer=transformers.AutoTokenizer.from_pretrained(path,local_files_only=True)
        model=transformers.AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,local_files_only=True)
        b=HuggingFaceModelBackend(str(path),device='mps',tokenizer=tokenizer,model=model)
        del tokenizer,model
        assert b.device=='mps' and b.metadata.dtype=='torch.bfloat16' and b.metadata.device_fallback_reason is None
        old_settings=json.loads((args.input_dir/'settings.json').read_text())
        contract=inspect_model_contract(b)
        identity=dict(sender=backend_cost_identity(b),receiver=backend_cost_identity(b))
        observed=dict(contract=asdict(contract),model_identity=identity)
        dump(args.output_dir/'observed-identity.json',observed)
        canonical=json.loads(json.dumps(observed))
        assert canonical['contract']==old_settings['contract'],'ModelContract differs from frozen 8.3A'
        assert canonical['model_identity']==old_settings['model_identity'],'backend cost identity differs from frozen 8.3A'
        settings=dict(metadata=asdict(b.metadata),contract=asdict(contract),model_identity=identity,
            torch=torch.__version__,transformers=transformers.__version__,revision=REVISIONS['05b'][1],
            hardware=dict(platform=__import__('platform').platform(),machine=__import__('platform').machine(),
                cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))),
            model_files={p:file_hash(path/p) for p in old_settings['model_files']},model_load_count=1)
        assert settings['model_files']==old_settings['model_files']
        dump(args.output_dir/'settings.json',settings)
        original=b._timed_forward
        def forward(**kw):
            monitor.check();monitor.active_forward_since=monotonic()
            try:
                out=original(**kw)
                forward_rows.append(dict(input_count=int(kw['input_ids'].shape[1]),
                    cache_tokens=int(kw['attention_mask'].shape[1]-kw['input_ids'].shape[1]),
                    forward_ms=out[1],device=str(kw['input_ids'].device)))
                monitor.check();return out
            finally:monitor.active_forward_since=None
        b._timed_forward=forward
        runtime=SecretSumRuntime(b);s=runtime.session;s._kv_contract=contract
        short={n:frozen_state(args.input_dir,'short-'+n,s.state(n)) for n in ('A','B')}
        options=[dict(label='token',candidate='token',path='legacy'),
                 dict(label='legacy:v0',candidate='kv:v0',path='legacy'),
                 dict(label='packed:v0',candidate='kv:v0',path='packed')]
        # Mandatory small real trial before any matrix. All samples retained.
        _,pilot=group(s,short,'pilot-eos',options,args.output_dir,monitor,forward_rows,
                      warmup=2,runs=1,modes=('instrumented',))
        host={p:median(sum(x['adapter_submetrics']['device_to_host_and_host_bytes_ms']) for x in pilot
                       if x['path']==p and x['candidate']=='kv:v0') for p in ('legacy','packed')}
        dump(args.output_dir/'pilot-gate.json',dict(correctness=True,host_median_ms=host,
            decision='continue_single_scheme',reason='packet/committed bits exact; inspect actual full cost in bounded matrix'))
        print('pilot gate passed',host,flush=True)
        def run(saved,label,opts=options):
            g,samples=group(s,saved,label,opts,args.output_dir,monitor,forward_rows,
                           warmup=args.warmup,runs=args.runs)
            groups.append(dict(**g,_samples=samples));dump(args.output_dir/'groups.json',[{k:v for k,v in x.items() if k!='_samples'} for x in groups])
        buffer_probe(s,short,'eos-2',args.output_dir)
        run(short,'eos-2')
        del short,runtime;gc.collect()
        old_spec=json.loads((args.input_dir/'frozen-spec.json').read_text())
        hc=old_spec['historical_config']
        runtime=DiscussionRuntime(b,config=DiscussionConfig(question=hc['question'],rounds=1,max_new_tokens=512,
            agent_a_system_prompt=hc['agent_a_system_prompt'],agent_b_system_prompt=hc['agent_b_system_prompt']))
        s=runtime.session;s._kv_contract=contract
        sender=frozen_state(args.input_dir,'long-source-A',s.state('A'))
        receiver=frozen_state(args.input_dir,'history-128-B',s.state('B'))
        p32=real_prefix(sender,32)
        more=options+[dict(label='legacy:v1',candidate='kv:v1',path='legacy'),
                      dict(label='packed:v1',candidate='kv:v1',path='packed')]
        run(dict(A=p32,B=receiver),'pending-32',more)
        p512=real_prefix(sender,512)
        run(dict(A=p512,B=receiver),'pending-512')
        buffer_probe(s,dict(A=p512,B=receiver),'pending-512',args.output_dir)
        history=frozen_state(args.input_dir,'history-2048-B',s.state('B'))
        run(dict(A=p32,B=history),'pending-32-history-2048')
        correctness_probe(s,dict(A=p32,B=receiver),args.output_dir,monitor,forward_rows)
        paired,networks,profiles=analyze(groups,settings)
        for name,row in [('paired-deltas',paired),('network',networks),('profiles',profiles)]:dump(args.output_dir/(name+'.json'),row)
        verify_index(args.input_dir)
        assert all(file_hash(Path(p))==sha for p,sha in spec['protected_user_assets'].items())
        dump(args.output_dir/'forward-summary.json',dict(count=len(forward_rows),calls=forward_rows))
        dump(args.output_dir/'completion.json',dict(status='verified',core_handoffs=sum(g['successful_handoffs'] for g in groups),
            pilot_handoffs=len(pilot),compatibility_handoffs=8,model_load_count=1,new_sender_generations=0,
            valid_samples=sum(not s['warmup'] for g in groups for s in g['_samples']),
            failures=0,resource_stop=monitor.stop_reason,cpu_fallback=False,token_fallback=False,
            source83a_unchanged=True,user_assets_unchanged=True,physical_network='not_measured'))
    except Exception as exc:
        dump(args.output_dir/'failure.json',dict(error=f'{type(exc).__name__}: {exc}',category=classify_failure(exc),
            completed_groups=[g['config']['label'] for g in groups],resource_stop=monitor.stop_reason))
        raise
    finally:
        monitor.close();dump(args.output_dir/'resource-samples.json',monitor.samples)
        dump(args.output_dir/'sha256-index.json',[dict(path=str(p.relative_to(args.output_dir)),bytes=p.stat().st_size,sha256=file_hash(p))
            for p in sorted(args.output_dir.rglob('*')) if p.is_file() and p.name!='sha256-index.json'])
    return 0
