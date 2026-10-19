"""Bounded 8.3A real Decode same-snapshot MPS handoff characterization."""
import argparse
from dataclasses import asdict, replace
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter, monotonic

from .performance_characterization import (measured_handoff, real_prefix, cache_sha256,
                                            statistics)
from .local_qwen_comparison import REVISIONS, snapshot_path, file_hash, memory
from .local_qwen_resources import ResourceMonitor, classify_failure, ResourceConfig

HISTORICAL_INPUT = Path('benchmark-results/discussion-82c-20261015/validated/58b11518-0790-4faf-a05f-6ff47f8d4bf4.json')


def dump(path, row):
    path.write_text(json.dumps(row,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def pad_history(state, target):
    """Controlled local user history, actual model forward, outside timing."""
    from .agent.state import working_state, append_working_ids
    # Preserve role prompt and question; insert an explicitly labelled local turn.
    frame = state.framing.local_user_ids('')
    # Empty local user template has a leading role prefix and actual closing.
    closing = state.framing.assistant.closing_ids
    assert frame[-len(closing):]==closing
    prefix = frame[:-len(closing)]
    extra = target-state.cached_token_count-len(frame)
    if extra < 0:
        raise ValueError('target history too short for local framing')
    token = state.backend.chat_tokenizer.encode(' history',add_special_tokens=False)[0]
    assert token not in state.backend.chat_tokenizer.all_special_ids
    work = append_working_ids(working_state(state),prefix+(token,)*extra+closing)
    assert work.cached_token_count==target
    work.validate()
    return work


def freeze_state(output_dir, label, state):
    from .model.kv_adapter import serialize_huggingface_kv_cache
    encoded = serialize_huggingface_kv_cache(state.cache,
        logical_cached_token_count=state.cached_token_count,model_id=state.backend.model_name,
        model_config=state.backend.kv_model_config,torch_module=state.backend._torch)
    path = output_dir/(label+'.kv')
    path.write_bytes(encoded.packet)
    row = dict(agent_id=state.agent_id,system_prompt=state.system_prompt,token_ledger=state.token_ledger,
               cached_token_count=state.cached_token_count,pending_token_id=state.pending_token_id,
               assistant_open=state.assistant_open,draft_message=asdict(state.draft_message) if state.draft_message else None,
               tensor_sha256=cache_sha256(state.cache),cache_file=path.name,cache_file_sha256=file_hash(path))
    dump(output_dir/(label+'.json'),row)
    return row


def summarize_group(samples):
    summary = {}
    for mode in ('steady','instrumented'):
        summary[mode] = {}
        for candidate in sorted(set(s['candidate_label'] for s in samples)):
            rows = [s for s in samples if not s['warmup'] and s['mode']==mode and s['candidate_label']==candidate]
            stage_names = set().union(*(s['stages'] for s in rows))
            wires = {s['wire_bytes'] for s in rows}
            assert len(wires)==1
            summary[mode][candidate] = dict(**statistics(s['local_ms'] for s in rows),wire_bytes=wires.pop(),
                stages={name:{field:statistics(s['stages'].get(name,{}).get(field,0) for s in rows)
                    for field in ('inclusive_ms','exclusive_ms')} for name in sorted(stage_names)},
                adapter_submetrics={key:statistics(sum(s['adapter_submetrics'].get(key,[])) for s in rows)
                    for key in set().union(*(s['adapter_submetrics'] for s in rows))})
    return summary


def run_group(session,saved,label,origin,options,output_dir,monitor,*,warmup,runs,forward_rows):
    from .model.huggingface import synchronize_device
    from .incremental_auto import estimate_incremental_wire_bytes
    b=saved['A'].backend
    before={n:cache_sha256(s.cache) for n,s in saved.items()}
    message=saved['A'].draft_message
    n=message.token_count-int(message.termination_reason=='eos')
    config=dict(label=label,origin=origin,message_tokens=n,message_ids=message.message_token_ids,
        message_text=b.decode(message.message_token_ids,skip_special_tokens=False),termination=message.termination_reason,
        sender_span=message.sender_message_span,sender_cache_tokens=saved['A'].cached_token_count,
        sender_history_tokens=message.sender_message_span[0],receiver_cache_tokens=saved['B'].cached_token_count,
        total_cache_tokens=sum(s.cached_token_count for s in saved.values()),snapshot_sha256=before,
        framing=asdict(saved['B'].framing.incoming('A','B')),options=options,warmup=warmup,runs=runs,
        snapshot_reset='same immutable base Cache; production work clones per handoff; no previous result reused')
    dump(output_dir/(label+'-config.json'),config)
    samples=[]
    # Pair order alternates each repetition. Instrumented and steady are separate.
    for mode in ('steady','instrumented'):
        for i in range(warmup+runs):
            ordered=options if i%2==0 else list(reversed(options))
            for option in ordered:
                monitor.check()
                start_index=len(forward_rows)
                sample=measured_handoff(session,saved,option['candidate'],mode=mode,
                    recompute_tokens=option.get('recompute_tokens'),recompute_ratio=option.get('recompute_ratio'))
                sample.update(candidate=option['candidate'],candidate_label=option['label'],index=i,
                    warmup=i<warmup,forward_range=[start_index,len(forward_rows)])
                calls=forward_rows[start_index:]
                # Check the actual receiver content-forward boundary without CPU trace IO.
                frame=config['framing'];start=saved['B'].cached_token_count
                if option['candidate']=='token':
                    assert any(c['cache_tokens']==start and c['input_count']==len(sample['receipt']['receiver_appended_ids']) for c in calls)
                elif option['candidate'] in ('kv:v0','kv:v1'):
                    target=start+len(frame['prefix_ids'])
                    assert not any(c['cache_tokens']==target for c in calls)
                    assert any(c['cache_tokens']==start and c['input_count']==len(frame['prefix_ids']) for c in calls)
                    assert any(c['cache_tokens']==target+n and c['input_count']==len(frame['closing_ids']) for c in calls)
                    assert sample['wire_bytes']==estimate_incremental_wire_bytes(saved['A'],message)
                sample['forward_calls']=calls
                sample['mps_allocated_after_bytes']=b._torch.mps.current_allocated_memory()
                sample['mps_driver_after_bytes']=b._torch.mps.driver_allocated_memory()
                samples.append(sample)
                with (output_dir/(label+'-samples.jsonl')).open('a') as f:
                    f.write(json.dumps(sample,ensure_ascii=False,allow_nan=False)+'\n')
            # Resource commands/garbage collection outside handoff boundaries.
        print(label,mode,'complete',flush=True)
    synchronize_device(b._torch,b.device)
    after={name:cache_sha256(state.cache) for name,state in saved.items()}
    assert before==after,'base snapshot mutated across measurements'
    summary=summarize_group(samples)
    resource=monitor.sample()
    row=dict(config=config,summary=summary,snapshot_unchanged=True,resource=resource,
             successful_handoffs=len(samples),failed_handoffs=0)
    dump(output_dir/(label+'-summary.json'),row)
    print(label,{k:round(v['median'],2) for k,v in summary['steady'].items()},flush=True)
    session._states=dict(saved)
    return row


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('benchmark-results/performance-83a'))
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--runs',type=int,default=5)
    parser.add_argument('--historical-input',type=Path,default=HISTORICAL_INPUT)
    args=parser.parse_args(argv)
    if args.warmup<1 or args.runs<5:
        parser.error('at least one warmup and five measured samples required')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OTEL_SDK_DISABLED='true')
    # Historical JSON provides frozen inputs only. No claim of Tensor recovery.
    old=json.loads(args.historical_input.read_text())
    historical_config=old['session']['config']
    spec=dict(task='8.3A',model=REVISIONS['05b'],device='mps',dtype='bfloat16',batch=1,
        offline=True,shared_weights=True,real_decode=True,warmup=args.warmup,samples=args.runs,
        historical_input=str(args.historical_input),historical_input_sha256=file_hash(args.historical_input),
        historical_config=historical_config,historical_kv_recovered=False,
        boundary='Sender generation complete to both states committed and device synchronized; external full handoff wall',
        modes=['steady_no_stage_wrappers_no_tensor_trace','instrumented_nested_sync_wrappers'],
        network='offline simulation only, actual protocol bytes, 2ms one-way base latency',
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):file_hash(p) for p in sorted(Path('src').rglob('*.py'))},
        entrypoint_sha256=file_hash(Path('run_performance_characterization.py')),
        hardware=dict(platform=platform.platform(),machine=platform.machine(),
            cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
            memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))),
        resource_config=asdict(ResourceConfig()))
    dump(args.output_dir/'frozen-spec.json',spec)
    import torch,transformers
    from .model import HuggingFaceModelBackend
    from .model.huggingface import synchronize_device
    from .incremental_kv_protocol import inspect_model_contract
    from .incremental_auto import backend_cost_identity
    from .secret_sum import SecretSumRuntime,PUBLIC_TASK,STAGES
    from .discussion import DiscussionConfig,DiscussionRuntime
    baseline=memory(torch)
    monitor=ResourceMonitor(lambda:memory(torch),baseline,args.output_dir)
    groups=[];forward_rows=[];generations=[]
    try:
        monitor.start();monitor.check()
        path=snapshot_path('05b')
        if not all((path/p).is_file() for p in ('config.json','tokenizer.json','tokenizer_config.json','model.safetensors')):
            raise FileNotFoundError('incomplete offline checkpoint')
        torch.manual_seed(77)
        tokenizer=transformers.AutoTokenizer.from_pretrained(path,local_files_only=True)
        model=transformers.AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,local_files_only=True)
        b=HuggingFaceModelBackend(str(path),device='mps',tokenizer=tokenizer,model=model)
        del tokenizer,model
        assert b.device=='mps' and b.metadata.dtype=='torch.bfloat16' and b.metadata.device_fallback_reason is None
        original=b._timed_forward
        def forward(**kw):
            monitor.check(); monitor.active_forward_since=monotonic()
            try:
                result=original(**kw)
                forward_rows.append(dict(input_count=int(kw['input_ids'].shape[1]),
                    cache_tokens=int(kw['attention_mask'].shape[1])-int(kw['input_ids'].shape[1]),
                    forward_ms=result[1],device=str(kw['input_ids'].device)))
                monitor.check();return result
            finally:
                monitor.active_forward_since=None
        b._timed_forward=forward
        synchronize_device(torch,'mps');t=perf_counter();contract=inspect_model_contract(b);synchronize_device(torch,'mps')
        startup_ms=(perf_counter()-t)*1000
        identity={'sender':backend_cost_identity(b),'receiver':backend_cost_identity(b)}
        dump(args.output_dir/'settings.json',dict(metadata=asdict(b.metadata),contract=asdict(contract),
             model_identity=identity,revision=REVISIONS['05b'][1],torch=torch.__version__,transformers=transformers.__version__,
             contract_startup_ms=startup_ms,model_files={p:file_hash(path/p) for p in ('config.json','tokenizer.json','tokenizer_config.json','model.safetensors')}))
        def generate(session,prompt,label,budget):
            start=len(forward_rows);synchronize_device(torch,'mps');t=perf_counter()
            m=session.generate_message('A','B',max_new_tokens=budget,local_prompt=prompt)
            synchronize_device(torch,'mps')
            generations.append(dict(label=label,message=asdict(m),text=b.decode(m.message_token_ids,skip_special_tokens=False),
                generation_wall_ms=(perf_counter()-t)*1000,forward_range=[start,len(forward_rows)],prompt=prompt,
                shared_cost_excluded_from_handoff=True))
            dump(args.output_dir/'generations.json',generations)
            return m
        pair=[dict(candidate=c,label=c) for c in ('token','kv:v0')]
        def run(session,saved,label,origin,options=pair):
            session._kv_contract=contract
            row=run_group(session,saved,label,origin,options,args.output_dir,monitor,
                warmup=args.warmup,runs=args.runs,forward_rows=forward_rows)
            groups.append(row);dump(args.output_dir/'groups.json',groups)
            return row
        short=SecretSumRuntime(b);ss=short.session
        for name in ('A','B'):ss.append_local_user(name,PUBLIC_TASK)
        generate(ss,STAGES.a_first,'production-secret-first',96)
        frozen_short=dict(ss._states)
        for name,state in frozen_short.items():freeze_state(args.output_dir,'short-'+name,state)
        run(ss,frozen_short,'short-real','real_decode')
        del short,ss,frozen_short;gc.collect()
        config=DiscussionConfig(question=historical_config['question'],rounds=1,max_new_tokens=512,
            agent_a_system_prompt=historical_config['agent_a_system_prompt'],agent_b_system_prompt=historical_config['agent_b_system_prompt'])
        runtime=DiscussionRuntime(b,config=config);s=runtime.session
        for name in ('A','B'):s.append_local_user(name,config.question)
        generate(s,runtime.prompt_config.stages.a_first,'82c-frozen-input-new-decode',512)
        sender=s.state('A');receiver=s.state('B')
        freeze_state(args.output_dir,'long-source-A',sender)
        base=pad_history(receiver,128)
        freeze_state(args.output_dir,'history-128-B',base)
        lengths=(2,8,32,128,256,512)
        prefixes={}
        for n in lengths:
            prefixes[n]=real_prefix(sender,n)
            run(s,dict(A=prefixes[n],B=base),f'length-{n}','real_decode_prefix')
        for history in (512,1024,2048):
            target=pad_history(receiver,history)
            freeze_state(args.output_dir,f'history-{history}-B',target)
            run(s,dict(A=prefixes[8],B=target),f'history-{history}','real_decode_prefix')
            del target;gc.collect()
        options=[dict(candidate='kv:v1',label='kv:v1'),
                 *[dict(candidate='kv:v2',label=f'kv:v2:k{k}',recompute_tokens=k) for k in (0,1,2,32)],
                 dict(candidate='kv:v2',label='kv:v2:ratio25',recompute_ratio=.25),
                 dict(candidate='kv:v3',label='kv:v3')]
        run(s,dict(A=prefixes[32],B=base),'corrections-32','real_decode_prefix',options)
        # Offline phase uses only captured metadata, independently testable.
        from .performance_characterization import offline_analysis
        network,profiles,decisions=offline_analysis(groups,json.loads((args.output_dir/'settings.json').read_text()))
        dump(args.output_dir/'network.json',network);dump(args.output_dir/'profiles.json',profiles)
        dump(args.output_dir/'selector-decisions.json',decisions)
        dump(args.output_dir/'forward-summary.json',dict(count=len(forward_rows),calls=forward_rows))
        dump(args.output_dir/'completion.json',dict(status='verified',groups=len(groups),generations=len(generations),
            handoffs=sum(r['successful_handoffs'] for r in groups),valid_samples=sum(
                v['sample_count'] for r in groups for mode in r['summary'].values() for v in mode.values()),
            forward_count=len(forward_rows),resource_stop=monitor.stop_reason,resource=monitor.sample(),
            cpu_fallback=False,token_fallback=False,physical_network='not_measured'))
        return 0
    except Exception as exc:
        dump(args.output_dir/'failure.json',dict(error=f'{type(exc).__name__}: {exc}',category=classify_failure(exc),
            completed_groups=[r['config']['label'] for r in groups],resource_stop=monitor.stop_reason))
        print(type(exc).__name__,str(exc),flush=True)
        raise
    finally:
        monitor.close();dump(args.output_dir/'resource-samples.json',monitor.samples)
        files=sorted(p for p in args.output_dir.rglob('*') if p.is_file() and p.name!='sha256-index.json')
        dump(args.output_dir/'sha256-index.json',[dict(path=str(p.relative_to(args.output_dir)),bytes=p.stat().st_size,
                                                   sha256=file_hash(p)) for p in files])
