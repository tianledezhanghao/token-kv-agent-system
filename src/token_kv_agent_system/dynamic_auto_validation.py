"""Bounded offline 0.5B/MPS validation; calibration is separate from AUTO runs."""
import argparse
from dataclasses import asdict, replace
import gc
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

from .communication import HandoffChoice
from .discussion import DiscussionConfig, DiscussionRuntime
from .secret_sum import SecretSumConfig, SecretSumRuntime, PUBLIC_TASK, STAGES, score_result
from .incremental_auto import (CANDIDATES, IncrementalAutoConfig, IncrementalCostProfile,
                               StageCost, backend_cost_identity)
from .incremental_kv_protocol import inspect_model_contract
from .local_qwen_comparison import REVISIONS, snapshot_path, file_hash, memory
from .local_qwen_resources import ResourceConfig, ResourceMonitor, classify_failure
from .network.handoff_network import HandoffNetworkConfig
from .model.state_cache import require_disjoint_caches


def synthetic_profile(identity):
    """Fixed BEFORE generation: routing fixture, never calibrated performance."""
    return IncrementalCostProfile('82a-synthetic-routing-v1','synthetic/test_only',identity,
        {'kv_contract_startup':StageCost(0), 'token_handoff':StageCost(1000), 'kv_base':StageCost(1), 'rope':StageCost(1),
         'suffix':StageCost(0,per_recomputed_token_ms=1), 'selection':StageCost(1), 'slot':StageCost(1)},
        0,(0,128),(0,8192),'Fabricated constants, solely to test AUTO routing. No real speed claim.')


def calibrate(backend, contract, emit, *, startup_ms=None):
    """One independent calibration; two actual Decode messages, warmup + 2 samples.

    Candidate branches use the SAME accepted sender and receiver snapshots. They
    are explicitly offline calibration, not online choices or quality oracles.
    Only TOKEN baseline continues generation to create the second message.
    """
    runtime = SecretSumRuntime(backend)
    session = runtime.session
    session._kv_contract = contract
    for name in ('A','B'):session.append_local_user(name,PUBLIC_TASK)
    samples=[]
    for source,target,prompt in [('A','B',STAGES.a_first),('B','A',STAGES.b_first)]:
        message=session.generate_message(source,target,max_new_tokens=96,local_prompt=prompt)
        saved=dict(session._states)
        n=message.token_count-int(message.termination_reason=='eos')
        cache=sum(s.cached_token_count for s in saved.values())
        for candidate in CANDIDATES:
            for sample_index in range(3):
                session._states=dict(saved)
                choice=HandoffChoice.parse(candidate)
                started=perf_counter()
                receipt=(session.handoff(source,target,measure_timing=True) if candidate=='token' else
                    session.handoff_kv(source,target,strategy=choice.kv_strategy,measure_timing=True))
                samples.append(dict(candidate=candidate, sample_index=sample_index,
                    warmup=sample_index==0,message_tokens=n,receiver_cache_tokens=saved[target].cached_token_count,
                    sender_cache_tokens=saved[source].cached_token_count,total_cache_tokens=cache,
                    observed_local_ms=(perf_counter()-started)*1000, observed_stage_ms=receipt.timings_ms,
                    actual_wire_bytes=receipt.wire_byte_count, model_identity=backend_cost_identity(backend),
                    measurement_scope='synchronized_instrumented_with_resource_monitor',
                    startup_contract_excluded=True, cold_start=False, received_ids=receipt.received_token_ids))
        session._states=dict(saved)
        session.handoff(source,target)  # TOKEN baseline only, before the next real generation.
    emit('calibration-samples',samples)
    warm=[s for s in samples if not s['warmup']]
    def stage(candidate,key):return median(s['observed_stage_ms'].get(key,0) for s in warm if s['candidate']==candidate)
    identity={'sender':backend_cost_identity(backend),'receiver':backend_cost_identity(backend)}
    profile=IncrementalCostProfile('82a-measured-incremental-v1','measured_incremental',identity,
        {'token_handoff':StageCost(median(s['observed_local_ms'] for s in warm if s['candidate']=='token')),
         'kv_base':StageCost(median(s['observed_local_ms'] for s in warm if s['candidate']=='kv:v0')),
         'rope':StageCost(stage('kv:v1','rope_rephase')),
         'suffix':StageCost(0,per_recomputed_token_ms=stage('kv:v2','suffix_recompute')),
         'selection':StageCost(stage('kv:v3','token_selection')),
         'slot':StageCost(stage('kv:v3','slot_repair'))},len(warm),
         (min(s['message_tokens'] for s in warm),max(s['message_tokens'] for s in warm)),
         (min(s['total_cache_tokens'] for s in warm),max(s['total_cache_tokens'] for s in warm)),
         'Development calibration only: two message/cache lengths, 2 warm samples per strategy per length; '
         'pooled medians within bounded domain, not independent throughput samples or statistical confidence. '
         'V0 base includes export/pending/host copy/encode/transfer/decode/restore/cache copies/framing/close/validation; '
         'V1 adds RoPE, V2 adds k-token suffix forward (only k=1 calibrated), V3 selection + crop/forward/scatter/validation. '
         'Backend/resource instrumentation included; startup/role prefill/decode generation excluded. '
         'No Canonical data. Exact frozen checkpoint; no cross-model reuse.', recompute_token_range=(0,1))
    if startup_ms is not None:
        profile = replace(profile, stages={**profile.stages, 'kv_contract_startup':StageCost(startup_ms)})
    emit('measured-profile',asdict(profile))
    return profile


def verify_runtime(runtime):
    events=[]
    runtime.event_sink=lambda kind,data:events.append(dict(kind=kind,data=data))
    result=runtime.run()
    assert result.status=='completed',result.error
    assert len(result.handoffs)==result.config.rounds*2
    for turn,h in zip(result.turns,result.handoffs):
        assert h.requested_mode=='auto' and h.actual_execution_status=='succeeded'
        assert h.executed_mode==h.selected_mode and h.executed_kv_strategy==h.selected_kv_strategy
        assert h.predicted_wire_bytes==h.actual_wire_bytes==turn.receipt.wire_byte_count
        assert h.auto_decision['decision_status']=='executed'
        assert h.fallback_reason is h.error is None
        assert h.observed_local_ms>0
    for name in ('A','B'):
        state=runtime.session.state(name);state.validate()
        assert all(str(t.device).startswith('mps') for layer in state.cache.layers for t in (layer.keys,layer.values))
    require_disjoint_caches(runtime.session.state('A').cache,runtime.session.state('B').cache)
    return result,events


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('benchmark-results/dynamic-auto-82a'))
    parser.add_argument('--phase',choices=('full','startup-smoke'),default='full')
    args=parser.parse_args(argv)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    def emit(name,row):
        (args.output_dir/(name+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    spec=dict(task='8.2A',model=REVISIONS['05b'],device='mps',dtype='bfloat16',offline=True,
        shared_weights=True,independent_caches=True,greedy=True,
        runs={'quality-discussion':dict(rounds=3,max_new_tokens=64,trace=[20,1000,50,500]),
              'measured-secret':dict(rounds=1,max_new_tokens=96,trace=[20,1000]),
              'synthetic-dynamic':dict(rounds=2,max_new_tokens=96,trace=[.001,1000,.001,1000]),
              'synthetic-v0-v1-v2-v3':dict(rounds=1,max_new_tokens=96,trace=[20,1000])},
        calibration=dict(messages=2,strategies=list(CANDIDATES),warmup_per_message_strategy=1,samples_per_message_strategy=2),
        prompts='existing production default Discussion and frozen SecretSum; no prompt search',
        synthetic_constants=dict(token_ms=1000,kv_base_ms=1,rope_ms=1,suffix_per_token_ms=1,selection_ms=1,slot_ms=1),
        no_real_network_sleep=True,resource_config=asdict(ResourceConfig()),
        source_sha256={str(p):file_hash(p) for p in sorted(Path('src').rglob('*.py'))},
        entrypoint_sha256=file_hash(Path('run_dynamic_auto_validation.py')))
    spec['phase']=args.phase
    if args.phase=='startup-smoke':
        spec['runs']={'startup-dynamic':dict(rounds=2,max_new_tokens=64,trace=[.001,1000,.001,1000])}
        spec['calibration']=None
    emit('frozen-spec',spec)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OTEL_SDK_DISABLED='true')
    import torch,transformers
    from .model import HuggingFaceModelBackend
    baseline=memory(torch);emit('preload',baseline)
    monitor=ResourceMonitor(lambda:memory(torch),baseline,args.output_dir)
    backend=None
    results=[]
    try:
        monitor.start();monitor.check();path=snapshot_path('05b')
        if not all((path/p).is_file() for p in ('config.json','tokenizer.json','tokenizer_config.json','model.safetensors')):
            raise FileNotFoundError('incomplete frozen offline checkpoint')
        torch.manual_seed(77)
        tokenizer=transformers.AutoTokenizer.from_pretrained(path,local_files_only=True)
        model=transformers.AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,local_files_only=True)
        backend=HuggingFaceModelBackend(str(path),device='mps',tokenizer=tokenizer,model=model)
        del tokenizer,model
        assert backend.device=='mps' and backend.metadata.dtype=='torch.bfloat16'
        assert backend.metadata.device_fallback_reason is None
        original=backend._timed_forward
        backend._timed_forward=lambda **kw:monitor.forward(original,**kw)
        started=perf_counter();contract=inspect_model_contract(backend)
        startup_ms=(perf_counter()-started)*1000
        emit('settings',dict(metadata=asdict(backend.metadata),contract=asdict(contract),revision=REVISIONS['05b'][1],
            torch=torch.__version__,transformers=transformers.__version__,contract_startup_ms=startup_ms))
        identity={'sender':backend_cost_identity(backend),'receiver':backend_cost_identity(backend)}
        measured=calibrate(backend,contract,emit,startup_ms=startup_ms) if args.phase=='full' else None
        synthetic=synthetic_profile(identity);emit('synthetic-profile',asdict(synthetic))
        configs=[('quality-discussion',DiscussionConfig(rounds=3,max_new_tokens=64,auto=IncrementalAutoConfig(),
            network=HandoffNetworkConfig(mode='trace',trace_mbps=(20,1000,50,500)))),
            ('measured-secret',SecretSumConfig(auto=IncrementalAutoConfig('performance_experiment',profile=measured),
                network=HandoffNetworkConfig(mode='trace',trace_mbps=(20,1000)))),
            ('synthetic-dynamic',DiscussionConfig(rounds=2,max_new_tokens=96,
                auto=IncrementalAutoConfig('performance_experiment',('token','kv:v1'),False,synthetic),
                network=HandoffNetworkConfig(mode='trace',trace_mbps=(.001,1000,.001,1000))))]
        configs += [(candidate.replace(':','-'),SecretSumConfig(auto=IncrementalAutoConfig('performance_experiment',
            ('token',candidate),candidate=='kv:v3',synthetic),network=HandoffNetworkConfig(mode='trace',trace_mbps=(20,1000)))) for candidate in CANDIDATES[1:]]
        if args.phase=='startup-smoke':
            configs=[('startup-dynamic',DiscussionConfig(rounds=2,max_new_tokens=64,
                auto=IncrementalAutoConfig('performance_experiment',('token','kv:v1'),False,synthetic),
                network=HandoffNetworkConfig(mode='trace',trace_mbps=(.001,1000,.001,1000))))]
        for name,config in configs:
            monitor.check()
            runtime=(SecretSumRuntime if isinstance(config,SecretSumConfig) else DiscussionRuntime)(backend,config=config)
            if args.phase=='full':runtime.session._kv_contract=contract
            start_forward=len(monitor.forwards)
            result,events=verify_runtime(runtime)
            if name=='quality-discussion':assert all(h.executed_mode=='token' for h in result.handoffs)
            if name in ('synthetic-dynamic','startup-dynamic'):assert [h.executed_mode for h in result.handoffs]==['token','kv','token','kv']
            if name.startswith('kv-'):assert all(h.executed_mode=='kv' for h in result.handoffs)
            row=dict(result=asdict(result),events=events,forward_index_range=[start_forward,len(monitor.forwards)],
                quality_score=score_result(config,result) if isinstance(config,SecretSumConfig) else None,
                physical_network_ms='not_measured',cpu_communication_fallback=False,memory=monitor.sample())
            emit(name,row);results.append(dict(name=name,status=result.status,
                selected=[h.auto_decision['selected_candidate'] for h in result.handoffs],
                bandwidths=[h.network['bandwidth_mbps'] for h in result.handoffs],
                wire_bytes=[h.actual_wire_bytes for h in result.handoffs],
                observed_local_ms=[h.observed_local_ms for h in result.handoffs],
                predicted_local_ms=[h.predicted_local_ms for h in result.handoffs],
                profile_provenance=result.handoffs[0].auto_decision['profile_provenance'],
                final=result.final_text,quality_score=row['quality_score']))
            print(name,results[-1]['selected'],flush=True)
            del runtime,result,row;gc.collect();torch.mps.empty_cache()
        emit('completion',dict(status='verified',runs=results,forward_count=len(monitor.forwards),
            resource_stop=monitor.stop_reason,memory=monitor.sample()))
        return 0
    except Exception as exc:
        emit('failure',dict(status='failed',error=f'{type(exc).__name__}: {exc}',category=classify_failure(exc),
            completed_runs=results,forward_count=len(monitor.forwards),resource_stop=monitor.stop_reason))
        print(type(exc).__name__,str(exc),flush=True);return 1
    finally:
        monitor.close();emit('memory-samples',dict(samples=monitor.samples,forwards=monitor.forwards))
