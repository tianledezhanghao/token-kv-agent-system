"""Offline paired real-Decode calibration for bounded generalized AUTO profiles.

Single local model, no downloads, no distributed inference, no CPU fallback.
The CLI refuses an active model service. Large data belongs in ignored outputs.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import sys
import threading
from time import monotonic, perf_counter
from urllib.request import urlopen

from .communication import HandoffChoice, suffix_recompute_count
from .generalized_cost import (GeneralizedCostProfile, digest, environment_identity,
    workload_features, MEASUREMENT_SCOPE, load_cost_profile)
from .incremental_auto import backend_cost_identity, estimate_incremental_wire_bytes, IncrementalAutoConfig, IncrementalSelector
from .network.handoff_network import HandoffNetwork, HandoffNetworkConfig


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def csv_counts(value):
    try:
        counts=tuple(int(x) for x in value.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('use comma-separated integer counts') from exc
    if not counts or any(x<=0 for x in counts) or len(set(counts))!=len(counts):
        raise argparse.ArgumentTypeError('counts must be unique positive integers')
    return tuple(sorted(counts))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', help='Existing local snapshot directory; never downloaded')
    p.add_argument('--cache-dir', type=Path, help='Optional HF cache root; model remains a local snapshot')
    p.add_argument('--device',default='mps',help='Explicit mps or cuda:N; no auto/CPU fallback')
    p.add_argument('--dtype',choices=('bfloat16','float16','float32'),default='bfloat16')
    p.add_argument('--kv-version',choices=('v0','v1','v2','v3'),default='v0')
    r=p.add_mutually_exclusive_group()
    r.add_argument('--recompute-tokens',type=int,default=None)
    r.add_argument('--recompute-ratio',type=float,default=None)
    p.add_argument('--message-lengths',type=csv_counts,default=(4,12))
    p.add_argument('--receiver-histories',type=csv_counts,default=(64,128))
    p.add_argument('--sender-histories',type=csv_counts,default=(128,))
    p.add_argument('--termination',choices=('pending','eos'),default='pending',help='Pending preserves real Decode prefixes; EOS uses only actual full generated messages')
    p.add_argument('--receiver-pending',action='store_true',help='Calibrate receiver open draft with genuine generated pending token')
    p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--samples',type=int,default=2)
    p.add_argument('--holdout-samples',type=int,default=2)
    p.add_argument('--max-handoffs',type=int,default=128)
    p.add_argument('--max-seconds',type=float,default=600)
    p.add_argument('--max-model-gib',type=float,default=8)
    p.add_argument('--max-device-memory-gib',type=float,default=10)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--service-url',default='http://127.0.0.1:8000/api/discussions/latest')
    p.add_argument('--serve-profile',type=Path,help='Load and serve an existing identity-matched profile; no calibration or automatic discussions')
    p.add_argument('--discussion-smoke',action='store_true',help='One real REST/WS synthetic PATCH discussion and one measured-profile discussion')
    p.add_argument('--serve-seconds',type=int,default=0,help='Keep the same 0.5B smoke service briefly for browser QA; bounded by max-seconds')
    p.add_argument('--port',type=int,default=8015)
    p.add_argument('--verify',action='store_true',help='Read-only SHA/statistics verification; --model is ignored')
    return p


def validate_args(args):
    if not args.model:
        raise ValueError('--model requires a complete local snapshot')
    from .communication import validate_recompute_policy
    from .model.config import validate_device_spec
    validate_device_spec(args.device)
    if args.device not in ('mps','cuda') and not args.device.startswith('cuda:'):
        raise ValueError('calibration requires explicit mps or cuda:N; CPU/auto forbidden')
    validate_recompute_policy(args.recompute_tokens,args.recompute_ratio)
    if args.kv_version!='v2' and (args.recompute_tokens is not None or args.recompute_ratio is not None):
        raise ValueError('recompute parameters require V2')
    if args.warmup<1 or args.samples<2 or args.holdout_samples<1:
        raise ValueError('require warmup >=1, samples >=2, holdout >=1')
    for name in ('max_handoffs','max_seconds','max_model_gib','max_device_memory_gib'):
        from .network.handoff_network import finite_number
        finite_number(getattr(args,name),name,positive=True)
    # Two independent source messages (training and holdout); pair budget includes all warmups.
    estimate=2*len(args.sender_histories)*len(args.message_lengths)*len(args.receiver_histories)*(args.warmup+args.samples)
    estimate+=4*len(args.sender_histories)*args.holdout_samples
    if args.discussion_smoke:
        estimate+=6  # One measured round plus two idealized rounds; no retries.
    if estimate>args.max_handoffs:
        raise ValueError(f'planned handoffs {estimate} exceed budget {args.max_handoffs}')
    if args.serve_seconds<0 or args.serve_seconds>args.max_seconds or not 1024<=args.port<=65535:
        raise ValueError('invalid bounded service configuration')
    if max(args.message_lengths)>4096 or max((*args.receiver_histories,*args.sender_histories))>16384:
        raise ValueError('workload exceeds bounded calibration limits')
    if args.discussion_smoke and (args.kv_version != 'v0' or args.receiver_pending or args.termination != 'pending'):
        raise ValueError('minimal Discussion smoke uses V0 pending calibration')


def refuse_active_service(url):
    try:
        with urlopen(url,timeout=2) as response:
            data=json.load(response)
    except OSError:
        return
    if data.get('available') or data.get('active_session_id') or data.get('model_switch',{}).get('loading'):
        raise RuntimeError('active model service detected; stop it cleanly before exclusive calibration')


@contextmanager
def exclusive_calibration_lock():
    import fcntl
    path=Path.home()/'.cache/token-kv-generalized-calibration.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another calibration process owns this device budget') from exc
        yield


def make_history(state,count,text):
    """Real prefilled history only, exact length. Message KV is never rebuilt here."""
    from .agent.state import append_working_ids, working_state
    needed=count-state.cached_token_count
    ids=state.framing.local_user_ids(text)
    if needed<len(ids):
        raise ValueError(f'history target {count} too short for initial role and calibration prompt')
    padding=state.backend.encode(' context')
    if not padding:
        raise ValueError('empty calibration history filler')
    # Insert regular text-token padding immediately after the local user prefix.
    closing=len(state.framing.assistant.closing_ids)
    # Prompt content is at the end. Leading whitespace is valid user content.
    prefix_len=len(state.framing.local_user_ids(''))-closing
    pad=(padding*((needed-len(ids)+len(padding)-1)//len(padding)))[:needed-len(ids)]
    ids=ids[:prefix_len]+pad+ids[prefix_len:]
    work=append_working_ids(working_state(state),ids)
    assert work.cached_token_count==count
    return work


def source_snapshot(backend,sender_history,max_n,*,holdout=False,termination='pending',receiver_pending=False):
    from .agent import LLMAgent
    from .stateful_token import StatefulTokenSession
    from .transport import InMemoryByteTransport
    from .incremental_kv_protocol import inspect_model_contract
    session=StatefulTokenSession((LLMAgent('A',backend,system_prompt='Follow the user instruction.'),
                                 LLMAgent('B',backend,system_prompt='Read the incoming message.')),
                                InMemoryByteTransport(),max_context_tokens=32768)
    base_b=session.state('B')
    prompt=('List the integers from 31 to 99, one by one, separated by commas.' if holdout else
            'List the integers from 1 to 99, one by one, separated by commas.') if termination=='pending' else (
            'Reply only with 43.' if holdout else 'Reply only with 17.')
    assistant_prefix=len(session.state('A').framing.assistant.prefix_ids)
    session._states['A']=make_history(session.state('A'),sender_history-assistant_prefix,prompt)
    message=session.generate_message('A','B',max_new_tokens=max_n+1 if termination=='pending' else max_n)
    if termination=='eos' and message.termination_reason!='eos':
        raise ValueError('EOS calibration requires a genuinely EOS-terminated generated message')
    # One contract is prepared at channel setup; startup measured separately once.
    # Reuse the exact actual contract across snapshots, never rehash each branch.
    session._kv_contract=backend._calibration_contract
    return session,session.state('A'),base_b,message


def measure_pair(session,saved,candidate,k,ratio,*,samples,warmup,check,tag,rows):
    from .performance_characterization import cache_sha256
    before={a:cache_sha256(s.cache) for a,s in saved.items()}
    message=saved['A'].draft_message
    n=message.token_count-int(message.termination_reason=='eos')
    recomputed=suffix_recompute_count(n,tokens=k,ratio=ratio) if candidate=='kv:v2' else int(candidate=='kv:v3')
    features=workload_features(saved['A'],saved['B'],message,candidate,contract_ready=True,recomputed_tokens=recomputed)
    wire_expected=10+4*message.token_count if candidate=='token' else estimate_incremental_wire_bytes(saved['A'],message)
    choice=HandoffChoice.parse(candidate,**(dict(recompute_tokens=k,recompute_ratio=ratio) if candidate=='kv:v2' else {}))
    result=[]
    for i in range(warmup+samples):
        check();session._states=dict(saved)
        started=perf_counter()
        receipt=(session.handoff('A','B',measure_timing=True) if candidate=='token' else
            session.handoff_kv('A','B',strategy=choice.kv_strategy,recompute_tokens=choice.recompute_tokens,
                recompute_ratio=choice.recompute_ratio,measure_timing=True))
        elapsed=(perf_counter()-started)*1000
        assert receipt.received_token_ids==message.message_token_ids and receipt.wire_byte_count==wire_expected
        assert session.state('B').received_messages[-1].message_id==message.message_id
        row=dict(tag=tag,candidate=candidate,features=features,local_ms=elapsed,wire_bytes=receipt.wire_byte_count,
            index=i,warmup=i<warmup,measurement_scope=MEASUREMENT_SCOPE,
            message_ids=list(message.message_token_ids),message_sha256=digest(message.message_token_ids),
            observed_stage_ms=receipt.timings_ms,source_cache_sha256=before,
            receipt=asdict(receipt))
        rows.append(row);result.append(row)
    assert before=={a:cache_sha256(s.cache) for a,s in saved.items()}
    session._states=dict(saved)
    valid=[r for r in result if not r['warmup']]
    center=median(r['local_ms'] for r in valid)
    return dict(candidate=candidate,features=features,local_ms=center,
        spread_ms=max(abs(r['local_ms']-center) for r in valid),samples=len(valid),wire_bytes=wire_expected)


def calibration(backend,args,check,output):
    from .performance_characterization import real_prefix
    from .incremental_kv_protocol import inspect_model_contract
    from .agent.state import append_working_ids,working_state
    points=[];rows=[];holdout=[];sources=[]
    candidates=('token','kv:'+args.kv_version)
    identity={key:backend_cost_identity(backend) for key in ('sender','receiver')}
    environment={key:environment_identity(backend) for key in ('sender','receiver')}
    check();started=perf_counter();backend._calibration_contract=inspect_model_contract(backend)
    startup_ms=(perf_counter()-started)*1000
    write_json(output/'startup.json',dict(observed_ms=startup_ms,samples=1,uncertainty_ms=startup_ms,
        scope='channel_model_contract_hashing_only',contract=asdict(backend._calibration_contract)))
    def sample_check():
        check()
        if len(rows)>=args.max_handoffs:
            raise RuntimeError('handoff budget exceeded')
    def snapshots(holdout_content):
        for sender_h in args.sender_histories:
            check()
            session,sender,base_b,message=source_snapshot(backend,sender_h,max(args.message_lengths),
                holdout=holdout_content,termination=args.termination)
            sources.append(dict(holdout=holdout_content,message=asdict(message),cache_sha256=__import__(
                'token_kv_agent_system.performance_characterization',fromlist=['cache_sha256']).cache_sha256(sender.cache)))
            if args.termination=='pending':
                lengths=(args.message_lengths if not holdout_content else tuple(dict.fromkeys((
                    (min(args.message_lengths)+max(args.message_lengths))//2,max(args.message_lengths)))))
            else:
                lengths=(message.token_count-1,)
            histories=args.receiver_histories if not holdout_content else ((min(args.receiver_histories)+max(args.receiver_histories))//2,)
            for n in lengths:
                source=real_prefix(sender,n) if args.termination=='pending' else sender
                for h in histories:
                    receiver=make_history(base_b,h,'Receive the next message.')
                    if args.receiver_pending:
                        # Generate a genuine one-token open Receiver draft, adjusting the
                        # base length so cached H is exact after its assistant prefix.
                        prefix=len(receiver.framing.assistant.prefix_ids)
                        receiver=make_history(base_b,h-prefix,'Reply with a long list of numbers.')
                        session._states={'A':source,'B':receiver}
                        session.generate_message('B','A',max_new_tokens=1)
                        receiver=session.state('B')
                    yield session,{'A':source,'B':receiver},n,h
            session._states={};gc.collect()
    try:
        for session,saved,n,h in snapshots(False):
            for candidate in candidates:
                point=measure_pair(session,saved,candidate,args.recompute_tokens,args.recompute_ratio,
                    samples=args.samples,warmup=args.warmup,check=sample_check,tag='train',rows=rows)
                points.append(point)
            write_json(output/'samples.json',rows)
            print(f'calibration N={n} H={h} S={saved["A"].draft_message.sender_message_span[0]} complete',flush=True)
        profile=GeneralizedCostProfile(version='generalized-calibration-v1',model_identity=identity,
            environment=environment,points=points,sample_count=sum(p['samples'] for p in points),
            calibrated_at=datetime.now(timezone.utc).isoformat(),evidence_sha256=digest(points),
            notes='Experimental bounded joint interpolation of paired genuine Decode handoffs. Full instrumented Session call, generation/setup excluded. No semantic certification. Empirical uncertainty, no extrapolation.',
            startup_cost_ms=startup_ms,startup_uncertainty_ms=startup_ms)
        write_json(output/'profile.json',asdict(profile))
        for session,saved,n,h in snapshots(True):
            actual={};predictions={}
            for candidate in candidates:
                actual[candidate]=measure_pair(session,saved,candidate,args.recompute_tokens,args.recompute_ratio,
                    samples=args.holdout_samples,warmup=0,check=sample_check,tag='holdout',rows=rows)
                try:
                    predictions[candidate]=profile.predict(candidate,actual[candidate]['features'],identity,environment)
                except ValueError as exc:
                    predictions[candidate]=dict(rejected=str(exc))
            row=dict(n=n,h=h,actual=actual,predictions=predictions,
                     message_sha256=digest(saved['A'].draft_message.message_token_ids), comparisons=[])
            for bandwidth in (20,10000):
                net=HandoffNetwork(HandoffNetworkConfig(bandwidth_mbps=bandwidth)).snapshot(1)
                try:
                    _,decision=IncrementalSelector(IncrementalAutoConfig('performance_experiment',candidates,
                        args.kv_version=='v3',profile,args.recompute_tokens,args.recompute_ratio)).select(
                            saved['A'],saved['B'],saved['A'].draft_message,net,kv_contract_ready=True,require_workload_domain=True)
                    measured_best=min(candidates,key=lambda c:actual[c]['local_ms']+net.transfer_ms(actual[c]['wire_bytes']))
                    row['comparisons'].append(dict(bandwidth=bandwidth,decision=decision,measured_best=measured_best,
                        agreement=decision['selected_candidate']==measured_best))
                except ValueError as exc:
                    row['comparisons'].append(dict(bandwidth=bandwidth,rejected=str(exc)))
            holdout.append(row)
            write_json(output/'holdout.json',holdout)
        # Keep all observed errors. Widen the empirical envelope if this small
        # independent holdout exceeds the initial 25% floor; no accuracy badge.
        errors=[dict(candidate=c,absolute_ms=abs(r['predictions'][c]['predicted_local_ms']-r['actual'][c]['local_ms']),
            relative=abs(r['predictions'][c]['predicted_local_ms']-r['actual'][c]['local_ms'])/r['actual'][c]['local_ms'],
            relative_to_prediction=abs(r['predictions'][c]['predicted_local_ms']-r['actual'][c]['local_ms'])/r['predictions'][c]['predicted_local_ms'],
            within_envelope=abs(r['predictions'][c]['predicted_local_ms']-r['actual'][c]['local_ms'])<=r['predictions'][c]['uncertainty_ms'])
            for r in holdout for c in candidates if 'rejected' not in r['predictions'][c]]
        from dataclasses import replace
        floor=max([.25]+[e['relative_to_prediction'] for e in errors])
        final=replace(profile,relative_error_floor=floor)
        write_json(output/'profile.json',asdict(final))
        report=dict(holdout_pairs=len(holdout),errors=errors,training_samples=profile.sample_count,
            holdout_samples=sum(not r['warmup'] and r['tag']=='holdout' for r in rows),
            initial_error_floor=.25,final_error_floor=floor,profile_reliability=final.reliability,
            note='Errors use pre-holdout predictions. Final envelope may be widened using holdout, not re-evaluated or certified.',
            source_messages=len(sources),sources=sources,observed_successful_handoffs=len(rows))
        write_json(output/'report.json',report)
        return final,report
    finally:
        write_json(output/'samples.json',rows)
        write_json(output/'sources.json',sources)
        write_json(output/'holdout.json',holdout)


def verify_output(output):
    manifest=json.loads((output/'manifest.json').read_text())
    for name,sha in manifest.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or not (output/relative).resolve().is_relative_to(output.resolve()) or hashlib.sha256((output/relative).read_bytes()).hexdigest()!=sha:
            raise ValueError('evidence SHA mismatch: '+name)
    profile_path=output/('profile.json' if (output/'profile.json').exists() else 'measured-profile.json')
    profile=load_cost_profile(json.loads(profile_path.read_text()))
    rows=json.loads((output/'samples.json').read_text())
    valid=[r for r in rows if r['tag']=='train' and not r['warmup']]
    assert len(valid)==profile.sample_count
    for p in profile.points:
        values=[r['local_ms'] for r in valid if r['candidate']==p['candidate'] and r['features']==p['features']]
        assert median(values)==p['local_ms']
        assert max(abs(x-p['local_ms']) for x in values)==p['spread_ms']
    report=json.loads((output/'report.json').read_text())
    assert report['observed_successful_handoffs']==len(rows)
    assert report['holdout_samples']==sum(r['tag']=='holdout' and not r['warmup'] for r in rows)
    holdouts=json.loads((output/'holdout.json').read_text())
    expected=[]
    for row in holdouts:
        for candidate,actual in row['actual'].items():
            prediction=row['predictions'][candidate]
            if 'rejected' in prediction:
                continue
            error=abs(prediction['predicted_local_ms']-actual['local_ms'])
            expected.append(dict(candidate=candidate,absolute_ms=error,relative=error/actual['local_ms'],
                relative_to_prediction=error/prediction['predicted_local_ms'],
                within_envelope=error<=prediction['uncertainty_ms']))
        for comparison in row['comparisons']:
            if 'rejected' in comparison:
                continue
            network=HandoffNetwork(HandoffNetworkConfig(bandwidth_mbps=comparison['bandwidth'])).snapshot(1)
            actual_best=min(row['actual'],key=lambda c:row['actual'][c]['local_ms']+network.transfer_ms(row['actual'][c]['wire_bytes']))
            assert actual_best==comparison['measured_best']
            assert comparison['agreement']==(actual_best==comparison['decision']['selected_candidate'])
    assert expected==report['errors']
    assert len(holdouts)==report['holdout_pairs']
    spec=json.loads((output/'spec.json').read_text())
    for source,sha in spec['source_sha256'].items():
        frozen=output/'measured-sources'/Path(source).name
        if frozen.exists():
            assert hashlib.sha256(frozen.read_bytes()).hexdigest()==sha
    for row in rows:
        assert row['measurement_scope']==MEASUREMENT_SCOPE and row['wire_bytes']==row['receipt']['wire_byte_count']
        assert row['message_sha256']==digest(row['message_ids'])
    assert set(r['message_sha256'] for r in rows if r['tag']=='train').isdisjoint(
        r['message_sha256'] for r in rows if r['tag']=='holdout')
    return dict(verified_files=len(manifest),training_samples=len(valid),holdout_pairs=report['holdout_pairs'])


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if args.verify:
        print(json.dumps(verify_output(args.output_dir)));return 0
    try:
        validate_args(args)
        refuse_active_service(args.service_url)
        path=Path(args.model).expanduser().resolve(strict=True)
        if not path.is_dir() or not all((path/f).is_file() for f in ('config.json','tokenizer_config.json','tokenizer.json')):
            raise ValueError('model must be a complete existing local snapshot')
        weights=list(path.glob('*.safetensors'))
        if not weights or sum(f.stat().st_size for f in weights)>args.max_model_gib*1024**3:
            raise ValueError('model missing or exceeds weight-file resource budget')
        index=path/'model.safetensors.index.json'
        if index.exists() and any(not (path/f).is_file() for f in set(json.loads(index.read_text())['weight_map'].values())):
            raise ValueError('incomplete model shard index')
        if args.serve_profile is not None and args.discussion_smoke:
            raise ValueError('serve-profile and discussion-smoke are separate operations')
        if args.serve_profile is not None and args.serve_seconds <= 0:
            raise ValueError('serve-profile requires positive --serve-seconds')
        args.output_dir.mkdir(parents=True,exist_ok=False)
    except (ValueError,RuntimeError,OSError) as exc:
        p.error(str(exc))
    output=args.output_dir
    write_json(output/'spec.json',{**vars(args),'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('generalized_cost.py'),Path(__file__).with_name('generalized_validation.py'),Path(__file__).with_name('incremental_auto.py'))},'serve_profile':str(args.serve_profile) if args.serve_profile else None,'model':str(path),'output_dir':str(output),'cache_dir':str(args.cache_dir) if args.cache_dir else None})
    archive=output/'measured-sources'
    archive.mkdir()
    frozen_spec=json.loads((output/'spec.json').read_text())
    for name,sha in frozen_spec['source_sha256'].items():
        body=Path(name).read_bytes()
        if hashlib.sha256(body).hexdigest()!=sha:
            raise RuntimeError('source changed while freezing calibration')
        (archive/Path(name).name).write_bytes(body)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
    if args.cache_dir:
        os.environ['HF_HOME']=str(args.cache_dir)
    started=monotonic();monitor=None
    def timeout():
        write_json(output/'timeout.json',dict(error='max-seconds exceeded',own_process_terminated=True))
        os._exit(2)
    timer=threading.Timer(args.max_seconds,timeout);timer.daemon=True;timer.start()
    try:
        with exclusive_calibration_lock():
            import torch
            import transformers
            from .model import HuggingFaceModelBackend
            from .model.huggingface import resolve_device
            from .local_qwen_resources import ResourceMonitor
            resolve_device(args.device,torch)  # reject unavailable CUDA before load
            def sample_resources():
                if sys.platform=='darwin':
                    from .local_qwen_comparison import memory
                    return memory(torch)
                # Linux: preserve RSS/swap/CUDA evidence; no invented macOS pressure.
                status=Path('/proc/self/status').read_text()
                import re
                value=lambda key:float(re.search(r'^'+key+r':\s+(\d+)',status,re.M)[1])
                return dict(monotonic=monotonic(),swap_used_mib=value('VmSwap')/1024,
                    rss_kib=value('VmRSS'),pressure_level=1,pressure_source='linux_process_rss_swap_cuda_budget',
                    cuda_allocated_bytes=torch.cuda.memory_allocated(args.device),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(args.device))
            baseline=sample_resources()
            monitor=ResourceMonitor(sample_resources,baseline,output);monitor.start()
            dtype=getattr(torch,args.dtype)
            tokenizer=transformers.AutoTokenizer.from_pretrained(path,local_files_only=True)
            model=transformers.AutoModelForCausalLM.from_pretrained(path,dtype=dtype,local_files_only=True)
            backend=HuggingFaceModelBackend(str(path),device=args.device,tokenizer=tokenizer,model=model)
            del tokenizer,model
            if backend.metadata.device_fallback_reason or backend.metadata.dtype!='torch.'+args.dtype:
                raise ValueError('actual device/dtype differs from requested')
            original=backend._timed_forward
            def check():
                monitor.check()
                if monotonic()-started>args.max_seconds:
                    raise RuntimeError('time budget exceeded')
                allocated=(torch.mps.current_allocated_memory() if backend.device=='mps' else torch.cuda.memory_allocated(backend.device))
                if allocated>args.max_device_memory_gib*1024**3:
                    raise RuntimeError('device memory budget exceeded')
            backend._timed_forward=lambda **kw:monitor.forward(original,**kw)
            check()
            if args.serve_profile:
                profile=load_cost_profile(json.loads(args.serve_profile.read_text()))
                identity={k:backend_cost_identity(backend) for k in ('sender','receiver')}
                environment={k:environment_identity(backend) for k in ('sender','receiver')}
                if not isinstance(profile,GeneralizedCostProfile) or profile.model_identity != identity or profile.environment != environment:
                    raise ValueError('model/environment/implementation mismatch for served profile')
                write_json(output/'loaded-profile.json',asdict(profile))
                from .generalized_validation import serve_dashboard
                serve_dashboard(backend,args,check)
            else:
                profile,report=calibration(backend,args,check,output)
            if args.discussion_smoke:
                from .generalized_validation import discussion_smoke
                discussion_smoke(backend,profile,args,output,check)
            write_json(output/'complete.json',dict(status='completed',elapsed_seconds=monotonic()-started,
                metadata=asdict(backend.metadata),profile_version=profile.version))
    except Exception as exc:
        write_json(output/'failure.json',dict(type=type(exc).__name__,error=str(exc),elapsed_seconds=monotonic()-started))
        raise
    finally:
        if monitor:monitor.close()
        timer.cancel()
        files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in output.iterdir() if f.suffix in ('.json','.jsonl') and f.name!='manifest.json'}
        write_json(output/'manifest.json',files)
    print(json.dumps(dict(status='server_completed') if args.serve_profile else verify_output(output)));return 0
