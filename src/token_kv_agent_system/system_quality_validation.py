"""Opt-in 8.3C quality study over unchanged production Session and runtimes.

Only this exclusive research process installs temporary evidence wrappers.
Never import/run the CLI concurrently with another model in this process.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter
from unittest.mock import patch

from .communication import CommunicationMode, HandoffChoice
from .discussion import DiscussionConfig, DiscussionRuntime
from .incremental_auto import (IncrementalAutoConfig, IncrementalCostProfile,
    IncrementalSelector, backend_cost_identity, workload_identity)
from .incremental_kv import prepare_incremental_message
from .incremental_kv_protocol import inspect_model_contract, encode_incremental_message, canonical_json
from .local_qwen_comparison import REVISIONS, snapshot_path, file_hash, memory
from .local_qwen_resources import ResourceConfig, ResourceMonitor, classify_failure
from .message_diagnostics import ForwardTrace, cache_fingerprint, cache_shapes
from .model.state_cache import require_disjoint_caches
from .network.handoff_network import HandoffNetwork, HandoffNetworkConfig
from .secret_sum import SecretSumConfig, SecretSumRuntime, score_result
from .system_quality_tasks import (PAIRS, PILOTS, SECRETS, DISCUSSIONS, VARIANTS,
    PAIR_SYSTEM_A, PAIR_SYSTEM_B, PUBLIC, SEND_STAGE, score_pair, freeze_tasks)
from .transport import InMemoryByteTransport

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def write_once(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def source_hashes():
    paths = sorted((ROOT/'src').rglob('*.py')) + sorted((ROOT/'tests').glob('*.py'))
    paths += [ROOT/'run_system_quality_validation.py']
    return {str(p.relative_to(ROOT)):file_hash(p) for p in paths}


def frozen_spec(phase):
    from .discussion_prompts import resolve_prompts
    from .secret_sum import STAGES, PUBLIC_TASK, private_prompt
    path=snapshot_path('05b')
    # Files and template fixed BEFORE model load, not inferred from model results.
    tokenizer_config=json.loads((path/'tokenizer_config.json').read_text())
    return dict(task='8.3C',phase=phase,head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        taskset=freeze_tasks(),source_sha256=source_hashes(),model=REVISIONS['05b'],checkpoint=str(path),
        checkpoint_files={p.name:file_hash(p) for p in path.iterdir() if p.is_file()},
        chat_template=tokenizer_config.get('chat_template'),default_prompts=resolve_prompts().to_dict(),
        secret_prompts=dict(public=PUBLIC_TASK,stages=asdict(STAGES),systems={f'{a}-{b}':[private_prompt('A',a),private_prompt('B',b)] for a,b in SECRETS}),
        generation=dict(device='mps',dtype='bfloat16',batch=1,seed=77,sampling='raw argmax greedy; no penalties',
            paired_budget=96,secret_budget=96,discussion_budget=256,rounds=2,summarizer='A',context_budget=4096),
        repair=dict(v0='raw',v1='original RoPE',v2=dict(recompute_tokens=1),v3='unchanged first ASCII numeric slot'),
        network=dict(mode='trace',trace_mbps=[20,10000,50,1000],base_latency_ms=2,physical='not_measured'),
        resource_config=asdict(ResourceConfig()),stopping=freeze_tasks()['stop'])


def state_evidence(state):
    state.validate()
    return dict(agent=state.agent_id,system=state.system_prompt,ledger=state.token_ledger,
        cached_tokens=state.cached_token_count,pending=state.pending_token_id,
        assistant_open=state.assistant_open,version=state.version,cache_sha256=cache_fingerprint(state.cache),
        cache_shapes=cache_shapes(state.cache),local_input_spans=state.local_input_spans,
        sent=[asdict(m) for m in state.sent_messages],received=[asdict(m) for m in state.received_messages])


class EvidenceTransport(InMemoryByteTransport):
    def __init__(self):
        super().__init__();self.packets=[]
    def transfer(self,data,**kw):
        assert type(data) is bytes
        delivered=super().transfer(data,**kw)
        self.packets.append(dict(**kw,wire_bytes=len(data),magic=data[:4].decode('ascii'),
            sha256=hashlib.sha256(data).hexdigest(),delivered_sha256=hashlib.sha256(delivered).hexdigest()))
        return delivered


def choice_for(variant):
    return HandoffChoice.parse(variant,**(dict(recompute_tokens=1) if variant=='kv:v2' else {}))


def deliver(session, variant, source='A', target='B'):
    c=choice_for(variant)
    if c.mode==CommunicationMode.TOKEN:
        return session.handoff(source,target,measure_timing=True)
    return session.handoff_kv(source,target,strategy=c.kv_strategy,recompute_tokens=c.recompute_tokens,measure_timing=True)


@contextmanager
def record_session(session, output, label):
    """Full raw IDs, actual forwarding, actual packet and committed Cache evidence."""
    b=session.state('A').backend
    generations=[];handoffs=[];paths=[]
    generation=session._generate_working
    def generate(old,target,limit,local_prompt=None,generation_budget=None):
        before=state_evidence(old)
        inputs=old.token_ledger + (old.framing.local_user_ids(local_prompt) if local_prompt is not None else ()) + old.framing.assistant.prefix_ids
        row=dict(before=before,local_prompt=local_prompt,input_ids=inputs,target=target,limit=limit)
        generations.append(row)
        try:
            with ForwardTrace(b) as trace:
                result=generation(old,target,limit,local_prompt,generation_budget)
            row.update(message=asdict(result.draft_message),text=b.decode(result.draft_message.message_token_ids,skip_special_tokens=True),
                forward_calls=trace.calls,after=state_evidence(result))
            return result
        finally:
            write_once(output/f'{label}-generation-{len(generations):03}.json',row)
    def wrap_handoff(original,variant):
        def handoff(source,target,**kw):
            before={n:session.state(n) for n in ('A','B')}
            message=before[source].draft_message
            row=dict(source=source,target=target,variant=variant,kwargs=kw,message=asdict(message),
                before={n:state_evidence(x) for n,x in before.items()},packet_start=len(session.transport.packets))
            handoffs.append(row);started=perf_counter();pstart=len(paths);trace=None
            try:
                with ForwardTrace(b) as trace:
                    receipt=original(source,target,**kw)
                b._torch.mps.synchronize() if b.device=='mps' else None
                elapsed=(perf_counter()-started)*1000  # full return includes dual commit; trace overhead included
                assert receipt.received_token_ids==message.message_token_ids
                content=message.message_token_ids[:-1] if message.termination_reason=='eos' else message.message_token_ids
                sender,receiver=session.state(source),session.state(target)
                for state in (sender,receiver):state.validate()
                require_disjoint_caches(sender.cache,receiver.cache)
                assert sender.token_ledger[slice(*message.sender_message_span)]==message.message_token_ids
                start=receipt.message.receiver_message_start
                assert receiver.token_ledger[start:start+len(content)]==content
                assert receiver.received_messages[-1]==sender.sent_messages[-1]==receipt.message
                assert receiver.pending_token_id is sender.pending_token_id is None
                assert receiver.cached_token_count==receipt.receiver_cached_count_after
                assert len(receiver.token_ledger)==receiver.cached_token_count
                packets=session.transport.packets[row['packet_start']:]
                assert len(packets)==1 and packets[0]['wire_bytes']==receipt.wire_byte_count
                assert packets[0]['magic']==('TKVT' if variant=='token' else 'TKVI')
                # Old committed snapshots must survive every successful transaction.
                assert all(cache_fingerprint(before[n].cache)==row['before'][n]['cache_sha256'] for n in before)
                row.update(engineering=True,receipt=asdict(receipt),full_handoff_ms=elapsed,forward_calls=trace.calls,
                    after={n:state_evidence(session.state(n)) for n in before},packets=packets,
                    transfer_paths=paths[pstart:],ids_exact=True,ledgers_exact=True,independent_caches=True)
                return receipt
            except Exception as exc:
                # An attempted wire send is not a receipt. Preserve the exact failure.
                row.update(engineering=False,error=f'{type(exc).__name__}: {exc}',
                    failure_category=classify_failure(exc),packets=session.transport.packets[row['packet_start']:],
                    forward_calls=trace.calls if trace else [],transfer_paths=paths[pstart:],
                    rollback_verified=all(session.state(n) is before[n] and cache_fingerprint(before[n].cache)==row['before'][n]['cache_sha256'] for n in before))
                raise
            finally:
                write_once(output/f'{label}-handoff-{len(handoffs):03}.json',row)
        return handoff
    from .model import kv_adapter
    original_path=kv_adapter._transfer_path
    def transfer_path(*a,**kw):
        path=original_path(*a,**kw);paths.append(dict(arguments=list(a),kwargs=kw,executed=path));return path
    with patch.object(session,'_generate_working',generate), \
         patch.object(session,'handoff',wrap_handoff(session.handoff,'token')), \
         patch.object(session,'handoff_kv',wrap_handoff(session.handoff_kv,'kv')), \
         patch.object(kv_adapter,'_transfer_path',transfer_path):
        yield dict(generations=generations,handoffs=handoffs)


def pair_session(backend,task):
    runtime=DiscussionRuntime(backend,config=DiscussionConfig(question=PUBLIC,rounds=1,max_new_tokens=96,
        agent_a_system_prompt=PAIR_SYSTEM_A,agent_b_system_prompt=PAIR_SYSTEM_B),transport=EvidenceTransport())
    for name in ('A','B'):runtime.session.append_local_user(name,PUBLIC)
    runtime.session.append_local_user('A',task.private)
    # Assert isolation on actual model input ledger, not merely config strings.
    initial_b=backend.decode(runtime.session.state('B').token_ledger,skip_special_tokens=False)
    assert task.private not in initial_b and all(not __import__('re').search(r'(?<![A-Za-z0-9])'+__import__('re').escape(x)+r'(?![A-Za-z0-9])',initial_b,__import__('re').I) for x in task.extraction)
    return runtime.session


def paired(backend,contract,task,output):
    s=pair_session(backend,task);s._kv_contract=contract
    with record_session(s,output,task.name+'-source') as evidence:
        m=s.generate_message('A','B',max_new_tokens=96,local_prompt=SEND_STAGE)
    source_text=backend.decode(m.message_token_ids,skip_special_tokens=True)
    snapshots=dict(s._states)
    hashes={n:cache_fingerprint(x.cache) for n,x in snapshots.items()}
    envelope,_=prepare_incremental_message(snapshots['A'],m,contract=contract)
    packet=encode_incremental_message(envelope)
    (output/(task.name+'-source.tkvi')).write_bytes(packet)
    source=dict(task=asdict(task),sender_text=source_text,message=asdict(m),source_packet_sha256=hashlib.sha256(packet).hexdigest(),
        initial={n:state_evidence(x) for n,x in snapshots.items()},generation=evidence,
        workload_domain=workload_identity(snapshots['A'],snapshots['B'],m))
    write_once(output/(task.name+'-source.json'),source)
    rows=[];input_ids=None
    for variant in VARIANTS:
        s._states=snapshots.copy();s.transport=EvidenceTransport()
        row=dict(task=task.name,category=task.category,variant=variant,snapshot_hashes=hashes,engineering=False)
        with record_session(s,output,task.name+'-'+variant.replace(':','-')) as evidence:
            try:
                receipt=deliver(s,variant)
                row.update(engineering=True,receipt=asdict(receipt))
                if variant!='token':assert s.transport.packets[-1]['sha256']==source['source_packet_sha256']
                received=s.state('B')
                reply=s.generate_message('B','A',max_new_tokens=96,local_prompt=task.question)
                actual_input=evidence['generations'][-1]['input_ids']
                if input_ids is None:input_ids=actual_input
                assert actual_input==input_ids,'paired receiver generation IDs differ'
                text=backend.decode(reply.message_token_ids,skip_special_tokens=True)
                row.update(engineering=True,receipt=asdict(receipt),reply=asdict(reply),reply_text=text,
                    score=score_pair(task,source_text,text,delivered=True),receiver_before_decode=state_evidence(received))
            except Exception as exc:
                if classify_failure(exc) in ('oom','resource_policy'):raise
                row.update(error=f'{type(exc).__name__}: {exc}',failure_category=classify_failure(exc),
                    applicability='unsupported_no_numeric_slot' if 'ASCII numeric' in str(exc) else 'failed',
                    score=score_pair(task,source_text,None,delivered=False))
        assert all(cache_fingerprint(snapshots[n].cache)==hashes[n] for n in snapshots)
        row.update(evidence=evidence,snapshots_unchanged=True)
        rows.append(row);write_once(output/(task.name+'-'+variant.replace(':','-')+'.json'),row)
        print('PAIR',task.name,variant,row.get('score'),row.get('reply_text',row.get('error')),flush=True)
    return rows


def runtime_case(backend,kind,task,variant,output,contract):
    events=[];label=f'{kind}-{task[0]}-{variant.replace(":","-")}' if kind=='secret' else f'discussion-{task["name"]}-{variant.replace(":","-")}'
    network=HandoffNetworkConfig(mode='trace',trace_mbps=(20,10000,50,1000),base_latency_ms=2)
    kwargs=dict(auto=IncrementalAutoConfig() if variant=='auto' else None,
        handoff_choice=HandoffChoice() if variant=='auto' else choice_for(variant),network=network)
    if kind=='secret':
        config=SecretSumConfig(secret_a=task[0],secret_b=task[1],**kwargs)
        label=f'secret-{task[0]}-{task[1]}-{variant.replace(":","-")}'
        r=SecretSumRuntime(backend,config=config,transport=EvidenceTransport())
    else:
        config=DiscussionConfig(question=task['question'],rounds=2,max_new_tokens=256,
            agent_a_system_prompt=task.get('a'),agent_b_system_prompt=task.get('b'),**kwargs)
        r=DiscussionRuntime(backend,config=config,transport=EvidenceTransport())
    # Precompute actual immutable contract; no fake startup estimate/profile.
    r.session._kv_contract=contract
    initial={n:state_evidence(r.session.state(n)) for n in ('A','B')}
    if kind=='secret':
        assert str(task[1]) not in backend.decode(r.session.state('A').token_ledger,skip_special_tokens=False)
        assert str(task[0]) not in backend.decode(r.session.state('B').token_ledger,skip_special_tokens=False)
    r.event_sink=lambda k,v:events.append(dict(kind=k,data=v))
    started=perf_counter()
    with record_session(r.session,output,label) as evidence:
        result=r.run()
    row=dict(kind=kind,task=task,variant=variant,result=asdict(result),events=events,
        initial=initial,evidence=evidence,session_ms=(perf_counter()-started)*1000,
        comparison='system outcome; messages naturally diverge after first handoff, not a paired causal estimate')
    if kind=='secret':row['historical_score']=score_result(config,result)
    write_once(output/(label+'.json'),row)
    print('RUNTIME',label,result.status,repr(result.final_text),flush=True)
    return row


def absent_controls(backend,output):
    rows=[]
    for task in (PAIRS[0],PAIRS[6]):
        s=pair_session(backend,task)
        label='absent-'+task.name
        with record_session(s,output,label) as evidence:
            m=s.generate_local_message('B',max_new_tokens=96,local_prompt=task.question)
        text=backend.decode(m.message_token_ids,skip_special_tokens=True)
        row=dict(task=task.name,origin='no_handoff',text=text,message=asdict(m),evidence=evidence,
            absent_acknowledged='unknown' in text.lower(),score=score_pair(task,'',text,delivered=False))
        write_once(output/(label+'.json'),row);rows.append(row)
    return rows


def profile_audit(backend,output):
    """Old real profiles are evidence, never widened or fitted to this study."""
    profile_file=ROOT/'docs/kv-transfer-83b/evidence/profiles.json'
    profiles=json.loads(profile_file.read_text());rows=[]
    from .kv_transfer_validation import frozen_state, verify_index
    from .performance_characterization import real_prefix
    inputs=ROOT/'benchmark-results/performance-83a'
    verify_index(inputs)
    # Replay the actual historical EOS point, with no new Sender generation.
    r=SecretSumRuntime(backend,transport=EvidenceTransport());s=r.session
    saved={n:frozen_state(inputs,'short-'+n,s.state(n)) for n in ('A','B')}
    s._states=dict(saved);m=saved['A'].draft_message
    for item in profiles:
        p=IncrementalCostProfile.from_dict(item['profile'])
        for mbps in (20,10000):
            try:
                _,decision=IncrementalSelector(IncrementalAutoConfig(policy='performance_experiment',
                    candidates=('token','kv:v0'),profile=p)).select(s.state('A'),s.state('B'),m,
                    HandoffNetwork(HandoffNetworkConfig(bandwidth_mbps=mbps)).snapshot(1),require_workload_domain=True)
                rows.append(dict(label=item['label'],mbps=mbps,decision=decision))
            except ValueError as exc:rows.append(dict(label=item['label'],mbps=mbps,rejected=str(exc)))
    row=dict(profile_source=str(profile_file.relative_to(ROOT)),source_sha256=file_hash(profile_file),
        identity={'sender':backend_cost_identity(backend),'receiver':backend_cost_identity(backend)},
        workload=workload_identity(s.state('A'),s.state('B'),m),results=rows,
        note='Historical measured values unchanged. No synthetic costs, no forced switch. In-domain replay experiment is separate.')
    write_once(output/'profile-audit.json',row)
    replays=[]
    for label in ('eos-2','pending-512'):
        if label=='pending-512':
            hc=json.loads((inputs/'frozen-spec.json').read_text())['historical_config']
            r=DiscussionRuntime(backend,config=DiscussionConfig(question=hc['question'],rounds=1,max_new_tokens=512,
                agent_a_system_prompt=hc['agent_a_system_prompt'],agent_b_system_prompt=hc['agent_b_system_prompt']),
                transport=EvidenceTransport())
            s=r.session
            saved=dict(A=real_prefix(frozen_state(inputs,'long-source-A',s.state('A')),512),
                       B=frozen_state(inputs,'history-128-B',s.state('B')))
        s._states=dict(saved)
        original=next(x['profile'] for x in profiles if x['label']==label)
        p=replace(IncrementalCostProfile.from_dict(original),version='83c-strict-replay-'+label,
            workload_domain=workload_identity(saved['A'],saved['B'],saved['A'].draft_message),
            notes=original['notes']+'; 8.3C exact replay guard only; costs unchanged; no generalized deployment claim')
        identity={'sender':backend_cost_identity(backend),'receiver':backend_cost_identity(backend)}
        assert identity==p.model_identity,'strict replay must match actual model identity'
        selector=IncrementalSelector(IncrementalAutoConfig(policy='performance_experiment',candidates=('token','kv:v0'),profile=p))
        write_once(output/(label+'-strict-profile.json'),asdict(p))
        for mbps in (20,10000):
            s._states=dict(saved);s.transport=EvidenceTransport()
            net=HandoffNetwork(HandoffNetworkConfig(bandwidth_mbps=mbps,base_latency_ms=2)).snapshot(1)
            cold=selector.select(saved['A'],saved['B'],saved['A'].draft_message,net,
                kv_contract_ready=False,require_workload_domain=True)[1]
            s._kv_contract=inspect_model_contract(backend)
            choice,decision=selector.select(saved['A'],saved['B'],saved['A'].draft_message,net,
                kv_contract_ready=True,require_workload_domain=True)
            candidate=decision['selected_candidate']
            tag=f'performance-replay-{label}-{mbps}'
            with record_session(s,output,tag) as evidence:
                receipt=deliver(s,candidate)
                assert receipt.wire_byte_count==decision['predicted_wire_bytes']
                reply=s.generate_local_message('B',max_new_tokens=16,
                    local_prompt='Briefly describe the information in the received message.')
            replays.append(dict(label=label,mbps=mbps,network=asdict(net),cold_decision=cold,
                decision=decision,receipt=asdict(receipt),reply=asdict(reply),evidence=evidence,
                nature='real Selector and selected Session execution from exact historical Decode snapshot; not full Discussion AUTO or quality certification'))
            write_once(output/(tag+'.json'),replays[-1])
        # Same sum but different A/B lengths must not pass the exact replay guard.
        shifted=replace(saved['A'],cached_token_count=saved['A'].cached_token_count+1)
        shifted_b=replace(saved['B'],cached_token_count=saved['B'].cached_token_count-1)
        try:
            selector.select(shifted,shifted_b,saved['A'].draft_message,net,
                kv_contract_ready=True,require_workload_domain=True)
        except ValueError as exc:
            write_once(output/(label+'-domain-rejection.json'),dict(error=str(exc),same_sum=True))
        else:raise AssertionError('out-of-domain exact profile accepted')
    return dict(audit=row,replays=replays)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('pilot','formal','discussion','profile'),required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args(argv)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    spec=frozen_spec(args.phase)
    write_once(args.output_dir/'frozen-spec.json',spec)
    if args.freeze_only:return 0
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OTEL_SDK_DISABLED='true')
    import torch,transformers
    from .model import HuggingFaceModelBackend
    baseline=memory(torch);write_once(args.output_dir/'preload.json',baseline)
    monitor=ResourceMonitor(lambda:memory(torch),baseline,args.output_dir)
    backend=None;original=None;allrows=[]
    try:
        monitor.start();monitor.check();torch.manual_seed(77)
        path=snapshot_path('05b')
        tokenizer=transformers.AutoTokenizer.from_pretrained(path,local_files_only=True)
        model=transformers.AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,local_files_only=True)
        backend=HuggingFaceModelBackend(str(path),device='mps',tokenizer=tokenizer,model=model)
        del model,tokenizer
        assert backend.device=='mps' and backend.metadata.dtype=='torch.bfloat16'
        original=backend._timed_forward
        backend._timed_forward=lambda **kw:monitor.forward(original,**kw)
        contract=inspect_model_contract(backend)
        write_once(args.output_dir/'settings.json',dict(metadata=asdict(backend.metadata),contract=asdict(contract),
            identity=backend_cost_identity(backend),torch=torch.__version__,transformers=transformers.__version__,
            template_sha256=hashlib.sha256(backend.chat_tokenizer.chat_template.encode()).hexdigest(),
            note='instrumented single-run quality evidence; timing includes evidence overhead; not a speed benchmark'))
        if args.phase in ('pilot','formal'):
            for task in PILOTS if args.phase=='pilot' else PAIRS:
                monitor.check();allrows.extend(paired(backend,contract,task,args.output_dir))
                gc.collect();torch.mps.empty_cache()
            if args.phase=='formal':
                allrows.extend(absent_controls(backend,args.output_dir))
                for task in SECRETS:
                    for variant in VARIANTS:
                        monitor.check();allrows.append(runtime_case(backend,'secret',task,variant,args.output_dir,contract))
                        gc.collect();torch.mps.empty_cache()
        elif args.phase=='discussion':
            for task in DISCUSSIONS:
                for variant in (*VARIANTS,'auto'):
                    monitor.check();allrows.append(runtime_case(backend,'discussion',task,variant,args.output_dir,contract))
                    gc.collect();torch.mps.empty_cache()
        else:profile_audit(backend,args.output_dir)
        write_once(args.output_dir/'completion.json',dict(status='executed',rows=len(allrows),forward_count=len(monitor.forwards),
            resource_stop=monitor.stop_reason,last_resource=monitor.sample()))
        return 0
    except Exception as exc:
        write_once(args.output_dir/'failure.json',dict(error=f'{type(exc).__name__}: {exc}',category=classify_failure(exc),
            forward_count=len(monitor.forwards),resource_stop=monitor.stop_reason))
        raise
    finally:
        if original is not None:backend._timed_forward=original
        monitor.close()
        index=[dict(path=str(p.relative_to(args.output_dir)),bytes=p.stat().st_size,sha256=file_hash(p))
            for p in sorted(args.output_dir.rglob('*')) if p.is_file()]
        write_once(args.output_dir/'sha256-index.json',index)
