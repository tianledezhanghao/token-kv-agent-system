"""Real asymmetric collaboration and fixed Decode-message reception controls."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform

from .communication import HandoffChoice
from .message_diagnostics import ForwardTrace, context_view
from .message_understanding_probe import branch, initialize_question
from .model import HuggingFaceModelBackend
from .model.config import DEFAULT_MODEL_NAME
from .incremental_kv_protocol import inspect_model_contract
from .secret_sum import (PUBLIC_TASK, STAGES, SecretSumConfig, SecretSumRuntime,
                         output_evidence, score_result, score_outputs)


def run_scene(backend, config):
    contexts=[]
    runtime=None
    def event(kind,data):
        if kind in ('AGENT_TURN_STARTED','DISCUSSION_FINALIZING'):
            agent=data['agent']
            stage=STAGES.final if kind=='DISCUSSION_FINALIZING' else STAGES.for_turn(agent,1)
            contexts.append(dict(agent=agent,stage=stage,context=context_view(
                runtime.session.state(agent),question=PUBLIC_TASK,stage=stage,generation_prefix=True)))
    runtime=SecretSumRuntime(backend,config=config,event_sink=event)
    with ForwardTrace(backend) as trace:
        result=runtime.run()
    initial={n:backend.decode(ids,skip_special_tokens=False) for n,ids in runtime.initial_token_ledgers.items()}
    if config.secret_a != config.secret_b:
        # Exact numeric words; equal secrets naturally share a value, not provenance.
        import re
        for name,peer in (('A',config.secret_b),('B',config.secret_a)):
            assert str(peer) not in re.findall(r'\b\d+\b',initial[name])
    assert all(c['context']['exact_span_coverage'] and c['context']['template_ids_equal'] for c in contexts)
    if result.status=='completed':
        assert len(runtime.session.transport.records)==2
        assert len(result.turns)==2 and result.final_message is not None
        assert [h.source for h in result.handoffs]==['A','B']
        assert all(h.fallback_reason is None for h in result.handoffs)
        for h in result.handoffs:
            if h.executed_mode=='kv':
                assert h.receiver_message_prefill_count==0
                start,end=h.receiver_kv_target_span
                frame=runtime.session.state(h.target).framing.incoming(h.source,h.target)
                idx=next(i for i,c in enumerate(trace.calls) if c['input_ids']==list(frame.prefix_ids) and c['cache_length_after']==start)
                assert trace.calls[idx+1]['input_ids']==list(frame.closing_ids)
                assert trace.calls[idx+1]['cache_length_before']==end
    return dict(config=asdict(config),result=asdict(result),score=score_result(config,result),
        initial_token_ledgers=runtime.initial_token_ledgers,initial_contexts=initial,
        generation_contexts=contexts,forward_calls=trace.calls,
        wire_records=[asdict(r) for r in runtime.session.transport.records])


def no_handoff(backend,config):
    r=SecretSumRuntime(backend,config=config)
    initialize_question(r)
    contexts=[]; texts=[]; messages=[]
    for name,stage in (('A',STAGES.a_first),('B',STAGES.b_first),('A',STAGES.final)):
        contexts.append(context_view(r.session.state(name),question=PUBLIC_TASK,stage=stage,generation_prefix=True))
        message=r.session.generate_local_message(name,max_new_tokens=config.max_new_tokens,local_prompt=stage)
        messages.append(asdict(message))
        texts.append(backend.decode(message.message_token_ids,skip_special_tokens=True))
    r._verify()
    assert not r.session.transport.records
    assert not r.session.state('A').received_messages and not r.session.state('B').received_messages
    return dict(config=asdict(config),texts=texts,messages=messages,contexts=contexts,
        score=score_outputs(config,*texts,engineering_success=True,delivered=(False,False)),
        wire_bytes=0,final_cache_lengths=r._lengths())


def fixed_decode(backend,config):
    r=SecretSumRuntime(backend,config=config)
    initialize_question(r)
    sender_context=context_view(r.session.state('A'),question=PUBLIC_TASK,stage=STAGES.a_first,generation_prefix=True)
    message=r.session.generate_message('A','B',max_new_tokens=config.max_new_tokens,local_prompt=STAGES.a_first)
    snapshots={n:r.session.state(n) for n in ('A','B')}
    result=branch(r,snapshots,message=message,stage=STAGES.b_first,origin='sender_decode',
        modes=('token','kv:v0','kv:v1'),oracle=False,sender_context=sender_context)
    for row in result['modes']:
        row['evidence']=output_evidence(row.get('reply_text'),'B')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=DEFAULT_MODEL_NAME)
    parser.add_argument('--device',default='mps')
    parser.add_argument('--max-new-tokens',type=int,default=96)
    parser.add_argument('--output-dir',type=Path,default=Path('benchmark-results/secret-sum-77e'))
    args=parser.parse_args(argv)
    if args.output_dir.exists(): parser.error('choose a new output directory')
    if not 1 <= args.max_new_tokens <=128: parser.error('token limit must be 1–128')
    os.environ['HF_HUB_OFFLINE']='1'
    import torch,transformers
    torch.manual_seed(77)
    backend=HuggingFaceModelBackend(args.model,device=args.device)
    args.output_dir.mkdir(parents=True)
    def write(name,row):
        (args.output_dir/(name+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
    write('settings',dict(model_contract=asdict(inspect_model_contract(backend)),model=asdict(backend.metadata),torch=torch.__version__,transformers=transformers.__version__,
        platform=platform.platform(),seed=77,do_sample=False,repetition_penalty=1.0,
        max_new_tokens=args.max_new_tokens,public_task=PUBLIC_TASK,stages=asdict(STAGES),
        chat_template=backend.chat_tokenizer.chat_template,eos_token_ids=backend.eos_token_ids))
    rows={}
    for a,b,plan in ((17,26,('token','token')),(17,31,('token','token')),(42,19,('token','token')),
                     (17,26,('kv:v0','kv:v0')),(17,26,('kv:v1','kv:v1')),(17,26,('token','kv:v1')),
                     (17,31,('kv:v0','kv:v0')),(17,31,('kv:v1','kv:v1'))):
        name=f'{a}-{b}-'+('-'.join(plan)).replace(':','-')
        row=run_scene(backend,SecretSumConfig(secret_a=a,secret_b=b,max_new_tokens=args.max_new_tokens,
            handoff_plan=tuple(map(HandoffChoice.parse,plan))))
        rows[name]=row; write(name,row)
        print(name,json.dumps(row['score'],ensure_ascii=False),flush=True)
        for turn in row['result']['turns']: print(turn['message']['source_agent'],repr(turn['text']),flush=True)
        print('Final',repr(row['result']['final_text']),flush=True)
    for b in (26,31):
        name=f'no-handoff-17-{b}'; row=no_handoff(backend,SecretSumConfig(secret_b=b,max_new_tokens=args.max_new_tokens))
        rows[name]=row; write(name,row); print(name,repr(row['texts']),flush=True)
    fixed=fixed_decode(backend,SecretSumConfig(max_new_tokens=args.max_new_tokens))
    write('fixed-decode',fixed)
    for row in fixed['modes']: print('fixed',row['mode'],repr(row.get('reply_text')),flush=True)
    passed=all(row.get('result',{}).get('status','completed')=='completed' for row in rows.values()) and all(
        row['engineering_status']=='verified' for row in fixed['modes'])
    write('completion',dict(engineering_passed=passed,scenes=len(rows),fixed_modes=len(fixed['modes'])))
    return int(not passed)
