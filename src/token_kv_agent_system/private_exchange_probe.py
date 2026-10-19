"""7.8B independent source/compute diagnostics. No production behavior changes."""
import argparse
from dataclasses import asdict, replace
import gc
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from .autogen_style_reference import generate, group_incoming, Event, template_ids
from .local_qwen_comparison import snapshot_path, memory, file_hash, REVISIONS
from .local_qwen_resources import ResourceConfig, ResourceMonitor, classify_failure
from .message_diagnostics import (ForwardTrace, context_view, token_injection,
                                 greedy_from_prefill, logits_difference, tensor_difference, cache_shapes)
from .message_understanding_probe import initialize_question
from .secret_sum import SecretSumRuntime, SecretSumConfig, PUBLIC_TASK, STAGES, private_prompt, output_evidence

FIXTURES = {
 'original':'My private number is 17. Please tell me your private number.',
 'no_request':'My private number is 17.',
 'identity':"I am Agent A. Agent A's private number is 17.",
 'no_pronouns':'Agent A private number: 17.',
 'system_reminder':"Agent A's private number is 17. Agent B should use Agent B's own private number from its system instructions.",
 'explicit_request':"Agent A's number is 17. Agent B, please state your own private number and calculate the sum.",
 'bare':'17'}
STAGE_VARIANTS = {
 'S0':None, 'S1':STAGES.b_first,
 'S2':"State Agent A's number, your own number, and their sum.",
 'S3':"You are Agent B. Agent A provided its number in the received message. Your own number is defined in your system instructions. State both with correct ownership and calculate the sum.",
 'S4':"Only mark Agent A's number as unknown if Agent A did not provide it. Your own private number must be read from your system instructions."}
OWNER_STAGE=STAGES.b_first.replace('your own number', "Agent B's private number from its system instructions")
# Frozen before results: only delete the request fragment in A first Stage.
A_NO_REQUEST='Tell Agent B your private number. Do not calculate yet.'


def cache_snapshot(session):
 from .model.cache_inspection import inspect_kv_cache
 from .model.state_cache import require_disjoint_caches
 a,b=session.state("A"),session.state("B")
 require_disjoint_caches(a.cache,b.cache)
 assert a.backend is b.backend
 return dict(shared_backend=True,independent_caches_verified=True,agents={
  n:asdict(inspect_kv_cache(session.state(n).cache,session.state(n).cached_token_count)) for n in ("A","B")})


def score_b(text,b):
 e=output_evidence(text,'B')
 return dict(evidence=e, a_ownership=e['numbers']['A']==17 if e['numbers']['A'] is not None else None,
             b_ownership=e['numbers']['B']==b if e['numbers']['B'] is not None else None,
             arithmetic=e['stated_sum']==17+b if e['stated_sum'] is not None else None,
             strict_pass=e['numbers']=={'A':17,'B':b} and e['stated_sum']==17+b,
             human_review_required=True)


def prepare(backend,b,fixture=None,real_sender=False,public=PUBLIC_TASK):
 r=SecretSumRuntime(backend,config=SecretSumConfig(secret_b=b,max_new_tokens=96))
 # Controller question override is local to diagnostic only; no number injection.
 for n in ('A','B'):r.session.append_local_user(n,public)
 initial={n:context_view(r.session.state(n)) for n in ('A','B')}
 caches=cache_snapshot(r.session)
 receipt=message=None
 with ForwardTrace(backend) as delivery:
  if real_sender:
   message=r.session.generate_message('A','B',max_new_tokens=96,local_prompt=STAGES.a_first)
  elif fixture is not None:message=token_injection(r.session,fixture)
  if message is not None:receipt=r.session.handoff('A','B')
 if receipt is not None:
  assert receipt.received_token_ids==message.message_token_ids
 return r,dict(initial_contexts=initial,initial_caches=caches,after_delivery_caches=cache_snapshot(r.session),sender=asdict(message) if message else None,
   receipt=asdict(receipt) if receipt else None,delivery_forward_calls=delivery.calls,
   origin='sender_decode' if real_sender else 'fixed_fixture' if fixture is not None else 'no_message',
   wire_records=[asdict(x) for x in r.session.transport.records])


def top5(logits):
 v,i=logits.detach().float().cpu().topk(5)
 return dict(ids=i.tolist(),scores=v.tolist(),margin=(v[0]-v[1]).item())


def first_logits(full,inc):
 r=full.detach().float().cpu();a=inc.detach().float().cpu()
 mask=r.abs()>=1.0
 relative=((r-a).abs()[mask]/r.abs()[mask]).max().item() if mask.any() else None
 return dict(**logits_difference(full,inc),max_relative_abs_reference_ge_1=relative,
             relative_eligible_count=int(mask.sum().item()),full_top5=top5(full),incremental_top5=top5(inc))


def reply(r,stage,*,paired=False):
 b=r.session.state('B').backend;state=r.session.state('B')
 view=context_view(state,question=PUBLIC_TASK,stage=stage,generation_prefix=True)
 with ForwardTrace(b,state.framing.assistant.prefix_ids) as trace:
  m=r.session.generate_local_message('B',max_new_tokens=96,local_prompt=stage)
 assert tuple(view['ledger_token_ids'])==r.session.state('B').token_ledger[:m.sender_message_span[0]]
 record=dict(context=view,input_ids=view['ledger_token_ids'],output_ids=m.message_token_ids,
  text=b.decode(m.message_token_ids,skip_special_tokens=True),stop_reason=m.termination_reason,
  eos_ids=b.eos_token_ids,stage=stage,prefill='production_TOKEN_incremental',
  generation_forward_calls=trace.calls,final_context=context_view(r.session.state('B')))
 if paired:
  t_logits,t_cache=trace.prefix_result
  f_ids=tuple(view['template_token_ids']);actual=tuple(view['ledger_token_ids'])
  with ForwardTrace(b) as ft:full=b.prefill(input_token_ids=f_ids)
  first=first_logits(full.last_token_logits,t_logits)
  f_output=greedy_from_prefill(b,full,96)
  fork=next((i for i,(x,y) in enumerate(zip(f_output,m.message_token_ids)) if x!=y),None)
  if fork is None and len(f_output)!=len(m.message_token_ids):fork=min(len(f_output),len(m.message_token_ids))
  fork_metrics=None
  if actual==f_ids and fork is not None and fork<min(len(f_output),len(m.message_token_ids)):
   fc,tc=b.clone_cache(full.cache),b.clone_cache(t_cache)
   fl,tl=full.last_token_logits,t_logits;count=len(actual)
   for token in m.message_token_ids[:fork]:
    fo=b.continue_from_cache((token,),previous_cache=fc,logical_cached_token_count=count)
    to=b.continue_from_cache((token,),previous_cache=tc,logical_cached_token_count=count)
    fc,tc,fl,tl=fo.cache,to.cache,fo.last_token_logits,to.last_token_logits;count+=1
   fork_metrics=dict(index=fork,common_input_ids=actual+tuple(m.message_token_ids[:fork]),logits=first_logits(fl,tl))
  record['full_pair']=dict(input_ids=f_ids,input_ids_equal=actual==f_ids,full_forward_calls=ft.calls,
   logical_messages=view['logical_messages'],output_ids=f_output,text=b.decode(f_output,skip_special_tokens=True),
   stop_reason='eos' if f_output[-1] in b.eos_token_ids else 'max_new_tokens',first_logits=first,
   output_ids_equal=tuple(f_output)==m.message_token_ids,first_fork=fork_metrics,
   t_cache_shapes=cache_shapes(t_cache),f_cache_shapes=cache_shapes(full.cache),
   kv_differences=[dict(layer=i,key=tensor_difference(f.keys,t.keys),value=tensor_difference(f.values,t.values))
                   for i,(f,t) in enumerate(zip(full.cache.layers,t_cache.layers))] if actual==f_ids else None,
   first_input_difference=view['first_input_difference'])
 return record


def b_case(backend,b,fixture,stage=STAGES.b_first,*,paired=False,real=False,public=PUBLIC_TASK):
 r,delivery=prepare(backend,b,fixture,real_sender=real,public=public)
 row=reply(r,stage,paired=paired)
 row.update(delivery=delivery,secret_b=b,fixture=fixture,public_task=public,score=score_b(row['text'],b))
 if 'full_pair' in row:row['full_pair']['score']=score_b(row['full_pair']['text'],b)
 return row


def own_messages(b,condition):
 system=private_prompt('B',b)
 if condition=='A':return [dict(role='system',content=f'You are Agent B. Your private number is {b}.'),dict(role='user',content='What is your private number?')]
 query={'A1':'What is your private number?','A2':'What is your private number?',
        'B':'State your own private number.','C':STAGES.b_first,
        'D':'You are Agent B. State the private number assigned to you in your system instructions.'}[condition]
 rows=[dict(role='system',content=system)]
 if condition!='A1':rows.append(dict(role='user',content=PUBLIC_TASK))
 return rows+[dict(role='user',content=query)]


def structural_messages(b,fixture,structure):
 rows=[dict(role='system',content=private_prompt('B',b))]
 events=[Event('user',PUBLIC_TASK,'user_task'),Event('A',fixture,'fixed_fixture')]
 if structure in ('A','D'):return rows+group_incoming(structure,'B',events,STAGES.b_first)
 rows.append(dict(role='user',content=PUBLIC_TASK))
 label='Message sent by Agent A to Agent B:\n' if structure=='B' else 'Message from A to B:\n'
 if structure=='B':return rows+[dict(role='user',content=label+fixture),dict(role='user',content=STAGES.b_first)]
 if structure=='C':return rows+[dict(role='user',content=label+fixture+'\n\n'+STAGES.b_first)]
 raise ValueError('unknown structure')


def segment_control(backend,ids,cuts):
 ids=tuple(ids);cuts=tuple(cuts)
 if not cuts or sorted(set(cuts))!=list(cuts) or cuts[-1]!=len(ids) or cuts[0]<=0:raise ValueError('invalid boundaries')
 with ForwardTrace(backend) as trace:
  first=backend.prefill(input_token_ids=ids[:cuts[0]])
  cache=first.cache;logits=first.last_token_logits;pred=first.first_token_id;count=cuts[0]
  for end in cuts[1:]:
   out=backend.continue_from_cache(ids[count:end],previous_cache=cache,logical_cached_token_count=count)
   cache,logits,pred,count=out.cache,out.last_token_logits,out.predicted_next_token_id,end
  output=greedy_from_prefill(backend,SimpleNamespace(cache=cache,logical_cached_token_count=count,first_token_id=pred),96)
 return dict(input_ids=ids,cuts=cuts,first_top5=top5(logits),output_ids=output,
             text=backend.decode(output,skip_special_tokens=True),stop_reason='eos' if output[-1] in backend.eos_token_ids else 'max_new_tokens',forward_calls=trace.calls)


def candidate_scene(backend,b,candidate,*,negative=False):
 if candidate not in ('A_no_request','S2','S3','S4','S1_owner'):raise ValueError('prespecified candidate required')
 r=SecretSumRuntime(backend,config=SecretSumConfig(secret_b=b,max_new_tokens=96))
 stages=replace(STAGES,a_first=A_NO_REQUEST) if candidate=='A_no_request' else replace(STAGES,b_first=OWNER_STAGE if candidate=='S1_owner' else STAGE_VARIANTS[candidate])
 r._prompt_config=replace(r.prompt_config,stages=stages,version='78b-diagnostic-'+candidate)
 contexts=[];outputs=[];receipts=[];cache_timeline=[]
 initialize_question(r)
 cache_timeline.append(dict(event='initial',**cache_snapshot(r.session)))
 with ForwardTrace(backend) as trace:
  for agent,target,stage in (('A','B',stages.a_first),('B','A',stages.b_first)):
   view=context_view(r.session.state(agent),question=PUBLIC_TASK,stage=stage,generation_prefix=True)
   m=r.session.generate_local_message(agent,max_new_tokens=96,local_prompt=stage) if negative and agent=='A' else r.session.generate_message(agent,target,max_new_tokens=96,local_prompt=stage)
   contexts.append(view);outputs.append(dict(message=asdict(m),text=backend.decode(m.message_token_ids,skip_special_tokens=True)))
   if not (negative and agent=='A'):
    receipt=r.session.handoff(agent,target);assert receipt.received_token_ids==m.message_token_ids;receipts.append(asdict(receipt))
   cache_timeline.append(dict(event=agent+'-turn',**cache_snapshot(r.session)))
  view=context_view(r.session.state('A'),question=PUBLIC_TASK,stage=stages.final,generation_prefix=True)
  m=r.session.generate_local_message('A',max_new_tokens=96,local_prompt=stages.final)
  contexts.append(view);outputs.append(dict(message=asdict(m),text=backend.decode(m.message_token_ids,skip_special_tokens=True)))
 cache_timeline.append(dict(event='final',**cache_snapshot(r.session)))
 if negative:assert not r.session.state('B').received_messages
 return dict(candidate=candidate,secret_b=b,negative=negative,origin='real_sender_decode',contexts=contexts,
  outputs=outputs,receipts=receipts,cache_timeline=cache_timeline,forward_calls=trace.calls,wire_records=[asdict(x) for x in r.session.transport.records],
  b_score=score_b(outputs[1]['text'],b),a_final_evidence=output_evidence(outputs[2]['text'],'A'),
  prompt_changes=dict(a_first=stages.a_first,b_first=stages.b_first,final=stages.final))


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',choices=REVISIONS,default='3b')
 p.add_argument('--phase',choices=('pair','own','fixtures','stage','public','structure','cross','segments','owner','candidate'),required=True)
 p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--candidate',choices=('A_no_request','S2','S3','S4','S1_owner'),default='S3')
 p.add_argument('--pair-file',type=Path)
 for key,v in asdict(ResourceConfig()).items():p.add_argument('--'+key.replace('_','-'),type=float,default=v)
 a=p.parse_args(argv);cfg=ResourceConfig(**{k:getattr(a,k) for k in asdict(ResourceConfig())})
 a.output_dir.mkdir(parents=True,exist_ok=False)
 def emit(name,r):
  (a.output_dir/(name+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(name,flush=True)
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OTEL_SDK_DISABLED='true')
 import torch,transformers
 from .model import HuggingFaceModelBackend
 baseline=memory(torch);monitor=ResourceMonitor(lambda:memory(torch),baseline,a.output_dir,cfg)
 backend=original=None
 try:
  monitor.start();t=time.perf_counter();path=snapshot_path(a.model)
  tok=transformers.AutoTokenizer.from_pretrained(path,local_files_only=True)
  model=transformers.AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,local_files_only=True)
  backend=HuggingFaceModelBackend(str(path),device='mps',tokenizer=tok,model=model);del tok,model
  torch.mps.synchronize();load=time.perf_counter()-t
  original=backend._timed_forward
  backend._timed_forward=lambda **kw:monitor.forward(original,**kw)
  emit('settings',dict(repo=REVISIONS[a.model][0],revision=REVISIONS[a.model][1],model=asdict(backend.metadata),
   config=backend._model.config.to_dict(),chat_template=backend.chat_tokenizer.chat_template,
   files_sha256={f.name:file_hash(f) for f in path.iterdir() if f.is_file()},torch=torch.__version__,transformers=transformers.__version__,
   load_seconds=load,resource_config=asdict(cfg),phase=a.phase,do_sample=False,max_new_tokens=96,preload=baseline))
  def run(name,fn):
   gc.collect();torch.mps.empty_cache();monitor.sample();monitor.check();start=time.perf_counter();row=fn()
   row.update(elapsed_seconds=time.perf_counter()-start,memory_after=monitor.sample(),model=a.model)
   emit(name,row)
  if a.phase=='pair':
   for b in (26,31):run(f'real-pair-{b}',lambda b=b:b_case(backend,b,None,paired=True,real=True))
  elif a.phase=='own':
   for b in (26,31):
    for c in ('A','A1','A2','B','C','D'):
     run(f'own-{b}-{c}',lambda b=b,c=c:dict(generate(backend,own_messages(b,c),96),secret_b=b,condition=c,origin='no_message'))
  elif a.phase in ('fixtures','cross'):
   names=list(FIXTURES) if a.phase=='fixtures' else ('bare','original')
   for b in (26,31):
    for name in names:run(f'fixture-{b}-{name}',lambda b=b,n=name:b_case(backend,b,FIXTURES[n]))
  elif a.phase=='stage':
   for b in (26,31):
    for name,stage in STAGE_VARIANTS.items():run(f'stage-{b}-{name}',lambda b=b,s=stage:b_case(backend,b,FIXTURES['original'],s))
  elif a.phase=='owner':
   for b in (26,31):run(f'owner-{b}',lambda b=b:b_case(backend,b,FIXTURES['original'],OWNER_STAGE,paired=True))
  elif a.phase=='public':
   for b in (26,31):run(f'public-no-unknown-{b}',lambda b=b:b_case(backend,b,FIXTURES['original'],public=PUBLIC_TASK.split(' If ')[0]))
  elif a.phase=='structure':
   for b in (26,31):
    for name in ('A','B','C','D'):run(f'structure-{b}-{name}',lambda b=b,n=name:dict(generate(backend,structural_messages(b,FIXTURES['original'],n),96),secret_b=b,structure=n,origin='fixed_fixture',prefill='full_history_structure_control'))
  elif a.phase=='segments':
   if a.pair_file is None:raise ValueError('segments requires a saved pair-file')
   data=json.loads(a.pair_file.read_text());ids=data['input_ids'];segments=data['context']['segments']
   cuts=sorted({x['end'] for x in segments})
   run('same-ids-boundary-chunks',lambda:segment_control(backend,ids,cuts))
   run('same-ids-two-chunks',lambda:segment_control(backend,ids,[len(ids)//2,len(ids)]))
   for fixture in (None,'Agent A private number: 17.'):
    msgs=[dict(role='system',content=private_prompt('B',26)),dict(role='user',content='State your own private number.')]
    if fixture:msgs.append(dict(role='user',content='Message from A to B:\n'+fixture))
    tokens=template_ids(backend,msgs)
    run('plain-agent-control' if fixture else 'plain-no-agent-control',lambda ids=tokens:dict(segment_control(backend,ids,[len(ids)//2,len(ids)]),full=generate(backend,msgs,96)))
  else:
   for b in (26,31):
    for negative in (False,True):run(f'candidate-{a.candidate}-{b}-'+('absent' if negative else 'token'),lambda b=b,n=negative:candidate_scene(backend,b,a.candidate,negative=n))
  emit('completion',dict(execution_completed=True,quality_requires_human_review=True))
  return 0
 except Exception as exc:
  emit('failure',dict(type=type(exc).__name__,error=str(exc),failure_category=classify_failure(exc),memory=memory(torch)))
  return 1
 finally:
  monitor.close();emit('memory-samples',dict(samples=monitor.samples,forwards=monitor.forwards))
  if backend is not None:backend._timed_forward=original
  del backend;gc.collect();torch.mps.empty_cache()


if __name__=='__main__':raise SystemExit(main())
