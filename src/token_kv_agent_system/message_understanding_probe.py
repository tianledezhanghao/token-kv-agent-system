"""7.7C controlled message understanding, context and TOKEN/KV diagnostics.

Model quality is reported separately from engineering assertions. No expected
answer enters Receiver inputs; KV sources are always real Sender Decode.
"""
import argparse
from dataclasses import asdict
import json
import hashlib
import os
from pathlib import Path
import platform

from .communication import HandoffChoice, CommunicationMode
from .discussion import DiscussionConfig, DiscussionRuntime, error_text
from .discussion_prompts import STAGE_INSTRUCTIONS
from .message_diagnostics import (ForwardTrace, cache_fingerprint, content_ids, context_view,
                                  logits_difference, tensor_difference, token_injection, token_oracle)
from .model import HuggingFaceModelBackend
from .model.config import DEFAULT_MODEL_NAME
from .token_protocol import deserialize_token_packet, serialize_token_ids

SECRETS = ('8391', '紫色袋鼠5729', 'silver otter 6832')
RECALL_QUESTION = '请准确复述 Agent A 刚刚通过通信告诉你的测试暗号。如果没有收到暗号，请明确说明。'
RECALL_STAGE = '复述最近收到的 Agent A 消息中的测试暗号。不要猜测。'
SENDER_STAGE = '把你私有上下文中的测试暗号准确告诉 Agent B。只输出暗号。'
REVIEW_CASES = (
    ('math-wrong', '23 + 48 = ?', 'Agent A says: The answer is 72.', '71'),
    ('math-correct', '23 + 48 = ?', 'Agent A says: The answer is 71.', '71'),
    ('fact-wrong', 'What is the capital of France?', 'Agent A says: The capital of France is Berlin.', 'Paris'),
    ('off-topic', '23 + 48 = ?', 'Agent A says: Bananas are yellow.', '71'),
    ('no-answer', '23 + 48 = ?', None, None),
)


def new_runtime(backend, question, limit):
    return DiscussionRuntime(backend, config=DiscussionConfig(
        question=question, max_new_tokens=limit, rounds=1))


def initialize_question(runtime):
    for name in ('A', 'B'):
        runtime.session.append_local_user(name, runtime.config.question)


def boundary_oracles(backend, initial, groups):
    """Locate first numerical/top-1 divergence using the same append grouping."""
    cache = backend.clone_cache(initial.cache)
    ledger = initial.token_ledger
    rows = []
    for label, ids in groups:
        if not ids:
            continue
        inc = backend.continue_from_cache(tuple(ids), previous_cache=cache,
                                         logical_cached_token_count=len(ledger))
        cache = inc.cache
        ledger += tuple(ids)
        full = backend.prefill(input_token_ids=ledger)
        rows.append(dict(boundary=label, position=len(ledger),
                         logits=logits_difference(full.last_token_logits, inc.last_token_logits)))
    return dict(boundaries=rows,
                first_numerical_difference=next((r['boundary'] for r in rows
                    if r['logits']['max_abs'] > 1e-6), None),
                first_top1_difference=next((r['boundary'] for r in rows
                    if not r['logits']['top1_equal']), None))


def respond(runtime, stage, *, oracle=False, initial=None, receipt=None):
    s, b = runtime.session, runtime.session.state('B').backend
    before = s.state('B')
    view = context_view(before, question=runtime.config.question, stage=stage, generation_prefix=True)
    assert view['exact_span_coverage'], 'unaccounted/overlapping context segments'
    assert view['template_ids_equal'], f"chat template differs at {view['first_input_difference']}"
    hash_before = cache_fingerprint(before.cache)
    with ForwardTrace(b, before.framing.assistant.prefix_ids) as trace:
        reply = s.generate_local_message('B', max_new_tokens=runtime.config.max_new_tokens,
                                         local_prompt=stage)
    assert trace.prefix_result is not None, 'generation prefix was not forwarded'
    actual = s.state('B').token_ledger[:reply.sender_message_span[0]]
    assert actual == tuple(view['ledger_token_ids']), 'preview differs from actual generation ledger'
    assert cache_fingerprint(before.cache) == hash_before, 'diagnostic modified previous history'
    ids, cache = trace.prefix_result
    result = dict(context=view, reply=asdict(reply),
                  reply_text=b.decode(reply.message_token_ids, skip_special_tokens=True),
                  generation_forward_calls=trace.calls,
                  final_context=context_view(s.state('B'), question=runtime.config.question),
                  history_unchanged=True)
    if oracle:
        result['token_oracle'] = token_oracle(b, view, ids, cache, reply.message_token_ids,
                                              runtime.config.max_new_tokens)
        groups = [('receive', receipt.receiver_appended_ids if receipt else ()),
                  ('stage', before.framing.local_user_ids(stage)),
                  ('assistant prefix', before.framing.assistant.prefix_ids)]
        result['token_oracle']['localization'] = boundary_oracles(b, initial or before, groups)
    return result, ids


def branch(runtime, snapshots, *, message, stage, origin, modes, expected=None, oracle=True, sender_context=None):
    """Private harness snapshot reset; production methods clone before mutation."""
    s, b = runtime.session, runtime.session.state('B').backend
    hashes = {n: cache_fingerprint(v.cache) for n,v in snapshots.items()}
    rows, reference_logits = [], None
    for mode in modes:
        phase='handoff'
        # Same B history, A draft and immutable source storage for every path.
        s._states.update(snapshots)
        try:
            if mode != 'none':
                if mode != 'token' and origin != 'sender_decode':
                    raise ValueError('artificial messages are TOKEN-only, never Decode KV')
                choice = HandoffChoice.parse(mode)
                wire_packets=[]
                transfer=s.transport.transfer
                def capture_wire(data, **kwargs):
                    wire_packets.append(data)
                    return transfer(data, **kwargs)
                s.transport.transfer=capture_wire
                try:
                    with ForwardTrace(b) as delivery_trace:
                        receipt = (s.handoff('A','B') if choice.mode == CommunicationMode.TOKEN
                                   else s.handoff_kv('A','B', strategy=choice.kv_strategy))
                finally:
                    s.transport.transfer=transfer
                assert len(wire_packets)==1
                assert len(wire_packets[0])==receipt.wire_byte_count
                content = content_ids(message, b.eos_token_ids)
                received = s.state('B')
                start = receipt.message.receiver_message_start
                end = start + receipt.message.receiver_content_token_count
                assert received.token_ledger[start:end] == content
                assert receipt.received_token_ids == message.message_token_ids
                assert receipt.wire_byte_count == s.transport.records[-1].byte_count
                if mode == 'token':
                    assert wire_packets[0] == serialize_token_ids(message.message_token_ids)
                    assert deserialize_token_packet(wire_packets[0]).token_ids == message.message_token_ids
                    assert delivery_trace.calls[-1]['input_ids'] == list(receipt.receiver_appended_ids)
                else:
                    # Actual forward spans must never intersect KV message M.
                    for call in delivery_trace.calls:
                        # Sender close is allowed; Receiver calls are prefix/close.
                        if call['input_ids'] == list(received.framing.incoming('A','B').prefix_ids):
                            assert call['cache_length_after'] == start
                    frame=received.framing.incoming('A','B')
                    prefix_index=next(i for i,c in enumerate(delivery_trace.calls)
                        if c['input_ids']==list(frame.prefix_ids) and c['cache_length_after']==start)
                    receiver_calls=delivery_trace.calls[prefix_index:prefix_index+2]
                    assert receiver_calls[0]['input_ids']==list(frame.prefix_ids)
                    assert receiver_calls[1]['input_ids']==list(frame.closing_ids)
                    assert receiver_calls[1]['cache_length_before']==end
                    assert all(c['cache_length_after'] <= start or c['cache_length_before'] >= end
                               for c in receiver_calls)
                    assert receipt.receiver_message_prefill_count == 0
            else:
                receipt, delivery_trace = None, None
            phase='receiver_generation_or_oracle'
            result, logits = respond(runtime, stage, oracle=oracle and mode in ('token','none'),
                                     initial=snapshots['B'], receipt=receipt)
            if reference_logits is None:
                reference_logits = logits
            result.update(mode=mode, origin=origin, receipt=asdict(receipt) if receipt else None,
                          handoff_forward_calls=delivery_trace.calls if delivery_trace else [],
                          wire_bytes=receipt.wire_byte_count if receipt else 0,
                          wire_sha256=hashlib.sha256(wire_packets[0]).hexdigest() if receipt else None,
                          token_raw_payload_bytes=len(serialize_token_ids(message.message_token_ids)) if message else 0,
                          token_content_payload_bytes=len(serialize_token_ids(content_ids(message,b.eos_token_ids))) if message else 0,
                          receiver_message_prefill_count=(receipt.message.receiver_content_token_count
                              if mode=='token' else 0),
                          next_logits_vs_token=logits_difference(reference_logits, logits),
                          model_assessment=dict(expected_fragment=expected,
                              contains_expected_fragment=(expected in result['reply_text'])
                                  if expected is not None else None,
                              note='Post-generation substring evidence; not a semantic correctness verdict.'),
                          engineering_status='verified')
            if expected is not None and message is not None and origin in ('sender_decode','artificial_token_fixture'):
                sender_text=b.decode(content_ids(message,b.eos_token_ids),skip_special_tokens=True)
                result['model_assessment']['sender_contains_expected']=expected in sender_text
                result['model_assessment']['receiver_recall_given_sender_success']=(
                    expected in result['reply_text']) if expected in sender_text else None
            text=result['reply_text'].lower()
            result['model_assessment'].update(
                mentions_agent_a='agent a' in text,
                claims_missing_answer=any(fragment in text for fragment in
                    ('not provided','no answer','没有收到','未提供')),
                rejects_peer=any(fragment in text for fragment in ('incorrect','wrong','错误')),
                endorses_peer=any(fragment in text for fragment in ('is correct','正确')),
                candidates_found=[secret for secret in SECRETS if secret in result['reply_text']])
            rows.append(result)
        except Exception as exc:
            rows.append(dict(mode=mode, origin=origin, engineering_status='failed', failure_stage=phase, error=error_text(exc),
                             receiver_state=context_view(s.state('B'),question=runtime.config.question)))
        finally:
            for n,v in snapshots.items():
                assert cache_fingerprint(v.cache) == hashes[n], f'{n} snapshot polluted'
    s._states.update(snapshots)
    return dict(sender_message=asdict(message) if message else None,
                sender_text=b.decode(message.message_token_ids,skip_special_tokens=True) if message else None,
                sender_generation_context=sender_context,
                sender_ledger_token_ids=snapshots['A'].token_ledger,
                sender_cached_count=snapshots['A'].cached_token_count,
                sender_pending_token_id=snapshots['A'].pending_token_id,
                snapshot_hashes=hashes, receiver_initial=context_view(snapshots['B'],question=runtime.config.question),
                modes=rows)


def run_probes(backend, limit=64, emit=lambda name, result: None):
    results = {}
    # All fixed-content controls share exactly the same initial B Cache.
    controls = new_runtime(backend, RECALL_QUESTION, limit)
    initialize_question(controls)
    initial = {n:controls.session.state(n) for n in ('A','B')}
    for secret in SECRETS:
        assert secret not in initial['B'].system_prompt + RECALL_QUESTION + RECALL_STAGE
        assert secret not in backend.decode(initial['B'].token_ledger,skip_special_tokens=False)
    def save(name, result):
        results[name] = result
        emit(name,result)
    save('recall-no-message', branch(controls,initial,message=None,stage=RECALL_STAGE,
                                    origin='no_message',modes=('none',)))
    for i,secret in enumerate(SECRETS):
        controls.session._states.update(initial)
        message = token_injection(controls.session, '测试暗号是：'+secret)
        snapshots = {n:controls.session.state(n) for n in ('A','B')}
        save(f'recall-injected-{i}',branch(controls,snapshots,message=message,stage=RECALL_STAGE,
             origin='artificial_token_fixture',modes=('token',),expected=secret))
    # Real private Sender input and actual Decode output; never repair missing secret.
    token_baseline_ok = all(row['engineering_status']=='verified'
        for record in results.values() for row in record['modes'])
    for i,secret in enumerate(SECRETS):
        r = new_runtime(backend,RECALL_QUESTION,limit)
        initialize_question(r)
        r.session.append_local_user('A','仅 Agent A 可见的私有信息：测试暗号是：'+secret)
        sender_context=context_view(r.session.state('A'),question=RECALL_QUESTION,stage=SENDER_STAGE,generation_prefix=True)
        assert secret not in backend.decode(r.session.state('B').token_ledger,skip_special_tokens=False)
        try:
            message = r.session.generate_message('A','B',max_new_tokens=limit,local_prompt=SENDER_STAGE)
        except Exception as exc:
            save(f'recall-decode-{i}',dict(sender_generation_context=sender_context,
                modes=[dict(mode='token',origin='sender_decode',engineering_status='failed',
                            failure_stage='sender_generation',error=error_text(exc))]))
            continue
        snapshots = {n:r.session.state(n) for n in ('A','B')}
        save(f'recall-decode-{i}',branch(r,snapshots,message=message,stage=RECALL_STAGE,
            origin='sender_decode',modes=('token',),expected=secret,sender_context=sender_context))
    for name,question,text,expected in REVIEW_CASES:
        r = new_runtime(backend,question,limit)
        initialize_question(r)
        message = token_injection(r.session,text) if text else None
        snapshots={n:r.session.state(n) for n in ('A','B')}
        save('review-'+name,branch(r,snapshots,message=message,stage=STAGE_INSTRUCTIONS.b_first,
             origin='artificial_reviewer_fixture' if text else 'no_message',
             modes=('token',) if text else ('none',),expected=expected))
    # Reproduce reported university case with unchanged default prompts, two rounds.
    r = DiscussionRuntime(backend,config=DiscussionConfig(question='中山大学在哪里',
                           rounds=2,max_new_tokens=limit))
    views=[]
    original=r.session.generate_message
    def capture(source,target,**kwargs):
        before=r.session.state(source)
        view=context_view(before,question=r.config.question,
                          stage=kwargs.get('local_prompt'),generation_prefix=True)
        with ForwardTrace(backend,before.framing.assistant.prefix_ids) as trace:
            message=original(source,target,**kwargs)
        record=dict(context=view,forward_calls=trace.calls)
        if source=='B':
            logits,cache=trace.prefix_result
            record['token_oracle']=token_oracle(backend,view,logits,cache,
                                                message.message_token_ids,limit)
            record['token_oracle']['localization']=boundary_oracles(backend,before,[
                ('stage',before.framing.local_user_ids(kwargs['local_prompt'])),
                ('assistant prefix',before.framing.assistant.prefix_ids)])
        views.append(record)
        return message
    r.session.generate_message=capture
    discussion=r.run()
    save('university-discussion',dict(discussion=asdict(discussion),
          generation_contexts=views,
          engineering_status='verified' if discussion.status=='completed' and all(
              v['context']['template_ids_equal'] and v['context']['exact_span_coverage'] for v in views) else 'failed'))
    # Gate KV comparison on engineered TOKEN baseline and actual recall evidence.
    real=results['recall-decode-1']
    row=real['modes'][0]
    recall_ok=row.get('model_assessment',{}).get('receiver_recall_given_sender_success') is True
    if token_baseline_ok and recall_ok:
        # Generate once for this comparison, then reset SAME snapshots for all modes.
        secret=SECRETS[1]
        r=new_runtime(backend,RECALL_QUESTION,limit)
        initialize_question(r)
        r.session.append_local_user('A','仅 Agent A 可见的私有信息：测试暗号是：'+secret)
        sender_context=context_view(r.session.state('A'),question=RECALL_QUESTION,stage=SENDER_STAGE,generation_prefix=True)
        try:
            message=r.session.generate_message('A','B',max_new_tokens=limit,local_prompt=SENDER_STAGE)
        except Exception as exc:
            save('fixed-decode-token-v0-v1',dict(sender_generation_context=sender_context,
                modes=[dict(mode='token',origin='sender_decode',engineering_status='failed',
                            failure_stage='sender_generation',error=error_text(exc))]))
            return results
        snapshots={n:r.session.state(n) for n in ('A','B')}
        result=branch(r,snapshots,message=message,stage=RECALL_STAGE,origin='sender_decode',
                     modes=('token','kv:v0','kv:v1'),expected=secret,sender_context=sender_context)
        result['gate']=dict(token_engineering_verified=token_baseline_ok,real_token_recall_verified=recall_ok)
        save('fixed-decode-token-v0-v1',result)
    else:
        save('fixed-decode-token-v0-v1',dict(status='blocked_by_token_baseline',
             token_engineering_verified=token_baseline_ok,real_token_recall_verified=recall_ok))
    return results


def cpu_precision_control(backend, results):
    """Supplemental same-weight CPU/FP32 check; primary MPS/BF16 stays unchanged."""
    from copy import deepcopy
    import torch
    model=deepcopy(backend._model).to(device='cpu',dtype=torch.float32)
    cpu=HuggingFaceModelBackend(backend.model_name,device='cpu',
        tokenizer=backend.chat_tokenizer,model=model,torch_module=torch)
    rows=[]
    for name in ('recall-decode-1','review-math-wrong'):
        record=results[name]
        row=record['modes'][0]
        initial=record['receiver_initial']
        system_end=initial['segments'][0]['end']
        context=tuple(row['context']['ledger_token_ids'])
        groups=[tuple(initial['ledger_token_ids'][system_end:]),
                tuple(row['receipt']['receiver_appended_ids'])]
        groups += [tuple(c['input_ids']) for c in row['generation_forward_calls'][:2]]
        state=cpu.prefill(input_token_ids=tuple(initial['ledger_token_ids'][:system_end]))
        ledger=state.input_token_ids
        cache=state.cache
        for ids in groups:
            out=cpu.continue_from_cache(ids,previous_cache=cache,logical_cached_token_count=len(ledger))
            ledger+=ids
            cache=out.cache
        assert ledger==context, 'precision control must compare identical primary input IDs'
        full=cpu.prefill(input_token_ids=context)
        rows.append(dict(case=name,input_ids=context,append_groups=groups,
            cache_length=len(context),logits=logits_difference(full.last_token_logits,out.last_token_logits),
            layers=[dict(layer=i,key=tensor_difference(r.keys,a.keys),value=tensor_difference(r.values,a.values))
                for i,(r,a) in enumerate(zip(full.cache.layers,cache.layers))]))
    return dict(model=asdict(cpu.metadata),same_checkpoint_weights='deepcopy of primary model, converted to FP32',
                supplemental=True,primary_device_dtype_unchanged=asdict(backend.metadata),cases=rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=DEFAULT_MODEL_NAME)
    parser.add_argument('--device',default='mps')
    parser.add_argument('--max-new-tokens',type=int,default=64)
    parser.add_argument('--cpu-fp32-control',action='store_true',
                        help='supplement primary diagnostics with same-weight CPU/FP32 numeric control')
    parser.add_argument('--output-dir',type=Path,default=Path('benchmark-results/message-understanding-77c'))
    args=parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error('--max-new-tokens must be positive')
    os.environ['HF_HUB_OFFLINE']='1'
    b=HuggingFaceModelBackend(args.model,device=args.device)
    import torch, transformers
    torch.manual_seed(77)
    settings=dict(model=asdict(b.metadata),torch=torch.__version__,transformers=transformers.__version__,
                  platform=platform.platform(),seed=77,do_sample=False,repetition_penalty=1.0,
                  max_new_tokens=args.max_new_tokens,eos_token_ids=b.eos_token_ids,
                  prompts='general-v1 unchanged',sender_stage=SENDER_STAGE,
                  recall_question=RECALL_QUESTION,recall_stage=RECALL_STAGE,
                  reviewer_stage=STAGE_INSTRUCTIONS.b_first,
                  offline=True,transport='InMemoryByteTransport, real bytes; no physical network')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    def emit(name,result):
        (args.output_dir/(name+'.json')).write_text(json.dumps(dict(settings=settings,result=result),
                                                               ensure_ascii=False,indent=2)+'\n')
        print(name,flush=True)
        for row in result.get('modes',[]):
            print(row['mode'],row['engineering_status'],row.get('reply_text',row.get('error')),flush=True)
    results=run_probes(b,args.max_new_tokens,emit)
    if args.cpu_fp32_control:
        control=cpu_precision_control(b,results)
        emit('cpu-fp32-control',control)
    failed=any(row['engineering_status']!='verified' for r in results.values() for row in r.get('modes',[]))
    failed |= results['university-discussion']['engineering_status']!='verified'
    failed |= results['fixed-decode-token-v0-v1'].get('status')=='blocked_by_token_baseline'
    (args.output_dir/'settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2)+'\n')
    print('Engineering failures:',failed,'; model quality is assessed separately.',flush=True)
    return int(failed)
