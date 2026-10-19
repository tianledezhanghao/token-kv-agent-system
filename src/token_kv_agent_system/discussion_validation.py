"""Offline Qwen acceptance: all manual modes plus a fixed-state single handoff.

Records every output. Expected arithmetic is never supplied to generation.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
import os
from pathlib import Path

from .communication import HandoffChoice, CommunicationMode
from .discussion import DiscussionConfig, DiscussionRuntime
from .discussion_demo import print_result
from .incremental_kv import prepare_incremental_message
from .incremental_kv_protocol import inspect_model_contract
from .model import HuggingFaceModelBackend
from .model.config import DEFAULT_MODEL_NAME


def fingerprint(cache):
    import torch

    h = hashlib.sha256()
    for layer in cache.layers:
        for tensor in (layer.keys, layer.values):
            h.update(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def fixed_step(backend, *, question, max_new_tokens):
    runtime = DiscussionRuntime(backend, config=DiscussionConfig(
        question=question, max_new_tokens=max_new_tokens))
    s = runtime.session
    for n in ('A','B'): s.append_local_user(n, question)
    sender_message = s.generate_message('A','B',max_new_tokens=max_new_tokens,
        local_prompt=runtime.prompt_config.stages.a_first)
    snapshots = {n:s.state(n) for n in ('A','B')}
    hashes = {n:fingerprint(snapshots[n].cache) for n in snapshots}
    contract = inspect_model_contract(backend)
    envelope, _ = prepare_incremental_message(snapshots['A'], sender_message, contract=contract)
    delta_hash = hashlib.sha256(b''.join(t.data for layer in envelope.delta.layers
                                         for t in (layer.key,layer.value))).hexdigest()
    rows=[]
    original = backend.continue_from_cache
    calls=[]
    logits=[]
    capture_position=None
    def traced(ids, **kwargs):
        calls.append((tuple(ids),kwargs['logical_cached_token_count']))
        out=original(ids,**kwargs)
        if kwargs['logical_cached_token_count']==capture_position:
            logits.append(out.last_token_logits.detach().float().cpu().clone())
        return out
    backend.continue_from_cache = traced
    try:
        for mode in ('token','kv:v0','kv:v1'):
            # Validation only: reuse the SAME immutable effective state snapshots.
            # All runtime mutations occur on work clones; no clean Prefill oracle.
            s._states.update(snapshots)
            choice=HandoffChoice.parse(mode)
            calls.clear()
            capture_position=(snapshots['B'].cached_token_count
                + len(snapshots['B'].framing.incoming('A','B').prefix_ids)
                + envelope.message_token_count) if choice.mode==CommunicationMode.KV else snapshots['B'].cached_token_count
            receipt=(s.handoff('A','B') if choice.mode==CommunicationMode.TOKEN else
                     s.handoff_kv('A','B',strategy=choice.kv_strategy))
            start=receipt.message.receiver_message_start
            count=receipt.message.receiver_content_token_count
            forwarded = any(pos==start for ids,pos in calls)
            if choice.mode==CommunicationMode.KV and forwarded:
                raise AssertionError('Receiver forwarded content instead of appending KV')
            handoff_calls=tuple(calls)
            assert len(logits)==len(rows)+1
            diff=(logits[-1]-logits[0]).abs()
            logit_comparison={'max_abs':diff.max().item(),'mean_abs':diff.mean().item(),
                'top1_equal':int(logits[-1].argmax())==int(logits[0].argmax())}
            after_receive=s.state('B')
            reply=s.generate_local_message('B',max_new_tokens=max_new_tokens,
                local_prompt=runtime.prompt_config.stages.b_first)
            for n in snapshots:
                assert fingerprint(snapshots[n].cache)==hashes[n]
            rows.append({'mode':mode,'receipt':asdict(receipt),
                'receiver_content_span':[start,start+count],
                'receiver_message_prefill_count':0 if choice.mode==CommunicationMode.KV else count,
                'handoff_forward_calls':handoff_calls,
                'closing_next_logits_vs_token':logit_comparison,
                'receiver_cache_after':after_receive.cached_token_count,
                'reply':asdict(reply),'reply_text':backend.decode(reply.message_token_ids,skip_special_tokens=True),
                'status':'completed','snapshot_unchanged':True})
    finally:
        backend.continue_from_cache=original
    return {'question':question,'sender_message':asdict(sender_message),
        'sender_text':backend.decode(sender_message.message_token_ids,skip_special_tokens=True),
        'sender_cache_sha256':hashes['A'],'receiver_cache_sha256':hashes['B'],
        'receiver_history_ids':snapshots['B'].token_ledger,
        'sender_delta_sha256':delta_hash,'modes':rows}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=DEFAULT_MODEL_NAME)
    parser.add_argument('--device',default='mps')
    parser.add_argument('--max-new-tokens',type=int,default=64)
    parser.add_argument('--output-dir',type=Path,default=Path('benchmark-results/discussion-76b'))
    args=parser.parse_args(argv)
    os.environ['HF_HUB_OFFLINE']='1'
    b=HuggingFaceModelBackend(args.model,device=args.device)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    failed=False
    for name,question in (('add','23 + 48 = ?'),('multiply','12 * 13 = ?')):
        for mode in ('token','v0','v1','mixed'):
            choice=HandoffChoice.parse('token' if mode in ('token','mixed') else 'kv:'+mode)
            plan=tuple(map(HandoffChoice.parse,('token','kv:v1','token','kv:v0'))) if mode=='mixed' else ()
            config=DiscussionConfig(question=question,max_new_tokens=args.max_new_tokens,
                                    handoff_choice=choice,handoff_plan=plan)
            runtime=DiscussionRuntime(b,config=config)
            result=runtime.run()
            print(f'\n==== {name} / {mode} ====')
            print_result(result)
            record={'model':asdict(b.metadata),'discussion':asdict(result),
                    'undelivered_drafts':{n:asdict(runtime.session.state(n).draft_message)
                        for n in ('A','B') if runtime.session.state(n).draft_message is not None}}
            (args.output_dir/f'{name}-{mode}.json').write_text(
                json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            failed |= result.status!='completed'
    record=fixed_step(b,question='23 + 48 = ?',max_new_tokens=args.max_new_tokens)
    (args.output_dir/'fixed-step.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print('\nFixed Sender / Receiver snapshot comparison:')
    for row in record['modes']: print(row['mode'],row['reply_text'])
    return int(failed)
