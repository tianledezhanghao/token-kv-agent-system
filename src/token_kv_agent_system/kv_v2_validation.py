"""Frozen, bounded 7.9B same-Sender V2 study, local 0.5B MPS BF16 only."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from .communication import HandoffChoice, CommunicationMode, KVStrategy
from .discussion import DiscussionConfig, DiscussionRuntime
from .incremental_kv import prepare_incremental_message
from .incremental_kv_protocol import inspect_model_contract, encode_incremental_message
from .local_qwen_comparison import REVISIONS, snapshot_path, file_hash, memory
from .local_qwen_resources import ResourceConfig, ResourceMonitor, classify_failure
from .message_diagnostics import ForwardTrace, cache_fingerprint, cache_shapes, context_view, logits_difference
from .message_understanding_probe import initialize_question, RECALL_QUESTION, RECALL_STAGE, SENDER_STAGE
from .secret_sum import SecretSumConfig, SecretSumRuntime, STAGES, PUBLIC_TASK, score_outputs
from .transport import InMemoryByteTransport

VARIANTS = ('token','kv:v0','kv:v1','v2-k0','v2-k1','v2-k2','v2-r25','v2-full')


def choice_for(variant):
    if variant in ('token','kv:v0','kv:v1'):return HandoffChoice.parse(variant)
    return HandoffChoice.parse('kv:v2', **(dict(recompute_ratio=.25) if variant=='v2-r25'
        else dict(recompute_tokens={'v2-k0':0,'v2-k1':1,'v2-k2':2,'v2-full':4096}[variant])))


class PacketEvidence(InMemoryByteTransport):
    def __init__(self):
        super().__init__();self.packets=[]
    def transfer(self,data,**kwargs):
        assert type(data) is bytes
        self.packets.append(dict(source=kwargs['source'],target=kwargs['target'],
            wire_bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        return super().transfer(data,**kwargs)


def controlled_case(backend, *, secret_b=None, emit=lambda name,row:None):
    """Generate A once, reset immutable snapshots, real handoffs in each branch."""
    config=SecretSumConfig(secret_b=secret_b) if secret_b is not None else DiscussionConfig(
        question=RECALL_QUESTION,rounds=1,max_new_tokens=96)
    runtime=SecretSumRuntime(backend,config=config) if secret_b is not None else DiscussionRuntime(backend,config=config)
    initialize_question(runtime)
    stage_a=STAGES.a_first if secret_b is not None else SENDER_STAGE
    stage_b=STAGES.b_first if secret_b is not None else RECALL_STAGE
    if secret_b is None:runtime.session.append_local_user('A','仅 Agent A 可见的私有信息：测试暗号是：紫色袋鼠5729')
    s=runtime.session
    sender_context=context_view(s.state('A'),question=runtime.config.question,stage=stage_a,generation_prefix=True)
    with ForwardTrace(backend) as sender_trace:
        message=s.generate_message('A','B',max_new_tokens=96,local_prompt=stage_a)
    snapshots={n:s.state(n) for n in ('A','B')}
    hashes={n:cache_fingerprint(snapshots[n].cache) for n in snapshots}
    contract=inspect_model_contract(backend);s._kv_contract=contract
    envelope,export=prepare_incremental_message(snapshots['A'],message,contract=contract)
    source_packet=encode_incremental_message(envelope)
    source=dict(sender=asdict(message),sender_text=backend.decode(message.message_token_ids,skip_special_tokens=True),
        sender_context=sender_context,sender_trace=sender_trace.calls,export=asdict(export),
        source_packet_sha256=hashlib.sha256(source_packet).hexdigest(),snapshot_hashes=hashes,
        initial_receiving_context=context_view(snapshots['B']))
    emit('source',source)
    rows=[];reference_logits=None;reference_ids=None;reference_input=None
    for variant in VARIANTS:
        s._states=snapshots.copy();s.transport=PacketEvidence();choice=choice_for(variant)
        gc.collect()
        started=perf_counter()
        with ForwardTrace(backend) as handoff_trace:
            receipt=(s.handoff('A','B') if choice.mode==CommunicationMode.TOKEN else s.handoff_kv('A','B',
                strategy=choice.kv_strategy,recompute_tokens=choice.recompute_tokens,
                recompute_ratio=choice.recompute_ratio,measure_timing=True))
        handoff_ms=(perf_counter()-started)*1000
        assert receipt.received_token_ids==message.message_token_ids
        if choice.mode==CommunicationMode.KV:
            assert s.transport.packets[-1]['sha256']==source['source_packet_sha256']
        before=s.state('B');before.validate()
        view=context_view(before,question=runtime.config.question,stage=stage_b,generation_prefix=True)
        assert view['template_ids_equal'] and view['exact_span_coverage']
        if reference_input is None:reference_input=view['ledger_token_ids']
        else:assert view['ledger_token_ids']==reference_input
        generation_started=perf_counter()
        with ForwardTrace(backend,before.framing.assistant.prefix_ids) as reply_trace:
            reply=s.generate_message('B','A',max_new_tokens=96,local_prompt=stage_b)
        generation_ms=(perf_counter()-generation_started)*1000
        assert s.state('B').token_ledger[:reply.sender_message_span[0]]==tuple(view['ledger_token_ids'])
        logits,prefix_cache=reply_trace.prefix_result
        assert backend._torch.isfinite(logits).all().item()
        logits=logits.detach().float().cpu();prefix_shapes=cache_shapes(prefix_cache)
        del prefix_cache;reply_trace.prefix_result=None
        top=logits.topk(5)
        if reference_logits is None:reference_logits=logits;reference_ids=reply.message_token_ids
        common=next((i for i,(a,b) in enumerate(zip(reference_ids,reply.message_token_ids)) if a!=b),min(len(reference_ids),len(reply.message_token_ids)))
        reply_text=backend.decode(reply.message_token_ids,skip_special_tokens=True)
        row=dict(variant=variant,choice=asdict(choice),receipt=asdict(receipt),
            receiver_generation_context=view,receiver_after_handoff_shapes=cache_shapes(before.cache),
            handoff_forward_calls=handoff_trace.calls,receiver_generation_forward_calls=reply_trace.calls,
            reply=asdict(reply),reply_text=reply_text,first_logits_top5=dict(ids=top.indices.tolist(),values=top.values.tolist()),
            first_logits_vs_token=logits_difference(reference_logits,logits),
            common_output_prefix_with_token=common,first_output_divergence_with_token=(None if reply.message_token_ids==reference_ids else common),
            receiver_generation_cache_shapes=prefix_shapes,handoff_with_trace_ms=handoff_ms,receiver_stage_and_decode_ms=generation_ms,
            token_stage_breakdown='not_measured' if variant=='token' else None,
            simulated_link_ms={str(mbps):receipt.wire_byte_count*8/(mbps*1e6)*1000+5 for mbps in (20,1000)},
            physical_network='not_measured',fallback=None)
        if secret_b is not None:
            with ForwardTrace(backend) as reverse_trace:
                reverse=(s.handoff('B','A') if choice.mode==CommunicationMode.TOKEN else s.handoff_kv('B','A',
                    strategy=choice.kv_strategy,recompute_tokens=choice.recompute_tokens,
                    recompute_ratio=choice.recompute_ratio,measure_timing=True))
            final_view=context_view(s.state('A'),question=PUBLIC_TASK,stage=STAGES.final,generation_prefix=True)
            with ForwardTrace(backend) as final_trace:
                final=s.generate_local_message('A',max_new_tokens=96,local_prompt=STAGES.final)
            final_text=backend.decode(final.message_token_ids,skip_special_tokens=True)
            row.update(reverse_receipt=asdict(reverse),reverse_forward_calls=reverse_trace.calls,
                final_context=final_view,final=asdict(final),final_text=final_text,final_forward_calls=final_trace.calls,
                score=score_outputs(config,source['sender_text'],reply_text,final_text,engineering_success=True),
                reverse_comparison_note='B reply differs by branch; only A→B has identical source input')
        else:row['recall_exact_secret_present']='紫色袋鼠5729' in reply_text
        for n in snapshots:
            assert cache_fingerprint(snapshots[n].cache)==hashes[n]
            s.state(n).validate()
        row.update(packets=s.transport.packets,engineering_status='verified',snapshots_unchanged=True)
        rows.append(row);emit(variant,row)
        print(variant,repr(reply_text),repr(row.get('final_text')),flush=True)
    # k=0 must produce same execution IDs as V1; numerical BF16 tolerance need not be bitwise.
    assert rows[2]['reply']['message_token_ids']==rows[3]['reply']['message_token_ids']
    if secret_b is None:
        assert envelope.message_token_count>2,'long-message case did not satisfy 0<k<N'
    return dict(source=source,modes=rows)


def frozen_spec(output):
    from .secret_sum import private_prompt
    root=Path(__file__).resolve().parents[2]
    sources=sorted((root/'src').rglob('*.py'))+sorted((root/'tests').glob('*kv*.py'))+[Path(__file__),root/'run_kv_v2_validation.py']
    return dict(task='7.9B',algorithm='V1 + contiguous suffix real receiver forward; full KV packet',
        variants=VARIANTS,policies={v:asdict(choice_for(v)) for v in VARIANTS},
        model=REVISIONS['05b'],device='mps',dtype='bfloat16',batch=1,greedy=True,seed=77,max_new_tokens=96,
        cases=['secret-17-26','secret-17-31','77c-purple-kangaroo-long-message'],
        public_task=PUBLIC_TASK,stages=asdict(STAGES),systems={str(b):[private_prompt('A',17),private_prompt('B',b)] for b in (26,31)},
        long_case=dict(question=RECALL_QUESTION,private_input='仅 Agent A 可见的私有信息：测试暗号是：紫色袋鼠5729',
            sender_stage=SENDER_STAGE,receiver_stage=RECALL_STAGE),
        scoring=['A/B ownership','B preserves system number','B reads actual A','correct sum','no repeated request',
            'no invented source','A final integration','no contradiction'],
        output_directory=str(output.resolve()),resource_config=asdict(ResourceConfig()),
        source_sha256={str(p.relative_to(root)):file_hash(p) for p in dict.fromkeys(sources)})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('benchmark-results/kv-v2-79b'))
    args=parser.parse_args(argv)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    def emit(name,row):
        (args.output_dir/(name+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
    emit('frozen-spec',frozen_spec(args.output_dir)) # before importing/loading real model
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OTEL_SDK_DISABLED='true')
    import torch,transformers
    from .model import HuggingFaceModelBackend
    baseline=memory(torch);emit('preload',baseline)
    monitor=ResourceMonitor(lambda:memory(torch),baseline,args.output_dir)
    original=None;backend=None
    try:
        monitor.start();monitor.check()
        path=snapshot_path('05b')
        # Check completeness before any loader; cache-only, no download or CPU fallback.
        required=('config.json','tokenizer.json','tokenizer_config.json','model.safetensors')
        if not all((path/p).is_file() for p in required):raise FileNotFoundError('incomplete frozen local checkpoint')
        torch.manual_seed(77)
        tokenizer=transformers.AutoTokenizer.from_pretrained(path,local_files_only=True)
        model=transformers.AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,local_files_only=True)
        backend=HuggingFaceModelBackend(str(path),device='mps',tokenizer=tokenizer,model=model)
        del model,tokenizer
        assert backend.device=='mps' and backend.metadata.dtype=='torch.bfloat16', 'expected MPS / torch.bfloat16'
        original=backend._timed_forward
        backend._timed_forward=lambda **kw:monitor.forward(original,**kw)
        emit('settings',dict(metadata=asdict(backend.metadata),model_contract=asdict(inspect_model_contract(backend)),
            revision=REVISIONS['05b'][1],torch=torch.__version__,transformers=transformers.__version__,
            files_sha256={p.name:file_hash(p) for p in path.iterdir() if p.is_file()},
            template_sha256=hashlib.sha256(backend.chat_tokenizer.chat_template.encode()).hexdigest(),
            measurement_note='per-stage MPS sync + ForwardTrace CPU copies + ResourceMonitor; instrumented single runs, no throughput claim'))
        for name,b in (('secret-17-26',26),('secret-17-31',31),('long-message',None)):
            monitor.check()
            row=controlled_case(backend,secret_b=b,emit=lambda key,row:emit(name+'-'+key,row))
            emit(name,row)
            del row;gc.collect();torch.mps.empty_cache()
        monitor.check();emit('completion',dict(status='verified',real_cases=3,variants_per_case=8,
            forward_count=len(monitor.forwards),resource_stop=monitor.stop_reason,memory=monitor.sample()))
        return 0
    except Exception as exc:
        emit('failure',dict(status='failed',category=classify_failure(exc),error=f'{type(exc).__name__}: {exc}',
            forward_count=len(monitor.forwards),resource_stop=monitor.stop_reason))
        print(type(exc).__name__,str(exc),flush=True);return 1
    finally:
        if original is not None:backend._timed_forward=original
        monitor.close()
