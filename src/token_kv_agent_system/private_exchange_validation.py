"""Frozen 7.8C validation; private facts enter owner System only, scoring is post-hoc."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time

from .secret_sum import SecretSumRuntime, SecretSumConfig, PUBLIC_TASK, STAGES, output_evidence
from .private_exchange_probe import cache_snapshot, STAGE_VARIANTS
from .message_diagnostics import ForwardTrace, context_view, token_injection
from .local_qwen_comparison import REVISIONS, snapshot_path, memory, file_hash
from .local_qwen_resources import ResourceConfig, ResourceMonitor, classify_failure

PROMPTS = {
    'P0': STAGES.b_first,
    'P1': STAGE_VARIANTS['S4'],
    'P2': ('You are Agent B. Read Agent A\'s number from the received message, if present; '
           'otherwise mark Agent A\'s number as unknown. Read Agent B\'s own private number '
           'from your system instructions. State both numbers with explicit ownership. '
           'If both are known, calculate their sum. If Agent A already provided its number, '
           'do not ask Agent A to provide it again. Do not invent missing messages or numbers. Answer briefly.'),
}
POSITIVES = ((17, 26), (17, 31), (13, 29), (37, 18), (42, 57), (9, 64))
NEGATIVES = tuple((17, b, n) for b in (26, 31) for n in ('N1', 'N2', 'N3'))
REPEAT = POSITIVES[:2]
EXTRA = ((13, 26),)  # Conditional Gate 4: change A only, compare 17/26; B-only already in core.
NO_NUMBER = 'I have not decided my private number yet.'
CRITERIA = {
    'B1': 'Explicitly identify received A number, or explicitly identify A as unknown when absent/non-numeric.',
    'B2': 'Explicitly identify own System number without calling it unknown.',
    'B3': 'Use both owned numbers for correct sum; no numerical total if A number unavailable.',
    'B4': 'Do not request A number again after receiving it.',
    'B5': 'No wrong ownership, fabricated peer number/message or controller attribution.',
    'B6': 'A final retains own number and uses actually delivered B number; missing B stays unknown with no numerical sum.',
    'B7': 'Real A/B Decode for positive scenes; exact TOKEN wire IDs, incremental receive, independent caches/shared weights. Controls verify only intended edges/origins.',
}


def frozen_spec():
    data = dict(model=REVISIONS['3b'], device='mps', dtype='bfloat16', batch=1,
                max_new_tokens=96, do_sample=False, max_context_tokens=4096,
                public_task=PUBLIC_TASK, original_stages=asdict(STAGES), prompts=PROMPTS,
                positives=POSITIVES, negatives=NEGATIVES, repeat=REPEAT, extra=EXTRA,
                fixture=NO_NUMBER, criteria=CRITERIA,
                stop_rule='No P3 or prompt/budget/model changes after sampling.',
                grading='Conservative objective aids plus mandatory manual semantic review; unknown is not pass.')
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return dict(spec=data, sha256=hashlib.sha256(canonical.encode()).hexdigest())


def unknown_peer(text, agent):
    return bool(re.search(rf"(?:Agent\s+{agent}(?:['’]s)?|{agent}['’]s)\s+(?:private\s+)?number\s*(?:is|=|:)\s*(?:unknown|not\s+known)", text, re.I))


def repeats_request(text):
    # Conservative English request evidence. Negated prohibition is not a request.
    return bool(re.search(r'(?:please\s+)?(?:tell\s+me|provide|share|send|give\s+me)\s+(?:your|(?:Agent\s+)?A[\'’]s)\s+(?:own\s+)?(?:private\s+)?number|what\s+is\s+your\s+(?:private\s+)?number', text, re.I)) and not bool(re.search(r'do\s+not\s+(?:ask|request)', text, re.I))


def score_texts(a, b, texts, *, received_a, received_b, engineering):
    """No model input generation here. All seven results require manual review.

    received values come from actual Sender output evidence, never inferred from
    final numerical sum. Unknown textual forms stay unknown until manual review.
    """
    ea, eb, ef = (output_evidence(t, who, allow_bare_own=(i == 0))
                  for i, (t, who) in enumerate(zip(texts, ('A', 'B', 'A'))))
    bt, ft = texts[1:]
    def equal(value, expected):
        return 'unknown' if value is None else 'yes' if value == expected else 'no'
    a_missing = received_a is None
    b_missing = received_b is None
    b1 = ('no' if eb['numbers']['A'] is not None else 'yes' if unknown_peer(bt, 'A') else 'unknown') if a_missing else equal(eb['numbers']['A'], received_a)
    b2 = equal(eb['numbers']['B'], b)
    b3 = ('no' if eb['stated_sum'] is not None or eb['equations'] else 'yes') if a_missing else equal(eb['stated_sum'], received_a + b)
    b4 = 'no' if not a_missing and repeats_request(bt) else 'yes'
    wrong = (eb['numbers']['B'] not in (None, b) or (a_missing and eb['numbers']['A'] is not None)
             or (not a_missing and eb['numbers']['A'] not in (None, received_a)))
    b5 = 'no' if wrong else 'unknown'  # Natural-language contradictions need human review.
    own = equal(ef['numbers']['A'], a)
    if b_missing:
        peer = 'no' if ef['numbers']['B'] is not None else 'yes' if unknown_peer(ft, 'B') else 'unknown'
        total = 'no' if ef['stated_sum'] is not None or ef['equations'] else 'yes'
    else:
        peer, total = equal(ef['numbers']['B'], received_b), equal(ef['stated_sum'], a + received_b)
    b6 = 'no' if 'no' in (own, peer, total) else 'yes' if {own, peer, total} == {'yes'} else 'unknown'
    criteria = dict(B1=b1, B2=b2, B3=b3, B4=b4, B5=b5, B6=b6, B7='yes' if engineering else 'no')
    return dict(criteria=criteria, evidence=dict(a_first=ea, b_reply=eb, a_final=ef),
                repeated_request=repeats_request(bt), a_sent_correct=equal(ea['numbers']['A'], a),
                strict_pass=False, manual_review_required=True,
                note='Objective aids only. B5 is not auto-approved. All criteria must be yes after semantic review.')


def run_case(backend, a, b, *, prompt='P2', control='positive', progress_sink=None):
    if prompt not in PROMPTS or control not in ('positive', 'N1', 'N2', 'N3'):
        raise ValueError('Only frozen prompt/control choices are allowed')
    r = SecretSumRuntime(backend, config=SecretSumConfig(secret_a=a, secret_b=b, max_new_tokens=96))
    session = r.session
    initial = {n: context_view(session.state(n)) for n in ('A', 'B')}
    for n in ('A', 'B'):
        session.append_local_user(n, PUBLIC_TASK)
    contexts, outputs, deliveries, timeline = [], [], [], [dict(event='initial', **cache_snapshot(session))]
    def progress():
        if progress_sink is not None:
            progress_sink(dict(initial_contexts=initial,contexts=contexts,outputs=outputs,deliveries=deliveries,cache_timeline=timeline))
    progress()
    def decode(agent, target, stage, local=False):
        view = context_view(session.state(agent), question=PUBLIC_TASK, stage=stage, generation_prefix=True)
        contexts.append(view)
        start = len(trace.calls)
        message = (session.generate_local_message(agent, max_new_tokens=96, local_prompt=stage) if local
                   else session.generate_message(agent, target, max_new_tokens=96, local_prompt=stage))
        assert tuple(view['ledger_token_ids']) == session.state(agent).token_ledger[:message.sender_message_span[0]]
        outputs.append(dict(message=asdict(message), text=backend.decode(message.message_token_ids, skip_special_tokens=True),
                            origin='sender_decode', forward_calls=trace.calls[start:]))
        progress()
        return message
    def deliver(message, origin):
        source, target = message.source_agent, message.target_agent
        old = session.state(target)
        before = context_view(old)
        start = len(trace.calls)
        receipt = session.handoff(source, target)
        assert receipt.received_token_ids == message.message_token_ids
        handoff_calls = trace.calls[start:]
        calls = [c for c in handoff_calls if c['input_ids'] == list(receipt.receiver_appended_ids)
                 and c['cache_length_before'] == receipt.receiver_cached_count_before]
        assert len(calls) == 1
        after = context_view(session.state(target))
        # Captured Receiver-only calls concatenate exactly the newly appended ledger.
        new_ids = tuple(i for call in calls for i in call['input_ids'])
        assert tuple(after['ledger_token_ids']) == tuple(before['ledger_token_ids']) + new_ids
        assert all(c['cache_length_before'] > 0 for c in calls)
        deliveries.append(dict(source=source, target=target, origin=origin, sender_message=asdict(message),
                               receipt=asdict(receipt), receiver_before=before, receiver_after=after,
                               forward_calls=calls, all_handoff_forward_calls=handoff_calls, new_prefill_ids=new_ids, token_ids_equal=True,
                               incremental_prefill_verified=True))
        timeline.append(dict(event=source+'-handoff', **cache_snapshot(session)))
        progress()
    with ForwardTrace(backend) as trace:
        if control == 'N2':
            contexts.append(context_view(session.state('A')))
            m = token_injection(session, NO_NUMBER)
            outputs.append(dict(message=asdict(m), text=NO_NUMBER, origin='fixed_fixture'))
        else:
            m = decode('A', 'B', STAGES.a_first, local=control == 'N1')
        if control != 'N1':
            deliver(m, outputs[-1]['origin'])
        m = decode('B', 'A', PROMPTS[prompt], local=control == 'N3')
        if control != 'N3':
            deliver(m, 'sender_decode')
        decode('A', 'A', STAGES.final, local=True)
    timeline.append(dict(event='final', **cache_snapshot(session)))
    a_delivery = next((d for d in deliveries if d['source'] == 'A'), None)
    b_delivery = next((d for d in deliveries if d['source'] == 'B'), None)
    if control == 'N1':
        assert not session.state('B').received_messages
    if control == 'N3':
        assert not session.state('A').received_messages
    texts = [o['text'] for o in outputs]
    received_a = output_evidence(texts[0], 'A', allow_bare_own=True)['numbers']['A'] if a_delivery else None
    received_b = output_evidence(texts[1], 'B', allow_bare_own=True)['numbers']['B'] if b_delivery else None
    assert len(deliveries) == (1 if control in ('N1','N3') else 2)
    engineering = all(d['token_ids_equal'] and d['incremental_prefill_verified'] for d in deliveries)
    return dict(prompt=prompt, control=control, secret_a=a, secret_b=b, spec_sha256=frozen_spec()['sha256'],
                initial_contexts=initial, contexts=contexts, outputs=outputs, deliveries=deliveries,
                wire_records=[asdict(x) for x in session.transport.records], cache_timeline=timeline,
                forward_calls=trace.calls, blocked_edge={'positive':None, 'N1':'A->B', 'N2':None, 'N3':'B->A'}[control],
                origin='fixed_fixture_A' if control == 'N2' else 'real_sender_decode',
                positive_real_exchange=control == 'positive',
                score=score_texts(a,b,texts,received_a=received_a,received_b=received_b,engineering=engineering))


def planned_cases(phase):
    if phase == 'core':
        return [('positive',a,b) for a,b in POSITIVES]+[(n,a,b) for a,b,n in NEGATIVES]
    if phase == 'repeat':
        return [('positive',a,b) for a,b in REPEAT]
    if phase == 'extra':
        return [('positive',a,b) for a,b in EXTRA]
    raise ValueError('unknown phase')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('core','repeat','extra'), required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    for key,v in asdict(ResourceConfig()).items():
        parser.add_argument('--'+key.replace('_','-'),type=float,default=v)
    args=parser.parse_args(argv)
    cfg=ResourceConfig(**{k:getattr(args,k) for k in asdict(ResourceConfig())})
    args.output_dir.mkdir(parents=True,exist_ok=False)
    def emit(name,row):
        (args.output_dir/(name+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
        print(name,flush=True)
    cases=planned_cases(args.phase)
    status={f'{n}-{a}-{b}':'not_run' for n,a,b in cases}
    emit('frozen-spec',frozen_spec());emit('case-status',status)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OTEL_SDK_DISABLED='true')
    import torch,transformers
    from .model import HuggingFaceModelBackend
    baseline=memory(torch)
    monitor=ResourceMonitor(lambda:memory(torch),baseline,args.output_dir,cfg)
    backend=original=None;active=None
    try:
        monitor.start();t=time.perf_counter();path=snapshot_path('3b')
        tok=transformers.AutoTokenizer.from_pretrained(path,local_files_only=True)
        model=transformers.AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,local_files_only=True)
        backend=HuggingFaceModelBackend(str(path),device='mps',tokenizer=tok,model=model);del tok,model
        torch.mps.synchronize();load=time.perf_counter()-t
        original=backend._timed_forward
        backend._timed_forward=lambda **kw:monitor.forward(original,**kw)
        emit('settings',dict(model=asdict(backend.metadata),revision=REVISIONS['3b'][1],
             config=backend._model.config.to_dict(),chat_template=backend.chat_tokenizer.chat_template,
             eos_ids=backend.eos_token_ids,torch=torch.__version__,transformers=transformers.__version__,
             preload=baseline,load_seconds=load,resource_config=asdict(cfg),spec=frozen_spec(),
             files_sha256={p.name:file_hash(p) for p in path.iterdir() if p.is_file()}))
        for control,a,b in cases:
            active=f'{control}-{a}-{b}'
            gc.collect();torch.mps.empty_cache();monitor.sample();monitor.check();t=time.perf_counter()
            row=run_case(backend,a,b,control=control,progress_sink=lambda partial:emit(active+'-progress',partial))
            row.update(elapsed_seconds=time.perf_counter()-t,memory_after=monitor.sample())
            emit(active,row);status[active]='completed';emit('case-status',status)
        emit('completion',dict(execution_completed=True,model_quality_requires_manual_review=True))
        return 0
    except Exception as exc:
        if active is not None:status[active]='execution_failed'
        emit('case-status',status)
        emit('failure',dict(type=type(exc).__name__,error=str(exc),category=classify_failure(exc),active_case=active,memory=memory(torch)))
        return 1
    finally:
        monitor.close();emit('memory-samples',dict(samples=monitor.samples,forwards=monitor.forwards))
        if backend is not None:backend._timed_forward=original
        del backend;gc.collect();torch.mps.empty_cache()


if __name__=='__main__':raise SystemExit(main())
