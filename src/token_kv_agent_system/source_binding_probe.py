"""7.7D opt-in framing/stage factorial harness over unchanged session bytes.

Fixed artificial messages are TOKEN-only. Real Decode is generated once and
branched from immutable snapshots. Reading and reviewing never share outputs.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets

from .agent.chat_framing import ChatFraming
from .discussion_prompts import STAGE_INSTRUCTIONS
from .incremental_kv_protocol import inspect_model_contract
from .message_diagnostics import content_ids, context_view, token_injection
from .message_understanding_probe import branch, initialize_question, new_runtime
from .model import HuggingFaceModelBackend
from .model.config import DEFAULT_MODEL_NAME

VARIANTS = ('A-default', 'B-external', 'C-external-split', 'D-default-split')
TASKS = ('read', 'review', 'production_stage')
READ_TASK = 'Agent A 刚才的答案是什么？请准确复述。'
REVIEW_TASK = 'Agent A 的回答有没有解决用户原问题？有没有具体错误？'
SPLIT_READ = (
    '先识别用户原问题，再识别收到的 Agent A 消息。'
    '复述 A 的原话，不改写成你自己的答案；如果 A 未作答或没有消息，如实说明。'
)
SPLIT_REVIEW = (
    '按顺序完成两步：1. 识别用户原问题，准确引用收到的 Agent A 原话。'
    '如果 A 未作答或没有消息，如实说明。2. 独立核查这条原话是否解决原问题，'
    '指出具体错误或给出具体检查依据；不确定就说明。不要把自己的答案当成 A 的原话。'
)
SENDER_STAGE = '把你私有上下文中的测试暗号准确告诉 Agent B。只输出暗号。'


def incoming_label(source, target, external):
    return (f'Received external message\nSender: Agent {source}\nRecipient: Agent {target}\n'
            f'Agent {source}\'s verbatim message (content follows):\n') if external else (
            f'Message from {source} to {target}:\n')


class ProbeFraming(ChatFraming):
    """Harness-local label adapter; role/control boundaries remain template-owned."""
    def __init__(self, tokenizer, system_prompt, *, external):
        super().__init__(tokenizer, system_prompt)
        self.external = external

    def incoming(self, source, target):
        # Reuse production identity, history and assistant-boundary validation.
        default = super().incoming(source, target)
        if not self.external:
            return default
        label = incoming_label(source, target, True)
        frame = self._frame(self.system, 'user', label)
        self._check_closing(frame.closing_ids)
        prior = self.system + [dict(role='user', content='Earlier input.'),
                               dict(role='assistant', content='Earlier reply.')]
        if self._frame(prior, 'user', label) != frame:
            raise ValueError('research label changes with history')
        messages = prior + [dict(role='user', content=label+self._marker)]
        closed = self._render_ids(messages)
        if self._render_ids(messages, True) != closed+self.assistant.prefix_ids:
            raise ValueError('research label changes generation prefix')
        if frame.closing_ids != default.closing_ids:
            raise ValueError('research framing must preserve template closing')
        return frame


def stage_for(variant, task):
    if variant not in VARIANTS or task not in TASKS:
        raise ValueError('unknown framing variant/task')
    split = variant.endswith('-split')
    if task == 'read':
        return READ_TASK + ('\n'+SPLIT_READ if split else '')
    if task == 'review':
        return REVIEW_TASK + ('\n'+SPLIT_REVIEW if split else '')
    return STAGE_INSTRUCTIONS.b_first + ('\n'+SPLIT_REVIEW if split else '')


def framed_snapshots(snapshots, variant):
    if variant not in VARIANTS:
        raise ValueError('unknown framing variant')
    receiver = snapshots['B']
    if receiver.received_messages or receiver.assistant_open:
        raise ValueError('framing must be selected before receiving a message')
    framing = ProbeFraming(receiver.backend.chat_tokenizer, receiver.system_prompt,
                           external=variant in ('B-external', 'C-external-split'))
    assert framing.initial_ids == receiver.framing.initial_ids
    assert framing.assistant == receiver.framing.assistant
    return dict(snapshots, B=replace(receiver, framing=framing))


def fixed_cases(secret):
    if not isinstance(secret, str) or not secret or len(secret) > 100:
        raise ValueError('secret must be 1–100 ordinary characters')
    return (
        dict(name='math-correct', question='23 + 48 = ?', message='23 + 48 = 71'),
        dict(name='math-wrong', question='23 + 48 = ?', message='23 + 48 = 72'),
        dict(name='private-secret', question='请 Agent A 提供它私有上下文中的测试暗号。',
             message='测试暗号是：'+secret),
        dict(name='university-wrong', question='中山大学在哪里',
             message='中山大学位于中国广东省中山市。'),
        dict(name='explicit-no-answer', question='23 + 48 = ?', message='我还没有提供答案。'),
        dict(name='absent-message', question='23 + 48 = ?', message=None),
    )


def output_evidence(result, case):
    """Post-generation literal evidence ONLY, never a semantic quality verdict."""
    for row in result['modes']:
        text = row.get('reply_text', '')
        row['literal_evidence'] = dict(
            exact_quote_present=case['message'] in text if case['message'] else None,
            original_question_present=case['question'] in text,
            note='Substring evidence only; human assessment is required for all seven quality dimensions.')
        row['human_assessment'] = dict.fromkeys((
            'accurate_restatement', 'identifies_original_question', 'claims_no_answer',
            'specific_check_basis', 'unconditional_agreement', 'unsupported_rejection',
            'rewrites_peer_information', 'failure_attribution'))
    return result


def run_token_matrix(backend, secret, *, limit=64, repeats=3, emit=lambda name, result: None):
    records = {}
    for case in fixed_cases(secret):
        runtime = new_runtime(backend, case['question'], limit)
        initialize_question(runtime)
        message = token_injection(runtime.session, case['message']) if case['message'] else None
        snapshots = {n: runtime.session.state(n) for n in ('A', 'B')}
        if case['name'] == 'private-secret':
            assert secret not in backend.decode(snapshots['B'].token_ledger, skip_special_tokens=False)
        # Reverse run order on alternate repetitions to detect order contamination.
        for repetition in range(1, repeats+1):
            variants = VARIANTS if repetition % 2 else VARIANTS[::-1]
            for variant in variants:
                framed = framed_snapshots(snapshots, variant)
                for task in TASKS:
                    stage = stage_for(variant, task)
                    if case['name'] == 'private-secret':
                        assert secret not in stage and secret not in incoming_label('A', 'B', True)
                    result = branch(runtime, framed, message=message, stage=stage,
                                    origin='artificial_token_fixture' if message else 'no_message',
                                    modes=('token',) if message else ('none',), oracle=False)
                    result.update(case=case, variant=variant, task=task, repetition=repetition,
                                  framing=incoming_label('A', 'B', variant in ('B-external', 'C-external-split')))
                    name = f"token-{case['name']}-{variant}-{task}-r{repetition}"
                    records[name] = output_evidence(result, case)
                    emit(name, records[name])
    return records


def verified(records):
    return bool(records) and all(record['modes'] and all(
        row['engineering_status'] == 'verified' for row in record['modes'])
        for record in records.values())


def run_decode_comparison(backend, secret, *, token_records, limit=64, emit=lambda name, result: None):
    """TOKEN engineering and actual Decode recall gate precede all KV calls."""
    if not verified(token_records):
        return dict(status='blocked_by_token_engineering')
    case = fixed_cases(secret)[2]
    runtime = new_runtime(backend, case['question'], limit)
    initialize_question(runtime)
    runtime.session.append_local_user('A', '仅 Agent A 可见的私有信息：测试暗号是：'+secret)
    sender_context = context_view(runtime.session.state('A'), question=case['question'],
                                  stage=SENDER_STAGE, generation_prefix=True)
    assert secret not in backend.decode(runtime.session.state('B').token_ledger, skip_special_tokens=False)
    message = runtime.session.generate_message('A', 'B', max_new_tokens=limit, local_prompt=SENDER_STAGE)
    snapshots = {n: runtime.session.state(n) for n in ('A', 'B')}
    sender_text = backend.decode(content_ids(message, backend.eos_token_ids), skip_special_tokens=True)
    gate = branch(runtime, framed_snapshots(snapshots, 'C-external-split'), message=message,
                  stage=stage_for('C-external-split', 'read'), origin='sender_decode',
                  modes=('token',), expected=secret, oracle=False, sender_context=sender_context)
    gate.update(variant='C-external-split', task='read', case=case)
    emit('decode-token-gate', gate)
    if secret not in sender_text or not verified({'gate': gate}) or secret not in gate['modes'][0]['reply_text']:
        return dict(status='blocked_by_real_decode_recall', gate=gate)
    records = {}
    for variant in VARIANTS:
        for task in ('read', 'review'):
            result = branch(runtime, framed_snapshots(snapshots, variant), message=message,
                            stage=stage_for(variant, task), origin='sender_decode',
                            modes=('token', 'kv:v0', 'kv:v1'), expected=secret, oracle=False,
                            sender_context=sender_context)
            result.update(case=dict(case, message=sender_text), variant=variant, task=task,
                          framing=incoming_label('A', 'B', variant in ('B-external', 'C-external-split')))
            name = f'decode-{variant}-{task}'
            records[name] = output_evidence(result, dict(case, message=sender_text))
            emit(name, records[name])
    return dict(status='compared', gate=gate, records=records)


def repetition_summary(records):
    groups = {}
    for record in records.values():
        key = '/'.join((record['case']['name'], record['variant'], record['task']))
        groups.setdefault(key, []).append(record)
    return {key: dict(repetitions=len(rows),
                      identical_effective_ids=len({tuple(r['modes'][0]['context']['ledger_token_ids'])
                          for r in rows if r['modes'][0]['engineering_status']=='verified'}) == 1,
                      identical_output_ids=len({tuple(r['modes'][0]['reply']['message_token_ids'])
                          for r in rows if r['modes'][0]['engineering_status']=='verified'}) == 1,
                      engineering_verified=verified({str(i): r for i, r in enumerate(rows)}))
            for key, rows in groups.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=DEFAULT_MODEL_NAME)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--fixtures', type=Path, help='reuse a previously saved fixtures.json (same random secret)')
    parser.add_argument('--token-only', action='store_true', help='run TOKEN diagnosis without later KV comparison')
    parser.add_argument('--output-dir', type=Path, default=Path('benchmark-results/source-binding-77d'))
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0 or args.repeats <= 0:
        parser.error('token limit and repeats must be positive')
    if (args.output_dir/'settings.json').exists():
        parser.error('output directory already contains a run; use a new directory')
    secret = json.loads(args.fixtures.read_text())['secret'] if args.fixtures else '青色水獭-'+secrets.token_hex(4)
    cases = fixed_cases(secret)
    os.environ['HF_HUB_OFFLINE'] = '1'
    backend = HuggingFaceModelBackend(args.model, device=args.device)
    import torch, transformers
    torch.manual_seed(77)
    contract = inspect_model_contract(backend)
    settings = dict(model=asdict(backend.metadata), model_contract=asdict(contract),
                    torch=torch.__version__, transformers=transformers.__version__, platform=platform.platform(),
                    seed=77, do_sample=False, repetition_penalty=1.0, max_new_tokens=args.max_new_tokens,
                    repeats=args.repeats, eos_token_ids=backend.eos_token_ids, offline=True,
                    variants=VARIANTS, stages={v: {t: stage_for(v,t) for t in TASKS} for v in VARIANTS},
                    chat_template=backend.chat_tokenizer.chat_template,
                    transport='unchanged InMemoryByteTransport and production TOKEN/KV bytes',
                    evaluation='Literal evidence and separate human assessment, no answer injection or repair')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def write(name, result):
        (args.output_dir/(name+'.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    write('settings', settings)
    write('fixtures', dict(secret=secret, cases=cases))
    def emit(name, result):
        write(name, result)
        print(name, flush=True)
        for row in result['modes']:
            print(row['mode'], row['engineering_status'], json.dumps(row.get('reply_text', row.get('error')), ensure_ascii=False), flush=True)
    token = run_token_matrix(backend, secret, limit=args.max_new_tokens, repeats=args.repeats, emit=emit)
    write('repetition-summary', repetition_summary(token))
    kv = dict(status='not_requested') if args.token_only else run_decode_comparison(
        backend, secret, token_records=token, limit=args.max_new_tokens, emit=emit)
    write('kv-summary', dict(status=kv['status']))
    success = verified(token) and (args.token_only or kv['status']=='compared' and verified(kv['records']))
    write('completion', dict(engineering_passed=success, token_rows=len(token), kv_status=kv['status'],
                             fixtures_sha256=hashlib.sha256((args.output_dir/'fixtures.json').read_bytes()).hexdigest()))
    print('Engineering passed:', success, '; quality requires independent assessment.', flush=True)
    return int(not success)
