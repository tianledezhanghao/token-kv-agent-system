"""7.7G offline local orchestration experiment (not a production Runtime)."""
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from .autogen_style_reference import (ReferenceHarness, STRUCTURES, MODES, READ_TASK,
    STAGES, systems, token_comparison, capture_token)
from .discussion import DiscussionConfig, DiscussionRuntime
from .discussion_prompts import AGENT_B_SYSTEM_PROMPT
from .secret_sum import SecretSumConfig, PUBLIC_TASK, STAGES as SECRET_STAGES, private_prompt
from .secret_sum_validation import run_scene
from .communication import HandoffChoice

SNAPSHOT = Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775'
READ_SYSTEM = ('You are Agent B. Read messages from Agent A and explain what A actually said. '
               'Be brief. Do not invent messages.')
FIXTURES = (('correct', '我认为答案是20。'), ('wrong', '我认为答案是21。'),
            ('uncertain', '我不确定答案是多少。'), ('absent', None))
QUESTIONS = (('ten', '10 + 10 = ?'), ('equation', '3x = 9, x = ?'),
             ('addition', '23 + 48 = ?'), ('chinese', '10+10=? 用中文讨论'),
             ('university', '中山大学在哪里？'), ('multiply', '12 * 13 = ?'))


def fixed(backend, limits, repeats, emit):
    for limit in limits:
        for name, fixture in FIXTURES:
            for structure in STRUCTURES:
                for task, stage in (('read', READ_TASK), ('review', STAGES['critic'].b_first)):
                    for repetition in range(repeats):
                        harness = ReferenceHarness(backend, '10 + 10 = ?', structure=structure)
                        if fixture is not None:
                            harness.deliver('A', 'B', fixture, origin='fixed_fixture')
                        row = harness.turn('B', stage, limit)
                        row.update(fixture=fixture, input_origin='fixed_fixture' if fixture else 'no_message',
                                   task=task, repetition=repetition+1)
                        emit(f'fixed-{limit}-{name}-{structure}-{task}-r{repetition+1}', row)


def understanding(backend, limits, repeats, emit):
    # Separate neutral reader role; Reviewer system confounds a reading task.
    for limit in limits:
        for name, fixture in FIXTURES:
            for structure in STRUCTURES:
                for repetition in range(repeats):
                    prompts = systems('critic') | {'B': READ_SYSTEM}
                    h = ReferenceHarness(backend, '10 + 10 = ?', structure=structure, prompts=prompts)
                    if fixture is not None:
                        h.deliver('A', 'B', fixture, origin='fixed_fixture')
                    row = h.turn('B', READ_TASK, limit)
                    row.update(fixture=fixture, input_origin='fixed_fixture' if fixture else 'no_message',
                               task='read', system_variant='neutral_reader', repetition=repetition+1)
                    emit(f'neutral-{limit}-{name}-{structure}-r{repetition+1}', row)


def dialogue(backend, limits, emit):
    for limit in limits:
        cases = QUESTIONS if limit == 128 else tuple(c for c in QUESTIONS if c[0] in ('ten','multiply','university'))
        for name, question in cases:
            for mode in MODES:
                for structure in STRUCTURES:
                    emit(f'dialogue-{limit}-{name}-{mode}-{structure}', ReferenceHarness(
                        backend, question, structure=structure, mode=mode).run(limit=limit))
                emit(f'token-{limit}-{name}-{mode}', token_comparison(backend, question, mode, limit))
        # Same private prompts and stages as the existing asymmetric experiment.
        prompts = {n: private_prompt(n, number) for n, number in (('A',17),('B',26))}
        for structure in STRUCTURES:
            emit(f'secret-reference-{limit}-{structure}', ReferenceHarness(
                backend, PUBLIC_TASK, structure=structure, prompts=prompts).run(
                    limit=limit, rounds=1, stages=SECRET_STAGES))
        emit(f'secret-token-{limit}', run_scene(backend, SecretSumConfig(max_new_tokens=limit)))
        for name, question in (c for c in QUESTIONS if c[0] in ('ten','multiply','university')):
            emit(f'production-default-{limit}-{name}', capture_token(DiscussionRuntime(backend,
                config=DiscussionConfig(question=question, rounds=2, max_new_tokens=limit))))
    # Prespecified repetition subset; no selection based on successful outputs.
    for name, question in (c for c in QUESTIONS if c[0] in ('ten','multiply','university')):
        for structure in ('B','D'):
            for mode in MODES:
                emit(f'dialogue-128-{name}-{mode}-{structure}-repeat', ReferenceHarness(
                    backend, question, structure=structure, mode=mode).run(limit=128))


def official(backend, limits, emit):
    from .autogen_local_client import official_fixed, official_team
    for limit in limits:
        for name, fixture in FIXTURES:
            for task, stage in (('read', READ_TASK), ('review', STAGES['critic'].b_first)):
                emit(f'official-fixed-{limit}-{name}-{task}', asyncio.run(official_fixed(
                    backend, '10 + 10 = ?', fixture, AGENT_B_SYSTEM_PROMPT, stage, limit)))
        cases = QUESTIONS if limit == 128 else tuple(c for c in QUESTIONS if c[0] in ('ten','multiply','university'))
        for name, question in cases:
            for mode in MODES:
                emit(f'official-team-{limit}-{name}-{mode}', asyncio.run(official_team(
                    backend, question, systems(mode), limit)))
        for mode in MODES:
            emit(f'official-raw-source-{limit}-ten-{mode}', asyncio.run(official_team(
                backend, '10 + 10 = ?', systems(mode), limit, prefix_sources=False)))
        prompts = {n: private_prompt(n, number) for n, number in (('A',17),('B',26))}
        emit(f'official-secret-{limit}', asyncio.run(official_team(backend, PUBLIC_TASK, prompts, limit)))


def regressions(backend, emit):
    from .discussion_regression_probe import RegressionRuntime
    from .message_understanding_probe import initialize_question, branch
    r = RegressionRuntime(backend, variant='D', question='10 + 10 = ?', limit=128)
    initialize_question(r)
    message = r.session.generate_message('A','B',max_new_tokens=128,local_prompt=r.prompt_config.stages.a_first)
    record = branch(r, {n:r.session.state(n) for n in ('A','B')}, message=message,
        stage=r.prompt_config.stages.b_first, origin='sender_decode',modes=('token','kv:v0','kv:v1'),oracle=False)
    assert all(row['engineering_status']=='verified' for row in record['modes'])
    emit('decode-token-v0-v1', record)
    # Same REAL sender draft, unlike the artificial diagnostic fixture matrix.
    # A can have byte-identical model IDs to TOKEN while using complete Prefill.
    sender_text=backend.decode(message.message_token_ids,skip_special_tokens=True)
    token_row=record['modes'][0]
    for structure in STRUCTURES:
        h=ReferenceHarness(backend,r.config.question,structure=structure)
        h.deliver('A','B',sender_text,origin='sender_decode')
        row=h.turn('B',r.prompt_config.stages.b_first,128)
        row.update(sender_message=asdict(message), sender_text=sender_text,
                   input_origin='one_frozen_real_sender_decode',
                   token_input_ids=token_row['context']['ledger_token_ids'],
                   token_reply=token_row['reply_text'],
                   input_ids_equal_token=tuple(row['input_ids'])==tuple(token_row['context']['ledger_token_ids']))
        if structure=='A':
            assert row['input_ids_equal_token'], 'A packaging must match captured production IDs'
        emit('same-sender-'+structure,row)
    for mode in ('token','kv:v0','kv:v1'):
        emit('regression-secret-'+mode.replace(':','-'), run_scene(backend,
            SecretSumConfig(handoff_choice=HandoffChoice.parse(mode))))
    emit('regression-secret-changed-b',run_scene(backend, SecretSumConfig(secret_b=31)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=SNAPSHOT)
    parser.add_argument('--device', choices=('mps','cpu'), default='mps')
    parser.add_argument('--limits', type=int, nargs='+', default=[128,64])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--phase', choices=('fixed','understanding','dialogue','official','regression','all'), default='all')
    parser.add_argument('--output-dir', type=Path, default=Path('benchmark-results/autogen-77g/all'))
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error('choose a new output directory; existing evidence is never overwritten')
    if args.repeats < 2 or any(i not in (64,128) for i in args.limits):
        parser.error('repeats >= 2 and limits 64/128 required')
    if not args.model.is_dir():
        parser.error('local checkpoint directory required; no download')
    if args.phase in ('official','all'):
        from .autogen_local_client import client_class
        client_class()  # fail before model load if optional environment is unavailable
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['OTEL_SDK_DISABLED']='true'
    import torch, transformers
    from .model import HuggingFaceModelBackend
    backend = HuggingFaceModelBackend(str(args.model),device=args.device)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.model.iterdir() if p.is_file()}
    args.output_dir.mkdir(parents=True)
    def emit(name, record):
        (args.output_dir/(name+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        print(name, flush=True)
        for key in ('discussion','result'):
            if isinstance(record.get(key),dict) and record[key].get('status')=='failed':
                raise RuntimeError(f'{name}: production execution failed; raw evidence saved')
    emit('settings', dict(model=asdict(backend.metadata), checkpoint=str(args.model), files_sha256=hashes,
        torch=torch.__version__, transformers=transformers.__version__, eos_ids=tuple(backend.eos_token_ids),
        chat_template=backend.chat_tokenizer.chat_template, limits=args.limits, do_sample=False,
        systems={m:systems(m) for m in MODES}, stages={m:asdict(s) for m,s in STAGES.items()},
        phase=args.phase, production_changed=False, official_version='0.7.5', neutral_reader_system=READ_SYSTEM,
        official_name_adapter='AutoGen _set_prepend_text_content: source said:\n + content on UserMessage; assistant unprefixed',
        reference_is_official=False, original_python_environment_modified=False))
    for phase, function in (('fixed',fixed),('understanding',understanding),('dialogue',dialogue),('official',official),('regression',regressions)):
        if args.phase not in (phase,'all'):
            continue
        if phase in ('fixed','understanding'):
            function(backend,args.limits,args.repeats,emit)
        elif phase=='regression':
            function(backend,emit)
        else:
            function(backend,args.limits,emit)
    emit('completion',dict(engineering_completed=True,phase=args.phase,model_quality_not_asserted=True))
    return 0
