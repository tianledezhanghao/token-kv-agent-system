"""7.7F frozen historical/current prompt factorial; diagnostic only.

No production prompts, runtime policy, transport or KV algorithm are changed.
Artificial fixed messages are labelled fixtures and travel through TOKEN bytes.
"""
import argparse
from collections.abc import Mapping
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

from .discussion import DiscussionConfig, DiscussionRuntime
from .discussion_prompts import StageInstructions, STAGE_INSTRUCTIONS
from .message_diagnostics import (context_view, token_injection,
                                  greedy_from_prefill, logits_difference)
from .message_understanding_probe import branch, initialize_question
from .model import HuggingFaceModelBackend
from .model.config import DEFAULT_MODEL_NAME
from .secret_sum import SecretSumConfig
from .secret_sum_validation import run_scene

# Literal fixtures verified against fe888ee, never used as new runtime defaults.
LEGACY_A = (
    "You are Agent A, a solver. Solve the user's arithmetic question. "
    "Be concise. Consider feedback from Agent B and correct mistakes when necessary. "
    "Do not invent disagreements."
)
LEGACY_B = (
    "You are Agent B, a reviewer. Independently check Agent A's calculation "
    "for the user's original arithmetic question. If it is correct, say so. "
    "If it is wrong, identify and correct the error. Be concise."
)
LEGACY_STAGES = StageInstructions(
    a_first='Solve the original question and send your initial calculation to Agent B.',
    a_followup="Consider Agent B's feedback above. Confirm or correct your calculation "
               'of the original question. Reply briefly to Agent B.',
    b_first="Independently check Agent A's latest calculation above against "
            'the original question. Reply briefly to Agent A.',
    b_followup="Independently check Agent A's latest calculation above against "
               'the original question. Reply briefly to Agent A.',
    final='Give the final answer to the original arithmetic question. '
          'Use the discussion above to check your result. Keep it concise.',
)
VARIANTS = {'A': ('old', 'old'), 'B': ('old', 'new'),
            'C': ('new', 'old'), 'D': ('new', 'new')}
READ_STAGE = "Quote the exact message received from Agent A. Do not substitute your own answer."
CASES = (
    ('correct', '10 + 10 = ?', '10 + 10 = 20'),
    ('wrong', '10 + 10 = ?', '10 + 10 = 21'),
    ('university', '中山大学在哪里？', '中山大学位于中国广东省中山市。'),
    ('absent', '10 + 10 = ?', None),
)
FEEDBACK = 'Your previous answer said 21. Please check it again.'
REFUSAL = 'I cannot provide a review or check for a message that I have not seen.'


class RegressionRuntime(DiscussionRuntime):
    """Harness-local injection; no Dashboard/config API or default changes."""
    def __init__(self, backend, *, variant, question, rounds=2, limit=64):
        system, stages = VARIANTS[variant]
        super().__init__(backend, config=DiscussionConfig(question=question, rounds=rounds,
            max_new_tokens=limit, agent_a_system_prompt=LEGACY_A if system == 'old' else None,
            agent_b_system_prompt=LEGACY_B if system == 'old' else None))
        self._prompt_config = replace(self.prompt_config, version='77f-' + variant,
            stages=LEGACY_STAGES if stages == 'old' else STAGE_INSTRUCTIONS)


def capture_discussion(runtime):
    contexts = []
    originals = {}
    for method in ('generate_message', 'generate_local_message'):
        original = getattr(runtime.session, method)
        originals[method] = original
        def wrap(source, *args, _original=original, **kwargs):
            before = runtime.session.state(source)
            view = context_view(before, question=runtime.config.question,
                                stage=kwargs['local_prompt'], generation_prefix=True)
            assert view['exact_span_coverage'] and view['template_ids_equal']
            message = _original(source, *args, **kwargs)
            assert runtime.session.state(source).token_ledger[:message.sender_message_span[0]] == tuple(view['ledger_token_ids'])
            contexts.append(dict(context=view, output=asdict(message)))
            return message
        setattr(runtime.session, method, wrap)
    try:
        result = runtime.run()
    finally:
        for method, original in originals.items():
            setattr(runtime.session, method, original)
    assert result.status == 'completed', result.error
    return dict(discussion=asdict(result), contexts=contexts,
        final_contexts={n: context_view(runtime.session.state(n)) for n in ('A', 'B')})


def standard_chat(backend, question, message, limit):
    """Different-ID semantic oracle, never used for formal communication."""
    from .discussion_prompts import AGENT_B_SYSTEM_PROMPT
    messages = [dict(role='system', content=AGENT_B_SYSTEM_PROMPT), dict(role='user', content=(
        f'Original user question:\n{question}\nReceived from Agent A (verbatim):\n{message}\n'
        "Quote Agent A's message and independently check whether it answers the original question. "
        'Use the language requested by the user.'))]
    rendered = backend.chat_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(rendered, Mapping):
        rendered = rendered['input_ids']
    ids = tuple(int(i) for i in rendered)
    generated = greedy_from_prefill(backend, backend.prefill(input_token_ids=ids), limit)
    return dict(origin='different_input_standard_chat_oracle', messages=messages,
        input_ids=ids, generated_ids=generated,
        text=backend.decode(generated, skip_special_tokens=True),
        termination_reason='eos' if generated[-1] in backend.eos_token_ids else 'max_new_tokens')


def fixed_matrix(backend, *, limit=64, repeats=2, emit):
    stability = {}
    for name, question, message in CASES:
        for variant in VARIANTS:
            runtime = RegressionRuntime(backend, variant=variant, question=question, limit=limit)
            initialize_question(runtime)
            fixture = token_injection(runtime.session, message) if message else None
            snapshots = {n: runtime.session.state(n) for n in ('A', 'B')}
            for task, stage in [('read', READ_STAGE), ('review', runtime.prompt_config.stages.b_first)]:
                key = f'fixed-{name}-{variant}-{task}'
                rows = []
                for repetition in range(repeats):
                    record = branch(runtime, snapshots, message=fixture, stage=stage,
                        origin='artificial_token_fixture' if fixture else 'no_message',
                        modes=('token',) if fixture else ('none',),
                        oracle=(variant == 'D' and task == 'review' and repetition == 0))
                    record.update(variant=variant, task=task, repetition=repetition+1,
                                  question=question, fixture_text=message)
                    assert all(r['engineering_status'] == 'verified' for r in record['modes']), record
                    emit(key + f'-r{repetition+1}', record)
                    rows.append(record['modes'][0])
                stability[key] = dict(repeats=repeats,
                    input_ids_identical=len({tuple(r['context']['ledger_token_ids']) for r in rows}) == 1,
                    generated_ids_identical=len({tuple(r['reply']['message_token_ids']) for r in rows}) == 1)
    return stability


def feedback_control(backend, variant, feedback, limit):
    runtime = RegressionRuntime(backend, variant=variant, question='10 + 10 = ?', limit=limit)
    initialize_question(runtime)
    token_injection(runtime.session, '10 + 10 = 21')
    runtime.session.handoff('A', 'B')
    token_injection(runtime.session, feedback, source='B', target='A')
    runtime.session.handoff('B', 'A')
    view = context_view(runtime.session.state('A'), question=runtime.config.question,
                        stage=runtime.prompt_config.stages.a_followup, generation_prefix=True)
    assert view['latest_received'] == feedback
    assert view['exact_span_coverage'] and view['template_ids_equal']
    message = runtime.session.generate_local_message('A', max_new_tokens=limit,
                                local_prompt=runtime.prompt_config.stages.a_followup)
    assert runtime.session.state('A').token_ledger[:message.sender_message_span[0]] == tuple(view['ledger_token_ids'])
    return dict(origin='artificial_token_fixture_both_directions', context=view,
        output=asdict(message), text=backend.decode(message.message_token_ids, skip_special_tokens=True))


def precision_controls(backend, fixed_dir, *, limit=64, emit):
    """Same recorded MPS IDs on one CPU/FP32 model, with actual append groups."""
    from .agent.chat_framing import ChatFraming
    if backend.metadata.device != 'cpu' or backend.metadata.dtype != 'torch.float32':
        raise ValueError('precision control requires CPU/FP32')
    for name in ('correct', 'wrong', 'university'):
        record = json.loads((fixed_dir/f'fixed-{name}-D-review-r1.json').read_text())
        row = record['modes'][0]
        view = row['context']
        ids = tuple(view['ledger_token_ids'])
        framing = ChatFraming(backend.chat_tokenizer, view['system_prompt'])
        groups = []
        previous = ()
        for i in range(1, len(view['logical_messages']) + 1):
            rendered = framing._render_ids(view['logical_messages'][:i])
            assert rendered[:len(previous)] == previous
            groups.append(rendered[len(previous):])
            previous = rendered
        groups.append(framing.assistant.prefix_ids)
        assert tuple(t for group in groups for t in group) == ids
        initial = backend.prefill(input_token_ids=groups[0])
        cache, count = initial.cache, len(groups[0])
        for group in groups[1:]:
            incremental = backend.continue_from_cache(group, previous_cache=cache,
                                                      logical_cached_token_count=count)
            cache, count = incremental.cache, incremental.logical_cached_token_count
        full = backend.prefill(input_token_ids=ids)
        # Construct a PrefillResult from the final incremental logits/cache.
        from dataclasses import replace
        inc_result = replace(full, cache=cache, first_token_id=incremental.predicted_next_token_id,
                             last_token_logits=incremental.last_token_logits)
        inc_ids = greedy_from_prefill(backend, inc_result, limit)
        full_ids = greedy_from_prefill(backend, full, limit)
        emit('cpu-' + name, dict(origin='same_recorded_input_cpu_fp32_oracle',
            input_ids=ids, append_groups=groups, model=asdict(backend.metadata),
            logits=logits_difference(full.last_token_logits, incremental.last_token_logits),
            incremental_generated_ids=inc_ids, full_generated_ids=full_ids,
            generation_ids_equal=inc_ids == full_ids,
            incremental_text=backend.decode(inc_ids, skip_special_tokens=True),
            full_text=backend.decode(full_ids, skip_special_tokens=True),
            mps_incremental_text=row['reply_text'], mps_full_text=row['token_oracle']['reference_text']))


def run_probe(backend, *, limit=64, repeats=2, phase='all', emit):
    if phase not in ('all', 'fixed', 'continuation'):
        raise ValueError('unknown phase')
    if phase != 'continuation':
        stability = fixed_matrix(backend, limit=limit, repeats=repeats, emit=emit)
        emit('stability', stability)
    if phase == 'fixed':
        return
    for variant in VARIANTS:
        for label, feedback in [('specific', FEEDBACK), ('refusal', REFUSAL)]:
            emit(f'feedback-{variant}-{label}', feedback_control(backend, variant, feedback, limit))
    for name, question, message in CASES:
        if message:
            emit('standard-' + name, standard_chat(backend, question, message, limit))
    for variant in VARIANTS:
        for name, question, rounds in [('ten', '10 + 10 = ?', 3),
                ('chinese', '10+10=? 用中文讨论', 3), ('university', '中山大学在哪里？', 2)]:
            emit(f'discussion-{variant}-{name}', capture_discussion(RegressionRuntime(
                backend, variant=variant, question=question, rounds=rounds, limit=limit)))
    for name, question in [('add', '23 + 48 = ?'), ('multiply', '12 * 13 = ?'), ('ten-two', '10 + 10 = ?')]:
        emit('current-legacy-' + name, capture_discussion(RegressionRuntime(
            backend, variant='A', question=question, limit=limit)))
    r = RegressionRuntime(backend, variant='D', question='10 + 10 = ?', limit=limit)
    initialize_question(r)
    message = r.session.generate_message('A', 'B', max_new_tokens=limit,
                                         local_prompt=r.prompt_config.stages.a_first)
    record = branch(r, {n: r.session.state(n) for n in ('A','B')}, message=message,
        stage=r.prompt_config.stages.b_first, origin='sender_decode',
        modes=('token','kv:v0','kv:v1'), oracle=False)
    assert all(row['engineering_status'] == 'verified' for row in record['modes'])
    emit('decode-token-v0-v1', record)
    from .communication import HandoffChoice
    for mode in ('token', 'kv:v0', 'kv:v1'):
        emit('secret-' + mode.replace(':', '-'), run_scene(backend,
            SecretSumConfig(handoff_choice=HandoffChoice.parse(mode))))
    emit('secret-token-changed-b', run_scene(backend, SecretSumConfig(secret_b=31)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=DEFAULT_MODEL_NAME)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--phase', choices=('all','fixed','continuation','cpu'), default='all',
                        help='Split expensive diagnosis into independent output directories')
    parser.add_argument('--fixed-input-dir', type=Path, help='Recorded fixed MPS inputs for --phase cpu')
    parser.add_argument('--output-dir', type=Path, default=Path('benchmark-results/discussion-77f/matrix'))
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error('choose a new output directory')
    if args.max_new_tokens <= 0 or args.repeats < 2:
        parser.error('token limit must be positive and repeats must be at least two')
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch, transformers
    if args.phase == 'cpu':
        if args.fixed_input_dir is None:
            parser.error('--phase cpu requires --fixed-input-dir')
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
        backend = HuggingFaceModelBackend(args.model, device='cpu', tokenizer=tokenizer, model=model)
    else:
        backend = HuggingFaceModelBackend(args.model, device=args.device)
    from .incremental_kv_protocol import inspect_model_contract
    settings = dict(model=asdict(backend.metadata), contract=asdict(inspect_model_contract(backend)),
        torch=torch.__version__, transformers=transformers.__version__,
        chat_template=backend.chat_tokenizer.chat_template,
        max_new_tokens=args.max_new_tokens, secret_max_new_tokens=96, phase=args.phase,
        do_sample=False, variants=VARIANTS, legacy_systems=[LEGACY_A, LEGACY_B],
        legacy_stages=asdict(LEGACY_STAGES), current_stages=asdict(STAGE_INSTRUCTIONS))
    args.output_dir.mkdir(parents=True)
    def emit(name, record):
        (args.output_dir/(name+'.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        texts = ([r.get('reply_text',r.get('error')) for r in record['modes']] if 'modes' in record
                 else [t['text'] for t in record['discussion']['turns']]+[record['discussion']['final_text']]
                 if 'discussion' in record else [record.get('text')])
        print(name, json.dumps(texts,ensure_ascii=False), flush=True)
    emit('settings', settings)
    if args.phase == 'cpu':
        precision_controls(backend, args.fixed_input_dir, limit=args.max_new_tokens, emit=emit)
    else:
        run_probe(backend, limit=args.max_new_tokens, repeats=args.repeats, phase=args.phase, emit=emit)
    emit('completion', dict(engineering_passed=True, production_runtime_changed=False, phase=args.phase))
    return 0
