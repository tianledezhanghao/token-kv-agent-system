"""Offline 7.7B prompt experiments; preserve every output, including mistakes."""
from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path

from .communication import HandoffChoice
from .discussion import DiscussionConfig, DiscussionRuntime
from .discussion_demo import print_result
from .model import HuggingFaceModelBackend
from .model.config import DEFAULT_MODEL_NAME

CUSTOM_B = (
    "You are Agent B. Check whether Agent A answered the user's actual question. "
    "Do your own check before agreeing. State a specific error or a brief reason. "
    "Say if you cannot verify a fact. Use the user's requested language."
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=DEFAULT_MODEL_NAME)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--output-dir', type=Path, default=Path('benchmark-results/discussion-77b'))
    args = parser.parse_args(argv)
    os.environ['HF_HUB_OFFLINE'] = '1'
    b = HuggingFaceModelBackend(args.model, device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    cases = [
        ('math-default-token-A', '23 + 48 = ?', 'token', None, 'A'),
        ('fact-default-token-A', '中山大学在哪里，用中文回答', 'token', None, 'A'),
        ('math-custom-token-A', '23 + 48 = ?', 'token', CUSTOM_B, 'A'),
        ('math-custom-v1-A', '23 + 48 = ?', 'kv:v1', CUSTOM_B, 'A'),
        ('math-default-token-B', '23 + 48 = ?', 'token', None, 'B'),
        ('fact-custom-token-A', '中山大学在哪里，用中文回答', 'token', CUSTOM_B, 'A'),
        ('fact-custom-v1-B', '中山大学在哪里，用中文回答', 'kv:v1', CUSTOM_B, 'B'),
    ]
    original = b.prefill
    for name, question, mode, override, summarizer in cases:
        prefills = []
        def traced(**kwargs):
            prefills.append(tuple(kwargs['input_token_ids']))
            return original(**kwargs)
        b.prefill = traced
        config = DiscussionConfig(question=question, rounds=2, max_new_tokens=64,
            summarizer=summarizer, handoff_choice=HandoffChoice.parse(mode),
            agent_b_system_prompt=override)
        runtime = DiscussionRuntime(b, config=config)
        initial = {n: runtime.session.state(n) for n in ('A', 'B')}
        result = runtime.run()
        assert prefills == [initial[n].token_ledger for n in ('A', 'B')]
        assert all(initial[n].system_prompt == getattr(result.prompt_config, f'agent_{n.lower()}').text for n in initial)
        if result.status == 'completed':
            assert len(result.turns) == len(result.handoffs) == 4
            assert result.final_message.source_agent == summarizer
            assert all(h.fallback_reason is None for h in result.handoffs)
            assert all(h.receiver_message_prefill_count == 0 for h in result.handoffs if h.executed_mode == 'kv')
        failed |= result.status != 'completed'
        record = {'model': asdict(b.metadata), 'discussion': asdict(result),
            'initial_system_contexts': {n: {
                'token_ids': initial[n].token_ledger,
                'cached_count': initial[n].cached_token_count,
                'rendered_template': b.decode(initial[n].token_ledger, skip_special_tokens=False),
            } for n in initial},
            'full_prefill_count': len(prefills),
            'undelivered_drafts': {n: asdict(runtime.session.state(n).draft_message)
                for n in initial if runtime.session.state(n).draft_message is not None}}
        (args.output_dir / f'{name}.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        print(f'\nCase: {name}')
        print_result(result)
        b.prefill = original
    # Controlled diagnostic only. This fixed A text is local input, not generated
    # output or an actual TOKEN/KV handoff. No expected answer reaches B.
    runtime = DiscussionRuntime(b, config=DiscussionConfig(rounds=1))
    s = runtime.session
    s.append_local_user('B', '23 + 48 = ?')
    s.append_local_user('B', 'Message from A to B:\nThe answer is 72.')
    before = s.state('B')
    instruction = runtime.prompt_config.stages.b_first
    message = s.generate_local_message('B', max_new_tokens=64, local_prompt=instruction)
    record = {'kind': 'controlled local reviewer input; no handoff claimed',
        'model': asdict(b.metadata), 'question': '23 + 48 = ?',
        'fixed_a_message': 'The answer is 72.', 'prompt_config': runtime.prompt_config.to_dict(),
        'reviewer_history_before_stage': b.decode(before.token_ledger, skip_special_tokens=False),
        'stage_instruction': instruction,
        'reviewer_text': b.decode(message.message_token_ids, skip_special_tokens=True),
        'message': asdict(message)}
    (args.output_dir / 'controlled-review.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print('\nControlled reviewer:\n' + record['reviewer_text'])
    return int(failed)
