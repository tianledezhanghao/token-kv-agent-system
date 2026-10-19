"""7.7F reproducible controls, actual stage/source inputs and diagnostic isolation."""
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from token_kv_agent_system.discussion import DiscussionRuntime
from token_kv_agent_system.discussion_prompts import (AGENT_A_SYSTEM_PROMPT,
    AGENT_B_SYSTEM_PROMPT, STAGE_INSTRUCTIONS)
from token_kv_agent_system.discussion_regression_probe import (
    VARIANTS, LEGACY_A, LEGACY_B, LEGACY_STAGES, RegressionRuntime,
    capture_discussion, fixed_matrix, feedback_control, standard_chat, FEEDBACK)
from token_kv_agent_system.message_diagnostics import context_view, token_injection
from token_kv_agent_system.token_protocol import deserialize_token_packet
from test_message_diagnostics import DisplayBackend
from test_incremental_kv_transport import make_backend
from test_stateful_token import build


def test_legacy_fixtures_match_independent_frozen_history_snapshot():
    # Exported from the original blob before release; clean clones need no private Git history.
    root = Path(__file__).resolve().parents[1]
    snapshot = json.loads((root / 'tests/fixtures/legacy-discussion-fe888ee.json').read_text())
    assert snapshot['source_commit'].startswith('fe888ee')
    constants = snapshot['constants']
    assert constants['SOLVER_PROMPT'] == LEGACY_A
    assert constants['REVIEWER_PROMPT'] == LEGACY_B
    assert constants['FINAL_PROMPT'] == LEGACY_STAGES.final
    assert all(s in snapshot['string_literals'] for s in asdict(LEGACY_STAGES).values())


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('summarizer', ['A', 'B'])
def test_factorial_full_run_exact_stages_sources_spans_and_no_history_prefill(variant, summarizer):
    b = make_backend()
    b.decode = lambda ids, **kw: DisplayBackend.decode(b, ids, **kw)
    r = RegressionRuntime(b, variant=variant, question='10 + 10 = ?', limit=2)
    from dataclasses import replace
    r.config = replace(r.config, summarizer=summarizer)
    old = {n: r.session.state(n) for n in ('A','B')}
    b.prefill = lambda **kw: pytest.fail('full Prefill of history in discussion')
    record = capture_discussion(r)
    stages = r.prompt_config.stages
    assert [c['context']['stage_prompt'] for c in record['contexts']] == [
        stages.a_first, stages.b_first, stages.a_followup, stages.b_followup, stages.final]
    assert [c['context']['agent'] for c in record['contexts']] == ['A','B','A','B',summarizer]
    assert old['A'].system_prompt == (LEGACY_A if variant in ('A','B') else AGENT_A_SYSTEM_PROMPT)
    assert old['B'].system_prompt == (LEGACY_B if variant in ('A','B') else AGENT_B_SYSTEM_PROMPT)
    contexts = record['contexts']
    assert contexts[1]['context']['latest_sender'] == 'A'
    assert contexts[2]['context']['latest_sender'] == 'B'
    assert len(contexts[-1]['context']['logical_messages']) > len(contexts[0]['context']['logical_messages'])
    for row in contexts:
        view = row['context']
        state = r.session.state(view['agent'])
        assert tuple(view['ledger_token_ids']) == state.token_ledger[:row['output']['sender_message_span'][0]]
        assert view['segments'][-1]['source'] == 'assistant generation prefix'
        assert view['logical_messages'][1]['content'] == r.config.question
        assert all('21' not in m['content'] for m in view['logical_messages'])
    for turn in record['discussion']['turns']:
        message, receipt = turn['message'], turn['receipt']
        state = r.session.state(message['target_agent'])
        start, count = message['receiver_message_start'], message['receiver_content_token_count']
        assert state.token_ledger[start:start+count] == message['message_token_ids'][:count]
        assert receipt['receiver_cached_count_after']-receipt['receiver_cached_count_before'] == len(receipt['receiver_appended_ids'])
    for n in old:
        state = r.session.state(n)
        assert state.token_ledger[:len(old[n].token_ledger)] == old[n].token_ledger
    # Harness selection has no effect on default sessions or secret_sum stages.
    ordinary = DiscussionRuntime(DisplayBackend())
    assert ordinary.prompt_config.stages == STAGE_INSTRUCTIONS
    from token_kv_agent_system.secret_sum import SecretSumRuntime, STAGES
    assert SecretSumRuntime(DisplayBackend()).prompt_config.stages == STAGES


def test_reverse_feedback_fixture_uses_bytes_correct_identity_and_no_hidden_local_content():
    s = build(DisplayBackend())
    s.append_local_user('A', 'Original question')
    before = s.state('A')
    fixture = token_injection(s, FEEDBACK, source='B', target='A')
    assert s.state('A') is before
    packets = []
    transfer = s.transport.transfer
    def capture(data, **kw):
        packets.append(data)
        return transfer(data, **kw)
    s.transport.transfer = capture
    receipt = s.handoff('B','A')
    assert deserialize_token_packet(packets[0]).token_ids == fixture.message_token_ids
    view = context_view(s.state('A'), stage='Read feedback', generation_prefix=True)
    assert view['latest_sender'] == 'B' and view['latest_received'] == FEEDBACK
    assert view['template_ids_equal'] and view['exact_span_coverage']
    assert s.state('A').local_input_spans == before.local_input_spans
    assert receipt.received_token_ids == fixture.message_token_ids
    assert 'Message from B to A:' in view['decoded_ledger']
    with pytest.raises(ValueError, match='distinct|differ'):
        token_injection(s, 'bad', source='A', target='A')


@pytest.mark.parametrize('variant', VARIANTS)
def test_feedback_control_preserves_actual_feedback_and_original_own_answer(variant):
    b = make_backend()
    b.decode = lambda ids, **kw: DisplayBackend.decode(b, ids, **kw)
    result = feedback_control(b, variant, FEEDBACK, 2)
    view = result['context']
    assert view['latest_received'] == FEEDBACK
    assert any(m['role']=='assistant' and m['content']=='10 + 10 = 21' for m in view['logical_messages'])
    assert view['logical_messages'][-1]['content'] == (
        LEGACY_STAGES if variant in ('A','C') else STAGE_INSTRUCTIONS).a_followup
    assert result['origin'] == 'artificial_token_fixture_both_directions'


def test_fixed_read_review_repeats_use_identical_history_without_output_feedback(monkeypatch):
    import token_kv_agent_system.discussion_regression_probe as probe
    calls = []
    def independent(runtime, snapshots, *, message, stage, origin, modes, oracle):
        assert not snapshots['B'].received_messages and not snapshots['B'].local_messages
        runtime.session._states.update(snapshots)
        if message:
            runtime.session.handoff('A','B')
        view = context_view(runtime.session.state('B'), stage=stage, generation_prefix=True)
        assert all('read-output-canary' not in m['content'] for m in view['logical_messages'])
        calls.append((origin, modes, oracle, view))
        return dict(modes=[dict(engineering_status='verified', context=view,
            reply=dict(message_token_ids=(100,2)), reply_text='read-output-canary')])
    monkeypatch.setattr(probe,'branch',independent)
    records = {}
    stability = fixed_matrix(DisplayBackend(), limit=2, repeats=2, emit=lambda k,v: records.update({k:v}))
    assert len(records) == 64 and len(stability) == 32
    assert all(r['input_ids_identical'] and r['generated_ids_identical'] for r in stability.values())
    assert sum(oracle for _,_,oracle,_ in calls) == 4
    assert all(modes == (('none',) if origin=='no_message' else ('token',)) for origin,modes,_,_ in calls)
    assert all(r['fixture_text'] is None for k,r in records.items() if 'absent' in k)


def test_standard_template_is_explicit_different_input_oracle_with_no_answer_key():
    b = make_backend()
    record = standard_chat(b, '10 + 10 = ?', '10 + 10 = 21', 2)
    assert record['origin'] == 'different_input_standard_chat_oracle'
    assert len(record['messages']) == 2
    assert record['messages'][0]['content'] == AGENT_B_SYSTEM_PROMPT
    assert '10 + 10 = 21' in record['messages'][1]['content']
    assert '20' not in record['messages'][1]['content']
    expected = tuple(b.chat_tokenizer.apply_chat_template(record['messages'], tokenize=True, add_generation_prompt=True))
    assert record['input_ids'] == expected


def test_standard_template_accepts_transformers_batchencoding_mapping():
    b = make_backend()
    original = b.chat_tokenizer.apply_chat_template
    def mapped(*args, **kwargs):
        result = original(*args, **kwargs)
        return {'input_ids': result} if kwargs.get('tokenize') else result
    b.chat_tokenizer.apply_chat_template = mapped
    record = standard_chat(b, '10 + 10 = ?', '10 + 10 = 21', 1)
    assert all(type(i) is int for i in record['input_ids'])
    assert record['generated_ids']


def test_precision_control_requires_real_cpu_fp32():
    from token_kv_agent_system.discussion_regression_probe import precision_controls
    b = DisplayBackend()
    class Metadata:
        device = 'mps'
        dtype = 'torch.bfloat16'
    b.metadata = Metadata()
    with pytest.raises(ValueError, match='CPU/FP32'):
        precision_controls(b, Path('unused'), emit=lambda *args: pytest.fail('unexpected output'))


def test_historical_runner_restores_capture_hooks_on_failed_discussion():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'run_discussion_history_probe.py'
    spec = importlib.util.spec_from_file_location('history77f', path)
    history = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(history)
    from test_discussion import ScriptBackend
    b = ScriptBackend()
    runtime = DiscussionRuntime(b)
    originals = (runtime.session.generate_message, runtime.session.generate_local_message)
    b.continue_from_cache = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('failure'))
    record = history.capture_run(runtime, b)
    assert record['discussion']['status'] == 'failed'
    assert (runtime.session.generate_message, runtime.session.generate_local_message) == originals


def test_cpu_precision_control_replays_exact_saved_ids_and_append_groups(tmp_path):
    import json
    from test_stateful_token_cache import backend
    from token_kv_agent_system.discussion_regression_probe import precision_controls
    b = backend()
    b.decode = lambda ids, **kw: DisplayBackend.decode(b, ids, **kw)
    for name, text in [('correct','10 + 10 = 20'), ('wrong','10 + 10 = 21'), ('university','A place')]:
        r = RegressionRuntime(b, variant='D', question='Question?', limit=2)
        for n in ('A','B'):
            r.session.append_local_user(n, 'Question?')
        token_injection(r.session, text)
        r.session.handoff('A','B')
        view = context_view(r.session.state('B'), stage=STAGE_INSTRUCTIONS.b_first, generation_prefix=True)
        record = dict(modes=[dict(context=view, reply_text='recorded', token_oracle=dict(reference_text='recorded full'))])
        (tmp_path/f'fixed-{name}-D-review-r1.json').write_text(json.dumps(record))
    rows = {}
    precision_controls(b, tmp_path, limit=2, emit=lambda k,v: rows.update({k:v}))
    assert len(rows) == 3
    for row in rows.values():
        assert tuple(t for group in row['append_groups'] for t in group) == row['input_ids']
        assert len(row['append_groups']) == 5  # system, question, received, stage, assistant prefix
        assert row['generation_ids_equal'] and row['logits']['max_abs'] < 1e-6
