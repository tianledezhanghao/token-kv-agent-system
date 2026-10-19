"""7.7D controlled framing, task isolation and real Decode KV boundaries."""
import pytest

from token_kv_agent_system.message_diagnostics import context_view, token_injection
from token_kv_agent_system.source_binding_probe import (
    VARIANTS, TASKS, ProbeFraming, fixed_cases, framed_snapshots, incoming_label,
    run_token_matrix, run_decode_comparison, stage_for, repetition_summary)
from token_kv_agent_system.token_protocol import deserialize_token_packet
from test_message_diagnostics import DisplayBackend
from test_stateful_token import build, snapshot


@pytest.mark.parametrize('variant', VARIANTS)
def test_labels_preserve_system_raw_peer_ids_and_exact_template(variant):
    b = DisplayBackend()
    s = build(b)
    for n in ('A', 'B'):
        s.append_local_user(n, 'Original question?')
    text = 'Only A knows: 青色水獭-53bc'
    message = token_injection(s, text)
    initial = {n: s.state(n) for n in ('A', 'B')}
    hashes = {n: snapshot(state.cache) for n, state in initial.items()}
    states = framed_snapshots(initial, variant)
    assert states['B'].cache is initial['B'].cache
    assert states['B'].token_ledger == initial['B'].token_ledger
    assert states['B'].system_prompt == initial['B'].system_prompt
    assert text not in b.decode(states['B'].token_ledger)
    s._states.update(states)
    packets = []
    transfer = s.transport.transfer
    def capture(data, **kw):
        packets.append(data)
        return transfer(data, **kw)
    s.transport.transfer = capture
    receipt = s.handoff('A', 'B')
    assert deserialize_token_packet(packets[0]).token_ids == message.message_token_ids
    frame = states['B'].framing.incoming('A', 'B')
    assert receipt.message.receiver_message_start == len(states['B'].token_ledger)+len(frame.prefix_ids)
    view = context_view(s.state('B'), stage=stage_for(variant, 'read'), generation_prefix=True)
    assert view['template_ids_equal'] and view['exact_span_coverage']
    assert view['latest_received'] == text and view['latest_sender'] == 'A'
    assert view['logical_messages'][2]['content'] == incoming_label(
        'A', 'B', variant in ('B-external', 'C-external-split')) + text
    assert all(snapshot(initial[n].cache) == hashes[n] for n in initial)
    assert initial['B'].received_messages == ()


def test_factorial_stage_controls_and_no_answer_vs_no_message():
    assert stage_for('A-default','read') == stage_for('B-external','read')
    assert stage_for('A-default','review') == stage_for('B-external','review')
    assert stage_for('C-external-split','review') == stage_for('D-default-split','review')
    cases = fixed_cases('PRIVATE-f13a')
    assert cases[-2]['message'] and cases[-1]['message'] is None
    assert cases[0]['question'] == cases[1]['question']
    for variant in VARIANTS:
        for task in TASKS:
            assert 'PRIVATE-f13a' not in stage_for(variant, task)
            assert '71' not in stage_for(variant, task) and '72' not in stage_for(variant, task)
    with pytest.raises(ValueError):
        stage_for('unknown','read')
    with pytest.raises(ValueError):
        fixed_cases('')


def test_matrix_never_feeds_read_reply_to_review_or_reuses_delivered_state(monkeypatch):
    import token_kv_agent_system.source_binding_probe as probe
    b = DisplayBackend()
    calls = []
    def independent(runtime, states, *, message, stage, origin, modes, oracle):
        assert not oracle and states['B'].received_messages == ()
        assert states['B'].local_messages == ()
        assert not states['B'].assistant_open
        from token_kv_agent_system.discussion_prompts import AGENT_B_SYSTEM_PROMPT
        assert states['B'].system_prompt == AGENT_B_SYSTEM_PROMPT
        calls.append((states['B'].token_ledger, stage, modes))
        runtime.session._states.update(states)
        if message:
            runtime.session.handoff('A', 'B')
        context = context_view(runtime.session.state('B'), stage=stage, generation_prefix=True)
        runtime.session._states.update(states)
        return dict(modes=[dict(engineering_status='verified', context=context,
                               reply=dict(message_token_ids=(100,2)), reply_text='Fixed mock output')])
    monkeypatch.setattr(probe, 'branch', independent)
    result = run_token_matrix(b, 'PRIVATE-f13a', limit=2, repeats=2)
    assert len(calls) == len(result) == 6*4*3*2
    assert all(r['modes'][0]['context']['template_ids_equal'] for r in result.values())
    assert all(r['modes'][0]['human_assessment']['accurate_restatement'] is None for r in result.values())
    summary = repetition_summary(result)
    assert len(summary) == 72
    assert all(r['identical_effective_ids'] and r['identical_output_ids'] for r in summary.values())
    group = next(iter(result.values()))
    group['modes'][0]['reply']['message_token_ids'] = (101,2)
    assert not all(r['identical_output_ids'] for r in repetition_summary(result).values())


def test_kv_gate_does_not_load_or_generate_when_token_engineering_failed():
    for records in ({}, {'empty': dict(modes=[])},
                    {'failed': dict(modes=[dict(engineering_status='failed')])}):
        result = run_decode_comparison(None, 'SECRET', token_records=records)
        assert result['status'] == 'blocked_by_token_engineering'


def test_framing_rejects_replacing_already_received_history_and_invalid_identity():
    s = build(DisplayBackend())
    token_injection(s, 'Peer message')
    s.handoff('A','B')
    with pytest.raises(ValueError, match='before receiving'):
        framed_snapshots({n:s.state(n) for n in ('A','B')}, 'B-external')
    with pytest.raises(ValueError, match='Agent IDs'):
        ProbeFraming(s.state('B').backend.chat_tokenizer, 'System', external=True).incoming('A\nforged','B')


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('raw', [(100,101,2), (100,101,3), (100,101), (100,)])
def test_framed_real_decode_span_same_snapshot_three_paths_without_hidden_prefill(variant, raw):
    from test_incremental_kv_transport import make_backend
    from token_kv_agent_system.discussion import DiscussionRuntime, DiscussionConfig
    from token_kv_agent_system.message_understanding_probe import branch
    b = make_backend()
    b.decode = lambda ids, **kw: DisplayBackend.decode(b, ids, **kw)
    runtime = DiscussionRuntime(b, config=DiscussionConfig(rounds=1, max_new_tokens=len(raw)))
    for n in ('A','B'):
        runtime.session.append_local_user(n, 'Read A.')
    b.outputs = list(raw)
    message = runtime.session.generate_message('A','B',max_new_tokens=len(raw))
    states = framed_snapshots({n:runtime.session.state(n) for n in ('A','B')}, variant)
    # Fail if delivery or decoding secretly recomputes history/content.
    b.prefill = lambda **kw: pytest.fail('hidden Prefill')
    result = branch(runtime, states, message=message, stage='Quote A.', origin='sender_decode',
                    modes=('token','kv:v0','kv:v1'), oracle=False)
    assert all(r['engineering_status']=='verified' for r in result['modes']), result
    assert len({tuple(r['context']['ledger_token_ids']) for r in result['modes']}) == 1
    assert all(r['context']['template_ids_equal'] and r['context']['exact_span_coverage'] for r in result['modes'])
    content_count = len(raw)-int(raw[-1] in (2,3))
    assert [r['receiver_message_prefill_count'] for r in result['modes']] == [content_count,0,0]
    for row in result['modes'][1:]:
        receipt = row['receipt']
        assert receipt['sender_content_span'] == (message.sender_message_span[0], message.sender_message_span[0]+content_count)
        assert receipt['receiver_target_span'] == (receipt['message']['receiver_message_start'],
            receipt['message']['receiver_message_start']+content_count)
        assert receipt['sender_supplemented_token_count'] == int(raw[-1] not in (2,3))
    assert result['modes'][1]['wire_sha256'] == result['modes'][2]['wire_sha256']
    assert all(runtime.session.state(n) is states[n] for n in states)


def test_sender_missing_secret_blocks_kv_even_when_receiver_guesses_it(monkeypatch):
    import token_kv_agent_system.source_binding_probe as probe
    b = DisplayBackend()
    calls = []
    def gate_only(runtime, states, **kwargs):
        calls.append(kwargs['modes'])
        return dict(modes=[dict(engineering_status='verified', reply_text='PRIVATE-f13a')])
    monkeypatch.setattr(probe, 'branch', gate_only)
    result = run_decode_comparison(b, 'PRIVATE-f13a', token_records={
        'fixed': dict(modes=[dict(engineering_status='verified')])}, limit=2)
    assert result['status'] == 'blocked_by_real_decode_recall'
    assert calls == [('token',)]
