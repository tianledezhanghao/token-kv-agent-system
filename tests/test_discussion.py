"""Discussion policy/bytes/limits tests with opaque scripted model predictions."""
import pytest

from token_kv_agent_system.discussion import (
    DiscussionConfig, DiscussionRuntime, FINAL_PROMPT, SOLVER_PROMPT, REVIEWER_PROMPT)
from token_kv_agent_system.model.state_cache import require_disjoint_caches
from token_kv_agent_system.token_protocol import deserialize_token_packet
from token_kv_agent_system.transport import InMemoryByteTransport
from test_stateful_token import Backend, snapshot


class ScriptBackend(Backend):
    def __init__(self, scripts=None):
        super().__init__()
        self.scripts = list(scripts or [(100, 101, 2)] * 7)
        self.generation_histories = []
        self.display_ids = []

    def continue_from_cache(self, ids, **kwargs):
        if ids == tuple(self.chat_tokenizer.encode("<s>assistant\n")):
            self.generation_histories.append(tuple(kwargs['previous_cache'].layers[0].keys.data))
            self.outputs = list(self.scripts.pop(0))
        return super().continue_from_cache(ids, **kwargs)

    def decode(self, ids, skip_special_tokens=True):
        self.display_ids.append(ids)
        return ''.join(chr(i - 10) for i in ids if i not in self.eos_token_ids)


class CaptureTransport(InMemoryByteTransport):
    def __init__(self):
        super().__init__()
        self.packets = []

    def transfer(self, data, **kwargs):
        self.packets.append(data)
        return super().transfer(data, **kwargs)


def build(*, rounds=2, summarizer='A', scripts=None, max_new_tokens=4, **kwargs):
    backend = ScriptBackend(scripts)
    runtime = DiscussionRuntime(backend, config=DiscussionConfig(
        question='35 + 27 = ?', rounds=rounds, summarizer=summarizer,
        max_new_tokens=max_new_tokens), **kwargs)
    return runtime, backend


@pytest.mark.parametrize('rounds', [1, 2, 3])
def test_round_order_shared_question_roles_and_long_lived_independent_histories(rounds):
    runtime, backend = build(rounds=rounds)
    initial = {name: runtime.session.state(name) for name in ('A', 'B')}
    result = runtime.run()
    assert result.status == 'completed'
    assert [(t.round, t.message.source_agent) for t in result.turns] == [
        (r, n) for r in range(1, rounds + 1) for n in ('A', 'B')]
    assert initial['A'].system_prompt == SOLVER_PROMPT
    assert initial['B'].system_prompt == REVIEWER_PROMPT
    assert SOLVER_PROMPT != REVIEWER_PROMPT
    assert sum(c[0] == 'prefill' for c in backend.calls) == 2
    for name in ('A', 'B'):
        state = runtime.session.state(name)
        state.validate()
        question_span = state.local_input_spans[0]
        assert state.token_ledger[slice(*question_span)] == state.framing.local_user_ids(result.config.question)
        assert len(state.sent_messages) == len(state.received_messages) == rounds
        assert len(state.token_ledger) == state.cached_token_count
        assert state.pending_token_id is None and state.draft_message is None
        assert state.token_ledger[:initial[name].cached_token_count] == initial[name].token_ledger
        assert snapshot(initial[name].cache)[0] == initial[name].token_ledger
        lengths = [t.cache_after[name] for t in result.turns]
        assert all(a < b for a, b in zip(lengths, lengths[1:]))
    require_disjoint_caches(runtime.session.state('A').cache, runtime.session.state('B').cache)
    assert len(backend.generation_histories) == 2 * rounds + 1
    question_ids = tuple(backend.chat_tokenizer.encode(result.config.question))
    for history in backend.generation_histories:
        assert any(history[i:i+len(question_ids)] == question_ids for i in range(len(history)))


def test_actual_token_bytes_only_new_ids_and_incremental_receiver_input():
    transport = CaptureTransport()
    runtime, backend = build(transport=transport)
    result = runtime.run()
    assert len(transport.packets) == 4  # final answer never goes through transport
    for turn, packet in zip(result.turns, transport.packets):
        m, receipt = turn.message, turn.receipt
        assert deserialize_token_packet(packet).token_ids == m.message_token_ids == receipt.received_token_ids
        assert len(packet) == receipt.wire_byte_count == 10 + 4 * m.token_count
        assert m.token_count == 3
        assert m.token_count < turn.cache_before[m.source_agent]
        assert receipt.receiver_cached_count_after - receipt.receiver_cached_count_before == len(receipt.receiver_appended_ids)
        assert ('continue', receipt.receiver_appended_ids, receipt.receiver_cached_count_before) in backend.calls
        state = runtime.session.state(m.target_agent)
        start = receipt.message.receiver_message_start
        count = receipt.message.receiver_content_token_count
        assert state.token_ledger[start:start+count] == m.message_token_ids[:count]
    assert backend.display_ids == [t.message.message_token_ids for t in result.turns] + [result.final_message.message_token_ids]


@pytest.mark.parametrize('stop_ids,reason', [((100, 2), 'eos'), ((100, 3), 'eos'),
                                            ((2,), 'eos'), ((100, 101), 'max_new_tokens')])
def test_stop_and_truncation_close_all_turns_including_final(stop_ids, reason):
    runtime, _ = build(rounds=1, scripts=[stop_ids] * 3, max_new_tokens=len(stop_ids))
    result = runtime.run()
    assert result.status == 'completed'
    for turn in result.turns:
        assert turn.message.message_token_ids == stop_ids
        assert turn.message.termination_reason == reason
        content = stop_ids[:-1] if reason == 'eos' else stop_ids
        assert turn.receipt.message.receiver_content_token_count == len(content)
        state = runtime.session.state(turn.message.source_agent)
        start, end = turn.message.sender_message_span
        assert state.token_ledger[start:end] == stop_ids
    final_state = runtime.session.state('A')
    assert final_state.token_ledger[-2:] == final_state.framing.assistant.closing_ids
    assert result.final_message.message_token_ids == stop_ids
    assert result.final_message.termination_reason == reason
    assert final_state.local_messages == (result.final_message,)
    assert result.final_message.sequence == 1
    assert result.final_message not in final_state.sent_messages
    assert result.final_message not in runtime.session.state('B').received_messages


@pytest.mark.parametrize('summarizer', ['A', 'B'])
def test_final_uses_selected_agents_real_history_and_generation_entry(summarizer):
    runtime, backend = build(summarizer=summarizer, scripts=[(100, 2), (101, 2), (102, 2), (103, 2), (104, 2)])
    final_entries = []
    original = runtime.session.generate_local_message
    def traced(agent_id, **kwargs):
        final_entries.append((agent_id, runtime.session.state(agent_id), kwargs))
        return original(agent_id, **kwargs)
    runtime.session.generate_local_message = traced
    result = runtime.run()
    assert result.status == 'completed'
    agent_id, before, kwargs = final_entries[0]
    assert agent_id == summarizer and kwargs['local_prompt'] == FINAL_PROMPT
    assert len(before.received_messages) == len(before.sent_messages) == 2
    assert result.final_message.message_token_ids == (104, 2)  # opaque model output, never arithmetic evaluation
    assert result.final_text == backend.decode((104, 2))
    assert result.final_message.source_agent == result.final_message.target_agent == summarizer
    assert backend.generation_histories[-1][:before.cached_token_count] == before.token_ledger
    latest_peer = before.received_messages[-1]
    peer_start = latest_peer.receiver_message_start
    assert backend.generation_histories[-1][peer_start:peer_start+1] == latest_peer.message_token_ids[:1]
    for turn in result.turns[1:]:
        state = runtime.session.state(turn.message.source_agent)
        prior = next(t for t in result.turns if t.message.target_agent == turn.message.source_agent)
        start = prior.receipt.message.receiver_message_start
        history = backend.generation_histories[result.turns.index(turn)]
        assert history[start:start+prior.receipt.message.receiver_content_token_count] == prior.message.message_token_ids[:-1]
        assert state.received_messages


def test_generation_failure_rolls_back_local_prompt_cache_and_stops_runtime():
    runtime, backend = build()
    original = runtime.session.generate_message
    saved = []
    def failing(source, target, **kwargs):
        saved.append({n: runtime.session.state(n) for n in ('A', 'B')})
        backend.fail_at = len(backend.calls) + 2  # local prompt succeeds; prefix mutates then raises
        return original(source, target, **kwargs)
    runtime.session.generate_message = failing
    result = runtime.run()
    assert result.status == 'failed' and result.failure_stage == 'round 1 / Agent A generation'
    assert not result.turns and result.final_message is None
    assert len(runtime.session.transport.records) == 0
    for name in ('A', 'B'):
        assert runtime.session.state(name) is saved[0][name]
        assert snapshot(saved[0][name].cache)[0] == saved[0][name].token_ledger
    calls = len(backend.calls)
    with pytest.raises(RuntimeError, match='terminal'):
        runtime.run()
    assert len(backend.calls) == calls


def test_failed_transfer_is_not_a_completed_turn_and_cannot_continue():
    class BrokenTransport(CaptureTransport):
        def transfer(self, data, **kwargs):
            super().transfer(data, **kwargs)
            return data[:-1]
    runtime, backend = build(transport=BrokenTransport())
    result = runtime.run()
    assert result.status == 'failed' and result.failure_stage == 'round 1 / TOKEN A -> B'
    assert not result.turns and result.final_message is None
    assert len(backend.generation_histories) == 1
    a, b = runtime.session.state('A'), runtime.session.state('B')
    assert a.draft_message is not None and a.pending_token_id is not None
    assert not a.sent_messages and not b.received_messages
    a.validate(); b.validate()
    assert len(runtime.session.transport.records) == 1  # diagnostic attempt only


def test_final_close_failure_preserves_discussion_state_and_has_no_answer():
    runtime, backend = build(rounds=1)
    original = runtime.session.generate_local_message
    saved = []
    def failing(agent_id, **kwargs):
        saved.append(runtime.session.state(agent_id))
        backend.fail_at = len(backend.calls) + 5  # prompt, prefix, 2 Decode, closing
        return original(agent_id, **kwargs)
    runtime.session.generate_local_message = failing
    result = runtime.run()
    assert result.status == 'failed' and result.failure_stage.startswith('final / Agent A')
    assert len(result.turns) == 2 and result.final_message is None
    assert runtime.session.state('A') is saved[0]
    assert snapshot(saved[0].cache)[0] == saved[0].token_ledger


def test_generation_budget_fails_before_forward_or_prompt_commit():
    runtime, backend = build()
    s = runtime.session
    old = s.state('A')
    s.max_context_tokens = old.cached_token_count + 1
    calls = len(backend.calls)
    with pytest.raises(ValueError, match='context budget'):
        s.generate_message('A', 'B', max_new_tokens=64, local_prompt='Local stage instruction')
    assert len(backend.calls) == calls and s.state('A') is old


def test_receiver_budget_fails_before_transfer_or_state_commit():
    runtime, backend = build()
    s = runtime.session
    s.generate_message('A', 'B', max_new_tokens=4)
    old = {n: s.state(n) for n in ('A', 'B')}
    s.max_context_tokens = len(old['B'].token_ledger) + 1
    calls = len(backend.calls)
    with pytest.raises(ValueError, match='context budget'):
        s.handoff('A', 'B')
    assert len(backend.calls) == calls and not s.transport.records
    assert all(s.state(n) is old[n] for n in old)


def test_backend_context_window_caps_configured_budget():
    backend = ScriptBackend()
    backend.context_window_tokens = 1024
    runtime = DiscussionRuntime(backend, config=DiscussionConfig(max_context_tokens=4096))
    assert runtime.context_budget == runtime.session.max_context_tokens == 1024


@pytest.mark.parametrize('overrides', [{'rounds': 0}, {'rounds': 21}, {'rounds': True},
                                      {'question': ''}, {'max_new_tokens': 0},
                                      {'max_context_tokens': -1}, {'summarizer': 'C'}])
def test_invalid_config(overrides):
    with pytest.raises(ValueError):
        DiscussionConfig(**overrides)


def test_cli_help_and_invalid_args_do_not_load_model(capsys):
    from token_kv_agent_system.discussion_demo import main
    with pytest.raises(SystemExit) as help_exit:
        main(['--help'])
    assert help_exit.value.code == 0
    assert '--summarizer' in capsys.readouterr().out
    with pytest.raises(SystemExit) as invalid_exit:
        main(['--max-new-tokens', '0'])
    assert invalid_exit.value.code == 2


def test_cli_success_and_failure_exit_status_with_exact_ids_json(monkeypatch, capsys, tmp_path):
    from token_kv_agent_system import discussion_demo
    class CLIBackend(ScriptBackend):
        def __init__(self, *args, **kwargs):
            super().__init__()
            from token_kv_agent_system.model import ModelMetadata
            self.metadata = ModelMetadata('mock', 'cpu', 256, 2, None, None, 0.0, dtype='float32')
    monkeypatch.setattr(discussion_demo, 'HuggingFaceModelBackend', CLIBackend)
    path = tmp_path / 'result.json'
    assert discussion_demo.main(['--rounds', '1', '--output', str(path)]) == 0
    import json
    record = json.loads(path.read_text())
    assert record['discussion']['final_message']['message_token_ids'] == [100, 101, 2]
    assert 'Final Answer / Agent A:' in capsys.readouterr().out
    assert discussion_demo.main(['--max-context-tokens', '10']) == 1
    assert 'FAILED at initialize Agent roles' in capsys.readouterr().out


def test_compatible_independent_backend_endpoints_and_display_never_retokenizes_peer():
    a = ScriptBackend([(100, 2), (101, 2)])
    b = ScriptBackend([(102, 2)])
    runtime = DiscussionRuntime(a, b, config=DiscussionConfig(rounds=1, max_new_tokens=4))
    result = runtime.run()
    assert result.status == 'completed'
    assert result.turns[0].message.message_token_ids == (100, 2)
    assert result.turns[1].message.message_token_ids == (102, 2)
    assert result.final_message.message_token_ids == (101, 2)
    assert len(a.generation_histories) == 2 and len(b.generation_histories) == 1
    assert sum(c[0] == 'prefill' for c in a.calls) == sum(c[0] == 'prefill' for c in b.calls) == 1
    # No peer decode: only each source's own generated IDs reach its display API.
    assert a.display_ids == [(100, 2), (101, 2)] and b.display_ids == [(102, 2)]


def test_context_failure_during_initial_question_is_terminal():
    runtime, backend = build()
    runtime.session.max_context_tokens = runtime.session.state('A').cached_token_count
    result = runtime.run()
    assert result.status == 'failed' and result.failure_stage == 'initialize question A'
    assert not result.turns and not backend.generation_histories
    assert not runtime.session.transport.records


def test_final_budget_failure_never_generates_or_commits_summary_instruction():
    runtime, backend = build(rounds=1)
    original = runtime.session.generate_local_message
    saved = []
    def limited(agent_id, **kwargs):
        old = runtime.session.state(agent_id)
        saved.append(old)
        runtime.session.max_context_tokens = old.cached_token_count + 1
        return original(agent_id, **kwargs)
    runtime.session.generate_local_message = limited
    result = runtime.run()
    assert result.status == 'failed' and len(result.turns) == 2
    assert result.final_message is None and len(backend.generation_histories) == 2
    assert runtime.session.state('A') is saved[0]
