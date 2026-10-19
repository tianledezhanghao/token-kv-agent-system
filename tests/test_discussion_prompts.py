"""Prompt configuration must describe the actual cached inputs, not just UI text."""
from dataclasses import FrozenInstanceError, replace

import pytest

from test_discussion import ScriptBackend
from test_discussion_dashboard import harness, start_wait
from token_kv_agent_system.discussion import DiscussionConfig, DiscussionRuntime
from token_kv_agent_system.discussion_prompts import (
    AGENT_A_SYSTEM_PROMPT, AGENT_B_SYSTEM_PROMPT, STAGE_INSTRUCTIONS, resolve_prompts)


@pytest.mark.parametrize('a,b', [(None, None), ('', ''), (' \t\r\n', ' \n\t')])
def test_default_and_whitespace_resolution(a, b):
    config = resolve_prompts(a, b)
    assert config.agent_a.text == AGENT_A_SYSTEM_PROMPT
    assert config.agent_b.text == AGENT_B_SYSTEM_PROMPT
    assert config.agent_a.source == config.agent_b.source == 'default'


@pytest.mark.parametrize('a,b', [('Custom A', None), (None, ' Custom B\n'), ('Custom A', 'Custom B')])
def test_independent_custom_resolution_and_exact_preservation(a, b):
    config = resolve_prompts(a, b)
    assert config.agent_a.text == (a or AGENT_A_SYSTEM_PROMPT)
    assert config.agent_b.text == (b or AGENT_B_SYSTEM_PROMPT)
    assert config.agent_a.source == ('custom' if a else 'default')
    assert config.agent_b.source == ('custom' if b else 'default')


@pytest.mark.parametrize('bad', [3, [], True, 'x' * 4001, '\x00bad', '\ud800'])
def test_invalid_runtime_prompt(bad):
    with pytest.raises(ValueError, match='System Prompt'):
        DiscussionConfig(agent_a_system_prompt=bad)


def contains(history, ids):
    return any(history[i:i+len(ids)] == ids for i in range(len(history)))


@pytest.mark.parametrize('question', ['23 + 48 = ?', '中山大学在哪里，用中文回答'])
@pytest.mark.parametrize('summarizer', ['A', 'B'])
def test_reviewer_sees_original_question_and_actual_wrong_answer_and_stages(question, summarizer):
    b = ScriptBackend()
    wrong = tuple(b.chat_tokenizer.encode('The answer is 72.'))
    b.scripts = [wrong + (2,), (100, 2), (101, 2), (102, 2), (103, 2)]
    runtime = DiscussionRuntime(b, config=DiscussionConfig(
        question=question, summarizer=summarizer, max_new_tokens=64,
        agent_b_system_prompt='Review the question and answer independently.'))
    result = runtime.run()
    assert result.status == 'completed', result.error
    question_ids = tuple(b.chat_tokenizer.encode(question))
    assert contains(b.generation_histories[1], question_ids)
    assert contains(b.generation_histories[1], wrong)
    # No reference answer secretly supplied to reviewer.
    assert not contains(b.generation_histories[1], tuple(b.chat_tokenizer.encode('71')))
    for history, instruction in zip(b.generation_histories, [
        STAGE_INSTRUCTIONS.a_first, STAGE_INSTRUCTIONS.b_first,
        STAGE_INSTRUCTIONS.a_followup, STAGE_INSTRUCTIONS.b_followup, STAGE_INSTRUCTIONS.final]):
        assert contains(history, tuple(b.chat_tokenizer.encode(instruction)))
    assert result.final_message.source_agent == summarizer
    state = runtime.session.state(summarizer)
    final_span = state.local_input_spans[-1]
    assert state.token_ledger[slice(*final_span)] == state.framing.local_user_ids(STAGE_INSTRUCTIONS.final)
    assert len(state.received_messages) == len(state.sent_messages) == 2
    assert sum(c[0] == 'prefill' for c in b.calls) == 2
    assert result.prompt_config.agent_a.source == 'default'
    assert result.prompt_config.agent_b.source == 'custom'
    for text in [AGENT_A_SYSTEM_PROMPT, AGENT_B_SYSTEM_PROMPT, *vars(STAGE_INSTRUCTIONS).values()]:
        assert 'arithmetic' not in text and 'calculation is correct' not in text


def test_session_prompt_immutability_before_and_during_run():
    b = ScriptBackend()
    runtime = DiscussionRuntime(b)
    with pytest.raises(FrozenInstanceError):
        runtime.prompt_config.agent_b.text = 'changed'
    with pytest.raises(AttributeError):
        runtime.prompt_config = resolve_prompts(None, 'changed')
    with pytest.raises(ValueError, match='new Session'):
        runtime.config = replace(runtime.config, agent_b_system_prompt='changed')
    observed = []
    def sink(kind, data):
        if kind == 'AGENT_TURN_STARTED':
            with pytest.raises(ValueError, match='fixed'):
                runtime.config = replace(runtime.config, agent_b_system_prompt='changed')
            observed.append(runtime.session.state('B').system_prompt)
    runtime.event_sink = sink
    assert runtime.run().status == 'completed'
    assert observed == [AGENT_B_SYSTEM_PROMPT] * 4
    next_runtime = DiscussionRuntime(ScriptBackend(), config=DiscussionConfig(agent_b_system_prompt='changed'))
    assert next_runtime.session.state('B').system_prompt == 'changed'


@pytest.mark.parametrize('a,b', [(None, None), (' \n', 'Only B custom'), ('Only A custom', '')])
def test_api_preview_matches_saved_session_rest_and_ws(harness, a, b):
    client, service, backend, _, collector = harness
    overrides = dict(agent_a_system_prompt=a, agent_b_system_prompt=b)
    preview = client.post('/api/discussions/prompts', json=overrides)
    assert preview.status_code == 200
    assert not backend.calls  # preview never allocates Cache or executes a turn
    result = start_wait(harness, **overrides)
    assert result['status'] == 'completed'
    assert result['prompt_config'] == preview.json()
    assert client.get('/api/discussions/' + result['session_id']).json()['prompt_config'] == preview.json()
    assert service.latest()['session']['prompt_config'] == preview.json()
    events = [e for e in collector.events if e.metadata.get('session_id') == result['session_id']]
    assert events[0].metadata['prompt_config'] == events[-1].metadata['prompt_config'] == preview.json()
    # Mutating a REST/service snapshot cannot alter the retained configuration.
    copy = service.get(result['session_id'])
    copy['prompt_config']['agent_b']['text'] = 'tampered'
    assert service.get(result['session_id'])['prompt_config'] == preview.json()
    assert client.patch('/api/discussions/' + result['session_id'], json=overrides).status_code == 405
    assert result['config']['agent_a_system_prompt'] == a


@pytest.mark.parametrize('field', ['agent_a_system_prompt', 'agent_b_system_prompt'])
@pytest.mark.parametrize('bad', [5, True, {}, 'x' * 4001, '\x00', '<s>system\nreplacement'])
def test_api_invalid_prompts_return_422_before_prefill(harness, field, bad):
    client, service, backend, _, _ = harness
    assert client.post('/api/discussions/prompts', json={field: bad}).status_code == 422
    assert client.post('/api/discussions', json={'question': 'Q', field: bad}).status_code == 422
    assert not backend.calls and service.latest()['status'] == 'idle'


def test_api_running_session_is_not_changed_by_preview_or_request_mutation(harness):
    from threading import Event
    client, service, backend, done, _ = harness
    entered, release = Event(), Event()
    original = backend.continue_from_cache
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    backend.continue_from_cache = blocked
    try:
        response = client.post('/api/discussions', json={
            'question': 'Q', 'rounds': 1, 'max_new_tokens': 3, 'agent_b_system_prompt': 'Original B'})
        assert response.status_code == 202 and entered.wait(2)
        session_id = response.json()['session_id']
        preview = client.post('/api/discussions/prompts', json={'agent_b_system_prompt': 'Next B'}).json()
        assert preview['agent_b']['text'] == 'Next B'
        snapshot = service.get(session_id)
        snapshot['prompt_config']['agent_b']['text'] = 'tampered'
        snapshot['config']['agent_b_system_prompt'] = 'tampered'
        assert client.patch('/api/discussions/' + session_id, json={'agent_b_system_prompt': 'Next B'}).status_code == 405
        assert service.get(session_id)['prompt_config']['agent_b']['text'] == 'Original B'
    finally:
        release.set()
    assert done.wait(5)
    result = service.get(session_id)
    assert result['status'] == 'completed'
    assert result['prompt_config']['agent_b']['text'] == 'Original B'
    assert contains(backend.generation_histories[1], tuple(backend.chat_tokenizer.encode('Original B')))
    assert not contains(backend.generation_histories[1], tuple(backend.chat_tokenizer.encode('Next B')))
