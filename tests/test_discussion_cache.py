"""End-to-end discussion using real random tiny Qwen2 CPU KV, no downloaded model."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from token_kv_agent_system.discussion import DiscussionConfig, DiscussionRuntime
from token_kv_agent_system.model.state_cache import require_disjoint_caches
from test_stateful_token_cache import backend, copies, unchanged


def test_cpu_discussion_full_history_oracle_and_no_history_prefill():
    b = backend()
    prefills = []
    original = b.prefill
    def traced(**kwargs):
        prefills.append(kwargs['input_token_ids'])
        return original(**kwargs)
    b.prefill = traced
    runtime = DiscussionRuntime(b, config=DiscussionConfig(
        question='12 * 13 = ?', rounds=2, max_new_tokens=2, summarizer='B'))
    initial = {n: runtime.session.state(n) for n in ('A', 'B')}
    saved = {n: copies(initial[n].cache) for n in initial}
    result = runtime.run()
    assert result.status == 'completed', result.error
    assert len(result.turns) == 4 and result.final_message.source_agent == 'B'
    assert len(prefills) == 2
    for n in ('A', 'B'):
        state = runtime.session.state(n)
        state.validate()
        assert state.cached_token_count == len(state.token_ledger)
        unchanged(initial[n].cache, saved[n])
        # Full Prefill is strictly a TEST oracle, outside the discussion path.
        full = original(input_token_ids=state.token_ledger)
        for layer, reference in zip(state.cache.layers, full.cache.layers):
            assert torch.allclose(layer.keys, reference.keys, atol=1e-6, rtol=1e-5)
            assert torch.allclose(layer.values, reference.values, atol=1e-6, rtol=1e-5)
    require_disjoint_caches(runtime.session.state('A').cache, runtime.session.state('B').cache)


def test_cpu_final_partial_forward_failure_rolls_back_real_dynamiccache():
    b = backend()
    runtime = DiscussionRuntime(b, config=DiscussionConfig(rounds=1, max_new_tokens=2))
    original_final = runtime.session.generate_local_message
    saved = []
    original_forward = b._model.forward
    def fail_after_forward(**kwargs):
        out = original_forward(**kwargs)
        out.past_key_values.layers[0].keys.zero_()
        raise RuntimeError('injected final forward mutation')
    def failing(agent_id, **kwargs):
        state = runtime.session.state(agent_id)
        saved.append((state, copies(state.cache)))
        b._model.forward = fail_after_forward
        return original_final(agent_id, **kwargs)
    runtime.session.generate_local_message = failing
    result = runtime.run()
    assert result.status == 'failed' and len(result.turns) == 2
    assert result.failure_stage == 'final / Agent A generation and close'
    assert 'injected final forward mutation' in result.error
    state, tensors = saved[0]
    assert runtime.session.state('A') is state
    unchanged(state.cache, tensors)
    state.validate()
    assert result.final_message is None


@pytest.mark.parametrize('mode', ['token', 'kv:v0', 'kv:v1', 'mixed'])
def test_cpu_custom_prompts_real_initial_cache_and_dynamic_handoff_positions(mode):
    from test_incremental_kv_transport import make_backend
    from token_kv_agent_system.communication import HandoffChoice
    b = make_backend()
    plan = ('token', 'kv:v1', 'token', 'kv:v0') if mode == 'mixed' else (mode,) * 4
    prefills = []
    original = b.prefill
    def traced(**kwargs):
        prefills.append(kwargs['input_token_ids'])
        return original(**kwargs)
    b.prefill = traced
    runtime = DiscussionRuntime(b, config=DiscussionConfig(
        rounds=2, max_new_tokens=2, agent_a_system_prompt='A short custom role.',
        agent_b_system_prompt='B has a different, longer custom role. Independently check the user request.',
        handoff_plan=tuple(map(HandoffChoice.parse, plan))))
    initial = {n: runtime.session.state(n) for n in ('A', 'B')}
    for index, n in enumerate(('A', 'B')):
        state = initial[n]
        expected = tuple(b.chat_tokenizer.apply_chat_template(
            [{'role': 'system', 'content': state.system_prompt}], tokenize=True, add_generation_prompt=False))
        assert state.token_ledger == prefills[index] == expected
        oracle = original(input_token_ids=expected)
        for layer, reference in zip(state.cache.layers, oracle.cache.layers):
            assert torch.equal(layer.keys, reference.keys)
            assert torch.equal(layer.values, reference.values)
    saved = {n: copies(initial[n].cache) for n in initial}
    result = runtime.run()
    assert result.status == 'completed', result.error
    assert len(prefills) == 2  # no historical Full Prefill
    assert len(result.handoffs) == 4
    for h in result.handoffs:
        state = runtime.session.state(h.target)
        assert h.receiver_kv_target_span[0] == h.receiver_cache_length_before + len(state.framing.incoming(h.source, h.target).prefix_ids)
        assert h.receiver_message_prefill_count == (2 if h.executed_mode == 'token' else 0)
        assert h.fallback_reason is None
    for n in initial:
        unchanged(initial[n].cache, saved[n])
        runtime.session.state(n).validate()
    require_disjoint_caches(runtime.session.state('A').cache, runtime.session.state('B').cache)
