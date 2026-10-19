from dataclasses import replace
import pytest
from token_kv_agent_system.discussion import DiscussionConfig, DiscussionRuntime
from token_kv_agent_system.secret_sum import SecretSumConfig
from token_kv_agent_system.dashboard.discussion_service import DiscussionStartRequest
from token_kv_agent_system.communication import HandoffChoice
from token_kv_agent_system.local_qwen_resources import ResourceStop
from token_kv_agent_system.token_protocol import deserialize_token_packet
from test_discussion import ScriptBackend, CaptureTransport


def short_stages(runtime):
    runtime._prompt_config = replace(runtime.prompt_config, stages=replace(runtime.prompt_config.stages,
        a_first='a', b_first='b', a_followup='a', b_followup='b', final='f'))


@pytest.mark.parametrize('budget', [129, 256, 512, 1024])
def test_extended_rest_and_runtime_budgets_real_ids_and_token_transport(budget):
    assert DiscussionStartRequest(question='Q', max_new_tokens=budget).runtime_config().max_new_tokens == budget
    b = ScriptBackend([(100,) * 150 + (2,)] * 3)
    transport = CaptureTransport()
    r = DiscussionRuntime(b, config=DiscussionConfig(question='Q', rounds=1, max_new_tokens=budget,
        agent_a_system_prompt='A', agent_b_system_prompt='B'), transport=transport)
    short_stages(r)
    result = r.run(); assert result.status == 'completed', result.error
    for t, packet in zip(result.turns, transport.packets):
        assert t.message.token_count == min(budget, 151)
        assert t.message.termination_reason == ('max_new_tokens' if budget < 151 else 'eos')
        assert t.message.configured_max_new_tokens == t.message.effective_max_new_tokens == budget
        assert deserialize_token_packet(packet).token_ids == t.message.message_token_ids == t.receipt.received_token_ids
        r.session.state(t.message.source_agent).validate()


@pytest.mark.parametrize('budget', [512, 1024])
def test_context_shrink_reports_reason_preserves_ids_and_leaves_future_stages(budget):
    b = ScriptBackend([(100,) * 1100] * 3)
    r = DiscussionRuntime(b, config=DiscussionConfig(question='Q', rounds=1, max_new_tokens=budget,
        max_context_tokens=400, agent_a_system_prompt='A', agent_b_system_prompt='B'))
    short_stages(r)
    result = r.run(); assert result.status == 'completed', result.error
    first = result.turns[0]
    assert 128 < first.message.token_count < budget
    assert first.message.termination_reason == 'context_limit'
    assert first.message.configured_max_new_tokens == budget
    assert first.message.effective_max_new_tokens == first.message.token_count
    assert first.message.budget_limit_reason == 'context_limit'
    assert first.receipt.received_token_ids == first.message.message_token_ids
    assert first.receipt.message.receiver_content_token_count == first.message.token_count
    assert all(t.message.token_count >= 1 for t in result.turns)
    assert all(n <= 400 for n in result.final_cache_lengths.values())
    for n in ('A', 'B'):
        state = r.session.state(n); state.validate()
        assert state.pending_token_id is None and len(state.token_ledger) == state.cached_token_count


def test_eos_with_reduced_budget_is_still_eos():
    b = ScriptBackend([(100, 2)] * 3)
    r = DiscussionRuntime(b, config=DiscussionConfig(question='Q', rounds=1, max_new_tokens=1024,
        max_context_tokens=400, agent_a_system_prompt='A', agent_b_system_prompt='B'))
    short_stages(r)
    result = r.run(); assert result.status == 'completed'
    assert all(t.message.termination_reason == 'eos' and t.message.effective_max_new_tokens < 1024 for t in result.turns)


def test_no_context_fails_before_decode_without_partial_commit():
    b = ScriptBackend()
    r = DiscussionRuntime(b, config=DiscussionConfig(question='Q', rounds=1, max_context_tokens=120,
        agent_a_system_prompt='A', agent_b_system_prompt='B'))
    result = r.run()
    assert result.status == 'failed' and result.termination_reason == 'context_limit'
    assert not b.generation_histories and not r.session.transport.records
    assert not result.turns
    for n in ('A', 'B'): r.session.state(n).validate()


@pytest.mark.parametrize('error,reason', [(ResourceStop('sustained_critical_pressure'), 'resource_abort'), (RuntimeError('MPS out of memory'), 'resource_abort'), (RuntimeError('forward failed'), 'error')])
def test_failure_reasons_and_atomic_generation(error, reason):
    b = ScriptBackend()
    r = DiscussionRuntime(b, config=DiscussionConfig(question='Q', rounds=1, max_new_tokens=512))
    original = b.continue_from_cache
    def failing(ids, **kw):
        if ids == tuple(b.chat_tokenizer.encode('<s>assistant\n')): raise error
        return original(ids, **kw)
    b.continue_from_cache = failing
    result = r.run()
    assert result.termination_reason == reason and result.status == 'failed'
    assert not result.turns and not r.session.transport.records
    for n in ('A', 'B'): r.session.state(n).validate()


def test_product_default_and_frozen_secret_budget():
    assert DiscussionConfig().max_new_tokens == DiscussionStartRequest(question='Q').max_new_tokens == 512
    assert SecretSumConfig().max_new_tokens == 96
    for n in (64, 96, 128):
        assert DiscussionConfig(max_new_tokens=n).max_new_tokens == n
        assert SecretSumConfig(max_new_tokens=n).runtime_config().max_new_tokens == n


@pytest.mark.parametrize('mode', ['kv:v0', 'kv:v1', 'kv:v2', 'kv:v3'])
def test_long_truncated_kv_pending_span_and_receiver_ledger(mode):
    from test_incremental_kv_transport import make_backend
    b = make_backend()
    b.decode = lambda ids, **kw: '17'  # deterministic V3 slot fixture, not a quality test
    r = DiscussionRuntime(b, config=DiscussionConfig(question='Q', rounds=1, max_new_tokens=160,
        agent_a_system_prompt='A', agent_b_system_prompt='B', handoff_choice=HandoffChoice.parse(mode)))
    short_stages(r)
    result = r.run(); assert result.status == 'completed', result.error
    for t in result.turns:
        assert t.message.token_count == 160 and t.message.termination_reason == 'max_new_tokens'
        assert t.receipt.sender_supplemented_token_count == 1
        assert t.receipt.received_token_ids == t.message.message_token_ids
        a, z = t.receipt.sender_content_span
        assert z-a == 160
        state = r.session.state(t.message.target_agent)
        x, y = t.receipt.receiver_target_span
        assert state.token_ledger[x:y] == t.message.message_token_ids
    for n in ('A', 'B'):
        state = r.session.state(n); state.validate()
        assert state.pending_token_id is None and state.cached_token_count == len(state.token_ledger)


def test_kv_protocol_capacity_rejected_before_generation(monkeypatch):
    from test_incremental_kv_transport import make_backend
    import token_kv_agent_system.incremental_kv_protocol as protocol
    real = protocol.IncrementalLimits
    monkeypatch.setattr(protocol, 'IncrementalLimits', lambda: real(max_packet_bytes=4096))
    b = make_backend()
    r = DiscussionRuntime(b, config=DiscussionConfig(question='Q', rounds=1, max_new_tokens=512,
        agent_a_system_prompt='A', agent_b_system_prompt='B', handoff_choice=HandoffChoice.parse('kv:v1')))
    short_stages(r)
    result = r.run()
    assert result.status == 'failed' and 'protocol capacity' in result.error
    assert not result.turns and not r.session.transport.records
    assert not any(ids == tuple(b.chat_tokenizer.encode('<s>assistant\n')) for ids, _ in b.calls)


def test_context_limited_kv_content_is_not_eos_and_pending_still_materializes():
    from test_incremental_kv_transport import make_backend
    b = make_backend()
    r = DiscussionRuntime(b, config=DiscussionConfig(question='Q', rounds=1, max_new_tokens=1024,
        max_context_tokens=400, agent_a_system_prompt='A', agent_b_system_prompt='B',
        handoff_choice=HandoffChoice.parse('kv:v1')))
    short_stages(r)
    result = r.run(); assert result.status == 'completed', result.error
    first = result.turns[0]
    assert first.message.termination_reason == 'context_limit' and first.message.token_count > 128
    assert first.receipt.sender_supplemented_token_count == 1
    assert first.receipt.received_token_ids == first.message.message_token_ids
    assert first.receipt.sender_content_span[1] - first.receipt.sender_content_span[0] == first.message.token_count
    for n in ('A', 'B'): r.session.state(n).validate()


@pytest.mark.parametrize('layers,heads,dim,dtype,accepted', [(24,2,64,'bfloat16',True), (36,2,128,'bfloat16',True), (36,2,128,'float32',False), (80,8,128,'bfloat16',False)])
def test_packet_guard_uses_loaded_model_shape_and_dtype(layers, heads, dim, dtype, accepted, monkeypatch):
    from test_incremental_kv_transport import make_backend
    from types import SimpleNamespace
    b = make_backend()
    r = DiscussionRuntime(b, config=DiscussionConfig(question='Q', rounds=1, max_new_tokens=1024,
        agent_a_system_prompt='A', agent_b_system_prompt='B', handoff_choice=HandoffChoice.parse('kv:v1')))
    short_stages(r)
    b._metadata = replace(b.metadata, dtype=dtype)
    monkeypatch.setattr(type(b), 'kv_model_config', property(lambda _: SimpleNamespace(num_hidden_layers=layers, num_key_value_heads=heads, head_dim=dim)))
    if accepted:
        assert r._generation_budget(1)['effective_max_new_tokens'] == 1024
    else:
        with pytest.raises(ValueError, match='protocol capacity'): r._generation_budget(1)


def test_sender_position_guard_rejects_before_generation(monkeypatch):
    from test_incremental_kv_transport import make_backend
    import token_kv_agent_system.incremental_kv_protocol as protocol
    real = protocol.IncrementalLimits
    monkeypatch.setattr(protocol, 'IncrementalLimits', lambda: real(max_sender_positions=100))
    b = make_backend()
    r = DiscussionRuntime(b, config=DiscussionConfig(question='Q', rounds=1, max_new_tokens=512,
        agent_a_system_prompt='A', agent_b_system_prompt='B', handoff_choice=HandoffChoice.parse('kv:v2')))
    short_stages(r)
    result = r.run()
    assert result.status == 'failed' and 'protocol capacity' in result.error
    assert not result.turns and not r.session.transport.records
