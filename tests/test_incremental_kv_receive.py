"""Dense tiny CPU KV, byte transport, and persistent mixed discussion contracts."""
from dataclasses import replace

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from test_incremental_kv_transport import make_backend
from test_stateful_token_cache import copies, unchanged
from token_kv_agent_system.communication import CommunicationMode, KVStrategy, HandoffChoice
from token_kv_agent_system.discussion import DiscussionConfig, DiscussionRuntime
from token_kv_agent_system.incremental_kv import prepare_incremental_message
from token_kv_agent_system.incremental_kv_protocol import inspect_model_contract, encode_incremental_message
from token_kv_agent_system.incremental_kv_receive import receive_incremental_kv
from token_kv_agent_system.model.delta_repair import rephase_key, rotary_tables
from token_kv_agent_system.model.state_cache import require_disjoint_caches
from token_kv_agent_system.transport import InMemoryByteTransport


def build(plan=(), choice=HandoffChoice(), rounds=1, raw=(100, 101, 2)):
    b = make_backend()
    b.outputs = list(raw)
    runtime = DiscussionRuntime(b, config=DiscussionConfig(
        rounds=rounds, max_new_tokens=len(raw), handoff_choice=choice, handoff_plan=plan))
    return b, runtime


@pytest.mark.parametrize('strategy', [KVStrategy.RAW_DELTA, KVStrategy.ROPE_ALIGNED_DELTA])
@pytest.mark.parametrize('raw', [(100, 101, 2), (100, 101, 3), (100, 101), (100,)])
def test_receiver_real_decode_kv_exact_content_framing_positions_and_no_m_forward(strategy, raw):
    b, runtime = build(raw=raw)
    s = runtime.session
    s.append_local_user('B', 'Different receiver history')
    m = s.generate_message('A', 'B', max_new_tokens=len(raw))
    old = {n:s.state(n) for n in ('A', 'B')}
    saved = {n:copies(old[n].cache) for n in old}
    contract = inspect_model_contract(b)
    envelope, _ = prepare_incremental_message(old['A'], m, contract=contract)
    b.calls.clear()
    # With EOS no sender content forward; truncation may materialize the accepted
    # last token at the Sender only. Receiver must never consume content IDs.
    receipt = s.handoff_kv('A', 'B', strategy=strategy)
    state = s.state('B')
    frame = state.framing.incoming('A', 'B')
    start, end = receipt.receiver_target_span
    assert start == old['B'].cached_token_count + len(frame.prefix_ids)
    content = envelope.message_token_ids
    assert state.token_ledger == old['B'].token_ledger + frame.prefix_ids + content + frame.closing_ids
    receiver_calls = [(ids, pos) for ids, pos in b.calls if pos in (old['B'].cached_token_count, end)]
    assert receiver_calls == [(frame.prefix_ids, old['B'].cached_token_count), (frame.closing_ids, end)]
    assert not any(pos == start for _, pos in b.calls)
    assert receipt.receiver_message_prefill_count == 0
    assert receipt.rope_position_delta == start - envelope.sender_span_start
    for layer, neutral in zip(state.cache.layers, envelope.delta.layers):
        k = torch.frombuffer(bytearray(neutral.key.data), dtype=torch.float32).reshape(neutral.key.shape)
        v = torch.frombuffer(bytearray(neutral.value.data), dtype=torch.float32).reshape(neutral.value.shape)
        if strategy == KVStrategy.ROPE_ALIGNED_DELTA:
            old_tables = rotary_tables(b, list(range(envelope.sender_span_start, envelope.sender_span_start + len(content))))
            new_tables = rotary_tables(b, list(range(start, end)))
            k = rephase_key(k, *old_tables, *new_tables)
        assert torch.equal(layer.keys[...,start:end,:], k)
        assert torch.equal(layer.values[...,start:end,:], v)
    for layer, (k,v) in zip(state.cache.layers, saved['B']):
        assert torch.equal(layer.keys[...,:k.shape[2],:], k)
        assert torch.equal(layer.values[...,:v.shape[2],:], v)
    for n in old:
        unchanged(old[n].cache, saved[n])
        s.state(n).validate()
        assert s.state(n).cached_token_count == len(s.state(n).token_ledger)
    require_disjoint_caches(s.state('A').cache, s.state('B').cache)
    assert receipt.wire_byte_count == s.transport.records[0].byte_count
    assert receipt.message.payload_mode == 'KV'
    # Subsequent greedy Decode works from this retained state.
    s.generate_message('B', 'A', max_new_tokens=2)


@pytest.mark.parametrize('plan', [
    ('token', 'token'), ('kv:v0','kv:v0'), ('kv:v1','kv:v1'),
    ('token','kv:v1','token','kv:v0'), ('kv:v0','kv:v1','kv:v0','kv:v1'),
    ('kv:v1','token','kv:v1','token','kv:v0','kv:v1')])
@pytest.mark.parametrize('summarizer', ['A','B'])
def test_bidirectional_modes_final_actual_cache_and_requested_executed(plan, summarizer):
    choices = tuple(HandoffChoice.parse(x) for x in plan)
    b, runtime = build(plan=choices, rounds=len(plan)//2)
    runtime.config = replace(runtime.config, summarizer=summarizer)
    original = runtime.session.generate_local_message
    final_states = []
    def final(agent_id, **kw):
        final_states.append(runtime.session.state(agent_id))
        return original(agent_id, **kw)
    runtime.session.generate_local_message = final
    b.prefill = lambda **kw: pytest.fail('hidden clean prefill')
    result = runtime.run()
    assert result.status == 'completed', result.error
    assert len(result.turns) == len(plan) == len(result.handoffs)
    for choice, log, turn in zip(choices, result.handoffs, result.turns):
        assert log.requested_mode == log.executed_mode == choice.mode.value
        assert log.requested_kv_strategy == log.executed_kv_strategy == (
            choice.kv_strategy.value if choice.kv_strategy else None)
        assert log.receiver_message_prefill_count == (log.message_content_token_count if choice.mode == CommunicationMode.TOKEN else 0)
        assert log.cache_ledger_consistent and log.append_succeeded
        assert log.error is log.fallback_reason is None
        assert log.token_protocol_bytes + log.incremental_kv_envelope_bytes == turn.receipt.wire_byte_count
    assert result.final_message.source_agent == summarizer
    final = runtime.session.state(summarizer)
    before = final_states[0]
    assert final.token_ledger[:len(before.token_ledger)] == before.token_ledger
    for layer, (k,v) in zip(final.cache.layers, copies(before.cache)):
        assert torch.equal(layer.keys[...,:k.shape[2],:], k)
        assert torch.equal(layer.values[...,:v.shape[2],:], v)
    for n in ('A','B'):
        state = runtime.session.state(n)
        state.validate()
        assert len(state.sent_messages) == len(state.received_messages) == len(plan)//2
    require_disjoint_caches(runtime.session.state('A').cache, runtime.session.state('B').cache)


@pytest.mark.parametrize('failure', ['corrupt','contract','append','closing','rope','sender_close','ledger'])
def test_failed_kv_transaction_does_not_commit_sequence_cache_or_fallback(monkeypatch, failure):
    from token_kv_agent_system import incremental_kv_receive as module
    b, runtime = build(choice=HandoffChoice.parse('kv:v1'))
    s = runtime.session
    s.generate_message('A','B',max_new_tokens=3)
    states = {n:s.state(n) for n in ('A','B')}
    saved = {n:copies(states[n].cache) for n in states}
    if failure == 'corrupt':
        class Corrupt(InMemoryByteTransport):
            def transfer(self, data, **kw):
                return super().transfer(data, **kw)[:-1]
        s.transport = Corrupt()
    elif failure == 'contract':
        contract = inspect_model_contract(b)
        from token_kv_agent_system.incremental_kv_protocol import decode_incremental_message
        class ChangedContract(InMemoryByteTransport):
            def transfer(self, data, **kw):
                data = super().transfer(data, **kw)
                envelope = decode_incremental_message(data, expected_contract=contract,
                    source='A', target='B', sequence=1)
                return encode_incremental_message(replace(envelope,
                    model_contract=replace(contract, config_sha256='0'*64)))
        s.transport = ChangedContract()
    elif failure == 'append':
        monkeypatch.setattr(module, 'append_delta', lambda *a, **kw: (_ for _ in ()).throw(ValueError('append failure')))
    elif failure == 'rope':
        monkeypatch.setattr(b._model.model.rotary_emb, 'rope_type', 'dynamic')
    elif failure == 'ledger':
        s._states['B'] = replace(states['B'], cached_token_count=states['B'].cached_token_count-1)
        states['B'] = s.state('B')
    else:
        original = b.continue_from_cache
        frame = states['B'].framing.incoming('A','B')
        def fail(ids, **kw):
            out = original(ids, **kw)
            if (failure == 'closing' and tuple(ids) == frame.closing_ids) or (
                    failure == 'sender_close' and kw['logical_cached_token_count'] == states['A'].cached_token_count):
                raise RuntimeError('partial forward mutation')
            return out
        b.continue_from_cache = fail
    with pytest.raises((ValueError,RuntimeError)):
        s.handoff_kv('A','B',strategy=KVStrategy.ROPE_ALIGNED_DELTA)
    for n in states:
        assert s.state(n) is states[n]
        unchanged(states[n].cache, saved[n])
        assert not states[n].sent_messages and not states[n].received_messages


@pytest.mark.parametrize('bad', ['v2','v3','garbage','v0'])
def test_invalid_strategy_rejected_before_forward(bad):
    b,runtime = build()
    s=runtime.session
    s.generate_message('A','B',max_new_tokens=3)
    calls=len(b.calls)
    with pytest.raises(ValueError): s.handoff_kv('A','B',strategy=bad)
    assert len(b.calls)==calls and not s.transport.records


@pytest.mark.parametrize('mode_request', ['kv:v4','auto','kv:foo',''])
def test_invalid_manual_plan(mode_request):
    with pytest.raises(ValueError): HandoffChoice.parse(mode_request)


def test_plan_length_and_cli_validation_before_model_load():
    from token_kv_agent_system.discussion_demo import main
    with pytest.raises(ValueError, match='plan length'):
        DiscussionConfig(handoff_plan=(HandoffChoice(),))
    for args in [['--mode','kv'], ['--mode','token','--kv-strategy','v0'],
                 ['--handoff-plan','token,kv:v1'], ['--handoff-plan','kv:v2']]:
        with pytest.raises(SystemExit) as e: main(args)
        assert e.value.code == 2


def test_empty_content_is_explicit_failure_and_runtime_stops_round():
    b,runtime=build(choice=HandoffChoice.parse('kv:v0'), raw=(2,))
    result=runtime.run()
    assert result.status=='failed' and not result.turns and result.final_message is None
    assert 'empty message content' in result.error
    assert len(result.handoffs)==1 and result.handoffs[0].executed_mode is None
    assert result.handoffs[0].fallback_reason is None
    assert not runtime.session.transport.records


def test_actual_corrupt_decode_stops_round_with_attempt_wire_bytes():
    b,runtime=build(choice=HandoffChoice.parse('kv:v1'))
    class Corrupt(InMemoryByteTransport):
        def transfer(self,data,**kw): return super().transfer(data,**kw)[:-1]
    runtime.session.transport=Corrupt()
    result=runtime.run()
    assert result.status=='failed' and not result.turns and result.final_message is None
    assert len(result.handoffs)==1
    h=result.handoffs[0]
    assert h.incremental_kv_envelope_bytes>0 and h.error
    assert h.executed_mode is None and h.executed_kv_strategy is None
    assert h.receiver_cache_length_before==h.receiver_cache_length_after
    assert h.fallback_reason is None


def test_incompatible_actual_model_weights_rejected_before_transport():
    a=make_backend(); b=make_backend()
    with torch.no_grad(): next(b._model.parameters()).add_(0.01)
    runtime=DiscussionRuntime(a,b,config=DiscussionConfig(rounds=1,max_new_tokens=2,
        handoff_choice=HandoffChoice.parse('kv:v0')))
    result=runtime.run()
    assert result.status=='failed' and 'incompatible checkpoint' in result.error
    assert not result.turns and not runtime.session.transport.records
    assert runtime.session.state('A').draft_message is not None
    assert not runtime.session.state('B').received_messages


def test_v1_unaudited_model_rejected_and_no_v0_fallback(monkeypatch):
    b,runtime=build(choice=HandoffChoice.parse('kv:v1'))
    monkeypatch.setattr(b._model.config,'model_type','unaudited')
    result=runtime.run()
    assert result.status=='failed' and 'audited' in result.error
    assert not result.turns and result.final_message is None
    assert not runtime.session.state('B').received_messages


def test_receiver_duplicate_sequence_entry_does_not_forward():
    b,runtime=build()
    s=runtime.session
    m=s.generate_message('A','B',max_new_tokens=3)
    contract=inspect_model_contract(b)
    envelope,_=prepare_incremental_message(s.state('A'),m,contract=contract)
    packet=encode_incremental_message(envelope)
    s.handoff_kv('A','B',strategy=KVStrategy.RAW_DELTA)
    old=s.state('B'); saved=copies(old.cache); b.calls.clear()
    with pytest.raises(ValueError,match='sequence'):
        receive_incremental_kv(old,packet,contract=contract,source='A',sequence=1,
            strategy=KVStrategy.RAW_DELTA)
    assert b.calls==[]
    unchanged(old.cache,saved)


def test_receiver_budget_before_framing_or_wire_and_retry_success():
    b,runtime=build()
    s=runtime.session
    s.generate_message('A','B',max_new_tokens=3)
    old={n:s.state(n) for n in ('A','B')}
    s.max_context_tokens=len(old['B'].token_ledger)+1
    with pytest.raises(ValueError,match='budget'):
        s.handoff_kv('A','B',strategy=KVStrategy.RAW_DELTA)
    assert not s.transport.records and all(s.state(n) is old[n] for n in old)
    s.max_context_tokens=4096
    receipt=s.handoff_kv('A','B',strategy=KVStrategy.RAW_DELTA)
    assert receipt.message.sequence==1 and len(s.state('A').sent_messages)==1


def test_fixed_step_all_three_modes_same_snapshots_and_content():
    from token_kv_agent_system.discussion_validation import fixed_step
    b=make_backend()
    result=fixed_step(b,question='23 + 48 = ?',max_new_tokens=2)
    assert [row['mode'] for row in result['modes']]==['token','kv:v0','kv:v1']
    assert len({tuple(row['receipt']['received_token_ids']) for row in result['modes']})==1
    assert all(row['status']=='completed' and row['snapshot_unchanged'] for row in result['modes'])
    assert [row['receiver_message_prefill_count'] for row in result['modes']]==[2,0,0]
