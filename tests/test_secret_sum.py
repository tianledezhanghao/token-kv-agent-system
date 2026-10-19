"""Scene isolation/provenance tests; scripted Decode is a mock, never a demo."""
from dataclasses import asdict
import pytest
from test_discussion import ScriptBackend, CaptureTransport
from token_kv_agent_system.secret_sum import (
    PUBLIC_TASK, STAGES, SecretSumConfig, SecretSumRuntime, output_evidence, score_result, score_outputs)
from token_kv_agent_system.communication import HandoffChoice
from token_kv_agent_system.model.state_cache import require_disjoint_caches
from token_kv_agent_system.token_protocol import deserialize_token_packet


def fixture_backend():
    texts = ('My number is 17. What is your number?',
             "Agent A's number is 17. My number is 26. 17 + 26 = 43.",
             "Agent A's number is 17. Agent B's number is 26. 17 + 26 = 43.")
    backend = ScriptBackend()
    backend.scripts = [tuple(backend.chat_tokenizer.encode(t)) + (2,) for t in texts]
    return backend, texts


@pytest.mark.parametrize('a,b', [(17,26),(17,31),(42,19),(1,99),(99,1)])
def test_initial_token_ledgers_public_and_stages_isolate_private_information(a,b):
    backend = ScriptBackend()
    runtime = SecretSumRuntime(backend, config=SecretSumConfig(secret_a=a, secret_b=b))
    for name, own, peer in (('A',a,b),('B',b,a)):
        state = runtime.session.state(name)
        runtime.session.append_local_user(name, PUBLIC_TASK)
        text = backend.decode(tuple(i for i in runtime.session.state(name).token_ledger if i>=10))
        assert f'private number is {own}.' in text
        assert f'private number is {peer}.' not in text
        assert str(peer) not in PUBLIC_TASK
        assert all(str(peer) not in v for v in asdict(STAGES).values())
        assert len(runtime.session.state(name).token_ledger) == runtime.session.state(name).cached_token_count
        assert state.system_prompt == runtime.session.state(name).system_prompt
    require_disjoint_caches(runtime.session.state('A').cache, runtime.session.state('B').cache)


def test_real_decode_order_token_wire_long_term_cache_and_posthoc_score():
    backend, texts = fixture_backend()
    transport = CaptureTransport()
    events=[]
    runtime=SecretSumRuntime(backend,config=SecretSumConfig(max_new_tokens=96),transport=transport,
        event_sink=lambda kind,data: events.append((kind,data)))
    result=runtime.run()
    assert result.status=='completed',result.error
    assert [t.text for t in result.turns]+[result.final_text] == list(texts)
    assert len(backend.generation_histories)==3
    assert [kind for kind,_ in events] == ['AGENT_TURN_STARTED','AGENT_TURN_COMPLETED','HANDOFF_STARTED','HANDOFF_COMPLETED']*2+['DISCUSSION_FINALIZING']
    assert [(h.source,h.target) for h in result.handoffs]==[('A','B'),('B','A')]
    assert len(transport.packets)==2
    for packet,turn in zip(transport.packets,result.turns):
        assert deserialize_token_packet(packet).token_ids==turn.message.message_token_ids
        assert turn.receipt.wire_byte_count==len(packet)
    for name in ('A','B'):
        state=runtime.session.state(name)
        assert state.token_ledger[:len(runtime.initial_token_ledgers[name])]==runtime.initial_token_ledgers[name]
        state.validate()
        assert state.pending_token_id is None and not state.assistant_open
    score=score_result(runtime.secret_config,result)
    assert all(score[k] is True for k in ('engineering_success','b_received_a','a_received_b','information_exchange_success','task_success'))
    # Scoring cannot change any model input/output or forward call.
    calls=list(backend.calls); score_result(runtime.secret_config,result)
    assert backend.calls==calls
    assert sum(c[0]=='prefill' for c in calls)==2
    assert result.final_message.message_token_ids == tuple(backend.chat_tokenizer.encode(texts[-1])) + (2,)


@pytest.mark.parametrize('value',[0,100,-1,True,'17',1.5])
def test_invalid_secrets(value):
    with pytest.raises(ValueError): SecretSumConfig(secret_a=value)


def test_new_session_changed_secret_and_no_old_secret_reuse():
    b=ScriptBackend()
    old=SecretSumRuntime(b,config=SecretSumConfig(secret_b=26))
    new=SecretSumRuntime(b,config=SecretSumConfig(secret_b=31))
    assert '26' not in b.decode(tuple(i for i in new.session.state('B').token_ledger if i>=10))
    assert '31' in b.decode(tuple(i for i in new.session.state('B').token_ledger if i>=10))
    require_disjoint_caches(old.session.state('B').cache,new.session.state('B').cache)
    with pytest.raises(ValueError): old.config=new.config


@pytest.mark.parametrize('text',[None,'The answer is 43.','17 26 43',"Agent B's number is 26. Agent B's number is 31."])
def test_ambiguous_or_unstructured_output_is_not_success(text):
    score=score_outputs(SecretSumConfig(),text,text,text,engineering_success=True)
    assert score['information_exchange_success'] is None
    assert score['task_success'] is None


def test_correct_guess_without_handoff_is_not_information_success():
    _,texts=fixture_backend()
    score=score_outputs(SecretSumConfig(),*texts,engineering_success=True,delivered=(False,False))
    assert score['task_success'] is True
    assert score['information_exchange_success'] is False


def test_wrong_own_number_reading_and_arithmetic_are_separate():
    score=score_outputs(SecretSumConfig(),'My number is 17.',
        "Agent A's number is 18. My number is 26.",
        "Agent B's number is 26. 17 + 26 = 44.",engineering_success=True)
    assert score['b_received_a'] is False and score['a_received_b'] is True
    assert score['final_sum_correct'] is False and score['task_success'] is False


def test_failure_terminal_no_fake_answer(monkeypatch):
    b,_=fixture_backend(); runtime=SecretSumRuntime(b)
    monkeypatch.setattr(runtime.session.transport,'transfer',lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('wire broken')))
    result=runtime.run()
    assert result.status=='failed' and 'wire broken' in result.error
    assert result.final_text is None and len(b.generation_histories)==1
    assert score_result(runtime.secret_config,result)['engineering_success'] is False
    with pytest.raises(RuntimeError): runtime.run()


@pytest.mark.parametrize('plan',[('kv:v0','kv:v0'),('kv:v1','kv:v1'),('token','kv:v1')])
def test_cpu_real_delta_receive_zero_message_prefill_independent_cache_no_fallback(plan,monkeypatch):
    from test_incremental_kv_transport import make_backend
    from token_kv_agent_system.message_diagnostics import ForwardTrace
    b=make_backend()
    runtime=SecretSumRuntime(b,config=SecretSumConfig(max_new_tokens=2,handoff_plan=tuple(map(HandoffChoice.parse,plan))))
    monkeypatch.setattr(b,'prefill',lambda **kw: pytest.fail('unexpected Full Prefill'))
    token=runtime.session.handoff
    calls=[]
    def tracked(*a,**kw):
        calls.append(1)
        return token(*a,**kw)
    monkeypatch.setattr(runtime.session,'handoff',tracked)
    with ForwardTrace(b) as trace: result=runtime.run()
    assert result.status=='completed',result.error
    assert len(calls)==plan.count('token')
    for h,m in zip(result.handoffs,plan):
        assert h.fallback_reason is None and h.error is None
        assert h.receiver_message_prefill_count==(2 if m=='token' else 0)
        if m!='token':
            start,end=h.receiver_kv_target_span
            frame=runtime.session.state(h.target).framing.incoming(h.source,h.target)
            idx=next(i for i,c in enumerate(trace.calls) if c['input_ids']==list(frame.prefix_ids) and c['cache_length_after']==start)
            assert trace.calls[idx+1]['input_ids']==list(frame.closing_ids)
            assert trace.calls[idx+1]['cache_length_before']==end
            assert h.incremental_kv_envelope_bytes>0 and h.append_succeeded
    runtime._verify()


@pytest.mark.parametrize('stop',[(100,2),(100,3),(100,101)])
def test_scene_eos_or_truncation_closes_message_boundaries(stop):
    b=ScriptBackend([stop]*3)
    r=SecretSumRuntime(b,config=SecretSumConfig(max_new_tokens=len(stop)))
    result=r.run()
    assert result.status=='completed'
    for n in ('A','B'):
        state=r.session.state(n)
        assert state.pending_token_id is None and not state.assistant_open
        assert state.token_ledger[-len(state.framing.assistant.closing_ids):] == state.framing.assistant.closing_ids
    for turn in result.turns:
        assert turn.message.termination_reason==('eos' if stop[-1] in (2,3) else 'max_new_tokens')
        assert turn.receipt.message.receiver_content_token_count==len(stop)-int(stop[-1] in (2,3))


def test_scene_kv_failure_never_enters_token_or_final(monkeypatch):
    from test_incremental_kv_transport import make_backend
    from token_kv_agent_system import incremental_kv_receive
    b=make_backend(); r=SecretSumRuntime(b,config=SecretSumConfig(max_new_tokens=2,handoff_choice=HandoffChoice.parse('kv:v1')))
    monkeypatch.setattr(incremental_kv_receive,'receive_incremental_kv',lambda *a,**kw: (_ for _ in ()).throw(ValueError('append broken')))
    monkeypatch.setattr(r.session,'handoff',lambda *a,**kw: pytest.fail('hidden fallback'))
    result=r.run()
    assert result.status=='failed' and 'append broken' in result.error
    assert result.final_text is None and not result.turns
    assert len(result.handoffs)==1 and not result.handoffs[0].append_succeeded
    assert result.handoffs[0].fallback_reason is None
    assert r.session.transport.records[0].byte_count>0


@pytest.mark.parametrize('text',["Agent B's number is 26.5.","Agent B's number is -26."])
def test_parser_does_not_turn_noninteger_into_private_number(text):
    assert output_evidence(text,'A')['numbers']['B'] is None


def test_parser_supported_prose_bare_sender_and_conflicting_sums():
    assert output_evidence('A number is 17.', 'B')['numbers']['A'] is None
    assert output_evidence('-17 + 26 = 9.', 'A')['equations'] == []
    assert output_evidence("Agent B's number is 26/2.", 'A')['numbers']['B'] is None
    score=score_outputs(SecretSumConfig(),'17',"A's number is 17. My number is 26. The sum of our numbers is 43.",
        "A's number is 17. B's number is 26. The sum of their numbers is 43.",engineering_success=True)
    assert score['information_exchange_success'] is score['task_success'] is True
    assert output_evidence('17','A')['numbers']['A'] is None  # Only Sender stages allow bare ownership.
    assert output_evidence('17 + 26 = 43. The sum is 44.','A')['stated_sum'] is None


def test_real_validation_harness_preserves_decode_snapshot_branches_and_no_message_control():
    from test_incremental_kv_transport import make_backend
    from token_kv_agent_system.secret_sum_validation import run_scene, fixed_decode, no_handoff
    from test_message_diagnostics import DisplayBackend
    b=make_backend()
    b.decode = lambda ids, **kw: DisplayBackend.decode(b, ids, **kw)
    config=SecretSumConfig(max_new_tokens=2)
    scene=run_scene(b,config)
    assert scene['score']['engineering_success']
    assert len(scene['wire_records'])==2
    fixed=fixed_decode(b,config)
    assert [row['mode'] for row in fixed['modes']]==['token','kv:v0','kv:v1']
    assert all(row['engineering_status']=='verified' and row['history_unchanged'] for row in fixed['modes'])
    assert fixed['modes'][1]['receiver_message_prefill_count']==0
    negative=no_handoff(b,config)
    assert negative['wire_bytes']==0 and negative['score']['information_exchange_success'] is False
