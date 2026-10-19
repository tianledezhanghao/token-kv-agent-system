from dataclasses import asdict, replace
from threading import Event
import json
import pytest
from token_kv_agent_system.incremental_auto import (
    IncrementalAutoConfig, IncrementalCostProfile, StageCost, backend_cost_identity, estimate_incremental_wire_bytes)
from token_kv_agent_system.network.handoff_network import HandoffNetworkConfig
from token_kv_agent_system.discussion import DiscussionConfig, DiscussionRuntime
from token_kv_agent_system.communication import HandoffChoice
from token_kv_agent_system.transport import InMemoryByteTransport
from test_incremental_kv_transport import make_backend


def profile(b, mode, token_cost=100000):
    return IncrementalCostProfile('synthetic-routing-v1','synthetic/test_only',
        {'sender':backend_cost_identity(b), 'receiver':backend_cost_identity(b)},
        {'kv_contract_startup':StageCost(0), 'token_handoff':StageCost(token_cost), 'kv_base':StageCost(1), 'rope':StageCost(1),
         'suffix':StageCost(0,per_recomputed_token_ms=1), 'selection':StageCost(1), 'slot':StageCost(1)},
        0,(0,128),(0,10000),'Routing only; artificial costs cannot evidence real speed.')


@pytest.mark.parametrize('candidate',['token','kv:v0','kv:v1','kv:v2','kv:v3'])
def test_auto_calls_unique_real_session_path_and_continues(candidate, monkeypatch):
    b=make_backend()
    if candidate=='kv:v3':
        original=b.decode
        b.decode=lambda ids, **kw: '17' if tuple(ids)==(100,) else original(ids,**kw)
    auto=IncrementalAutoConfig('performance_experiment',('token',candidate if candidate!='token' else 'kv:v0'),candidate=='kv:v3',profile(b,candidate,token_cost=.001 if candidate=='token' else 100000))
    events=[]
    r=DiscussionRuntime(b,config=DiscussionConfig(rounds=2,max_new_tokens=2,auto=auto,
        network=HandoffNetworkConfig(mode='trace',trace_mbps=(20,1000,50,500))),event_sink=lambda k,v:events.append((k,v)))
    b.prefill=lambda **kw: pytest.fail('Canonical or Full Prefill after initialization')
    calls=[]
    token,kv=r.session.handoff,r.session.handoff_kv
    def traced_token(*a,**kw):calls.append('token');return token(*a,**kw)
    def traced_kv(*a,**kw):calls.append(kw['strategy'].value);return kv(*a,**kw)
    monkeypatch.setattr(r.session,'handoff',traced_token)
    monkeypatch.setattr(r.session,'handoff_kv',traced_kv)
    result=r.run()
    assert result.status=='completed',result.error
    assert len(calls)==4
    assert calls==[HandoffChoice.parse(candidate).kv_strategy.value if candidate!='token' else 'token']*4
    assert [h.network['bandwidth_mbps'] for h in result.handoffs]==[20,1000,50,500]
    assert [k for k,v in events].count('AUTO_DECISION')==4
    for turn,h in zip(result.turns,result.handoffs):
        assert h.requested_mode=='auto' and h.actual_execution_status=='succeeded'
        assert h.predicted_wire_bytes==h.actual_wire_bytes==turn.receipt.wire_byte_count
        assert h.selected_mode==h.executed_mode
        assert h.selected_kv_strategy==h.executed_kv_strategy
        assert h.auto_decision['decision_status']=='executed'
        assert h.observed_local_ms>0 and h.observed_stage_ms['handoff_total']>0
        assert h.fallback_reason is h.error is None
        assert not any('observed' in e for e in h.auto_decision['candidate_estimates'])
    assert r.session.state('A').cached_token_count==len(r.session.state('A').token_ledger)


def test_quality_default_with_opaque_backend():
    from test_discussion import ScriptBackend
    b=ScriptBackend()
    result=DiscussionRuntime(b,config=DiscussionConfig(rounds=3,auto=IncrementalAutoConfig(),
        network=HandoffNetworkConfig(mode='trace',trace_mbps=(20,1000,50,500)))).run()
    assert result.status=='completed',result.error
    assert len(result.handoffs)==6
    assert all(h.executed_mode=='token' and h.predicted_local_ms is None for h in result.handoffs)


def test_no_silent_fallback_failed_selected_kv_preserves_transaction():
    b=make_backend()
    class Corrupt(InMemoryByteTransport):
        def transfer(self,data,**kw):return super().transfer(data,**kw)[:-1]
    r=DiscussionRuntime(b,config=DiscussionConfig(rounds=1,max_new_tokens=2,
        auto=IncrementalAutoConfig('performance_experiment',('token','kv:v1'),False,profile(b,'kv:v1'))),transport=Corrupt())
    r.session.handoff=lambda *a,**kw:pytest.fail('silent TOKEN fallback')
    saved={}
    original=r.session.handoff_kv
    def capture(*a,**kw):
        saved.update({n:r.session.state(n) for n in ('A','B')})
        return original(*a,**kw)
    r.session.handoff_kv=capture
    result=r.run()
    assert result.status=='failed' and len(result.handoffs)==1
    h=result.handoffs[0]
    assert h.executed_mode is None and h.selected_mode=='kv'
    assert h.auto_decision['decision_status']=='execution_failed' and h.actual_wire_bytes>0
    assert h.fallback_reason is None
    assert all(r.session.state(n) is saved[n] for n in saved)


def test_decision_failure_no_transfer_and_event():
    b=make_backend()
    p=replace(profile(b,'kv:v0'),model_identity={'wrong':1})
    events=[]
    r=DiscussionRuntime(b,config=DiscussionConfig(rounds=1,max_new_tokens=2,
        auto=IncrementalAutoConfig('performance_experiment',('token','kv:v0'),False,p)),event_sink=lambda k,v:events.append((k,v)))
    result=r.run()
    assert result.status=='failed' and not r.session.transport.records
    assert events[-1][0]=='AUTO_DECISION_FAILED'
    assert 'model_profile_mismatch' in result.error


@pytest.mark.parametrize('manual',[dict(handoff_choice=HandoffChoice.parse('kv:v1')),
    dict(handoff_plan=(HandoffChoice(),)*4)])
def test_auto_manual_conflict_explicit(manual):
    with pytest.raises(ValueError,match='coexist'):DiscussionConfig(auto=IncrementalAutoConfig(),**manual)


def test_rest_auto_ws_events_and_snapshot_projection():
    from fastapi.testclient import TestClient
    from token_kv_agent_system.dashboard.app import create_app
    from token_kv_agent_system.dashboard.discussion_service import DiscussionService
    from token_kv_agent_system.metrics import MetricCollector
    from test_discussion import ScriptBackend
    c=MetricCollector();s=DiscussionService(ScriptBackend(),collector=c);done=Event()
    c.subscribe(lambda e:done.set() if e.event_type.value=='DISCUSSION_COMPLETED' else None)
    with TestClient(create_app(collector=c,discussion_service=s)) as client:
        with client.websocket_connect('/ws/events') as ws:
            resp=client.post('/api/discussions',json=dict(question='Q',mode='auto',rounds=2,
                max_new_tokens=3,network_mode='trace',bandwidth_trace=[20,1000,50,500]))
            assert resp.status_code==202,resp.text
            events=[]
            while True:
                e=ws.receive_json();events.append(e)
                if e['type'] in ('DISCUSSION_COMPLETED','DISCUSSION_FAILED'):break
        assert done.wait(5)
        record=client.get('/api/discussions/'+resp.json()['session_id']).json()
        assert record['status']=='completed'
        assert len(record['auto_decisions'])==4
        assert [h['network']['bandwidth_mbps'] for h in record['handoffs']]==[20,1000,50,500]
        assert [e['metadata']['revision'] for e in events]==list(range(1,len(events)+1))
        assert [e['metadata']['handoff'] for e in events if e['type']=='HANDOFF_COMPLETED']==record['handoffs']
        assert events[-1]['metadata']['summary']['token_handoffs']==4


@pytest.mark.parametrize('bad',[dict(bandwidth_mbps=0),dict(base_latency_ms=-1),dict(network_mode='trace'),
    dict(network_mode='trace',bandwidth_trace=[-1]),dict(bandwidth_trace=[20]),dict(auto_candidates=[]),
    dict(auto_policy='oracle'),dict(auto_candidates=['kv:v4']),dict(cost_profile={'canonical':True}),
    dict(auto_policy='performance_experiment',auto_candidates=['kv:v3']),
    dict(kv_recompute_ratio=2),dict(kv_recompute_tokens=1,kv_recompute_ratio=.2)])
def test_auto_rest_schema_boundaries(bad):
    from token_kv_agent_system.dashboard.discussion_service import DiscussionStartRequest
    with pytest.raises(ValueError):DiscussionStartRequest(question='Q',mode='auto',**bad)


def test_cli_auto_options_validation_before_load(monkeypatch,tmp_path):
    from token_kv_agent_system import discussion_demo as module
    monkeypatch.setattr(module,'HuggingFaceModelBackend',lambda *a,**kw:pytest.fail('should not load'))
    for argv in [['--mode','auto','--handoff-plan','token,token'],['--mode','auto','--bandwidth-mbps','0'],
        ['--mode','auto','--network-mode','trace'],['--mode','auto','--auto-policy','performance_experiment','--auto-candidates','kv:v3']]:
        with pytest.raises(SystemExit) as error:module.main(argv)
        assert error.value.code==2


def test_independent_calibration_samples_warmup_domains_and_provenance():
    from token_kv_agent_system.dynamic_auto_validation import calibrate
    from token_kv_agent_system.incremental_kv_protocol import inspect_model_contract
    b=make_backend();original=b.decode
    b.decode=lambda ids,**kw:'17' if tuple(ids)==(100,) else original(ids,**kw)
    records={}
    p=calibrate(b,inspect_model_contract(b),lambda name,row:records.update({name:row}))
    samples=records['calibration-samples']
    assert len(samples)==30 and sum(s['warmup'] for s in samples)==10
    assert p.provenance=='measured_incremental' and p.sample_count==20
    assert all(s['actual_wire_bytes']>0 and s['observed_local_ms']>0 for s in samples)
    assert 'No Canonical data' in p.notes


def test_valid_cli_auto_serialized_output(monkeypatch,tmp_path):
    from token_kv_agent_system import discussion_demo as module
    b=make_backend()
    monkeypatch.setattr(module,'HuggingFaceModelBackend',lambda *a,**kw:b)
    out=tmp_path/'auto.json'
    assert module.main(['--mode','auto','--rounds','1','--max-new-tokens','2',
        '--network-mode','trace','--bandwidth-trace','20,1000','--output',str(out)])==0
    logs=json.loads(out.read_text())['discussion']['handoffs']
    assert [h['requested_mode'] for h in logs]==['auto']*2
    assert [h['executed_mode'] for h in logs]==['token']*2


def test_performance_rest_profile_real_kv_path_and_v3_risk():
    from token_kv_agent_system.dashboard.discussion_service import DiscussionStartRequest, DiscussionService
    from token_kv_agent_system.metrics import MetricCollector
    b=make_backend();original=b.decode
    b.decode=lambda ids,**kw:'17' if tuple(ids)==(100,) else original(ids,**kw)
    c=MetricCollector();s=DiscussionService(b,collector=c);done=Event()
    c.subscribe(lambda e:done.set() if e.event_type.value in ('DISCUSSION_COMPLETED','DISCUSSION_FAILED') else None)
    try:
        req=DiscussionStartRequest(question='Q',rounds=1,max_new_tokens=2,mode='auto',
            auto_policy='performance_experiment',auto_candidates=['token','kv:v3'],allow_experimental_v3=True,
            cost_profile=json.loads(json.dumps(asdict(profile(b,'kv:v3')))))
        rec=s.start(req);assert done.wait(10)
        rec=s.get(rec['session_id'])
        assert rec['status']=='completed',rec['error']
        assert rec['summary']['kv_v3_handoffs']==2
        assert all(h['auto_decision']['v3_semantic_risk'] for h in rec['handoffs'])
        assert all(h['selected_kv_strategy']==h['executed_kv_strategy']=='numeric_slot_delta' for h in rec['handoffs'])
    finally:s.close()
