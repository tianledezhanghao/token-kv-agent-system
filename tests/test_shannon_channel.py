"""Independent unit oracles, reproducibility, PATCH and bounded actual CPU routing."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import math
import statistics
import threading

import pytest

from token_kv_agent_system.network.shannon import (
    ShannonChannelConfig, capacity_mbps, crossover_analysis)
from token_kv_agent_system.network.handoff_network import HandoffNetwork, HandoffNetworkConfig
from token_kv_agent_system.dashboard.discussion_service import (
    DiscussionService, DiscussionStartRequest, DiscussionNetworkRequest)
from token_kv_agent_system.discussion import DiscussionConfig, DiscussionRuntime, estimate_discussion_context
from token_kv_agent_system.incremental_auto import IncrementalAutoConfig
from test_discussion import ScriptBackend


def network(**kw):
    return HandoffNetwork(HandoffNetworkConfig(mode='shannon',channel=ShannonChannelConfig(**kw)))


def test_independent_units_and_network_time_oracles():
    assert capacity_mbps(20,0) == pytest.approx(20)
    assert capacity_mbps(1,10) == pytest.approx(math.log2(11))
    s = network(bandwidth_mhz=20,snr_mean_db=0,snr_sigma_db=0,efficiency=.8).snapshot(1)
    assert s.bandwidth_mbps == pytest.approx(16)
    assert s.transfer_ms(2000) == pytest.approx(1)  # 16000 bits / 16e6 bps = 1ms
    s = replace(s,base_latency_ms=5)
    assert s.transfer_ms(2000) == pytest.approx(6)
    for snr in [-120,120]:
        assert math.isfinite(capacity_mbps(100000,snr)) and capacity_mbps(.001,snr)>0
    with pytest.raises(ValueError): s.transfer_ms(10**400)


@pytest.mark.parametrize('kw',[
    {'bandwidth_mhz':0},{'bandwidth_mhz':float('inf')},{'snr_mean_db':float('nan')},
    {'snr_sigma_db':-1},{'correlation':1},{'correlation':-1},{'efficiency':0},{'efficiency':1.1},
    {'seed':True},{'seed':-1},{'snr_min_db':20,'snr_max_db':10},{'snr_max_db':121},{'efficiency':5e-324},
])
def test_illegal_parameters_fail(kw):
    with pytest.raises(ValueError): ShannonChannelConfig(**kw)


def test_unknown_channel_fields_are_validation_errors_not_server_errors():
    with pytest.raises(ValueError,match='unknown'):
        DiscussionStartRequest(question='q',network_mode='shannon',channel={'bandwidth_mbps':20})
    with pytest.raises(ValueError): ShannonChannelConfig.from_dict([])


def trajectory(seed=42,**kw):
    n=network(seed=seed,**kw)
    return [n.snapshot(i).channel['snr_db'] for i in range(1,50)]


def test_seeds_zero_sigma_clipping_and_session_isolation():
    a=trajectory(); assert a==trajectory() and a!=trajectory(43)
    assert trajectory(snr_sigma_db=0)==[10]*49
    assert all(8<=v<=12 for v in trajectory(snr_min_db=8,snr_max_db=12))
    n=network(); original=n.snapshot(1)
    for _ in range(20): n.current(); assert n.snapshot(1) is original
    assert n.snapshot(2).channel['snr_db']==a[1]
    # Other sessions' consumption cannot alter this stream.
    trajectory(999)
    assert n.snapshot(3).channel['snr_db']==a[2]


@pytest.mark.parametrize('rho',[0,.85])
def test_stationary_distribution_and_empirical_lag_correlation(rho):
    n=network(correlation=rho,snr_mean_db=0,snr_sigma_db=4)
    xs=[n.snapshot(i).channel['snr_db'] for i in range(1,12001)]
    assert abs(statistics.mean(xs)) < .3
    assert statistics.stdev(xs)==pytest.approx(4,abs=.2)
    assert statistics.correlation(xs[:-1],xs[1:])==pytest.approx(rho,abs=.04)


def test_concurrent_reads_freeze_once_and_patch_preserves_normalized_state():
    n=network(); control=network()
    with ThreadPoolExecutor(max_workers=8) as pool:
        snapshots=list(pool.map(lambda _:n.snapshot(1),range(32)))
    assert all(s is snapshots[0] for s in snapshots)
    first=asdict(snapshots[0]); control.snapshot(1)
    config=HandoffNetworkConfig(mode='shannon',channel=ShannonChannelConfig(snr_mean_db=20,snr_sigma_db=8))
    n.update(config)
    assert n.current()['next_unfrozen_handoff_index']==2
    after=n.snapshot(2); reference=control.snapshot(2)
    assert after.channel['snr_db']==pytest.approx(20+2*(reference.channel['snr_db']-10))
    assert after.config_revision==1 and asdict(n.snapshot(1))==first
    n.update(replace(config,channel=replace(config.channel,seed=123)))
    assert n.snapshot(3).channel['snr_db']==network(seed=123,snr_mean_db=20,snr_sigma_db=8).snapshot(1).channel['snr_db']


def costs(p_t=100,p_k=20,d_t=100,d_k=100100,source='predicted'):
    return [dict(candidate=c,predicted_local_ms=p,predicted_wire_bytes=d,cost_source=source)
            for c,p,d in [('token',p_t,d_t),('kv:v0',p_k,d_k)]]


def test_crossover_units_inverse_and_nonexistent_cases():
    c=network(bandwidth_mhz=10).snapshot(1).channel
    crossing=crossover_analysis(costs(),c)
    assert crossing['effective_rate_mbps']==10
    assert .8*capacity_mbps(10,crossing['snr_threshold_db'])==pytest.approx(10)
    assert crossing['above']=='kv:v0' and crossing['below']=='token'
    assert crossover_analysis(costs(p_k=120))['status']=='no_positive_crossover'
    assert crossover_analysis(costs(p_k=100,d_k=100))['preferred']=='equal'
    assert crossover_analysis(costs(p_k=100))['status']=='no_positive_crossover'
    assert crossover_analysis(costs()[:1])['status']=='unavailable'
    assert crossover_analysis(costs(source='synthetic'))['cost_source']=='synthetic_assumption'
    assert math.isfinite(crossover_analysis(costs(d_k=10**15),c)['snr_threshold_db'])


@pytest.mark.parametrize('rounds',[1,4,10,20])
def test_bounded_discussion_rounds_real_token_receipts_and_final(rounds):
    b=ScriptBackend([(100,2)]*(2*rounds+1))
    config=DiscussionConfig(rounds=rounds,max_new_tokens=2,max_context_tokens=32768,
                            network=HandoffNetworkConfig(mode='shannon',channel=ShannonChannelConfig()))
    assert estimate_discussion_context(b,config)['context_limit']==32768
    result=DiscussionRuntime(b,config=config).run()
    assert result.status=='completed',result.error
    assert len(result.turns)==len(result.handoffs)==2*rounds
    assert result.final_message and result.final_text
    assert all(h.append_succeeded and h.executed_mode=='token' for h in result.handoffs)
    assert [h.network['handoff_index'] for h in result.handoffs]==list(range(1,2*rounds+1))
    assert len({h.network['channel']['snr_db'] for h in result.handoffs})>1


def test_long_session_rejected_before_prefill_or_slot_reservation():
    b=ScriptBackend(); service=DiscussionService(b)
    before=list(b.calls)
    try:
        with pytest.raises(ValueError,match='context'):
            service.start(DiscussionStartRequest(question='q',rounds=20,advanced_rounds=True,max_context_tokens=128))
        assert b.calls==before and not service.latest()['active_session_id']
        with pytest.raises(ValueError): DiscussionStartRequest(question='q',rounds=11)
    finally: service.close()


def test_live_patch_api_future_only_and_no_legacy_rate_override():
    b=ScriptBackend([(100,2)]*9); service=DiscussionService(b)
    blocked=threading.Event(); resume=threading.Event()
    def event(e):
        if e.event_type.value=='HANDOFF_STARTED' and e.metadata['sequence']==1:
            blocked.set(); resume.wait(10)
    service.collector.subscribe(event)
    try:
        record=service.start(DiscussionStartRequest(question='q',rounds=4,max_new_tokens=2,max_context_tokens=32768,
            network_mode='shannon',channel=asdict(ShannonChannelConfig())))
        assert blocked.wait(5); sid=record['session_id']
        before=service.get(sid)['active_handoff']['network']
        with pytest.raises(ValueError,match='derived'): service.update_network(sid,DiscussionNetworkRequest(bandwidth_mbps=99))
        update=service.update_network(sid,DiscussionNetworkRequest(channel={'snr_mean_db':-10,'bandwidth_mhz':5}))
        assert update['effective_from_handoff_index']==2 and update['channel_bandwidth_changed']
        assert service.get(sid)['active_handoff']['network']==before
        resume.set(); service.close()
        final=service.get(sid)
        assert final['status']=='completed'
        assert final['handoffs'][0]['network']['config_revision']==0
        assert all(h['network']['config_revision']==1 for h in final['handoffs'][1:])
        assert final['handoffs'][0]['network']['channel']['bandwidth_mhz']==20
    finally: resume.set();service.close()


def test_shannon_auto_actual_kv_and_token_paths_use_one_snapshot():
    from test_incremental_kv_transport import make_backend
    b=make_backend(); b.outputs=[59]*100
    n=network(bandwidth_mhz=1000,snr_mean_db=40,snr_sigma_db=0)
    def sink(kind,data):
        if kind=='HANDOFF_STARTED' and data['sequence']==1:
            n.update(HandoffNetworkConfig(mode='shannon',channel=ShannonChannelConfig(
                bandwidth_mhz=.001,snr_mean_db=-10,snr_sigma_db=0)))
    result=DiscussionRuntime(b,config=DiscussionConfig(rounds=1,max_new_tokens=2,
        auto=IncrementalAutoConfig(policy='performance_experiment',idealized=True)),network=n,event_sink=sink).run()
    assert result.status=='completed',result.error
    assert [h.executed_mode for h in result.handoffs]==['kv','token']
    assert all(h.wire_prediction_error_bytes==0 for h in result.handoffs)
    for h in result.handoffs:
        d=h.auto_decision
        assert d['network']==h.network and d['profile_provenance']=='synthetic'
        assert d['candidate_set']==['token','kv:v0']
        assert h.predicted_local_ms+h.simulated_transfer_ms==pytest.approx(h.predicted_total_ms)
    assert result.handoffs[0].receiver_message_prefill_count==0
    assert result.handoffs[1].receiver_message_prefill_count>0


def test_valid_generalized_profile_uses_shannon_and_rejects_other_model():
    from test_generalized_cost import runtime_fixture
    from token_kv_agent_system.generalized_cost import GeneralizedCostProfile, digest, workload_features, environment_identity
    from token_kv_agent_system.incremental_auto import backend_cost_identity, IncrementalSelector
    b,r,a,z,m=runtime_fixture()
    identity={k:backend_cost_identity(b) for k in ('sender','receiver')}
    env={k:environment_identity(b) for k in ('sender','receiver')}
    f=workload_features(a,z,m,'token',contract_ready=True,recomputed_tokens=0)
    rows=[dict(candidate=c,features=f,local_ms=t,spread_ms=1,samples=2,wire_bytes=100)
          for c,t in [('token',100),('kv:v0',20)]]
    p=GeneralizedCostProfile('shannon-fixture-only',identity,env,rows,4,'2026-10-16T00:00:00Z',digest(rows),'Engineering fixture; not hardware evidence.')
    selector=IncrementalSelector(IncrementalAutoConfig('performance_experiment',profile=p))
    n=network(bandwidth_mhz=1000,snr_mean_db=40,snr_sigma_db=0).snapshot(1)
    _,d=selector.select(a,z,m,n,kv_contract_ready=True,require_workload_domain=True)
    assert d['selected_candidate']=='kv:v0' and d['profile_provenance']=='measured'
    assert all(e['cost_source']=='predicted' for e in d['candidate_estimates'])
    other_identity={**identity,'sender':{'model_id':'another-3B'}}
    other_env={**env,'sender':{**env['sender'],'model':other_identity['sender']}}
    p=replace(p,model_identity=other_identity,environment=other_env)
    with pytest.raises(ValueError,match='model_profile_mismatch'):
        IncrementalSelector(IncrementalAutoConfig('performance_experiment',profile=p)).select(a,z,m,n,kv_contract_ready=True)


def test_twenty_rounds_actual_random_tiny_cpu_cache_decode():
    from test_incremental_kv_transport import make_backend
    b=make_backend()
    b._model.config.max_position_embeddings=8192
    b.outputs=[59]*100
    result=DiscussionRuntime(b,config=DiscussionConfig(rounds=20,max_new_tokens=1,max_context_tokens=8192,
        network=HandoffNetworkConfig(mode='shannon',channel=ShannonChannelConfig()))).run()
    assert result.status=='completed',result.error
    assert len(result.handoffs)==40 and result.final_message
    assert all(h.append_succeeded and h.receiver_message_prefill_count==1 for h in result.handoffs)
    assert all(result.final_cache_lengths[n]<8192 for n in ('A','B'))


def test_long_session_resource_failure_is_actionable_http_503():
    from types import SimpleNamespace
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from token_kv_agent_system.dashboard.app import create_app
    b=ScriptBackend();b.metadata=SimpleNamespace(device='mps');b._torch=None
    service=DiscussionService(b)
    with patch('token_kv_agent_system.local_qwen_comparison.memory',return_value={'pressure_level':4}), \
         TestClient(create_app(collector=service.collector,discussion_service=service)) as client:
        r=client.post('/api/discussions',json=dict(question='q',rounds=4,max_new_tokens=2,max_context_tokens=32768))
        assert r.status_code==503 and 'critical resource pressure' in r.json()['detail']
        assert not service._active and not b.calls


def test_mid_run_resource_abort_retains_completed_rounds_without_final_answer():
    from types import SimpleNamespace
    from unittest.mock import patch
    b=ScriptBackend([(100,2)]*9)
    runtime=DiscussionRuntime(b,config=DiscussionConfig(rounds=4,max_new_tokens=2,max_context_tokens=32768))
    runtime._resource_backends=[(SimpleNamespace(_torch=None),{'swap_used_mib':100})]
    def memory(_):
        return {'pressure_level':4 if runtime.session.state('B').sent_messages else 1,'swap_used_mib':100}
    with patch('token_kv_agent_system.local_qwen_comparison.memory',side_effect=memory):
        result=runtime.run()
    assert result.status=='failed' and result.termination_reason=='resource_abort'
    assert len(result.turns)==len(result.handoffs)==2
    assert result.final_message is None and result.final_text is None
    assert all(h.append_succeeded for h in result.handoffs)
