from dataclasses import asdict, replace
from types import SimpleNamespace
import pytest

from token_kv_agent_system.network.handoff_network import HandoffNetwork, HandoffNetworkConfig
from token_kv_agent_system.incremental_auto import (
    IncrementalAutoConfig, IncrementalCostProfile, StageCost, IncrementalSelector, CANDIDATES)


@pytest.mark.parametrize('bad', [0, -1, float('nan'), float('inf'), True, '20'])
def test_invalid_bandwidth(bad):
    with pytest.raises(ValueError): HandoffNetworkConfig(bandwidth_mbps=bad)
    with pytest.raises(ValueError): HandoffNetworkConfig(mode='trace', trace_mbps=(bad,))


@pytest.mark.parametrize('kwargs', [dict(base_latency_ms=-1), dict(base_latency_ms=float('nan')),
    dict(base_latency_ms=float('inf')), dict(mode='trace'), dict(trace_mbps=(20,)), dict(mode='clock')])
def test_invalid_network_config(kwargs):
    with pytest.raises(ValueError): HandoffNetworkConfig(**kwargs)


def test_trace_replay_failure_retry_exhaustion_and_mode_switch():
    c = HandoffNetworkConfig(mode='trace', trace_mbps=(20,1000,50,500))
    a,b = HandoffNetwork(c), HandoffNetwork(c)
    for i in range(1,7):
        s = a.snapshot(i)
        assert s == b.snapshot(i) == a.snapshot(i)
        assert s.bandwidth_mbps == (20,1000,50,500,500,500)[i-1]
        assert s.trace_exhausted == (i>4)
    a.update(HandoffNetworkConfig(bandwidth_mbps=99, base_latency_ms=4))
    assert a.snapshot(6).bandwidth_mbps == 500
    assert a.snapshot(7).bandwidth_mbps == 99
    a.update(c)
    assert a.snapshot(8).bandwidth_mbps == 20
    assert a.snapshot(8).simulation_step == 0
    with pytest.raises(ValueError): a.snapshot(10)
    with pytest.raises(ValueError): a.snapshot(0)


def test_network_updates_freeze_read_consistency():
    from concurrent.futures import ThreadPoolExecutor
    net = HandoffNetwork()
    net.snapshot(1)
    def read(_): return net.snapshot(1)
    with ThreadPoolExecutor(4) as executor:
        before = list(executor.map(read, range(20)))
        net.update(HandoffNetworkConfig(bandwidth_mbps=1000))
        after = list(executor.map(read, range(20)))
    assert before == after
    assert net.snapshot(2).bandwidth_mbps == 1000


def fixture_selector(monkeypatch, stages=None, candidates=('token','kv:v0'), policy='performance_experiment', **kw):
    import token_kv_agent_system.incremental_auto as module
    monkeypatch.setattr(module, 'backend_cost_identity', lambda b: {'id':'test'})
    monkeypatch.setattr(module, 'estimate_incremental_wire_bytes', lambda *a: 10000)
    profile = IncrementalCostProfile('synthetic-v1','synthetic/test_only',
        {'sender':{'id':'test'},'receiver':{'id':'test'}}, stages or {
        'token_handoff':StageCost(100), 'kv_base':StageCost(10), 'rope':StageCost(2),
        'suffix':StageCost(0, per_recomputed_token_ms=4), 'selection':StageCost(1), 'slot':StageCost(5)},
        0,(0,100000),(0,100000),'Fabricated decision logic fixture, never a speed measurement.')
    profile = replace(profile, stages={**profile.stages, 'kv_contract_startup':StageCost(0)})
    config = IncrementalAutoConfig(policy,candidates,'kv:v3' in candidates,profile,**kw)
    sender = receiver = SimpleNamespace(backend=None, cached_token_count=50)
    message = SimpleNamespace(token_count=10,termination_reason='max_new_tokens')
    return IncrementalSelector(config), sender, receiver, message


def test_hand_calculated_cost_low_high_bandwidth_and_switch(monkeypatch):
    s,a,b,m=fixture_selector(monkeypatch)
    choice,low=s.select(a,b,m,HandoffNetwork(HandoffNetworkConfig(bandwidth_mbps=.01,base_latency_ms=3)).snapshot(1))
    assert low['selected_candidate']=='token'
    assert low['predicted_wire_bytes']==50
    assert low['simulated_transfer_ms']==43
    assert low['predicted_total_ms']==143
    choice,high=s.select(a,b,m,HandoffNetwork(HandoffNetworkConfig(bandwidth_mbps=1000,base_latency_ms=3)).snapshot(1))
    assert high['selected_candidate']=='kv:v0'
    assert high['predicted_total_ms']==pytest.approx(13.08)
    assert len({e['predicted_wire_bytes'] for e in high['candidate_estimates'] if e['candidate']!='token'})==1
    assert [e['predicted_local_ms'] for e in high['candidate_estimates']]==[100,10]


def test_same_cost_stable_tie_break_and_big_message(monkeypatch):
    s,a,b,m=fixture_selector(monkeypatch,candidates=('kv:v1','token'), stages={'token_handoff':StageCost(10),'kv_base':StageCost(10),'rope':StageCost(0)})
    assert s.select(a,b,m,HandoffNetwork().snapshot(1))[1]['selected_candidate']=='token'
    m.token_count=1024
    assert s.select(a,b,m,HandoffNetwork().snapshot(1))[1]['candidate_estimates'][1]['predicted_wire_bytes']==10000


def test_v2_parameters_change_cost(monkeypatch):
    s,a,b,m=fixture_selector(monkeypatch,candidates=('token','kv:v2'),recompute_ratio=.5)
    d=s.select(a,b,m,HandoffNetwork().snapshot(1))[1]
    assert next(e for e in d['candidate_estimates'] if e['candidate']=='kv:v2')['predicted_local_ms']==32


def test_quality_filter_and_missing_profile(monkeypatch):
    s,a,b,m=fixture_selector(monkeypatch,policy='quality_first')
    d=s.select(a,b,m,HandoffNetwork().snapshot(1))[1]
    assert d['selected_candidate']=='token'
    assert [e['reason'] for e in d['excluded_candidates']]==['quality_not_certified']+['not_whitelisted']*3
    s=IncrementalSelector(IncrementalAutoConfig())
    d=s.select(a,b,m,HandoffNetwork().snapshot(1))[1]
    assert d['predicted_local_ms'] is d['predicted_total_ms'] is None
    s=IncrementalSelector(IncrementalAutoConfig(policy='performance_experiment'))
    d=s.select(a,b,m,HandoffNetwork().snapshot(1))[1]
    assert d['excluded_candidates'][0]['reason']=='missing_cost_profile'


def test_profile_mismatch_domain_missing_and_zero_cost(monkeypatch):
    s,a,b,m=fixture_selector(monkeypatch)
    p=s.config.profile
    for change,reason in [(dict(model_identity={'wrong':1}),'model_profile_mismatch'),
        (dict(cache_token_range=(0,1)),'profile_out_of_domain'), (dict(stages={}),'missing_cost_profile'),
        (dict(stages={'token_handoff':StageCost(0)}),'missing_cost_profile')]:
        s=IncrementalSelector(replace(s.config,profile=replace(p,**change)))
        with pytest.raises(ValueError,match=reason):s.select(a,b,m,HandoffNetwork().snapshot(1))


@pytest.mark.parametrize('kwargs',[dict(policy='bad'),dict(candidates=()),dict(candidates=('token','token')),
    dict(candidates=('kv:v9',)),dict(policy='quality_first',candidates=('kv:v1',)),
    dict(policy='performance_experiment',candidates=('kv:v3',)),dict(recompute_tokens=-1),dict(profile={})])
def test_invalid_auto_config(kwargs):
    with pytest.raises(ValueError):IncrementalAutoConfig(**kwargs)


def test_profile_roundtrip_and_invalid_coefficients(monkeypatch):
    s,*_=fixture_selector(monkeypatch)
    assert IncrementalCostProfile.from_dict(asdict(s.config.profile))==s.config.profile
    for v in (-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):StageCost(v)
    with pytest.raises(ValueError):IncrementalCostProfile.from_dict({'canonical':True})


def test_token_kv_exact_tie_prefers_token(monkeypatch):
    s,a,b,m=fixture_selector(monkeypatch,candidates=('kv:v0','token'),
        stages={'token_handoff':StageCost(100),'kv_base':StageCost(10)})
    monkeypatch.setattr('token_kv_agent_system.incremental_auto.estimate_incremental_wire_bytes',lambda *a:11300)
    # TOKEN: 100 + 50*8/1000 = 100.4; KV: 10 + 11300*8/1000 = 100.4.
    _,d=s.select(a,b,m,HandoffNetwork(HandoffNetworkConfig(bandwidth_mbps=1)).snapshot(1))
    assert d['selected_candidate']=='token' and d['cost_gap_ms']==0


def test_context_and_message_cost_terms_and_recompute_domain(monkeypatch):
    s,a,b,m=fixture_selector(monkeypatch,candidates=('token','kv:v2'),recompute_tokens=2,
        stages={'token_handoff':StageCost(1,2,3), 'kv_base':StageCost(1),
                'rope':StageCost(1),'suffix':StageCost(0,per_recomputed_token_ms=5)})
    _,d=s.select(a,b,m,HandoffNetwork().snapshot(1))
    assert next(e for e in d['candidate_estimates'] if e['candidate']=='token')['predicted_local_ms']==321
    s=IncrementalSelector(replace(s.config,profile=replace(s.config.profile,recompute_token_range=(0,1))))
    _,d=s.select(a,b,m,HandoffNetwork().snapshot(1))
    assert any(e['reason']=='profile_recompute_out_of_domain' for e in d['excluded_candidates'])


def test_network_overflow_is_explicit_error():
    net=HandoffNetwork(HandoffNetworkConfig(bandwidth_mbps=1e-320))
    with pytest.raises(ValueError,match='finite'):net.snapshot(1).transfer_ms(100000)


def test_cold_kv_contract_cost_is_charged_once_and_missing_is_excluded(monkeypatch):
    s,a,b,m=fixture_selector(monkeypatch,candidates=('token','kv:v0'))
    p=replace(s.config.profile,stages={**s.config.profile.stages,'kv_contract_startup':StageCost(200)})
    s=IncrementalSelector(replace(s.config,profile=p))
    net=HandoffNetwork(HandoffNetworkConfig(bandwidth_mbps=1000)).snapshot(1)
    _,cold=s.select(a,b,m,net,kv_contract_ready=False)
    _,warm=s.select(a,b,m,net,kv_contract_ready=True)
    assert cold['selected_candidate']=='token' and warm['selected_candidate']=='kv:v0'
    kv_cold=next(e for e in cold['candidate_estimates'] if e['candidate']=='kv:v0')
    kv_warm=next(e for e in warm['candidate_estimates'] if e['candidate']=='kv:v0')
    assert kv_cold['predicted_local_ms']-kv_warm['predicted_local_ms']==200
    s=IncrementalSelector(replace(s.config,profile=replace(p,stages={k:v for k,v in p.stages.items() if k!='kv_contract_startup'})))
    _,d=s.select(a,b,m,net)
    assert any(e['reason']=='missing_startup_cost_profile' for e in d['excluded_candidates'])


def test_quality_profile_long_message_no_extrapolation_and_performance_rejects(monkeypatch):
    s,a,b,m = fixture_selector(monkeypatch, policy='quality_first')
    profile = replace(s.config.profile, message_token_range=(0,128))
    s = IncrementalSelector(replace(s.config, profile=profile))
    m.token_count = 512
    d = s.select(a,b,m,HandoffNetwork().snapshot(1))[1]
    assert d['selected_candidate'] == 'token' and d['cost_status'] == 'profile_out_of_domain'
    assert d['predicted_local_ms'] is d['predicted_total_ms'] is None
    assert d['predicted_wire_bytes'] == 10 + 4*512
    unsafe = IncrementalSelector(replace(s.config, policy='performance_experiment'))
    with pytest.raises(ValueError, match='profile_out_of_domain'):
        unsafe.select(a,b,m,HandoffNetwork().snapshot(1))


def test_auto_excludes_oversize_kv_without_executing(monkeypatch):
    s,a,b,m = fixture_selector(monkeypatch)
    m.token_count = 5000
    d = s.select(a,b,m,HandoffNetwork().snapshot(1))[1]
    assert d['selected_candidate'] == 'token'
    assert [e['reason'] for e in d['excluded_candidates'] if e['candidate'] in s.config.candidates] == ['protocol_capacity']
