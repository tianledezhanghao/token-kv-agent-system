"""9.1: exact equations, strict input, frozen sessions and actual AUTO routing."""
from dataclasses import FrozenInstanceError
import pytest
from token_kv_agent_system.synthetic_cost import SyntheticCostAssumptions as Assumptions
from token_kv_agent_system.dashboard.discussion_service import DiscussionStartRequest, DiscussionNetworkRequest
from token_kv_agent_system.incremental_auto import IncrementalAutoConfig, IncrementalSelector
from token_kv_agent_system.network.handoff_network import HandoffNetwork, HandoffNetworkConfig
from test_generalized_cost import runtime_fixture
from test_discussion_dashboard import start_wait
from threading import Event
from fastapi.testclient import TestClient
from token_kv_agent_system.dashboard.app import create_app
from token_kv_agent_system.dashboard.discussion_service import DiscussionService
from token_kv_agent_system.metrics import MetricCollector


@pytest.fixture
def harness():
    from test_incremental_kv_transport import make_backend
    backend = make_backend()
    collector = MetricCollector()
    service = DiscussionService(backend, collector=collector)
    done = Event()
    collector.subscribe(lambda e: done.set() if e.event_type.value in ("DISCUSSION_COMPLETED", "DISCUSSION_FAILED") else None)
    with TestClient(create_app(collector=collector, discussion_service=service)) as client:
        yield client, service, backend, done, collector


@pytest.mark.parametrize('bad', [True, False, '2', None, -1, float('inf'), float('nan'), 1e10, [], {}])
@pytest.mark.parametrize('key', list('abcdef'))
def test_strict_coefficients(key, bad):
    with pytest.raises(ValueError):
        Assumptions.from_dict({key: bad})


def test_exact_default_custom_zero_and_metadata():
    features = dict(message_tokens=100, receiver_history_tokens=1000, recomputed_tokens=25)
    assert Assumptions().predict(features) == (420, 15)
    assert Assumptions(a=1,b=3,c=.01,d=2,e=.1,f=4).predict(features) == (1301,112)
    assert Assumptions(**dict.fromkeys('abcdef', 0)).predict(features) == (0,0)
    for bad in ({'units':'seconds'}, {'source':'measured'}, {'schema':'v2'}, {'x':1}):
        with pytest.raises(ValueError): Assumptions.from_dict(bad)
    with pytest.raises(FrozenInstanceError): Assumptions().a = 30


@pytest.mark.parametrize('kv,k', [('v0',0), ('v1',0), ('v2',2), ('v3',1)])
def test_actual_candidate_k_and_coefficients_change_selector(kv,k):
    b,r,a,z,m = runtime_fixture()
    b.decode = lambda ids, **kw: '42'
    def select(assumptions):
        config = IncrementalAutoConfig('performance_experiment', ('token','kv:'+kv),
            allow_experimental_v3=True, recompute_tokens=2 if kv=='v2' else None,
            idealized=True, synthetic_assumptions=assumptions)
        return IncrementalSelector(config).select(a,z,m,HandoffNetwork(HandoffNetworkConfig(bandwidth_mbps=100000)).snapshot(1))[1]
    cheap_kv = Assumptions(a=10000,b=0,c=0,d=1,e=0,f=3)
    decision = select(cheap_kv)
    assert decision['selected_candidate'] == 'kv:'+kv
    estimate = next(e for e in decision['candidate_estimates'] if e['candidate']=='kv:'+kv)
    assert estimate['recomputed_tokens'] == k
    assert estimate['workload_features']['recomputed_tokens'] == k
    assert estimate['predicted_local_ms'] == 1+3*k
    assert decision['synthetic_assumptions'] == cheap_kv.to_dict()
    assert select(Assumptions(a=1,b=0,c=0,d=10000,e=0,f=0))['selected_candidate']=='token'
    assert all(e['predicted_total_ms']==e['predicted_local_ms']+e['simulated_transfer_ms'] for e in decision['candidate_estimates'])


def test_request_normalizes_and_freezes_defaults_and_rejects_wrong_modes():
    req = DiscussionStartRequest(question='Q',mode='auto',auto_policy='performance_experiment',idealized_simulation=True)
    assert req.model_dump()['synthetic_assumptions'] == Assumptions().to_dict()
    config = req.auto_config()
    req.synthetic_assumptions['a'] = 99
    assert config.synthetic_assumptions.a == 20
    for kwargs in (dict(mode='token'),dict(mode='auto'),dict(mode='auto',auto_policy='performance_experiment')):
        with pytest.raises(ValueError): DiscussionStartRequest(question='Q',synthetic_assumptions={'a':3},**kwargs)
    with pytest.raises(ValueError): DiscussionNetworkRequest(bandwidth_mbps=10,synthetic_assumptions={'a':3})


def test_rest_history_replays_actual_frozen_assumptions(harness):
    # Tiny random CPU model executes the real protocol, without pretrained weights.
    used = Assumptions(a=31,b=3,c=.004,d=1e8).to_dict()
    first = start_wait(harness,mode='auto',rounds=1,auto_policy='performance_experiment',idealized_simulation=True,synthetic_assumptions=used)
    assert first['status']=='completed', first['error']
    assert first['config']['synthetic_assumptions']==used
    for decision in first['auto_decisions']:
        assert decision['synthetic_assumptions']==used
        token = next(e for e in decision['candidate_estimates'] if e['candidate']=='token')
        assert token['predicted_local_ms']==Assumptions.from_dict(used).predict(token['workload_features'])[0]
    start_wait(harness,mode='auto',rounds=1,auto_policy='performance_experiment',idealized_simulation=True)
    client = harness[0]
    assert client.get('/api/discussions/'+first['session_id']).json()==first
    events = client.get('/api/events?limit=100').json()['events']
    started = next(e for e in events if e['type']=='DISCUSSION_STARTED' and e['metadata']['session_id']==first['session_id'])
    assert started['metadata']['config']['synthetic_assumptions']==used
    assert client.patch('/api/discussions/'+first['session_id']+'/network',json={'synthetic_assumptions': {'a':2}}).status_code==422


def test_serialized_runtime_config_keeps_immutable_assumptions():
    from dataclasses import asdict
    from token_kv_agent_system.discussion import DiscussionConfig
    from token_kv_agent_system.distributed_discussion import config_from_dict
    original = DiscussionConfig(auto=IncrementalAutoConfig(policy='performance_experiment', idealized=True,
        synthetic_assumptions=Assumptions(a=41)))
    restored = config_from_dict(asdict(original))
    assert restored.auto == original.auto
    assert isinstance(restored.auto.synthetic_assumptions, Assumptions)


def test_v2_ratio_k_is_rounded_and_clamped_by_existing_algorithm():
    from token_kv_agent_system.communication import suffix_recompute_count
    b,r,a,z,m = runtime_fixture()
    config = IncrementalAutoConfig('performance_experiment',('token','kv:v2'),
        recompute_ratio=.75,idealized=True,synthetic_assumptions=Assumptions(f=7))
    decision = IncrementalSelector(config).select(a,z,m,HandoffNetwork().snapshot(1))[1]
    estimate = next(e for e in decision['candidate_estimates'] if e['candidate']=='kv:v2')
    n = estimate['workload_features']['message_tokens']
    assert estimate['recomputed_tokens'] == suffix_recompute_count(n, ratio=.75)
    assert estimate['predicted_local_ms'] == 5+.05*n+7*estimate['recomputed_tokens']
