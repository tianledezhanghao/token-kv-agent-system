from copy import deepcopy
from dataclasses import asdict, replace
from itertools import product
import json
from types import SimpleNamespace

import pytest

from token_kv_agent_system.generalized_cost import (GeneralizedCostProfile, digest,
    load_cost_profile, workload_features, environment_identity, idealized_prediction)
from token_kv_agent_system.incremental_auto import (IncrementalAutoConfig, IncrementalSelector,
    backend_cost_identity)
from token_kv_agent_system.network.handoff_network import HandoffNetwork, HandoffNetworkConfig


def features(n=4,h=64,s=128,**kw):
    return dict(message_tokens=n,receiver_history_tokens=h,sender_history_tokens=s,
        termination_reason='max_new_tokens',sender_pending=True,receiver_pending=False,
        sender_close_tokens=3,receiver_close_tokens=0,prefix_tokens=12,closing_tokens=2,
        transfer_path='packed',contract_ready=True,recomputed_tokens=0,numeric_slot_index=None,**kw)


def profile(*,token=100,kv=80,spread=1,env=None,ident=None):
    ident=ident or {'sender':{'model_id':'test'},'receiver':{'model_id':'test'}}
    env=env or {k:dict(model=v,hardware={'name':'fixture'},implementation={'version':'test'}) for k,v in ident.items()}
    points=[dict(candidate=c,features=features(n,h,s),local_ms=base+.2*n+.001*n*h+.1*s,
        spread_ms=spread,samples=2,wire_bytes=10+4*n if c=='token' else n*12288+2000)
        for c,base in [('token',token),('kv:v0',kv)] for n,h,s in product((4,12),(64,128),(128,256))]
    return GeneralizedCostProfile('test',ident,env,points,len(points)*2,'2026-10-16T01:00:00+00:00',
        digest(points),'Fixture only; not actual hardware evidence.')


def test_joint_cell_prediction_including_interaction_and_boundaries():
    p=profile()
    for n,h,s in ((8,96,192),(4,64,128),(12,128,256)):
        r=p.predict('token',features(n,h,s),p.model_identity,p.environment)
        assert r['predicted_local_ms']==pytest.approx(100+.2*n+.001*n*h+.1*s)
        assert r['cost_source']=='predicted' and r['calibration_source']=='measured'
        assert r['support_points']==(8 if n==8 else 1)
        assert r['uncertainty_ms']>0 and r['training_sample_count']==32
    assert load_cost_profile(json.loads(json.dumps(asdict(p))))==p


@pytest.mark.parametrize('axis,value',[('message_tokens',3),('message_tokens',13),
    ('receiver_history_tokens',63),('sender_history_tokens',257),('sender_pending',False),
    ('termination_reason','eos'),('contract_ready',False),('transfer_path','legacy'),
    ('recomputed_tokens',1),('prefix_tokens',11),('receiver_pending',True)])
def test_domain_conditions_are_not_extrapolated(axis,value):
    p=profile();f={**features(),axis:value}
    with pytest.raises(ValueError,match='profile|startup'):
        p.predict('kv:v0',f,p.model_identity,p.environment)


def test_sparse_grid_cannot_be_claimed_as_full_domain():
    p=profile();rows=p.points[:-1]
    p=replace(p,points=rows,evidence_sha256=digest(rows),sample_count=len(rows)*2)
    with pytest.raises(ValueError,match='sparse_cell'):
        p.predict('kv:v0',features(8,96,192),p.model_identity,p.environment)


def test_model_device_adapter_algorithm_and_software_identity_rejection():
    p=profile()
    with pytest.raises(ValueError,match='model_profile_mismatch'):
        p.predict('token',features(),{},p.environment)
    for part in ('model','hardware','implementation'):
        env=deepcopy(p.environment);env['sender'][part]={'other':'changed'}
        with pytest.raises(ValueError,match='environment_profile_mismatch'):
            p.predict('token',features(),p.model_identity,env)


@pytest.mark.parametrize('change',[dict(schema='old'),dict(provenance='synthetic'),dict(units='seconds'),
    dict(sample_count=1),dict(relative_error_floor=0),dict(evidence_sha256='0'*64),
    dict(calibrated_at='2026-10-16'),dict(startup_cost_ms=1),dict(environment={})])
def test_generalized_profile_rejects_forged_or_invalid_metadata(change):
    with pytest.raises(ValueError):replace(profile(),**change)


def test_profile_bad_samples_and_missing_features_are_rejected():
    for change in (dict(local_ms=-1),dict(spread_ms=float('nan')),dict(samples=0),dict(wire_bytes=0),dict(features={})):
        p=profile();points=deepcopy(p.points);points[0].update(change)
        with pytest.raises(ValueError):replace(p,points=points,evidence_sha256=digest(points))
    p=profile()
    with pytest.raises(ValueError):GeneralizedCostProfile.from_dict({**asdict(p),'path':'/server'})


def runtime_fixture():
    from test_incremental_kv_transport import make_backend
    from token_kv_agent_system.discussion import DiscussionConfig,DiscussionRuntime
    b=make_backend();r=DiscussionRuntime(b,config=DiscussionConfig(rounds=1,max_new_tokens=4))
    m=r.session.generate_message('A','B',max_new_tokens=4)
    return b,r,r.session.state('A'),r.session.state('B'),m


def test_unseen_content_uses_features_not_ids_or_sha_and_uncertain_tie_is_token():
    b,r,a,z,m=runtime_fixture()
    identity={k:backend_cost_identity(b) for k in ('sender','receiver')}
    env={k:environment_identity(b) for k in ('sender','receiver')}
    f=workload_features(a,z,m,'token',contract_ready=True,recomputed_tokens=0)
    rows=[dict(candidate=c,features=f,local_ms=t,spread_ms=10,samples=2,wire_bytes=100)
          for c,t in [('token',100),('kv:v0',90)]]
    p=GeneralizedCostProfile('actual-features-fixture',identity,env,rows,4,'2026-10-16T00:00:00Z',digest(rows),'Fixture cost only.')
    selector=IncrementalSelector(IncrementalAutoConfig('performance_experiment',profile=p))
    network=HandoffNetwork(HandoffNetworkConfig(bandwidth_mbps=100000)).snapshot(1)
    _,d=selector.select(a,z,m,network,kv_contract_ready=True,require_workload_domain=True)
    assert d['selected_candidate']=='token' and d['selection_reason']=='uncertainty_overlap_prefer_token'
    assert d['candidate_set']==['token','kv:v0']
    different=replace(m,message_token_ids=tuple(x+1 for x in m.message_token_ids))
    d2=selector.select(a,z,different,network,kv_contract_ready=True,require_workload_domain=True)[1]
    assert [x['predicted_local_ms'] for x in d2['candidate_estimates']]==[100,90]
    assert d2['profile_provenance']=='measured'
    assert all('observed_local_ms' not in e for e in d2['candidate_estimates'])
    cold=replace(p,startup_cost_ms=200,startup_uncertainty_ms=100)
    f={**f,'contract_ready':False}
    assert cold.predict('kv:v0',f,identity,env)['predicted_local_ms']==290
    assert cold.predict('token',f,identity,env)['predicted_local_ms']==100


@pytest.mark.parametrize('kv',['v0','v1','v2','v3'])
def test_pair_configuration_and_default_v2_parameter(kv):
    from token_kv_agent_system.dashboard.discussion_service import DiscussionStartRequest
    c=DiscussionStartRequest(question='Q',mode='auto',auto_kv_version=kv).auto_config()
    assert c.candidates==('token','kv:'+kv)
    assert c.recompute_tokens==(1 if kv=='v2' else None)
    assert c.policy=='quality_first'
    with pytest.raises(ValueError):IncrementalAutoConfig(candidates=('token','kv:v0','kv:v1'))
    with pytest.raises(ValueError):DiscussionStartRequest(question='Q',mode='auto',auto_kv_version='v2',auto_candidates=['token','kv:v1'])


def test_idealized_uses_actual_bytes_and_v3_rejection_is_visible():
    b,r,a,z,m=runtime_fixture()
    selector=IncrementalSelector(IncrementalAutoConfig('performance_experiment',('token','kv:v1'),idealized=True))
    low=selector.select(a,z,m,HandoffNetwork(HandoffNetworkConfig(bandwidth_mbps=.001)).snapshot(1))[1]
    high=selector.select(a,z,m,HandoffNetwork(HandoffNetworkConfig(bandwidth_mbps=10000)).snapshot(1))[1]
    assert low['selected_candidate']=='token' and high['selected_candidate']=='kv:v1'
    assert high['profile_provenance']=='synthetic' and high['idealized']
    assert high['candidate_set']==['token','kv:v1']
    assert all(e['cost_source']=='synthetic' for e in high['candidate_estimates'])
    c=IncrementalAutoConfig('performance_experiment',('token','kv:v3'),True,idealized=True)
    b.decode=lambda ids,**kw:'letters'
    d=IncrementalSelector(c).select(a,z,m,HandoffNetwork().snapshot(1))[1]
    assert d['selected_candidate']=='token'
    assert next(e['reason'] for e in d['excluded_candidates'] if e['candidate']=='kv:v3')=='v3_no_numeric_slot'
    for kwargs in (dict(idealized=True),dict(idealized=True,policy='performance_experiment',profile=profile())):
        with pytest.raises(ValueError):IncrementalAutoConfig(**kwargs)


def test_idealized_discussion_patch_executes_real_tensor_protocol_without_timing_pollution():
    from test_incremental_kv_transport import make_backend
    from token_kv_agent_system.discussion import DiscussionConfig,DiscussionRuntime
    from token_kv_agent_system.network.handoff_network import HandoffNetwork
    b=make_backend();net=HandoffNetwork(HandoffNetworkConfig(bandwidth_mbps=10000))
    events=[]
    def event(kind,data):
        events.append((kind,data))
        if kind=='HANDOFF_COMPLETED' and data['sequence']==1:
            net.update(HandoffNetworkConfig(bandwidth_mbps=.001))
    r=DiscussionRuntime(b,config=DiscussionConfig(rounds=1,max_new_tokens=4,
        auto=IncrementalAutoConfig('performance_experiment',('token','kv:v1'),idealized=True)),network=net,event_sink=event)
    result=r.run()
    assert result.status=='completed',result.error
    assert [h.executed_mode for h in result.handoffs]==['kv','token']
    assert result.handoffs[0].executed_kv_strategy=='rope_aligned_delta'
    assert [h.network['config_revision'] for h in result.handoffs]==[0,1]
    for h in result.handoffs:
        assert h.actual_wire_bytes==h.predicted_wire_bytes
        assert h.observed_local_ms>0 and h.observation_scope=='synchronized_instrumented_handoff'
        assert h.auto_decision['cost_source']=='synthetic' and h.fallback_reason is None
    assert result.final_message is not None


def test_calibration_budget_cuda_entry_and_complete_help(tmp_path,capsys):
    from token_kv_agent_system.generalized_calibration import parser,validate_args,main
    args=parser().parse_args(['--model','/local/snapshot','--device','cuda:0','--output-dir',str(tmp_path)])
    validate_args(args)
    for field,value in [('max_handoffs',1),('warmup',0),('device','auto'),('samples',1),('max_seconds',float('inf'))]:
        with pytest.raises(ValueError):validate_args(SimpleNamespace(**{**vars(args),field:value}))
    with pytest.raises(SystemExit) as exc:main(['--help'])
    assert exc.value.code==0 and '--receiver-pending' in capsys.readouterr().out


def test_cuda_uses_legacy_adapter_not_mps_optimization():
    b,r,a,z,m=runtime_fixture()
    b._device='cuda:0'  # Metadata-only routing test; no CUDA inference is claimed.
    f=workload_features(a,z,m,'kv:v0',contract_ready=True,recomputed_tokens=0)
    assert f['transfer_path']=='legacy'


def test_calibration_entry_rejects_unavailable_cuda_before_any_model_load(monkeypatch,tmp_path):
    from token_kv_agent_system import generalized_calibration as module
    from token_kv_agent_system.model import huggingface
    import transformers
    model=tmp_path/'local-model';model.mkdir()
    for name in ('config.json','tokenizer_config.json','tokenizer.json'):
        (model/name).write_text('{}')
    (model/'model.safetensors').write_bytes(b'not-weights-for-preflight-test')
    monkeypatch.setattr(module,'refuse_active_service',lambda url:None)
    calls=[]
    def reject(device,torch):
        calls.append(device)
        raise RuntimeError('CUDA requested but unavailable')
    monkeypatch.setattr(huggingface,'resolve_device',reject)
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',lambda *a,**k:pytest.fail('unexpected model loading'))
    output=tmp_path/'output'
    with pytest.raises(RuntimeError,match='unavailable'):
        module.main(['--model',str(model),'--device','cuda:0','--output-dir',str(output)])
    assert calls==['cuda:0']
    assert json.loads((output/'failure.json').read_text())['type']=='RuntimeError'
    assert (output/'manifest.json').exists()
    assert not (output/'profile.json').exists()


def test_idealized_failure_does_not_fall_back_or_commit(monkeypatch):
    from test_incremental_kv_transport import make_backend
    from token_kv_agent_system.discussion import DiscussionConfig,DiscussionRuntime
    from token_kv_agent_system.performance_characterization import cache_sha256
    b=make_backend();before={};events=[]
    r=DiscussionRuntime(b,config=DiscussionConfig(rounds=1,max_new_tokens=4,
        auto=IncrementalAutoConfig('performance_experiment',('token','kv:v1'),idealized=True),
        network=HandoffNetworkConfig(bandwidth_mbps=10000)))
    def event(kind,data):
        events.append((kind,data))
        if kind=='HANDOFF_STARTED':
            before.update({a:cache_sha256(s.cache) for a,s in r.session._states.items()})
    r.event_sink=event
    def fail(*args,**kwargs):raise RuntimeError('injected transport failure')
    monkeypatch.setattr(r.session.transport,'transfer',fail)
    result=r.run()
    assert result.status=='failed' and len(result.handoffs)==1
    h=result.handoffs[0]
    assert h.selected_kv_strategy=='rope_aligned_delta' and h.executed_mode is None
    assert h.fallback_reason is None and h.actual_execution_status=='failed'
    assert before=={a:cache_sha256(s.cache) for a,s in r.session._states.items()}
    assert not any(s.received_messages for s in r.session._states.values())


def test_tiny_cpu_calibration_toolchain_pairs_true_decode_and_keeps_holdout_separate(monkeypatch,tmp_path):
    """Random tiny CPU unit fixture; never pretrained-model research evidence."""
    from token_kv_agent_system import generalized_calibration as module
    from test_incremental_kv_transport import make_backend
    b=make_backend();source=module.source_snapshot
    def different_content(backend,*args,**kwargs):
        b.outputs=[101 if kwargs.get('holdout') else 100]*200
        return source(backend,*args,**kwargs)
    monkeypatch.setattr(module,'source_snapshot',different_content)
    args=module.parser().parse_args(['--model','unit-fixture','--output-dir',str(tmp_path),
        '--message-lengths','4,8','--receiver-histories','256,384','--sender-histories','256,384','--max-handoffs','80'])
    p,report=module.calibration(b,args,lambda:None,tmp_path)
    assert p.sample_count==32 and report['holdout_samples']==16
    assert report['source_messages']==4 and len(p.points)==16
    assert report['observed_successful_handoffs']==64
    module.write_json(tmp_path/'spec.json',dict(source_sha256={}))
    import hashlib
    module.write_json(tmp_path/'manifest.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in tmp_path.glob('*.json')})
    assert module.verify_output(tmp_path)['training_samples']==32
    samples=json.loads((tmp_path/'samples.json').read_text())
    assert all(r['receipt']['received_token_ids']==r['message_ids'] for r in samples)
    assert set(r['message_sha256'] for r in samples if r['tag']=='train').isdisjoint(
        r['message_sha256'] for r in samples if r['tag']=='holdout')
    # Removing a slow sample is detected by manifest/sample count verification.
    samples.pop();module.write_json(tmp_path/'samples.json',samples)
    with pytest.raises(ValueError,match='SHA'):module.verify_output(tmp_path)


def test_missing_token_prediction_never_certifies_a_sole_kv_cost():
    b,r,a,z,m=runtime_fixture()
    ident={k:backend_cost_identity(b) for k in ('sender','receiver')}
    env={k:environment_identity(b) for k in ('sender','receiver')}
    f=workload_features(a,z,m,'kv:v0',contract_ready=True,recomputed_tokens=0)
    points=[dict(candidate='kv:v0',features=f,local_ms=1,spread_ms=1,samples=2,wire_bytes=100)]
    p=GeneralizedCostProfile('fixture',ident,env,points,2,'2026-10-16T00:00:00Z',digest(points),'Fixture missing TOKEN calibration.')
    with pytest.raises(ValueError,match='token_reference_unavailable'):
        IncrementalSelector(IncrementalAutoConfig('performance_experiment',profile=p)).select(
            a,z,m,HandoffNetwork(HandoffNetworkConfig(bandwidth_mbps=100000)).snapshot(1),kv_contract_ready=True)


def test_discussion_smoke_reserves_all_six_handoffs_in_budget(tmp_path):
    from token_kv_agent_system.generalized_calibration import parser,validate_args
    args=parser().parse_args(['--model','fixture','--output-dir',str(tmp_path),'--discussion-smoke','--max-handoffs','38'])
    validate_args(args)  # default calibration/holdout=32, smoke=6
    args.max_handoffs=37
    with pytest.raises(ValueError,match='budget'):validate_args(args)
