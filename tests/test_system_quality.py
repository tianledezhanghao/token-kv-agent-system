from dataclasses import asdict, replace
import json
import pytest

from token_kv_agent_system.system_quality_tasks import PAIRS,PILOTS,DISCUSSIONS,freeze_tasks,score_pair
from token_kv_agent_system.system_quality_validation import write_once,digest,record_session,paired
from token_kv_agent_system.incremental_auto import (IncrementalCostProfile,IncrementalAutoConfig,
    IncrementalSelector,StageCost,workload_identity,backend_cost_identity)
from token_kv_agent_system.incremental_kv_protocol import inspect_model_contract
from token_kv_agent_system.network.handoff_network import HandoffNetwork


def test_frozen_task_distribution_and_answer_isolation():
    from collections import Counter
    assert Counter(t.category for t in PAIRS)==dict(number=3,code=3,fact=3,multiple_facts=3,calculation=3)
    assert not set(t.name for t in PAIRS)&set(t.name for t in PILOTS)
    spec=freeze_tasks()
    for t in PAIRS:
        assert t.private not in spec['public']+spec['systems'][1]+t.question
        assert all(x not in spec['public']+spec['systems'][1] for x in t.extraction)
    assert all(str(n) not in d['question'] for d in DISCUSSIONS if d['name']=='integration' for n in (12,7,19))
    assert '19' not in DISCUSSIONS[1]['a']+DISCUSSIONS[1]['b']


@pytest.mark.parametrize('idx,text,extract,final',[(0,'27',True,True),(0,'27 or 64',True,False),
    (3,'K7M2',True,True),(3,'K7M2 or K7M3',True,False),(6,'blue',True,True),
    (6,'blue and red',True,False),(9,'Locker 7 is blue. Key: box.',True,True),
    (9,'Locker 7 is blue. Key: drawer.',False,False),(12,'8 + 3 = 11',True,True),
    (12,'8 + 3 = 13',True,False),(12,'There are 8 coins.',True,None)])
def test_frozen_scorer_correct_conflict_and_unknown(idx,text,extract,final):
    task=PAIRS[idx]
    s=score_pair(task,task.private,text,delivered=True)
    assert s['extraction']==extract and s['final'] is final


def test_score_never_infers_delivery_and_sender_error_retained():
    t=PAIRS[0]
    assert score_pair(t,'27','27',delivered=False)['use'] is False
    s=score_pair(t,'64','27',delivered=True)
    assert s['final'] and not s['sender_correct'] and not s['use']
    assert score_pair(t,'27',None,delivered=False)['final'] is None


def test_artifacts_refuse_overwrite(tmp_path):
    p=tmp_path/'x.json';write_once(p,{'a':1})
    with pytest.raises(FileExistsError):write_once(p,{'a':2})
    assert json.loads(p.read_text())=={'a':1}


def state_pair():
    from test_incremental_kv_receive import build
    b,r=build(raw=(100,101,102,2));s=r.session
    m=s.generate_message('A','B',max_new_tokens=4)
    return b,s,m


def profile_for(b,s,m):
    ident={'sender':backend_cost_identity(b),'receiver':backend_cost_identity(b)}
    return IncrementalCostProfile('exact-test','measured_incremental',ident,
        {'token_handoff':StageCost(40),'kv_base':StageCost(80)},10,(3,3),
        (sum(x.cached_token_count for x in s._states.values()),)*2,'Test cost fixture, not measured speed evidence.',
        recompute_token_range=(0,0),workload_domain=workload_identity(s.state('A'),s.state('B'),m))


def test_measured_profile_requires_workload_domain_and_quality_safe_token():
    b,s,m=state_pair();p=replace(profile_for(b,s,m),workload_domain=None)
    c=IncrementalAutoConfig(policy='performance_experiment',candidates=('token','kv:v0'),profile=p)
    with pytest.raises(ValueError,match='profile_workload_domain_unverified'):
        IncrementalSelector(c).select(s.state('A'),s.state('B'),m,HandoffNetwork().snapshot(1),require_workload_domain=True)
    _,d=IncrementalSelector(replace(c,policy='quality_first')).select(s.state('A'),s.state('B'),m,HandoffNetwork().snapshot(1),require_workload_domain=True)
    assert d['selected_candidate']=='token' and d['predicted_local_ms'] is None
    assert d['cost_status']=='profile_workload_domain_unverified'


@pytest.mark.parametrize('key',['sender_cache_tokens','receiver_cache_tokens','sender_span_start',
    'termination_reason','sender_pending_token_id','receiver_pending_token_id',
    'sender_ledger_sha256','receiver_ledger_sha256','message_sha256','framing_sha256'])
def test_each_exact_domain_coordinate_rejects(key):
    b,s,m=state_pair();p=profile_for(b,s,m);domain=dict(p.workload_domain)
    if key.endswith('sha256'):domain[key]='0'*64
    elif key=='termination_reason':domain[key]='max_new_tokens'
    elif 'pending' in key:domain[key]=999
    else:domain[key]+=1
    c=IncrementalAutoConfig(policy='performance_experiment',candidates=('token','kv:v0'),profile=replace(p,workload_domain=domain))
    with pytest.raises(ValueError,match='profile_out_of_domain'):
        IncrementalSelector(c).select(s.state('A'),s.state('B'),m,HandoffNetwork().snapshot(1),kv_contract_ready=True)


def test_same_cache_sum_different_endpoints_rejected():
    b,s,m=state_pair();p=profile_for(b,s,m)
    a=replace(s.state('A'),cached_token_count=s.state('A').cached_token_count+1)
    receiver=replace(s.state('B'),cached_token_count=s.state('B').cached_token_count-1)
    c=IncrementalAutoConfig(policy='performance_experiment',candidates=('token','kv:v0'),profile=p)
    with pytest.raises(ValueError,match='profile_out_of_domain'):
        IncrementalSelector(c).select(a,receiver,m,HandoffNetwork().snapshot(1),kv_contract_ready=True)


def test_exact_profile_roundtrip_startup_missing_and_identity_rejection():
    b,s,m=state_pair();p=profile_for(b,s,m)
    assert IncrementalCostProfile.from_dict(asdict(p))==p
    c=IncrementalAutoConfig(policy='performance_experiment',candidates=('token','kv:v0'),profile=p)
    _,d=IncrementalSelector(c).select(s.state('A'),s.state('B'),m,HandoffNetwork().snapshot(1),require_workload_domain=True)
    assert any(x['reason']=='missing_startup_cost_profile' for x in d['excluded_candidates'])
    _,d=IncrementalSelector(c).select(s.state('A'),s.state('B'),m,HandoffNetwork().snapshot(1),kv_contract_ready=True)
    assert len(d['candidate_estimates'])==2
    with pytest.raises(ValueError,match='model_profile_mismatch'):
        IncrementalSelector(replace(c,profile=replace(p,model_identity={'wrong':1}))).select(s.state('A'),s.state('B'),m,HandoffNetwork().snapshot(1),require_workload_domain=True)


@pytest.mark.parametrize('domain',[{}, {'sender_cache_tokens':True}, {'garbage':'x'}])
def test_bad_workload_schema(domain):
    b,s,m=state_pair()
    with pytest.raises(ValueError,match='workload'):replace(profile_for(b,s,m),workload_domain=domain)


def test_harness_actual_tensor_pairing_and_packet_identity(tmp_path):
    b,s,m=state_pair()
    rows=paired(b,inspect_model_contract(b),PILOTS[0],tmp_path)
    assert len(rows)==5
    assert all(r['snapshots_unchanged'] for r in rows)
    assert all(r['engineering'] for r in rows[:4])
    assert len({r['evidence']['generations'][0]['input_ids'].__repr__() for r in rows if r['engineering']})==1
    assert len({r['snapshot_hashes']['B'] for r in rows})==1
    for r in rows[1:4]:assert r['evidence']['handoffs'][0]['packets'][0]['magic']=='TKVI'


def test_runtime_enforces_exact_domain_for_real_profile_not_legacy_offline():
    from token_kv_agent_system.discussion import DiscussionConfig,DiscussionRuntime
    b,s,m=state_pair()
    p=replace(profile_for(b,s,m),workload_domain=None,message_token_range=(0,4096),cache_token_range=(0,99999))
    c=IncrementalAutoConfig(policy='performance_experiment',candidates=('token','kv:v0'),profile=p)
    r=DiscussionRuntime(b,config=DiscussionConfig(rounds=1,max_new_tokens=4,auto=c))
    events=[];r.event_sink=lambda k,v:events.append((k,v))
    result=r.run()
    assert result.status=='failed' and 'profile_workload_domain_unverified' in result.error
    assert not r.session.transport.records
    assert any(k=='AUTO_DECISION_FAILED' for k,_ in events)
    r=DiscussionRuntime(b,config=DiscussionConfig(rounds=1,max_new_tokens=4,auto=replace(c,policy='quality_first')))
    result=r.run()
    assert result.status=='completed' and all(h.executed_mode=='token' for h in result.handoffs)
    assert all(h.predicted_local_ms is None for h in result.handoffs)


def test_result_denominators_keep_unknown_and_token_failed_cases():
    from token_kv_agent_system.system_quality_export import count_values,paired_statistics
    assert count_values([True,False,None])==dict(n=3,yes=1,no=1,unknown=1)
    rows=[dict(task=t,category='code',variant=v,score={'final':value})
          for t,values in [('a',[True,False,True,None,True]),('b',[False,True,False,False,None])]
          for v,value in zip(('token','kv:v0','kv:v1','kv:v2','kv:v3'),values)]
    result=paired_statistics(rows)['code']
    assert result['kv:v0']['all']['n']==2 and result['kv:v0']['token_success_subset']['n']==1
    assert result['kv:v0']['kv_unique_success']==['b'] and result['kv:v0']['kv_new_failures']==['a']
    assert result['kv:v2']['unknown_with_token_success']==['a']


def test_export_sha_audit_rejects_corruption_and_escape(tmp_path):
    from token_kv_agent_system.system_quality_export import verified_index
    from token_kv_agent_system.local_qwen_comparison import file_hash
    p=tmp_path/'data.json';write_once(p,{'x':1})
    row=dict(path=p.name,bytes=p.stat().st_size,sha256=file_hash(p))
    (tmp_path/'sha256-index.json').write_text(json.dumps([row]))
    assert verified_index(tmp_path)==[row]
    p.write_text('{}')
    with pytest.raises(ValueError,match='SHA'):verified_index(tmp_path)
    row['path']='../outside'
    (tmp_path/'sha256-index.json').write_text(json.dumps([row]))
    with pytest.raises(ValueError,match='path'):verified_index(tmp_path)
