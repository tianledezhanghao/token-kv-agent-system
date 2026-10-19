from dataclasses import asdict
import time
import pytest

from token_kv_agent_system.performance_characterization import (
    StageRecorder, bandwidth_analysis, statistics, real_prefix, measured_handoff,
    cache_sha256, research_profile)


def test_nested_timing_counts_children_once_and_restores_after_failure():
    r = StageRecorder()
    with r.span('total'):
        with r.span('child'):
            time.sleep(.001)
        with pytest.raises(RuntimeError):
            with r.span('failure'):
                raise RuntimeError('expected')
    rows = r.aggregate()
    assert not r.stack
    assert sum(row['exclusive_ms'] for row in rows.values()) == pytest.approx(rows['total']['inclusive_ms'])
    assert rows['child']['calls']==1 and rows['failure']['calls']==1


def test_bandwidth_sign_units_latency_and_uncertain_sample_spread():
    r = bandwidth_analysis(40,80,18,30000)
    assert r['break_even_mbps'] is None
    assert r['curves'][0]['kv_network_ms']==242
    r = bandwidth_analysis(100,50,10,12510,bandwidths=(1,2,10))
    assert r['break_even_mbps']==2
    assert r['curves'][1]['token_total_ms']==pytest.approx(r['curves'][1]['kv_total_ms'])
    assert bandwidth_analysis(10,10,1,1)['reason']=='equal_all_bandwidths'
    assert bandwidth_analysis(10,10,1,2)['break_even_mbps'] is None
    assert statistics([1,2,9])==dict(sample_count=3,median=2,min=1,max=9,mad=1)
    for kw in (dict(token_ms=float('nan')),dict(kv_bytes=-1),dict(bandwidths=(0,)),dict(base_latency_ms=-1)):
        args=dict(token_ms=40,kv_ms=80,token_bytes=18,kv_bytes=30000)
        args.update(kw)
        with pytest.raises(ValueError):bandwidth_analysis(**args)
    with pytest.raises(ValueError):statistics([])


@pytest.mark.parametrize('candidate,kw', [('token',{}),('kv:v0',{}),('kv:v1',{}),
    ('kv:v2',dict(recompute_tokens=1)),('kv:v3',{})])
def test_same_snapshot_real_tensor_execution_instrumentation_and_full_commit(candidate,kw):
    from test_incremental_kv_receive import build
    from token_kv_agent_system.incremental_kv_protocol import inspect_model_contract
    import torch
    b,r=build(raw=(100,101,102,103,2)); s=r.session
    s.generate_message('A','B',max_new_tokens=5)
    s._kv_contract=inspect_model_contract(b)
    saved=dict(s._states)
    hashes={n:cache_sha256(x.cache) for n,x in saved.items()}
    original=b.continue_from_cache
    plain=measured_handoff(s,saved,candidate,**kw)
    expected=[(l.keys.clone(),l.values.clone()) for l in s.state('B').cache.layers]
    observed=measured_handoff(s,saved,candidate,mode='instrumented',**kw)
    assert b.continue_from_cache==original
    assert plain['wire_bytes']==observed['wire_bytes']
    assert observed['local_ms']>=observed['stages']['handoff_residual_validation_commit']['inclusive_ms']
    for l,(k,v) in zip(s.state('B').cache.layers,expected):
        assert torch.equal(l.keys,k) and torch.equal(l.values,v)
    assert {n:cache_sha256(x.cache) for n,x in saved.items()}==hashes
    if candidate!='token':
        assert observed['adapter_submetrics']['host_to_device_ms']
    # Failure leaves original states and patches restored, no TOKEN fallback.
    s._states=dict(saved)
    def fail(*a,**kw):raise RuntimeError('forward failed')
    b.continue_from_cache=fail
    with pytest.raises(RuntimeError):measured_handoff(s,saved,candidate,mode='instrumented',**kw)
    assert b.continue_from_cache is fail
    assert all(s.state(n) is saved[n] for n in saved)


def test_real_decode_prefix_pending_boundary_and_immutable_source():
    from test_incremental_kv_receive import build
    b,r=build(raw=(100,101,102,103,2));s=r.session
    s.generate_message('A','B',max_new_tokens=5)
    state=s.state('A');before=cache_sha256(state.cache)
    for n in (1,2,4):
        work=real_prefix(state,n)
        assert work.pending_token_id==state.draft_message.message_token_ids[n-1]
        assert work.cached_token_count==work.draft_message.sender_message_span[1]-1
        assert work.draft_message.termination_reason=='max_new_tokens'
        assert work.token_ledger[work.draft_message.sender_message_span[0]:]==work.draft_message.message_token_ids
        work.validate()
    assert cache_sha256(state.cache)==before
    for n in (0,5,True):
        with pytest.raises(ValueError):real_prefix(state,n)


def test_point_profile_identity_domain_origin_and_selector():
    from token_kv_agent_system.incremental_auto import IncrementalCostProfile
    summary={k:statistics(v) for k,v in {'token':[40]*5,'kv:v0':[80]*5}.items()}
    identity={'sender':{'revision':'fixed'},'receiver':{'revision':'fixed'}}
    p=research_profile(summary,identity,origin='real_decode',n=2,cache_count=220)
    assert p.sample_count==10 and p.cache_token_range==(220,220)
    assert IncrementalCostProfile.from_dict(asdict(p))==p
    for origin in ('synthetic','controlled_fixed_token','prefill_reconstruction'):
        with pytest.raises(ValueError):research_profile(summary,identity,origin=origin,n=2,cache_count=220)


def test_offline_real_selector_predicts_using_captured_metadata_and_no_model():
    from test_incremental_kv_receive import build
    from token_kv_agent_system.incremental_kv_protocol import inspect_model_contract
    from token_kv_agent_system.incremental_auto import backend_cost_identity
    from token_kv_agent_system.performance_characterization import offline_analysis
    b,r=build(raw=(100,101,2));s=r.session
    m=s.generate_message('A','B',max_new_tokens=3)
    contract=inspect_model_contract(b);s._kv_contract=contract;saved=dict(s._states)
    t=measured_handoff(s,saved,'token');k=measured_handoff(s,saved,'kv:v0')
    c=dict(label='test_fixture',origin='real_decode',message_tokens=2,message_ids=list(m.message_token_ids),
        termination='eos',sender_span=list(m.sender_message_span),sender_cache_tokens=saved['A'].cached_token_count,
        receiver_cache_tokens=saved['B'].cached_token_count,total_cache_tokens=sum(x.cached_token_count for x in saved.values()))
    group=dict(config=c,summary={'steady':{'token':dict(**statistics([40]*5),wire_bytes=t['wire_bytes']),
        'kv:v0':dict(**statistics([80]*5),wire_bytes=k['wire_bytes'])}})
    settings=dict(contract=asdict(contract),model_identity={'sender':backend_cost_identity(b),'receiver':backend_cost_identity(b)},
        metadata={'dtype':b.metadata.dtype})
    b.prefill=lambda **kw:pytest.fail('model called by offline analysis')
    networks,profiles,decisions=offline_analysis([group],settings)
    assert len(networks)==1 and len(profiles)==1 and len(decisions)==5
    assert all(d['decision']['selected_candidate']=='token' for d in decisions)
    assert all(len(d['decision']['candidate_estimates'])==2 for d in decisions)


def test_sha_index_rejects_corruption_and_path_escape(tmp_path):
    import json
    import hashlib
    from token_kv_agent_system.performance_export import verify_index
    p=tmp_path/'raw.json';p.write_bytes(b'{}')
    index=[dict(path=p.name,bytes=2,sha256=hashlib.sha256(b'{}').hexdigest())]
    (tmp_path/'sha256-index.json').write_text(json.dumps(index))
    assert verify_index(tmp_path)==index
    p.write_bytes(b'[]')
    with pytest.raises(ValueError,match='SHA'):verify_index(tmp_path)
    index[0]['path']='../outside.json'
    (tmp_path/'sha256-index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError,match='path'):verify_index(tmp_path)


def test_cli_help_does_not_load_model_or_create_output(capsys):
    from token_kv_agent_system.performance_validation import main
    with pytest.raises(SystemExit) as e:main(['--help'])
    assert e.value.code==0 and 'output-dir' in capsys.readouterr().out
