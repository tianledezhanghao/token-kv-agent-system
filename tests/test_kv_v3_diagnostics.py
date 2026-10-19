"""Aligned statistics and real sparse forward contracts, not semantic mocks."""
from dataclasses import replace
import pytest
import torch
from test_incremental_kv_receive import build
from test_stateful_token_cache import copies, unchanged
from token_kv_agent_system.communication import HandoffChoice, KVStrategy
from token_kv_agent_system.kv_v3_diagnostics import span_metrics, message_span, oracle_layer_intervention
from token_kv_agent_system.model.token_slot_repair import first_numeric, repair_slot
from token_kv_agent_system.model.delta_repair import reconstruct
from token_kv_agent_system.model.decode_kv import cache_layers
from token_kv_agent_system.message_diagnostics import ForwardTrace

V3=KVStrategy.NUMERIC_SLOT_DELTA

@pytest.mark.parametrize('pieces,expected', [(['17'],0),(['紫色','5','7'],1),(['A','B'],None),(['１２'],None),([' 2 '],0),(['-3'],None)])
def test_online_selection_id_only(pieces,expected):
    assert first_numeric(tuple(range(len(pieces))),lambda ids:pieces[ids[0]])==expected


def test_metrics_alignment_and_kv_separation():
    r=((torch.ones(1,2,3,4),torch.ones(1,2,3,4)),)
    a=((r[0][0]*2,r[0][1]),)
    rows=span_metrics(r,a)
    assert len(rows)==6
    assert all(x['relative_l2']==1 and x['cosine']==pytest.approx(1) for x in rows if x['kind']=='K')
    assert all(x['relative_l2']==0 for x in rows if x['kind']=='V')
    with pytest.raises(ValueError):span_metrics(r,((torch.ones(1,2,2,4),r[0][1]),))
    with pytest.raises(ValueError):span_metrics(r,())
    with pytest.raises(ValueError):span_metrics(r,((r[0][0]*float('nan'),r[0][1]),))


def numeric_backend(b):
    original=b.decode
    b.decode=lambda ids,**kw:'1' if tuple(ids)==(100,) else original(ids,**kw)

@pytest.mark.parametrize('raw',[(100,101,102,2),(100,101,102),(100,2),(100,)])
def test_production_slot_vs_independent_real_forward_and_no_suffix_refresh(raw):
    b,r=build(raw=raw);numeric_backend(b);s=r.session
    s.generate_message('A','B',max_new_tokens=len(raw));initial=s._states.copy()
    saved={n:copies(x.cache) for n,x in initial.items()}
    v1=s.handoff_kv('A','B',strategy=KVStrategy.ROPE_ALIGNED_DELTA);foreign=s.state('B');old=copies(foreign.cache)
    start,end=v1.receiver_target_span
    # All-layer forward from cropped receiver prefix gives exact chosen-slot reference.
    prefix=reconstruct(tuple((k[...,:start,:],v[...,:start,:]) for k,v in cache_layers(foreign.cache)),b._model.config)
    oracle=b.continue_from_cache((100,),previous_cache=prefix,logical_cached_token_count=start)
    s._states=initial.copy();b.prefill=lambda **kw:pytest.fail('hidden full prefill')
    with ForwardTrace(b) as trace:
        receipt=s.handoff_kv('A','B',strategy=V3,measure_timing=True)
    actual=s.state('B')
    assert receipt.repaired_token_indices==(0,)
    assert receipt.repaired_layer_indices==(0,1)
    assert receipt.repair_layer_token_count==2
    assert receipt.receiver_message_prefill_count==1 and receipt.recompute_forward_count==1
    assert receipt.wire_byte_count==v1.wire_byte_count
    calls=[c for c in trace.calls if c['cache_length_before']==start]
    assert len(calls)==1 and calls[0]['input_ids']==[100]
    assert calls[0]['effective_rotary_position_ids']==[[[start]]]
    assert calls[0]['attention_mask']==[[1]*(start+1)]
    for i,layer in enumerate(actual.cache.layers):
        for t,reference,prior in zip((layer.keys,layer.values),cache_layers(oracle.cache)[i],old[i]):
            assert torch.equal(t[...,start:start+1,:],reference[...,start:start+1,:])
            assert torch.equal(t[...,:start,:],prior[...,:start,:])
            assert torch.equal(t[...,start+1:end,:],prior[...,start+1:end,:])
    for n,state in initial.items():unchanged(state.cache,saved[n])
    actual.validate();s.generate_message('B','A',max_new_tokens=2)

@pytest.mark.parametrize('failure',['no-numeric','slot','closing','sender-close'])
def test_transaction_no_commit_no_fallback(monkeypatch,failure):
    from token_kv_agent_system import incremental_kv_receive as receive
    from token_kv_agent_system import stateful_token as session
    b,r=build();s=r.session
    if failure!='no-numeric':numeric_backend(b)
    else:b.decode=lambda ids,**kw:'letters'
    s.generate_message('A','B',max_new_tokens=3);initial=s._states.copy();saved={n:copies(x.cache) for n,x in initial.items()}
    if failure=='slot':
        def fail(b,cache,*args):
            cache.layers[0].keys.zero_();raise RuntimeError('slot error')
        monkeypatch.setattr(receive,'repair_slot',fail)
    elif failure=='closing':
        original=receive.append_working_ids
        def fail(state,ids):
            if state.cached_token_count>initial['B'].cached_token_count+len(state.framing.incoming('A','B').prefix_ids):
                raise RuntimeError('closing error')
            return original(state,ids)
        monkeypatch.setattr(receive,'append_working_ids',fail)
    elif failure=='sender-close':
        monkeypatch.setattr(session,'close_working_draft',lambda *x:(_ for _ in ()).throw(RuntimeError('sender close')))
    with pytest.raises((RuntimeError,ValueError)):s.handoff_kv('A','B',strategy=V3)
    for n,state in initial.items():
        assert s.state(n) is state;unchanged(state.cache,saved[n])
    assert not s.state('B').received_messages and s.state('A').draft_message is not None

@pytest.mark.parametrize('index',[-1,3,True])
def test_invalid_slot(index):
    b,r=build();state=r.session.state('B')
    with pytest.raises(ValueError):repair_slot(b,state.cache,1,(100,101,102),index)


def test_v3_policy_rejects_v2_parameters():
    assert HandoffChoice.parse('kv:v3').kv_strategy==V3
    with pytest.raises(ValueError):HandoffChoice.parse('kv:v3',recompute_tokens=1)
    with pytest.raises(ValueError):HandoffChoice.parse('kv:v3',recompute_ratio=.25)


def test_middle_layer_oracle_is_intervention_only():
    b,r=build();state=r.session.state('B');old=copies(state.cache);ref=tuple((k[...,:1,:]*0,v[...,:1,:]*0) for k,v in old)
    changed=oracle_layer_intervention(state,ref,0,1,layers=(1,))
    unchanged(state.cache,old)
    assert torch.equal(changed.cache.layers[0].keys,old[0][0])
    assert changed.cache.layers[1].keys[...,0,:].count_nonzero()==0

@pytest.mark.parametrize('plan',[('kv:v3','kv:v3'),('token','kv:v3','kv:v0','kv:v3','kv:v1','kv:v2')])
def test_mixed_real_runtime(plan):
    b,r=build(plan=tuple(HandoffChoice.parse(m) for m in plan),rounds=len(plan)//2);numeric_backend(b);b.outputs=[]
    result=r.run();assert result.status=='completed',result.error
    for h,c in zip(result.handoffs,r.config.handoff_plan):
        if c.kv_strategy==V3:
            assert h.executed_kv_strategy=='numeric_slot_delta' and h.repair_layer_token_count==2
            assert h.repaired_token_indices==(0,) and h.fallback_reason is None


def test_cli_dispatch_v3_without_loading(monkeypatch):
    from token_kv_agent_system import discussion_demo as cli
    def unavailable(*args,**kw):raise RuntimeError('cache-only loader')
    monkeypatch.setattr(cli,'HuggingFaceModelBackend',unavailable)
    assert cli.main(['--mode','kv','--kv-strategy','v3'])==1

@pytest.mark.parametrize('mode',['kv:v3','mixed'])
def test_actual_rest_v3_receipts(mode):
    from threading import Event
    from fastapi.testclient import TestClient
    from token_kv_agent_system.dashboard.app import create_app
    from token_kv_agent_system.dashboard.discussion_service import DiscussionService,DiscussionStartRequest
    from token_kv_agent_system.metrics import MetricCollector
    b,r=build();numeric_backend(b);b.outputs=[];collector=MetricCollector();done=Event()
    service=DiscussionService(b,collector=collector)
    collector.subscribe(lambda e:done.set() if e.event_type.value in ('DISCUSSION_COMPLETED','DISCUSSION_FAILED') else None)
    with pytest.raises(ValueError):DiscussionStartRequest(question='Q',mode='kv:v3',kv_recompute_tokens=1)
    request=dict(question='Q',rounds=1,mode=mode,max_new_tokens=2)
    if mode=='mixed':request['handoff_plan']=[dict(round=1,source='A',target='B',mode='token'),dict(round=1,source='B',target='A',mode='kv:v3')]
    with TestClient(create_app(collector=collector,discussion_service=service)) as client:
        assert client.post('/api/discussions',json=request).status_code==202
        assert done.wait(20)
        result=client.get('/api/discussions/latest').json()['session']
        assert result['status']=='completed',result['error']
        repairs=[h for h in result['handoffs'] if h['executed_kv_strategy']=='numeric_slot_delta']
        assert len(repairs)==(1 if mode=='mixed' else 2)
        assert result['summary']['kv_v3_handoffs']==len(repairs)
        assert all(h['repaired_token_indices']==[0] and h['repair_layer_token_count']==2 and h['fallback_reason'] is None for h in repairs)

@pytest.mark.parametrize('raw',[(100,101),(100,2),(100,3)])
def test_v3_receiver_pending_and_reverse(raw):
    b,r=build();s=r.session
    b.outputs=list(raw);s.generate_message('A','B',max_new_tokens=len(raw))
    b.outputs=list(raw);s.generate_message('B','A',max_new_tokens=len(raw))
    initial=s._states.copy();saved={n:copies(x.cache) for n,x in initial.items()}
    receipt=s.handoff_kv('A','B',strategy=V3)
    assert receipt.receiver_pending_materialized and s.state('B').pending_token_id is None
    assert s.state('B').draft_message==initial['B'].draft_message
    assert receipt.receiver_message_prefill_count==1
    reverse=s.handoff_kv('B','A',strategy=V3)
    assert reverse.received_token_ids==raw
    for n,state in initial.items():unchanged(state.cache,saved[n]);s.state(n).validate()

@pytest.mark.parametrize('failure',['corrupt','contract'])
def test_v3_strict_protocol_rejects_before_commit(failure):
    from token_kv_agent_system.transport import InMemoryByteTransport
    b,r=build();s=r.session;s.generate_message('A','B',max_new_tokens=3)
    initial=s._states.copy();saved={n:copies(x.cache) for n,x in initial.items()}
    if failure=='corrupt':
        class Corrupt(InMemoryByteTransport):
            def transfer(self,data,**kw):return super().transfer(data,**kw)[:-1]
        s.transport=Corrupt()
    else:
        from token_kv_agent_system.incremental_kv_protocol import inspect_model_contract
        s._kv_contract=replace(inspect_model_contract(b),model_id='wrong-model')
    with pytest.raises(ValueError):s.handoff_kv('A','B',strategy=V3)
    for n,state in initial.items():assert s.state(n) is state;unchanged(state.cache,saved[n])


def test_existing_output_directory_is_never_overwritten(tmp_path):
    from token_kv_agent_system.kv_v3_diagnostics import main
    marker=tmp_path/'source.json';marker.write_text('original')
    with pytest.raises(FileExistsError):main(['--output-dir',str(tmp_path)])
    assert marker.read_text()=='original'


def test_export_timing_matches_identical_traces_chronologically():
    module=load_exporter();trace_time,check_trace=module.trace_time,module.check_trace
    calls=[dict(input_ids=[16],cache_length_before=10,cache_length_after=11,
                attention_mask=[[1]*11],effective_rotary_position_ids=[[[10]]])]
    logs=[dict(input_ids=[16],previous_cache_tokens=10,backend_forward_ms=ms,
               started=i,elapsed_seconds=.1) for i,ms in enumerate((1.,2.,3.))]
    first=trace_time(logs,calls);second=trace_time(logs,calls,first['end_index'])
    assert first['forward_ms']==1 and second['forward_ms']==2
    check_trace(calls)
    with pytest.raises(ValueError):trace_time(logs,calls,3)


def test_export_refuses_existing_destination(tmp_path):
    export=load_exporter().export
    out=tmp_path/'evidence';out.mkdir();(out/'marker').write_text('preserved')
    with pytest.raises(FileExistsError):export(tmp_path/'absent',out)
    assert (out/'marker').read_text()=='preserved'


def load_exporter():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('kv_v3_export',Path(__file__).resolve().parents[1]/'export_kv_v3_diagnostics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
