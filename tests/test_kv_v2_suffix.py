"""V2 numerical cache oracle and transactional production paths on tiny CPU Qwen."""
from dataclasses import replace
import pytest
import torch
from test_incremental_kv_receive import build
from test_stateful_token_cache import copies, unchanged
from token_kv_agent_system.communication import HandoffChoice, KVStrategy, suffix_recompute_count
from token_kv_agent_system.model.delta_repair import delta_prefix, append_delta, rotary_tables, rephase_key
from token_kv_agent_system.model.decode_kv import cache_layers
from token_kv_agent_system.incremental_kv import prepare_incremental_message
from token_kv_agent_system.incremental_kv_protocol import inspect_model_contract
from token_kv_agent_system.model.kv_adapter import restore_huggingface_kv_cache
from token_kv_agent_system.agent.state import working_state, append_working_ids
from token_kv_agent_system.message_diagnostics import ForwardTrace

V2 = KVStrategy.SUFFIX_RECOMPUTE_DELTA

@pytest.mark.parametrize('n,t,r,k', [(1,None,None,1),(5,0,None,0),(5,1,None,1),(5,2,None,2),
    (5,5,None,5),(5,9,None,5),(5,None,.25,2),(1,None,.25,1),(5,None,0,0),(5,None,1,5)])
def test_count(n,t,r,k):
    assert suffix_recompute_count(n,tokens=t,ratio=r)==k
    assert HandoffChoice.parse('kv:v2',recompute_tokens=t,recompute_ratio=r).kv_strategy==V2

@pytest.mark.parametrize('kw',[dict(tokens=-1),dict(tokens=True),dict(tokens=1.2),dict(ratio=-.1),
    dict(ratio=1.1),dict(ratio=float('nan')),dict(ratio=float('inf')),dict(ratio=True),dict(tokens=1,ratio=.2)])
def test_invalid_policy(kw):
    with pytest.raises(ValueError): suffix_recompute_count(5,**kw)


def test_empty_and_other_strategies():
    with pytest.raises(ValueError): suffix_recompute_count(0)
    for mode in ('token','kv:v0','kv:v1'):
        with pytest.raises(ValueError): HandoffChoice.parse(mode,recompute_tokens=1)
    b,r=build(raw=(2,),choice=HandoffChoice.parse('kv:v2'))
    result=r.run()
    assert result.status=='failed' and 'empty message content' in result.error
    assert not r.session.transport.records

@pytest.mark.parametrize('raw',[(100,101,102,103,2),(100,101,102,103),(100,2),(100,)])
@pytest.mark.parametrize('k',[0,1,2,4,9])
def test_real_suffix_forward_positions_mask_numerical_oracle_and_isolation(raw,k):
    b,r=build(raw=raw);s=r.session
    s.append_local_user('B','independent history')
    m=s.generate_message('A','B',max_new_tokens=len(raw))
    initial={n:s.state(n) for n in ('A','B')}; saved={n:copies(initial[n].cache) for n in initial}
    contract=inspect_model_contract(b)
    envelope,_=prepare_incremental_message(initial['A'],m,contract=contract)
    content=envelope.message_token_ids; actual=min(k,len(content)); retained=len(content)-actual
    frame=initial['B'].framing.incoming('A','B')
    # Independent cache oracle: real model forward on precisely H_B + V1 prefix.
    oracle=append_working_ids(working_state(initial['B']),frame.prefix_ids); start=oracle.cached_token_count
    delta=cache_layers(restore_huggingface_kv_cache(envelope.delta,model_id=b.model_name,
        model_config=b.kv_model_config,model_dtype=contract.dtype,target_device=b.device,
        transformers_config=b._model.config,torch_module=b._torch).cache)
    old=rotary_tables(b,list(range(envelope.sender_span_start,envelope.sender_span_start+len(content))))
    new=rotary_tables(b,list(range(start,start+len(content))))
    delta=tuple((rephase_key(key,*old,*new),v) for key,v in delta)
    if retained:
        oracle=replace(oracle,cache=append_delta(oracle.cache,delta_prefix(delta,retained),b._model.config),
            cached_token_count=start+retained,token_ledger=oracle.token_ledger+content[:retained])
    if actual: oracle=append_working_ids(oracle,content[retained:])
    oracle=append_working_ids(oracle,frame.closing_ids)
    b.calls.clear()
    with ForwardTrace(b) as trace:
        receipt=s.handoff_kv('A','B',strategy=V2,recompute_tokens=k,measure_timing=True)
    assert receipt.receiver_message_prefill_count==actual
    assert receipt.recompute_forward_count==int(actual>0)
    start,end=receipt.receiver_target_span
    suffix_calls=[c for c in trace.calls if c['cache_length_before']==start+retained and c['input_ids']==list(content[retained:])]
    assert len(suffix_calls)==int(actual>0)
    if actual:
        call=suffix_calls[0]
        assert call['attention_mask']==[[1]*(start+len(content))]
        assert call['effective_rotary_position_ids']==[[list(range(start+retained,end))]]
        assert call['cache_length_after']==end
    assert receipt.wire_byte_count==s.transport.records[-1].byte_count
    assert receipt.timings_ms['suffix_recompute']>0 if actual else receipt.timings_ms['suffix_recompute']==0
    for layer,(key,value) in zip(s.state('B').cache.layers,copies(oracle.cache)):
        torch.testing.assert_close(layer.keys,key,rtol=0,atol=0)
        torch.testing.assert_close(layer.values,value,rtol=0,atol=0)
    for n in initial: unchanged(initial[n].cache,saved[n]);s.state(n).validate()
    # k=0 is V1 numerical execution, same wire, no additional message forward.
    if k==0:
        v2=s.state('B');s._states=initial.copy()
        v1=s.handoff_kv('A','B',strategy=KVStrategy.ROPE_ALIGNED_DELTA)
        assert v1.wire_byte_count==receipt.wire_byte_count
        for layer,(key,value) in zip(v2.cache.layers,copies(s.state('B').cache)):
            assert torch.equal(layer.keys,key) and torch.equal(layer.values,value)

@pytest.mark.parametrize('failure',['suffix','closing','sender_close','corrupt','rope'])
def test_mutation_failure_rolls_back_both_and_no_fallback(monkeypatch,failure):
    from token_kv_agent_system.transport import InMemoryByteTransport
    b,r=build(raw=(100,101,102,2));s=r.session
    s.generate_message('A','B',max_new_tokens=4)
    states={n:s.state(n) for n in ('A','B')};saved={n:copies(states[n].cache) for n in states}
    original=b.continue_from_cache;frame=states['B'].framing.incoming('A','B')
    if failure=='corrupt':
        class Corrupt(InMemoryByteTransport):
            def transfer(self,data,**kw):return super().transfer(data,**kw)[:-1]
        s.transport=Corrupt()
    elif failure=='rope':monkeypatch.setattr(b._model.model.rotary_emb,'rope_type','dynamic')
    else:
        def broken(ids,**kw):
            out=original(ids,**kw)
            if ((failure=='suffix' and tuple(ids)==(102,)) or
                (failure=='closing' and tuple(ids)==frame.closing_ids) or
                (failure=='sender_close' and kw['logical_cached_token_count']==states['A'].cached_token_count)):
                raise RuntimeError('mutated work forward failure')
            return out
        monkeypatch.setattr(b,'continue_from_cache',broken)
    with pytest.raises((ValueError,RuntimeError)):s.handoff_kv('A','B',strategy=V2,recompute_tokens=1)
    for n in states:assert s.state(n) is states[n];unchanged(states[n].cache,saved[n])

@pytest.mark.parametrize('plan',[('kv:v2',)*6,('token','kv:v2','kv:v0','kv:v2','kv:v1','token')])
def test_multi_round_mixed_runtime_receipts_and_events(plan):
    choices=tuple(HandoffChoice.parse(m,recompute_ratio=.25) if m=='kv:v2' else HandoffChoice.parse(m) for m in plan)
    b,r=build(plan=choices,rounds=3);events=[];r.event_sink=lambda kind,row:events.append((kind,row))
    b.prefill=lambda **kw:pytest.fail('full history Prefill')
    result=r.run();assert result.status=='completed',result.error
    for h,c in zip(result.handoffs,choices):
        assert h.executed_kv_strategy==(c.kv_strategy.value if c.kv_strategy else None)
        if c.kv_strategy==V2:assert h.receiver_recomputed_token_count==1 and h.recompute_forward_count==1
        assert h.fallback_reason is None and h.cache_ledger_consistent
    assert sum(kind=='HANDOFF_COMPLETED' for kind,_ in events)==6


def test_delta_prefix_layout_validation():
    pairs=((torch.ones(1,2,4,8),torch.ones(1,2,4,8)),)*2
    assert delta_prefix(pairs,2)[0][0].shape==(1,2,2,8)
    for bad,n in [(pairs,5),((),1),(pairs,-1),((pairs[0],(torch.ones(1,2,3,8),torch.ones(1,2,3,8))),1),
        ((pairs[0],(pairs[0][0].double(),pairs[0][1].double())),1)]:
        with pytest.raises(ValueError):delta_prefix(bad,n)


def test_api_policy_secret_runtime_and_cli(monkeypatch):
    from token_kv_agent_system.dashboard.discussion_service import DiscussionStartRequest
    from token_kv_agent_system.secret_sum import SecretSumRuntime
    from token_kv_agent_system.discussion_demo import main
    request=DiscussionStartRequest(question='Q',scenario='secret_sum',rounds=1,mode='kv:v2',kv_recompute_tokens=2,max_new_tokens=2)
    b,r=build();result=SecretSumRuntime(b,config=request.runtime_config()).run()
    assert result.status=='completed' and all(h.receiver_recomputed_token_count==2 for h in result.handoffs)
    for kw in [dict(kv_recompute_tokens=-1),dict(kv_recompute_ratio=2),dict(kv_recompute_tokens=1,kv_recompute_ratio=.25)]:
        with pytest.raises(ValueError):DiscussionStartRequest(question='Q',mode='kv:v2',**kw)
    import token_kv_agent_system.discussion_demo as demo
    monkeypatch.setattr(demo,'HuggingFaceModelBackend',lambda *a,**kw: b)
    # tiny test backend has no metadata: capture validated config before load instead.
    class Seen(Exception):pass
    configs=[]
    def runtime(*a,config):configs.append(config);raise Seen()
    monkeypatch.setattr(demo,'DiscussionRuntime',runtime)
    from types import SimpleNamespace
    b._metadata=SimpleNamespace(model_name='tiny',device='cpu',dtype='float32')
    assert main(['--mode','kv','--kv-strategy','v2','--kv-recompute-ratio','.25'])==1
    assert configs[0].handoff_choice.recompute_ratio==.25


def test_controlled_harness_single_sender_same_packets_full_ids_and_two_way():
    from token_kv_agent_system.kv_v2_validation import controlled_case
    b,r=build(raw=(100,101,102,103,2))
    from test_message_diagnostics import DisplayBackend
    b.decode=lambda ids,**kw:DisplayBackend.decode(b,ids,**kw)
    result=controlled_case(b,secret_b=26)
    assert len(result['modes'])==8
    hashes={row['packets'][0]['sha256'] for row in result['modes'][1:]}
    assert hashes=={result['source']['source_packet_sha256']}
    assert len({tuple(row['receiver_generation_context']['ledger_token_ids']) for row in result['modes']})==1
    assert all(row['engineering_status']=='verified' and row['reverse_receipt'] for row in result['modes'])

@pytest.mark.parametrize('scenario,mode',[('discussion','kv:v2'),('secret_sum','mixed')])
def test_actual_rest_v2_receipt_summary_and_no_token_fallback(scenario,mode):
    from threading import Event
    from fastapi.testclient import TestClient
    from token_kv_agent_system.dashboard.app import create_app
    from token_kv_agent_system.dashboard.discussion_service import DiscussionService
    from token_kv_agent_system.metrics import MetricCollector
    b,r=build();b.outputs=[];collector=MetricCollector();service=DiscussionService(b,collector=collector);done=Event()
    collector.subscribe(lambda e:done.set() if e.event_type.value in ('DISCUSSION_COMPLETED','DISCUSSION_FAILED') else None)
    request=dict(question='Q',scenario=scenario,rounds=1,mode=mode,kv_recompute_ratio=.25,max_new_tokens=2)
    if mode=='mixed':request['handoff_plan']=[dict(round=1,source='A',target='B',mode='token'),dict(round=1,source='B',target='A',mode='kv:v2')]
    with TestClient(create_app(collector=collector,discussion_service=service)) as client:
        assert client.post('/api/discussions',json=request).status_code==202
        assert done.wait(20)
        result=client.get('/api/discussions/latest').json()['session']
        assert result['status']=='completed',result['error']
        v2=[h for h in result['handoffs'] if h['executed_kv_strategy']=='suffix_recompute_delta']
        assert len(v2)==(1 if mode=='mixed' else 2)
        assert all(h['receiver_recomputed_token_count']==1 and h['recompute_forward_count']==1 and h['wire_bytes']>0 and h['fallback_reason'] is None for h in v2)
        assert result['summary']['kv_v2_handoffs']==len(v2)
