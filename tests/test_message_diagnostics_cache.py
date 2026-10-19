"""Real CPU numerical oracle, exact inputs and scoped diagnostic hooks."""
import pytest

from token_kv_agent_system.agent.chat_framing import ChatFraming
from token_kv_agent_system.message_diagnostics import (ForwardTrace, cache_fingerprint,
    greedy_from_prefill, token_oracle, logits_difference)
from token_kv_agent_system.model.state_cache import require_disjoint_caches
from test_stateful_token_cache import backend


def test_cpu_same_ids_incremental_vs_full_prefill_and_actual_position_mask():
    b=backend()
    prefix=ChatFraming(b.chat_tokenizer,'You are Agent B.').assistant.prefix_ids
    ids=(30,31,32)
    initial=b.prefill(input_token_ids=ids)
    saved=cache_fingerprint(initial.cache)
    continuation=b.continue_from_cache
    forward=b._timed_forward
    with ForwardTrace(b,prefix) as trace:
        out=b.continue_from_cache(prefix,previous_cache=b.clone_cache(initial.cache),
                                  logical_cached_token_count=3)
    assert b.continue_from_cache==continuation
    assert b._timed_forward==forward
    assert trace.prefix_result is not None
    row=trace.calls[0]
    assert row['cache_length_before']==3
    assert row['attention_mask']==[[1]*(3+len(prefix))]
    assert row['effective_rotary_position_ids']==[[list(range(3,3+len(prefix)))]]
    assert row['supplied_position_ids'] is None and row['supplied_cache_position'] is None
    full_ids=ids+prefix
    full=b.prefill(input_token_ids=full_ids)
    generated=greedy_from_prefill(b,full,2)
    view={'template_ids_equal':True,'ledger_token_ids':full_ids}
    report=token_oracle(b,view,out.last_token_logits,out.cache,generated,2)
    assert report['input_ids_equal'] and report['generation_ids_equal']
    assert report['logits']['max_abs']<1e-6
    assert report['logits']['top1_equal']
    assert all(layer['key']['max_abs']<1e-6 and layer['value']['max_abs']<1e-6
               for layer in report['kv_differences'])
    assert cache_fingerprint(initial.cache)==saved
    require_disjoint_caches(initial.cache,out.cache)


def test_oracle_refuses_different_template_input_before_forward():
    b=backend()
    def forbidden(**kwargs):
        raise AssertionError('should not compute logits for different inputs')
    b.prefill=forbidden
    result=token_oracle(b,{'template_ids_equal':False,'first_input_difference':7},None,None,(),2)
    assert result=={'status':'input_mismatch','first_difference':7}


def test_diagnostic_hooks_restore_after_forward_failure_and_do_not_mutate_source():
    b=backend()
    initial=b.prefill(input_token_ids=(30,31,32))
    saved=cache_fingerprint(initial.cache)
    forward=b._timed_forward
    continuation=b.continue_from_cache
    original=b._model.forward
    def fail(**kwargs):
        original(**kwargs)
        raise RuntimeError('diagnostic failure after cache growth')
    b._model.forward=fail
    rotary=b._model.model.rotary_emb
    hooks=len(rotary._forward_pre_hooks)
    with pytest.raises(RuntimeError):
        with ForwardTrace(b):
            b.continue_from_cache((33,),previous_cache=b.clone_cache(initial.cache),
                                  logical_cached_token_count=3)
    assert b._timed_forward==forward and b.continue_from_cache==continuation
    assert len(rotary._forward_pre_hooks)==hooks
    assert cache_fingerprint(initial.cache)==saved


def test_relative_error_and_top1_are_reported_without_bitwise_requirement():
    import torch
    report=logits_difference(torch.tensor([1.,2.,3.]),torch.tensor([1.01,2.,3.]))
    assert report['max_abs']>0 and report['mean_abs']>0 and report['relative_l2']>0
    assert report['top1_equal']


@pytest.mark.parametrize('raw', [(100,101,2),(100,101,3),(100,101),(100,)])
def test_harness_fixed_decode_three_paths_share_snapshot_and_never_prefill_m(raw):
    from test_incremental_kv_transport import make_backend
    from token_kv_agent_system.discussion import DiscussionRuntime, DiscussionConfig
    from token_kv_agent_system.message_understanding_probe import branch
    from test_message_diagnostics import DisplayBackend
    b=make_backend()
    # Diagnostic display only; no decode/re-encode occurs in handoff.
    b.decode=lambda ids,**kwargs:DisplayBackend.decode(b,ids,**kwargs)
    r=DiscussionRuntime(b,config=DiscussionConfig(rounds=1,max_new_tokens=len(raw)))
    for n in ('A','B'):
        r.session.append_local_user(n,'Read the peer message.')
    b.outputs=list(raw)
    message=r.session.generate_message('A','B',max_new_tokens=len(raw))
    snapshots={n:r.session.state(n) for n in ('A','B')}
    b.prefill=lambda **kwargs:pytest.fail('hidden Prefill in communication or diagnostic view')
    result=branch(r,snapshots,message=message,stage='Repeat A.',origin='sender_decode',
                  modes=('token','kv:v0','kv:v1'),oracle=False,expected='8391')
    assert all(row['engineering_status']=='verified' for row in result['modes']), result
    assert len({tuple(row['context']['ledger_token_ids']) for row in result['modes']})==1
    assert all(row['context']['template_ids_equal'] for row in result['modes'])
    assert [row['receiver_message_prefill_count'] for row in result['modes']]==[
        len(raw)-int(raw[-1] in (2,3)),0,0]
    assert all(row['model_assessment']['sender_contains_expected'] is False
               and row['model_assessment']['receiver_recall_given_sender_success'] is None
               for row in result['modes'])
    assert result['modes'][1]['wire_sha256']==result['modes'][2]['wire_sha256']
    assert result['modes'][1]['receipt']['sender_supplemented_token_count']==int(raw[-1] not in (2,3))
    assert all(r.session.state(n) is snapshots[n] for n in snapshots)


def test_harness_rejects_artificial_fixture_as_decode_kv_before_transport():
    from test_incremental_kv_transport import make_backend
    from token_kv_agent_system.discussion import DiscussionRuntime, DiscussionConfig
    from token_kv_agent_system.message_understanding_probe import branch
    from token_kv_agent_system.message_diagnostics import token_injection
    from test_message_diagnostics import DisplayBackend
    b=make_backend()
    b.decode=lambda ids,**kwargs:DisplayBackend.decode(b,ids,**kwargs)
    r=DiscussionRuntime(b,config=DiscussionConfig(rounds=1,max_new_tokens=2))
    m=token_injection(r.session,'Secret')
    snapshots={n:r.session.state(n) for n in ('A','B')}
    result=branch(r,snapshots,message=m,stage='Repeat A.',origin='artificial_token_fixture',
                  modes=('kv:v0',),oracle=False)
    assert result['modes'][0]['engineering_status']=='failed'
    assert 'never Decode KV' in result['modes'][0]['error']
    assert r.session.transport.records==()
