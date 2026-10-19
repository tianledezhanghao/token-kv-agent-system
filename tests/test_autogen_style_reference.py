"""Independent orchestration, trace integrity, stop safeguards and production isolation."""
import asyncio
from dataclasses import asdict
import importlib.util

import pytest

from token_kv_agent_system.autogen_style_reference import (Event, STRUCTURES, MODES,
    ReferenceHarness, group_incoming, generate, overlap, repetition_report,
    experimental_stop, STAGES, systems, token_comparison)
from token_kv_agent_system.discussion import DiscussionRuntime
from token_kv_agent_system.discussion_prompts import STAGE_INSTRUCTIONS
from test_message_diagnostics import DisplayBackend
from test_incremental_kv_transport import make_backend


@pytest.mark.parametrize('structure', STRUCTURES)
def test_grouping_preserves_sources_roles_question_content_and_stage(structure):
    events=[Event('user','Question-Q','user_task'), Event('A','A-actual','fixed_fixture')]
    messages=group_incoming(structure,'B',events,'Stage-S')
    assert all(m['role']=='user' for m in messages)
    assert [s for m in messages for s in m['sources']]==[asdict(e) for e in events]+[
        dict(source='controller',content='Stage-S',origin='stage_instruction')]
    assert all(text in '\n'.join(m['content'] for m in messages) for text in ('Question-Q','A-actual','Stage-S'))
    assert len(messages)=={'A':3,'B':2,'C':1,'D':3}[structure]
    if structure=='D':
        assert messages[1]['content']=='A said:\nA-actual'


@pytest.mark.parametrize('structure', STRUCTURES)
def test_no_message_stays_absent_without_synthesized_sender(structure):
    messages=group_incoming(structure,'B',[Event('user','Q','user_task')],'S')
    assert all(s['source']!='A' for m in messages for s in m['sources'])
    assert all('A_MESSAGE' not in m['content'] and 'Message from A' not in m['content'] for m in messages)


def stub_generate(backend,messages,limit):
    text=f'Local response {backend.next_id}'
    backend.next_id+=1
    return dict(messages=messages,input_ids=(10,),output_ids=(100,2),text=text,
                stop_reason='eos',truncated=False,prefill='complete_history',max_new_tokens=limit)


@pytest.mark.parametrize('structure',STRUCTURES)
@pytest.mark.parametrize('mode',MODES)
def test_independent_contexts_new_deliveries_own_assistant_and_real_decode(monkeypatch,structure,mode):
    import token_kv_agent_system.autogen_style_reference as reference
    b=DisplayBackend(); b.next_id=0
    monkeypatch.setattr(reference,'generate',stub_generate)
    h=ReferenceHarness(b,'User question',structure=structure,mode=mode)
    original_b=list(h.contexts['B'])
    row=h.turn('A',STAGES[mode].a_first,128)
    assert h.contexts['B']==original_b
    h.deliver('A','B',row['text'],origin='sender_decode')
    reply=h.turn('B',STAGES[mode].b_first,128)
    h.deliver('B','A',reply['text'],origin='sender_decode')
    update=h.turn('A',STAGES[mode].a_followup,128)
    for agent,context in h.contexts.items():
        source_events=[s for m in context for s in m['sources']]
        assert sum(s['source']=='user' for s in source_events)==1
        assert context[0]['role']=='system'
        assert context[-1]['role']=='assistant'
        assert context[-1]['sources'][0]['source']==agent
    assert any(m['role']=='assistant' and m['content']==row['text'] for m in update['messages'])
    received=[s for m in reply['messages'] for s in m['sources'] if s['source']=='A']
    assert received==[dict(source='A',content=row['text'],origin='sender_decode')]
    assert [r['agent'] for r in h.trace]==['A','B','A']
    assert all(r['origin']=='real_local_decode' for r in h.trace)
    assert h.contexts['A'] is not h.contexts['B']


def test_fixture_decode_origin_distinction_and_validation(monkeypatch):
    import token_kv_agent_system.autogen_style_reference as reference
    b=DisplayBackend(); b.next_id=0
    monkeypatch.setattr(reference,'generate',stub_generate)
    h=ReferenceHarness(b,'Q')
    h.deliver('A','B','fixture',origin='fixed_fixture')
    row=h.turn('B','read',64)
    assert row['deliveries'][0]['origin']=='fixed_fixture'
    assert row['origin']=='real_local_decode'
    for source,target,content,origin in [('A','A','x','sender_decode'),('A','B','','sender_decode'),('A','B','x','unknown')]:
        with pytest.raises(ValueError): h.deliver(source,target,content,origin=origin)
    with pytest.raises(ValueError): ReferenceHarness(b,'')
    with pytest.raises(ValueError): h.run(rounds=3)
    with pytest.raises(ValueError): systems('unknown')


@pytest.mark.parametrize('output,limit,stop', [((100,2),2,'eos'),((100,3),2,'eos'),((100,101),2,'max_new_tokens'),((2,),1,'eos')])
def test_full_prefill_actual_ids_and_eos_at_budget_not_truncated(monkeypatch,output,limit,stop):
    import token_kv_agent_system.autogen_style_reference as reference
    b=make_backend()
    b.decode=lambda ids,**kw: 'decoded output'
    monkeypatch.setattr(reference,'greedy_from_prefill',lambda backend,result,limit:output)
    messages=[dict(role='system',content='S'),dict(role='user',content='Q')]
    row=generate(b,messages,limit)
    assert tuple(b.chat_tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))==row['input_ids']
    assert row['output_ids']==output and row['stop_reason']==stop
    assert row['truncated']==(stop=='max_new_tokens')
    assert row['prefill']=='complete_history'
    with pytest.raises(ValueError): generate(b,messages,0)


def test_repetition_metrics_do_not_turn_short_agreement_into_failure():
    assert overlap('x=3','x=3')==dict(exact_match=True,character_overlap=1.0,long_exact_copy=False)
    assert not overlap('','')['exact_match']
    assert not overlap('answer20','answer21')['exact_match']
    long='As the reviewer, the original answer does not address the question. '*3
    turns=[dict(agent=a,text=long,stop_reason='eos') for a in ('A','B','A')]
    report=repetition_report(turns)
    assert report['adjacent_exact_rate']==1 and report['same_agent'][0]['long_exact_copy']
    assert experimental_stop(turns,max_turns=5)['reason']=='long_repetition'


@pytest.mark.parametrize('text', ['correct','The answer is correct, but it does not address the question.','APPROVE'])
def test_stop_does_not_treat_model_signal_as_verified_answer(text):
    turn=dict(agent='B',text=text,stop_reason='eos')
    assert not experimental_stop([turn],max_turns=5,review_signal='approve')['stop']
    assert experimental_stop([turn],max_turns=5,review_signal='correction')['reason']=='specific_correction'
    assert experimental_stop([turn],max_turns=5,review_signal='approve',content_verified=True)['success']
    assert not experimental_stop([{**turn,'agent':'A'}],max_turns=5,review_signal='approve',content_verified=True)['stop']
    assert experimental_stop([turn],max_turns=1)['reason']=='max_turns'
    assert experimental_stop([{**turn,'stop_reason':'max_new_tokens'}],max_turns=5)['reason']=='truncated'


@pytest.mark.parametrize('mode',MODES)
def test_comparison_runtime_real_token_no_full_prefill_and_defaults_unchanged(mode):
    b=make_backend(); b.decode=lambda ids,**kw:DisplayBackend.decode(b,ids,**kw)
    original=b.prefill
    calls=[]
    def init_prefill(**kwargs):
        calls.append(kwargs)
        if len(calls)>2: pytest.fail('production TOKEN must not full Prefill history')
        return original(**kwargs)
    b.prefill=init_prefill
    record=token_comparison(b,'10 + 10 = ?',mode,2)
    assert len(calls)==2
    assert record['discussion']['status']=='completed'
    assert len(record['discussion']['turns'])==4
    assert all(t['receipt']['received_token_ids']==t['message']['message_token_ids'] for t in record['discussion']['turns'])
    assert DiscussionRuntime(DisplayBackend()).prompt_config.stages==STAGE_INSTRUCTIONS


@pytest.mark.skipif(importlib.util.find_spec('autogen_agentchat') is None,reason='optional isolated AutoGen environment')
def test_official_agentchat_source_roles_independent_context_and_local_only(monkeypatch):
    import token_kv_agent_system.autogen_local_client as local
    from token_kv_agent_system.autogen_style_reference import group_incoming
    b=DisplayBackend(); b.next_id=0
    monkeypatch.setattr(local,'generate',stub_generate)
    record=asyncio.run(local.official_team(b,'Shared-Q',systems('critic'),128))
    turns=record['turns']
    assert [r['agent'] for r in turns]==['A','B','A','B','A']
    assert 'Maximum number of turns 5' in record['result']['stop_reason']
    assert [m['role'] for m in turns[1]['messages']]==['system','user','user']
    assert turns[1]['messages'][-1]['source']=='A'
    assert turns[2]['messages'][2]['role']=='assistant'
    assert turns[2]['messages'][2]['source']=='A'
    assert turns[2]['messages'][3]['source']=='B'
    assert sum(m['source']=='user' for m in turns[-1]['messages'])==1
    fixed=asyncio.run(local.official_fixed(b,'Q','actual',systems('critic')['B'],'S',128))
    grouped=group_incoming('D','B',[Event('user','Q','user_task'),Event('A','actual','fixed_fixture')],'S')
    assert [m['content'] for m in fixed['turns'][0]['messages'][1:]]==[m['content'] for m in grouped]
    assert fixed['context'][-1]['type']=='AssistantMessage'


def test_reference_full_execution_uses_five_fresh_prefills_not_persistent_peer_cache():
    from dataclasses import replace
    b=make_backend()
    b.decode=lambda ids,**kw:DisplayBackend.decode(b,ids,**kw)
    original=b.prefill
    prefills=[]
    def prefill(**kwargs):
        result=original(**kwargs)
        prefills.append(result)
        return replace(result,first_token_id=100)
    b.prefill=prefill
    h=ReferenceHarness(b,'Question-Q',structure='A',mode='peer')
    record=h.run(limit=2)
    assert len(prefills)==len(record['turns'])==5
    assert len({id(p.cache) for p in prefills})==5
    assert len(record['deliveries'])==4
    assert all(d['origin']=='sender_decode' for d in record['deliveries'])
    assert all(t['stop_reason']=='max_new_tokens' for t in record['turns'])
    assert 'cache' not in h.__dict__
    assert len(record['final_contexts']['A'])!=len(record['final_contexts']['B'])


def test_failed_reference_generation_does_not_consume_pending_delivery(monkeypatch):
    import token_kv_agent_system.autogen_style_reference as reference
    h=ReferenceHarness(DisplayBackend(),'Q')
    h.deliver('A','B','actual',origin='fixed_fixture')
    before=tuple(h.pending['B']); context=list(h.contexts['B'])
    def fail(*args,**kwargs): raise RuntimeError('forward failed')
    monkeypatch.setattr(reference,'generate',fail)
    with pytest.raises(RuntimeError): h.turn('B','S',128)
    assert tuple(h.pending['B'])==before and h.contexts['B']==context
    assert not h.trace


def test_actual_token_capture_records_noncanonical_text_without_changing_ledger(monkeypatch):
    import token_kv_agent_system.message_diagnostics as diagnostics
    from token_kv_agent_system.autogen_style_reference import capture_token
    from token_kv_agent_system.discussion import DiscussionConfig
    b=make_backend(); b.decode=lambda ids,**kw:DisplayBackend.decode(b,ids,**kw)
    r=DiscussionRuntime(b,config=DiscussionConfig(rounds=1,max_new_tokens=2))
    old=diagnostics.context_view
    def noncanonical(*args,**kwargs):
        view=old(*args,**kwargs)
        view['template_ids_equal']=False
        return view
    monkeypatch.setattr(diagnostics,'context_view',noncanonical)
    record=capture_token(r)
    assert record['discussion']['status']=='completed'
    assert all(not row['context']['template_ids_equal'] for row in record['contexts'])
    assert all(row['context']['exact_span_coverage'] for row in record['contexts'])
    for row in record['contexts']:
        state=r.session.state(row['context']['agent'])
        assert tuple(row['context']['ledger_token_ids'])==state.token_ledger[:row['output']['sender_message_span'][0]]


def test_neutral_reader_matrix_separates_role_from_message_structure(monkeypatch):
    import token_kv_agent_system.autogen_style_reference as reference
    from token_kv_agent_system.autogen_style_probe import understanding, READ_SYSTEM
    b=DisplayBackend(); b.next_id=0
    monkeypatch.setattr(reference,'generate',stub_generate)
    rows={}
    understanding(b,[128],2,lambda name,row:rows.update({name:row}))
    assert len(rows)==32
    for row in rows.values():
        assert row['messages'][0]['content']==READ_SYSTEM
        assert row['system_variant']=='neutral_reader'
        if row['fixture'] is None:
            assert row['input_origin']=='no_message' and not row['deliveries']
        else:
            assert row['deliveries'][0]['origin']=='fixed_fixture'
            assert row['deliveries'][0]['content']==row['fixture']


def test_evidence_exporter_understands_actual_token_and_secret_result_schemas(tmp_path):
    import runpy
    from pathlib import Path
    from token_kv_agent_system.secret_sum import SecretSumConfig
    from token_kv_agent_system.secret_sum_validation import run_scene
    exporter=runpy.run_path(str(Path(__file__).resolve().parents[1]/'experiments/summarize_autogen_77g.py'))
    b=make_backend(); b.decode=lambda ids,**kw:DisplayBackend.decode(b,ids,**kw)
    token=token_comparison(b,'Q','critic',2)
    secret=run_scene(b,SecretSumConfig(max_new_tokens=2))
    assert len(exporter['generations'](token))==5
    rows=exporter['generations'](secret)
    assert len(rows)==3
    assert [r['output_ids'] for r in rows]==[t['message']['message_token_ids'] for t in secret['result']['turns']]+[secret['result']['final_message']['message_token_ids']]
    with pytest.raises(ValueError,match='not completed'):
        exporter['export']([tmp_path/'incomplete'],tmp_path/'out')
