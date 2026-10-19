from dataclasses import asdict
import json
import pytest
from token_kv_agent_system.private_exchange_probe import (
 prepare,reply,own_messages,structural_messages,FIXTURES,STAGE_VARIANTS,
 score_b,candidate_scene,A_NO_REQUEST,main)
from token_kv_agent_system.secret_sum import STAGES,PUBLIC_TASK,private_prompt
from test_incremental_kv_transport import make_backend as tiny_backend

def make_backend():
 b=tiny_backend()
 specials={1:"<s>",2:"</s>",3:"<eos>"}
 b.decode=lambda ids,skip_special_tokens=False: "".join(("" if skip_special_tokens else specials[i]) if i in specials else chr(i-10) for i in ids)
 return b


@pytest.mark.parametrize('b',[26,31])
def test_private_context_isolation_and_exact_fixture_wire(b):
 r,d=prepare(make_backend(),b,FIXTURES['original'])
 assert d['origin']=='fixed_fixture'
 assert f'Your private number is {b}.' in d['initial_contexts']['B']['decoded_ledger']
 assert '17' not in d['initial_contexts']['B']['decoded_ledger']
 assert str(b) not in d['initial_contexts']['A']['decoded_ledger']
 assert d['sender']['message_token_ids']==d['receipt']['received_token_ids']
 assert r.session.state('B').pending_token_id is None
 row=reply(r,STAGES.b_first,paired=True)
 assert row['context']['template_ids_equal'] and row['full_pair']['input_ids_equal']
 assert row['input_ids']==row['full_pair']['input_ids']
 assert all(x['key_shape'][2]==len(row['input_ids']) for x in row['full_pair']['t_cache_shapes'])


def test_no_message_does_not_create_sender():
 r,d=prepare(make_backend(),26)
 assert d['origin']=='no_message' and d['receipt'] is None and d['sender'] is None
 assert not d['wire_records'] and not r.session.state('B').received_messages


@pytest.mark.parametrize('condition',['A','A1','A2','B','C','D'])
def test_own_controls_have_no_peer_secret(condition):
 rows=own_messages(26,condition)
 assert rows[0]['role']=='system' and '26' in rows[0]['content']
 assert all('17' not in x['content'] for x in rows)
 if condition=='C':assert rows[-1]['content']==STAGES.b_first


@pytest.mark.parametrize('structure',['A','B','C','D'])
def test_structures_never_masquerade_as_assistant(structure):
 rows=structural_messages(26,FIXTURES['original'],structure)
 assert rows[0]['content']==private_prompt('B',26)
 assert all(x['role']=='user' for x in rows[1:])
 assert FIXTURES['original'] in '\n'.join(x['content'] for x in rows)
 assert STAGES.b_first in '\n'.join(x['content'] for x in rows)


def test_stage_controls_do_not_publish_any_secret_or_sum():
 for name,s in STAGE_VARIANTS.items():
  assert s is None or not any(n in s for n in ('17','26','31','43','48'))
 assert STAGE_VARIANTS['S1']==STAGES.b_first
 assert A_NO_REQUEST=='Tell Agent B your private number. Do not calculate yet.'


def test_sum_alone_does_not_pass_source_score():
 assert not score_b('17 + 26 = 43',26)['strict_pass']
 assert score_b("Agent A's number is 17. My number is 26. The sum is 43.",26)['strict_pass']
 assert not score_b("Agent A's number is 26. My number is 17. The sum is 43.",26)['strict_pass']


def test_candidate_is_local_and_negative_has_no_a_handoff():
 before=asdict(STAGES)
 r=candidate_scene(make_backend(),31,'S3',negative=True)
 assert asdict(STAGES)==before
 assert r['prompt_changes']['a_first']==STAGES.a_first and r['prompt_changes']['final']==STAGES.final
 assert len(r['wire_records'])==1 and r['wire_records'][0]['source']=='B'
 assert r['contexts'][1]['latest_sender'] is None
 assert '17' not in r['contexts'][1]['decoded_ledger']


def test_reject_existing_results_before_model_load(tmp_path):
 (tmp_path/'old.json').write_text('evidence')
 with pytest.raises(FileExistsError):main(['--phase','pair','--output-dir',str(tmp_path)])
 assert (tmp_path/'old.json').read_text()=='evidence'


@pytest.mark.parametrize('error,category', [('MPS out of memory','oom'),('checkpoint missing','execution_exception')])
def test_probe_saves_load_failure_without_retry(monkeypatch,tmp_path,error,category):
 import sys
 from types import SimpleNamespace
 import token_kv_agent_system.private_exchange_probe as probe
 def fail(*args,**kwargs):raise RuntimeError(error)
 monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(mps=SimpleNamespace(empty_cache=lambda:None)))
 monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=fail)))
 monkeypatch.setattr(probe,'memory',lambda *a:dict(swap_used_mib=0,pressure_level=1))
 dest=tmp_path/'run'
 assert probe.main(['--phase','pair','--output-dir',str(dest)])==1
 r=json.loads((dest/'failure.json').read_text())
 assert r['error']==error and r['failure_category']==category
 assert (dest/'memory-samples.json').exists()
 assert not (dest/'completion.json').exists()
 assert not list(dest.glob('real-pair-*.json'))


def test_stage_is_only_variable_in_fixture_pair():
 r,d=prepare(make_backend(),26,FIXTURES['original'])
 from token_kv_agent_system.message_diagnostics import context_view
 before=r.session.state('B').token_ledger
 for stage in STAGE_VARIANTS.values():
  view=context_view(r.session.state('B'),stage=stage,generation_prefix=True)
  assert view['ledger_token_ids'][:len(before)]==before
  assert view['pending_token_id'] is None
 assert r.session.state('B').token_ledger==before
