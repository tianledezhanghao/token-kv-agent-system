"""7.8C contracts; tiny CPU outputs are engineering tests, never model quality."""
import json
from pathlib import Path
import pytest
from test_private_exchange_probe import make_backend
from token_kv_agent_system.private_exchange_validation import (
    PROMPTS, POSITIVES, NEGATIVES, CRITERIA, PUBLIC_TASK, STAGES, frozen_spec,
    run_case, score_texts, planned_cases, main, repeats_request)

FROZEN_SHA='0a7281eaa378165640fa63c313fe1b4178228a39d7538ef76e8ffa775f487376'


def test_exact_frozen_configuration_and_committed_artifact():
    spec=frozen_spec()
    assert spec['sha256']==FROZEN_SHA
    assert json.loads((Path(__file__).parents[1]/'docs/token-baseline-78c.json').read_text())==json.loads(json.dumps(spec))
    assert PROMPTS['P0']==STAGES.b_first
    assert PROMPTS['P1']=='Only mark Agent A\'s number as unknown if Agent A did not provide it. Your own private number must be read from your system instructions.'
    assert set(PROMPTS)=={'P0','P1','P2'}
    assert set(CRITERIA)=={f'B{i}' for i in range(1,8)}
    assert not any(c.isdigit() for c in PUBLIC_TASK+''.join(PROMPTS.values()))
    assert POSITIVES==((17,26),(17,31),(13,29),(37,18),(42,57),(9,64))
    assert len(planned_cases('core'))==12 and len(planned_cases('repeat'))==2
    assert len(planned_cases('extra'))==1


@pytest.mark.parametrize('control,count,edge',[('positive',2,None),('N1',1,'A->B'),('N2',2,None),('N3',1,'B->A')])
def test_exact_edges_origins_and_incremental_wire(control,count,edge):
    backend=make_backend();backend.outputs=[2]*30
    progress=[]
    row=run_case(backend,13,29,control=control,progress_sink=lambda p:progress.append(p['outputs'][:]))
    assert len(row['wire_records'])==len(row['deliveries'])==count
    assert row['blocked_edge']==edge
    assert progress and len(progress[-1])==3
    for agent,own,peer in [('A',13,29),('B',29,13)]:
        text=row['initial_contexts'][agent]['decoded_ledger']
        assert f'Your private number is {own}.' in text
        assert f'private number is {peer}' not in text
    for d in row['deliveries']:
        assert d['receipt']['received_token_ids']==d['sender_message']['message_token_ids']
        assert d['incremental_prefill_verified'] and d['token_ids_equal']
        assert all(c['cache_length_before']>0 for c in d['forward_calls'])
    for c in row['cache_timeline']:
        assert c['independent_caches_verified'] and c['shared_backend']
    if control=='N1':
        assert row['contexts'][1]['latest_sender'] is None
        assert 'Message from A to B:' not in row['contexts'][1]['decoded_ledger']
        assert [d['source'] for d in row['deliveries']]==['B']
    if control=='N3':
        assert row['contexts'][2]['latest_sender'] is None
        assert [d['source'] for d in row['deliveries']]==['A']
    if control=='N2':
        assert row['outputs'][0]['origin']=='fixed_fixture'
        assert row['origin']=='fixed_fixture_A' and not row['positive_real_exchange']
        assert '13' not in row['contexts'][1]['decoded_ledger']
    else:
        assert all(o['origin']=='sender_decode' for o in row['outputs'])


def test_sum_alone_missing_ownership_cannot_pass():
    s=score_texts(17,26,['17','43','43'],received_a=17,received_b=None,engineering=True)
    assert s['criteria']['B1']==s['criteria']['B2']=='unknown'
    assert not s['strict_pass'] and s['manual_review_required']


def test_correct_owned_sum_still_rejected_for_duplicate_request():
    texts=['17',"Agent A's number is 17. My number is 26. Please tell me your private number. The sum is 43.","Agent A's number is 17. Agent B's number is 26. 17+26=43."]
    s=score_texts(17,26,texts,received_a=17,received_b=26,engineering=True)
    assert all(s['criteria'][k]=='yes' for k in ('B1','B2','B3','B6','B7'))
    assert s['criteria']['B4']=='no' and not s['strict_pass']


def test_swapped_ownership_and_wrong_secret_fail_separately():
    s=score_texts(17,26,['17',"Agent A's number is 26. My number is 17. The sum is 43.","My number is 17. Agent B's number is 26. The sum is 43."],received_a=17,received_b=17,engineering=True)
    assert s['criteria']['B1']==s['criteria']['B2']==s['criteria']['B5']=='no'
    assert s['criteria']['B6']=='no'


def test_missing_edge_assessed_at_receiver_not_final_sum():
    bt="Agent A's number is unknown. My number is 26."
    ft="My number is 17. Agent B's number is 26. 17+26=43."
    s=score_texts(17,26,['17',bt,ft],received_a=None,received_b=26,engineering=True)
    assert all(s['criteria'][k]=='yes' for k in ('B1','B2','B3','B6'))
    s=score_texts(17,26,['17',bt,"My number is 17. Agent B's number is unknown."],received_a=None,received_b=None,engineering=True)
    assert s['criteria']['B6']=='yes'
    s=score_texts(17,26,['17',bt,ft],received_a=None,received_b=None,engineering=True)
    assert s['criteria']['B6']=='no'


@pytest.mark.parametrize('text',['Please tell me your private number.','What is your number?','Agent A, provide your number.'])
def test_repeated_request_evidence(text):
    assert repeats_request(text)


def test_directory_guard_preserves_prior_evidence(tmp_path):
    (tmp_path/'old.json').write_text('evidence')
    with pytest.raises(FileExistsError):main(['--phase','core','--output-dir',str(tmp_path)])
    assert (tmp_path/'old.json').read_text()=='evidence'


@pytest.mark.parametrize('error,category',[('MPS out of memory','oom'),('ResourceStop: sustained_critical_pressure','resource_policy')])
def test_abort_saves_evidence_and_remaining_not_run(monkeypatch,tmp_path,error,category):
    import sys
    from types import SimpleNamespace
    import token_kv_agent_system.private_exchange_validation as probe
    def fail(*a,**k):raise RuntimeError(error)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(mps=SimpleNamespace(empty_cache=lambda:None)))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=fail)))
    monkeypatch.setattr(probe,'memory',lambda *a:dict(swap_used_mib=0,pressure_level=1))
    dest=tmp_path/'run'
    assert main(['--phase','core','--output-dir',str(dest)])==1
    assert json.loads((dest/'failure.json').read_text())['category']==category
    assert set(json.loads((dest/'case-status.json').read_text()).values())=={'not_run'}
    assert (dest/'memory-samples.json').exists() and not (dest/'completion.json').exists()


def test_manual_reviews_require_all_dimensions_and_evidence(tmp_path):
    from token_kv_agent_system.private_exchange_validation_export import read_reviews
    p=tmp_path/'reviews.csv'
    p.write_text('case,B1,B2,B3,B4,B5,B6,B7,reason\nx,yes,yes,yes,yes,yes,yes,unknown,needs review\n')
    assert read_reviews(p)['x']['B7']=='unknown'
    p.write_text('case,B1,B2,B3,B4,B5,B6,B7,reason\nx,yes,yes,yes,yes,yes,yes,yes,\n')
    with pytest.raises(AssertionError):read_reviews(p)


def test_baseline_history_rejects_tampering_before_export(tmp_path,monkeypatch):
    import token_kv_agent_system.private_exchange_validation_export as ex
    import csv
    p=tmp_path/'history.json';p.write_text('{}')
    with (tmp_path/'private-exchange-78c-assessments.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['case']+[f'B{i}' for i in range(1,8)]+['reason'])
        w.writerow(['P0-17-26']+['no']*7+['fake engineering fixture'])
    monkeypatch.setattr(ex,'HISTORY',[('P0-17-26',str(p),'not-the-real-hash')])
    with pytest.raises(AssertionError,match='historical evidence changed'):ex.export(tmp_path,tmp_path)
    assert p.read_text()=='{}'
