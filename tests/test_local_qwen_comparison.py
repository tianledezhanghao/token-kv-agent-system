"""Focused 7.8A contracts; model behavior is tested only by the real experiment."""
from types import SimpleNamespace
import pytest

from token_kv_agent_system.local_qwen_comparison import (
    snapshot_path, REVISIONS, check_memory, fixed_case, production)
from token_kv_agent_system.model.cache_inspection import inspect_kv_model_config


def test_3b_config_is_not_05b_layout():
    c=inspect_kv_model_config(SimpleNamespace(num_hidden_layers=36,num_attention_heads=16,
        num_key_value_heads=2,hidden_size=2048))
    assert c.head_dim==128 and c.num_hidden_layers==36
    assert 2*c.num_hidden_layers*c.num_key_value_heads*c.head_dim*2==36864


@pytest.mark.parametrize('key', REVISIONS)
def test_explicit_checkpoint_selection(key):
    repo,revision=REVISIONS[key]
    assert snapshot_path(key).name==revision
    assert repo.replace('/','--') in str(snapshot_path(key))


@pytest.mark.parametrize('pressure,swap', [(1,2000),(2,5500),(4,7000)])
def test_single_sample_warns_without_persistence_stop(pressure,swap):
    decision=check_memory(dict(pressure_level=pressure,swap_used_mib=swap),2000)
    assert decision['stop_reason'] is None
    assert decision['warning']==(pressure>=2 or swap-2000>1024)


@pytest.mark.parametrize('structure', ['A','D'])
def test_absence_not_synthetic_and_frozen_stages(monkeypatch,structure):
    import token_kv_agent_system.autogen_style_reference as ref
    from test_autogen_style_reference import stub_generate
    b=SimpleNamespace(next_id=0)
    monkeypatch.setattr(ref,'generate',stub_generate)
    row=fixed_case(b,'absent','10 + 10 = ?',None,structure,'read',128)
    assert row['input_origin']=='no_message' and not row['deliveries']
    assert not any(s['source']=='A' for m in row['messages'] for s in m['sources'])
    row=fixed_case(b,'wrong','10 + 10 = ?','我认为答案是21。',structure,'review',128)
    assert row['deliveries'][0]['origin']=='fixed_fixture'
    assert row['max_new_tokens']==128


def test_production_shared_weights_independent_caches_incremental_ids():
    from test_incremental_kv_transport import make_backend
    b=make_backend()
    row=production(b,'10 + 10 = ?',4)
    r=row['discussion']
    assert r['status']=='completed' and len(row['wire_records'])==4
    assert len(row['contexts'])==5
    assert row['shared_backend'] and row['independent_caches_verified']
    assert row['cache_timeline'][-1]['agents']['A']['raw_kv_tensor_bytes']>0
    for turn,handoff in zip(r['turns'],r['handoffs']):
        assert handoff['executed_mode']=='token' and handoff['append_succeeded']
        assert turn['receipt']['received_token_ids']==turn['message']['message_token_ids']
    # Runtime construction initialized states before tracing; every traced call is incremental.
    assert sum(c['cache_length_before']==0 for c in row['forward_calls'])==0


@pytest.mark.parametrize('error', ['MPS out of memory','checkpoint missing'])
def test_load_failure_saved_without_model_fallback(monkeypatch,tmp_path,error):
    import sys,json
    import token_kv_agent_system.local_qwen_comparison as probe
    fake_torch=SimpleNamespace(mps=SimpleNamespace(empty_cache=lambda:None))
    def fail(*args,**kwargs):raise RuntimeError(error)
    fake_tf=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=fail))
    monkeypatch.setitem(sys.modules,'torch',fake_torch)
    monkeypatch.setitem(sys.modules,'transformers',fake_tf)
    monkeypatch.setattr(probe,'memory',lambda *a:dict(swap_used_mib=0,pressure_level=1))
    dest=tmp_path/'run'
    assert probe.main(['--phase','boot','--model','3b','--output-dir',str(dest)])==1
    r=json.loads((dest/'failure.json').read_text())
    assert r['error']==error and not (dest/'completion.json').exists()
    assert not (dest/'boot-1.json').exists()


def test_existing_output_directory_rejected_before_load(tmp_path):
    from token_kv_agent_system.local_qwen_comparison import main
    (tmp_path/'old.json').write_text('evidence')
    with pytest.raises(FileExistsError):main(['--phase','boot','--output-dir',str(tmp_path)])
    assert (tmp_path/'old.json').read_text()=='evidence'
