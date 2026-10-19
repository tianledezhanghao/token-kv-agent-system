"""Packed adapter bits, bounded allocation, ownership and transaction oracles."""
from dataclasses import replace
import gc
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
from token_kv_agent_system.model import kv_adapter as adapter
from token_kv_agent_system.kv_protocol import deserialize_kv_packet
from token_kv_agent_system.model.cache_inspection import KVModelConfig
from token_kv_agent_system.model.state_cache import require_disjoint_caches


def config():
    return KVModelConfig(2, 4, 2, 16, 4, 'config.head_dim')


def cache(n, dtype, strided):
    # Include every BF16 pattern (NaNs, signed zero, subnormal) without arithmetic.
    pattern = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    if dtype != torch.bfloat16:
        pattern = pattern.to(dtype)
    tensors = []
    for i in range(4):
        t = pattern.roll(i * 13).repeat((4*n+65535)//65536)[:4*n].reshape(1, 2, n, 2)
        t = torch.cat((t, t), -1)
        if strided:
            t = torch.stack((t, t), -1)[..., 0]
            assert not t.is_contiguous()
        tensors.append(t)
    return SimpleNamespace(layers=[SimpleNamespace(keys=tensors[i], values=tensors[i+1])
                                  for i in (0,2)], get_seq_length=lambda:n)


def encode(c, path):
    return adapter.serialize_huggingface_kv_cache(c,logical_cached_token_count=c.get_seq_length(),
        model_id='test',model_config=config(),torch_module=torch,transfer_path=path)


def restore(p, path, cache_class=None):
    return adapter.restore_huggingface_kv_cache(p,model_id='test',model_config=config(),
        model_dtype=p.layers[0].key.dtype,target_device='cpu',torch_module=torch,
        transformers_config=transformers.Qwen2Config(num_hidden_layers=2),transfer_path=path,
        dynamic_cache_class=cache_class).cache


def raw(t):
    return t.detach().contiguous().view(torch.uint8).numpy().tobytes()


@pytest.mark.parametrize('n',[1,2,32,512,16384])
@pytest.mark.parametrize('strided',[False,True])
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16,torch.float32])
def test_byte_identical_packet_restore_and_nonoverlapping_owned_views(n,strided,dtype):
    c=cache(n,dtype,strided)
    old=encode(c,'legacy');new=encode(c,'packed')
    assert old.packet==new.packet
    p=deserialize_kv_packet(new.packet).payload
    captured=[]
    def observed(**kw):
        captured.extend(t for pair in kw['ddp_cache_data'] for t in pair)
        return transformers.DynamicCache(**kw)
    r=restore(p,'packed',observed);other=restore(p,'packed')
    require_disjoint_caches(c,r);require_disjoint_caches(r,other)
    tensors=[t for l in r.layers for t in (l.keys,l.values)]
    assert len({t.untyped_storage().data_ptr() for t in captured})==1
    size=captured[0].numel()
    assert [t.storage_offset() for t in captured]==[i*size for i in range(4)]
    # TF5.17 public constructor cats from empty storage, copying each input.
    assert len({t.untyped_storage().data_ptr() for t in tensors})==4
    assert not ({t.untyped_storage().data_ptr() for t in tensors}
                & {t.untyped_storage().data_ptr() for t in captured})
    del captured
    for l,src in zip(r.layers,c.layers):
        assert l.keys.shape==src.keys.shape and l.keys.dtype==src.keys.dtype
        assert raw(l.keys)==raw(src.keys) and raw(l.values)==raw(src.values)
    originals=[raw(t) for t in tensors]
    expected_other=raw(other.layers[0].keys)
    del p,new,old;gc.collect()
    tensors[0].zero_()
    assert [raw(t) for t in tensors[1:]]==originals[1:]
    assert raw(other.layers[0].keys)==expected_other
    assert raw(c.layers[0].keys)==expected_other


def test_invalid_inputs_and_buffer_limit_rejected_before_stack_or_upload(monkeypatch):
    c=cache(2,torch.bfloat16,False)
    p=deserialize_kv_packet(encode(c,'legacy').packet).payload
    monkeypatch.setattr(adapter,'MAX_PACKED_TRANSFER_BYTES',1)
    monkeypatch.setattr(torch,'stack',lambda *a,**kw:pytest.fail('allocation before guard'))
    with pytest.raises(ValueError,match='capacity'):encode(c,'packed')
    with pytest.raises(ValueError,match='capacity'):restore(p,'packed')
    with pytest.raises(ValueError,match='identifier'):restore(replace(p,model=replace(p.model,model_id='wrong')),'packed')
    with pytest.raises(ValueError,match='greater than zero'):
        adapter.extract_huggingface_kv_payload(c,logical_cached_token_count=0,model_id='test',model_config=config(),transfer_path='packed')
    with pytest.raises(ValueError,match='transfer_path'):encode(c,'unknown')


def test_mixed_dtype_is_not_silently_promoted():
    c=cache(2,torch.bfloat16,False)
    c.layers[1].keys=c.layers[1].keys.float();c.layers[1].values=c.layers[1].values.float()
    encode(c,'legacy')
    with pytest.raises(ValueError,match='uniform dtype'):encode(c,'packed')


@pytest.mark.parametrize('candidate',['kv:v0','kv:v1','kv:v2','kv:v3'])
@pytest.mark.parametrize('eos',[True,False])
def test_handoff_exact_cache_continue_decode_and_lifecycle(monkeypatch,candidate,eos):
    from test_incremental_kv_receive import build
    from token_kv_agent_system.performance_characterization import measured_handoff,cache_sha256
    from token_kv_agent_system.incremental_kv_protocol import inspect_model_contract
    b,r=build(raw=(100,101,102,103,2) if eos else (100,101,102,103,104))
    s=r.session;s.generate_message('A','B',max_new_tokens=5)
    saved=dict(s._states);s._kv_contract=inspect_model_contract(b)
    before={n:cache_sha256(x.cache) for n,x in saved.items()}
    out=[]
    for path in ('legacy','packed'):
        monkeypatch.setattr(adapter,'DEFAULT_TRANSFER_PATH',path)
        sample=measured_handoff(s,saved,candidate)
        out.append((sample['wire_bytes'],cache_sha256(s.state('B').cache),s.state('B').token_ledger))
        del sample;gc.collect()
        # Real tiny model forward uses restored/concatenated cache after buffers die.
        s.generate_message('B','A',max_new_tokens=2)
        s.state('B').validate()
        require_disjoint_caches(s.state('A').cache,s.state('B').cache)
    assert out[0]==out[1]
    assert before=={n:cache_sha256(x.cache) for n,x in saved.items()}


@pytest.mark.parametrize('failure',['corrupt','contract','append','closing','rope','sender_close','ledger'])
def test_existing_transaction_failures_with_packed_adapter(monkeypatch,failure):
    from test_incremental_kv_receive import test_failed_kv_transaction_does_not_commit_sequence_cache_or_fallback
    monkeypatch.setattr(adapter,'DEFAULT_TRANSFER_PATH','packed')
    test_failed_kv_transaction_does_not_commit_sequence_cache_or_fallback(monkeypatch,failure)


def test_repeated_bidirectional_handoffs_with_packed_buffers(monkeypatch):
    from test_incremental_kv_receive import build
    from token_kv_agent_system.communication import KVStrategy
    monkeypatch.setattr(adapter,'DEFAULT_TRANSFER_PATH','packed')
    b,r=build(raw=(100,101,102,103,2));s=r.session
    retained=[]
    for source,target in [('A','B'),('B','A')]*3:
        s.generate_message(source,target,max_new_tokens=5)
        s.handoff_kv(source,target,strategy=KVStrategy.RAW_DELTA)
        for n in ('A','B'):s.state(n).validate()
        require_disjoint_caches(s.state('A').cache,s.state('B').cache)
        retained.append(s.state(target).cache)
        gc.collect()
    for earlier,later in zip(retained,retained[1:]):require_disjoint_caches(earlier,later)


def test_auto_domain_and_no_exception_fallback(monkeypatch):
    assert adapter._transfer_path('auto','mps',4096,dtype='bfloat16')=='packed'
    assert adapter._transfer_path('auto','mps:0',4096,dtype='bfloat16')=='packed'
    assert adapter._transfer_path('auto','mps',4096,dtype='float16')=='legacy'
    assert adapter._transfer_path('auto','mps',6*1024*1024+1,dtype='bfloat16')=='legacy'
    assert adapter._transfer_path('auto','cpu',4096)=='legacy'
    assert adapter._transfer_path('auto','cuda:1',4096)=='legacy'
    assert adapter._transfer_path('auto','mps',2**30)=='legacy'
    assert adapter._transfer_path('auto','mps',4096,False)=='legacy'
    def fail(*a,**kw):raise RuntimeError('packing failed')
    monkeypatch.setattr(torch,'stack',fail)
    with pytest.raises(RuntimeError,match='packing failed'):encode(cache(2,torch.bfloat16,False),'packed')


def test_signed_statistics_and_input_index_tamper(tmp_path):
    import json
    from token_kv_agent_system.kv_transfer_validation import signed_statistics,verify_index
    from token_kv_agent_system.local_qwen_comparison import file_hash
    assert signed_statistics([-4,0,1])['median']==0
    with pytest.raises(ValueError):signed_statistics([float('nan')])
    p=tmp_path/'cache';p.write_bytes(b'actual')
    (tmp_path/'sha256-index.json').write_text(json.dumps([dict(path='cache',bytes=6,sha256=file_hash(p))]))
    verify_index(tmp_path);p.write_bytes(b'broken')
    with pytest.raises(ValueError,match='SHA'):verify_index(tmp_path)


@pytest.mark.parametrize('boundary',['export','upload'])
def test_new_transfer_failure_rolls_back_without_legacy_retry(monkeypatch,boundary):
    from test_incremental_kv_receive import build
    from token_kv_agent_system.performance_characterization import cache_sha256
    from token_kv_agent_system.communication import KVStrategy
    b,r=build(raw=(100,101,2));s=r.session
    s.generate_message('A','B',max_new_tokens=3)
    saved=dict(s._states);before={n:cache_sha256(x.cache) for n,x in saved.items()}
    monkeypatch.setattr(adapter,'DEFAULT_TRANSFER_PATH','packed')
    def forbidden(*a,**kw):pytest.fail('legacy retry after packed failure')
    monkeypatch.setattr(adapter,'_tensor_to_neutral',forbidden)
    monkeypatch.setattr(adapter,'_neutral_to_cpu_tensor',forbidden)
    if boundary=='export':
        def fail(*a,**kw):raise RuntimeError('injected transfer failure')
        monkeypatch.setattr(torch,'stack',fail)
    else:
        original=torch.Tensor.to
        def fail(t,*a,**kw):
            result=original(t,*a,**kw)
            if t.ndim==1 and t.dtype==torch.float32:
                raise RuntimeError('injected transfer failure after to')
            return result
        monkeypatch.setattr(torch.Tensor,'to',fail)
    with pytest.raises(RuntimeError,match='injected transfer failure'):
        s.handoff_kv('A','B',strategy=KVStrategy.RAW_DELTA)
    assert all(s.state(n) is saved[n] for n in saved)
    assert before=={n:cache_sha256(x.cache) for n,x in saved.items()}


def test_original_empty_and_capacity_guards_under_packed_path(monkeypatch):
    from test_incremental_kv_transport import (
        test_empty_content_and_forged_ledger_fail_explicitly,
        test_sender_size_bound_and_bad_cache_fail_before_supplement)
    monkeypatch.setattr(adapter,'DEFAULT_TRANSFER_PATH','packed')
    test_empty_content_and_forged_ledger_fail_explicitly()
    test_sender_size_bound_and_bad_cache_fail_before_supplement()


def test_export_audit_rejects_measured_branch_drift():
    import importlib.util
    from pathlib import Path
    p=Path(__file__).resolve().parents[1]/'export_kv_transfer_validation.py'
    spec=importlib.util.spec_from_file_location('export83b',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=Path(adapter.__file__).read_text()
    assert module.measured_adapter_unchanged(source,source)==''
    with pytest.raises(ValueError,match='execution changed'):
        module.measured_adapter_unchanged(source,source.replace('packed.cpu()', 'packed.clone().cpu()'))
