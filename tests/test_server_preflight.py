from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from token_kv_agent_system import server_preflight
from token_kv_agent_system.model import IndependentBackends, PrefillResult, CachedContinuationResult
from token_kv_agent_system.model.kv_adapter import restore_huggingface_kv_cache
from token_kv_agent_system.transport import InMemoryByteTransport
from token_kv_agent_system.agent import LLMAgent
from token_kv_agent_system.kv_handoff import perform_kv_handoff, CanonicalHandoffSpan
from token_kv_agent_system.network import SimulatedNetwork
from test_independent_backends import FakeIndependentBackend


def test_environment_inventory_is_read_only_and_distinguishes_visible_ids(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1,2,0")
    class Cache:
        def __init__(self, ddp_cache_data=None, config=None):
            pass
        def update(self, key, value, layer_index):
            pass
    torch = SimpleNamespace(__version__="fixture-torch", version=SimpleNamespace(cuda="fixture-cuda"),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 4,
            get_device_properties=lambda i: SimpleNamespace(name=f"fixture GPU {i}"),
            mem_get_info=lambda i: (100 - i, 200), memory_allocated=lambda i: i))
    result = server_preflight.inspect_environment(torch, SimpleNamespace(__version__="fixture-hf", DynamicCache=Cache))
    assert result["CUDA_VISIBLE_DEVICES"] == "3,1,2,0"
    assert result["visible_gpu_count"] == 4
    assert [g["visible_device"] for g in result["gpus"]] == [f"cuda:{i}" for i in range(4)]
    assert result["gpus"][1]["allocated_memory_bytes"] == 1


class FakeKVBackend(FakeIndependentBackend):
    """Actual tiny CPU cache tensors; no pretrained or randomly initialized LM."""
    def __init__(self):
        super().__init__()
        self.torch = pytest.importorskip("torch")
        self.transformers = pytest.importorskip("transformers")
        self.restores = []
        self.continued_caches = []
        self.hf_config = self.transformers.LlamaConfig(
            vocab_size=16, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        )

    def _append(self, cache, count):
        for index in range(2):
            key = self.torch.arange(count * 16, dtype=self.torch.float32).reshape(1, 2, count, 8)
            cache.update(key, key + 1, index)

    def prefill(self, prompt=None, *, input_token_ids=None):
        ids = tuple(input_token_ids) if input_token_ids is not None else self.encode(prompt)
        cache = self.transformers.DynamicCache(config=self.hf_config)
        self._append(cache, len(ids))
        self.prefill_inputs.append(ids)
        self.prefill_caches.append(cache)
        return PrefillResult(ids, 6, cache, 1.0, len(ids), (1, len(ids), 16), (16,), self.torch.zeros(16))

    def continue_from_cache(self, input_token_ids, *, previous_cache, logical_cached_token_count):
        ids = tuple(input_token_ids)
        assert previous_cache.get_seq_length() == logical_cached_token_count
        self.continued_caches.append(previous_cache)
        self._append(previous_cache, len(ids))
        return CachedContinuationResult(ids, 6, previous_cache, 1.0,
                                        logical_cached_token_count + len(ids), self.torch.zeros(16))

    def decode_step(self, input_token_id, *, previous_cache, logical_cached_token_count):
        from token_kv_agent_system.model import DecodeStepResult
        result = self.continue_from_cache((input_token_id,), previous_cache=previous_cache,
                                          logical_cached_token_count=logical_cached_token_count)
        return DecodeStepResult(input_token_id, 6, result.cache, 1.0, result.logical_cached_token_count)

    def restore_kv_cache(self, payload):
        self.restores.append(payload)
        return restore_huggingface_kv_cache(payload, model_id=self.model_name,
            model_config=self.kv_model_config, model_dtype=self.metadata.dtype,
            target_device=self.device, transformers_config=self.hf_config)

    def manual_generate(self, prompt, config=None, *, warmup_runs=0):
        return SimpleNamespace(input_token_ids=self.encode(prompt), output_token_ids=self.outputs)


def test_cpu_cache_single_smoke_without_loading_any_model():
    result = server_preflight.single_model_smoke(FakeKVBackend(), "prompt", 4)
    assert result["manual_hf_ids_exact"] and result["all_layers_bitwise_exact"]
    assert result["raw_kv_bytes"] == result["body_bytes"] == 256
    assert [s["runtime_cached_token_count"] for s in result["cache_snapshots"]] == [1, 2, 3, 4]
    assert result["restored_one_token_id_exact"]
    assert result["remaining_server_acceptance"]


def test_dual_canonical_kv_uses_source_cache_and_target_restore_via_bytes(monkeypatch):
    import token_kv_agent_system.kv_handoff as handoff
    from token_kv_agent_system.kv_protocol import deserialize_kv_packet
    a, b = FakeKVBackend(), FakeKVBackend()
    pair = IndependentBackends(a, b)
    serialized_caches, target_devices = [], []
    original_serializer = handoff.serialize_huggingface_kv_cache
    def serializer(cache, **kwargs):
        serialized_caches.append(cache)
        return original_serializer(cache, **kwargs)
    monkeypatch.setattr(handoff, "serialize_huggingface_kv_cache", serializer)
    original_restore = restore_huggingface_kv_cache
    def restore(payload, **kwargs):
        target_devices.append(kwargs["target_device"])
        return original_restore(payload, **kwargs)
    monkeypatch.setitem(globals(), "restore_huggingface_kv_cache", restore)
    class ByteSpy(InMemoryByteTransport):
        def transfer(self, data, **kwargs):
            assert type(data) is bytes
            assert deserialize_kv_packet(data).payload.source_device == "cpu"
            return super().transfer(memoryview(data).tobytes(), **kwargs)
    transport = ByteSpy()
    for source, target in ((a, b), (b, a)):
        target_prefills = len(target.prefill_inputs)
        result = perform_kv_handoff(source=LLMAgent("source", source), target=LLMAgent("target", target),
            span=CanonicalHandoffSpan((1, 2, 3)), receiver_suffix_token_ids=(4,), transport=transport,
            networks=(SimulatedNetwork(1000),), receiver_max_new_tokens=3)
        assert serialized_caches[-1] is source.prefill_caches[-1]
        assert len(target.prefill_inputs) == target_prefills
        assert target.continued_caches[-1] is not serialized_caches[-1]
        restored_layers = target.continued_caches[-1].layers
        for source_layer, target_layer in zip(serialized_caches[-1].layers, restored_layers):
            assert source_layer.keys.untyped_storage().data_ptr() != target_layer.keys.untyped_storage().data_ptr()
        assert result.wire_identity_verified and not result.receiver_prefix_prefill_executed
        assert result.wire_tensor_body_bytes == result.raw_kv_tensor_bytes
    assert target_devices == ["cpu", "cpu"]
    assert len(transport.records) == 2
    assert pair.to_dict()["same_device"]


def test_dual_cli_smoke_uses_four_byte_transfers():
    result = server_preflight.dual_handoff_smoke(IndependentBackends(FakeKVBackend(), FakeKVBackend()), "prompt", 2)
    assert len(result["transfer_records"]) == 4
    assert len(result["directions"]) == 2
    assert result["transport"] == "InMemoryByteTransport"


def test_preflight_default_never_loads_a_checkpoint(monkeypatch, capsys):
    import token_kv_agent_system.model as model
    monkeypatch.setattr(server_preflight, "inspect_environment", lambda: {"cuda_available": False})
    monkeypatch.setattr(model, "HuggingFaceModelBackend", lambda *a, **k: pytest.fail("unexpected model load"))
    assert server_preflight.main(["--agent-a-device", "cpu", "--agent-b-device", "cpu"]) == 0
    assert '"model_loaded": false' in capsys.readouterr().out


def test_preflight_requires_actual_checkpoint_for_config_stage(monkeypatch):
    monkeypatch.setattr(server_preflight, "inspect_environment", lambda: {})
    assert server_preflight.main(["--stage", "config"]) == 1


def test_config_preflight_reads_supplied_config_and_tokenizer_without_model(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from test_independent_backends import Tokenizer, config
    calls = []
    def read_config(name, **kwargs):
        calls.append((name, kwargs))
        return config(architectures=["ProvidedArchitecture"], dtype="float32", head_dim=12)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", read_config)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    result = server_preflight.inspect_checkpoint("actual-supplied-id")
    assert result["declared_architectures"] == ["ProvidedArchitecture"]
    assert result["structure"]["head_dim"] == 12
    assert result["attention_kind"] == "GQA"
    assert calls == [("actual-supplied-id", {"local_files_only": True})]


def test_dual_cli_never_loads_second_backend_if_single_gate_fails(monkeypatch):
    import token_kv_agent_system.model as model
    calls = []
    monkeypatch.setattr(server_preflight, "inspect_environment", lambda: {})
    monkeypatch.setattr(server_preflight, "inspect_checkpoint", lambda *a, **k: {})
    def factory(*args, **kwargs):
        calls.append(kwargs["device"])
        return FakeIndependentBackend()
    monkeypatch.setattr(model, "HuggingFaceModelBackend", factory)
    def fail(*args):
        raise RuntimeError("single gate failed")
    monkeypatch.setattr(server_preflight, "single_model_smoke", fail)
    assert server_preflight.main(["--stage", "dual", "--model", "fixture",
                                   "--agent-a-device", "cpu", "--agent-b-device", "cpu"]) == 1
    assert calls == ["cpu"]


def test_adapter_preserves_source_and_target_cuda_indices_without_gpu_execution(monkeypatch):
    import token_kv_agent_system.model.kv_adapter as adapter
    from token_kv_agent_system.kv_protocol import deserialize_kv_packet
    backend = FakeKVBackend()
    torch = backend.torch
    class LabelledTensor:
        def __init__(self, tensor, device):
            self.tensor, self.device = tensor, device
        def __getattr__(self, name):
            return getattr(self.tensor, name)
    source = backend.prefill(input_token_ids=(1, 2))
    cache = SimpleNamespace(get_seq_length=lambda: 2, layers=[SimpleNamespace(
        keys=LabelledTensor(layer.keys, "cuda:0"), values=LabelledTensor(layer.values, "cuda:0"))
        for layer in source.cache.layers])
    syncs = []
    monkeypatch.setattr(adapter, "synchronize_device", lambda torch_module, device: syncs.append(device))
    encoded = adapter.serialize_huggingface_kv_cache(cache, logical_cached_token_count=2,
        model_id=backend.model_name, model_config=backend.kv_model_config)
    assert syncs == ["cuda:0", "cuda:0"]
    payload = deserialize_kv_packet(encoded.packet).payload
    uploads = []
    def fake_upload(tensor, device):
        uploads.append(device)
        return LabelledTensor(tensor, device)
    monkeypatch.setattr(torch.Tensor, "to", fake_upload)
    class PublicCache:
        def __init__(self, ddp_cache_data, config):
            self.layers = [SimpleNamespace(keys=k, values=v) for k, v in ddp_cache_data]
        def get_seq_length(self):
            return self.layers[0].keys.shape[2]
    restored = adapter.restore_huggingface_kv_cache(payload, model_id=backend.model_name,
        model_config=backend.kv_model_config, model_dtype=backend.metadata.dtype,
        target_device="cuda:1", transformers_config=backend.hf_config, dynamic_cache_class=PublicCache)
    assert uploads == ["cuda:1"] * 4
    assert syncs[2:] == ["cuda:1"] * 4
    assert restored.target_device == "cuda:1"


def test_unknown_cache_constructor_fails_before_allocating_receiver_tensors(monkeypatch):
    import token_kv_agent_system.model.kv_adapter as adapter
    from token_kv_agent_system.kv_protocol import deserialize_kv_packet
    backend = FakeKVBackend()
    source = backend.prefill(input_token_ids=(1,))
    encoded = adapter.serialize_huggingface_kv_cache(source.cache, logical_cached_token_count=1,
        model_id=backend.model_name, model_config=backend.kv_model_config)
    monkeypatch.setattr(adapter, "_neutral_to_cpu_tensor", lambda *a: pytest.fail("premature allocation"))
    with pytest.raises(adapter.HuggingFaceKVAdapterError, match="public"):
        adapter.restore_huggingface_kv_cache(deserialize_kv_packet(encoded.packet).payload,
            model_id=backend.model_name, model_config=backend.kv_model_config, model_dtype=backend.metadata.dtype,
            target_device="cpu", transformers_config=backend.hf_config, dynamic_cache_class=object)
