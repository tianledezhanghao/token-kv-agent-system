"""No checkpoint downloads/models: fake backends, config objects and tiny CPU tensors."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from token_kv_agent_system.agent import LLMAgent
from token_kv_agent_system.model.backend_pair import (
    BackendIdentity, IndependentBackendConfig, IndependentBackends,
    inspect_backend_identity, validate_backend_compatibility,
)
from token_kv_agent_system.model.cache_inspection import inspect_kv_model_config
from token_kv_agent_system.model.config import LLMConfig
from token_kv_agent_system.model.huggingface import resolve_device, synchronize_device
from token_kv_agent_system.model.kv_adapter import _validate_restored_cache, HuggingFaceKVAdapterError
from token_kv_agent_system.token_handoff import TokenHandoffSession, TokenHandoffError
from token_kv_agent_system.transport import InMemoryByteTransport
from test_token_handoff import FakeTokenBackend


def config(**overrides):
    values = dict(model_type="llama", num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, hidden_size=32, vocab_size=16)
    return SimpleNamespace(**(values | overrides))


class Tokenizer:
    vocab_size = 16
    all_special_ids = []
    bos_token_id = eos_token_id = pad_token_id = unk_token_id = None

    def __len__(self):
        return 16

    def get_vocab(self):
        return {f"token-{i}": i for i in range(16)}


class StorageTensor:
    """Device-labelled test storage; never dispatches a CUDA/MPS operation."""
    def __init__(self, device):
        self.device = device

    def numel(self):
        return 1

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: id(self))


class FakeIndependentBackend(FakeTokenBackend):
    def __init__(self, device="cpu"):
        super().__init__()
        self._metadata = replace(self._metadata, model_name="fixture/same-checkpoint", device=device, dtype="torch.float32")
        self._tokenizer = Tokenizer()
        self.storage = StorageTensor(device)
        self._model = SimpleNamespace(config=config(), parameters=lambda: iter([self.storage]), buffers=lambda: iter([]))
        self.kv_model_config = inspect_kv_model_config(self._model.config)
        self.compatibility_metadata = inspect_backend_identity(self)


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda", "cuda:0", "cuda:3"])
def test_explicit_device_configuration_is_hardware_independent(device):
    assert LLMConfig(device=device).device == device
    assert IndependentBackendConfig("provided-checkpoint", device, device).agent_b_device == device


@pytest.mark.parametrize("device", ["cuda:-1", "cuda:1.0", "cuda:x", "cuda:01", "mps:3", "gpu", ""])
def test_invalid_device_syntax(device):
    with pytest.raises(ValueError):
        IndependentBackendConfig("provided-checkpoint", device, "cpu")


def test_explicit_placement_required():
    with pytest.raises(ValueError, match="explicit"):
        IndependentBackendConfig("provided-checkpoint", "auto", "cpu")


def test_cuda_device_preflight_and_exact_sync_are_centralized():
    queries, syncs = [], []
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
        get_device_properties=lambda index: queries.append(index), synchronize=syncs.append),
        mps=SimpleNamespace(synchronize=lambda: syncs.append("mps")))
    assert resolve_device("cuda:1", torch) == "cuda:1"
    assert queries == [1]
    with pytest.raises(RuntimeError, match="not visible"):
        resolve_device("cuda:2", torch)
    for device in ("cuda:1", "cuda:0", "mps:0", "cpu"):
        synchronize_device(torch, device)
    assert syncs == ["cuda:1", "cuda:0", "mps"]


def test_factory_checks_both_devices_before_allocating_any_model():
    calls = []
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
        get_device_properties=lambda index: None))
    with pytest.raises(RuntimeError, match="not visible"):
        IndependentBackends.load(IndependentBackendConfig("checkpoint"), torch_module=torch,
                                  backend_factory=lambda *args, **kwargs: calls.append(kwargs))
    assert calls == []


@pytest.mark.parametrize("devices", [("cpu", "cpu"), ("mps", "mps"), ("cuda:0", "cuda:1")])
def test_independent_factory_and_same_device_metadata(devices):
    calls = []
    def factory(name, *, device, torch_module):
        calls.append((name, device))
        return FakeIndependentBackend(device)
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 4,
        get_device_properties=lambda index: None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
    pair = IndependentBackends.load(IndependentBackendConfig("provided-checkpoint", *devices),
                                     torch_module=torch, backend_factory=factory)
    assert calls == [("provided-checkpoint", d) for d in devices]
    assert pair.agent_a_backend is not pair.agent_b_backend
    assert pair.to_dict()["same_device"] == (devices[0] == devices[1])
    assert pair.to_dict()["separate_cuda_devices"] == (devices == ("cuda:0", "cuda:1"))


@pytest.mark.parametrize("alias", ["backend", "model", "tokenizer", "storage"])
def test_rejects_false_independence(alias):
    a, b = FakeIndependentBackend(), FakeIndependentBackend()
    if alias == "backend":
        b = a
    elif alias == "model":
        b._model = a._model
    elif alias == "tokenizer":
        b._tokenizer = a._tokenizer
    else:
        b.storage = a.storage
    with pytest.raises(ValueError, match="independent"):
        IndependentBackends(a, b)


@pytest.mark.parametrize("field,value", [
    ("model_id", "different-checkpoint"), ("architecture", "DifferentClass"),
    ("dtype", "torch.bfloat16"), ("tokenizer_class", "OtherTokenizer"),
    ("model_vocab_size", 32), ("tokenizer_vocab_sha256", "different-mapping"),
    ("special_token_ids", (("eos_token_id", 2),)),
])
def test_structural_and_tokenizer_mismatch_fails_before_bytes(field, value):
    a, b = FakeIndependentBackend(), FakeIndependentBackend()
    b.compatibility_metadata = replace(b.compatibility_metadata, **{field: value})
    transport = InMemoryByteTransport()
    session = TokenHandoffSession([LLMAgent("a", a), LLMAgent("b", b)], transport)
    with pytest.raises(TokenHandoffError, match=field):
        session.handoff_token_ids("a", "b", (1, 2))
    assert transport.records == () and b.prefill_inputs == []


def test_llama_like_config_extraction_and_no_qwen_constants():
    implicit = inspect_kv_model_config(config())
    explicit = inspect_kv_model_config(config(head_dim=12, num_hidden_layers=3, num_key_value_heads=1))
    assert implicit.head_dim == 8 and implicit.head_dim_source.startswith("derived")
    assert explicit.head_dim == 12 and explicit.head_dim_source == "config.head_dim"
    assert (explicit.num_hidden_layers, explicit.num_key_value_heads) == (3, 1)
    a, b = FakeIndependentBackend(), FakeIndependentBackend()
    b.compatibility_metadata = replace(b.compatibility_metadata, structure=explicit)
    with pytest.raises(ValueError, match="structure"):
        validate_backend_compatibility(a, b)
    del a._model.config.num_key_value_heads
    with pytest.raises(ValueError):
        inspect_kv_model_config(a._model.config)  # Do not guess unknown architectures.


def test_tokenizer_id_mapping_and_model_domain_are_checked():
    backend = FakeIndependentBackend()
    first = inspect_backend_identity(backend)
    backend._tokenizer.get_vocab = lambda: {"token-0": 1, "token-1": 0}
    assert inspect_backend_identity(backend).tokenizer_vocab_sha256 != first.tokenizer_vocab_sha256
    backend._tokenizer.get_vocab = lambda: {"bad": 999}
    with pytest.raises(ValueError, match="outside"):
        inspect_backend_identity(backend)


def test_token_handoff_uses_each_receiver_backend_and_independent_caches():
    a, b = FakeIndependentBackend("cuda:0"), FakeIndependentBackend("cuda:1")
    pair = IndependentBackends(a, b)
    transport = InMemoryByteTransport()
    session = TokenHandoffSession([LLMAgent("a", a), LLMAgent("b", b)], transport)
    forward = session.generate_and_handoff("a", "b", "prompt")
    assert a.prefill_inputs == [] and b.prefill_inputs == [(4, 5)]
    reverse = session.handoff_token_ids("b", "a", forward.receiver_output_token_ids)
    assert a.prefill_inputs == [forward.receiver_output_token_ids]
    assert a.prefill_caches[0] is not b.prefill_caches[0]
    assert forward.token_identity_preserved and reverse.token_identity_preserved
    assert a.encode_calls == b.encode_calls == 0
    assert [(r.source, r.target) for r in transport.records] == [("a", "b"), ("b", "a")]
    assert pair.to_dict()["agent_b_device"] == "cuda:1"


def test_restore_rejects_wrong_cuda_index_even_when_device_type_matches():
    from token_kv_agent_system.kv_protocol import KVModelCompatibility, NeutralKVPayload, NeutralKVLayer, NeutralKVTensor
    tensor = NeutralKVTensor("float32", (1, 2, 1, 8), bytes(64))
    payload = NeutralKVPayload(KVModelCompatibility("fixture", 1, 2, 8), 1, "cuda:0",
                               (NeutralKVLayer(0, tensor, tensor),))
    runtime = SimpleNamespace(shape=tensor.shape, dtype="torch.float32", device="cuda:0")
    cache = SimpleNamespace(layers=[SimpleNamespace(keys=runtime, values=runtime)], get_seq_length=lambda: 1)
    with pytest.raises(HuggingFaceKVAdapterError, match="device"):
        _validate_restored_cache(cache, payload=payload, target_device="cuda:1")
