import pickle
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from token_kv_agent_system.kv_protocol import deserialize_kv_packet
from token_kv_agent_system.model.cache_inspection import KVModelConfig
from token_kv_agent_system.model.kv_adapter import (
    HuggingFaceKVAdapterError,
    serialize_huggingface_kv_cache,
)


def _config(layers=2):
    return KVModelConfig(layers, 4, 2, 16, 4, "config.head_dim")


def _bf16_tensor(seed):
    patterns = torch.tensor(
        [0x0000, 0x8000, 0x7FC1, 0x3F80, 0xBF80, 0x7F80], dtype=torch.uint16
    )
    repeated = patterns.repeat(4)[:24]
    return repeated.view(torch.bfloat16).reshape(1, 2, 3, 4).clone().add_(seed)


def _cache():
    return SimpleNamespace(
        get_seq_length=lambda: 3,
        layers=[
            SimpleNamespace(keys=_bf16_tensor(0), values=_bf16_tensor(1)),
            SimpleNamespace(keys=_bf16_tensor(2), values=_bf16_tensor(3)),
        ]
    )


def _raw_bytes(tensor):
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


def test_adapter_and_wire_round_trip_preserve_all_bfloat16_bits():
    cache = _cache()
    before = [
        (_raw_bytes(layer.keys), _raw_bytes(layer.values)) for layer in cache.layers
    ]

    encoded = serialize_huggingface_kv_cache(
        cache,
        logical_cached_token_count=3,
        model_id="Qwen/test",
        model_config=_config(),
        torch_module=torch,
    )
    restored = deserialize_kv_packet(encoded.packet).payload

    assert encoded.raw_kv_tensor_bytes == 2 * 2 * 24 * 2
    assert encoded.wire_tensor_body_bytes == encoded.raw_kv_tensor_bytes
    assert [
        (layer.key.data, layer.value.data) for layer in restored.layers
    ] == before
    assert all(layer.key.dtype == "bfloat16" for layer in restored.layers)
    assert all(layer.key.shape == (1, 2, 3, 4) for layer in restored.layers)
    assert [
        (_raw_bytes(layer.keys), _raw_bytes(layer.values)) for layer in cache.layers
    ] == before


def test_restored_payload_does_not_reference_sender_cache():
    cache = _cache()
    encoded = serialize_huggingface_kv_cache(
        cache,
        logical_cached_token_count=3,
        model_id="Qwen/test",
        model_config=_config(),
        torch_module=torch,
    )
    restored = deserialize_kv_packet(encoded.packet).payload
    restored_bytes = restored.layers[0].key.data

    cache.layers[0].keys.zero_()

    assert restored.layers[0].key.data == restored_bytes
    assert restored.layers[0].key.data != _raw_bytes(cache.layers[0].keys)


def test_sender_cache_growth_does_not_change_restored_payload():
    cache = _cache()
    encoded = serialize_huggingface_kv_cache(
        cache,
        logical_cached_token_count=3,
        model_id="Qwen/test",
        model_config=_config(),
        torch_module=torch,
    )
    restored = deserialize_kv_packet(encoded.packet).payload
    original = restored.layers[1].value.data

    cache.layers[1].values = torch.cat(
        (cache.layers[1].values, torch.ones((1, 2, 1, 4), dtype=torch.bfloat16)),
        dim=2,
    )

    assert restored.layers[1].value.data == original
    assert restored.layers[1].value.shape == (1, 2, 3, 4)


def test_serializer_never_uses_pickle_or_torch_save(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("object serialization is forbidden")

    monkeypatch.setattr(pickle, "dumps", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)

    encoded = serialize_huggingface_kv_cache(
        _cache(),
        logical_cached_token_count=3,
        model_id="Qwen/test",
        model_config=_config(),
        torch_module=torch,
    )

    assert encoded.packet[:4] == b"TKVK"


def test_adapter_rejects_sequence_length_mismatch():
    cache = _cache()
    cache.get_seq_length = lambda: 4
    with pytest.raises(HuggingFaceKVAdapterError, match="sequence length"):
        serialize_huggingface_kv_cache(
            cache,
            logical_cached_token_count=4,
            model_id="Qwen/test",
            model_config=_config(),
            torch_module=torch,
        )


def test_adapter_rejects_runtime_cache_length_mismatch():
    cache = _cache()
    cache.get_seq_length = lambda: 2
    with pytest.raises(HuggingFaceKVAdapterError, match="runtime length"):
        serialize_huggingface_kv_cache(
            cache,
            logical_cached_token_count=3,
            model_id="Qwen/test",
            model_config=_config(),
            torch_module=torch,
        )


def test_adapter_rejects_missing_kv_and_layer_count_mismatch():
    cache = _cache()
    cache.layers[0].keys = None
    with pytest.raises(HuggingFaceKVAdapterError, match="initialized"):
        serialize_huggingface_kv_cache(
            cache,
            logical_cached_token_count=3,
            model_id="Qwen/test",
            model_config=_config(),
            torch_module=torch,
        )

    with pytest.raises(HuggingFaceKVAdapterError, match="model config"):
        serialize_huggingface_kv_cache(
            _cache(),
            logical_cached_token_count=3,
            model_id="Qwen/test",
            model_config=_config(3),
            torch_module=torch,
        )


def test_adapter_rejects_shape_head_and_dtype_incompatibility():
    cache = _cache()
    cache.layers[1].values = torch.zeros((1, 2, 2, 4), dtype=torch.bfloat16)
    with pytest.raises(HuggingFaceKVAdapterError, match="shapes differ"):
        serialize_huggingface_kv_cache(
            cache,
            logical_cached_token_count=3,
            model_id="Qwen/test",
            model_config=_config(),
            torch_module=torch,
        )

    cache = _cache()
    cache.layers[0].values = cache.layers[0].values.float()
    with pytest.raises(HuggingFaceKVAdapterError, match="dtypes differ"):
        serialize_huggingface_kv_cache(
            cache,
            logical_cached_token_count=3,
            model_id="Qwen/test",
            model_config=_config(),
            torch_module=torch,
        )
