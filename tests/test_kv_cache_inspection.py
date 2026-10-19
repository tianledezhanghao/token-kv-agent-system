from math import prod
from types import SimpleNamespace

import pytest

from token_kv_agent_system.model import (
    calculate_raw_kv_tensor_bytes,
    config_derived_raw_kv_tensor_bytes,
    format_byte_count,
    inspect_kv_cache,
    inspect_kv_model_config,
)


class FakeTensor:
    def __init__(
        self,
        shape,
        *,
        dtype="torch.bfloat16",
        device="mps:0",
        element_size=2,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self._element_size = element_size

    def numel(self):
        return prod(self.shape)

    def element_size(self):
        return self._element_size


class FakeLayer:
    def __init__(self, key, value):
        self.keys = key
        self.values = value


class FakeDynamicCache:
    def __init__(self, layers):
        self.layers = list(layers)


def make_cache(*, layers=1, batch=1, heads=2, tokens=3, dim=4, element_size=2):
    return FakeDynamicCache(
        [
            FakeLayer(
                FakeTensor((batch, heads, tokens, dim), element_size=element_size),
                FakeTensor((batch, heads, tokens, dim), element_size=element_size),
            )
            for _ in range(layers)
        ]
    )


def test_inspects_one_layer_shape_dtype_device_numel_and_bytes():
    snapshot = inspect_kv_cache(make_cache(), 3)
    layer = snapshot.layers[0]

    assert layer.key_shape == (1, 2, 3, 4)
    assert layer.value_shape == (1, 2, 3, 4)
    assert layer.key_dtype == layer.value_dtype == "torch.bfloat16"
    assert layer.key_device == layer.value_device == "mps:0"
    assert layer.key_numel == layer.value_numel == 24
    assert layer.key_element_size == layer.value_element_size == 2
    assert layer.key_bytes == layer.value_bytes == 48


def test_sums_multiple_layer_key_value_and_total_bytes():
    snapshot = inspect_kv_cache(make_cache(layers=3), 3)

    assert snapshot.layer_count == 3
    assert snapshot.total_key_bytes == 144
    assert snapshot.total_value_bytes == 144
    assert snapshot.raw_kv_tensor_bytes == 288
    assert snapshot.layers_are_uniform is True
    assert snapshot.runtime_cached_token_count == 3


def test_snapshot_contains_metadata_not_source_objects_or_tensor_references():
    cache = make_cache()
    source_layer = cache.layers[0]
    source_key = source_layer.keys
    source_value = source_layer.values

    snapshot = inspect_kv_cache(cache, 3)

    assert snapshot is not cache
    assert snapshot.layers[0] is not source_layer
    assert all(
        value is not source_key and value is not source_value
        for value in vars(snapshot.layers[0]).values()
    )


def test_old_snapshot_is_unchanged_after_mutable_cache_grows():
    cache = make_cache(tokens=3)
    before = inspect_kv_cache(cache, 3)

    cache.layers[0].keys = FakeTensor((1, 2, 4, 4))
    cache.layers[0].values = FakeTensor((1, 2, 4, 4))
    after = inspect_kv_cache(cache, 4)

    assert before.layers[0].key_shape == (1, 2, 3, 4)
    assert before.raw_kv_tensor_bytes == 96
    assert after.layers[0].key_shape == (1, 2, 4, 4)
    assert after.raw_kv_tensor_bytes == 128


def test_records_nonuniform_layers_instead_of_assuming_uniformity():
    cache = make_cache(layers=2)
    cache.layers[1].keys = FakeTensor((1, 2, 2, 4))
    cache.layers[1].values = FakeTensor((1, 2, 2, 4))

    snapshot = inspect_kv_cache(cache, 3)

    assert snapshot.layers_are_uniform is False
    assert snapshot.runtime_cached_token_count is None
    assert snapshot.layers[1].key_shape == (1, 2, 2, 4)


def test_model_config_distinguishes_attention_and_kv_heads_and_derives_head_dim():
    config = inspect_kv_model_config(
        SimpleNamespace(
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            hidden_size=896,
        )
    )

    assert config.num_attention_heads == 14
    assert config.num_key_value_heads == 2
    assert config.head_dim == 64
    assert config.head_dim_source == "derived: hidden_size / num_attention_heads"


def test_model_config_prefers_real_head_dim_field_when_present():
    config = inspect_kv_model_config(
        SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            hidden_size=1024,
            head_dim=80,
        )
    )

    assert config.head_dim == 80
    assert config.head_dim_source == "config.head_dim"


def test_formula_uses_batch_and_kv_heads_not_attention_heads():
    assert calculate_raw_kv_tensor_bytes(
        num_layers=3,
        batch_size=2,
        num_key_value_heads=4,
        cached_tokens=5,
        head_dim=8,
        key_element_size=2,
        value_element_size=2,
    ) == 3 * 2 * 4 * 5 * 8 * (2 + 2)


def test_config_derived_formula_matches_runtime_snapshot():
    config = inspect_kv_model_config(
        SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            hidden_size=32,
        )
    )
    snapshot = inspect_kv_cache(make_cache(layers=2, batch=2), 3)

    assert config_derived_raw_kv_tensor_bytes(config, snapshot) == 384
    assert snapshot.raw_kv_tensor_bytes == 384


def test_bytes_grow_by_formula_amount_for_each_cached_token():
    cache = make_cache(layers=2, tokens=3)
    snapshots = [inspect_kv_cache(cache, 3)]
    for tokens in (4, 5, 6):
        for layer in cache.layers:
            layer.keys = FakeTensor((1, 2, tokens, 4))
            layer.values = FakeTensor((1, 2, tokens, 4))
        snapshots.append(inspect_kv_cache(cache, tokens))

    growth = [
        after.raw_kv_tensor_bytes - before.raw_kv_tensor_bytes
        for before, after in zip(snapshots, snapshots[1:])
    ]
    assert growth == [64, 64, 64]


def test_format_byte_count_uses_bytes_kib_and_mib():
    assert format_byte_count(2048) == "2,048 B (2.00 KiB, 0.00 MiB)"
    assert format_byte_count(1048576) == "1,048,576 B (1024.00 KiB, 1.00 MiB)"


def test_old_key_cache_shape_is_not_mistaken_for_current_layers_api():
    legacy_like_cache = SimpleNamespace(key_cache=[], value_cache=[])

    with pytest.raises(TypeError, match="Cache.layers"):
        inspect_kv_cache(legacy_like_cache, 1)
