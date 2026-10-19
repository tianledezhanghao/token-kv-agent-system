"""Dense DynamicCache validation and isolation for persistent Agent state.

This intentionally supports full-attention, batch-one caches only. Sliding,
offloaded and static caches need a separately validated adapter.
"""
from __future__ import annotations

from typing import Any

from .cache_inspection import inspect_kv_cache


def validate_state_cache(cache: Any, count: int, layer_count: int) -> None:
    snapshot = inspect_kv_cache(cache, count)
    if snapshot.layer_count != layer_count or int(cache.get_seq_length()) != count:
        raise ValueError("state cache layer count/length mismatch")
    for layer in snapshot.layers:
        if (len(layer.key_shape) != 4 or layer.key_shape != layer.value_shape
                or layer.key_shape[0] != 1 or layer.key_shape[2] != count
                or layer.key_dtype != layer.value_dtype
                or layer.key_device != layer.value_device):
            raise ValueError("state cache has incomplete or inconsistent K/V layers")


def cache_storage_keys(cache: Any) -> set[tuple[str, int]]:
    return {(str(t.device), t.untyped_storage().data_ptr())
            for layer in cache.layers for t in (layer.keys, layer.values)}


def require_disjoint_caches(left: Any, right: Any) -> None:
    if left is right or cache_storage_keys(left) & cache_storage_keys(right):
        raise ValueError("Agent/work caches must not share mutable storage")


def clone_state_cache(cache: Any, config: Any) -> Any:
    from transformers import DynamicCache

    # DynamicCache can mutate during forward, even if forward later raises.
    # Copy on the current device; never serialize or re-Prefill the history.
    count = int(cache.get_seq_length())
    validate_state_cache(cache, count, config.num_hidden_layers)
    if type(cache) is not DynamicCache or any(
        type(layer).__name__ != "DynamicLayer" for layer in cache.layers
    ):
        raise ValueError("persistent state currently requires dense DynamicCache")
    result = DynamicCache(
        ddp_cache_data=tuple((layer.keys.detach().clone(), layer.values.detach().clone())
                            for layer in cache.layers),
        config=config,
    )
    validate_state_cache(result, count, config.num_hidden_layers)
    require_disjoint_caches(cache, result)
    return result
