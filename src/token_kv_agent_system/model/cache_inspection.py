"""Read-only inspection and raw tensor accounting for runtime KV caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KVLayerStats:
    """Immutable metadata captured from one attention cache layer."""

    layer_index: int
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    key_dtype: str
    value_dtype: str
    key_device: str
    value_device: str
    key_numel: int
    value_numel: int
    key_element_size: int
    value_element_size: int
    key_bytes: int
    value_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.key_bytes + self.value_bytes


@dataclass(frozen=True)
class KVCacheSnapshot:
    """Point-in-time KV metadata with no cache, layer, or tensor references."""

    cache_type: str
    logical_cached_token_count: int
    layer_count: int
    total_key_bytes: int
    total_value_bytes: int
    raw_kv_tensor_bytes: int
    layers_are_uniform: bool
    layers: tuple[KVLayerStats, ...]

    @property
    def runtime_cached_token_count(self) -> int | None:
        lengths = {
            shape[-2]
            for layer in self.layers
            for shape in (layer.key_shape, layer.value_shape)
            if len(shape) >= 2
        }
        return next(iter(lengths)) if len(lengths) == 1 else None


@dataclass(frozen=True)
class KVModelConfig:
    """Cache-relevant model configuration normalized without tensor data."""

    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    head_dim: int
    head_dim_source: str


def inspect_kv_model_config(config: Any) -> KVModelConfig:
    """Read the current model config and derive head_dim only when absent."""

    num_hidden_layers = _positive_config_int(config, "num_hidden_layers")
    num_attention_heads = _positive_config_int(config, "num_attention_heads")
    num_key_value_heads = _positive_config_int(config, "num_key_value_heads")
    hidden_size = _positive_config_int(config, "hidden_size")

    configured_head_dim = getattr(config, "head_dim", None)
    if configured_head_dim is None:
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "cannot derive head_dim: hidden_size is not divisible by "
                "num_attention_heads"
            )
        head_dim = hidden_size // num_attention_heads
        head_dim_source = "derived: hidden_size / num_attention_heads"
    else:
        head_dim = _positive_int(configured_head_dim, "head_dim")
        head_dim_source = "config.head_dim"

    return KVModelConfig(
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        hidden_size=hidden_size,
        head_dim=head_dim,
        head_dim_source=head_dim_source,
    )


def inspect_kv_cache(
    cache: object,
    logical_cached_token_count: int,
) -> KVCacheSnapshot:
    """Inspect a Transformers 5.17 Cache through ``layers[*].keys/values``.

    Only tensor metadata is read. The function does not run a forward pass,
    clone a tensor, or move cache data between devices.
    """

    if logical_cached_token_count <= 0:
        raise ValueError("logical_cached_token_count must be greater than zero")
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError(
            "unsupported cache: expected the Transformers Cache.layers API"
        )

    layer_stats: list[KVLayerStats] = []
    for layer_index, layer in enumerate(layers):
        key = getattr(layer, "keys", None)
        value = getattr(layer, "values", None)
        if key is None or value is None:
            raise ValueError(f"cache layer {layer_index} has no initialized K/V tensors")
        layer_stats.append(_inspect_layer(layer_index, key, value))

    if not layer_stats:
        raise ValueError("cache has no initialized layers")

    frozen_layers = tuple(layer_stats)
    total_key_bytes = sum(layer.key_bytes for layer in frozen_layers)
    total_value_bytes = sum(layer.value_bytes for layer in frozen_layers)
    comparison_fields = (
        "key_shape",
        "value_shape",
        "key_dtype",
        "value_dtype",
        "key_device",
        "value_device",
        "key_element_size",
        "value_element_size",
    )
    first = frozen_layers[0]
    layers_are_uniform = all(
        all(getattr(layer, field) == getattr(first, field) for field in comparison_fields)
        for layer in frozen_layers[1:]
    )
    cache_type = f"{type(cache).__module__}.{type(cache).__qualname__}"
    return KVCacheSnapshot(
        cache_type=cache_type,
        logical_cached_token_count=logical_cached_token_count,
        layer_count=len(frozen_layers),
        total_key_bytes=total_key_bytes,
        total_value_bytes=total_value_bytes,
        raw_kv_tensor_bytes=total_key_bytes + total_value_bytes,
        layers_are_uniform=layers_are_uniform,
        layers=frozen_layers,
    )


def calculate_raw_kv_tensor_bytes(
    *,
    num_layers: int,
    batch_size: int,
    num_key_value_heads: int,
    cached_tokens: int,
    head_dim: int,
    key_element_size: int,
    value_element_size: int,
) -> int:
    """Calculate K+V bytes for the verified dense four-dimensional layout."""

    values = {
        "num_layers": num_layers,
        "batch_size": batch_size,
        "num_key_value_heads": num_key_value_heads,
        "cached_tokens": cached_tokens,
        "head_dim": head_dim,
        "key_element_size": key_element_size,
        "value_element_size": value_element_size,
    }
    for name, value in values.items():
        _positive_int(value, name)
    elements_per_tensor_kind = (
        num_layers * batch_size * num_key_value_heads * cached_tokens * head_dim
    )
    return elements_per_tensor_kind * (key_element_size + value_element_size)


def config_derived_raw_kv_tensor_bytes(
    config: KVModelConfig,
    snapshot: KVCacheSnapshot,
) -> int:
    """Validate runtime layout against config and return formula-derived bytes."""

    if snapshot.layer_count != config.num_hidden_layers:
        raise ValueError(
            f"cache has {snapshot.layer_count} layers; config expects "
            f"{config.num_hidden_layers}"
        )
    runtime_tokens = snapshot.runtime_cached_token_count
    if runtime_tokens is None:
        raise ValueError("cache sequence length is inconsistent across K/V tensors")
    if runtime_tokens != snapshot.logical_cached_token_count:
        raise ValueError(
            f"runtime cache length {runtime_tokens} does not match logical length "
            f"{snapshot.logical_cached_token_count}"
        )

    first = snapshot.layers[0]
    if len(first.key_shape) != 4 or len(first.value_shape) != 4:
        raise ValueError("expected K/V layout [batch, KV heads, tokens, head dim]")
    batch_size = first.key_shape[0]
    expected_shape = (
        batch_size,
        config.num_key_value_heads,
        runtime_tokens,
        config.head_dim,
    )
    for layer in snapshot.layers:
        if layer.key_shape != expected_shape or layer.value_shape != expected_shape:
            raise ValueError(
                f"layer {layer.layer_index} K/V shape does not match config-derived "
                f"shape {expected_shape}: {layer.key_shape}, {layer.value_shape}"
            )
        if (
            layer.key_element_size != first.key_element_size
            or layer.value_element_size != first.value_element_size
        ):
            raise ValueError("K/V element sizes are inconsistent across layers")

    return calculate_raw_kv_tensor_bytes(
        num_layers=config.num_hidden_layers,
        batch_size=batch_size,
        num_key_value_heads=config.num_key_value_heads,
        cached_tokens=runtime_tokens,
        head_dim=config.head_dim,
        key_element_size=first.key_element_size,
        value_element_size=first.value_element_size,
    )


def format_byte_count(byte_count: int) -> str:
    """Format one raw byte count using binary KiB and MiB units."""

    if byte_count < 0:
        raise ValueError("byte_count cannot be negative")
    return (
        f"{byte_count:,} B "
        f"({byte_count / 1024:.2f} KiB, {byte_count / (1024**2):.2f} MiB)"
    )


def _inspect_layer(layer_index: int, key: Any, value: Any) -> KVLayerStats:
    key_shape = _shape_tuple(key, layer_index, "K")
    value_shape = _shape_tuple(value, layer_index, "V")
    key_numel = _tensor_int_method(key, "numel", layer_index, "K")
    value_numel = _tensor_int_method(value, "numel", layer_index, "V")
    key_element_size = _tensor_int_method(key, "element_size", layer_index, "K")
    value_element_size = _tensor_int_method(value, "element_size", layer_index, "V")
    return KVLayerStats(
        layer_index=layer_index,
        key_shape=key_shape,
        value_shape=value_shape,
        key_dtype=str(getattr(key, "dtype")),
        value_dtype=str(getattr(value, "dtype")),
        key_device=str(getattr(key, "device")),
        value_device=str(getattr(value, "device")),
        key_numel=key_numel,
        value_numel=value_numel,
        key_element_size=key_element_size,
        value_element_size=value_element_size,
        key_bytes=key_numel * key_element_size,
        value_bytes=value_numel * value_element_size,
    )


def _shape_tuple(tensor: Any, layer_index: int, label: str) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise TypeError(f"cache layer {layer_index} {label} has no tensor shape")
    return tuple(int(size) for size in shape)


def _tensor_int_method(tensor: Any, method: str, layer_index: int, label: str) -> int:
    callback = getattr(tensor, method, None)
    if not callable(callback):
        raise TypeError(f"cache layer {layer_index} {label} has no tensor {method}()")
    return _positive_int(callback(), f"layer {layer_index} {label} {method}")


def _positive_config_int(config: Any, attribute: str) -> int:
    value = getattr(config, attribute, None)
    if value is None:
        raise ValueError(f"model config does not expose {attribute}")
    return _positive_int(value, attribute)


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
