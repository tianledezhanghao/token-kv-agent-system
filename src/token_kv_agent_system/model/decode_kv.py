"""Dense Decode KV operations shared by the research probe and message exporter.

No Receiver append, RoPE conversion, or clean Prefill belongs here.
"""
from __future__ import annotations
from typing import Any

def cache_layers(cache: Any) -> tuple[tuple[Any, Any], ...]:
    """Validate dense, ordered [1,Hkv,S,D] DynamicCache; reject partial layers."""
    layers = getattr(cache, "layers", None)
    if layers is None or not layers:
        raise ValueError("expected initialized DynamicCache.layers")
    count = int(cache.get_seq_length())
    pairs = []
    shape = None
    for index, layer in enumerate(layers):
        key, value = getattr(layer, "keys", None), getattr(layer, "values", None)
        if key is None or value is None:
            raise ValueError(f"uninitialized layer {index}")
        if key.ndim != 4 or tuple(key.shape) != tuple(value.shape):
            raise ValueError(f"invalid [B,Hkv,S,D] layout at layer {index}")
        if key.shape[0] != 1 or key.shape[2] != count or count <= 0:
            raise ValueError(f"inconsistent cache length/batch at layer {index}")
        if key.dtype != value.dtype or key.device != value.device:
            raise ValueError(f"K/V dtype/device mismatch at layer {index}")
        descriptor = (tuple(key.shape), key.dtype, key.device)
        if shape is not None and descriptor != shape:
            raise ValueError("inconsistent layers")
        shape = descriptor
        pairs.append((key, value))
    return tuple(pairs)


def extract_span(cache: Any, start: int, end: int) -> tuple[tuple[Any, Any], ...]:
    pairs = cache_layers(cache)
    if not 0 <= start < end <= int(cache.get_seq_length()):
        raise ValueError("invalid or unmaterialized KV span [start,end)")
    return tuple((k[..., start:end, :].detach().clone(),
                  v[..., start:end, :].detach().clone()) for k, v in pairs)


def complete_message(backend: Any, cache: Any, history_length: int,
                     message_ids: tuple[int, ...]) -> tuple[Any, dict[str, Any]]:
    """Accept only an exact span or one final pending prediction; never re-prefill."""
    if history_length <= 0 or not message_ids:
        raise ValueError("history and message must be nonempty")
    before = int(cache.get_seq_length())
    wanted = history_length + len(message_ids)
    if before not in (wanted, wanted - 1):
        raise ValueError("cache does not match complete message or one pending token")
    cache_layers(cache)
    ms = 0.0
    if before == wanted - 1:
        out = backend.materialize_token_into_cache(
            message_ids[-1], previous_cache=cache, logical_cached_token_count=before)
        cache, ms = out.cache, out.materialization_forward_ms
    if int(cache.get_seq_length()) != wanted:
        raise ValueError("materialization did not grow exactly one position")
    cache_layers(cache)
    return cache, {"cache_before_materialization": before, "cache_after_materialization": wanted,
                   "pending_token_count": wanted - before,
                   "pending_token_id": message_ids[-1] if wanted != before else None,
                   "pending_token_materialization_ms": ms}


