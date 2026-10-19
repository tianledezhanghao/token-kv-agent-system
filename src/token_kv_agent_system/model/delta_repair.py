"""Audited dense delta append and static split-half post-RoPE conversion.

Shared by the 7.2A probe and persistent 7.6B receiver; no message Prefill.
"""
from __future__ import annotations
import inspect
from typing import Any
from .decode_kv import cache_layers

def reconstruct(pairs: tuple[tuple[Any, Any], ...], config: Any,
                cache_class: type | None = None) -> Any:
    if cache_class is None:
        from transformers import DynamicCache
        cache_class = DynamicCache
    # Constructor can retain storage: always supply privately owned tensors.
    cache = cache_class(ddp_cache_data=tuple((k.clone(), v.clone()) for k, v in pairs),
                        config=config)
    actual = cache_layers(cache)
    if len(actual) != len(pairs):
        raise ValueError("cache reconstruction changed layer count")
    if any(a.shape != k.shape for (a, _), (k, _) in zip(actual, pairs)):
        raise ValueError("cache reconstruction changed sequence length/layout")
    return cache


def clone_cache(cache: Any, config: Any, cache_class: type | None = None) -> Any:
    return reconstruct(cache_layers(cache), config, cache_class)


def append_delta(history: Any, delta: tuple[tuple[Any, Any], ...], config: Any,
                 cache_class: type | None = None) -> Any:
    import torch
    prefix = cache_layers(history)
    if len(prefix) != len(delta):
        raise ValueError("delta layer count/order mismatch")
    n = None
    joined = []
    for index, ((pk, pv), (dk, dv)) in enumerate(zip(prefix, delta)):
        if (dk.ndim != 4 or dk.shape != dv.shape or dk.shape[2] <= 0
                or pk.shape[:2] != dk.shape[:2] or pk.shape[3] != dk.shape[3]
                or pk.dtype != dk.dtype or pv.dtype != dv.dtype
                or pk.device != dk.device or pv.device != dv.device):
            raise ValueError(f"incompatible delta layer {index}")
        if n is not None and n != dk.shape[2]:
            raise ValueError("delta lengths differ between layers")
        n = dk.shape[2]
        joined.append((torch.cat((pk, dk), dim=2), torch.cat((pv, dv), dim=2)))
    result = reconstruct(tuple(joined), config, cache_class)
    if int(result.get_seq_length()) != int(history.get_seq_length()) + n:
        raise ValueError("append produced wrong cache length")
    return result


def delta_prefix(delta: tuple[tuple[Any, Any], ...], count: int):
    """Validated BHSD prefix views; append_delta reconstructs private storage.

    Empty prefix is handled by the caller, never appended to DynamicCache.
    """
    if not delta or type(count) is not int or count < 0:
        raise ValueError('invalid delta prefix count/layers')
    shape, dtype, device = delta[0][0].shape, delta[0][0].dtype, delta[0][0].device
    for k, v in delta:
        if (k.ndim != 4 or k.shape != shape or v.shape != shape or count > k.shape[2]
                or k.dtype != dtype or v.dtype != dtype
                or k.device != device or v.device != device):
            raise ValueError('delta prefix layer shape/dtype/device mismatch')
    return tuple((k[..., :count, :], v[..., :count, :]) for k, v in delta)


def rotate_half(x: Any) -> Any:
    import torch
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def rephase_key(key: Any, old_cos: Any, old_sin: Any,
                new_cos: Any, new_sin: Any) -> Any:
    """Invert old split-half RoPE (including scaling), then apply target RoPE.

    Tables are [1,S,rotary_dim]. Compute in FP32, cast once to cache dtype.
    This corrects stored keys, not position metadata. Values are never rotated.
    """
    import torch
    dim = old_cos.shape[-1]
    expected = (1, key.shape[2], dim)
    if (dim <= 0 or dim % 2 or dim > key.shape[-1]
            or any(tuple(t.shape) != expected for t in (old_cos, old_sin, new_cos, new_sin))):
        raise ValueError("invalid rotary dimensions/tables")
    if any(not torch.isfinite(t).all().item() for t in (key, old_cos, old_sin, new_cos, new_sin)):
        raise ValueError("nonfinite RoPE input")
    # Split-half representation requires equal coefficients in both halves.
    for table in (old_cos, old_sin, new_cos, new_sin):
        a, b = table.chunk(2, dim=-1)
        if not torch.equal(a, b):
            raise ValueError("unsupported rotary pairing")
    c, s, nc, ns = (t.float().unsqueeze(1) for t in (old_cos, old_sin, new_cos, new_sin))
    denom = c.square() + s.square()
    if (denom <= 0).any().item():
        raise ValueError("singular old RoPE transform")
    x = key[..., :dim].float()
    unrotated = (x * c - rotate_half(x) * s) / denom
    changed = (unrotated * nc + rotate_half(unrotated) * ns).to(key.dtype)
    return torch.cat((changed, key[..., dim:].clone()), dim=-1)



def rotary_tables(backend: Any, positions: list[int]) -> tuple[Any, Any]:
    import torch
    rotary = backend._model.model.rotary_emb
    # Use actual model-dtype tables, matching the quantization during forward.
    x = torch.empty((1, len(positions), backend.kv_model_config.head_dim),
                    device=backend.device, dtype=backend._model.dtype)
    ids = torch.tensor([positions], device=backend.device, dtype=torch.long)
    with torch.inference_mode():
        return rotary(x, ids)


def inspect_rope(backend: Any) -> dict[str, Any]:
    config = backend._model.config
    if config.model_type not in {"qwen2", "llama"}:
        raise ValueError("probe supports only audited Qwen2/Llama split-half RoPE")
    rotary = backend._model.model.rotary_emb
    rope_type = getattr(rotary, "rope_type", None)
    if rope_type not in {"default", "llama3"}:
        raise ValueError("sequence-dependent/unaudited RoPE is not supported")
    if any(t != "full_attention" for t in getattr(config, "layer_types", ())):
        raise ValueError("sliding/hybrid cache not supported")
    dim = rotary.inv_freq.numel() * 2
    if dim != backend.kv_model_config.head_dim:
        raise ValueError("model rotary dimension not audited")
    return {"rope_type": rope_type, "rope_parameters": getattr(config, "rope_parameters", None),
            "rope_scaling": getattr(config, "rope_scaling", None),
            "rotary_dim": dim, "pairing": "split_half", "K_is_post_RoPE": True,
            "V_is_rotated": False, "layout": "[batch,kv_heads,sequence,head_dim]",
            "attention_scaling": float(rotary.attention_scaling),
            "inv_freq": rotary.inv_freq.detach().float().cpu().tolist(),
            "position_ids": "implicit arange(new_tokens) + cache.get_seq_length(); no padding",
            "cache_position_declared": "cache_position" in inspect.signature(backend._model.model.forward).parameters,
            "forward_signature": str(inspect.signature(backend._model.model.forward))}

