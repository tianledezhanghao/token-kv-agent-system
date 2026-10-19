"""Experimental V3: ID-only first numeric slot repair, all decoder layers.

No semantic reliability claim. Earlier reused slots condition the repair; later
slots remain stale. No oracle, hidden-state transport, or custom model forward.
"""
from .decode_kv import cache_layers
from .delta_repair import reconstruct


def first_numeric(ids, decode):
    for index, token in enumerate(ids):
        piece = decode((token,)).strip()
        if piece.isascii() and piece.isdigit():
            return index
    return None


def repair_slot(backend, cache, start, ids, index):
    """Crop privately before selected absolute position, forward, scatter one slot.

    Normal causal mask sees only positions <= selected index, position_ids derive
    from cropped cache length. Preserves all nonselected slots exactly.
    """
    if type(index) is not int or not 0 <= index < len(ids):
        raise ValueError('invalid sparse slot')
    pairs = cache_layers(cache)
    count = int(cache.get_seq_length())
    if type(start) is not int or start <= 0 or start + len(ids) > count:
        raise ValueError('invalid repair span')
    backend.validate_cache(cache, count)
    position = start + index
    prefix = reconstruct(tuple((k[..., :position, :], v[..., :position, :]) for k, v in pairs), backend._model.config)
    out = backend.continue_from_cache((ids[index],), previous_cache=prefix,
                                     logical_cached_token_count=position)
    backend.validate_cache(out.cache, position+1)
    if out.input_token_ids != (ids[index],) or out.logical_cached_token_count != position+1:
        raise ValueError('slot forward input/count mismatch')
    repaired = []
    for original, computed in zip(pairs, cache_layers(out.cache)):
        slot = []
        for old, new in zip(original, computed):
            changed = old.clone()
            changed[..., position:position+1, :] = new[..., position:position+1, :]
            slot.append(changed)
        repaired.append(tuple(slot))
    result = reconstruct(tuple(repaired), backend._model.config)
    backend.validate_cache(result, count)
    return result
