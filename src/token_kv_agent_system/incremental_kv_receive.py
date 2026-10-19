"""Formal transactional Receiver entry point for incremental KV envelope bytes."""
from dataclasses import dataclass, replace
from time import perf_counter

from .agent.state import AgentState, append_working_ids, close_working_draft, working_state
from .communication import KVStrategy, HandoffChoice, CommunicationMode, suffix_recompute_count
from .incremental_kv_protocol import (ModelContract, decode_incremental_message,
                                     IncrementalLimits)
from .model.decode_kv import cache_layers
from .model.token_slot_repair import first_numeric, repair_slot
from .model.delta_repair import append_delta, delta_prefix, inspect_rope, rotary_tables, rephase_key
from .model.kv_adapter import restore_huggingface_kv_cache
from .model.state_cache import require_disjoint_caches
from .model.huggingface import synchronize_device


@dataclass(frozen=True)
class ReceiverAppendInfo:
    target_span: tuple[int, int]
    repair_method: str
    rope_position_delta: int
    message_prefill_count: int = 0
    recompute_forward_count: int = 0
    timings_ms: dict | None = None
    repaired_token_indices: tuple[int, ...] = ()
    repaired_layer_indices: tuple[int, ...] = ()
    repair_layer_token_count: int = 0


def receive_incremental_kv(state: AgentState, packet: bytes, *, contract: ModelContract,
                           source: str, sequence: int, strategy: KVStrategy,
                           recompute_tokens=None, recompute_ratio=None, measure_timing=False,
                           max_context_tokens: int | None = None,
                           limits: IncrementalLimits = IncrementalLimits()):
    """Return a fully validated work state; caller commits only after Sender closes.

    Sequence belongs to the common TOKEN/KV message ledger. No unpack counter is
    advanced here. Empty KV content is rejected by envelope v1, without fallback.
    """
    strategy = KVStrategy(strategy)
    HandoffChoice(CommunicationMode.KV, strategy, recompute_tokens, recompute_ratio)
    state.validate()
    expected_sequence = 1 + sum(m.source_agent == source for m in state.received_messages)
    if sequence != expected_sequence:
        raise ValueError('duplicate/out-of-order message sequence')
    b = state.backend
    timings = {} if measure_timing else None
    def measured(name, operation):
        if timings is None:
            return operation()
        synchronize_device(b._torch, b.device)
        started = perf_counter()
        result = operation()
        synchronize_device(b._torch, b.device)
        timings[name] = (perf_counter() - started) * 1000
        return result
    message = measured('decode', lambda: decode_incremental_message(packet, expected_contract=contract,
                                         source=source, target=state.agent_id,
                                         sequence=sequence, limits=limits))
    k = (suffix_recompute_count(message.message_token_count, tokens=recompute_tokens,
                               ratio=recompute_ratio)
         if strategy == KVStrategy.SUFFIX_RECOMPUTE_DELTA else 0)
    if strategy in (KVStrategy.ROPE_ALIGNED_DELTA, KVStrategy.SUFFIX_RECOMPUTE_DELTA, KVStrategy.NUMERIC_SLOT_DELTA):
        inspect_rope(b)  # Reject dynamic/unaudited rotations before local forward.
    selected = None
    if strategy == KVStrategy.NUMERIC_SLOT_DELTA:
        selected = measured('token_selection', lambda: first_numeric(message.message_token_ids,
            lambda ids: b.decode(ids, skip_special_tokens=False)))
        if selected is None:
            raise ValueError('V3 numeric_slot_delta requires an ASCII numeric content token')
    frame = state.framing.incoming(source, state.agent_id)
    extra = len(frame.prefix_ids) + message.message_token_count + len(frame.closing_ids)
    if state.assistant_open:
        extra += len(state.framing.assistant.closing_ids) - int(
            state.pending_token_id == state.framing.assistant.closing_ids[0])
    if max_context_tokens is not None and len(state.token_ledger) + extra > max_context_tokens:
        raise ValueError('Receiver exceeds context budget')
    restored = measured('restore', lambda: restore_huggingface_kv_cache(
        message.delta, model_id=b.model_name, model_config=b.kv_model_config,
        model_dtype=contract.dtype, target_device=b.device,
        transformers_config=b._model.config, torch_module=b._torch).cache)
    delta = cache_layers(restored)
    work = measured('work_clone_and_previous_close', lambda: close_working_draft(working_state(state)))
    work = measured('receiver_prefix', lambda: append_working_ids(work, frame.prefix_ids))
    start = work.cached_token_count  # Only now, after actual local framing forward.
    shift = start - message.sender_span_start
    def aligned_delta():
        old = rotary_tables(b, list(range(message.sender_span_start,
                                         message.sender_span_start + message.message_token_count)))
        new = rotary_tables(b, list(range(start, start + message.message_token_count)))
        return tuple((rephase_key(key, *old, *new), value) for key, value in delta)
    if strategy in (KVStrategy.ROPE_ALIGNED_DELTA, KVStrategy.SUFFIX_RECOMPUTE_DELTA, KVStrategy.NUMERIC_SLOT_DELTA):
        delta = measured('rope_rephase', aligned_delta)
    elif timings is not None:
        timings['rope_rephase'] = 0.0  # algorithmically absent in V0
    retained = message.message_token_count - k
    # k=0 uses exactly V1's full append; k=N appends no foreign cache.
    if retained:
        prefix = delta if k == 0 else delta_prefix(delta, retained)
        joined = measured('cache_append_copy', lambda: append_delta(work.cache, prefix, b._model.config))
        b.validate_cache(joined, start + retained)
        work = replace(work, cache=joined, cached_token_count=start + retained,
                       token_ledger=work.token_ledger + message.message_token_ids[:retained])
    elif timings is not None:
        timings['cache_append_copy'] = 0.0  # explicitly skipped at k=N
    work.validate()
    if k:
        work = measured('suffix_recompute', lambda: append_working_ids(work, message.message_token_ids[retained:]))
    elif timings is not None:
        timings['suffix_recompute'] = 0.0  # no message forward at k=0
    end = start + message.message_token_count
    b.validate_cache(work.cache, end)
    work.validate()
    if selected is not None:
        work = measured('slot_repair', lambda: replace(work, cache=repair_slot(
            b, work.cache, start, message.message_token_ids, selected)))
        work.validate()
    repaired = (selected,) if selected is not None else tuple(range(retained, message.message_token_count)) if k else ()
    layers = tuple(range(len(cache_layers(work.cache)))) if repaired else ()
    work = measured('receiver_closing', lambda: append_working_ids(work, frame.closing_ids))
    work.validate()
    require_disjoint_caches(state.cache, work.cache)
    # Append checks every layer against history dtype/device/layout; closing has
    # just continued Decode with the new cache length and full attention mask.
    pairs = cache_layers(work.cache)
    original = cache_layers(state.cache)
    for (key, v), (pk, pv) in zip(pairs, original):
        if (key.shape[:2] != pk.shape[:2] or key.shape[-1] != pk.shape[-1]
                or key.dtype != pk.dtype or v.dtype != pv.dtype
                or key.device != pk.device or v.device != pv.device):
            raise ValueError('Receiver cache layout/dtype/device changed')
    return work, message, ReceiverAppendInfo((start, end), strategy.value, shift, len(repaired), int(bool(repaired)), timings,
        repaired, layers, len(repaired)*len(layers))
