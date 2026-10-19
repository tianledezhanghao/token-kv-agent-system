"""Decode message KV export and isolated byte receive; no receiver cache append."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import perf_counter

from .agent.state import AgentMessage, AgentState
from .incremental_kv_protocol import (IncrementalKVMessage, IncrementalLimits, ModelContract,
                                     decode_incremental_message,
                                     encode_incremental_message_with_metrics,
                                     inspect_model_contract, validate_message)
from .model.decode_kv import complete_message
from .model.huggingface import synchronize_device
from .model.kv_adapter import extract_huggingface_kv_span
from .model.state_cache import require_disjoint_caches
from .token_protocol import serialize_token_ids
from .transport import BaseByteTransport


@dataclass(frozen=True)
class ExportDiagnostics:
    raw_generated_token_ids: tuple[int, ...]
    message_token_ids: tuple[int, ...]
    termination_reason: str
    sender_cache_length: int
    sender_span_start: int
    sender_span_length: int
    original_message_token_count: int
    cached_message_token_count: int
    supplemented_token_count: int
    pending_token_ms: float
    tensor_extraction_ms: float
    delta_raw_bytes: int
    layer_count: int
    model: str
    device: str
    dtype: str


@dataclass(frozen=True)
class IncrementalTransportDiagnostics:
    export: ExportDiagnostics
    token_protocol_wire_bytes: int  # Existing discussion raw IDs, including EOS.
    envelope_token_protocol_wire_bytes: int  # Content packet actually nested here.
    kv_protocol_wire_bytes: int
    envelope_total_bytes: int
    envelope_overhead_bytes: int
    kv_serialize_ms: float
    envelope_serialize_ms: float
    deserialize_ms: float
    bitwise_exact: bool


def prepare_incremental_message(state: AgentState, message: AgentMessage, *,
                                contract: ModelContract,
                                limits: IncrementalLimits = IncrementalLimits()
                                ) -> tuple[IncrementalKVMessage, ExportDiagnostics]:
    """Export ledger-proven content only. Pending forward uses private storage."""
    state.validate()
    if message != state.draft_message and message not in state.sent_messages:
        raise ValueError('message is not an accepted Sender draft/sent record')
    if message.source_agent != state.agent_id or message.payload_mode != 'TOKEN':
        raise ValueError('incorrect Sender or source message mode')
    raw = message.message_token_ids
    start, raw_end = message.sender_message_span
    if (type(start) is not int or type(raw_end) is not int or start <= 0
            or raw_end - start != len(raw) or state.token_ledger[start:raw_end] != raw):
        raise ValueError('message span does not match Sender ledger')
    if not raw or (message.termination_reason == 'eos') != (raw[-1] in state.backend.eos_token_ids):
        raise ValueError('raw generation/termination mismatch')
    content = raw[:-1] if message.termination_reason == 'eos' else raw
    if not content:
        raise ValueError('empty message content has no nonempty KV v1 span')
    end = start + len(content)
    # Reject excessive allocations before cache clone, slicing or host copy.
    raw_bytes = (2 * contract.num_hidden_layers * contract.num_key_value_heads
                 * len(content) * contract.head_dim
                 * {'bfloat16': 2, 'float16': 2, 'float32': 4}[contract.dtype])
    if len(content) > limits.max_message_tokens or raw_bytes > limits.max_packet_bytes or end > limits.max_sender_positions:
        raise ValueError('incremental message exceeds size/position limits')
    before = state.cached_token_count
    cached = max(0, min(before, end) - start)
    cache = state.cache
    pending_ms = 0.0
    missing = len(content) - cached
    if missing:
        if (missing != 1 or before != end - 1 or state.pending_token_id != content[-1]
                or state.draft_message != message):
            raise ValueError('message cache incomplete; expected only one accepted pending content Token')
        cache = state.backend.clone_cache(cache)
        require_disjoint_caches(state.cache, cache)
        cache, info = complete_message(state.backend, cache, start, content)
        pending_ms = info['pending_token_materialization_ms']
        state.backend.validate_cache(cache, end)
    synchronize_device(state.backend._torch, state.backend.device)
    started = perf_counter()
    delta = extract_huggingface_kv_span(
        cache, start, end, model_id=state.backend.model_name,
        model_config=state.backend.kv_model_config, torch_module=state.backend._torch)
    synchronize_device(state.backend._torch, state.backend.device)
    extract_ms = (perf_counter() - started) * 1000
    envelope = IncrementalKVMessage(
        message.message_id, message.sequence, message.source_agent, message.target_agent,
        content, len(content), message.termination_reason,
        raw[-1] if message.termination_reason == 'eos' else None,
        start, len(content), contract, delta)
    validate_message(envelope, expected_contract=contract, source=message.source_agent,
                     target=message.target_agent, sequence=message.sequence, limits=limits)
    if message.receiver_content_token_count is not None and message.receiver_content_token_count != len(content):
        raise ValueError('TOKEN Receiver content boundary disagrees with Delta')
    info = ExportDiagnostics(raw, content, message.termination_reason, before, start, len(content),
                             len(content), cached, missing, pending_ms, extract_ms,
                             delta.raw_kv_tensor_bytes, len(delta.layers), state.backend.model_name,
                             state.backend.device, contract.dtype)
    return envelope, info


class IncrementalKVReceiver:
    """Strict unpack-only endpoint. Validated sequence commits after all checks.

    No AgentState/cache/model forward is owned or changed by this endpoint.
    """
    def __init__(self, contract: ModelContract, source: str, target: str, *,
                 limits: IncrementalLimits = IncrementalLimits()) -> None:
        self.contract, self.source, self.target, self.limits = contract, source, target, limits
        self.next_sequence = 1
        self._lock = Lock()

    def unpack(self, packet: bytes, *, expected_message: IncrementalKVMessage | None = None) -> IncrementalKVMessage:
        with self._lock:
            message = decode_incremental_message(
                packet, expected_contract=self.contract, source=self.source, target=self.target,
                sequence=self.next_sequence, limits=self.limits)
            if expected_message is not None and message != expected_message:
                raise ValueError('incremental round-trip is not bitwise exact')
            self.next_sequence += 1
            return message


class IncrementalKVChannel:
    """Separate side channel: existing TOKEN discussion execution is untouched."""
    def __init__(self, backend_a, backend_b, transport: BaseByteTransport, *,
                 limits: IncrementalLimits = IncrementalLimits()) -> None:
        a = inspect_model_contract(backend_a)
        b = a if backend_a is backend_b else inspect_model_contract(backend_b)
        if a != b:
            raise ValueError('incompatible checkpoint/tokenizer/config/cache contract')
        self.contract = a
        self.transport, self.limits = transport, limits
        self.receivers = {(s, t): IncrementalKVReceiver(a, s, t, limits=limits)
                          for s, t in (('A', 'B'), ('B', 'A'))}
        self.backends = {'A': backend_a, 'B': backend_b}
        self._lock = Lock()

    def send(self, state: AgentState, message: AgentMessage
             ) -> tuple[IncrementalKVMessage, IncrementalTransportDiagnostics]:
        with self._lock:
            if state.backend is not self.backends.get(message.source_agent):
                raise ValueError('Sender is not a registered channel backend')
            receiver = self.receivers.get((message.source_agent, message.target_agent))
            if receiver is None:
                raise ValueError('incorrect source/target endpoint')
            if message.sequence != receiver.next_sequence:
                raise ValueError('duplicate/out-of-order message sequence')
            envelope, export = prepare_incremental_message(
                state, message, contract=self.contract, limits=self.limits)
            validate_message(envelope, expected_contract=self.contract,
                             source=receiver.source, target=receiver.target,
                             sequence=receiver.next_sequence, limits=self.limits)
            started = perf_counter()
            encoded = encode_incremental_message_with_metrics(envelope, limits=self.limits)
            encode_ms = (perf_counter() - started) * 1000
            delivered = self.transport.transfer(encoded.packet, source=receiver.source, target=receiver.target)
            # Identity transport contract, before committing receipt sequence.
            if delivered != encoded.packet:
                raise ValueError('transport altered incremental payload')
            started = perf_counter()
            decoded = receiver.unpack(delivered, expected_message=envelope)
            decode_ms = (perf_counter() - started) * 1000
            exact = decoded == envelope  # Includes every K/V byte, dtype and shape.
            if not exact:
                raise RuntimeError('incremental round-trip is not bitwise exact')
            token_bytes = len(serialize_token_ids(envelope.message_token_ids))
            return decoded, IncrementalTransportDiagnostics(
                export, len(serialize_token_ids(message.message_token_ids)), token_bytes,
                encoded.kv_wire_bytes, len(encoded.packet),
                len(encoded.packet) - token_bytes - encoded.kv_wire_bytes,
                encoded.kv_serialize_ms, encode_ms, decode_ms, exact)
