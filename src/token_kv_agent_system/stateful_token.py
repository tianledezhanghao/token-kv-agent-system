"""Independent state with incremental TOKEN/KV receive, without discussion policy."""
from __future__ import annotations

from dataclasses import dataclass, replace, field
from threading import Lock
from time import perf_counter
from typing import Iterable

from .agent.llm_agent import LLMAgent
from .agent.state import (AgentMessage, AgentState, append_working_ids,
                          close_working_draft, initialize_agent_state, working_state)
from .model.backend_pair import validate_backend_compatibility
from .model.state_cache import require_disjoint_caches
from .token_protocol import deserialize_token_packet, serialize_token_ids
from .transport import BaseByteTransport


@dataclass(frozen=True)
class IncrementalTokenReceipt:
    message: AgentMessage
    wire_byte_count: int
    received_token_ids: tuple[int, ...]
    receiver_appended_ids: tuple[int, ...]
    receiver_cached_count_before: int
    receiver_cached_count_after: int
    receiver_pending_materialized: bool
    token_timings_ms: dict | None = field(default=None, kw_only=True)

    @property
    def timings_ms(self):
        return self.token_timings_ms


@dataclass(frozen=True)
class IncrementalKVReceipt(IncrementalTokenReceipt):
    kv_strategy: str
    sender_content_span: tuple[int, int]
    receiver_target_span: tuple[int, int]
    receiver_message_prefill_count: int
    repair_method: str
    rope_position_delta: int
    kv_protocol_wire_bytes: int
    sender_supplemented_token_count: int
    recompute_forward_count: int = 0
    timings_ms: dict | None = None
    repaired_token_indices: tuple[int, ...] = ()
    repaired_layer_indices: tuple[int, ...] = ()
    repair_layer_token_count: int = 0


class ContextBudgetError(ValueError):
    """No safe output fits; raised before generation/communication mutation."""


class StatefulTokenSession:
    """Synchronous owner of two long-lived states, not a discussion runtime.

    Metadata is local session data, outside the unchanged v1 binary packet.
    Source draft survives failed delivery for retry. Neither state's effective
    cache/message sequence is committed until the complete handoff succeeds.
    """

    def __init__(self, agents: Iterable[LLMAgent], transport: BaseByteTransport,
                 *, max_context_tokens: int | None = None) -> None:
        if max_context_tokens is not None and (type(max_context_tokens) is not int or max_context_tokens <= 0):
            raise ValueError("max_context_tokens must be a positive integer")
        self.max_context_tokens = max_context_tokens
        agents = tuple(agents)
        if len(agents) != 2 or agents[0].agent_id == agents[1].agent_id:
            raise ValueError("stateful TOKEN session requires two distinct Agents")
        validate_backend_compatibility(agents[0].backend, agents[1].backend)
        self._states = {agent.agent_id: initialize_agent_state(agent, max_context_tokens=max_context_tokens)
                        for agent in agents}
        require_disjoint_caches(*(state.cache for state in self._states.values()))
        self.transport = transport
        self._lock = Lock()

    def state(self, agent_id: str) -> AgentState:
        return self._states[agent_id]

    def _pair(self, source: str, target: str) -> tuple[AgentState, AgentState]:
        if source == target:
            raise ValueError("source and target must differ")
        left, right = self.state(source), self.state(target)
        validate_backend_compatibility(left.backend, right.backend)
        left.validate()
        right.validate()
        require_disjoint_caches(left.cache, right.cache)
        return left, right

    def _reserve(self, state: AgentState, extra: int) -> None:
        if self.max_context_tokens is not None and len(state.token_ledger) + extra > self.max_context_tokens:
            raise ContextBudgetError(f"Agent {state.agent_id} exceeds context budget {self.max_context_tokens}")

    @staticmethod
    def _closing_extra(state: AgentState) -> int:
        if not state.assistant_open:
            return 0
        closing = state.framing.assistant.closing_ids
        return len(closing) - int(state.pending_token_id == closing[0])

    def append_local_user(self, agent_id: str, text: str) -> AgentState:
        with self._lock:
            old = self.state(agent_id)
            ids = old.framing.local_user_ids(text)
            self._reserve(old, self._closing_extra(old) + len(ids))
            work = close_working_draft(working_state(old))
            start = work.cached_token_count
            work = append_working_ids(work, ids)
            work = replace(work, version=old.version + 1,
                           local_input_spans=old.local_input_spans + ((start, work.cached_token_count),))
            work.validate()
            self._states[agent_id] = work
            return work

    def generate_message(self, source: str, target: str, *, max_new_tokens: int = 8,
                         local_prompt: str | None = None, generation_budget: dict | None = None) -> AgentMessage:
        """Cached greedy output; optional local instruction commits with generation."""
        with self._lock:
            old, _ = self._pair(source, target)
            work = self._generate_working(old, target, max_new_tokens, local_prompt, generation_budget)
            self._states[source] = work
            return work.draft_message

    def generate_local_message(self, agent_id: str, *, max_new_tokens: int = 64,
                               local_prompt: str, generation_budget: dict | None = None) -> AgentMessage:
        """Generate and close a local answer atomically, without a wire handoff."""
        with self._lock:
            old = self.state(agent_id)
            old.validate()
            work = self._generate_working(old, agent_id, max_new_tokens, local_prompt, generation_budget)
            message = replace(work.draft_message,
                              message_id=f"{agent_id}:local:{len(old.local_messages) + 1}",
                              sequence=len(old.local_messages) + 1)
            work = close_working_draft(work)
            work = replace(work, draft_message=None,
                           local_messages=old.local_messages + (message,))
            work.validate()
            self._states[agent_id] = work
            return message

    def _generate_working(self, old: AgentState, target: str, max_new_tokens: int,
                          local_prompt: str | None, generation_budget: dict | None = None) -> AgentState:
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        source = old.agent_id
        if old.draft_message is not None:
            raise ValueError("deliver existing draft before generating another message")
        local_ids = old.framing.local_user_ids(local_prompt) if local_prompt is not None else ()
        self._reserve(old, len(local_ids) + len(old.framing.assistant.prefix_ids)
                      + max_new_tokens + len(old.framing.assistant.closing_ids))
        work = working_state(old)
        if local_ids:
            local_start = work.cached_token_count
            work = append_working_ids(work, local_ids)
            work = replace(work, local_input_spans=old.local_input_spans
                           + ((local_start, work.cached_token_count),))
        prefix = work.framing.assistant.prefix_ids
        if not prefix:
            raise ValueError("empty assistant generation prefix")
        # Only this new prefix enters forward; receive leaves a closed user turn.
        out = work.backend.continue_from_cache(
            prefix, previous_cache=work.cache,
            logical_cached_token_count=work.cached_token_count)
        expected = work.cached_token_count + len(prefix)
        if out.logical_cached_token_count != expected or out.input_token_ids != prefix:
            raise ValueError("assistant prefix forward mismatch")
        work.backend.validate_cache(out.cache, expected)
        work = replace(work, cache=out.cache, token_ledger=work.token_ledger + prefix,
                       cached_token_count=expected)
        start = expected
        ids = [out.predicted_next_token_id]
        while ids[-1] not in work.backend.eos_token_ids and len(ids) < max_new_tokens:
            consumed = ids[-1]
            out = work.backend.continue_from_cache(
                (consumed,), previous_cache=work.cache,
                logical_cached_token_count=work.cached_token_count)
            expected = work.cached_token_count + 1
            if out.logical_cached_token_count != expected or out.input_token_ids != (consumed,):
                raise ValueError("cached Decode count/input mismatch")
            work.backend.validate_cache(out.cache, expected)
            work = replace(work, cache=out.cache, cached_token_count=expected,
                           token_ledger=work.token_ledger + (consumed,))
            ids.append(out.predicted_next_token_id)
        reason = "eos" if ids[-1] in work.backend.eos_token_ids else (
            "context_limit" if generation_budget and generation_budget['budget_limit_reason'] else "max_new_tokens")
        message = AgentMessage(f"{source}:{len(old.sent_messages) + 1}",
                               len(old.sent_messages) + 1, source, target, tuple(ids),
                               reason, (start, start + len(ids)),
                               **(generation_budget or dict(configured_max_new_tokens=max_new_tokens,
                                   effective_max_new_tokens=max_new_tokens)))
        self._validate_message(message, work)
        work = replace(work, token_ledger=work.token_ledger + (ids[-1],),
                       pending_token_id=ids[-1], assistant_open=True,
                       draft_message=message, version=old.version + 1)
        work.validate()
        return work

    def materialize_message(self, agent_id: str) -> AgentState:
        """Finish an accepted output and close its turn; no further prediction is accepted."""
        with self._lock:
            old = self.state(agent_id)
            if not old.assistant_open:
                old.validate()
                return old
            self._reserve(old, self._closing_extra(old))
            work = close_working_draft(working_state(old))
            work = replace(work, version=old.version + 1)
            work.validate()
            self._states[agent_id] = work
            return work

    @staticmethod
    def _validate_message(message: AgentMessage, source: AgentState) -> None:
        ids = message.message_token_ids
        eos = source.backend.eos_token_ids
        if message.termination_reason not in ("eos", "max_new_tokens", "context_limit"):
            raise ValueError("unsupported message termination reason")
        if message.payload_mode != "TOKEN" or not ids:
            raise ValueError("only nonempty TOKEN messages are implemented")
        if any(type(i) is not int or not 0 <= i < source.backend.vocab_size for i in ids):
            raise ValueError("message Token ID outside model vocabulary")
        if any(i in eos for i in ids[:-1]):
            raise ValueError("message contains an interior EOS")
        if (message.termination_reason == "eos") != (ids[-1] in eos):
            raise ValueError("termination reason and final EOS disagree")
        # A model-generated role opener/control token is not ordinary content.
        allowed = set(eos)
        if any(i in source.backend.chat_tokenizer.all_special_ids and i not in allowed for i in ids):
            raise ValueError("message contains a chat control token")

    def handoff(self, source: str, target: str, *, measure_timing=False) -> IncrementalTokenReceipt:
        with self._lock:
            sender, receiver = self._pair(source, target)
            message = sender.draft_message
            if message is None or message.target_agent != target:
                raise ValueError("source has no draft for this target")
            self._validate_message(message, sender)
            expected_sequence = 1 + sum(m.source_agent == source for m in receiver.received_messages)
            if message.sequence != expected_sequence:
                raise ValueError("duplicate/out-of-order message sequence")
            frame = receiver.framing.incoming(source, target)
            received = message.message_token_ids
            # Terminal generation stop is control metadata, not peer utterance
            # content. Reuse a matching EOT as the user close, or replace a
            # different EOS with the actual local template close.
            content = received[:-1] if message.termination_reason == "eos" else received
            if any(i in receiver.backend.chat_tokenizer.all_special_ids for i in content):
                raise ValueError("received content contains a chat control token")
            closing = frame.closing_ids
            self._reserve(sender, self._closing_extra(sender))
            self._reserve(receiver, self._closing_extra(receiver)
                          + len(frame.prefix_ids) + len(content) + len(closing))
            from .model.huggingface import synchronize_device
            timings = {} if measure_timing else None
            def measured(name, operation):
                if timings is None:
                    return operation()
                for b in (sender.backend, receiver.backend):
                    synchronize_device(getattr(b, '_torch', None), getattr(b, 'device', 'cpu'))
                started = perf_counter()
                result = operation()
                for b in (sender.backend, receiver.backend):
                    synchronize_device(getattr(b, '_torch', None), getattr(b, 'device', 'cpu'))
                timings[name] = (perf_counter() - started) * 1000
                return result
            total_started = perf_counter()
            wire = measured('encode', lambda: serialize_token_ids(message.message_token_ids))
            delivered = measured('local_transport', lambda: self.transport.transfer(wire, source=source, target=target))
            packet = measured('decode', lambda: deserialize_token_packet(delivered))
            if packet.token_ids != message.message_token_ids:
                raise ValueError("transport altered TOKEN payload")
            received = packet.token_ids
            # Rebuild from the IDs actually delivered, after strict identity validation.
            content = received[:-1] if message.termination_reason == "eos" else received
            body_and_close = content + closing
            if message.termination_reason == "eos" and received[-1] == closing[0]:
                body_and_close = received + closing[1:]
            sender_work = measured('sender_clone_close', lambda: close_working_draft(working_state(sender)))
            receiver_work = measured('receiver_clone_close', lambda: close_working_draft(working_state(receiver)))
            message_start = receiver_work.cached_token_count + len(frame.prefix_ids)
            appended = frame.prefix_ids + body_and_close
            receiver_work = measured('receiver_framing_and_message_prefill', lambda: append_working_ids(receiver_work, appended))
            record = replace(message, receiver_message_start=message_start,
                             receiver_content_token_count=len(content))
            sender_work = replace(sender_work, draft_message=None,
                                  sent_messages=sender.sent_messages + (record,),
                                  version=sender.version + 1)
            receiver_work = replace(receiver_work,
                                    received_messages=receiver.received_messages + (record,),
                                    version=receiver.version + 1)
            sender_work.validate()
            receiver_work.validate()
            require_disjoint_caches(sender_work.cache, receiver_work.cache)
            receipt = IncrementalTokenReceipt(record, len(wire), received, appended,
                                              receiver.cached_token_count,
                                              receiver_work.cached_token_count,
                                              receiver.pending_token_id is not None,
                                              token_timings_ms=timings)
            if timings is not None:
                timings['handoff_total'] = (perf_counter() - total_started) * 1000
            # Both assignments happen after all forward/validation work succeeds.
            self._states.update({source: sender_work, target: receiver_work})
            return receipt

    def handoff_kv(self, source: str, target: str, *, strategy, recompute_tokens=None,
                   recompute_ratio=None, measure_timing=False) -> IncrementalKVReceipt:
        """Send Decode delta through envelope bytes and commit both states atomically."""
        from .communication import KVStrategy, HandoffChoice, CommunicationMode
        from .model.huggingface import synchronize_device
        from .incremental_kv import prepare_incremental_message
        from .incremental_kv_protocol import (inspect_model_contract,
                                             encode_incremental_message_with_metrics)
        from .incremental_kv_receive import receive_incremental_kv

        strategy = KVStrategy(strategy)
        HandoffChoice(CommunicationMode.KV, strategy, recompute_tokens, recompute_ratio)
        with self._lock:
            sender, receiver = self._pair(source, target)
            timings = {} if measure_timing else None
            def measured(name, operation):
                if timings is None:
                    return operation()
                synchronize_device(sender.backend._torch, sender.backend.device)
                synchronize_device(receiver.backend._torch, receiver.backend.device)
                started = perf_counter()
                result = operation()
                synchronize_device(sender.backend._torch, sender.backend.device)
                synchronize_device(receiver.backend._torch, receiver.backend.device)
                timings[name] = (perf_counter() - started) * 1000
                return result
            total_started = perf_counter()
            message = sender.draft_message
            if message is None or message.target_agent != target:
                raise ValueError('source has no draft for this target')
            self._validate_message(message, sender)
            sequence = 1 + sum(m.source_agent == source for m in receiver.received_messages)
            if sequence != message.sequence:
                raise ValueError('duplicate/out-of-order message sequence')
            # Contracts fingerprint immutable model weights/config/tokenizer once.
            if not hasattr(self, '_kv_contract'):
                a = inspect_model_contract(sender.backend)
                b = a if sender.backend is receiver.backend else inspect_model_contract(receiver.backend)
                if a != b:
                    raise ValueError('incompatible checkpoint/tokenizer/config/cache contract')
                self._kv_contract = a
            self._reserve(sender, self._closing_extra(sender))
            envelope, export = measured('export', lambda: prepare_incremental_message(sender, message, contract=self._kv_contract))
            frame = receiver.framing.incoming(source, target)
            self._reserve(receiver, self._closing_extra(receiver) + len(frame.prefix_ids)
                          + envelope.message_token_count + len(frame.closing_ids))
            encoded = measured('encode', lambda: encode_incremental_message_with_metrics(envelope))
            delivered = measured('local_transport', lambda: self.transport.transfer(encoded.packet, source=source, target=target))
            work, decoded, info = receive_incremental_kv(
                receiver, delivered, contract=self._kv_contract, source=source,
                sequence=sequence, strategy=strategy, recompute_tokens=recompute_tokens,
                recompute_ratio=recompute_ratio, measure_timing=measure_timing,
                max_context_tokens=self.max_context_tokens)
            if decoded != envelope:
                raise ValueError('transport altered incremental payload')
            sender_work = measured('sender_close', lambda: close_working_draft(working_state(sender)))
            if timings is not None:
                timings.update(info.timings_ms)
                timings['sender_pending'] = export.pending_token_ms
                timings['tensor_extraction'] = export.tensor_extraction_ms
                timings['handoff_total'] = (perf_counter() - total_started) * 1000
            record = replace(message, payload_mode='KV', receiver_message_start=info.target_span[0],
                             receiver_content_token_count=decoded.message_token_count)
            sender_work = replace(sender_work, draft_message=None,
                                  sent_messages=sender.sent_messages + (record,), version=sender.version + 1)
            work = replace(work, received_messages=receiver.received_messages + (record,),
                           version=receiver.version + 1)
            sender_work.validate()
            work.validate()
            require_disjoint_caches(sender_work.cache, work.cache)
            receipt = IncrementalKVReceipt(
                record, len(encoded.packet), decoded.raw_generated_token_ids,
                frame.prefix_ids + decoded.message_token_ids + frame.closing_ids,
                receiver.cached_token_count, work.cached_token_count,
                receiver.pending_token_id is not None, strategy.value,
                (decoded.sender_span_start, decoded.sender_span_start + decoded.sender_span_length),
                info.target_span, info.message_prefill_count, info.repair_method,
                info.rope_position_delta, encoded.kv_wire_bytes, export.supplemented_token_count,
                info.recompute_forward_count, timings, info.repaired_token_indices,
                info.repaired_layer_indices, info.repair_layer_token_count)
            self._states.update({source: sender_work, target: work})
            return receipt
