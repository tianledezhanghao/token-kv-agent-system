"""Persistent Agent state separate from the canonical handoff runtimes."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .chat_framing import ChatFraming
from .llm_agent import LLMAgent
from ..model.state_cache import require_disjoint_caches


@dataclass(frozen=True)
class AgentMessage:
    message_id: str
    sequence: int
    source_agent: str
    target_agent: str
    message_token_ids: tuple[int, ...]
    termination_reason: Literal["eos", "max_new_tokens", "context_limit"]
    sender_message_span: tuple[int, int]
    payload_mode: str = "TOKEN"
    receiver_message_start: int | None = None
    receiver_content_token_count: int | None = None

    configured_max_new_tokens: int | None = None
    effective_max_new_tokens: int | None = None
    context_remaining_tokens: int | None = None
    budget_limit_reason: str | None = None

    @property
    def token_count(self) -> int:
        return len(self.message_token_ids)


@dataclass(frozen=True)
class AgentState:
    """Ledger[:cached_token_count] maps exactly to effective KV positions.

    At most one accepted output token may await forward. An unaccepted argmax
    prediction is not a ledger entry. Cache is session-owned: callers must not
    mutate it. Every update uses separate work storage before replacing state.
    """

    agent_id: str
    system_prompt: str
    backend: Any = field(repr=False, compare=False)
    framing: ChatFraming = field(repr=False, compare=False)
    token_ledger: tuple[int, ...]
    cache: Any = field(repr=False, compare=False)
    cached_token_count: int
    pending_token_id: int | None = None
    assistant_open: bool = False
    draft_message: AgentMessage | None = None
    sent_messages: tuple[AgentMessage, ...] = ()
    received_messages: tuple[AgentMessage, ...] = ()
    local_input_spans: tuple[tuple[int, int], ...] = ()
    version: int = 0
    local_messages: tuple[AgentMessage, ...] = ()

    def validate(self) -> None:
        self.backend.validate_cache(self.cache, self.cached_token_count)
        pending = int(self.pending_token_id is not None)
        if len(self.token_ledger) != self.cached_token_count + pending:
            raise ValueError("ledger and cached/pending positions disagree")
        if pending and (self.token_ledger[-1] != self.pending_token_id
                        or not self.assistant_open or self.draft_message is None):
            raise ValueError("invalid accepted pending token")
        if self.assistant_open and not pending:
            raise ValueError("open assistant must have one pending output token")
        if self.draft_message is not None:
            message = self.draft_message
            start, end = message.sender_message_span
            if (message.source_agent != self.agent_id
                    or self.token_ledger[start:end] != message.message_token_ids
                    or message.sequence != len(self.sent_messages) + 1):
                raise ValueError("draft span/sequence disagrees with ledger")

        for sequence, message in enumerate(self.local_messages, 1):
            start, end = message.sender_message_span
            if (message.source_agent != self.agent_id or message.target_agent != self.agent_id
                    or message.sequence != sequence
                    or not 0 <= start < end <= self.cached_token_count
                    or self.token_ledger[start:end] != message.message_token_ids):
                raise ValueError("local message span/sequence disagrees with ledger")


def initialize_agent_state(agent: LLMAgent, *, max_context_tokens: int | None = None) -> AgentState:
    backend = agent.backend
    for method in ("continue_from_cache", "clone_cache", "validate_cache"):
        if not callable(getattr(backend, method, None)):
            raise TypeError(f"persistent state backend requires {method}")
    framing = ChatFraming(backend.chat_tokenizer, agent.system_prompt or "")
    # Validate endpoint label and template before allocating the initial cache.
    framing.incoming(agent.agent_id, agent.agent_id)
    if max_context_tokens is not None and len(framing.initial_ids) > max_context_tokens:
        raise ValueError("system prompt exceeds context budget")
    initial = backend.prefill(input_token_ids=framing.initial_ids)
    if initial.logical_cached_token_count != len(framing.initial_ids):
        raise ValueError("initial Prefill count mismatch")
    state = AgentState(agent.agent_id, agent.system_prompt or "", backend, framing,
                       framing.initial_ids, initial.cache, len(framing.initial_ids))
    state.validate()
    return state


def working_state(state: AgentState) -> AgentState:
    state.validate()
    cache = state.backend.clone_cache(state.cache)
    require_disjoint_caches(state.cache, cache)
    state.backend.validate_cache(cache, state.cached_token_count)
    return replace(state, cache=cache)


def append_working_ids(state: AgentState, ids: tuple[int, ...]) -> AgentState:
    """Internal only: consume uncached IDs on already-isolated work storage."""
    if not ids or state.pending_token_id is not None:
        raise ValueError("append requires nonempty IDs and no pending token")
    out = state.backend.continue_from_cache(
        ids, previous_cache=state.cache,
        logical_cached_token_count=state.cached_token_count)
    expected = state.cached_token_count + len(ids)
    if out.logical_cached_token_count != expected or out.input_token_ids != ids:
        raise ValueError("incremental forward input/count mismatch")
    state.backend.validate_cache(out.cache, expected)
    return replace(state, cache=out.cache, cached_token_count=expected,
                   token_ledger=state.token_ledger + ids)


def close_working_draft(state: AgentState) -> AgentState:
    if not state.assistant_open:
        return state
    closing = state.framing.assistant.closing_ids
    # EOT predicted by the model already closes the turn; only append its tail.
    if state.pending_token_id == closing[0]:
        closing = closing[1:]
    pending = state.pending_token_id
    state = replace(state, token_ledger=state.token_ledger[:-1],
                    pending_token_id=None, assistant_open=False)
    return append_working_ids(state, (pending,) + closing)
