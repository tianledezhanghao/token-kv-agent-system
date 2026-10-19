"""Small, validated append-only chat-template adapter; no message text round-trip."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class MessageFrame:
    prefix_ids: tuple[int, ...]
    closing_ids: tuple[int, ...]


class ChatFraming:
    """Derive boundaries from the loaded template and fail closed if non-additive.

    A peer message is a user turn with a local identity label. Only the raw
    content IDs come from the peer; template framing belongs to the receiver.
    Templates requiring tools, role-dependent rewriting, or nonlocal history
    edits are outside this minimal adapter.
    """

    _marker = "TOKEN_BODY_PLACEHOLDER_75A"

    def __init__(self, tokenizer: Any, system_prompt: str) -> None:
        if not system_prompt.strip():
            raise ValueError("persistent Agents require a nonempty system prompt")
        self.tokenizer = tokenizer
        self.system = [{"role": "system", "content": system_prompt}]
        self.initial_ids = self._render_ids(self.system)
        if not self.initial_ids:
            raise ValueError("empty system context")
        self.assistant = self._frame(self.system, "assistant", "")
        rendered = self._render_ids(self.system, generation=True)
        if rendered != self.initial_ids + self.assistant.prefix_ids:
            raise ValueError("template generation prefix is not append-compatible")
        self._check_closing(self.assistant.closing_ids)

    def _encode(self, text: str) -> tuple[int, ...]:
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    def _text(self, messages: list[dict[str, str]], generation: bool = False) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation)

    def _render_ids(self, messages: list[dict[str, str]], generation: bool = False) -> tuple[int, ...]:
        text = self._text(messages, generation)
        ids = self._encode(text)
        actual = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=generation)
        if isinstance(actual, Mapping):
            actual = actual["input_ids"]
        if tuple(actual) != ids:
            raise ValueError("template tokenize/text paths disagree")
        return ids

    def _frame(self, prior: list[dict[str, str]], role: str, label: str) -> MessageFrame:
        prior_text = self._text(prior)
        full_messages = prior + [{"role": role, "content": label + self._marker}]
        text = self._text(full_messages)
        if text.count(self._marker) != 1 or not text.startswith(prior_text):
            raise ValueError("template rewrites history/content; cannot incrementally frame")
        before, after = text[len(prior_text):].split(self._marker)
        frame = MessageFrame(self._encode(before), self._encode(after))
        if self._render_ids(full_messages) != (
            self._render_ids(prior) + frame.prefix_ids
            + self._encode(self._marker) + frame.closing_ids
        ):
            raise ValueError("template token boundaries are not append-compatible")
        return frame

    def _check_closing(self, ids: tuple[int, ...]) -> None:
        if not ids or ids[0] not in self.tokenizer.all_special_ids:
            raise ValueError("template needs a leading special EOT in its closing fragment")
        if any(i in self.tokenizer.all_special_ids for i in ids[1:]):
            raise ValueError("unsupported multi-special-token closing fragment")

    def incoming(self, source: str, target: str) -> MessageFrame:
        for identity in (source, target):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", identity):
                raise ValueError("Agent IDs must be plain letters, digits, underscore or hyphen")
        label = f"Message from {source} to {target}:\n"
        frame = self._frame(self.system, "user", label)
        self._check_closing(frame.closing_ids)
        # Validate the same fragments after a completed assistant turn, including
        # assistant generation start. Never tokenize the actual received content.
        prior = self.system + [{"role": "user", "content": "Earlier input."},
                               {"role": "assistant", "content": "Earlier reply."}]
        if self._frame(prior, "user", label) != frame:
            raise ValueError("template framing changes with history")
        messages = prior + [{"role": "user", "content": label + self._marker}]
        closed = self._render_ids(messages)
        if self._render_ids(messages, True) != closed + self.assistant.prefix_ids:
            raise ValueError("assistant generation prefix changes with history")
        assistant_prior = prior + [{"role": "user", "content": label + "Prior peer input."}]
        if self._frame(assistant_prior, "assistant", "") != self.assistant:
            raise ValueError("assistant boundary changes with history")
        return frame

    def local_user_ids(self, text: str) -> tuple[int, ...]:
        # Local text is allowed to be tokenized; peer IDs are never decoded here.
        if any(token in text for token in self.tokenizer.all_special_tokens):
            raise ValueError("local message must not contain chat control tokens")
        before = self._render_ids(self.system)
        after = self._render_ids(self.system + [{"role": "user", "content": text}])
        if after[:len(before)] != before:
            raise ValueError("local user template rewrites context")
        return after[len(before):]
