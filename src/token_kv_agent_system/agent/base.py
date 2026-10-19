"""Replaceable agent contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from token_kv_agent_system.models import KVPayload, TokenPayload


@dataclass(frozen=True)
class AgentGeneration:
    text: str
    token_payload: TokenPayload
    kv_payload: KVPayload
    receiver_prefill_latency_ms: float


class Agent(ABC):
    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Stable identifier used in communication events."""

    @abstractmethod
    def generate(self, prompt: str) -> AgentGeneration:
        """Generate output plus transferable TOKEN and KV representations."""

