"""Deterministic CPU-only agent used by the v0.1 demo and tests."""

from __future__ import annotations

from token_kv_agent_system.metrics import CommunicationEvent, EventType, MetricCollector
from token_kv_agent_system.models import KVPayload, TokenPayload

from .base import Agent, AgentGeneration


class MockAgent(Agent):
    """Generate deterministic mock payloads whose sizes are estimates, not wire bytes."""

    def __init__(
        self,
        agent_id: str,
        *,
        token_count: int = 250,
        bytes_per_token_id: int = 4,
        kv_size_bytes: int = 1_000_000,
        receiver_prefill_latency_ms: float = 20.0,
        collector: MetricCollector | None = None,
    ) -> None:
        if token_count < 0 or bytes_per_token_id <= 0 or kv_size_bytes < 0:
            raise ValueError("mock payload dimensions must be non-negative")
        if receiver_prefill_latency_ms < 0:
            raise ValueError("receiver_prefill_latency_ms cannot be negative")
        self._agent_id = agent_id
        self._token_count = token_count
        self._bytes_per_token_id = bytes_per_token_id
        self._kv_size_bytes = kv_size_bytes
        self._receiver_prefill_latency_ms = receiver_prefill_latency_ms
        self._collector = collector
        self._generation_index = 0

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def generate(self, prompt: str) -> AgentGeneration:
        if self._collector is not None:
            self._collector.record(
                CommunicationEvent(
                    event_type=EventType.AGENT_GENERATION_STARTED,
                    source_agent_id=self.agent_id,
                    metadata={"prompt": prompt},
                )
            )

        self._generation_index += 1
        token_ids = tuple(range(self._token_count))
        generation = AgentGeneration(
            text=f"{self.agent_id} 模拟响应 {self._generation_index}：{prompt}",
            token_payload=TokenPayload(
                token_ids=token_ids,
                estimated_size_bytes=len(token_ids) * self._bytes_per_token_id,
            ),
            kv_payload=KVPayload(
                cache_key=f"{self.agent_id}-mock-kv-{self._generation_index}",
                estimated_size_bytes=self._kv_size_bytes,
            ),
            receiver_prefill_latency_ms=self._receiver_prefill_latency_ms,
        )

        if self._collector is not None:
            self._collector.record(
                CommunicationEvent(
                    event_type=EventType.AGENT_GENERATION_FINISHED,
                    source_agent_id=self.agent_id,
                    metadata={
                        "text": generation.text,
                        "token_count": len(token_ids),
                        "token_payload_bytes": generation.token_payload.estimated_size_bytes,
                        "kv_payload_bytes": generation.kv_payload.estimated_size_bytes,
                    },
                )
            )
        return generation
