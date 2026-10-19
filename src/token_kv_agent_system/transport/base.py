"""Payload transport contract and shared event emission."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from token_kv_agent_system.metrics import CommunicationEvent, EventType, MetricCollector
from token_kv_agent_system.models import CommunicationMode
from token_kv_agent_system.network import NetworkAdapter


@dataclass(frozen=True)
class TransmissionResult:
    mode: CommunicationMode
    payload_bytes: int
    transmission_latency_ms: float
    estimated_e2e_latency_ms: float


class Transport(ABC):
    mode: CommunicationMode

    def __init__(self, network: NetworkAdapter, collector: MetricCollector) -> None:
        self._network = network
        self._collector = collector

    @abstractmethod
    def payload_size_bytes(self, payload: Any) -> int:
        """Validate a mode-specific payload and return its estimated size."""

    def send(
        self,
        payload: Any,
        *,
        source_agent_id: str,
        target_agent_id: str,
        receiver_prefill_latency_ms: float = 0.0,
    ) -> TransmissionResult:
        payload_bytes = self.payload_size_bytes(payload)
        estimated_e2e_latency_ms = self._network.estimate_transmission_latency_ms(
            payload_bytes
        ) + receiver_prefill_latency_ms
        common = {
            "source_agent_id": source_agent_id,
            "target_agent_id": target_agent_id,
            "mode": self.mode,
            "payload_bytes": payload_bytes,
            "estimated_e2e_latency_ms": estimated_e2e_latency_ms,
        }
        self._collector.record(
            CommunicationEvent(event_type=EventType.TRANSMISSION_STARTED, **common)
        )
        transmission_latency_ms = self._network.transmit(payload_bytes)
        result = TransmissionResult(
            mode=self.mode,
            payload_bytes=payload_bytes,
            transmission_latency_ms=transmission_latency_ms,
            estimated_e2e_latency_ms=estimated_e2e_latency_ms,
        )
        self._collector.record(
            CommunicationEvent(
                event_type=EventType.TRANSMISSION_FINISHED,
                transmission_latency_ms=transmission_latency_ms,
                **common,
            )
        )
        return result

