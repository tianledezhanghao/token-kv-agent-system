"""Deterministic network simulation with optional wall-clock sleeping."""

from __future__ import annotations

import time

from token_kv_agent_system.metrics import CommunicationEvent, EventType, MetricCollector
from token_kv_agent_system.models import NetworkState

from .base import NetworkAdapter


class SimulatedNetwork(NetworkAdapter):
    BITS_PER_BYTE = 8
    BITS_PER_MEGABIT = 1_000_000
    MILLISECONDS_PER_SECOND = 1_000

    def __init__(
        self,
        bandwidth_mbps: float,
        base_latency_ms: float = 0.0,
        *,
        sleep_enabled: bool = False,
        collector: MetricCollector | None = None,
    ) -> None:
        self._state = NetworkState(bandwidth_mbps, base_latency_ms)
        self._sleep_enabled = sleep_enabled
        self._collector = collector

    @property
    def state(self) -> NetworkState:
        return self._state

    def estimate_transmission_latency_ms(self, payload_bytes: int) -> float:
        if payload_bytes < 0:
            raise ValueError("payload_bytes cannot be negative")
        payload_bits = payload_bytes * self.BITS_PER_BYTE
        bandwidth_bits_per_second = (
            self._state.bandwidth_mbps * self.BITS_PER_MEGABIT
        )
        serialization_ms = (
            payload_bits / bandwidth_bits_per_second * self.MILLISECONDS_PER_SECOND
        )
        return self._state.base_latency_ms + serialization_ms

    def transmit(self, payload_bytes: int) -> float:
        latency_ms = self.estimate_transmission_latency_ms(payload_bytes)
        if self._sleep_enabled:
            time.sleep(latency_ms / self.MILLISECONDS_PER_SECOND)
        return latency_ms

    def set_bandwidth_mbps(self, bandwidth_mbps: float) -> None:
        previous_state = self._state
        self._state = NetworkState(bandwidth_mbps, previous_state.base_latency_ms)
        if self._collector is not None:
            self._collector.record(
                CommunicationEvent(
                    event_type=EventType.NETWORK_STATE_CHANGED,
                    metadata={
                        "previous_bandwidth_mbps": previous_state.bandwidth_mbps,
                        "bandwidth_mbps": bandwidth_mbps,
                        "base_latency_ms": self._state.base_latency_ms,
                    },
                )
            )

