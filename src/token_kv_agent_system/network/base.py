"""Replaceable network contract used by selectors and transports."""

from __future__ import annotations

from abc import ABC, abstractmethod

from token_kv_agent_system.models import NetworkState


class NetworkAdapter(ABC):
    @property
    @abstractmethod
    def state(self) -> NetworkState:
        """Return a snapshot of current network conditions."""

    @abstractmethod
    def estimate_transmission_latency_ms(self, payload_bytes: int) -> float:
        """Estimate one-way latency, including base latency, in milliseconds."""

    @abstractmethod
    def transmit(self, payload_bytes: int) -> float:
        """Transmit/simulate a payload and return measured latency in milliseconds."""

