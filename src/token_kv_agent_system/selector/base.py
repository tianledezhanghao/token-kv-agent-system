"""Mode selection contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from token_kv_agent_system.models import CommunicationContext, SelectionDecision
from token_kv_agent_system.network import NetworkAdapter


class ModeSelector(ABC):
    @abstractmethod
    def select(
        self, context: CommunicationContext, network: NetworkAdapter
    ) -> SelectionDecision:
        """Choose a mode from estimated end-to-end costs."""

