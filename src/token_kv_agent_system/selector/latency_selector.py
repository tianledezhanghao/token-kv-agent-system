"""Select the lower estimated-latency communication mode."""

from token_kv_agent_system.models import (
    CommunicationContext,
    CommunicationMode,
    SelectionDecision,
)
from token_kv_agent_system.network import NetworkAdapter

from .base import ModeSelector


class LatencyBasedSelector(ModeSelector):
    def select(
        self, context: CommunicationContext, network: NetworkAdapter
    ) -> SelectionDecision:
        token_latency_ms = network.estimate_transmission_latency_ms(
            context.token_payload.estimated_size_bytes
        ) + context.receiver_prefill_latency_ms
        kv_latency_ms = network.estimate_transmission_latency_ms(
            context.kv_payload.estimated_size_bytes
        )
        mode = (
            CommunicationMode.TOKEN
            if token_latency_ms <= kv_latency_ms
            else CommunicationMode.KV
        )
        return SelectionDecision(mode, token_latency_ms, kv_latency_ms)

