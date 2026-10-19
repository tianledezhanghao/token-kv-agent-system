"""Application service connecting replaceable agents and infrastructure."""

from __future__ import annotations

from dataclasses import dataclass

from token_kv_agent_system.agent import Agent, AgentGeneration
from token_kv_agent_system.metrics import CommunicationEvent, EventType, MetricCollector
from token_kv_agent_system.models import (
    CommunicationContext,
    CommunicationMode,
    SelectionDecision,
)
from token_kv_agent_system.network import NetworkAdapter
from token_kv_agent_system.selector import ModeSelector
from token_kv_agent_system.transport import Transport, TransmissionResult


@dataclass(frozen=True)
class CommunicationResult:
    generation: AgentGeneration
    decision: SelectionDecision
    transmission: TransmissionResult


class CommunicationOrchestrator:
    def __init__(
        self,
        *,
        network: NetworkAdapter,
        selector: ModeSelector,
        transports: dict[CommunicationMode, Transport],
        collector: MetricCollector,
    ) -> None:
        missing = set(CommunicationMode) - set(transports)
        if missing:
            raise ValueError(f"missing transports for modes: {sorted(m.value for m in missing)}")
        self._network = network
        self._selector = selector
        self._transports = transports
        self._collector = collector

    def communicate(
        self, source: Agent, target: Agent, prompt: str
    ) -> CommunicationResult:
        generation = source.generate(prompt)
        context = CommunicationContext(
            source_agent_id=source.agent_id,
            target_agent_id=target.agent_id,
            network_state=self._network.state,
            token_payload=generation.token_payload,
            kv_payload=generation.kv_payload,
            receiver_prefill_latency_ms=generation.receiver_prefill_latency_ms,
        )
        decision = self._selector.select(context, self._network)
        self._collector.record(
            CommunicationEvent(
                event_type=EventType.MODE_SELECTED,
                source_agent_id=source.agent_id,
                target_agent_id=target.agent_id,
                mode=decision.mode,
                metadata={
                    "bandwidth_mbps": context.network_state.bandwidth_mbps,
                    "base_latency_ms": context.network_state.base_latency_ms,
                    "estimated_token_latency_ms": decision.estimated_token_latency_ms,
                    "estimated_kv_latency_ms": decision.estimated_kv_latency_ms,
                },
            )
        )
        if decision.mode is CommunicationMode.TOKEN:
            payload = generation.token_payload
            receiver_prefill_latency_ms = generation.receiver_prefill_latency_ms
        else:
            payload = generation.kv_payload
            receiver_prefill_latency_ms = 0.0
        transmission = self._transports[decision.mode].send(
            payload,
            source_agent_id=source.agent_id,
            target_agent_id=target.agent_id,
            receiver_prefill_latency_ms=receiver_prefill_latency_ms,
        )
        return CommunicationResult(generation, decision, transmission)

