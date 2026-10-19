"""Two-stage deterministic CLI demonstration."""

from __future__ import annotations

from collections.abc import Callable

from token_kv_agent_system.agent import MockAgent
from token_kv_agent_system.metrics import CommunicationEvent, EventType, MetricCollector
from token_kv_agent_system.models import CommunicationMode
from token_kv_agent_system.network import SimulatedNetwork
from token_kv_agent_system.selector import LatencyBasedSelector
from token_kv_agent_system.transport import KVTransport, TokenTransport

from .orchestrator import CommunicationOrchestrator, CommunicationResult

Output = Callable[[str], None]


def _event_printer(output: Output) -> Callable[[CommunicationEvent], None]:
    def print_event(event: CommunicationEvent) -> None:
        if event.event_type is EventType.MODE_SELECTED:
            output("[Selector]")
            output(f"{event.mode.value} selected")
            output(
                "estimated TOKEN latency = "
                f"{event.metadata['estimated_token_latency_ms']:.3f} ms"
            )
            output(
                "estimated KV latency = "
                f"{event.metadata['estimated_kv_latency_ms']:.3f} ms"
            )
        elif event.event_type is EventType.TRANSMISSION_FINISHED:
            output("[Transport]")
            output(f"{event.source_agent_id} -> {event.target_agent_id}")
            output(f"mode = {event.mode.value}")
            output(f"payload = {event.payload_bytes} bytes")
            output(f"transmission latency = {event.transmission_latency_ms:.3f} ms")
            output(f"estimated E2E latency = {event.estimated_e2e_latency_ms:.3f} ms")
        elif event.event_type is EventType.NETWORK_STATE_CHANGED:
            output("[Network]")
            output(f"bandwidth changed to {event.metadata['bandwidth_mbps']:g} Mbps")

    return print_event


def run_demo(
    output: Output = print,
    *,
    collector: MetricCollector | None = None,
) -> tuple[list[CommunicationResult], MetricCollector]:
    collector = collector if collector is not None else MetricCollector()
    collector.subscribe(_event_printer(output))
    network = SimulatedNetwork(
        bandwidth_mbps=20,
        base_latency_ms=5,
        sleep_enabled=False,
        collector=collector,
    )
    selector = LatencyBasedSelector()
    transports = {
        CommunicationMode.TOKEN: TokenTransport(network, collector),
        CommunicationMode.KV: KVTransport(network, collector),
    }
    orchestrator = CommunicationOrchestrator(
        network=network,
        selector=selector,
        transports=transports,
        collector=collector,
    )
    agent_a = MockAgent("Agent A", collector=collector)
    agent_b = MockAgent("Agent B", collector=collector)

    output("[Network]")
    output(
        f"bandwidth = {network.state.bandwidth_mbps:g} Mbps, "
        f"base latency = {network.state.base_latency_ms:g} ms"
    )
    results = [
        orchestrator.communicate(agent_a, agent_b, "提出一个确定性方案")
    ]

    network.set_bandwidth_mbps(1_000)
    results.append(
        orchestrator.communicate(agent_b, agent_a, "审查已提出的方案")
    )
    return results, collector


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
