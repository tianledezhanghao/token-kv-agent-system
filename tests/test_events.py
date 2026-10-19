from token_kv_agent_system.agent import MockAgent
from token_kv_agent_system.metrics import EventType, MetricCollector
from token_kv_agent_system.models import CommunicationMode
from token_kv_agent_system.network import SimulatedNetwork
from token_kv_agent_system.orchestrator import CommunicationOrchestrator
from token_kv_agent_system.selector import LatencyBasedSelector
from token_kv_agent_system.transport import KVTransport, TokenTransport


def test_one_communication_records_structured_events() -> None:
    collector = MetricCollector()
    network = SimulatedNetwork(20, 5, collector=collector)
    orchestrator = CommunicationOrchestrator(
        network=network,
        selector=LatencyBasedSelector(),
        transports={
            CommunicationMode.TOKEN: TokenTransport(network, collector),
            CommunicationMode.KV: KVTransport(network, collector),
        },
        collector=collector,
    )

    result = orchestrator.communicate(
        MockAgent("A", collector=collector),
        MockAgent("B", collector=collector),
        "hello",
    )

    assert result.decision.mode is CommunicationMode.TOKEN
    assert [event.event_type for event in collector.events] == [
        EventType.AGENT_GENERATION_STARTED,
        EventType.AGENT_GENERATION_FINISHED,
        EventType.MODE_SELECTED,
        EventType.TRANSMISSION_STARTED,
        EventType.TRANSMISSION_FINISHED,
    ]
    finished = collector.events_of_type(EventType.TRANSMISSION_FINISHED)[0]
    assert finished.source_agent_id == "A"
    assert finished.target_agent_id == "B"
    assert finished.mode is CommunicationMode.TOKEN
    assert finished.payload_bytes == 1_000
    assert finished.transmission_latency_ms is not None
    assert finished.estimated_e2e_latency_ms is not None

