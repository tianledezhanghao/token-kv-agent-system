from token_kv_agent_system.models import (
    CommunicationContext,
    CommunicationMode,
    KVPayload,
    TokenPayload,
)
from token_kv_agent_system.network import SimulatedNetwork
from token_kv_agent_system.selector import LatencyBasedSelector


def make_context(
    network: SimulatedNetwork,
    *,
    token_bytes: int = 1_000,
    kv_bytes: int = 1_000_000,
    prefill_ms: float = 20,
) -> CommunicationContext:
    return CommunicationContext(
        source_agent_id="A",
        target_agent_id="B",
        network_state=network.state,
        token_payload=TokenPayload((1, 2, 3), token_bytes),
        kv_payload=KVPayload("mock", kv_bytes),
        receiver_prefill_latency_ms=prefill_ms,
    )


def test_low_bandwidth_selects_token_from_estimated_costs() -> None:
    network = SimulatedNetwork(bandwidth_mbps=20, base_latency_ms=5)

    decision = LatencyBasedSelector().select(make_context(network), network)

    assert decision.mode is CommunicationMode.TOKEN
    assert decision.estimated_token_latency_ms < decision.estimated_kv_latency_ms


def test_high_bandwidth_selects_kv_from_estimated_costs() -> None:
    network = SimulatedNetwork(bandwidth_mbps=1_000, base_latency_ms=5)

    decision = LatencyBasedSelector().select(make_context(network), network)

    assert decision.mode is CommunicationMode.KV
    assert decision.estimated_kv_latency_ms < decision.estimated_token_latency_ms


def test_selection_is_not_a_bandwidth_threshold() -> None:
    network = SimulatedNetwork(bandwidth_mbps=100, base_latency_ms=5)
    selector = LatencyBasedSelector()

    low_prefill = selector.select(
        make_context(network, kv_bytes=10_000, prefill_ms=0), network
    )
    high_prefill = selector.select(
        make_context(network, kv_bytes=10_000, prefill_ms=10), network
    )

    assert low_prefill.mode is CommunicationMode.TOKEN
    assert high_prefill.mode is CommunicationMode.KV

