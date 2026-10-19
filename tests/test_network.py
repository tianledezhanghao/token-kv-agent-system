import pytest

from token_kv_agent_system.network import SimulatedNetwork


def test_mbps_byte_bit_conversion_is_correct() -> None:
    network = SimulatedNetwork(bandwidth_mbps=20, base_latency_ms=5)

    # 1,000,000 bytes = 8,000,000 bits; at 20 Mbps that is 0.4 s.
    assert network.estimate_transmission_latency_ms(1_000_000) == pytest.approx(405)


def test_larger_payload_has_higher_latency_at_fixed_bandwidth() -> None:
    network = SimulatedNetwork(bandwidth_mbps=100, base_latency_ms=2)

    assert network.estimate_transmission_latency_ms(
        10_000
    ) > network.estimate_transmission_latency_ms(1_000)


def test_higher_bandwidth_has_lower_latency_for_fixed_payload() -> None:
    slow = SimulatedNetwork(bandwidth_mbps=20, base_latency_ms=2)
    fast = SimulatedNetwork(bandwidth_mbps=1_000, base_latency_ms=2)

    assert fast.estimate_transmission_latency_ms(
        1_000_000
    ) < slow.estimate_transmission_latency_ms(1_000_000)


@pytest.mark.parametrize("payload_bytes", [-1, -100])
def test_negative_payload_is_rejected(payload_bytes: int) -> None:
    network = SimulatedNetwork(bandwidth_mbps=100)

    with pytest.raises(ValueError):
        network.estimate_transmission_latency_ms(payload_bytes)

