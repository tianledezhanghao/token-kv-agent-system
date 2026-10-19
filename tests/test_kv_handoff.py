from __future__ import annotations

from dataclasses import fields
import json

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from token_kv_agent_system.agent import LLMAgent  # noqa: E402
from token_kv_agent_system.kv_handoff import (  # noqa: E402
    BidirectionalKVHandoffSession,
    CanonicalHandoffSpan,
    KVHandoffResult,
    compare_canonical_token_kv,
    measure_kv_scaling,
    perform_kv_handoff,
)
from token_kv_agent_system.model import HuggingFaceModelBackend  # noqa: E402
from token_kv_agent_system.network import SimulatedNetwork  # noqa: E402
from token_kv_agent_system.transport import (  # noqa: E402
    BaseByteTransport,
    InMemoryByteTransport,
)


class _Tokenizer:
    vocab_size = 64
    eos_token_id = 63
    bos_token_id = 1
    pad_token_id = 0

    def encode(self, text: str):
        return [1, 2, 3]

    def decode(self, token_ids, **kwargs):
        return " ".join(str(value) for value in token_ids)


@pytest.fixture(scope="module")
def agents():
    torch.manual_seed(17)
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        use_cache=True,
    )
    model = transformers.Qwen2ForCausalLM(config).eval()
    backend = HuggingFaceModelBackend(
        "tiny/qwen2-handoff",
        device="cpu",
        tokenizer=_Tokenizer(),
        model=model,
        torch_module=torch,
    )
    return LLMAgent("agent_a", backend), LLMAgent("agent_b", backend)


class _BytesOnlySpy(BaseByteTransport):
    def __init__(self):
        self.inputs = []

    def transfer(self, data: bytes, *, source: str, target: str) -> bytes:
        assert type(data) is bytes
        self.inputs.append((data, source, target))
        return bytes(data)


def _networks():
    return (
        SimulatedNetwork(20.0, base_latency_ms=5.0),
        SimulatedNetwork(1000.0, base_latency_ms=5.0),
    )


def _handoff(agents, *, ids=(1, 2, 3, 4), transport=None):
    source, target = agents
    return perform_kv_handoff(
        source=source,
        target=target,
        span=CanonicalHandoffSpan(tuple(ids)),
        receiver_suffix_token_ids=(5, 6),
        transport=transport or InMemoryByteTransport(),
        networks=_networks(),
        receiver_max_new_tokens=2,
    )


def test_kv_handoff_crosses_real_bytes_transport_and_restores_complete_cache(agents):
    transport = _BytesOnlySpy()
    result = _handoff(agents, transport=transport)

    assert len(transport.inputs) == 1
    packet, source, target = transport.inputs[0]
    assert packet[:4] == b"TKVK"
    assert (source, target) == ("agent_a", "agent_b")
    assert result.wire_total_bytes == result.received_byte_count == len(packet)
    assert result.wire_tensor_body_bytes == result.raw_kv_tensor_bytes
    assert result.wire_metadata_bytes + result.wire_tensor_body_bytes == len(packet)
    assert result.logical_cached_tokens == result.token_count == 4
    assert result.canonical_cache_complete
    assert result.compatibility_verified
    assert result.wire_identity_verified
    assert result.receiver_prefix_prefill_executed is False
    assert result.materialization_forward_ms is None
    assert result.restored_cache_length_after_suffix == 6


def test_kv_receiver_does_not_execute_handoff_prefix_prefill(agents, monkeypatch):
    source, target = agents
    backend = source.backend
    original_prefill = backend.prefill
    calls = []

    def counted_prefill(*args, **kwargs):
        calls.append(tuple(kwargs.get("input_token_ids", ())))
        return original_prefill(*args, **kwargs)

    monkeypatch.setattr(backend, "prefill", counted_prefill)
    result = _handoff(agents)
    assert calls == [result.handoff_token_ids]
    assert result.receiver_prefix_prefill_executed is False


def test_handoff_stage_metrics_and_derived_cost_do_not_double_count(agents):
    result = _handoff(agents)
    assert min(
        result.sender_kv_prepare_ms,
        result.device_to_host_ms,
        result.protocol_encode_ms,
        result.total_serialize_ms,
        result.local_transfer_ms,
        result.deserialize_ms,
        result.cpu_tensor_reconstruction_ms,
        result.host_to_device_ms,
        result.cache_construction_ms,
        result.total_restore_ms,
    ) >= 0
    assert result.total_serialize_ms >= result.device_to_host_ms + result.protocol_encode_ms
    assert result.total_restore_ms >= (
        result.cpu_tensor_reconstruction_ms
        + result.host_to_device_ms
        + result.cache_construction_ms
    )
    for link in result.simulated_links:
        expected_ready = (
            result.total_serialize_ms
            + link.simulated_link_latency_ms
            + result.deserialize_ms
            + result.total_restore_ms
        )
        assert link.kv_handoff_cost_ms == pytest.approx(expected_ready)
        assert link.kv_end_to_end_representation_cost_ms == pytest.approx(
            result.sender_kv_prepare_ms + expected_ready
        )
        assert result.local_transfer_ms not in (
            link.kv_handoff_cost_ms,
            link.kv_end_to_end_representation_cost_ms,
        )


def test_simulated_link_uses_real_packet_and_low_bandwidth_costs_more(agents):
    result = _handoff(agents)
    low, high = result.simulated_links
    assert low.bandwidth_mbps == 20.0
    assert high.bandwidth_mbps == 1000.0
    assert low.simulated_link_latency_ms == pytest.approx(
        5.0 + result.wire_total_bytes * 8 / 20_000_000 * 1000
    )
    assert low.simulated_link_latency_ms > high.simulated_link_latency_ms


def test_canonical_token_kv_and_full_recompute_oracle_match(agents):
    source, target = agents
    span = CanonicalHandoffSpan((1, 2, 3, 4))
    result = compare_canonical_token_kv(
        source=source,
        target=target,
        span=span,
        receiver_suffix_token_ids=(5, 6),
        transport=InMemoryByteTransport(),
    )
    assert result.handoff_token_ids == span.handoff_token_ids
    assert result.all_next_token_ids_equal
    assert result.token_vs_kv_close
    assert result.kv_vs_full_close
    assert result.kv_receiver_prefix_prefill_executed is False


def test_bidirectional_session_switches_only_after_success(agents):
    source, target = agents
    session = BidirectionalKVHandoffSession(
        source, target, InMemoryByteTransport(), _networks()
    )
    assert session.next_direction == ("agent_a", "agent_b")
    first = session.handoff_next(
        (1, 2, 3, 4), receiver_suffix_token_ids=(5,), receiver_max_new_tokens=1
    )
    assert (first.source_agent_id, first.target_agent_id) == ("agent_a", "agent_b")
    assert session.next_direction == ("agent_b", "agent_a")
    second = session.handoff_next(
        first.receiver_output_token_ids,
        receiver_suffix_token_ids=(6,),
        receiver_max_new_tokens=1,
    )
    assert (second.source_agent_id, second.target_agent_id) == ("agent_b", "agent_a")
    assert session.next_direction == ("agent_a", "agent_b")

    class FailingTransport(BaseByteTransport):
        def transfer(self, data: bytes, *, source: str, target: str) -> bytes:
            raise RuntimeError("link failed")

    failed = BidirectionalKVHandoffSession(source, target, FailingTransport(), _networks())
    with pytest.raises(RuntimeError, match="link failed"):
        failed.handoff_next((1, 2), receiver_suffix_token_ids=(3,))
    assert failed.next_direction == ("agent_a", "agent_b")


def test_result_is_json_serializable_and_retains_no_runtime_payload(agents):
    result = _handoff(agents)
    encoded = result.to_json()
    assert json.loads(encoded)["wire_total_bytes"] == result.wire_total_bytes
    field_names = {field.name for field in fields(KVHandoffResult)}
    assert not {"packet", "payload", "cache", "tensor"} & field_names
    assert not any(isinstance(value, (bytes, bytearray, memoryview)) for value in result.__dict__.values())
    assert not any(torch.is_tensor(value) for value in result.__dict__.values())


def test_stage_listener_receives_metadata_only_in_data_plane_order(agents):
    source, target = agents
    stages = []
    result = perform_kv_handoff(
        source=source,
        target=target,
        span=CanonicalHandoffSpan((1, 2, 3, 4)),
        receiver_suffix_token_ids=(5, 6),
        transport=InMemoryByteTransport(),
        networks=_networks(),
        receiver_max_new_tokens=2,
        stage_listener=lambda name, metadata: stages.append((name, dict(metadata))),
    )

    assert [name for name, _ in stages] == [
        "prepared", "serialized", "transferred", "deserialized", "restored"
    ]
    assert stages[1][1]["wire_total_bytes"] == result.wire_total_bytes
    assert all(
        not isinstance(value, (bytes, bytearray, memoryview))
        for _, metadata in stages
        for value in metadata.values()
    )
    assert all("packet" not in metadata and "token_ids" not in metadata for _, metadata in stages)


def test_scaling_uses_multiple_canonical_lengths_and_real_wire_sizes(agents):
    source, target = agents
    measurements = measure_kv_scaling(
        source=source,
        target=target,
        spans=(
            CanonicalHandoffSpan(tuple(range(1, 9))),
            CanonicalHandoffSpan(tuple(range(1, 17))),
            CanonicalHandoffSpan(tuple(range(1, 33))),
        ),
        receiver_suffix_token_ids=(5,),
        transport=InMemoryByteTransport(),
        networks=_networks(),
        measurement_runs=2,
    )
    assert [entry.token_count for entry in measurements] == [8, 16, 32]
    assert [entry.raw_kv_tensor_bytes for entry in measurements] == sorted(
        entry.raw_kv_tensor_bytes for entry in measurements
    )
    assert [entry.wire_total_bytes for entry in measurements] == sorted(
        entry.wire_total_bytes for entry in measurements
    )
    assert all(entry.links[0].simulated_link_latency_ms > entry.links[1].simulated_link_latency_ms for entry in measurements)
