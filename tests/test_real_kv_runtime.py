from __future__ import annotations

from dataclasses import replace
from threading import Event, Thread

from fastapi.testclient import TestClient
import pytest

from token_kv_agent_system.agent import LLMAgent
from token_kv_agent_system.dashboard import (
    InteractiveRuntimeBusyError,
    RealKVRuntime,
    create_app,
)
from token_kv_agent_system.kv_handoff import KVHandoffResult, KVSimulatedLinkCost
from token_kv_agent_system.metrics import EventType, MetricCollector
from token_kv_agent_system.model import (
    DecodeStepResult,
    GenerationResult,
    IncrementalCausalLMBackend,
    ModelMetadata,
    PrefillResult,
)
from token_kv_agent_system.network import SimulatedNetwork
from token_kv_agent_system.transport import InMemoryByteTransport


class FakeRealKVBackend(IncrementalCausalLMBackend):
    def __init__(self) -> None:
        self.generate_calls = 0
        self.prefill_inputs: list[tuple[int, ...]] = []
        self._metadata = ModelMetadata(
            model_name="fake/real-kv",
            device="cpu",
            vocab_size=32,
            eos_token_id=None,
            bos_token_id=None,
            pad_token_id=None,
            load_wall_time_seconds=0.0,
        )

    @property
    def metadata(self):
        return self._metadata

    @property
    def eos_token_ids(self):
        return ()

    @property
    def kv_model_config(self):
        return object()

    def encode(self, text):
        return (9,)

    def decode(self, token_ids, *, skip_special_tokens=False):
        return " ".join(map(str, token_ids))

    def generate(self, prompt, config=None):
        self.generate_calls += 1
        return GenerationResult(
            prompt=prompt,
            input_token_ids=(1, 2),
            generated_text="sender KV text",
            output_token_ids=(3, 4, 5),
            generation_wall_time_seconds=0.002,
            full_token_ids=(1, 2, 3, 4, 5),
        )

    def prefill(self, prompt=None, *, input_token_ids=None):
        ids = tuple(input_token_ids or ())
        self.prefill_inputs.append(ids)
        return PrefillResult(
            input_token_ids=ids,
            first_token_id=6,
            cache=object(),
            prefill_forward_ms=7.5,
            logical_cached_token_count=len(ids),
            logits_shape=(1, len(ids), 32),
            last_token_logits_shape=(32,),
            last_token_logits=object(),
        )

    def decode_step(self, input_token_id, *, previous_cache, logical_cached_token_count):
        return DecodeStepResult(
            input_token_id=input_token_id,
            predicted_next_token_id=7,
            cache=previous_cache,
            decode_forward_ms=1.5,
            logical_cached_token_count=logical_cached_token_count + 1,
        )

    def manual_generate(self, prompt, config=None, *, warmup_runs=1):
        raise NotImplementedError

    def generate_from_token_ids(self, input_token_ids, config=None):
        raise NotImplementedError

    def restore_kv_cache(self, payload):
        raise NotImplementedError

    def continue_from_cache(self, *args, **kwargs):
        raise NotImplementedError


def _result(source: str, target: str, token_ids: tuple[int, ...], network: SimulatedNetwork) -> KVHandoffResult:
    wire_bytes = 100_587
    link_ms = network.estimate_transmission_latency_ms(wire_bytes)
    ready_ms = 7.7 + link_ms + 0.1 + 7.3
    return KVHandoffResult(
        schema_version=1,
        source_agent_id=source,
        target_agent_id=target,
        handoff_token_ids=token_ids,
        receiver_suffix_token_ids=(9,),
        token_count=len(token_ids),
        logical_cached_tokens=len(token_ids),
        raw_kv_tensor_bytes=98_304,
        wire_tensor_body_bytes=98_304,
        wire_metadata_bytes=2_283,
        wire_total_bytes=wire_bytes,
        received_byte_count=wire_bytes,
        sender_kv_prepare_ms=27.2,
        materialization_forward_ms=None,
        device_to_host_ms=6.1,
        protocol_encode_ms=1.6,
        total_serialize_ms=7.7,
        local_transfer_ms=0.002,
        deserialize_ms=0.1,
        cpu_tensor_reconstruction_ms=2.1,
        host_to_device_ms=4.2,
        cache_construction_ms=1.0,
        total_restore_ms=7.3,
        receiver_continuation_forward_ms=14.0,
        receiver_output_token_ids=(6, 7),
        receiver_generated_text="receiver KV continuation",
        receiver_next_token_id=6,
        restored_cache_length_before_continuation=len(token_ids),
        restored_cache_length_after_suffix=len(token_ids) + 1,
        transport_name="InMemoryByteTransport",
        compatibility_verified=True,
        wire_identity_verified=True,
        canonical_cache_complete=True,
        receiver_prefix_prefill_executed=False,
        local_transfer_is_diagnostic_only=True,
        simulated_links=(
            KVSimulatedLinkCost(
                bandwidth_mbps=network.state.bandwidth_mbps,
                base_latency_ms=network.state.base_latency_ms,
                simulated_link_latency_ms=link_ms,
                kv_handoff_cost_ms=ready_ms,
                kv_end_to_end_representation_cost_ms=27.2 + ready_ms,
            ),
        ),
    )


class FakeKVSession:
    def __init__(self, network: SimulatedNetwork) -> None:
        self.network = network
        self.calls: list[tuple[int, ...]] = []
        self.direction = 0

    def handoff_next(self, handoff_token_ids, *, receiver_suffix_token_ids, receiver_max_new_tokens, stage_listener=None, advance_direction=True):
        ids = tuple(handoff_token_ids)
        self.calls.append(ids)
        source, target = (("Agent A", "Agent B"), ("Agent B", "Agent A"))[self.direction]
        if stage_listener:
            stage_listener("prepared", {"token_count": len(ids), "sender_kv_prepare_ms": 27.2})
            stage_listener("serialized", {"wire_total_bytes": 100_587})
            stage_listener("transferred", {"wire_total_bytes": 100_587})
            stage_listener("deserialized", {"deserialize_ms": 0.1})
            stage_listener("restored", {"total_restore_ms": 7.3, "receiver_prefix_prefill_executed": False})
        result = _result(source, target, ids, self.network)
        if advance_direction:
            self.advance_direction()
        return result

    def advance_direction(self):
        self.direction = 1 - self.direction


def make_runtime() -> tuple[RealKVRuntime, FakeRealKVBackend, FakeKVSession]:
    backend = FakeRealKVBackend()
    collector = MetricCollector()
    network = SimulatedNetwork(20, base_latency_ms=5, collector=collector)
    agents = (LLMAgent("Agent A", backend), LLMAgent("Agent B", backend))
    transport = InMemoryByteTransport()
    session = FakeKVSession(network)
    runtime = RealKVRuntime(
        agents=agents,
        backend=backend,
        session=session,  # type: ignore[arg-type]
        transport=transport,
        network=network,
        collector=collector,
        receiver_suffix_token_ids=(9,),
    )
    return runtime, backend, session


def make_app(runtime: RealKVRuntime):
    return create_app(
        collector=runtime.collector,
        initial_network_state=runtime.network.state,
        agent_ids=tuple(agent.agent_id for agent in runtime.agents),
        interactive_runtime=runtime,
        runtime_mode="real_kv",
    )


def test_runtime_initializes_fixed_kv_mode_with_one_shared_backend() -> None:
    runtime, backend, _ = make_runtime()

    assert runtime.backend is backend
    assert runtime.agents[0].backend is runtime.agents[1].backend is backend
    assert runtime.snapshot() == {
        "mode": "real_kv",
        "status": "ready",
        "running": False,
        "next_source": "Agent A",
        "next_target": "Agent B",
        "active_communication_mode": "KV",
        "selector_status": "not_applicable",
        "model": "fake/real-kv",
        "device": "cpu",
        "max_new_tokens": 8,
        "warmup_performed": False,
        "canonical_handoff_semantics": True,
        "last_error": None,
    }


def test_real_kv_handoff_projects_every_cost_and_alternates() -> None:
    runtime, backend, session = make_runtime()

    first = runtime.communicate_next()[2]
    second = runtime.communicate_next()[2]

    assert session.calls == [(3, 4, 5), (6, 7)]
    assert backend.generate_calls == 1
    assert (first.handoff.source_agent_id, first.handoff.target_agent_id) == ("Agent A", "Agent B")
    assert (second.handoff.source_agent_id, second.handoff.target_agent_id) == ("Agent B", "Agent A")
    assert first.handoff.token_count == 3
    assert first.handoff.raw_kv_tensor_bytes == 98_304
    assert first.handoff.wire_total_bytes == 100_587
    assert first.handoff.sender_kv_prepare_ms == 27.2
    assert first.handoff.device_to_host_ms == 6.1
    assert first.handoff.protocol_encode_ms == 1.6
    assert first.handoff.total_serialize_ms == 7.7
    assert first.handoff.deserialize_ms == 0.1
    assert first.handoff.cpu_tensor_reconstruction_ms == 2.1
    assert first.handoff.host_to_device_ms == 4.2
    assert first.handoff.cache_construction_ms == 1.0
    assert first.handoff.total_restore_ms == 7.3
    assert first.link.kv_handoff_cost_ms == pytest.approx(60.3348)
    assert first.link.kv_end_to_end_representation_cost_ms == pytest.approx(87.5348)
    assert first.handoff.receiver_prefix_prefill_executed is False
    assert first.handoff.receiver_generated_text == "receiver KV continuation"
    assert first.continuation_verified is True
    assert runtime.snapshot()["next_source"] == "Agent A"
    assert runtime.collector.events_of_type(EventType.MODE_SELECTED) == ()


def test_network_control_changes_real_link_only_and_state_is_payload_safe() -> None:
    runtime, _, _ = make_runtime()
    client = TestClient(make_app(runtime))

    first = client.post("/api/control/communication").json()["real_kv_metrics"]
    assert client.patch("/api/control/network", json={"bandwidth_mbps": 1000}).status_code == 200
    second = client.post("/api/control/communication").json()["real_kv_metrics"]
    state = client.get("/api/state").json()
    events = client.get("/api/events?limit=50").json()["events"]

    assert first["simulated_link_ms"] == pytest.approx(45.2348)
    assert second["simulated_link_ms"] == pytest.approx(5.804696)
    assert state["runtime"]["active_communication_mode"] == "KV"
    assert state["runtime"]["selector_status"] == "not_applicable"
    assert state["last_selection"] is None
    metrics = state["real_kv_metrics"]
    assert metrics["raw_kv_bytes"] == 98_304
    assert metrics["wire_bytes"] == 100_587
    assert metrics["receiver_prefix_prefill_executed"] is False
    assert metrics["continuation_verified"] is True
    assert "MODE_SELECTED" not in [event["type"] for event in events]
    forbidden = (bytes, bytearray)
    assert all(not isinstance(value, forbidden) for event in events for value in event["metadata"].values())
    assert all("packet" not in event["metadata"] for event in events)
    assert all("token_ids" not in event["metadata"] for event in events)
    assert all("cache" not in key or key in {"canonical_cache_complete", "cache_construction_ms"} for event in events for key in event["metadata"])


def test_kv_stages_stream_over_websocket() -> None:
    runtime, _, _ = make_runtime()
    client = TestClient(make_app(runtime))

    with client.websocket_connect("/ws/events") as websocket:
        request = Thread(target=lambda: client.post("/api/control/communication"))
        request.start()
        events = [websocket.receive_json() for _ in range(9)]
        request.join(timeout=2)

    types = [event["type"] for event in events]
    assert types[0] == "REAL_KV_HANDOFF_STARTED"
    assert types[-1] == "REAL_KV_HANDOFF_FINISHED"
    assert types[3:8] == ["KV_PREPARED", "KV_SERIALIZED", "KV_TRANSFERRED", "KV_DESERIALIZED", "KV_RESTORED"]


def test_failure_does_not_switch_direction_and_running_lock_rejects_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, _, session = make_runtime()
    before = runtime.snapshot()

    def fail(*args, **kwargs):
        raise RuntimeError("KV failed")

    monkeypatch.setattr(session, "handoff_next", fail)
    with pytest.raises(RuntimeError, match="KV failed"):
        runtime.communicate_next()
    assert runtime.snapshot()["next_source"] == before["next_source"]

    runtime, _, session = make_runtime()
    started = Event()
    release = Event()
    original = session.handoff_next

    def block(*args, **kwargs):
        started.set()
        assert release.wait(timeout=2)
        return original(*args, **kwargs)

    monkeypatch.setattr(session, "handoff_next", block)
    thread = Thread(target=runtime.communicate_next)
    thread.start()
    assert started.wait(timeout=2)
    with pytest.raises(InteractiveRuntimeBusyError):
        runtime.communicate_next()
    with pytest.raises(InteractiveRuntimeBusyError):
        runtime.set_bandwidth_mbps(1000)
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_failed_correctness_does_not_switch_direction() -> None:
    runtime, _, session = make_runtime()
    original = session.handoff_next

    def invalid(*args, **kwargs):
        return replace(original(*args, **kwargs), wire_identity_verified=False)

    session.handoff_next = invalid  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="verification failed"):
        runtime.communicate_next()
    assert runtime.snapshot()["next_source"] == "Agent A"


def test_missing_llm_dependencies_have_real_kv_specific_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import token_kv_agent_system.dashboard.runtime as runtime_module

    def missing(*args, **kwargs):
        raise RuntimeError("Local LLM support is not installed; run pip install -e '.[llm]'")

    monkeypatch.setattr(runtime_module, "HuggingFaceModelBackend", missing)
    with pytest.raises(RuntimeError, match="Real KV runtime requires the 'llm'"):
        RealKVRuntime.create_default(warmup=False)
