from __future__ import annotations

from threading import Event, Thread

from fastapi.testclient import TestClient
import pytest

from token_kv_agent_system.dashboard import (
    InteractiveRuntimeBusyError,
    RealTokenRuntime,
    create_app,
)
from token_kv_agent_system.metrics import EventType
from token_kv_agent_system.model import (
    DecodeStepResult,
    GenerationResult,
    IncrementalCausalLMBackend,
    ModelMetadata,
    PrefillResult,
    TokenIdGenerationResult,
)


class FakeRealTokenBackend(IncrementalCausalLMBackend):
    def __init__(self) -> None:
        self.generate_calls = 0
        self.prefill_inputs: list[tuple[int, ...]] = []
        self._metadata = ModelMetadata(
            model_name="fake/real-token",
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

    def encode(self, text):
        return (1, 2)

    def decode(self, token_ids, *, skip_special_tokens=False):
        return " ".join(map(str, token_ids))

    def generate(self, prompt, config=None):
        self.generate_calls += 1
        return GenerationResult(
            prompt=prompt,
            input_token_ids=(1, 2),
            generated_text="sender real text",
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
        prefill = self.prefill(input_token_ids=input_token_ids)
        step = self.decode_step(
            prefill.first_token_id,
            previous_cache=prefill.cache,
            logical_cached_token_count=prefill.logical_cached_token_count,
        )
        return TokenIdGenerationResult(
            input_token_ids=prefill.input_token_ids,
            output_token_ids=(6, 7),
            generated_text="receiver real text",
            prefill=prefill,
            decode_steps=(step,),
            stop_reason="max_new_tokens",
        )


def make_runtime() -> tuple[RealTokenRuntime, FakeRealTokenBackend]:
    backend = FakeRealTokenBackend()
    return RealTokenRuntime.create_with_backend(backend=backend), backend


def make_app(runtime: RealTokenRuntime):
    return create_app(
        collector=runtime.collector,
        initial_network_state=runtime.network.state,
        agent_ids=tuple(agent.agent_id for agent in runtime.agents),
        interactive_runtime=runtime,
        runtime_mode="real_token",
    )


def test_runtime_initializes_two_agents_with_one_shared_backend() -> None:
    runtime, backend = make_runtime()

    assert runtime.backend is backend
    assert runtime.agents[0].backend is runtime.agents[1].backend is backend
    assert runtime.snapshot() == {
        "mode": "real_token",
        "status": "ready",
        "running": False,
        "next_source": "Agent A",
        "next_target": "Agent B",
        "active_communication_mode": "TOKEN",
        "selector_status": "not_applicable",
        "model": "fake/real-token",
        "device": "cpu",
        "max_new_tokens": 8,
        "warmup_performed": False,
        "last_error": None,
    }


def test_real_handoff_alternates_and_projects_single_run_measurements() -> None:
    runtime, backend = make_runtime()

    first_source, first_target, first = runtime.communicate_next()
    second_source, second_target, second = runtime.communicate_next()

    assert (first_source, first_target) == ("Agent A", "Agent B")
    assert (second_source, second_target) == ("Agent B", "Agent A")
    assert backend.generate_calls == 1
    assert backend.prefill_inputs == [(3, 4, 5), (6, 7)]
    assert first.handoff.sender_generated_text == "sender real text"
    assert first.handoff.receiver_generated_text == "receiver real text"
    assert second.handoff.sender_output_token_ids == (6, 7)
    assert first.handoff.wire_byte_count == 22
    assert first.handoff.token_identity_preserved is True
    assert first.simulated_link_latency_ms == pytest.approx(5.0088)
    assert first.token_handoff_cost_ms == pytest.approx(
        first.handoff.serialization_ms
        + first.simulated_link_latency_ms
        + first.handoff.deserialization_ms
        + 7.5
    )
    assert runtime.snapshot()["next_source"] == "Agent A"
    assert runtime.collector.events_of_type(EventType.MODE_SELECTED) == ()


def test_network_control_uses_the_same_real_runtime_network() -> None:
    runtime, _ = make_runtime()
    client = TestClient(make_app(runtime))

    response = client.patch("/api/control/network", json={"bandwidth_mbps": 1000})
    result = client.post("/api/control/communication")

    assert response.status_code == 200
    assert runtime.network.state.bandwidth_mbps == 1000
    assert result.status_code == 200
    assert result.json()["communication_mode"] == "TOKEN"
    assert result.json()["selected_mode"] is None
    assert result.json()["real_token_metrics"]["simulated_link_ms"] == pytest.approx(5.000176)


def test_real_events_project_state_and_never_expose_packet_or_token_ids() -> None:
    runtime, _ = make_runtime()
    client = TestClient(make_app(runtime))

    response = client.post("/api/control/communication")
    state = client.get("/api/state").json()
    events = client.get("/api/events?limit=50").json()["events"]

    assert response.status_code == 200
    assert state["runtime"]["active_communication_mode"] == "TOKEN"
    assert state["runtime"]["selector_status"] == "not_applicable"
    assert state["last_selection"] is None
    assert state["real_token_metrics"]["wire_bytes"] == 22
    assert state["real_token_metrics"]["receiver_prefill_ms"] == 7.5
    assert state["real_token_metrics"]["sender_generated_text"] == "sender real text"
    assert state["real_token_metrics"]["receiver_generated_text"] == "receiver real text"
    assert state["real_token_metrics"]["token_identity_verified"] is True
    assert any(event["type"] == "REAL_TOKEN_HANDOFF_FINISHED" for event in events)
    assert all(not isinstance(value, (bytes, bytearray)) for event in events for value in event["metadata"].values())
    assert all("token_ids" not in event["metadata"] for event in events)
    assert all("packet" not in event["metadata"] for event in events)


def test_real_handoff_stages_stream_over_websocket_without_mode_selected() -> None:
    runtime, _ = make_runtime()
    client = TestClient(make_app(runtime))

    with client.websocket_connect("/ws/events") as websocket:
        request = Thread(target=lambda: client.post("/api/control/communication"))
        request.start()
        events = [websocket.receive_json() for _ in range(8)]
        request.join(timeout=2)

    assert not request.is_alive()
    assert events[0]["type"] == "REAL_TOKEN_HANDOFF_STARTED"
    assert events[-1]["type"] == "REAL_TOKEN_HANDOFF_FINISHED"
    assert "MODE_SELECTED" not in [event["type"] for event in events]
    assert all("packet" not in event["metadata"] for event in events)
    assert all("token_ids" not in event["metadata"] for event in events)


def test_failure_does_not_switch_direction(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, _ = make_runtime()
    before = runtime.snapshot()

    def fail(*args, **kwargs):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(runtime.session, "handoff_token_ids", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        runtime.communicate_next()

    after = runtime.snapshot()
    assert (after["next_source"], after["next_target"]) == (
        before["next_source"], before["next_target"]
    )
    assert after["status"] == "error"


def test_running_lock_rejects_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, _ = make_runtime()
    started = Event()
    release = Event()
    original = runtime.session.handoff_token_ids

    def block(*args, **kwargs):
        started.set()
        assert release.wait(timeout=2)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime.session, "handoff_token_ids", block)
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


def test_missing_llm_dependencies_have_real_runtime_specific_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import token_kv_agent_system.dashboard.runtime as runtime_module

    def missing(*args, **kwargs):
        raise RuntimeError("Local LLM support is not installed; run pip install -e '.[llm]'")

    monkeypatch.setattr(runtime_module, "HuggingFaceModelBackend", missing)
    with pytest.raises(RuntimeError, match="Real Token runtime requires the 'llm'"):
        RealTokenRuntime.create_default(warmup=False)
