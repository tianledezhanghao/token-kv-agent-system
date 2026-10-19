from __future__ import annotations

from threading import Event, Thread

from fastapi.testclient import TestClient
import pytest

from token_kv_agent_system.dashboard import (
    create_app,
    create_integrated_demo_app,
    create_interactive_app,
)
from token_kv_agent_system.demo import run_demo
from token_kv_agent_system.metrics import CommunicationEvent, EventType, MetricCollector
from token_kv_agent_system.models import CommunicationMode, NetworkState


def make_client() -> tuple[TestClient, MetricCollector]:
    collector = MetricCollector()
    app = create_app(
        collector=collector,
        initial_network_state=NetworkState(20, 5),
        agent_ids=("Agent A", "Agent B"),
    )
    return TestClient(app), collector


def test_health_and_initial_state() -> None:
    client, _ = make_client()

    assert client.get("/api/health").json() == {"status": "ok"}
    response = client.get("/api/state")

    assert response.status_code == 200
    assert response.json() == {
        "system_status": "idle",
        "current_task": None,
        "network": {"bandwidth_mbps": 20.0, "base_latency_ms": 5.0},
        "agents": {
            "Agent A": {"agent_id": "Agent A", "status": "idle"},
            "Agent B": {"agent_id": "Agent B", "status": "idle"},
        },
        "last_selection": None,
        "last_communication": None,
        "real_token_metrics": None,
        "real_kv_metrics": None,
        "recent_events": [],
        "runtime": {
            "mode": "observer",
            "running": False,
            "next_source": None,
            "next_target": None,
        },
    }


def test_existing_collector_events_are_projected_when_app_is_attached() -> None:
    collector = MetricCollector()
    collector.record(
        CommunicationEvent(
            event_type=EventType.NETWORK_STATE_CHANGED,
            metadata={"bandwidth_mbps": 80, "base_latency_ms": 3},
        )
    )

    client = TestClient(create_app(collector=collector))

    assert client.get("/api/state").json()["network"] == {
        "bandwidth_mbps": 80,
        "base_latency_ms": 3,
    }
    assert client.get("/api/events").json()["count"] == 1


def test_state_is_updated_by_real_demo_event_flow() -> None:
    client, collector = make_client()

    results, returned_collector = run_demo(lambda _: None, collector=collector)
    state = client.get("/api/state").json()

    assert returned_collector is collector
    assert [result.decision.mode for result in results] == [
        CommunicationMode.TOKEN,
        CommunicationMode.KV,
    ]
    assert state["system_status"] == "idle"
    assert state["network"] == {
        "bandwidth_mbps": 1000,
        "base_latency_ms": 5,
    }
    assert state["last_selection"] == {
        "source": "Agent B",
        "target": "Agent A",
        "mode": "KV",
        "estimated_token_latency_ms": 25.008,
        "estimated_kv_latency_ms": 13.0,
    }
    assert state["last_communication"] == {
        "source": "Agent B",
        "target": "Agent A",
        "mode": "KV",
        "payload_bytes": 1_000_000,
        "transmission_latency_ms": 13.0,
        "estimated_e2e_latency_ms": 13.0,
    }
    assert all(agent["status"] == "idle" for agent in state["agents"].values())
    assert [event["type"] for event in state["recent_events"]].count(
        "MODE_SELECTED"
    ) == 2


def test_network_selection_and_communication_projection() -> None:
    client, collector = make_client()
    collector.record(
        CommunicationEvent(
            event_type=EventType.NETWORK_STATE_CHANGED,
            metadata={"bandwidth_mbps": 1000, "base_latency_ms": 7},
        )
    )
    collector.record(
        CommunicationEvent(
            event_type=EventType.MODE_SELECTED,
            source_agent_id="A",
            target_agent_id="B",
            mode=CommunicationMode.TOKEN,
            metadata={
                "estimated_token_latency_ms": 11.5,
                "estimated_kv_latency_ms": 22.5,
            },
        )
    )
    collector.record(
        CommunicationEvent(
            event_type=EventType.TRANSMISSION_FINISHED,
            source_agent_id="A",
            target_agent_id="B",
            mode=CommunicationMode.TOKEN,
            payload_bytes=128,
            transmission_latency_ms=7.1,
            estimated_e2e_latency_ms=11.5,
        )
    )

    state = client.get("/api/state").json()

    assert state["network"] == {
        "bandwidth_mbps": 1000,
        "base_latency_ms": 7,
    }
    assert state["last_selection"]["mode"] == "TOKEN"
    assert state["last_selection"]["estimated_token_latency_ms"] == 11.5
    assert state["last_communication"]["payload_bytes"] == 128
    assert state["last_communication"]["estimated_e2e_latency_ms"] == 11.5


def test_events_limit_and_validation() -> None:
    client, collector = make_client()
    for index in range(4):
        collector.record(
            CommunicationEvent(
                event_type=EventType.AGENT_GENERATION_STARTED,
                source_agent_id=f"agent-{index}",
            )
        )

    response = client.get("/api/events?limit=2")

    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert [item["source"] for item in response.json()["events"]] == [
        "agent-2",
        "agent-3",
    ]
    assert client.get("/api/events?limit=0").status_code == 422
    assert client.get("/api/events?limit=201").status_code == 422
    assert client.get("/api/events?limit=not-an-int").status_code == 422


def test_websocket_receives_new_event_and_disconnect_isolated() -> None:
    client, collector = make_client()
    event = CommunicationEvent(
        event_type=EventType.MODE_SELECTED,
        source_agent_id="Agent A",
        target_agent_id="Agent B",
        mode=CommunicationMode.TOKEN,
        metadata={
            "estimated_token_latency_ms": 25.4,
            "estimated_kv_latency_ms": 405.0,
        },
    )

    with client.websocket_connect("/ws/events") as websocket:
        collector.record(event)
        received = websocket.receive_json()

    assert received["type"] == "MODE_SELECTED"
    assert received["source"] == "Agent A"
    assert received["mode"] == "TOKEN"

    # A closed client and then no clients must not make event recording fail.
    collector.record(
        CommunicationEvent(event_type=EventType.AGENT_GENERATION_FINISHED)
    )
    assert client.get("/api/events?limit=1").json()["events"][0]["type"] == (
        "AGENT_GENERATION_FINISHED"
    )


def test_integrated_demo_streams_its_real_event_flow_over_websocket() -> None:
    app = create_integrated_demo_app(demo_delay_seconds=0.05)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/events") as websocket:
            events = [websocket.receive_json() for _ in range(11)]

        assert [event["mode"] for event in events if event["type"] == "MODE_SELECTED"] == [
            "TOKEN",
            "KV",
        ]
        assert [
            (event["source"], event["target"])
            for event in events
            if event["type"] == "TRANSMISSION_FINISHED"
        ] == [("Agent A", "Agent B"), ("Agent B", "Agent A")]
        assert client.get("/api/state").json()["network"]["bandwidth_mbps"] == 1000


def test_two_websocket_clients_receive_the_same_event() -> None:
    client, collector = make_client()
    event = CommunicationEvent(event_type=EventType.NETWORK_STATE_CHANGED)

    with client.websocket_connect("/ws/events") as first:
        with client.websocket_connect("/ws/events") as second:
            collector.record(event)
            assert first.receive_json()["type"] == "NETWORK_STATE_CHANGED"
            assert second.receive_json()["type"] == "NETWORK_STATE_CHANGED"


def test_dashboard_serialization_contains_metadata_not_payload_objects() -> None:
    client, collector = make_client()
    _, collector = run_demo(lambda _: None, collector=collector)

    events = client.get("/api/events?limit=50").json()["events"]

    assert events
    assert all("token_payload" not in event for event in events)
    assert all("kv_payload" not in event for event in events)
    assert all("token_ids" not in event["metadata"] for event in events)
    assert all("cache_key" not in event["metadata"] for event in events)


def test_interactive_control_updates_real_network_and_dashboard_event() -> None:
    app = create_interactive_app()
    client = TestClient(app)

    response = client.patch("/api/control/network", json={"bandwidth_mbps": 1000})

    assert response.status_code == 200
    assert app.state.interactive_runtime.network.state.bandwidth_mbps == 1000
    state = client.get("/api/state").json()
    assert state["network"]["bandwidth_mbps"] == 1000
    assert state["recent_events"][-1]["type"] == "NETWORK_STATE_CHANGED"
    assert state["runtime"] == {
        "mode": "interactive",
        "running": False,
        "next_source": "Agent A",
        "next_target": "Agent B",
    }


def test_network_control_validation_and_mode_field_rejection() -> None:
    client = TestClient(create_interactive_app())

    assert client.patch(
        "/api/control/network", json={"bandwidth_mbps": 0}
    ).status_code == 422
    assert client.patch(
        "/api/control/network", json={"bandwidth_mbps": "not-a-number"}
    ).status_code == 422
    assert client.patch(
        "/api/control/network", json={"bandwidth_mbps": 20, "mode": "KV"}
    ).status_code == 422
    assert client.post(
        "/api/control/communication", json={"mode": "KV"}
    ).status_code == 422
    assert client.get("/api/state").json()["last_communication"] is None


def test_interactive_communication_uses_selector_and_alternates_direction() -> None:
    app = create_interactive_app()
    client = TestClient(app)

    low = client.post("/api/control/communication")
    client.patch("/api/control/network", json={"bandwidth_mbps": 1000})
    high = client.post("/api/control/communication")

    assert low.status_code == 200
    assert (low.json()["source"], low.json()["target"]) == ("Agent A", "Agent B")
    assert low.json()["selected_mode"] == "TOKEN"
    assert high.status_code == 200
    assert (high.json()["source"], high.json()["target"]) == ("Agent B", "Agent A")
    assert high.json()["selected_mode"] == "KV"
    assert client.get("/api/state").json()["runtime"] == {
        "mode": "interactive",
        "running": False,
        "next_source": "Agent A",
        "next_target": "Agent B",
    }
    assert len(
        app.state.collector.events_of_type(EventType.TRANSMISSION_FINISHED)
    ) == 2


def test_control_endpoints_require_interactive_mode() -> None:
    client = TestClient(create_integrated_demo_app(demo_delay_seconds=60))

    network = client.patch("/api/control/network", json={"bandwidth_mbps": 20})
    communication = client.post("/api/control/communication")

    assert network.status_code == 409
    assert communication.status_code == 409
    assert network.json()["detail"] == "interactive mode is required for control operations"


def test_communication_endpoint_returns_conflict_while_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_interactive_app()
    client = TestClient(app)
    runtime = app.state.interactive_runtime
    started = Event()
    release = Event()
    original_communicate = runtime.orchestrator.communicate

    def blocking_communicate(*args: object, **kwargs: object):
        started.set()
        assert release.wait(timeout=2)
        return original_communicate(*args, **kwargs)

    monkeypatch.setattr(runtime.orchestrator, "communicate", blocking_communicate)
    first = Thread(target=lambda: client.post("/api/control/communication"))
    first.start()
    assert started.wait(timeout=2)

    response = client.post("/api/control/communication")

    assert response.status_code == 409
    assert response.json()["detail"] == "communication is already running"
    assert client.get("/api/state").json()["runtime"]["running"] is True
    release.set()
    first.join(timeout=2)
    assert not first.is_alive()
