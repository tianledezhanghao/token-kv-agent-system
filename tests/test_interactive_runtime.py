from __future__ import annotations

from threading import Event, Thread

import pytest

from token_kv_agent_system.dashboard import (
    InteractiveRuntime,
    InteractiveRuntimeBusyError,
)
from token_kv_agent_system.metrics import EventType
from token_kv_agent_system.models import CommunicationMode


def test_interactive_runtime_initializes_real_components() -> None:
    runtime = InteractiveRuntime.create_default()

    assert runtime.network.state.bandwidth_mbps == 20
    assert [agent.agent_id for agent in runtime.agents] == ["Agent A", "Agent B"]
    assert runtime.snapshot()["next_source"] == "Agent A"


def test_bandwidth_update_changes_network_and_records_event() -> None:
    runtime = InteractiveRuntime.create_default()

    runtime.set_bandwidth_mbps(1000)

    assert runtime.network.state.bandwidth_mbps == 1000
    event = runtime.collector.events_of_type(EventType.NETWORK_STATE_CHANGED)[-1]
    assert event.metadata["previous_bandwidth_mbps"] == 20
    assert event.metadata["bandwidth_mbps"] == 1000


@pytest.mark.parametrize("bandwidth", [0, -1])
def test_bandwidth_update_rejects_non_positive_values(bandwidth: float) -> None:
    runtime = InteractiveRuntime.create_default()

    with pytest.raises(ValueError, match="greater than zero"):
        runtime.set_bandwidth_mbps(bandwidth)


def test_communication_selects_modes_and_alternates() -> None:
    runtime = InteractiveRuntime.create_default()

    first_source, first_target, low_result = runtime.communicate_next()
    runtime.set_bandwidth_mbps(1000)
    second_source, second_target, high_result = runtime.communicate_next()

    assert (first_source, first_target, low_result.decision.mode) == (
        "Agent A",
        "Agent B",
        CommunicationMode.TOKEN,
    )
    assert (second_source, second_target, high_result.decision.mode) == (
        "Agent B",
        "Agent A",
        CommunicationMode.KV,
    )
    assert len(runtime.collector.events_of_type(EventType.MODE_SELECTED)) == 2


def test_overlapping_communication_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = InteractiveRuntime.create_default()
    started = Event()
    release = Event()
    original_communicate = runtime.orchestrator.communicate

    def blocking_communicate(*args: object, **kwargs: object):
        started.set()
        assert release.wait(timeout=2)
        return original_communicate(*args, **kwargs)

    monkeypatch.setattr(runtime.orchestrator, "communicate", blocking_communicate)
    thread = Thread(target=runtime.communicate_next)
    thread.start()
    assert started.wait(timeout=2)

    assert runtime.snapshot()["running"] is True
    with pytest.raises(InteractiveRuntimeBusyError, match="already running"):
        runtime.communicate_next()
    with pytest.raises(InteractiveRuntimeBusyError, match="already running"):
        runtime.set_bandwidth_mbps(1000)

    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert runtime.snapshot()["running"] is False
