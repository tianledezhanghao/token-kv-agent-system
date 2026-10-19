"""Dashboard projection for real selection and the single selected execution."""

from fastapi.testclient import TestClient
import pytest

from test_real_adaptive_runtime import Backend, PROFILE, build
from token_kv_agent_system.dashboard import runtime as dashboard_runtime_module
from token_kv_agent_system.dashboard.app import create_app
from token_kv_agent_system.dashboard.runtime import RealAdaptiveDashboardRuntime
from token_kv_agent_system.metrics import EventType, MetricCollector
from token_kv_agent_system.models import CommunicationMode


@pytest.mark.parametrize("mode", [CommunicationMode.TOKEN, CommunicationMode.KV])
def test_adaptive_dashboard_selects_before_one_path_and_projects_actual(mode):
    core, a, b, token, kv, selector = build(mode)
    collector = MetricCollector()
    runtime = RealAdaptiveDashboardRuntime(core, collector)
    app = create_app(collector=collector, initial_network_state=core.network.state,
                     agent_ids=(a.agent_id, b.agent_id), interactive_runtime=runtime)
    client = TestClient(app)

    before = client.get("/api/state").json()
    assert before["runtime"]["mode"] == "real_adaptive"
    assert before["runtime"]["selector_status"] == "active"
    assert before["runtime"]["selector_kind"] == "real_cost_based"
    assert before["runtime"]["active_communication_mode"] is None
    assert before["runtime"]["selector_profile"]["calibration_token_counts"] == [2, 8]

    response = client.post("/api/control/communication").json()
    assert response["selected_mode"] == mode.value
    state = client.get("/api/state").json()
    selection = state["last_selection"]
    execution = state["adaptive_execution"]
    assert selection["token_estimate"]["wire_bytes"] > 0
    assert selection["kv_estimate"]["wire_bytes"] > 0
    assert selection["estimated_saving_ms"] >= 0
    assert selection["selected_mode"] == mode.value
    assert execution["executed_mode"] == mode.value
    assert execution["absolute_error_ms"] >= 0
    assert len(token.calls) == (mode is CommunicationMode.TOKEN)
    assert len(kv.calls) == (mode is CommunicationMode.KV)
    assert state["runtime"]["next_source"] == "B"
    kinds = [event["type"] for event in state["recent_events"]]
    assert kinds.index("AGENT_GENERATION_FINISHED") < kinds.index("REAL_SELECTION_MADE")
    assert kinds.index("REAL_SELECTION_MADE") < kinds.index("SELECTED_PATH_STARTED")
    assert kinds.index("SELECTED_PATH_STARTED") < kinds.index("REAL_ADAPTIVE_HANDOFF_FINISHED")
    for event in state["recent_events"]:
        assert "token_ids" not in event["metadata"]
        assert "packet" not in event["metadata"]
        assert "cache" not in event["metadata"]


def test_network_patch_uses_core_network_and_running_mutation_is_rejected():
    core, a, b, token, kv, selector = build(CommunicationMode.TOKEN)
    collector = MetricCollector()
    runtime = RealAdaptiveDashboardRuntime(core, collector)
    app = create_app(collector=collector, initial_network_state=core.network.state,
                     agent_ids=(a.agent_id, b.agent_id), interactive_runtime=runtime)
    client = TestClient(app)
    assert client.patch("/api/control/network", json={"bandwidth_mbps": 1000}).status_code == 200
    assert core.network.state.bandwidth_mbps == 1000
    original_select = selector.select

    def select(context, *, boundary):
        assert context.bandwidth_mbps == 1000
        assert client.patch("/api/control/network", json={"bandwidth_mbps": 20}).status_code == 409
        return original_select(context, boundary=boundary)

    selector.select = select
    assert client.post("/api/control/communication").status_code == 200
    assert core.network.state.bandwidth_mbps == 1000


def test_failure_does_not_advance_direction():
    core, a, b, token, kv, selector = build(CommunicationMode.TOKEN)
    token.fail = True
    collector = MetricCollector()
    runtime = RealAdaptiveDashboardRuntime(core, collector)
    app = create_app(collector=collector, initial_network_state=core.network.state,
                     agent_ids=(a.agent_id, b.agent_id), interactive_runtime=runtime)
    client = TestClient(app)
    assert client.post("/api/control/communication").status_code == 500
    assert client.get("/api/state").json()["runtime"]["next_source"] == "A"
    assert collector.events_of_type(EventType.REAL_ADAPTIVE_HANDOFF_FAILED)


def test_default_composition_loads_one_backend_and_rejects_incompatible_profile(tmp_path, monkeypatch):
    loaded = []

    def backend_factory(*args, **kwargs):
        backend = Backend()
        loaded.append(backend)
        return backend

    monkeypatch.setattr(dashboard_runtime_module, "HuggingFaceModelBackend", backend_factory)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(PROFILE.to_json(), encoding="utf-8")
    dashboard = RealAdaptiveDashboardRuntime.create_default(selector_profile=str(profile_path), device="mps")
    assert len(loaded) == 1
    assert dashboard.agents[0].backend is dashboard.agents[1].backend is loaded[0]
    assert dashboard.runtime.profile.profile_name == PROFILE.profile_name

    incompatible = PROFILE.to_dict()
    incompatible["model_id"] = "other/model"
    from token_kv_agent_system.real_cost import RealCommunicationCostProfile
    profile_path.write_text(RealCommunicationCostProfile.from_dict(incompatible).to_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        RealAdaptiveDashboardRuntime.create_default(selector_profile=str(profile_path), device="mps")
