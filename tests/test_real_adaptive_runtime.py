import json
from types import SimpleNamespace

import pytest

from token_kv_agent_system.models import CommunicationMode
from token_kv_agent_system.network import SimulatedNetwork
from token_kv_agent_system.real_adaptive_runtime import RealAdaptiveRuntime, RealAdaptiveRuntimeBusyError
from token_kv_agent_system.real_cost import (MESSAGE_READY, CalibrationPoint,
    RealCommunicationCostModel, RealCommunicationCostProfile, RealCostBasedSelector, RealSelectionContext)
from token_kv_agent_system.selection_stability import SelectionStabilityConfig, StableRealCostSelector
from token_kv_agent_system.synthetic_selection import synthetic_crossover_profile


PROFILE = RealCommunicationCostProfile(
    1, "test/model", 24, 2, 64, "bfloat16", 1, "mps", "test", "mps:0",
    1, 1, "test", 1, 2,
    (CalibrationPoint(2, 1, 1, 4, 4, 3, 2, 1),
     CalibrationPoint(8, 1, 1, 5, 5, 3, 2, 1)))


class Backend:
    metadata = SimpleNamespace(model_name=PROFILE.model_id, device=PROFILE.source_device,
                               dtype="torch.bfloat16")
    kv_model_config = SimpleNamespace(num_hidden_layers=PROFILE.num_hidden_layers,
        num_key_value_heads=PROFILE.num_key_value_heads, head_dim=PROFILE.head_dim)


class Agent:
    def __init__(self, name, backend):
        self.agent_id = name
        self.backend = backend
        self.calls = 0

    def generate_reply(self, prompt, config):
        self.calls += 1
        return SimpleNamespace(output_token_ids=(1, 2, 3), generated_text="sender",
                               generation_wall_time_seconds=.02)


class TokenPath:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.correct = True

    def handoff_token_ids(self, source, target, ids, **kwargs):
        self.calls.append((source, target, ids))
        if self.fail:
            raise RuntimeError("token failed")
        return SimpleNamespace(token_identity_preserved=self.correct,
            receiver_output_token_ids=(4, 5), receiver_generated_text="receiver",
            wire_byte_count=22, serialization_ms=1., deserialization_ms=2.,
            receiver_prefill_forward_ms=3.)


class KVPath:
    def __init__(self):
        self.calls = []
        self.fail = False

    def __call__(self, source, target, ids, network):
        self.calls.append((source.agent_id, target.agent_id, ids, network))
        if self.fail:
            raise RuntimeError("kv failed")
        return SimpleNamespace(handoff_token_ids=ids, compatibility_verified=True,
            wire_identity_verified=True, canonical_cache_complete=True,
            receiver_prefix_prefill_executed=False, receiver_output_token_ids=(4, 5),
            receiver_generated_text="receiver", raw_kv_tensor_bytes=100,
            wire_total_bytes=120, sender_kv_prepare_ms=4., total_serialize_ms=3.,
            deserialize_ms=2., total_restore_ms=1.)


class Selector:
    def __init__(self, mode, token, kv, network):
        self.mode, self.token, self.kv, self.network = mode, token, kv, network
        self.contexts = []
        self.fail = False

    def select(self, context, *, boundary):
        assert boundary == MESSAGE_READY
        assert not self.token.calls and not self.kv.calls
        assert self.network.state.bandwidth_mbps == context.bandwidth_mbps
        self.contexts.append(context)
        if self.fail:
            raise RuntimeError("selector failed")
        real = RealCostBasedSelector(RealCommunicationCostModel(PROFILE)).select(context)
        return SimpleNamespace(selected_mode=self.mode, cost_boundary=MESSAGE_READY,
            token_estimate=real.token_estimate, kv_estimate=real.kv_estimate,
            estimated_saving_ms=real.estimated_saving_ms,
            estimated_saving_ratio=real.estimated_saving_ratio)


def build(mode):
    backend = Backend()
    a, b = Agent("A", backend), Agent("B", backend)
    token, kv = TokenPath(), KVPath()
    network = SimulatedNetwork(20, 5)
    selector = Selector(mode, token, kv, network)
    runtime = RealAdaptiveRuntime(a, b, PROFILE, network, selector=selector,
                                  token_handoff=token, kv_handoff=kv)
    return runtime, a, b, token, kv, selector


@pytest.mark.parametrize("mode", [CommunicationMode.TOKEN, CommunicationMode.KV])
def test_route_once_before_transfer_and_message_ready_cost(mode):
    runtime, a, b, token, kv, selector = build(mode)
    result = runtime.communicate_next()
    assert a.calls == 1 and b.calls == 0
    assert selector.contexts[0].handoff_token_count == result.canonical_token_count == 3
    assert len(token.calls) == (mode == CommunicationMode.TOKEN)
    assert len(kv.calls) == (mode == CommunicationMode.KV)
    assert (token.calls[0][2] if token.calls else kv.calls[0][2]) == (1, 2, 3)
    link = runtime.network.estimate_transmission_latency_ms(22 if token.calls else 120)
    assert result.actual_selected_cost_ms == pytest.approx(6 + link if token.calls else 10 + link)
    assert result.absolute_error_ms == pytest.approx(abs(result.actual_selected_cost_ms - result.predicted_selected_cost_ms))
    assert result.sender_generation_ms == 20
    assert "packet" not in result.to_json().lower()
    assert "token_ids" not in result.to_json()
    json.dumps(result.to_dict())
    assert runtime.next_direction == ("B", "A")
    assert not runtime.running


def test_next_round_reuses_receiver_ids_without_generation():
    runtime, a, b, token, kv, selector = build(CommunicationMode.TOKEN)
    runtime.communicate_next()
    selector.token.calls.clear()  # selector assertion is scoped to a new communication
    runtime.communicate_next()
    assert a.calls == 1 and b.calls == 0
    assert token.calls[0] == ("B", "A", (4, 5))
    assert runtime.next_direction == ("A", "B")


@pytest.mark.parametrize("failure", ["selector", "token", "correctness"])
def test_failure_keeps_direction_and_clears_running(failure):
    runtime, a, b, token, kv, selector = build(CommunicationMode.TOKEN)
    if failure == "selector": selector.fail = True
    if failure == "token": token.fail = True
    if failure == "correctness": token.correct = False
    with pytest.raises(RuntimeError):
        runtime.communicate_next()
    assert runtime.next_direction == ("A", "B")
    assert not runtime.running


def test_kv_failure_keeps_direction():
    runtime, a, b, token, kv, selector = build(CommunicationMode.KV)
    kv.fail = True
    with pytest.raises(RuntimeError, match="kv failed"):
        runtime.communicate_next()
    assert runtime.next_direction == ("A", "B")


def test_network_mutation_blocked_during_selection():
    runtime, a, b, token, kv, selector = build(CommunicationMode.TOKEN)
    original = selector.select
    def select(context, *, boundary):
        with pytest.raises(RealAdaptiveRuntimeBusyError):
            runtime.set_bandwidth_mbps(1000)
        return original(context, boundary=boundary)
    selector.select = select
    runtime.communicate_next()
    assert runtime.network.state.bandwidth_mbps == 20


def test_incompatible_profile_rejected():
    runtime, a, b, token, kv, selector = build(CommunicationMode.TOKEN)
    bad = Backend()
    bad.metadata = SimpleNamespace(model_name=PROFILE.model_id, device="cuda:0", dtype="torch.bfloat16")
    with pytest.raises(ValueError, match="device_type"):
        RealAdaptiveRuntime(Agent("A", bad), Agent("B", bad), PROFILE, SimulatedNetwork(20))


def test_real_selector_is_default():
    backend = Backend()
    token, kv = TokenPath(), KVPath()
    actual = RealAdaptiveRuntime(Agent("A", backend), Agent("B", backend), PROFILE,
                                 SimulatedNetwork(20), token_handoff=token, kv_handoff=kv)
    assert isinstance(actual.selector, RealCostBasedSelector)
    result = actual.communicate_next()
    assert result.selected_mode == "TOKEN"
    assert len(token.calls) == 1 and len(kv.calls) == 0


def test_stability_state_only_advances_after_correct_execution():
    backend = Backend()
    token, kv = TokenPath(), KVPath()
    synthetic = synthetic_crossover_profile()
    class SyntheticRaw:
        def select(self, context, *, boundary):
            test_context = RealSelectionContext.from_profile(synthetic, context.handoff_token_count,
                context.bandwidth_mbps, context.base_latency_ms)
            return RealCostBasedSelector(RealCommunicationCostModel(synthetic)).select(test_context, boundary=boundary)
    selector = StableRealCostSelector(SyntheticRaw(), SelectionStabilityConfig(True, 2, .05))
    selector.record_success(CommunicationMode.TOKEN)
    runtime = RealAdaptiveRuntime(Agent("A", backend), Agent("B", backend), PROFILE,
                                  SimulatedNetwork(200, 5), selector=selector,
                                  token_handoff=token, kv_handoff=kv)
    kv.fail = True
    with pytest.raises(RuntimeError, match="kv failed"):
        runtime.communicate_next()
    assert selector.previous_successful_mode == CommunicationMode.TOKEN
    assert runtime.next_direction == ("A", "B")
    kv.fail = False
    result = runtime.communicate_next()
    assert result.selected_mode == "KV"
    assert result.stability["raw_candidate_mode"] == "KV"
    assert selector.previous_successful_mode == CommunicationMode.KV
    assert len(token.calls) == 0 and len(kv.calls) == 2
    runtime.set_bandwidth_mbps(1)
    token.correct = False
    with pytest.raises(RuntimeError, match="correctness"):
        runtime.communicate_next()
    assert selector.previous_successful_mode == CommunicationMode.KV
    token.correct = True
    result = runtime.communicate_next()
    assert result.selected_mode == "TOKEN"
    assert selector.previous_successful_mode == CommunicationMode.TOKEN
