from dataclasses import replace

import pytest

from token_kv_agent_system.kv_protocol import KVModelCompatibility, estimate_kv_wire_size
from token_kv_agent_system.models import CommunicationMode
from token_kv_agent_system.network import SimulatedNetwork
from token_kv_agent_system.real_cost import (
    CalibrationPoint, MESSAGE_READY, REPRESENTATION_READY, RealCommunicationCostModel,
    RealCommunicationCostProfile, RealCostBasedSelector, RealSelectionContext,
)
from token_kv_agent_system.token_protocol import estimate_token_wire_bytes, serialize_token_ids


def profile(prefill=20, prepare=20, restore=5):
    points = tuple(CalibrationPoint(n, 1, 2, prefill, prepare, 3, 4, restore)
                   for n in (8, 32, 128))
    return RealCommunicationCostProfile(1, "Qwen/Qwen2.5-0.5B-Instruct", 24, 2, 64,
        "bfloat16", 1, "mps", "macbook-development", "mps:0", 1, 1,
        "warmup + median", 1, 3, points)


def context(p, count=8, bandwidth=1000):
    return RealSelectionContext.from_profile(p, count, bandwidth, 5)


def test_wire_sizes_and_link_exact():
    model = KVModelCompatibility("Qwen/Qwen2.5-0.5B-Instruct", 24, 2, 64)
    for count, expected in ((8, 100587), (32, 395499), (128, 1575147)):
        assert estimate_kv_wire_size(count, model, dtype="bfloat16", source_device="mps:0").total_wire_bytes == expected
        assert estimate_token_wire_bytes(count) == len(serialize_token_ids([1] * count))
    assert SimulatedNetwork(20, 5).estimate_transmission_latency_ms(42) == pytest.approx(5.0168)


def test_profile_roundtrip_and_mismatch():
    p = profile()
    assert RealCommunicationCostProfile.from_json(p.to_json()) == p
    model = RealCommunicationCostModel(p)
    with pytest.raises(ValueError, match="device_type"):
        model.estimate(replace(context(p), device_type="cuda"))
    with pytest.raises(ValueError, match="profile_name"):
        model.estimate(replace(context(p), profile_name="lab-server"))
    with pytest.raises(ValueError, match="protocol"):
        replace(p, token_protocol_version=2)
    with pytest.raises(ValueError, match="ascending"):
        replace(p, points=(p.points[1], p.points[0]))


def test_exact_interpolation_extrapolation_and_formulas():
    p = profile()
    model = RealCommunicationCostModel(p)
    exact = model.estimate(context(p))
    assert exact.token.calibration_position == "exact"
    assert exact.token.total_ms == pytest.approx(1 + exact.token.link_ms + 2 + 20)
    assert exact.kv.representation_ready_total_ms == pytest.approx(3 + exact.kv.link_ms + 4 + 5)
    assert exact.kv.message_ready_total_ms == pytest.approx(20 + exact.kv.representation_ready_total_ms)
    assert model.estimate(context(p, 64)).token.calibration_position == "interpolated"
    assert model.estimate(context(p, 4)).kv.calibration_position == "extrapolated"
    assert model.estimate(context(p, 256)).token.calibration_position == "extrapolated"
    fast = model.estimate(context(p, 8, 1000))
    slow = model.estimate(context(p, 8, 20))
    assert fast.token.receiver_prefill_ms == slow.token.receiver_prefill_ms
    assert fast.kv.serialize_ms == slow.kv.serialize_ms
    assert fast.kv.link_ms < slow.kv.link_ms
    assert model.estimate(context(p, 32)).kv.wire_bytes > fast.kv.wire_bytes


def test_piecewise_values_and_nonnegative_extrapolation():
    p = profile()
    pts = (
        replace(p.points[0], receiver_prefill_ms=10, kv_restore_ms=10),
        replace(p.points[1], receiver_prefill_ms=34, kv_restore_ms=1),
        replace(p.points[2], receiver_prefill_ms=130, kv_restore_ms=1),
    )
    model = RealCommunicationCostModel(replace(p, points=pts))
    assert model.estimate(context(p, 20)).token.receiver_prefill_ms == pytest.approx(22)
    assert model.estimate(context(p, 64)).token.receiver_prefill_ms == pytest.approx(66)
    assert model.estimate(context(p, 1)).token.receiver_prefill_ms == pytest.approx(3)
    assert model.estimate(context(p, 64)).kv.restore_ms == pytest.approx(1)
    assert model.estimate(context(p, 1280)).kv.restore_ms == pytest.approx(1)


def test_selector_boundary_tie_winners_and_crossover():
    p = profile(prefill=20, prepare=20)
    model = RealCommunicationCostModel(p)
    selector = RealCostBasedSelector(model)
    assert selector.select(context(p)).selected_mode == CommunicationMode.TOKEN
    assert selector.select(context(p), boundary=REPRESENTATION_READY).selected_mode == CommunicationMode.KV
    assert model.crossover_bandwidth_mbps(context(p), boundary=MESSAGE_READY) is None
    crossover = model.crossover_bandwidth_mbps(context(p), boundary=REPRESENTATION_READY)
    assert crossover is not None and crossover > 0
    tie_p = profile(prefill=19, prepare=20, restore=10)
    # Check tie with zero wire difference using a minimal fake model estimate.
    tie_model = RealCommunicationCostModel(tie_p)
    original = tie_model.estimate
    def tied(ctx):
        e = original(ctx)
        return replace(e, kv=replace(e.kv, message_ready_total_ms=e.token.total_ms))
    tie_model.estimate = tied
    assert RealCostBasedSelector(tie_model).select(context(tie_p)).selected_mode == CommunicationMode.TOKEN
    cheaper = profile(prefill=100, prepare=1)
    assert RealCostBasedSelector(RealCommunicationCostModel(cheaper)).select(context(cheaper)).selected_mode == CommunicationMode.KV


def test_selector_is_metadata_only(monkeypatch):
    p = profile()
    model = RealCommunicationCostModel(p)
    monkeypatch.setattr("token_kv_agent_system.token_protocol.serialize_token_ids", lambda *_: pytest.fail("packet created"))
    monkeypatch.setattr("token_kv_agent_system.kv_protocol.serialize_kv_payload", lambda *_: pytest.fail("packet created"))
    RealCostBasedSelector(model).select(context(p))
