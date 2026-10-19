import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from token_kv_agent_system.models import CommunicationMode
from token_kv_agent_system.real_cost import (MESSAGE_READY, REPRESENTATION_READY,
    RealCommunicationCostModel, RealCommunicationCostProfile, RealCostBasedSelector, RealSelectionContext)
from token_kv_agent_system.selection_analysis import prediction_error_summary, sensitivity_rows
from token_kv_agent_system.selection_stability import SelectionStabilityConfig, StableRealCostSelector
from token_kv_agent_system.synthetic_selection import synthetic_crossover_profile


PROFILE = synthetic_crossover_profile()


def decision(selector, bandwidth):
    return selector.select(RealSelectionContext.from_profile(PROFILE, 8, bandwidth, 5))


def test_disabled_equals_raw_and_first_decision_and_tie():
    raw = RealCostBasedSelector(RealCommunicationCostModel(PROFILE))
    stable = StableRealCostSelector(raw)
    for bandwidth in (20, 50, 90, 200):
        assert decision(stable, bandwidth).selected_mode == decision(raw, bandwidth).selected_mode
    first = decision(stable, 200)
    assert first.previous_mode is None and first.selected_mode == CommunicationMode.KV
    crossover = RealCommunicationCostModel(PROFILE).crossover_bandwidth_mbps(
        RealSelectionContext.from_profile(PROFILE, 8, 80, 5))
    tie = decision(stable, crossover)
    assert tie.selected_mode == decision(raw, crossover).selected_mode


def test_both_margins_and_bidirectional_switches():
    selector = StableRealCostSelector(RealCostBasedSelector(RealCommunicationCostModel(PROFILE)),
                                      SelectionStabilityConfig(True, 2, .05))
    selector.record_success(CommunicationMode.TOKEN)
    small = decision(selector, 90)
    assert small.raw_candidate_mode == CommunicationMode.KV
    assert small.selected_mode == CommunicationMode.TOKEN and small.switch_suppressed
    assert small.stability_metadata()["previous_mode"] == "TOKEN"
    assert json.loads(json.dumps(small.stability_metadata()))["switch_suppressed"]
    large = decision(selector, 200)
    assert large.selected_mode == CommunicationMode.KV and large.switch_allowed
    selector.record_success(CommunicationMode.KV)
    small_back = decision(selector, 78)
    assert small_back.raw_candidate_mode == CommunicationMode.TOKEN
    assert small_back.selected_mode == CommunicationMode.KV and small_back.switch_suppressed
    large_back = decision(selector, 20)
    assert large_back.selected_mode == CommunicationMode.TOKEN and large_back.switch_allowed
    assert small.token_estimate == decision(RealCostBasedSelector(RealCommunicationCostModel(PROFILE)), 90).token_estimate


def test_absolute_and_relative_thresholds_each_required():
    raw = RealCostBasedSelector(RealCommunicationCostModel(PROFILE))
    absolute = StableRealCostSelector(raw, SelectionStabilityConfig(True, 2, 0))
    relative = StableRealCostSelector(raw, SelectionStabilityConfig(True, 0, .2))
    for selector in (absolute, relative):
        selector.record_success(CommunicationMode.TOKEN)
        assert decision(selector, 90).switch_suppressed
        assert decision(selector, 1000).switch_allowed
    with pytest.raises(ValueError):
        SelectionStabilityConfig(True, -1, 0)


def test_synthetic_flapping_is_suppressed_then_switches_both_ways():
    stable = StableRealCostSelector(RealCostBasedSelector(RealCommunicationCostModel(PROFILE)),
                                    SelectionStabilityConfig(True, 2, .05))
    sequence = (50, 88, 78, 90, 76, 200, 90, 78, 20)
    raw_modes, final_modes = [], []
    for bandwidth in sequence:
        outcome = decision(stable, bandwidth)
        raw_modes.append(outcome.raw_candidate_mode.value)
        final_modes.append(outcome.selected_mode.value)
        stable.record_success(outcome.selected_mode)
    assert raw_modes == ["TOKEN", "KV", "TOKEN", "KV", "TOKEN", "KV", "KV", "TOKEN", "TOKEN"]
    assert final_modes == ["TOKEN"] * 5 + ["KV"] * 3 + ["TOKEN"]


def test_sensitivity_positions_boundaries_and_latency_cancellation():
    rows = sensitivity_rows(PROFILE, boundaries=(MESSAGE_READY, REPRESENTATION_READY),
                            token_counts=(8, 16, 64), bandwidths=(20, 200),
                            base_latencies=(.5, 50), near_tie_ms=2)
    assert {r.calibration_position for r in rows} == {"exact", "interpolated", "extrapolated"}
    for row in rows:
        paired = next(r for r in rows if r.boundary == row.boundary and r.token_count == row.token_count
                      and r.bandwidth_mbps == row.bandwidth_mbps and r.base_latency_ms != row.base_latency_ms)
        assert paired.decision_margin_ms == pytest.approx(row.decision_margin_ms)
        assert paired.crossover_bandwidth_mbps == pytest.approx(row.crossover_bandwidth_mbps)
    assert all(r.kv_estimated_ms <= next(x for x in rows if x.boundary == MESSAGE_READY
        and x.token_count == r.token_count and x.bandwidth_mbps == r.bandwidth_mbps
        and x.base_latency_ms == r.base_latency_ms).kv_estimated_ms
        for r in rows if r.boundary == REPRESENTATION_READY)


def test_prediction_error_and_cuda_diagnostic_and_portable_schema():
    assert prediction_error_summary([{"predicted_selected_cost_ms": 10, "actual_selected_cost_ms": 12},
                                     {"predicted_selected_cost_ms": 20, "actual_selected_cost_ms": 19}]) == {
                                         "count": 2, "mae_ms": 1.5, "median_ape": .125, "max_error_ms": 2}
    cuda = replace(PROFILE, device_type="cuda", source_device="cuda:0")
    assert RealCommunicationCostProfile.from_json(cuda.to_json()) == cuda
    assert RealCostBasedSelector(RealCommunicationCostModel(cuda)).select(
        RealSelectionContext.from_profile(cuda, 8, 50, 5)).selected_mode == CommunicationMode.TOKEN


def test_readiness_cli_reports_cuda_unavailable_without_model_load():
    import torch
    if torch.cuda.is_available():
        pytest.skip("CUDA host has a device; unavailable diagnostic is Mac-only")
    result = subprocess.run([sys.executable, "run_server_readiness.py", "--device", "cuda",
                             "--offline", "--load-model"], capture_output=True, text=True)
    assert result.returncode == 1
    checks = json.loads(result.stdout)["checks"]
    assert checks["requested_device"] == {"ok": False, "detail": "CUDA unavailable on this machine"}
    assert checks["model_load"]["ok"] is False


def test_measured_mps_profile_preserves_known_message_ready_results():
    path = Path("benchmark-results/mac-mps-selector-profile.json")
    if not path.exists():
        pytest.skip("local gitignored development profile unavailable")
    profile = RealCommunicationCostProfile.from_json(path.read_text())
    model = RealCommunicationCostModel(profile)
    selector = RealCostBasedSelector(model)
    for count in (8, 32, 128):
        for bandwidth in (20, 1000):
            context = RealSelectionContext.from_profile(profile, count, bandwidth, 5)
            assert selector.select(context).selected_mode == CommunicationMode.TOKEN
            assert model.crossover_bandwidth_mbps(context) is None
