"""Read-only selector sensitivity and prediction-error summaries."""

from dataclasses import asdict, dataclass
from statistics import mean, median

from .real_cost import (MESSAGE_READY, RealCommunicationCostModel,
                        RealCommunicationCostProfile, RealCostBasedSelector,
                        RealSelectionContext)


@dataclass(frozen=True)
class SensitivityRow:
    boundary: str
    token_count: int
    calibration_position: str
    bandwidth_mbps: float
    base_latency_ms: float
    token_estimated_ms: float
    kv_estimated_ms: float
    raw_selected_mode: str
    decision_margin_ms: float
    decision_margin_ratio: float
    near_tie: bool
    crossover_bandwidth_mbps: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def sensitivity_rows(profile: RealCommunicationCostProfile, *, boundaries: tuple[str, ...],
                     token_counts: tuple[int, ...], bandwidths: tuple[float, ...],
                     base_latencies: tuple[float, ...], near_tie_ms: float = 0.0) -> list[SensitivityRow]:
    if near_tie_ms < 0:
        raise ValueError("near-tie threshold must be nonnegative")
    model = RealCommunicationCostModel(profile)
    selector = RealCostBasedSelector(model)
    rows = []
    for boundary in boundaries:
        for count in token_counts:
            for latency in base_latencies:
                for bandwidth in bandwidths:
                    context = RealSelectionContext.from_profile(profile, count, bandwidth, latency)
                    decision = selector.select(context, boundary=boundary)
                    token = decision.token_estimate.total_ms
                    kv = (decision.kv_estimate.message_ready_total_ms if boundary == MESSAGE_READY
                          else decision.kv_estimate.representation_ready_total_ms)
                    gap = abs(token - kv)
                    rows.append(SensitivityRow(boundary, count, decision.token_estimate.calibration_position,
                                               bandwidth, latency, token, kv, decision.selected_mode.value,
                                               gap, gap / max(token, kv) if max(token, kv) else 0.0,
                                               gap <= near_tie_ms,
                                               model.crossover_bandwidth_mbps(context, boundary=boundary)))
    return rows


def prediction_error_summary(results: list[dict]) -> dict:
    if not results:
        raise ValueError("at least one runtime result is required")
    errors = []
    percentages = []
    for result in results:
        predicted = float(result["predicted_selected_cost_ms"])
        actual = float(result["actual_selected_cost_ms"])
        if predicted <= 0 or actual < 0:
            raise ValueError("invalid predicted or actual cost")
        error = abs(actual - predicted)
        errors.append(error)
        percentages.append(error / predicted)
    return {"count": len(errors), "mae_ms": mean(errors), "median_ape": median(percentages),
            "max_error_ms": max(errors)}
