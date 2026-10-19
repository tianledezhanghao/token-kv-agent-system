"""Optional hysteresis around the pure real cost selector."""

from dataclasses import asdict, dataclass
from math import isfinite

from .models import CommunicationMode
from .real_cost import MESSAGE_READY, RealCostBasedSelector, RealSelectionContext, SelectionDecision


@dataclass(frozen=True)
class SelectionStabilityConfig:
    enabled: bool = False
    min_absolute_saving_ms: float = 0.0
    min_relative_saving: float = 0.0

    def __post_init__(self) -> None:
        for value in (self.min_absolute_saving_ms, self.min_relative_saving):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                raise ValueError("stability margins must be finite and nonnegative")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StableSelectionDecision:
    selected_mode: CommunicationMode
    cost_boundary: str
    token_estimate: object
    kv_estimate: object
    estimated_saving_ms: float
    estimated_saving_ratio: float
    raw_candidate_mode: CommunicationMode
    previous_mode: CommunicationMode | None
    candidate_saving_ms: float
    candidate_saving_ratio: float
    switch_allowed: bool
    switch_suppressed: bool
    stability_config: SelectionStabilityConfig

    def stability_metadata(self) -> dict:
        return {
            "raw_candidate_mode": self.raw_candidate_mode.value,
            "final_selected_mode": self.selected_mode.value,
            "previous_mode": self.previous_mode.value if self.previous_mode else None,
            "candidate_saving_ms": self.candidate_saving_ms,
            "candidate_saving_ratio": self.candidate_saving_ratio,
            "switch_allowed": self.switch_allowed,
            "switch_suppressed": self.switch_suppressed,
            "stability_config": self.stability_config.to_dict(),
        }


class StableRealCostSelector:
    """State advances only after the runtime confirms a correct executed path."""

    def __init__(self, raw_selector: RealCostBasedSelector,
                 config: SelectionStabilityConfig = SelectionStabilityConfig()) -> None:
        self.raw_selector = raw_selector
        self.config = config
        self.previous_successful_mode: CommunicationMode | None = None

    def select(self, context: RealSelectionContext, *, boundary: str = MESSAGE_READY) -> StableSelectionDecision:
        raw: SelectionDecision = self.raw_selector.select(context, boundary=boundary)
        previous = self.previous_successful_mode
        token_cost = raw.token_estimate.total_ms
        kv_cost = (raw.kv_estimate.message_ready_total_ms if boundary == MESSAGE_READY
                   else raw.kv_estimate.representation_ready_total_ms)
        candidate_cost = token_cost if raw.selected_mode == CommunicationMode.TOKEN else kv_cost
        previous_cost = (token_cost if previous == CommunicationMode.TOKEN else kv_cost) if previous else candidate_cost
        saving = max(0.0, previous_cost - candidate_cost) if previous != raw.selected_mode else 0.0
        ratio = saving / previous_cost if previous_cost else 0.0
        switching = previous is not None and previous != raw.selected_mode
        allowed = (not switching or not self.config.enabled or
                   (saving >= self.config.min_absolute_saving_ms and
                    ratio >= self.config.min_relative_saving))
        final = raw.selected_mode if allowed else previous
        return StableSelectionDecision(
            final, raw.cost_boundary, raw.token_estimate, raw.kv_estimate,
            raw.estimated_saving_ms, raw.estimated_saving_ratio,
            raw.selected_mode, previous, saving, ratio, allowed, switching and not allowed,
            self.config)

    def record_success(self, mode: CommunicationMode) -> None:
        if mode not in (CommunicationMode.TOKEN, CommunicationMode.KV):
            raise ValueError("unsupported successful mode")
        self.previous_successful_mode = mode
