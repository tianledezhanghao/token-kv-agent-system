"""Demonstrate hysteresis using synthetic costs only."""

from token_kv_agent_system.real_cost import RealCommunicationCostModel, RealCostBasedSelector, RealSelectionContext
from token_kv_agent_system.selection_stability import SelectionStabilityConfig, StableRealCostSelector
from token_kv_agent_system.synthetic_selection import synthetic_crossover_profile


def main() -> None:
    profile = synthetic_crossover_profile()
    model = RealCommunicationCostModel(profile)
    selector = StableRealCostSelector(RealCostBasedSelector(model), SelectionStabilityConfig(True, 2.0, 0.05))
    print("SYNTHETIC DEMO — NOT MEASURED HARDWARE PROFILE")
    print("Mbps | Raw | Previous | Saving ms | Allowed | Final")
    for bandwidth in (50, 88, 78, 90, 76, 200, 90, 50, 20):
        decision = selector.select(RealSelectionContext.from_profile(profile, 8, bandwidth, 5))
        print(f"{bandwidth:4g} | {decision.raw_candidate_mode.value:5s} | "
              f"{decision.previous_mode.value if decision.previous_mode else '-':8s} | "
              f"{decision.candidate_saving_ms:9.3f} | {str(decision.switch_allowed):7s} | {decision.selected_mode.value}")
        selector.record_success(decision.selected_mode)


if __name__ == "__main__":
    main()
