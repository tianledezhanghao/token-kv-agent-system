from token_kv_agent_system.demo import run_demo
from token_kv_agent_system.metrics import EventType
from token_kv_agent_system.models import CommunicationMode


def test_demo_runs_both_directions_and_switches_mode() -> None:
    lines: list[str] = []

    results, collector = run_demo(lines.append)

    assert [result.decision.mode for result in results] == [
        CommunicationMode.TOKEN,
        CommunicationMode.KV,
    ]
    finished = collector.events_of_type(EventType.TRANSMISSION_FINISHED)
    assert [(event.source_agent_id, event.target_agent_id) for event in finished] == [
        ("Agent A", "Agent B"),
        ("Agent B", "Agent A"),
    ]
    output = "\n".join(lines)
    assert "bandwidth = 20 Mbps" in output
    assert "TOKEN selected" in output
    assert "bandwidth changed to 1000 Mbps" in output
    assert "KV selected" in output

