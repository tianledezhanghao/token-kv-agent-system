"""Thread-safe in-memory projection of the core communication event stream."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from threading import RLock
from typing import Any, Iterable

from token_kv_agent_system.metrics import CommunicationEvent, EventType
from token_kv_agent_system.models import NetworkState

from .serialization import serialize_event


class DashboardStateStore:
    """Project existing CommunicationEvents into a JSON-compatible view."""

    def __init__(
        self,
        *,
        initial_network_state: NetworkState | None = None,
        agent_ids: Iterable[str] = (),
        max_events: int = 200,
    ) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be greater than zero")
        self._lock = RLock()
        self._system_status = "idle"
        self._current_task: str | None = None
        self._network = self._network_dict(initial_network_state)
        self._agents = {
            agent_id: {"agent_id": agent_id, "status": "idle"}
            for agent_id in agent_ids
        }
        self._last_selection: dict[str, Any] | None = None
        self._last_communication: dict[str, Any] | None = None
        self._real_token_metrics: dict[str, Any] | None = None
        self._real_kv_metrics: dict[str, Any] | None = None
        self._adaptive_execution: dict[str, Any] | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)

    @staticmethod
    def _network_dict(state: NetworkState | None) -> dict[str, float] | None:
        if state is None:
            return None
        return {
            "bandwidth_mbps": state.bandwidth_mbps,
            "base_latency_ms": state.base_latency_ms,
        }

    def _set_agent_status(self, agent_id: str | None, status: str) -> None:
        if agent_id is not None:
            self._agents[agent_id] = {"agent_id": agent_id, "status": status}

    def consume(self, event: CommunicationEvent) -> None:
        """Consume one source-of-truth event. Safe to use as collector listener."""
        serialized = serialize_event(event)
        with self._lock:
            self._events.append(serialized)
            if event.event_type is EventType.NETWORK_STATE_CHANGED:
                bandwidth = event.metadata.get("bandwidth_mbps")
                base_latency = event.metadata.get("base_latency_ms")
                if bandwidth is not None and base_latency is not None:
                    self._network = {
                        "bandwidth_mbps": bandwidth,
                        "base_latency_ms": base_latency,
                    }
            elif event.event_type is EventType.AGENT_GENERATION_STARTED:
                self._system_status = "running"
                prompt = event.metadata.get("prompt")
                self._current_task = str(prompt) if prompt is not None else None
                self._set_agent_status(event.source_agent_id, "generating")
            elif event.event_type is EventType.AGENT_GENERATION_FINISHED:
                self._set_agent_status(event.source_agent_id, "idle")
            elif event.event_type is EventType.MODE_SELECTED:
                self._last_selection = {
                    "source": event.source_agent_id,
                    "target": event.target_agent_id,
                    "mode": event.mode.value if event.mode is not None else None,
                    "estimated_token_latency_ms": event.metadata.get(
                        "estimated_token_latency_ms"
                    ),
                    "estimated_kv_latency_ms": event.metadata.get(
                        "estimated_kv_latency_ms"
                    ),
                }
                self._set_agent_status(event.source_agent_id, "sending")
                self._set_agent_status(event.target_agent_id, "waiting")
            elif event.event_type is EventType.REAL_ADAPTIVE_HANDOFF_STARTED:
                self._system_status = "running"
                self._current_task = "Real Adaptive handoff"
                self._set_agent_status(event.source_agent_id, "generating")
                self._set_agent_status(event.target_agent_id, "waiting")
            elif event.event_type is EventType.REAL_SELECTION_MADE:
                metadata = event.metadata
                self._last_selection = {
                    "source": event.source_agent_id, "target": event.target_agent_id,
                    "token_count": metadata["token_count"],
                    "bandwidth_mbps": metadata["bandwidth_mbps"],
                    "base_latency_ms": metadata["base_latency_ms"],
                    "cost_boundary": metadata["cost_boundary"],
                    "selector_kind": metadata["selector_kind"],
                    "selected_mode": metadata["selected_mode"],
                    "estimated_saving_ms": metadata["estimated_saving_ms"],
                    "estimated_saving_ratio": metadata["estimated_saving_ratio"],
                    "token_estimate": dict(metadata["token_estimate"]),
                    "kv_estimate": dict(metadata["kv_estimate"]),
                    "stability": dict(metadata.get("stability", {})),
                }
                self._adaptive_execution = None
                self._set_agent_status(event.source_agent_id, "sending")
            elif event.event_type is EventType.SELECTED_PATH_STARTED:
                self._set_agent_status(event.target_agent_id, "receiving")
            elif event.event_type is EventType.REAL_ADAPTIVE_HANDOFF_FINISHED:
                metadata = event.metadata
                self._adaptive_execution = {
                    "source": event.source_agent_id, "target": event.target_agent_id,
                    "executed_mode": metadata["actually_executed_mode"],
                    "predicted_ms": metadata["predicted_selected_cost_ms"],
                    "actual_ms": metadata["actual_selected_cost_ms"],
                    "absolute_error_ms": metadata["absolute_error_ms"],
                    "relative_error": metadata["relative_error"],
                    "selected_measurement": dict(metadata["selected_measurement"]),
                    "sender_text": metadata["sender_text"],
                    "receiver_text": metadata["receiver_text"],
                    "correctness": metadata["correctness"],
                }
                self._last_communication = {
                    "source": event.source_agent_id, "target": event.target_agent_id,
                    "mode": metadata["actually_executed_mode"],
                    "payload_bytes": metadata["selected_measurement"]["wire_bytes"],
                    "transmission_latency_ms": metadata["selected_measurement"]["link_ms"],
                    "estimated_e2e_latency_ms": metadata["predicted_selected_cost_ms"],
                }
                self._set_agent_status(event.source_agent_id, "idle")
                self._set_agent_status(event.target_agent_id, "idle")
                self._system_status = "idle"
                self._current_task = None
            elif event.event_type is EventType.REAL_ADAPTIVE_HANDOFF_FAILED:
                self._system_status = "error"
                self._current_task = None
                self._set_agent_status(event.source_agent_id, "idle")
                self._set_agent_status(event.target_agent_id, "idle")
            elif event.event_type is EventType.TRANSMISSION_STARTED:
                self._set_agent_status(event.source_agent_id, "sending")
                self._set_agent_status(event.target_agent_id, "receiving")
            elif event.event_type is EventType.TRANSMISSION_FINISHED:
                self._last_communication = {
                    "source": event.source_agent_id,
                    "target": event.target_agent_id,
                    "mode": event.mode.value if event.mode is not None else None,
                    "payload_bytes": event.payload_bytes,
                    "transmission_latency_ms": event.transmission_latency_ms,
                    "estimated_e2e_latency_ms": event.estimated_e2e_latency_ms,
                }
                self._set_agent_status(event.source_agent_id, "idle")
                self._set_agent_status(event.target_agent_id, "idle")
                self._system_status = "idle"
                self._current_task = None
            elif event.event_type is EventType.REAL_TOKEN_HANDOFF_STARTED:
                self._system_status = "running"
                self._current_task = "Real Token direct causal continuation"
                self._set_agent_status(event.source_agent_id, "sending")
                self._set_agent_status(event.target_agent_id, "waiting")
            elif event.event_type is EventType.TOKEN_TRANSFERRED:
                self._set_agent_status(event.source_agent_id, "sending")
                self._set_agent_status(event.target_agent_id, "receiving")
            elif event.event_type is EventType.RECEIVER_PREFILL_FINISHED:
                self._set_agent_status(event.target_agent_id, "generating")
            elif event.event_type is EventType.REAL_TOKEN_HANDOFF_FINISHED:
                metadata = event.metadata
                self._real_token_metrics = {
                    "source": event.source_agent_id,
                    "target": event.target_agent_id,
                    "communication_mode": metadata.get("communication_mode"),
                    "measurement_strategy": metadata.get("measurement_strategy"),
                    "token_count": metadata.get("token_count"),
                    "theoretical_token_bytes": metadata.get("theoretical_token_bytes"),
                    "wire_bytes": metadata.get("wire_bytes"),
                    "serialization_ms": metadata.get("serialization_ms"),
                    "deserialization_ms": metadata.get("deserialization_ms"),
                    "receiver_prefill_ms": metadata.get("receiver_prefill_ms"),
                    "simulated_link_ms": metadata.get("simulated_link_ms"),
                    "token_handoff_cost_ms": metadata.get("token_handoff_cost_ms"),
                    "receiver_decode_mean_ms": metadata.get("receiver_decode_mean_ms"),
                    "receiver_decode_median_ms": metadata.get("receiver_decode_median_ms"),
                    "sender_generated_text": metadata.get("sender_generated_text"),
                    "receiver_generated_text": metadata.get("receiver_generated_text"),
                    "token_identity_verified": metadata.get("token_identity_verified"),
                    "protocol_version": metadata.get("protocol_version"),
                    "transport": metadata.get("transport"),
                }
                self._last_communication = {
                    "source": event.source_agent_id,
                    "target": event.target_agent_id,
                    "mode": event.mode.value if event.mode is not None else None,
                    "payload_bytes": event.payload_bytes,
                    "transmission_latency_ms": event.transmission_latency_ms,
                    "estimated_e2e_latency_ms": event.estimated_e2e_latency_ms,
                }
                self._set_agent_status(event.source_agent_id, "idle")
                self._set_agent_status(event.target_agent_id, "idle")
                self._system_status = "idle"
                self._current_task = None
            elif event.event_type is EventType.REAL_TOKEN_HANDOFF_FAILED:
                self._set_agent_status(event.source_agent_id, "idle")
                self._set_agent_status(event.target_agent_id, "idle")
                self._system_status = "error"
                self._current_task = None
            elif event.event_type is EventType.REAL_KV_HANDOFF_STARTED:
                self._system_status = "running"
                self._current_task = "Real KV canonical handoff"
                self._set_agent_status(event.source_agent_id, "sending")
                self._set_agent_status(event.target_agent_id, "waiting")
            elif event.event_type is EventType.KV_TRANSFERRED:
                self._set_agent_status(event.source_agent_id, "sending")
                self._set_agent_status(event.target_agent_id, "receiving")
            elif event.event_type is EventType.KV_RESTORED:
                self._set_agent_status(event.target_agent_id, "generating")
            elif event.event_type is EventType.REAL_KV_HANDOFF_FINISHED:
                metadata = event.metadata
                keys = (
                    "communication_mode",
                    "measurement_strategy",
                    "canonical_handoff_semantics",
                    "token_count",
                    "raw_kv_bytes",
                    "wire_tensor_body_bytes",
                    "wire_metadata_bytes",
                    "wire_bytes",
                    "sender_kv_prepare_ms",
                    "device_to_host_ms",
                    "protocol_encode_ms",
                    "total_serialize_ms",
                    "local_transfer_ms",
                    "deserialize_ms",
                    "cpu_tensor_reconstruction_ms",
                    "host_to_device_ms",
                    "cache_construction_ms",
                    "total_restore_ms",
                    "receiver_continuation_forward_ms",
                    "bandwidth_mbps",
                    "base_latency_ms",
                    "simulated_link_ms",
                    "representation_ready_cost_ms",
                    "message_ready_e2e_cost_ms",
                    "receiver_prefix_prefill_executed",
                    "sender_generated_text",
                    "receiver_generated_text",
                    "compatibility_verified",
                    "wire_identity_verified",
                    "canonical_cache_complete",
                    "continuation_verified",
                    "transport",
                    "protocol_version",
                )
                self._real_kv_metrics = {
                    "source": event.source_agent_id,
                    "target": event.target_agent_id,
                    **{key: metadata.get(key) for key in keys},
                }
                self._last_communication = {
                    "source": event.source_agent_id,
                    "target": event.target_agent_id,
                    "mode": event.mode.value if event.mode is not None else None,
                    "payload_bytes": event.payload_bytes,
                    "transmission_latency_ms": event.transmission_latency_ms,
                    "estimated_e2e_latency_ms": event.estimated_e2e_latency_ms,
                }
                self._set_agent_status(event.source_agent_id, "idle")
                self._set_agent_status(event.target_agent_id, "idle")
                self._system_status = "idle"
                self._current_task = None
            elif event.event_type is EventType.REAL_KV_HANDOFF_FAILED:
                self._set_agent_status(event.source_agent_id, "idle")
                self._set_agent_status(event.target_agent_id, "idle")
                self._system_status = "error"
                self._current_task = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            snapshot = {
                    "system_status": self._system_status,
                    "current_task": self._current_task,
                    "network": self._network,
                    "agents": self._agents,
                    "last_selection": self._last_selection,
                    "last_communication": self._last_communication,
                    "real_token_metrics": self._real_token_metrics,
                    "real_kv_metrics": self._real_kv_metrics,
                    "recent_events": list(self._events),
                }
            if self._last_selection and "selector_kind" in self._last_selection:
                snapshot["adaptive_execution"] = self._adaptive_execution
            return deepcopy(snapshot)

    def recent_events(self, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(list(self._events)[-limit:])
