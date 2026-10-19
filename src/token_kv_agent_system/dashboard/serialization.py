"""JSON adapters for core event models.

The core domain remains unaware of HTTP and Pydantic. Dashboard responses are
plain JSON-compatible dictionaries derived from the existing event model.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from token_kv_agent_system.metrics import CommunicationEvent


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def serialize_event(event: CommunicationEvent) -> dict[str, Any]:
    """Serialize a core event without including any Token/KV payload object."""
    data: dict[str, Any] = {
        "type": event.event_type.value,
        "timestamp": event.timestamp.isoformat(),
        "source": event.source_agent_id,
        "target": event.target_agent_id,
        "mode": event.mode.value if event.mode is not None else None,
        "payload_bytes": event.payload_bytes,
        "transmission_latency_ms": event.transmission_latency_ms,
        "estimated_e2e_latency_ms": event.estimated_e2e_latency_ms,
        "metadata": _json_value(event.metadata),
    }
    return data
