from typing import Any

from token_kv_agent_system.models import CommunicationMode, KVPayload

from .base import Transport


class KVTransport(Transport):
    mode = CommunicationMode.KV

    def payload_size_bytes(self, payload: Any) -> int:
        if not isinstance(payload, KVPayload):
            raise TypeError("KVTransport requires KVPayload")
        return payload.estimated_size_bytes

