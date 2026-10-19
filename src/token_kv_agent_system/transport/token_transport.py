from typing import Any

from token_kv_agent_system.models import CommunicationMode, TokenPayload

from .base import Transport


class TokenTransport(Transport):
    """Phase 1 estimated-size transport; it does not send Token Protocol bytes."""

    mode = CommunicationMode.TOKEN

    def payload_size_bytes(self, payload: Any) -> int:
        if not isinstance(payload, TokenPayload):
            raise TypeError("TokenTransport requires TokenPayload")
        return payload.estimated_size_bytes
