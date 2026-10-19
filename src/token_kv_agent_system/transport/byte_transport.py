"""Byte-only transport boundary shared by real Token and KV data planes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ByteTransferRecord:
    """Small transfer metadata; the data-plane payload is not retained."""

    source: str
    target: str
    byte_count: int


class BaseByteTransport(ABC):
    """Transport that knows only opaque bytes and endpoint identities."""

    @abstractmethod
    def transfer(self, data: bytes, *, source: str, target: str) -> bytes:
        """Deliver opaque bytes across the transport boundary."""


class InMemoryByteTransport(BaseByteTransport):
    """Synchronous single-process byte boundary with identity delivery."""

    def __init__(self) -> None:
        self._records: list[ByteTransferRecord] = []

    @property
    def records(self) -> tuple[ByteTransferRecord, ...]:
        return tuple(self._records)

    def transfer(self, data: bytes, *, source: str, target: str) -> bytes:
        if not isinstance(data, bytes):
            raise TypeError("byte transport data must be bytes")
        if not source.strip() or not target.strip():
            raise ValueError("source and target cannot be empty")
        if source == target:
            raise ValueError("source and target must be different agents")
        self._records.append(
            ByteTransferRecord(source=source, target=target, byte_count=len(data))
        )
        return data
