"""Small thread-safe fan-out for Dashboard event WebSockets."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class EventSubscription:
    identifier: int
    queue: asyncio.Queue[dict[str, Any]]


class EventBroadcaster:
    def __init__(self, *, queue_size: int = 100) -> None:
        self._queue_size = queue_size
        self._lock = RLock()
        self._next_identifier = 0
        self._subscribers: dict[
            int, tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any]]]
        ] = {}

    def subscribe(self) -> EventSubscription:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(self._queue_size)
        with self._lock:
            identifier = self._next_identifier
            self._next_identifier += 1
            self._subscribers[identifier] = (loop, queue)
        return EventSubscription(identifier, queue)

    def unsubscribe(self, subscription: EventSubscription) -> None:
        with self._lock:
            self._subscribers.pop(subscription.identifier, None)

    @staticmethod
    def _enqueue(
        queue: asyncio.Queue[dict[str, Any]], event: dict[str, Any]
    ) -> None:
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(event)

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            subscribers = tuple(self._subscribers.values())
        for loop, queue in subscribers:
            try:
                loop.call_soon_threadsafe(self._enqueue, queue, event)
            except RuntimeError:
                # A closing client's event loop must never break the core event path.
                continue
