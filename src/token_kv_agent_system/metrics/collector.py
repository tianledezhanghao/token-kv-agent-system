"""In-memory event sink with a minimal subscription API."""

from __future__ import annotations

from collections.abc import Callable

from .events import CommunicationEvent, EventType

EventListener = Callable[[CommunicationEvent], None]


class MetricCollector:
    def __init__(self) -> None:
        self._events: list[CommunicationEvent] = []
        self._listeners: list[EventListener] = []

    @property
    def events(self) -> tuple[CommunicationEvent, ...]:
        return tuple(self._events)

    def subscribe(self, listener: EventListener) -> None:
        self._listeners.append(listener)

    def record(self, event: CommunicationEvent) -> None:
        self._events.append(event)
        for listener in tuple(self._listeners):
            listener(event)

    def events_of_type(self, event_type: EventType) -> tuple[CommunicationEvent, ...]:
        return tuple(event for event in self._events if event.event_type is event_type)

