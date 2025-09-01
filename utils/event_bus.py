"""
Simple in-process pub/sub event bus for task lifecycle events.

- Subscribers receive dict events via asyncio.Queue
- Publisher broadcasts to all subscribers
- Designed to be lightweight and dependency-free
"""
from __future__ import annotations

import asyncio
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subscribers.append(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)
            # drain
            try:
                while not q.empty():
                    q.get_nowait()
                    q.task_done()
            except Exception:
                pass

    async def publish(self, event: dict[str, Any]) -> None:
        # Best-effort broadcast; don't block publisher
        async with self._lock:
            targets = list(self._subscribers)
        for q in targets:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop if subscriber is too slow
                pass


# Global bus instance
_event_bus = EventBus()


def get_event_bus() -> EventBus:
    return _event_bus

