"""
Unified EventBus adapter for publishing lifecycle and messaging events.
- Inline bus: always publish to in-proc bus for local subscribers
- NATS (optional): publish to JetStream subjects when ZEN_EVENT_BUS=nats

This module centralizes the subject mapping and environment checks so callers
can use a single interface regardless of backend.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any


class EventPublisher:
    def __init__(self) -> None:
        self._mode = os.getenv("ZEN_EVENT_BUS", os.getenv("ZEN_EVENTS", "inline")).lower()
        self._lock = asyncio.Lock()

    async def _publish_inline(self, event: dict[str, Any]) -> None:
        try:
            from utils.event_bus import get_event_bus
            await get_event_bus().publish(event)
        except Exception:
            # Best-effort
            pass

    async def _publish_nats(self, subject: str, event: dict[str, Any]) -> None:
        try:
            from utils.nats_communicator import get_nats_communicator
            ncomm = await get_nats_communicator(None)
            await ncomm.publish(subject, event, use_jetstream=True)
        except Exception:
            # Best-effort
            pass

    async def publish_lifecycle(self, event_name: str, payload: dict[str, Any]) -> None:
        """Publish lifecycle events to inline bus and, if enabled, to NATS."""
        event = {"event": event_name, **payload}
        # Always publish inline first
        await self._publish_inline(event)
        # Optionally publish to NATS
        if self._mode == "nats":
            subject = {
                "task_created": "tasks.created",
                "task_updated": "tasks.updated.running",
                "task_completed": "tasks.completed",
                "task_failed": "tasks.failed",
                "task_timeout": "tasks.timeout",
                "task_cancelled": "tasks.cancelled",
                "batch_started": "batches.started",
                "batch_finished": "batches.finished",
            }.get(event_name, "system.events")
            await self._publish_nats(subject, event)

    async def publish_messaging(self, event_name: str, payload: dict[str, Any]) -> None:
        """Publish messaging-related events."""
        event = {"event": event_name, **payload}
        await self._publish_inline(event)
        if self._mode == "nats":
            subject = {
                "messaging_posted": "messaging.posted",
                "messaging_read": "messaging.read",
                "channel_created": "channels.created",
                "projects.created": "projects.created",
                "projects.agent_added": "projects.agent_added",
                "projects.artifact_added": "projects.artifact_added",
            }.get(event_name, "messaging.events")
            await self._publish_nats(subject, event)


_publisher: EventPublisher | None = None


def get_event_publisher() -> EventPublisher:
    global _publisher
    if _publisher is None:
        _publisher = EventPublisher()
    return _publisher

