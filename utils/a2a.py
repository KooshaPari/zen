from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class A2AIntent(str, Enum):
    DELEGATE = "delegate"
    REQUEST_TOOL = "request_tool"
    STATUS_UPDATE = "status_update"
    HANDOFF = "handoff"


class A2AEnvelope(BaseModel):
    spec: str = Field(default="a2a/1")
    type: str = Field(..., description="request|response|event|error")
    id: str = Field(...)
    correlation_id: str | None = None
    causation_id: str | None = None
    time: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    from_id: str | None = None
    to_id: str | None = None
    intent: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)


async def _nats_publish(subject: str, env: A2AEnvelope) -> None:
    if os.getenv("ZEN_EVENT_BUS", os.getenv("ZEN_EVENTS", "inline")).lower() != "nats":
        return
    from utils.nats_communicator import get_nats_communicator
    nats = await get_nats_communicator(None)
    await nats.publish(subject, json.loads(env.model_dump_json()), use_jetstream=True)


async def publish_task_event(task_id: str, event: str, payload: dict[str, Any]) -> None:
    """Publish a minimal a2a.task.<task_id>.events message if NATS is enabled."""
    try:
        subj = f"a2a.task.{task_id}.events"
        env = A2AEnvelope(
            type="event",
            id=f"evt-{task_id}-{int(datetime.now(timezone.utc).timestamp()*1000)}",
            intent=A2AIntent.STATUS_UPDATE.value,
            context={"task_id": task_id},
            payload={"event": event, **(payload or {})},
        )
        await _nats_publish(subj, env)
    except Exception:
        pass


async def publish_delegate(task_id: str, to_id: str, payload: dict[str, Any], from_id: str | None = None) -> None:
    try:
        subj = f"a2a.agent.{to_id}.in"
        env = A2AEnvelope(
            type="request",
            id=f"dlg-{int(datetime.now(timezone.utc).timestamp()*1000)}",
            from_id=from_id,
            to_id=to_id,
            intent=A2AIntent.DELEGATE.value,
            context={"task_id": task_id},
            payload=payload or {},
        )
        await _nats_publish(subj, env)
    except Exception:
        pass


async def publish_request_tool(task_id: str, to_id: str, tool_name: str, args: dict[str, Any], from_id: str | None = None) -> None:
    try:
        subj = f"a2a.agent.{to_id}.in"
        env = A2AEnvelope(
            type="request",
            id=f"tool-{int(datetime.now(timezone.utc).timestamp()*1000)}",
            from_id=from_id,
            to_id=to_id,
            intent=A2AIntent.REQUEST_TOOL.value,
            context={"task_id": task_id},
            payload={"tool": tool_name, "args": args or {}},
        )
        await _nats_publish(subj, env)
    except Exception:
        pass


async def publish_status_update(task_id: str, payload: dict[str, Any], from_id: str | None = None, to_id: str | None = None) -> None:
    try:
        subj = f"a2a.task.{task_id}.events"
        env = A2AEnvelope(
            type="event",
            id=f"st-{int(datetime.now(timezone.utc).timestamp()*1000)}",
            from_id=from_id,
            to_id=to_id,
            intent=A2AIntent.STATUS_UPDATE.value,
            context={"task_id": task_id},
            payload=payload or {},
        )
        await _nats_publish(subj, env)
    except Exception:
        pass


async def publish_handoff(task_id: str, to_id: str, context: dict[str, Any], from_id: str | None = None) -> None:
    try:
        subj = f"a2a.agent.{to_id}.in"
        env = A2AEnvelope(
            type="request",
            id=f"handoff-{int(datetime.now(timezone.utc).timestamp()*1000)}",
            from_id=from_id,
            to_id=to_id,
            intent=A2AIntent.HANDOFF.value,
            context={"task_id": task_id, **(context or {})},
            payload={},
        )
        await _nats_publish(subj, env)
    except Exception:
        pass
