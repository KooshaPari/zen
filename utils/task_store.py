"""
Redis-backed TaskStore facade (MVP)

Centralizes task persistence and task message streams behind a simple async API.
Used opportunistically by AgentTaskManager when Redis is enabled; otherwise,
fallback to in-memory behavior remains unchanged.

Env knobs:
- AGENT_TASK_RETENTION_SEC (TTL)
- TASK_MESSAGES_MAXLEN

This module does not import heavy deps at module import time.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class TaskStore:
    def __init__(self, redis_client: Any) -> None:
        self.redis = redis_client

    async def store_task(self, task: Any, retention: int | None = None) -> None:
        """Persist task JSON and update indexes with TTL."""
        try:
            retention = int(retention or os.getenv("AGENT_TASK_RETENTION_SEC", os.getenv("ZEN_AGENT_TASK_RETENTION_SEC", "3600")))
            key = f"task:{task.task_id}"
            data = json.loads(task.model_dump_json())
            data["schema_version"] = 1
            payload = json.dumps(data)
            # Main record with TTL
            self.redis.setex(key, retention, payload)
            # Indexes
            try:
                updated = (task.updated_at or task.created_at).timestamp()
                self.redis.zadd("inbox:status:" + task.status.value, {task.task_id: updated})
                self.redis.zadd("tasks:by_created_at", {task.task_id: task.created_at.timestamp()})
                max_keep = int(os.getenv("TASK_INDEX_MAX", "5000"))
                current = self.redis.zcard("tasks:by_created_at")
                if current and int(current) > max_keep:
                    self.redis.zremrangebyrank("tasks:by_created_at", 0, int(current) - max_keep - 1)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"TaskStore.store_task failed: {e}")

    async def load_task_json(self, task_id: str) -> str | None:
        try:
            for key in (f"task:{task_id}", f"agent_task:{task_id}"):
                data = self.redis.get(key)
                if data:
                    return data
        except Exception as e:
            logger.warning(f"TaskStore.load_task_json failed: {e}")
        return None

    async def append_task_message(self, task_id: str, event: str, data: dict) -> None:
        try:
            stream_key = f"task:{task_id}:messages"
            fields = {
                "ts": int(datetime.now(timezone.utc).timestamp() * 1000),
                "event": event,
                "data": json.dumps(data),
            }
            maxlen = int(os.getenv("TASK_MESSAGES_MAXLEN", "1000"))
            # XADD + XTRIM
            self.redis.xadd(stream_key, fields, maxlen=maxlen, approximate=True)  # type: ignore[attr-defined]
        except Exception:
            logger.debug("TaskStore.append_task_message failed", exc_info=True)

