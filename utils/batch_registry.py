"""
In-memory batch registry for mapping batch IDs to launched task IDs and metadata.

This keeps tracking simple for the current process lifetime.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Optional Redis support for persistence
try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore

_DEF_TTL = int(os.getenv("AGENT_TASK_RETENTION_SEC", os.getenv("ZEN_AGENT_TASK_RETENTION_SEC", "3600")))


def _get_redis():
    if os.getenv("ZEN_STORAGE", os.getenv("ZEN_STORAGE_MODE", "memory")).lower() != "redis":
        return None
    if not redis:
        return None
    try:
        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "1")),
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        client.ping()
        return client
    except Exception:
        return None


@dataclass
class BatchRecord:
    batch_id: str
    description: str = ""
    task_ids: list[str] = field(default_factory=list)


_BATCHES: dict[str, BatchRecord] = {}


def register_batch(batch_id: str, description: str = "", task_ids: list[str] | None = None) -> None:
    """Register or update a batch in memory and Redis (if enabled)."""
    rec = _BATCHES.get(batch_id)
    if not rec:
        rec = BatchRecord(batch_id=batch_id, description=description, task_ids=list(task_ids or []))
        _BATCHES[batch_id] = rec
    else:
        if description:
            rec.description = description
        if task_ids:
            for tid in task_ids:
                if tid not in rec.task_ids:
                    rec.task_ids.append(tid)
    # Persist to Redis if available
    r = _get_redis()
    if r:
        try:
            key = f"batch:{batch_id}"
            data = {"schema_version": 1, "description": rec.description, "created_at": datetime.now(timezone.utc).isoformat()}
            r.setex(key, _DEF_TTL, json.dumps(data))
            set_key = f"batch:{batch_id}:tasks"
            for tid in rec.task_ids:
                r.sadd(set_key, tid)
            r.expire(set_key, _DEF_TTL)
        except Exception:
            pass


def append_task(batch_id: str, task_id: str) -> None:
    rec = _BATCHES.get(batch_id)
    if not rec:
        rec = BatchRecord(batch_id=batch_id, task_ids=[task_id])
        _BATCHES[batch_id] = rec
    else:
        if task_id not in rec.task_ids:
            rec.task_ids.append(task_id)
    r = _get_redis()
    if r:
        try:
            r.sadd(f"batch:{batch_id}:tasks", task_id)
            r.expire(f"batch:{batch_id}:tasks", _DEF_TTL)
        except Exception:
            pass


def get_batch(batch_id: str) -> BatchRecord | None:
    rec = _BATCHES.get(batch_id)
    if rec:
        return rec
    r = _get_redis()
    if not r:
        return None
    try:
        data = r.get(f"batch:{batch_id}")
        if not data:
            return None
        obj = json.loads(data)
        tasks = list(r.smembers(f"batch:{batch_id}:tasks") or [])
        rec = BatchRecord(batch_id=batch_id, description=obj.get("description", ""), task_ids=tasks)
        _BATCHES[batch_id] = rec
        return rec
    except Exception:
        return None


def get_batches() -> dict[str, BatchRecord]:
    return _BATCHES


def list_batches(limit: int = 200, offset: int = 0) -> list[BatchRecord]:
    """Return batches, preferring Redis-sourced metadata when available."""
    r = _get_redis()
    items: list[BatchRecord] = []
    if r:
        try:
            # naive scan based listing
            cursor = 0
            keys: list[str] = []
            while True:
                cursor, ks = r.scan(cursor=cursor, match="batch:*", count=200)
                keys.extend([k for k in ks if not k.endswith(":tasks")])
                if cursor == 0:
                    break
            keys.sort()
            for k in keys[offset: offset + limit]:
                try:
                    bid = k.split(":", 1)[1]
                    data = r.get(k)
                    obj = json.loads(data) if data else {}
                    tasks = list(r.smembers(f"batch:{bid}:tasks") or [])
                    items.append(BatchRecord(batch_id=bid, description=obj.get("description", ""), task_ids=tasks))
                except Exception:
                    continue
            if items:
                return items
        except Exception:
            pass
    # fallback to in-memory
    all_items = list(_BATCHES.values())
    return all_items[offset: offset + limit]


def find_batches_for_task(task_id: str) -> list[str]:
    out: list[str] = []
    for bid, rec in _BATCHES.items():
        if task_id in rec.task_ids:
            out.append(bid)
    # If Redis is available, attempt reverse lookup as well
    r = _get_redis()
    if r:
        try:
            cursor = 0
            while True:
                cursor, ks = r.scan(cursor=cursor, match="batch:*:tasks", count=200)
                for k in ks:
                    if r.sismember(k, task_id):
                        bid = k.split(":")[1]
                        if bid not in out:
                            out.append(bid)
                if cursor == 0:
                    break
        except Exception:
            pass
    return out

# Optional: persistence to disk for process restarts
import os  # noqa: E402

_BATCH_FILE = os.path.expanduser("~/.zen_mcp_batches.json")


def save_batches_to_disk() -> None:
    try:
        data = {bid: {"description": rec.description, "task_ids": rec.task_ids} for bid, rec in _BATCHES.items()}
        with open(_BATCH_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def load_batches_from_disk() -> None:
    try:
        if not os.path.exists(_BATCH_FILE):
            return
        with open(_BATCH_FILE) as f:
            data = json.load(f)
        for bid, rec in data.items():
            _BATCHES[bid] = BatchRecord(batch_id=bid, description=rec.get("description", ""), task_ids=rec.get("task_ids", []))
    except Exception:
        pass
