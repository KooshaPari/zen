"""
Simple Redis-backed rate limiters with in-memory fallback.
Use token-bucket style: allow() returns True if within limit.
Keys:
- ratelimit:<scope>:<window>
"""
from __future__ import annotations

import os
import time

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore

_MEM: dict[str, tuple[int, float]] = {}


def _get_redis():
    if os.getenv("ZEN_STORAGE", os.getenv("ZEN_STORAGE_MODE", "memory")).lower() != "redis":
        return None
    if not redis:
        return None
    try:
        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "2")),
            decode_responses=True,
        )
        client.ping()
        return client
    except Exception:
        return None


def allow(scope: str, max_per_window: int, window_seconds: int = 60) -> bool:
    r = _get_redis()
    key = f"ratelimit:{scope}:{window_seconds}"
    now = int(time.time())
    if r:
        try:
            with r.pipeline() as p:  # type: ignore[attr-defined]
                p.incr(key)
                p.expire(key, window_seconds)
                count, _ = p.execute()
            return int(count) <= max_per_window
        except Exception:
            pass
    # Fallback memory bucket
    count, exp = _MEM.get(key, (0, now + window_seconds))
    if now > exp:
        count, exp = 0, now + window_seconds
    count += 1
    _MEM[key] = (count, exp)
    return count <= max_per_window

