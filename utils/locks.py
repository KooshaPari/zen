"""
Redis-backed locks with in-memory fallback.
- locks:wd:<normalized_path>
- locks:repo:<repo_id>

Use acquire_* with TTLs to avoid deadlocks. Non-blocking: returns bool.
"""
from __future__ import annotations

import hashlib
import os
import time

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore

# In-memory fallback (best-effort only, not cross-process)
_MEM_LOCKS: dict[str, float] = {}


def _norm(s: str) -> str:
    s = s.strip().lower()
    return hashlib.md5(s.encode("utf-8")).hexdigest()  # nosec


def _get_redis():
    if os.getenv("ZEN_STORAGE", os.getenv("ZEN_STORAGE_MODE", "memory")).lower() != "redis":
        return None
    if not redis:
        return None
    try:
        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "2")),  # separate DB for locks
            decode_responses=True,
            socket_timeout=3,
            socket_connect_timeout=3,
        )
        client.ping()
        return client
    except Exception:
        return None


def _acquire_mem(key: str, ttl: int) -> bool:
    now = time.time()
    exp = _MEM_LOCKS.get(key, 0)
    if exp > now:
        return False
    _MEM_LOCKS[key] = now + ttl
    return True


def _release_mem(key: str) -> None:
    _MEM_LOCKS.pop(key, None)


def acquire_wd_lock(path: str | None, ttl: int = 600) -> bool:
    if not path:
        return True
    key = f"locks:wd:{_norm(path)}"
    r = _get_redis()
    if r:
        try:
            return bool(r.set(key, "1", nx=True, ex=ttl))
        except Exception:
            pass
    return _acquire_mem(key, ttl)


def release_wd_lock(path: str | None) -> None:
    if not path:
        return
    key = f"locks:wd:{_norm(path)}"
    r = _get_redis()
    if r:
        try:
            r.delete(key)
            return
        except Exception:
            pass
    _release_mem(key)


def acquire_repo_lock(repo_id: str | None, ttl: int = 600) -> bool:
    if not repo_id:
        return True
    key = f"locks:repo:{_norm(repo_id)}"
    r = _get_redis()
    if r:
        try:
            return bool(r.set(key, "1", nx=True, ex=ttl))
        except Exception:
            pass
    return _acquire_mem(key, ttl)


def release_repo_lock(repo_id: str | None) -> None:
    if not repo_id:
        return
    key = f"locks:repo:{_norm(repo_id)}"
    r = _get_redis()
    if r:
        try:
            r.delete(key)
            return
        except Exception:
            pass
    _release_mem(key)

