import hashlib
import json
import os
from typing import Any, Optional

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore

# Best-effort in-memory fallback (non-persistent)
_MEM: dict[str, str] = {}


def _get_redis():
    if os.getenv("ZEN_STORAGE", os.getenv("ZEN_STORAGE_MODE", "memory")).lower() != "redis":
        return None
    if not redis:
        return None
    try:
        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "3")),  # separate DB for caching
            decode_responses=True,
            socket_timeout=3,
            socket_connect_timeout=3,
        )
        client.ping()
        return client
    except Exception:
        return None


def _sha(data: Any) -> str:
    s = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def build_generation_key(model: str, system_prompt: Optional[str], prompt: str, tools: Optional[Any] = None, project: Optional[str] = None) -> str:
    payload = {
        "model": model,
        "system": system_prompt or "",
        "prompt": prompt or "",
        "tools": tools or [],
        "project": project or "default",
    }
    return f"pc:generation:{_sha(payload)}"


def build_router_key(task_type: str, prompt: str, project: Optional[str] = None) -> str:
    payload = {
        "task_type": task_type,
        "prompt": prompt or "",
        "project": project or "default",
    }
    return f"pc:router:{_sha(payload)}"


def get_cached_generation(key: str) -> Optional[dict]:
    r = _get_redis()
    if r:
        try:
            v = r.get(key)
            return json.loads(v) if v else None
        except Exception:
            return None
    raw = _MEM.get(key)
    return json.loads(raw) if raw else None


def set_cached_generation(key: str, value: dict, ttl_seconds: int = 3600) -> None:
    r = _get_redis()
    if r:
        try:
            r.setex(key, ttl_seconds, json.dumps(value))
            return
        except Exception:
            pass
    _MEM[key] = json.dumps(value)


def get_cached_router_decision(key: str) -> Optional[dict]:
    return get_cached_generation(key)


def set_cached_router_decision(key: str, value: dict, ttl_seconds: int = 3600) -> None:
    set_cached_generation(key, value, ttl_seconds)

