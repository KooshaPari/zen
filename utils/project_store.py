"""
ProjectStore (MVP)

Redis-backed with in-memory fallback. Stores Projects, membership (agents), channels,
and artifacts registry. Provides minimal graph view suitable for injection and ACLs.

Env:
- ZEN_STORAGE=redis to enable Redis persistence
- PROJECT_TTL_SEC optional TTL for project hashes (no TTL by default)
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


_MEM_PROJECTS: dict[str, dict[str, Any]] = {}
_MEM_PROJECT_AGENTS: dict[str, list[str]] = {}
_MEM_PROJECT_CHANNELS: dict[str, list[str]] = {}
_MEM_PROJECT_ARTIFACTS: dict[str, list[dict[str, Any]]] = {}


def _get_redis() -> Any | None:
    if os.getenv("ZEN_STORAGE", os.getenv("ZEN_STORAGE_MODE", "memory")).lower() != "redis":
        return None


def _get_pg() -> Any | None:
    """Get Postgres connection when ZEN_STORAGE=postgres or PG_DSN set."""
    if (os.getenv("ZEN_STORAGE", os.getenv("ZEN_STORAGE_MODE", "")).lower() != "postgres"
        and not os.getenv("PG_DSN")):
        return None
    try:
        import psycopg
    except Exception:
        logger.warning("ProjectStore: psycopg not installed; skipping Postgres")
        return None
    try:
        dsn = os.getenv("PG_DSN") or (
            f"dbname={os.getenv('PG_DATABASE','zen_mcp')} user={os.getenv('PG_USER','postgres')} "
            f"password={os.getenv('PG_PASSWORD','')} host={os.getenv('PG_HOST','localhost')} port={os.getenv('PG_PORT','5432')}"
        )
        conn = psycopg.connect(dsn, autocommit=True)
        _ensure_pg_schema(conn)
        return conn
    except Exception as e:
        logger.warning(f"ProjectStore: Postgres unavailable, using memory: {e}")
        return None


def _ensure_pg_schema(conn: Any) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    description TEXT,
                    created_at BIGINT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS project_agents (
                    pid TEXT NOT NULL,
                    agent_id TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_project_agents ON project_agents (pid);
                CREATE TABLE IF NOT EXISTS project_channels (
                    pid TEXT NOT NULL,
                    channel_id TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_project_channels ON project_channels (pid);
                CREATE TABLE IF NOT EXISTS project_artifacts (
                    pid TEXT NOT NULL,
                    idx BIGSERIAL PRIMARY KEY,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_project_artifacts ON project_artifacts (pid, idx);
                """
            )
    except Exception:
        pass
    if not redis:
        return None
    try:
        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB_PROJECTS", os.getenv("REDIS_DB", "3"))),
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        client.ping()
        return client
    except Exception as e:
        logger.warning(f"ProjectStore: Redis unavailable, using memory: {e}")
        return None


def create_project(name: str, owner: str, description: str = "") -> dict[str, Any]:
    r = _get_redis()
    pg = _get_pg()
    now = int(time.time() * 1000)
    pid = f"proj:{owner}:{int(time.time())}"
    data = {
        "id": pid,
        "name": name,
        "owner": owner,
        "description": description,
        "schema_version": 1,
        "created_at": now,
    }
    if r:
        try:
            ttl = int(os.getenv("PROJECT_TTL_SEC", "0"))
            if ttl > 0:
                r.setex(f"project:{pid}", ttl, json.dumps(data))
            else:
                r.set(f"project:{pid}", json.dumps(data))
            r.sadd("projects:index", pid)
        except Exception:
            pass
    if pg:
        try:
            with pg.cursor() as cur:
                cur.execute(
                    "INSERT INTO projects (id,name,owner,description,created_at) VALUES (%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING",
                    (pid, name, owner, description, now),
                )
        except Exception:
            pass
    _MEM_PROJECTS[pid] = data
    _MEM_PROJECT_AGENTS.setdefault(pid, [owner])
    _MEM_PROJECT_CHANNELS.setdefault(pid, [])
    _MEM_PROJECT_ARTIFACTS.setdefault(pid, [])
    return data


def get_project(pid: str) -> dict[str, Any] | None:
    r = _get_redis()
    pg = _get_pg()
    if pg:
        try:
            with pg.cursor() as cur:
                cur.execute("SELECT id,name,owner,description,created_at FROM projects WHERE id=%s", (pid,))
                row = cur.fetchone()
                if row:
                    return {"id": row[0], "name": row[1], "owner": row[2], "description": row[3], "schema_version": 1, "created_at": row[4]}
        except Exception:
            pass
    if r:
        try:
            raw = r.get(f"project:{pid}")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    return _MEM_PROJECTS.get(pid)


def add_agent(pid: str, agent_id: str) -> bool:
    r = _get_redis()
    pg = _get_pg()
    if r:
        try:
            r.sadd(f"project:{pid}:agents", agent_id)
            return True
        except Exception:
            return False
    if pg:
        try:
            with pg.cursor() as cur:
                cur.execute("INSERT INTO project_agents (pid,agent_id) VALUES (%s,%s)", (pid, agent_id))
            return True
        except Exception:
            return False
    _MEM_PROJECT_AGENTS.setdefault(pid, [])
    if agent_id not in _MEM_PROJECT_AGENTS[pid]:
        _MEM_PROJECT_AGENTS[pid].append(agent_id)
    return True


def list_agents(pid: str) -> list[str]:
    r = _get_redis()
    pg = _get_pg()
    if r:
        try:
            return sorted(r.smembers(f"project:{pid}:agents"))
        except Exception:
            return []
    if pg:
        try:
            with pg.cursor() as cur:
                cur.execute("SELECT agent_id FROM project_agents WHERE pid=%s", (pid,))
                return sorted([row[0] for row in cur.fetchall()])
        except Exception:
            return []
    return sorted(_MEM_PROJECT_AGENTS.get(pid, []))


def link_channel(pid: str, channel_id: str) -> None:
    r = _get_redis()
    pg = _get_pg()
    if r:
        try:
            r.sadd(f"project:{pid}:channels", channel_id)
            return
        except Exception:
            pass
    if pg:
        try:
            with pg.cursor() as cur:
                cur.execute("INSERT INTO project_channels (pid,channel_id) VALUES (%s,%s)", (pid, channel_id))
            return
        except Exception:
            pass
    _MEM_PROJECT_CHANNELS.setdefault(pid, [])
    if channel_id not in _MEM_PROJECT_CHANNELS[pid]:
        _MEM_PROJECT_CHANNELS[pid].append(channel_id)


def add_artifact(pid: str, artifact: dict[str, Any]) -> None:
    r = _get_redis()
    pg = _get_pg()
    if r:
        try:
            key = f"project:{pid}:artifacts"
            r.rpush(key, json.dumps(artifact))
            return
        except Exception:
            pass
    if pg:
        try:
            with pg.cursor() as cur:
                cur.execute("INSERT INTO project_artifacts (pid,data) VALUES (%s,%s)", (pid, json.dumps(artifact)))
            return
        except Exception:
            pass
    _MEM_PROJECT_ARTIFACTS.setdefault(pid, []).append(artifact)


def list_artifacts(pid: str, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    r = _get_redis()
    pg = _get_pg()
    if r:
        try:
            key = f"project:{pid}:artifacts"
            end = offset + limit - 1
            raw = r.lrange(key, offset, end)
            return [json.loads(x) for x in raw]
        except Exception:
            pass
    if pg:
        try:
            with pg.cursor() as cur:
                cur.execute("SELECT data FROM project_artifacts WHERE pid=%s ORDER BY idx ASC LIMIT %s OFFSET %s", (pid, int(limit), int(offset)))
                return [json.loads(row[0]) for row in cur.fetchall()]
        except Exception:
            pass
    return _MEM_PROJECT_ARTIFACTS.get(pid, [])[offset:offset+limit]


def get_graph(pid: str) -> dict[str, Any]:
    # Prefer channels from messaging_store (which supports Postgres),
    # fall back to Redis-linked channel set, then in-memory list
    channels: list[str] = []
    try:
        from utils.messaging_store import list_channels as _list_channels
        chs = _list_channels(pid, limit=1000, offset=0)
        channels = [c.get("id") for c in chs if isinstance(c, dict) and c.get("id")]
    except Exception:
        pass
    if not channels:
        r = _get_redis()
        if r:
            try:
                channels = sorted(r.smembers(f"project:{pid}:channels"))
            except Exception:
                channels = []
    if not channels:
        channels = _MEM_PROJECT_CHANNELS.get(pid, [])

    return {
        "project": get_project(pid),
        "agents": list_agents(pid),
        "channels": channels,
        "artifacts_count": len(_MEM_PROJECT_ARTIFACTS.get(pid, [])),
        "schema_version": 1,
    }
