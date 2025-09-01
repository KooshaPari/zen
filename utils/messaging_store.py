import os
import time
import uuid  # noqa: E402
from typing import Any, Optional

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore

# Optional Postgres (psycopg3) support
_PG_CONN: Any | None = None

def _get_pg():
    global _PG_CONN
    mode = (os.getenv("ZEN_STORAGE_MODE") or os.getenv("ZEN_STORAGE") or "").lower()
    if (mode != "postgres" and not os.getenv("PG_DSN")):
        return None
    try:
        import psycopg  # type: ignore
    except Exception:
        return None
    try:
        dsn = os.getenv("PG_DSN") or (
            f"dbname={os.getenv('PG_DATABASE','zen_mcp')} user={os.getenv('PG_USER','postgres')} "
            f"password={os.getenv('PG_PASSWORD','')} host={os.getenv('PG_HOST','localhost')} port={os.getenv('PG_PORT','5432')}"
        )
        if _PG_CONN is None:
            _PG_CONN = psycopg.connect(dsn, autocommit=True)
            _ensure_pg_schema(_PG_CONN)
        return _PG_CONN
    except Exception:
        return None


def _ensure_pg_schema(conn) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS channels (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    visibility TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at BIGINT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_channels_project ON channels (project_id);

                CREATE TABLE IF NOT EXISTS channel_members (
                    channel_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_channel_members ON channel_members (channel_id);

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    channel_id TEXT,
                    members TEXT,
                    root TEXT,
                    sender TEXT NOT NULL,
                    body TEXT NOT NULL,
                    mentions TEXT,
                    blocking BOOLEAN,
                    resume_token TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_ts BIGINT,
                    ts BIGINT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_channel_ts ON messages (channel_id, ts DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_members_ts ON messages (members, ts DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_root_ts ON messages (root, ts DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_resume ON messages (resume_token);
                """
            )
    except Exception:
        pass

# In-memory fallback

_MEM_CHANNELS: dict[str, list[dict]] = {}
_MEM_DMS: dict[tuple[str, str], list[dict]] = {}
_MEM_UNREAD: dict[str, int] = {}
_MEM_MESSAGES: dict[str, dict] = {}
_MEM_RESUME_TOKENS: dict[str, str] = {}


def _get_redis():
    mode = (os.getenv("ZEN_STORAGE_MODE") or os.getenv("ZEN_STORAGE") or "memory").lower()
    if mode != "redis":
        return None
    if not redis:
        return None
    try:
        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "2")),
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        client.ping()
        return client
    except Exception:
        return None

# Channel and Thread registries (in-memory fallback)
_MEM_CHANNEL_META: dict[str, dict] = {}
_MEM_THREAD_MSGS: dict[str, list[str]] = {}



def create_channel(project_id: str, name: str, visibility: str = "project", created_by: str = "system") -> dict:
    r = _get_redis()
    channel_id = f"ch:{project_id}:{name}"
    meta = {"id": channel_id, "project_id": project_id, "name": name, "visibility": visibility, "created_by": created_by, "created_at": int(time.time()*1000)}
    # Postgres
    pg = _get_pg()
    if pg:
        try:
            with pg.cursor() as cur:
                cur.execute(
                    "INSERT INTO channels (id,project_id,name,visibility,created_by,created_at) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING",
                    (channel_id, project_id, name, visibility, created_by, meta["created_at"]),
                )
        except Exception:
            pass
    if r:
        try:
            r.hset(f"channel:{channel_id}", mapping=meta)  # type: ignore[arg-type]
            r.sadd(f"project:{project_id}:channels", channel_id)
        except Exception:
            pass
    _MEM_CHANNEL_META[channel_id] = meta
    _MEM_CHANNELS.setdefault(channel_id, [])
    # Best-effort linkage for project graph
    try:
        from utils.project_store import link_channel  # local import to avoid cycle
        link_channel(project_id, channel_id)
    except Exception:
        pass
    return meta


def list_channels(project_id: str, limit: int = 100, offset: int = 0) -> list[dict]:
    r = _get_redis()
    pg = _get_pg()
    if pg:
        try:
            with pg.cursor() as cur:
                cur.execute(
                    "SELECT id,project_id,name,visibility,created_by,created_at FROM channels WHERE project_id=%s ORDER BY created_at DESC LIMIT %s OFFSET %s",
                    (project_id, int(limit), int(offset)),
                )
                out: list[dict] = []
                for row in cur.fetchall():
                    out.append({
                        "id": row[0],
                        "project_id": row[1],
                        "name": row[2],
                        "visibility": row[3],
                        "created_by": row[4],
                        "created_at": row[5],
                    })
                return out
        except Exception:
            pass
    if r:
        try:
            ids = list(r.smembers(f"project:{project_id}:channels"))
            ids.sort()
            ids = ids[offset:offset+limit]
            out: list[dict] = []
            for cid in ids:
                m = r.hgetall(f"channel:{cid}")
                if m:
                    if "created_at" in m:
                        try:
                            m["created_at"] = int(m["created_at"])  # type: ignore[assignment]
                        except Exception:
                            pass
                    out.append(m)
            return out
        except Exception:
            pass
    # fallback
    all_ids = [cid for cid, meta in _MEM_CHANNEL_META.items() if meta.get("project_id") == project_id]
    all_ids.sort()
    return [ _MEM_CHANNEL_META[cid] for cid in all_ids[offset:offset+limit] ]


def reply_thread(channel_id: str, root_message_id: str, from_id: str, body: str) -> dict:
    r = _get_redis()
    ts = int(time.time()*1000)
    msg_id = f"th:{root_message_id}:{ts}:{uuid.uuid4().hex[:6]}"
    msg = {"id": msg_id, "type": "thread", "channel_id": channel_id, "root": root_message_id, "from": from_id, "body": body, "ts": ts}
    # Postgres write-through
    pg = _get_pg()
    if pg:
        try:
            with pg.cursor() as cur:
                cur.execute(
                    "INSERT INTO messages (id,type,channel_id,members,root,sender,body,mentions,blocking,resume_token,resolved,resolved_ts,ts) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING",
                    (msg_id, "thread", channel_id, None, root_message_id, from_id, body, None, False, None, False, None, ts),
                )
        except Exception:
            pass
    if r:
        try:
            r.hset(f"message:{msg_id}", mapping=msg)  # type: ignore[arg-type]
            r.rpush(f"thread:{root_message_id}:messages", msg_id)
            return msg
        except Exception:
            pass
    _MEM_MESSAGES[msg_id] = msg
    _MEM_THREAD_MSGS.setdefault(root_message_id, []).append(msg_id)
    return msg


def get_thread_history(root_message_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
    r = _get_redis()
    pg = _get_pg()
    if pg:
        try:
            with pg.cursor() as cur:
                cur.execute(
                    "SELECT id,type,channel_id,members,root,sender,body,mentions,blocking,resume_token,resolved,resolved_ts,ts FROM messages WHERE root=%s ORDER BY ts DESC LIMIT %s OFFSET %s",
                    (root_message_id, int(limit), int(offset)),
                )
                out: list[dict] = []
                for row in cur.fetchall():
                    out.append({
                        "id": row[0],
                        "type": row[1],
                        "channel_id": row[2],
                        "members": (row[3].split(',') if row[3] else None),
                        "root": row[4],
                        "from": row[5],
                        "body": row[6],
                        "mentions": (row[7].split(',') if row[7] else []),
                        "blocking": bool(row[8]),
                        "resume_token": row[9],
                        "resolved": bool(row[10]) if row[10] is not None else False,
                        "resolved_ts": row[11],
                        "ts": row[12],
                    })
                return out
        except Exception:
            pass
    if r:
        try:
            end = offset + limit - 1
            ids = r.lrange(f"thread:{root_message_id}:messages", offset, end)
            out: list[dict] = []
            for mid in ids:
                m = r.hgetall(f"message:{mid}")
                if m:
                    if "ts" in m:
                        try:
                            m["ts"] = int(m["ts"])  # type: ignore[assignment]
                        except Exception:
                            pass
                    out.append(m)
            return out
        except Exception:
            pass
    ids = _MEM_THREAD_MSGS.get(root_message_id, [])[offset:offset+limit]
    return [ _MEM_MESSAGES[mid] for mid in ids if mid in _MEM_MESSAGES ]


def _normalize_dm(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted([a, b]))  # deterministic key order


def post_channel_message(channel_id: str, from_id: str, body: str,
                         *, mentions: Optional[list[str]] = None,
                         artifacts: Optional[list[str]] = None,
                         blocking: bool = False,
                         importance: str = "normal") -> dict:
    ts = int(time.time() * 1000)
    msg_id = f"c:{channel_id}:{ts}"
    import uuid as _uuid
    msg = {
        "id": msg_id,
        "type": "channel",
        "channel_id": channel_id,
        "from": from_id,
        "body": body,
        "mentions": mentions or [],
        "artifacts": artifacts or [],
        "blocking": bool(blocking),
        "importance": importance,
        "ts": ts,
    }
    if blocking:
        token = _uuid.uuid4().hex
        msg["resume_token"] = token
    r = _get_redis()
    # Postgres write-through
    pg = _get_pg()
    if pg:
        try:
            with pg.cursor() as cur:
                cur.execute(
                    "INSERT INTO messages (id,type,channel_id,members,root,sender,body,mentions,blocking,resume_token,resolved,resolved_ts,ts) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING",
                    (
                        msg_id, "channel", channel_id, None, None, from_id, body,
                        (",".join(mentions or []) if mentions else None), bool(blocking),
                        msg.get("resume_token"), False, None, ts,
                    ),
                )
        except Exception:
            pass
    if r:
        try:
            # Store message hash and push to list for channel history
            r.hset(f"message:{msg_id}", mapping=msg)  # type: ignore[arg-type]
            r.rpush(f"messages:channel:{channel_id}", msg_id)
            if blocking:
                try:
                    r.set(f"resume:token:{msg.get('resume_token')}", msg_id, ex=int(os.getenv("MSG_RESUME_TTL", "86400")))
                except Exception:
                    pass
            # Add to inbox of mentioned agents
            for agent in (mentions or []):
                try:
                    r.zadd(f"inbox:agent:{agent}:unread", {msg_id: ts})
                except Exception:
                    pass
            return msg
        except Exception:
            pass
    # In-memory fallback
    _MEM_CHANNELS.setdefault(channel_id, []).append(msg)
    _MEM_MESSAGES[msg_id] = msg
    if blocking and msg.get("resume_token"):
        _MEM_RESUME_TOKENS[msg["resume_token"]] = msg_id
    for agent in (mentions or []):
        _MEM_UNREAD.setdefault(agent, 0)
        _MEM_UNREAD[agent] += 1
    return msg


def get_channel_history(channel_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
    r = _get_redis()
    pg = _get_pg()
    if pg:
        try:
            with pg.cursor() as cur:
                cur.execute(
                    "SELECT id,type,channel_id,members,root,sender,body,mentions,blocking,resume_token,resolved,resolved_ts,ts FROM messages WHERE channel_id=%s ORDER BY ts DESC LIMIT %s OFFSET %s",
                    (channel_id, int(limit), int(offset)),
                )
                out: list[dict] = []
                for row in cur.fetchall():
                    out.append({
                        "id": row[0],
                        "type": row[1],
                        "channel_id": row[2],
                        "members": (row[3].split(',') if row[3] else None),
                        "root": row[4],
                        "from": row[5],
                        "body": row[6],
                        "mentions": (row[7].split(',') if row[7] else []),
                        "blocking": bool(row[8]),
                        "resume_token": row[9],
                        "resolved": bool(row[10]) if row[10] is not None else False,
                        "resolved_ts": row[11],
                        "ts": row[12],
                    })
                return out
        except Exception:
            pass
    if r:
        try:
            end = offset + limit - 1
            ids = r.lrange(f"messages:channel:{channel_id}", offset, end)
            out: list[dict] = []
            for mid in ids:
                try:
                    m = r.hgetall(f"message:{mid}")
                    if m:
                        if "ts" in m:
                            try:
                                m["ts"] = int(m["ts"])  # type: ignore[assignment]
                            except Exception:
                                pass
                        out.append(m)
                except Exception:
                    continue
            return out
        except Exception:
            pass
    msgs = _MEM_CHANNELS.get(channel_id, [])
    return msgs[offset:offset + limit]


def post_dm_message(a: str, b: str, from_id: str, body: str,
                    *, blocking: bool = False,
                    importance: str = "normal") -> dict:
    a, b = _normalize_dm(a, b)
    ts = int(time.time() * 1000)
    msg_id = f"dm:{a}:{b}:{ts}"
    import uuid as _uuid
    msg = {
        "id": msg_id,
        "type": "dm",
        "members": [a, b],
        "from": from_id,
        "body": body,
        "blocking": bool(blocking),
        "importance": importance,
        "ts": ts,
    }
    if blocking:
        token = _uuid.uuid4().hex
        msg["resume_token"] = token
    r = _get_redis()
    # Postgres write-through
    pg = _get_pg()
    if pg:
        try:
            with pg.cursor() as cur:
                members_csv = ",".join([a, b])
                cur.execute(
                    "INSERT INTO messages (id,type,channel_id,members,root,sender,body,mentions,blocking,resume_token,resolved,resolved_ts,ts) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING",
                    (
                        msg_id, "dm", None, members_csv, None, from_id, body,
                        None, bool(blocking), msg.get("resume_token"), False, None, ts,
                    ),
                )
        except Exception:
            pass
    key = f"messages:dm:{a}:{b}"
    if r:
        try:
            r.hset(f"message:{msg_id}", mapping=msg)  # type: ignore[arg-type]
            r.rpush(key, msg_id)
            if blocking:
                try:
                    r.set(f"resume:token:{msg.get('resume_token')}", msg_id, ex=int(os.getenv("MSG_RESUME_TTL", "86400")))
                except Exception:
                    pass
            # Inbox for both members except the sender
            for member in (a, b):
                if member != from_id:
                    try:
                        r.zadd(f"inbox:agent:{member}:unread", {msg_id: ts})
                    except Exception:
                        pass
            return msg
        except Exception:
            pass
    _MEM_DMS.setdefault((a, b), []).append(msg)
    _MEM_MESSAGES[msg_id] = msg
    if blocking and msg.get("resume_token"):
        _MEM_RESUME_TOKENS[msg["resume_token"]] = msg_id
    for member in (a, b):
        if member != from_id:
            _MEM_UNREAD[member] = _MEM_UNREAD.get(member, 0) + 1
    return msg


def get_dm_history(a: str, b: str, limit: int = 50, offset: int = 0) -> list[dict]:
    a, b = _normalize_dm(a, b)
    r = _get_redis()
    pg = _get_pg()
    if pg:
        try:
            with pg.cursor() as cur:
                mems = ",".join([a, b])
                cur.execute(
                    "SELECT id,type,channel_id,members,root,sender,body,mentions,blocking,resume_token,resolved,resolved_ts,ts FROM messages WHERE members=%s ORDER BY ts DESC LIMIT %s OFFSET %s",
                    (mems, int(limit), int(offset)),
                )
                out: list[dict] = []
                for row in cur.fetchall():
                    out.append({
                        "id": row[0],
                        "type": row[1],
                        "channel_id": row[2],
                        "members": (row[3].split(',') if row[3] else None),
                        "root": row[4],
                        "from": row[5],
                        "body": row[6],
                        "mentions": (row[7].split(',') if row[7] else []),
                        "blocking": bool(row[8]),
                        "resume_token": row[9],
                        "resolved": bool(row[10]) if row[10] is not None else False,
                        "resolved_ts": row[11],
                        "ts": row[12],
                    })
                return out
        except Exception:
            pass
    key = f"messages:dm:{a}:{b}"
    if r:
        try:
            end = offset + limit - 1
            ids = r.lrange(key, offset, end)
            out: list[dict] = []
            for mid in ids:
                try:
                    m = r.hgetall(f"message:{mid}")
                    if m:
                        if "ts" in m:
                            try:
                                m["ts"] = int(m["ts"])  # type: ignore[assignment]
                            except Exception:
                                pass
                        out.append(m)
                except Exception:
                    continue
            return out
        except Exception:
            pass
    msgs = _MEM_DMS.get((a, b), [])
    return msgs[offset:offset + limit]


def get_unread_summary(agent_id: str) -> dict:
    r = _get_redis()
    if r:
        try:
            # Count ZSET size (unread messages for this agent)
            zkey = f"inbox:agent:{agent_id}:unread"
            total = int(r.zcard(zkey) or 0)
            # Optional: sample a few latest IDs
            ids = r.zrevrange(zkey, 0, 9)
            return {"total": total, "sample": ids}
        except Exception:
            pass
    # Fallback: only approximate total from in-memory
    return {"total": int(_MEM_UNREAD.get(agent_id, 0)), "sample": []}


def mark_read(agent_id: str, message_id: str) -> bool:
    """Mark a message as read for an agent (removes from unread ZSET)."""
    r = _get_redis()
    if r:
        try:
            removed = r.zrem(f"inbox:agent:{agent_id}:unread", message_id)
            return bool(removed)
        except Exception:
            return False
    # In-memory: decrement count if present
    if _MEM_UNREAD.get(agent_id, 0) > 0:
        _MEM_UNREAD[agent_id] -= 1
    return True


def get_unread(agent_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
    """List unread messages for an agent with basic pagination."""
    r = _get_redis()
    if r:
        try:
            start = offset
            end = offset + max(0, limit - 1)
            ids = r.zrevrange(f"inbox:agent:{agent_id}:unread", start, end)
            out: list[dict] = []
            for mid in ids:
                m = r.hgetall(f"message:{mid}")
                if m:
                    if "ts" in m:
                        try:
                            m["ts"] = int(m["ts"])  # type: ignore[assignment]
                        except Exception:
                            pass
                    out.append(m)
            return out
        except Exception:
            return []
    # Fallback: not tracking individual messages in memory yet
    return []


def get_message(message_id: str) -> Optional[dict]:
    """Retrieve a message by ID from Redis or memory."""
    r = _get_redis()
    pg = _get_pg()
    if pg:
        try:
            with pg.cursor() as cur:
                cur.execute(
                    "SELECT id,type,channel_id,members,root,sender,body,mentions,blocking,resume_token,resolved,resolved_ts,ts FROM messages WHERE id=%s",
                    (message_id,),
                )
                row = cur.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "type": row[1],
                        "channel_id": row[2],
                        "members": (row[3].split(',') if row[3] else None),
                        "root": row[4],
                        "from": row[5],
                        "body": row[6],
                        "mentions": (row[7].split(',') if row[7] else []),
                        "blocking": bool(row[8]),
                        "resume_token": row[9],
                        "resolved": bool(row[10]) if row[10] is not None else False,
                        "resolved_ts": row[11],
                        "ts": row[12],
                    }
        except Exception:
            pass
    if r:
        try:
            m = r.hgetall(f"message:{message_id}")
            if m:
                if "ts" in m:
                    try:
                        m["ts"] = int(m["ts"])  # type: ignore[assignment]
                    except Exception:
                        pass
                return m
        except Exception:
            return None
    return _MEM_MESSAGES.get(message_id)


def mark_block_resolved(message_id: str, agent_id: Optional[str] = None) -> bool:
    """Mark a blocking message as resolved, recording resolver and timestamp.

    Adds fields: resolved=true, resolved_ts, resolved_by (set of agent_ids as CSV).
    """
    ts = int(time.time() * 1000)
    r = _get_redis()
    pg = _get_pg()
    if pg:
        try:
            with pg.cursor() as cur:
                cur.execute("UPDATE messages SET resolved=TRUE, resolved_ts=%s WHERE id=%s", (ts, message_id))
            # No storage for resolved_by in PG for now
        except Exception:
            pass
    if r:
        try:
            # Update resolved fields
            r.hset(f"message:{message_id}", mapping={"resolved": True, "resolved_ts": ts})
            if agent_id:
                prev = r.hget(f"message:{message_id}", "resolved_by")
                if prev:
                    vals = set(prev.split(","))
                    vals.add(agent_id)
                    r.hset(f"message:{message_id}", "resolved_by", ",".join(sorted(vals)))
                else:
                    r.hset(f"message:{message_id}", "resolved_by", agent_id)
            return True
        except Exception:
            return False
    # Memory fallback
    msg = _MEM_MESSAGES.get(message_id)
    if not msg:
        return False
    msg["resolved"] = True
    msg["resolved_ts"] = ts
    if agent_id:
        rb = set(msg.get("resolved_by") or []) if isinstance(msg.get("resolved_by"), list) else set()
        rb_list = list(rb | {agent_id})
        msg["resolved_by"] = rb_list
    _MEM_MESSAGES[message_id] = msg
    return True


def resolve_by_token(resume_token: str, reply_body: Optional[str] = None, from_id: Optional[str] = None) -> dict:
    """Resolve a blocking message by resume_token and optionally add a reply.

    Returns a dict with fields: { ok, message, reply_id? }
    """
    r = _get_redis()
    msg_id: Optional[str] = None
    if msg_id is None:
        pg = _get_pg()
        if pg:
            try:
                with pg.cursor() as cur:
                    cur.execute("SELECT id FROM messages WHERE resume_token=%s", (resume_token,))
                    row = cur.fetchone()
                    if row:
                        msg_id = row[0]
            except Exception:
                pass
    if r:
        try:
            msg_id = r.get(f"resume:token:{resume_token}")
        except Exception:
            msg_id = None
    if not msg_id:
        msg_id = _MEM_RESUME_TOKENS.get(resume_token)
    if not msg_id:
        return {"ok": False, "error": "invalid_resume_token"}
    # Mark resolved
    ok = mark_block_resolved(msg_id, agent_id=from_id)
    reply_id = None
    msg = get_message(msg_id) or {}
    if ok and reply_body and from_id:
        try:
            if msg.get("type") == "channel":
                ch_id = msg.get("channel_id")
                rep = reply_thread(ch_id, msg_id, from_id, reply_body)
                reply_id = rep.get("id")
            elif msg.get("type") == "dm":
                a, b = tuple(sorted(msg.get("members", []))) if msg.get("members") else (None, None)
                if a and b:
                    rep = post_dm_message(a, b, from_id, reply_body)
                    reply_id = rep.get("id")
        except Exception:
            pass
    return {"ok": ok, "message": get_message(msg_id), "reply_id": reply_id}



from typing import Optional as _Opt  # noqa: E402

# --- ACL helpers and channel meta access ---


def get_channel_meta(channel_id: str) -> _Opt[dict]:
    r = _get_redis()
    pg = _get_pg()
    if pg:
        try:
            with pg.cursor() as cur:
                cur.execute("SELECT id,project_id,name,visibility,created_by,created_at FROM channels WHERE id=%s", (channel_id,))
                row = cur.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "project_id": row[1],
                        "name": row[2],
                        "visibility": row[3],
                        "created_by": row[4],
                        "created_at": row[5],
                    }
        except Exception:
            return None
    if r:
        try:
            m = r.hgetall(f"channel:{channel_id}")
            if m:
                if "created_at" in m:
                    try:
                        m["created_at"] = int(m["created_at"])  # type: ignore[assignment]
                    except Exception:
                        pass
                return m
        except Exception:
            return None
    return _MEM_CHANNEL_META.get(channel_id)


def add_channel_member(channel_id: str, agent_id: str) -> bool:
    r = _get_redis()
    pg = _get_pg()
    if pg:
        try:
            with pg.cursor() as cur:
                cur.execute("INSERT INTO channel_members (channel_id,agent_id) VALUES (%s,%s)", (channel_id, agent_id))
            return True
        except Exception:
            return False
    if r:
        try:
            r.sadd(f"channel:{channel_id}:members", agent_id)
            return True
        except Exception:
            return False
    meta = _MEM_CHANNEL_META.get(channel_id)
    if meta is None:
        return False
    meta.setdefault("members", [])
    if agent_id not in meta["members"]:
        meta["members"].append(agent_id)
    return True


def get_channel_members(channel_id: str) -> list[str]:
    r = _get_redis()
    pg = _get_pg()
    if pg:
        try:
            with pg.cursor() as cur:
                cur.execute("SELECT agent_id FROM channel_members WHERE channel_id=%s", (channel_id,))
                return sorted([row[0] for row in cur.fetchall()])
        except Exception:
            return []
    if r:
        try:
            return list(r.smembers(f"channel:{channel_id}:members"))
        except Exception:
            return []
    meta = _MEM_CHANNEL_META.get(channel_id)
    if not meta:
        return []
    return list(meta.get("members", []))
