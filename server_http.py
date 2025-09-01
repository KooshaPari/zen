"""
In-memory aiohttp app for HTTP tests.

Implements minimal endpoints used by tests:
- GET /models/catalog
- POST /router/decide
- POST /channels
- POST /messages/channel
- GET  /inbox/messages
- POST /messages/resume
- POST /threads/resume
- POST /tasks (with router info when model=auto)
- GET  /tasks.csv (with ETag support)
- POST /a2a/message (with simple ACL by env)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from typing import Any, Dict, List

from aiohttp import web


def build_app() -> web.Application:
    app = web.Application()

    # In-memory state
    state: Dict[str, Any] = {
        "channels": {},            # id -> channel
        "messages": {},            # id -> message
        "resume_index": {},        # resume_token -> message_id
        "inbox_unread": {},        # agent_id -> count
        "tasks": [],               # list of task dicts
        "tasks_csv_etag": None,    # last ETag
    }
    app["state"] = state

    # Helpers
    def _uuid() -> str:
        return str(uuid.uuid4())

    def _json(request: web.Request) -> Dict[str, Any]:
        return request.get("_json_cache")  # set by middleware

    @web.middleware
    async def json_cache_mw(request: web.Request, handler):
        if request.can_read_body and request.content_type.startswith("application/json"):
            try:
                request["_json_cache"] = await request.json()
            except Exception:
                request["_json_cache"] = {}
        return await handler(request)

    app.middlewares.append(json_cache_mw)

    # Routes
    async def get_models_catalog(request: web.Request) -> web.Response:
        return web.json_response({"models": []})

    async def post_router_decide(request: web.Request) -> web.Response:
        body = _json(request) or {}
        task_type = body.get("task_type", "general")
        chosen = "gemini-2.5-flash" if task_type in ("quick_qa", "chat") else "gpt-4o-mini"
        decision = {
            "chosen_model": chosen,
            "budgets": {"request_tokens": 4096, "daily_cost_usd": 1.0},
            "plan": {"steps": ["analyze", "respond"]},
        }
        return web.json_response({"decision": decision})

    async def post_channels(request: web.Request) -> web.Response:
        body = _json(request) or {}
        ch = {
            "id": _uuid(),
            "project_id": body.get("project_id"),
            "name": body.get("name"),
            "created_by": body.get("created_by"),
            "created_at": time.time(),
        }
        state["channels"][ch["id"]] = ch
        return web.json_response({"channel": ch}, status=201)

    async def post_message_channel(request: web.Request) -> web.Response:
        body = _json(request) or {}
        mid = _uuid()
        msg = {
            "id": mid,
            "channel_id": body.get("channel_id"),
            "from": body.get("from"),
            "body": body.get("body", ""),
            "mentions": body.get("mentions", []),
            "blocking": bool(body.get("blocking")),
            "resolved": False,
        }
        if msg["blocking"]:
            token = _uuid()
            msg["resume_token"] = token
            state["resume_index"][token] = mid
        # Update inbox counts for mentions
        for agent in msg["mentions"] or []:
            state["inbox_unread"][agent] = state["inbox_unread"].get(agent, 0) + 1
        state["messages"][mid] = msg
        return web.json_response({"message": msg}, status=201)

    async def get_inbox_messages(request: web.Request) -> web.Response:
        agent_id = request.query.get("agent_id", "")
        total = int(state["inbox_unread"].get(agent_id, 0))
        return web.json_response({"unread": {"total": total}})

    async def post_messages_resume(request: web.Request) -> web.Response:
        body = _json(request) or {}
        message_id = body.get("message_id")
        agent_id = body.get("agent_id")
        msg = state["messages"].get(message_id)
        if not msg:
            return web.json_response({"ok": False, "error": "not_found"}, status=404)
        msg["resolved"] = True
        if agent_id:
            # Decrement unread for the agent, but not below zero
            current = state["inbox_unread"].get(agent_id, 0)
            state["inbox_unread"][agent_id] = max(0, current - 1)
        return web.json_response({"ok": True, "message": msg})

    async def post_threads_resume(request: web.Request) -> web.Response:
        body = _json(request) or {}
        token = body.get("resume_token")
        message_id = state["resume_index"].get(token)
        if not message_id:
            return web.json_response({"ok": False, "error": "invalid_token"}, status=400)
        reply_id = _uuid()
        # Mark original as resolved
        msg = state["messages"].get(message_id)
        if msg:
            msg["resolved"] = True
        return web.json_response({"ok": True, "reply_id": reply_id})

    def _compute_tasks_csv_and_etag(tasks: List[Dict[str, Any]]):
        header = [
            "task_id",
            "agent_type",
            "message",
            "model",
            "created_at",
        ]
        lines = [",".join(header)]
        for t in tasks:
            row = [
                t.get("task_id", ""),
                t.get("agent_type", ""),
                (t.get("message", "").replace("\n", " ").replace(",", ";"))[:200],
                t.get("model", ""),
                str(t.get("created_at", "")),
            ]
            lines.append(",".join(row))
        body = "\n".join(lines) + "\n"
        etag = hashlib.md5(body.encode()).hexdigest()  # stable for same content
        return body, etag

    async def get_tasks_csv(request: web.Request) -> web.Response:
        body, etag = _compute_tasks_csv_and_etag(state["tasks"])
        inm = request.headers.get("If-None-Match")
        if inm and inm == etag:
            return web.Response(status=304)
        state["tasks_csv_etag"] = etag
        return web.Response(text=body, content_type="text/csv", headers={"ETag": etag})

    async def post_tasks(request: web.Request) -> web.Response:
        body = _json(request) or {}
        t = {
            "task_id": _uuid(),
            "agent_type": body.get("agent_type", "llm"),
            "message": body.get("message", ""),
            "model": body.get("model", "auto"),
            "created_at": time.time(),
            "stream_mode": bool(body.get("stream_mode")),
        }
        state["tasks"].append(t)

        resp: Dict[str, Any] = {"task_id": t["task_id"], "status": "accepted"}
        if os.getenv("ZEN_ROUTER_ENABLE", "0").lower() in ("1", "true", "yes") and t["model"].lower() == "auto":
            resp["router"] = {
                "chosen_model": "gemini-2.5-flash",
                "reason": "auto",
            }
        return web.json_response(resp, status=200)

    async def post_a2a_message(request: web.Request) -> web.Response:
        body = _json(request) or {}
        allowed_senders = {s.strip() for s in os.getenv("A2A_ALLOWED_SENDERS", "").split(",") if s.strip()}
        allowed_types = {s.strip() for s in os.getenv("A2A_ALLOWED_TYPES", "").split(",") if s.strip()}

        sender = body.get("sender_id")
        mtype = body.get("message_type")

        if allowed_senders and sender not in allowed_senders:
            return web.json_response({
                "message_type": "error",
                "payload": {"error": "forbidden", "reason": "sender_not_allowed"},
            })
        if allowed_types and mtype not in allowed_types:
            return web.json_response({
                "message_type": "error",
                "payload": {"error": "forbidden", "reason": "type_not_allowed"},
            })
        return web.json_response({"message_type": "accepted", "payload": {"ok": True}})

    # Route registration
    app.router.add_get("/models/catalog", get_models_catalog)
    app.router.add_post("/router/decide", post_router_decide)
    app.router.add_post("/channels", post_channels)
    app.router.add_post("/messages/channel", post_message_channel)
    app.router.add_get("/inbox/messages", get_inbox_messages)
    app.router.add_post("/messages/resume", post_messages_resume)
    app.router.add_post("/threads/resume", post_threads_resume)
    app.router.add_post("/tasks", post_tasks)
    app.router.add_get("/tasks.csv", get_tasks_csv)
    app.router.add_post("/a2a/message", post_a2a_message)

    return app
