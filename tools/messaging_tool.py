from __future__ import annotations

import os
from typing import Any

from mcp.types import TextContent

from tools.shared.base_tool import BaseTool


class MessagingTool(BaseTool):
    """Agent-facing messaging utilities: post, resume, histories, inbox summary.

    Methods (via action argument):
      - channel_post: channel_id, body, from, mentions?(list), blocking?(bool)
      - dm_post: a, b, body, from, blocking?(bool)
      - resume: message_id or resume_token, from?, reply_body?
      - channel_history: channel_id, limit?, offset?
      - dm_history: a, b, limit?, offset?
      - inbox: agent_id, limit?
    """

    def get_name(self) -> str:
        return "messaging"

    def get_description(self) -> str:
        return "Interact with channels/DMs: post messages, resume blocking items, fetch histories and inbox."

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": [
                    "channel_post", "dm_post", "resume", "channel_history", "dm_history", "inbox"
                ]},
                # common
                "from": {"type": "string"},
                "body": {"type": "string"},
                "blocking": {"type": "boolean"},
                "mentions": {"type": "array", "items": {"type": "string"}},
                # channel
                "channel_id": {"type": "string"},
                # dm
                "a": {"type": "string"},
                "b": {"type": "string"},
                # resume
                "message_id": {"type": "string"},
                "resume_token": {"type": "string"},
                "reply_body": {"type": "string"},
                # listing
                "limit": {"type": "integer"},
                "offset": {"type": "integer"},
                # inbox
                "agent_id": {"type": "string"},
            },
            "required": ["action"],
        }

    def requires_model(self) -> bool:
        return False

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        action = arguments.get("action")
        from mcp.types import TextContent

        # Import stores lazily
        from utils.messaging_store import (
            get_channel_history,
            get_dm_history,
            get_message,
            get_unread_summary,
            mark_block_resolved,
            post_channel_message,
            post_dm_message,
            resolve_by_token,
        )

        def _enforce_acl_channel(cid: str, actor: str):
            if os.getenv("ENFORCE_TOOL_ACL", os.getenv("ENFORCE_ACL", "0")).lower() in ("1","true","yes"):
                from utils.messaging_store import get_channel_members
                members = set(get_channel_members(cid) or [])
                if members and actor not in members:
                    raise PermissionError("not_channel_member")

        def _enforce_acl_dm(a: str, b: str, actor: str):
            if os.getenv("ENFORCE_TOOL_ACL", os.getenv("ENFORCE_ACL", "0")).lower() in ("1","true","yes"):
                if actor not in (a, b):
                    raise PermissionError("dm_members_only")

        if action == "channel_post":
            msg = post_channel_message(
                arguments.get("channel_id"),
                arguments.get("from", "agent"),
                arguments.get("body", ""),
                mentions=arguments.get("mentions") or [],
                blocking=bool(arguments.get("blocking", False)),
            )
            return [TextContent(type="text", text=f"posted {msg.get('id')}")]

        if action == "dm_post":
            msg = post_dm_message(
                arguments.get("a"), arguments.get("b"), arguments.get("from", "agent"), arguments.get("body", ""),
                blocking=bool(arguments.get("blocking", False))
            )
            return [TextContent(type="text", text=f"posted {msg.get('id')}")]

        if action == "resume":
            if arguments.get("message_id"):
                ok = mark_block_resolved(arguments["message_id"], agent_id=arguments.get("from"))
                msg = get_message(arguments["message_id"]) or {"id": arguments["message_id"]}
                return [TextContent(type="text", text=f"resolved={ok} message={msg.get('id')}")]
            if arguments.get("resume_token"):
                res = resolve_by_token(arguments["resume_token"], reply_body=arguments.get("reply_body"), from_id=arguments.get("from"))
                return [TextContent(type="text", text=f"resolved={res.get('ok')} message={res.get('message',{}).get('id')} reply_id={res.get('reply_id')}")]
            return [TextContent(type="text", text="missing message_id or resume_token")]

        if action == "channel_history":
            items = get_channel_history(arguments.get("channel_id"), limit=int(arguments.get("limit", 50)), offset=int(arguments.get("offset", 0)))
            return [TextContent(type="text", text=f"{len(items)} messages")]

        if action == "dm_history":
            items = get_dm_history(arguments.get("a"), arguments.get("b"), limit=int(arguments.get("limit", 50)), offset=int(arguments.get("offset", 0)))
            return [TextContent(type="text", text=f"{len(items)} messages")]

        if action == "inbox":
            s = get_unread_summary(arguments.get("agent_id", "agent"))
            return [TextContent(type="text", text=f"unread={s.get('total',0)}")]

        return [TextContent(type="text", text="unknown action")]
