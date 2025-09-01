from __future__ import annotations

import os
from typing import Any

from mcp.types import TextContent

from tools.shared.base_tool import BaseTool


class ProjectTool(BaseTool):
    """Minimal project operations for agents: create, graph, add_artifact, list_artifacts."""

    def get_name(self) -> str:
        return "project"

    def get_description(self) -> str:
        return "Create and inspect projects; manage artifacts."

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create", "graph", "add_artifact", "list_artifacts"]},
                "name": {"type": "string"},
                "owner": {"type": "string"},
                "project_id": {"type": "string"},
                "artifact": {"type": "object"},
                "limit": {"type": "integer"},
                "offset": {"type": "integer"},
            },
            "required": ["action"],
        }

    def requires_model(self) -> bool:
        return False

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        from utils.project_store import add_artifact, create_project, get_graph, list_artifacts
        action = arguments.get("action")
        if action == "create":
            owner = arguments.get("owner", "agent")
            if os.getenv("ENFORCE_TOOL_ACL", os.getenv("ENFORCE_ACL", "0")).lower() in ("1","true","yes"):
                actor = arguments.get("from") or owner
                if actor != owner:
                    raise PermissionError("owner_mismatch")
            p = create_project(arguments.get("name", "Untitled"), owner)
            return [TextContent(type="text", text=p["id"])]
        if action == "graph":
            g = get_graph(arguments.get("project_id"))
            return [TextContent(type="text", text=f"graph nodes={len(g.get('agents',[]))} artifacts={g.get('artifacts_count',0)}")]
        if action == "add_artifact":
            add_artifact(arguments.get("project_id"), arguments.get("artifact") or {"name": "artifact"})
            return [TextContent(type="text", text="ok")]
        if action == "list_artifacts":
            items = list_artifacts(arguments.get("project_id"), limit=int(arguments.get("limit", 50)), offset=int(arguments.get("offset", 0)))
            return [TextContent(type="text", text=f"{len(items)} artifacts")]
        return [TextContent(type="text", text="unknown action")]
