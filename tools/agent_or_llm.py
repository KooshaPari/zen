"""
Unified Agent/LLM Tool

Superset interface that can route to either:
- LLM providers (existing simple tools semantics via chat tool)
- CLI agents (agent_sync/agent_async/agent_batch semantics)

Selection:
- mode: "llm" | "agent" (explicit), OR
- provider/model present => llm; agent_type present => agent

Maintains existing functionality while providing one entry point.
"""
from __future__ import annotations

from typing import Any

from mcp.types import TextContent

from tools.shared.agent_models import AgentTaskRequest, AgentType
from tools.shared.base_tool import BaseTool


class AgentOrLLMTool(BaseTool):
    name = "agent_or_llm"
    description = (
        "Unified tool to call either LLM providers or CLI agents. Use mode=llm/agent, or specify model/provider for LLM and agent_type for agent."
    )

    def get_annotations(self) -> dict[str, Any] | None:
        return {
            "category": "orchestration",
            "tags": ["llm", "agent", "unified"],
            "destructiveHint": True,
            "readOnlyHint": False,
        }

    def requires_model(self) -> bool:
        # Only needed when routing to LLM
        return False

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["llm", "agent"], "description": "Route explicitly to LLM or Agent"},
                # LLM params
                "prompt": {"type": "string"},
                "model": {"type": "string"},
                "provider": {"type": "string"},
                # Agent params
                "agent_type": {"type": "string", "enum": [a.value for a in AgentType]},
                "task_description": {"type": "string"},
                "message": {"type": "string"},
                "agent_args": {"type": "array", "items": {"type": "string"}},
                "working_directory": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 30, "maximum": 7200, "default": 300},
                "files": {"type": "array", "items": {"type": "string"}},
                "async": {"type": "boolean", "description": "If true and mode=agent, run in background and return task id"},
            },
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        # Decide route
        mode = arguments.get("mode")
        if not mode:
            if arguments.get("agent_type"):
                mode = "agent"
            elif arguments.get("model") or arguments.get("provider"):
                mode = "llm"
        if mode == "llm":
            # delegate to chat tool
            from tools.chat import ChatTool

            prompt = arguments.get("prompt") or arguments.get("message") or ""
            model = arguments.get("model")
            provider = arguments.get("provider")
            chat = ChatTool()
            return await chat.execute({"prompt": prompt, **({"model": model} if model else {}), **({"provider": provider} if provider else {})})
        elif mode == "agent":
            from utils.agent_defaults import get_default_working_directory
            from utils.internal_agentapi import InternalAgentAPI

            request = AgentTaskRequest(
                agent_type=AgentType(arguments["agent_type"]),
                task_description=arguments.get("task_description") or (arguments.get("prompt") or "Agent task"),
                message=arguments.get("message") or arguments.get("prompt") or "",
                agent_args=arguments.get("agent_args", []),
                working_directory=get_default_working_directory(AgentType(arguments["agent_type"]), arguments.get("working_directory"), session_id=arguments.get("continuation_id")),
                timeout_seconds=arguments.get("timeout_seconds", 300),
                files=arguments.get("files", []),
            )
            api = InternalAgentAPI()
            if arguments.get("async"):
                task = await api.agent_async(request)
                # Reuse agent_async success format: return task id & one-liner
                return [TextContent(type="text", text=f"Launched background agent task: {task.task_id} ({request.agent_type.value})\nUse agent_inbox to monitor.")]
            else:
                result = await api.agent_sync(request)
                # Reuse agent_sync formatter with color default from env
                import os

                from tools.agent_sync import AgentSyncTool
                use_color = bool(arguments.get("color", False) or os.getenv("SYNC_COLOR") == "1")
                return [TextContent(type="text", text=AgentSyncTool()._format_task_result(result, color=use_color))]
        else:
            return [TextContent(type="text", text="Specify mode: 'llm' or 'agent', or provide model/provider or agent_type.")]

