"""
Synchronous Agent Tool

This tool executes single agent tasks with blocking wait for results,
integrating with AgentAPI HTTP endpoints for direct agent communication.
"""

import asyncio
import logging
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from mcp.types import TextContent

from tools.shared.agent_models import (
    AgentTask,
    AgentTaskRequest,
    AgentTaskResult,
    AgentType,
    Message,
    TaskStatus,
)
from tools.shared.base_tool import BaseTool
from utils.agent_defaults import get_default_working_directory

logger = logging.getLogger(__name__)


class AgentSyncTool(BaseTool):
    """Tool for synchronous agent task execution."""

    name = "agent_sync"
    description = """Execute a single agent task synchronously and wait for completion.

    This tool delegates a specific task to a CLI agent (claude, aider, goose, etc.) and waits
    for the agent to complete the task before returning results. Use this when you need the
    result immediately and can wait for completion.

    Implementation: Runs in-process via adapters (no external agentapi server).
    The tool will:
    1. Spawn the CLI agent in a controlled working directory
    2. Send your message to the agent (non-interactive)
    3. Wait for completion and capture stdout/stderr
    4. Return the final output and status

    Best for: Quick tasks, code analysis, single-file edits, debugging assistance
    """
    def get_annotations(self) -> Optional[dict[str, Any]]:
        """Agent orchestration tool; not read-only and potentially destructive.

        We tag it for UI grouping and warn about side effects.
        """
        return {
            "category": "agent-orchestration",
            "tags": ["agent", "sync", "cli"],
            "destructiveHint": True,
            "readOnlyHint": False,
        }

    def requires_model(self) -> bool:
        """This tool doesn't require AI model access."""
        return False

    def get_name(self) -> str:
        """Return the tool name."""
        return self.name

    def get_description(self) -> str:
        """Return the tool description."""
        return self.description

    def get_request_model(self):
        """Return the request model for this tool."""
        from typing import Optional

        from pydantic import BaseModel, Field

        class AgentSyncRequest(BaseModel):
            agent_type: str = Field(..., description="Type of agent to use (claude, aider, goose)")
            task_description: str = Field(..., description="Brief description of the task")
            message: str = Field(..., description="Detailed message/instructions for the agent")
            agent_args: Optional[list[str]] = Field([], description="Additional command-line arguments")
            working_directory: Optional[str] = Field(None, description="Working directory for the agent")
            timeout_seconds: Optional[int] = Field(300, description="Timeout in seconds")
            env_vars: Optional[dict[str, str]] = Field({}, description="Environment variables")
            files: Optional[list[str]] = Field([], description="Files to include in context")

        return AgentSyncRequest

    def get_system_prompt(self) -> str:
        """Return the system prompt for this tool."""
        return "You are a synchronous agent execution tool that coordinates with CLI agents."

    async def prepare_prompt(self, request) -> str:
        """Prepare the prompt for the agent sync tool."""
        return "Agent sync tool - no prompt preparation needed as this is a coordination tool."

    def get_input_schema(self) -> dict[str, Any]:
        """Get the input schema for the synchronous agent tool."""
        return {
            "type": "object",
            "properties": {
                "agent_type": {
                    "type": "string",
                    "enum": [agent.value for agent in AgentType],
                    "description": "Type of agent to use for the task",
                },
                "task_description": {
                    "type": "string",
                    "description": "Brief description of what you want the agent to do",
                },
                "message": {"type": "string", "description": "Detailed message/instructions to send to the agent"},
                "agent_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Additional command-line arguments for the agent",
                },
                "working_directory": {
                    "type": "string",
                    "description": "Working directory for the agent (defaults to current directory)",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "default": 300,
                    "minimum": 30,
                    "maximum": 1800,
                    "description": "Maximum time to wait for task completion (30-1800 seconds)",
                },
                "env_vars": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "default": {},
                    "description": "Environment variables to set for the agent",
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "List of files to make available to the agent",
                },
            },
            "required": ["agent_type", "task_description", "message"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Execute the synchronous agent tool using the task manager path.

        This integrates with utils.agent_manager so tests can patch the manager
        and the _wait_for_completion method.
        """
        try:
            # Create task request
            request = AgentTaskRequest(
                agent_type=AgentType(arguments["agent_type"]),
                task_description=arguments["task_description"],
                message=arguments["message"],
                agent_args=arguments.get("agent_args", []),
                working_directory=get_default_working_directory(
                    AgentType(arguments["agent_type"]),
                    arguments.get("working_directory"),
                    session_id=arguments.get("continuation_id"),
                ),
                timeout_seconds=arguments.get("timeout_seconds", 300),
                env_vars=arguments.get("env_vars", {}),
                files=arguments.get("files", []),
            )

            # Use task manager pipeline (mockable in tests)
            from utils.agent_manager import get_task_manager

            mgr = get_task_manager()
            task = await mgr.create_task(request)

            # If creation failed early
            if task.status == TaskStatus.FAILED and task.result:
                response = self._format_task_result(task.result)
                return [TextContent(type="text", text=response)]

            # Start the task and wait for completion
            await mgr.start_task(task.task_id)
            result = await self._wait_for_completion(task, request.timeout_seconds or 300)

            # Best-effort cleanup
            try:
                await mgr.cleanup_completed_tasks()
            except Exception:
                pass

            response = self._format_task_result(result)
            return [TextContent(type="text", text=response)]

        except Exception as e:
            logger.error(f"Error in synchronous agent execution: {e}")
            return [TextContent(type="text", text=f"Error executing agent task: {str(e)}")]

    async def _execute_sync_task(self, request: AgentTaskRequest) -> AgentTaskResult:
        """Execute a task synchronously and return the result using in-process AgentAPI."""
        from utils.internal_agentapi import InternalAgentAPI

        api = InternalAgentAPI()
        result = await api.agent_sync(request)
        return result

    async def _wait_for_completion(self, task: AgentTask, timeout_seconds: int) -> AgentTaskResult:
        """Wait for task completion and collect results."""
        start_time = time.time()
        messages = []

        try:
            while time.time() - start_time < timeout_seconds:
                # Check agent status
                status = await self._get_agent_status(task.agent_port)

                if status == "stable":
                    # Agent is idle, collect final messages and complete
                    messages = await self._get_agent_messages(task.agent_port)

                    return AgentTaskResult(
                        task_id=task.task_id,
                        agent_type=task.request.agent_type,
                        status=TaskStatus.COMPLETED,
                        messages=messages,
                        output=self._extract_output_from_messages(messages),
                        started_at=task.created_at,
                        completed_at=datetime.now(timezone.utc),
                        duration_seconds=time.time() - start_time,
                        agent_port=task.agent_port,
                    )

                elif status == "running":
                    # Agent is still working, continue waiting
                    await asyncio.sleep(2)
                    continue

                else:
                    # Unknown status or error
                    messages = await self._get_agent_messages(task.agent_port)
                    return self._create_error_result(task, f"Agent returned unexpected status: {status}", messages)

            # Timeout reached
            messages = await self._get_agent_messages(task.agent_port)
            return AgentTaskResult(
                task_id=task.task_id,
                agent_type=task.request.agent_type,
                status=TaskStatus.TIMEOUT,
                messages=messages,
                output=self._extract_output_from_messages(messages),
                error=f"Task timed out after {timeout_seconds} seconds",
                started_at=task.created_at,
                completed_at=datetime.now(timezone.utc),
                duration_seconds=timeout_seconds,
                agent_port=task.agent_port,
            )

        except Exception as e:
            logger.error(f"Error waiting for task completion: {e}")
            messages = await self._get_agent_messages(task.agent_port)
            return self._create_error_result(task, str(e), messages)

        finally:
            # Always try to stop the agent server
            await self._stop_agent_server(task)

    async def _get_agent_status(self, port: int) -> Optional[str]:
        """Get the current status of the agent."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:{port}/status", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("status")
        except Exception as e:
            logger.debug(f"Error getting agent status: {e}")
        return None

    async def _get_agent_messages(self, port: int) -> list[dict[str, Any]]:
        """Get all messages from the agent conversation."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:{port}/messages", timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("messages", [])
        except Exception as e:
            logger.debug(f"Error getting agent messages: {e}")
        return []

    async def _stop_agent_server(self, task: AgentTask) -> None:
        """Stop the agent server process."""
        if task.process_id:
            try:
                # Try to terminate gracefully first
                subprocess.run(["kill", str(task.process_id)], timeout=5)
                await asyncio.sleep(1)

                # Force kill if still running
                subprocess.run(["kill", "-9", str(task.process_id)], timeout=5)
            except Exception as e:
                logger.debug(f"Error stopping agent server: {e}")

    def _extract_output_from_messages(self, messages: list[Any]) -> str:
        """Extract the final output from agent messages.

        Supports both dict and Message model instances.
        """
        if not messages:
            return ""

        def get_role(m: Any) -> str:
            return (getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None) or "").lower()

        def get_text(m: Any) -> str:
            # Prefer 'message' if present (legacy), else string 'content'
            if isinstance(m, dict):
                if isinstance(m.get("message"), str):
                    return m.get("message") or ""
                c = m.get("content")
            else:
                # Message model
                c = getattr(m, "message", None) or getattr(m, "content", None)
            if isinstance(c, str):
                return c
            # If provider-style content list
            if isinstance(c, list) and c:
                for part in reversed(c):
                    txt = part.get("text") if isinstance(part, dict) else None
                    if isinstance(txt, str) and txt:
                        return txt
            return ""

        # Choose last agent/assistant message with text
        for m in reversed(messages):
            if get_role(m) in {"agent", "assistant"}:
                txt = get_text(m)
                if txt:
                    return txt
        # Fallback to any last message text
        for m in reversed(messages):
            txt = get_text(m)
            if txt:
                return txt
        return ""

    def _create_error_result(
        self, task: AgentTask, error: str, messages: Optional[list[Message]] = None
    ) -> AgentTaskResult:
        """Create an error result for a failed task."""
        return AgentTaskResult(
            task_id=task.task_id,
            agent_type=task.request.agent_type,
            status=TaskStatus.FAILED,
            messages=messages or [],
            output="",
            error=error,
            started_at=task.created_at,
            completed_at=datetime.now(timezone.utc),
            duration_seconds=0,
            agent_port=task.agent_port,
        )

    def _format_task_result(self, result: AgentTaskResult, color: bool = False) -> str:
        """Format the task result for display."""
        status_icons = {TaskStatus.COMPLETED: "✅", TaskStatus.FAILED: "❌", TaskStatus.TIMEOUT: "⏰"}

        icon = status_icons.get(result.status, "❓")

        def colorize(s: str, code: str) -> str:
            return f"\x1b[{code}m{s}\x1b[0m" if color else s

        status_name = colorize(result.status.name, "32" if result.status == TaskStatus.COMPLETED else ("31" if result.status == TaskStatus.FAILED else "33"))
        agent_name = colorize(result.agent_type.value, "36")
        duration_str = f"{result.duration_seconds:.1f}s" if result.duration_seconds else "Unknown"
        duration_str = colorize(duration_str, "35")

        response_parts = [
            f"# {icon} Agent Task Result",
            f"\n**Agent**: {agent_name}",
            f"**Status**: {status_name}",
            f"**Duration**: {duration_str}",
        ]

        if result.error:
            response_parts.append(f"**Error**: {colorize(result.error, '31')}")

        if result.output:
            response_parts.append(f"\n## {colorize('Final Report', '1;34')}\n\n{colorize(result.output, '0')}" )

        # Extract synthetic metrics and action trail from messages if present
        metrics_line = None
        metrics_obj = None
        actions = []
        if result.messages:
            for m in result.messages:
                content = getattr(m, "message", None) or getattr(m, "content", None) if not isinstance(m, dict) else (m.get("message") or m.get("content"))
                if not content:
                    continue
                if isinstance(content, str) and content.startswith("metrics: "):
                    metrics_line = content.replace("metrics: ", "").strip()
                    # Try to decode metrics dict for richer display
                    try:
                        import ast
                        metrics_obj = ast.literal_eval(metrics_line)
                    except Exception:
                        metrics_obj = None
                elif isinstance(content, str) and content.startswith("action: "):
                    actions.append(content.replace("action: ", "").strip())

        if metrics_line:
            response_parts.append(f"\n## {colorize('Metrics', '1;34')}\n\n{colorize(metrics_line, '90')}")
        if metrics_obj and isinstance(metrics_obj, dict):
            files = metrics_obj.get("files_touched") or []
            if isinstance(files, list) and files:
                response_parts.append("\n## Files Touched")
                max_show = 15
                for p in files[:max_show]:
                    response_parts.append(f"\n- {p}")
                if len(files) > max_show:
                    response_parts.append(f"\n- ... and {len(files) - max_show} more")
        if actions:
            response_parts.append(f"\n## {colorize('Action Trail', '1;34')}")
            for i, a in enumerate(actions[-20:], 1):
                if len(a) > 300:
                    a = a[:300] + "..."
                response_parts.append(f"\n{colorize(str(i)+'.', '33')} {colorize(a, '0')}")

        # Suggested next steps
        response_parts.append(f"\n## {colorize('Next Steps', '1;34')}")
        if result.status == TaskStatus.COMPLETED:
            response_parts.extend([
                colorize("- Review the Final Report and verify outputs.", '0'),
                colorize("- If code was generated, save/apply changes and run tests.", '0'),
                colorize("- If additional detail is needed, re-run with a more specific prompt or files context.", '0'),
            ])
        elif result.status == TaskStatus.FAILED:
            response_parts.extend([
                colorize("- Inspect the Metrics and Action Trail for failure context.", '0'),
                colorize("- Re-run with simplified instructions or longer timeout.", '0'),
            ])

        response_parts.append(f"\n---\n**Task ID**: {result.task_id}")

        return "\n".join(response_parts)
