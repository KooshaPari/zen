"""
Asynchronous Agent Tool

This tool launches background agent tasks and returns task IDs for later monitoring,
allowing the primary agent to continue working while sub-agents execute tasks.
"""

import asyncio
import logging
from typing import Any, Optional

from mcp.types import TextContent

import utils.agent_manager as agent_manager
from tools.shared.agent_models import (
    AgentTaskRequest,
    AgentType,
    TaskStatus,
)
from tools.shared.base_tool import BaseTool
from utils.agent_defaults import get_default_working_directory

logger = logging.getLogger(__name__)


class AgentAsyncTool(BaseTool):
    """Tool for asynchronous agent task execution."""

    name = "agent_async"
    description = """Launch a background agent task and return immediately with a task ID.

    This tool starts an agent task in the background and returns a task ID that you can use
    to check status and retrieve results later using the agent_inbox tool. This allows you
    to continue working on other tasks while the agent executes in parallel.

    Implementation: Runs in-process via adapters (no external agentapi server).
    The tool will:
    1. Spawn the CLI agent in a controlled working directory
    2. Send your task message to the agent (non-interactive)
    3. Return a task ID immediately
    4. Continue running the agent in the background

    Best for: Long-running tasks, parallel execution, complex refactoring, comprehensive analysis

    Use agent_inbox to check status and retrieve results when ready.
    """

    def get_annotations(self) -> Optional[dict[str, Any]]:
        """Agent orchestration tool; background execution with side effects.
        """
        return {
            "category": "agent-orchestration",
            "tags": ["agent", "async", "cli", "background"],
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

        class AgentAsyncRequest(BaseModel):
            agent_type: str = Field(..., description="Type of agent to use (claude, aider, goose)")
            task_description: str = Field(..., description="Brief description of the task")
            message: str = Field(..., description="Detailed message/instructions for the agent")
            priority: Optional[str] = Field("normal", description="Task priority (low, normal, high)")
            agent_args: Optional[list[str]] = Field([], description="Additional command-line arguments")
            working_directory: Optional[str] = Field(None, description="Working directory for the agent")
            timeout_seconds: Optional[int] = Field(1800, description="Timeout in seconds")
            env_vars: Optional[dict[str, str]] = Field({}, description="Environment variables")
            files: Optional[list[str]] = Field([], description="Files to include in context")

        return AgentAsyncRequest

    def get_system_prompt(self) -> str:
        """Return the system prompt for this tool."""
        return "You are an asynchronous agent execution tool that coordinates with CLI agents."

    async def prepare_prompt(self, request) -> str:
        """Prepare the prompt for the agent async tool."""
        return "Agent async tool - no prompt preparation needed as this is a coordination tool."

    def get_input_schema(self) -> dict[str, Any]:
        """Get the input schema for the asynchronous agent tool."""
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
                    "default": 1800,
                    "minimum": 60,
                    "maximum": 3600,
                    "description": "Maximum time to allow task to run (60-3600 seconds)",
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
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high"],
                    "default": "normal",
                    "description": "Task priority for resource allocation",
                },
            },
            "required": ["agent_type", "task_description", "message"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Execute the asynchronous agent tool."""
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
                timeout_seconds=arguments.get("timeout_seconds", 1800),
                env_vars=arguments.get("env_vars", {}),
                files=arguments.get("files", []),
            )

            # Launch task asynchronously
            task_id = await self._launch_async_task(request, arguments.get("priority", "normal"))

            # Format and return response
            response = self._format_launch_response(task_id, request)
            return [TextContent(type="text", text=response)]

        except Exception as e:
            logger.error(f"Error launching asynchronous agent task: {e}")
            return [TextContent(type="text", text=f"Error launching agent task: {str(e)}. Ensure working_directory is writable and PATH/API keys are configured.")]

    async def _launch_async_task(self, request: AgentTaskRequest, priority: str) -> str:
        """Launch a task asynchronously via the task manager and return the task ID."""
        mgr = agent_manager.get_task_manager()
        task = await mgr.create_task(request)
        # Best-effort background start without blocking
        try:
            asyncio.create_task(mgr.start_task(task.task_id))
        except Exception:
            logger.debug("Background start_task scheduling failed", exc_info=True)
        logger.info(f"Launched async task {task.task_id} for {request.agent_type}")
        return task.task_id

    async def _run_background_task(self, task_id: str, priority: str) -> None:
        """Run a task in the background with monitoring (internal)."""
        # No-op: InternalAgentAPI already manages background execution.
        return None

    async def _monitor_background_task(self, task_id: str, priority: str) -> None:
        """Monitor a background task until completion."""
        task_manager = agent_manager.get_task_manager()

        # Adjust monitoring frequency based on priority
        check_intervals = {
            "low": 10,  # Check every 10 seconds
            "normal": 5,  # Check every 5 seconds
            "high": 2,  # Check every 2 seconds
        }
        interval = check_intervals.get(priority, 5)

        max_iterations = 3600 // interval  # Maximum monitoring time based on interval
        iterations = 0

        while iterations < max_iterations:
            task = await task_manager.get_task(task_id)
            if not task:
                logger.error(f"Lost track of task {task_id}")
                break

            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED]:
                logger.info(f"Background task {task_id} completed with status: {task.status}")
                break

            await asyncio.sleep(interval)
            iterations += 1

        if iterations >= max_iterations:
            logger.warning(f"Background task {task_id} monitoring timed out")

    def _format_launch_response(self, task_id: str, request: AgentTaskRequest) -> str:
        """Format the launch response."""
        response_parts = [
            "# 🚀 Agent Task Launched",
            f"\n**Task ID**: `{task_id}`",
            f"**Agent**: {request.agent_type.value}",
            f"**Description**: {request.task_description}",
            f"**Working Directory**: {request.working_directory}",
            f"**Timeout**: {request.timeout_seconds}s",
        ]

        if request.agent_args:
            response_parts.append(f"**Agent Args**: {' '.join(request.agent_args)}")

        if request.files:
            response_parts.append(f"**Files**: {len(request.files)} files provided")

        response_parts.extend(
            [
                "\n## Next Steps",
                "",
                "The agent is now running in the background. You can:",
                "",
                f"1. **Check Status**: Use `agent_inbox` with task_id `{task_id}` to check progress",
                "2. **Continue Working**: The agent runs independently, so you can work on other tasks",
                "3. **Retrieve Results**: Use `agent_inbox` to get results when the task completes",
                "",
                "## Task Message",
                f"\n```\n{request.message}\n```",
                "",
                "---",
                "",
                "💡 **Tip**: Background tasks are automatically cleaned up after completion. Use `agent_inbox` to monitor progress and retrieve results.",
            ]
        )

        return "\n".join(response_parts)
