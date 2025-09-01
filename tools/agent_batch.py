"""
Batch Agent Tool

This tool launches multiple agent tasks in parallel with coordination
and result aggregation, enabling complex multi-agent workflows.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from mcp.types import TextContent

from tools.agent_batch_utils import format_batch_launch_response
from tools.shared.agent_models import (
    AgentTaskRequest,
    AgentType,
    BatchTaskRequest,
    TaskStatus,
)
from tools.shared.base_tool import BaseTool
from utils.agent_defaults import get_default_working_directory
from utils.batch_registry import append_task, register_batch
from utils import agent_manager as agent_manager

logger = logging.getLogger(__name__)


class AgentBatchTool(BaseTool):
    """Tool for batch agent task execution with coordination."""

    name = "agent_batch"
    description = """Launch multiple agent tasks in parallel with coordination and result aggregation.

    This tool enables complex multi-agent workflows by launching multiple agents simultaneously
    and coordinating their execution. Perfect for decomposing large projects into parallel tasks.

    Features:
    - Parallel task execution with configurable concurrency limits
    - Sequential execution option for dependent tasks
    - Fail-fast or continue-on-error strategies
    - Automatic result aggregation and coordination
    - Progress monitoring across all tasks

    Example Use Case: Building a CRUD todo app
    - Frontend tasks: Home Page, Login, Tasks Page, Actions Modal, Tests, Styling
    - Backend tasks: Auth, DB, API, Logic, Tests, Performance

    Each task can be assigned to the most appropriate agent type for optimal results.
    """
    def get_annotations(self) -> Optional[dict[str, Any]]:
        """Agent orchestration tool; batch launcher with side effects.
        """
        return {
            "category": "agent-orchestration",
            "tags": ["agent", "batch", "cli", "parallel"],
            "destructiveHint": True,
            "readOnlyHint": False,
        }

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

        class TaskRequest(BaseModel):
            agent_type: str = Field(..., description="Type of agent to use")
            task_description: str = Field(..., description="Brief description of the task")
            message: str = Field(..., description="Detailed message/instructions for the agent")
            agent_args: Optional[list[str]] = Field([], description="Additional command-line arguments")
            working_directory: Optional[str] = Field(None, description="Working directory for the agent")
            timeout_seconds: Optional[int] = Field(1800, description="Timeout in seconds")
            env_vars: Optional[dict[str, str]] = Field({}, description="Environment variables")
            files: Optional[list[str]] = Field([], description="Files to include in context")

        class AgentBatchRequest(BaseModel):
            batch_description: Optional[str] = Field("", description="Description of the batch operation")
            tasks: list[TaskRequest] = Field(..., description="List of tasks to execute")
            coordination_strategy: Optional[str] = Field(
                "parallel", description="Coordination strategy (parallel, sequential)"
            )
            max_concurrent: Optional[int] = Field(5, description="Maximum concurrent tasks")
            timeout_seconds: Optional[int] = Field(3600, description="Overall timeout in seconds")
            fail_fast: Optional[bool] = Field(False, description="Stop on first failure")

        return AgentBatchRequest

    def get_system_prompt(self) -> str:
        """Return the system prompt for this tool."""
        return "You are a batch agent execution tool that coordinates multiple CLI agents."

    async def prepare_prompt(self, request) -> str:
        """Prepare the prompt for the agent batch tool."""
        return "Agent batch tool - no prompt preparation needed as this is a coordination tool."

    def requires_model(self) -> bool:
        """This tool doesn't require AI model access."""
        return False

    def get_input_schema(self) -> dict[str, Any]:
        """Get the input schema for the batch agent tool."""
        return {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "agent_type": {
                                "type": "string",
                                "enum": [agent.value for agent in AgentType],
                                "description": "Type of agent to use for this task",
                            },
                            "task_description": {
                                "type": "string",
                                "description": "Brief description of what this agent should do",
                            },
                            "message": {
                                "type": "string",
                                "description": "Detailed message/instructions for this agent",
                            },
                            "agent_args": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": [],
                                "description": "Additional command-line arguments for this agent",
                            },
                            "working_directory": {"type": "string", "description": "Working directory for this agent"},
                            "timeout_seconds": {
                                "type": "integer",
                                "default": 1800,
                                "description": "Timeout for this specific task",
                            },
                            "env_vars": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                                "default": {},
                                "description": "Environment variables for this agent",
                            },
                            "files": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": [],
                                "description": "Files to make available to this agent",
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["low", "normal", "high"],
                                "default": "normal",
                                "description": "Priority for this task",
                            },
                        },
                        "required": ["agent_type", "task_description", "message"],
                    },
                    "minItems": 2,
                    "maxItems": 10,
                    "description": "List of tasks to execute (2-10 tasks)",
                },
                "coordination_strategy": {
                    "type": "string",
                    "enum": ["parallel", "sequential"],
                    "default": "parallel",
                    "description": "How to coordinate task execution",
                },
                "max_concurrent": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum number of tasks to run concurrently",
                },
                "fail_fast": {"type": "boolean", "default": False, "description": "Stop all tasks if any task fails"},
                "timeout_seconds": {
                    "type": "integer",
                    "default": 3600,
                    "minimum": 300,
                    "maximum": 7200,
                    "description": "Overall batch timeout (300-7200 seconds)",
                },
                "batch_description": {"type": "string", "description": "Description of the overall batch operation"},
            },
            "required": ["tasks"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Execute the batch agent tool."""
        try:
            # Create batch request
            task_requests = []
            for task_data in arguments["tasks"]:
                request = AgentTaskRequest(
                    agent_type=AgentType(task_data["agent_type"]),
                    task_description=task_data["task_description"],
                    message=task_data["message"],
                    agent_args=task_data.get("agent_args", []),
                    working_directory=get_default_working_directory(
                        AgentType(task_data["agent_type"]),
                        task_data.get("working_directory"),
                        session_id=arguments.get("continuation_id"),
                    ),
                    timeout_seconds=task_data.get("timeout_seconds", 1800),
                    env_vars=task_data.get("env_vars", {}),
                    files=task_data.get("files", []),
                )
                task_requests.append(request)

            batch_request = BatchTaskRequest(
                tasks=task_requests,
                coordination_strategy=arguments.get("coordination_strategy", "parallel"),
                max_concurrent=arguments.get("max_concurrent", 5),
                timeout_seconds=arguments.get("timeout_seconds", 3600),
                fail_fast=arguments.get("fail_fast", False),
            )

            # Determine color flag once
            try:
                color_flag = bool(arguments.get("color", False) or os.getenv("BATCH_COLOR") == "1")
            except Exception:
                color_flag = bool(arguments.get("color", False))

            # Execute batch; for parallel strategy, launch tasks immediately to surface IDs
            description = arguments.get("batch_description", "")
            launched: list[tuple[str, str, str]] | None = None
            if batch_request.coordination_strategy == "parallel":
                mgr = agent_manager.get_task_manager()
                launched = []
                batch_id = str(uuid.uuid4())
                for req in task_requests:
                    task = await mgr.create_task(req)
                    launched.append((task.task_id, req.agent_type.value, req.task_description))
                    append_task(batch_id, task.task_id)
                register_batch(batch_id, description, [tid for (tid, _agent, _desc) in launched])
                # Hand off batch execution in background exactly once
                asyncio.create_task(self._run_batch_execution(batch_id, batch_request, description))
            else:
                # For sequential, run coordinator in background
                batch_id = await self._execute_batch(batch_request, description)
                # Pre-register empty batch (tasks appended during run)
                register_batch(batch_id, description, [])
                # If sequential or after launching, hand off to background runner
                # We already created a batch_id above for sequential
                asyncio.create_task(self._run_batch_execution(batch_id, batch_request, description))

                # Format and return response with dashboard placeholders if any
                response = format_batch_launch_response(
                    batch_id, batch_request, description, launched, color_flag
                )
                return [TextContent(type="text", text=response)]


            # Format and return response with tracker when available
            response = format_batch_launch_response(
                batch_id, batch_request, description, launched, color_flag
            )
            return [TextContent(type="text", text=response)]

        except Exception as e:
            logger.error(f"Error executing batch agent tasks: {e}")
            return [TextContent(type="text", text=f"Error executing batch tasks: {str(e)}")]

    async def _execute_batch(self, batch_request: BatchTaskRequest, description: str) -> str:
        """Execute a batch of tasks and return batch ID."""
        batch_id = str(uuid.uuid4())

        # Launch batch execution in background
        asyncio.create_task(self._run_batch_execution(batch_id, batch_request, description))

        logger.info(f"Launched batch {batch_id} with {len(batch_request.tasks)} tasks")
        return batch_id

    async def _run_batch_execution(self, batch_id: str, batch_request: BatchTaskRequest, description: str) -> None:
        """Run batch execution with coordination."""
        mgr = agent_manager.get_task_manager()
        datetime.now(timezone.utc)

        try:
            if batch_request.coordination_strategy == "sequential":
                await self._run_sequential_batch(batch_id, batch_request, mgr)
            else:  # parallel
                await self._run_parallel_batch(batch_id, batch_request, mgr)

        except Exception as e:
            logger.error(f"Error in batch execution {batch_id}: {e}")
        finally:
            # Cleanup
            await mgr.cleanup_completed_tasks()
            logger.info(f"Batch {batch_id} execution completed")

    async def _run_parallel_batch(self, batch_id: str, batch_request: BatchTaskRequest, mgr) -> None:
        """Run tasks in parallel with concurrency control."""
        semaphore = asyncio.Semaphore(batch_request.max_concurrent)
        tasks = []

        async def run_single_task(request: AgentTaskRequest) -> str:
            async with semaphore:
                # Create and start task using manager
                task = await mgr.create_task(request)
                try:
                    await mgr.start_task(task.task_id)
                except Exception:
                    pass
                if task.status == TaskStatus.FAILED and batch_request.fail_fast:
                    raise Exception(f"Failed to create task: {task.result.error if task.result else 'Unknown error'}")
                append_task(batch_id, task.task_id)
                return task.task_id

        # Launch all tasks
        for request in batch_request.tasks:
            task_coro = run_single_task(request)
            tasks.append(task_coro)

        # Wait for all tasks to start
        try:
            await asyncio.gather(*tasks, return_exceptions=not batch_request.fail_fast)
        except Exception as e:
            if batch_request.fail_fast:
                logger.error(f"Batch {batch_id} failed fast: {e}")
                # Cancel remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()

    async def _run_sequential_batch(self, batch_id: str, batch_request: BatchTaskRequest, mgr) -> None:
        """Run tasks sequentially."""
        for i, request in enumerate(batch_request.tasks):
            logger.info(f"Starting sequential task {i+1}/{len(batch_request.tasks)} in batch {batch_id}")

            # Launch and wait synchronously for this one before next
            task = await mgr.create_task(request)
            try:
                await mgr.start_task(task.task_id)
            except Exception:
                pass
            if task.status == TaskStatus.FAILED and batch_request.fail_fast:
                raise Exception(
                    f"Failed to create task {i+1}: {task.result.error if task.result else 'Unknown error'}"
                )
            append_task(batch_id, task.task_id)
            await self._wait_for_task_completion(task.task_id, request.timeout_seconds or 1800)

    async def _wait_for_task_completion(self, task_id: str, timeout_seconds: int) -> None:
        """Wait for a single task to complete."""
        from utils.agent_manager import get_task_manager
        task_manager = get_task_manager()

        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout_seconds:
            task = await task_manager.get_task(task_id)
            if not task:
                break

            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED]:
                break

            await asyncio.sleep(5)  # Check every 5 seconds

    def _format_batch_launch_response(self, batch_id: str, batch_request: BatchTaskRequest, description: str) -> str:
        """Format the batch launch response."""
        response_parts = [
            "# 🚀 Batch Agent Tasks Launched",
            f"\nBatch ID: `{batch_id}`",
            f"Strategy: {batch_request.coordination_strategy}",
            f"Tasks: {len(batch_request.tasks)}",
            f"Max Concurrent: {batch_request.max_concurrent}",
            f"Fail Fast: {'Yes' if batch_request.fail_fast else 'No'}",
            f"Timeout: {batch_request.timeout_seconds}s",
        ]

        if description:
            response_parts.append(f"**Description**: {description}")

        response_parts.append("\n## Task Breakdown")

        for i, request in enumerate(batch_request.tasks, 1):
            response_parts.append(f"\n### Task {i}: {request.agent_type.value}")
            response_parts.append(f"**Description**: {request.task_description}")
            if request.working_directory:
                response_parts.append(f"**Directory**: {request.working_directory}")
            if request.agent_args:
                response_parts.append(f"**Args**: {' '.join(request.agent_args)}")

        response_parts.extend(
            [
                "\n## Monitoring",
                "",
                "Your batch is now running in the background. You can:",
                "",
                "1. **Check Progress**: Use `agent_inbox` with `action: list` to see all running tasks",
                "2. **Individual Status**: Use `agent_inbox` with specific task IDs to check individual progress",
                "3. **Continue Working**: All tasks run independently in the background",
                "",
                "## Coordination Strategy",
                "",
            ]
        )

        if batch_request.coordination_strategy == "parallel":
            response_parts.extend(
                [
                    f"🔄 **Parallel Execution**: Up to {batch_request.max_concurrent} tasks running simultaneously",
                    "- Tasks start immediately and run concurrently",
                    "- Faster overall completion time",
                    "- Resource usage distributed across tasks",
                ]
            )
        else:
            response_parts.extend(
                [
                    "➡️ **Sequential Execution**: Tasks run one after another",
                    "- Each task waits for the previous to complete",
                    "- Predictable resource usage",
                    "- Useful for dependent tasks",
                ]
            )

        if batch_request.fail_fast:
            response_parts.append("\n⚡ **Fail Fast**: If any task fails, remaining tasks will be cancelled")
        else:
            response_parts.append("\n🔄 **Continue on Error**: Tasks continue even if others fail")

        response_parts.extend(
            [
                "",
                "---",
                "",
                "💡 **Tip**: Use `agent_inbox` regularly to monitor progress and collect results as tasks complete.",
            ]
        )

        return "\n".join(response_parts)
