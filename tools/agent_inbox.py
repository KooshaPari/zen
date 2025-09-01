"""
Agent Inbox Tool

This tool checks status and retrieves results from asynchronous agent tasks,
providing a centralized way to monitor and collect results from background agents.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from mcp.types import TextContent

import utils.agent_manager as agent_manager
from tools.shared.agent_models import TaskStatus
from tools.shared.base_tool import BaseTool

logger = logging.getLogger(__name__)


class AgentInboxTool(BaseTool):
    """Tool for checking status and retrieving results from asynchronous agent tasks."""

    name = "agent_inbox"
    description = """📥 AGENT INBOX - Manage asynchronous agent tasks and background operations.

Use this tool to:
• 📊 CHECK STATUS: Monitor progress of running background tasks
• 📋 LIST TASKS: See all your pending, running, and completed tasks
• 📄 GET RESULTS: Retrieve outputs from completed tasks
• ❌ CANCEL TASKS: Stop running tasks if needed
• 📈 TASK SUMMARY: Get overview of all task activity

WHEN TO USE: After launching background tasks with deploy tool (execution_mode='async') or any agent operations. Essential for managing long-running workflows."""

    def get_annotations(self) -> Optional[dict[str, Any]]:
        """Agent orchestration tool; monitoring interface (read-only).
        """
        return {
            "category": "agent-orchestration",
            "tags": ["agent", "inbox", "monitoring"],
            "destructiveHint": False,
            "readOnlyHint": True,
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

        class AgentInboxRequest(BaseModel):
            action: Optional[str] = Field("status", description="Action to perform (status, results, list, cancel, summary)")
            task_id: Optional[str] = Field(None, description="Specific task ID to query")
            include_conversation: Optional[bool] = Field(False, description="Include full conversation history")
            limit: Optional[int] = Field(10, description="Maximum number of tasks to list")
            status_filter: Optional[str] = Field(None, description="Filter by status (running, completed, failed)")

        return AgentInboxRequest

    def get_system_prompt(self) -> str:
        """Return the system prompt for this tool."""
        return "You are an agent inbox tool that monitors and retrieves results from background agent tasks."

    async def prepare_prompt(self, request) -> str:
        """Prepare the prompt for the agent inbox tool."""
        return "Agent inbox tool - no prompt preparation needed as this is a monitoring tool."

    def get_input_schema(self) -> dict[str, Any]:
        """Get the input schema for the agent inbox tool."""
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Specific task ID to check (optional - returns all tasks if not specified)",
                },
                "action": {
                    "type": "string",
                    "enum": ["status", "results", "list", "cancel", "summary"],
                    "default": "status",
                    "description": "Action to perform: status (check status), results (get full results), list (list all tasks), cancel (cancel task), summary (aggregate view)",
                },
                "include_messages": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to include conversation messages in results",
                },
                "max_message_length": {
                    "type": "integer",
                    "default": 1000,
                    "minimum": 100,
                    "maximum": 5000,
                    "description": "Maximum length for individual messages (truncated if longer)",
                },
                "filter_status": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["pending", "starting", "running", "completed", "failed", "timeout", "cancelled"],
                    },
                    "description": "Filter tasks by status (for list action)",
                },
                "batch_id": {
                    "type": "string",
                    "description": "Filter inbox (list/summary) to a specific batch",
                },
                "color": {
                    "type": "boolean",
                    "default": False,
                    "description": "Use ANSI colors in status lines (opt-in)",
                },
            },
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Execute the agent inbox tool."""
        try:
            action = arguments.get("action", "status")
            task_id = arguments.get("task_id")

            if action == "list":
                response = await self._list_tasks(arguments)
            elif action == "cancel" and task_id:
                response = await self._cancel_task(task_id)
            elif action == "summary":
                response = await self._summary(arguments)
            elif task_id:
                if action == "results":
                    response = await self._get_task_results(task_id, arguments)
                else:  # status
                    response = await self._get_task_status(task_id)
            else:
                response = await self._list_tasks(arguments)

            return [TextContent(type="text", text=response)]

        except Exception as e:
            logger.error(f"Error in agent inbox: {e}")
            return [TextContent(type="text", text=f"Error checking agent inbox: {str(e)}")]

    async def _get_task_status(self, task_id: str) -> str:
        """Get status of a specific task."""
        task_manager = agent_manager.get_task_manager()
        task = await task_manager.get_task(task_id)

        if not task:
            return f"❌ Task `{task_id}` not found. It may have been cleaned up or the ID is incorrect."

        # Get real-time status from agent if running
        current_status = task.status
        agent_status = None

        if task.status == TaskStatus.RUNNING and task.agent_port:
            agent_status = await self._get_live_agent_status(task.agent_port)

        status_icons = {
            TaskStatus.PENDING: "⏳",
            TaskStatus.STARTING: "🚀",
            TaskStatus.RUNNING: "⚡",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.TIMEOUT: "⏰",
            TaskStatus.CANCELLED: "🚫",
        }

        icon = status_icons.get(current_status, "❓")

        response_parts = [
            f"# {icon} Task Status: {task_id}",
            f"\n**Agent**: {task.request.agent_type.value}",
            f"**Status**: {current_status.name}",
            f"**Description**: {task.request.task_description}",
            f"**Created**: {task.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Updated**: {task.updated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ]

        if task.agent_port:
            response_parts.append(f"**Agent Port**: {task.agent_port}")

        if agent_status:
            response_parts.append(f"**Live Agent Status**: {agent_status}")

        # Calculate duration
        if current_status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT]:
            duration = (task.updated_at - task.created_at).total_seconds()
            response_parts.append(f"**Duration**: {duration:.1f}s")
        else:
            duration = (datetime.now(timezone.utc) - task.created_at).total_seconds()
            response_parts.append(f"**Running Time**: {duration:.1f}s")

        if task.result and task.result.error:
            response_parts.append(f"**Error**: {task.result.error}")
        # Print batch id to ease cross-linking
        for _bid_rec in []:  # placeholder to avoid unused import in some linters
            pass
        if task.result:
            for _m in task.result.messages or []:
                pass  # no-op, keep result.messages intact
        # Include batch membership if any
        from utils.batch_registry import find_batches_for_task
        batches = find_batches_for_task(task_id)
        for b in batches:
            response_parts.append(f"**Batch**: {b}")


        # Add next steps
        response_parts.append("\n## Next Steps")

        if current_status == TaskStatus.COMPLETED:
            response_parts.append(
                f"✅ Task completed! Use `agent_inbox` with `action: results` and `task_id: {task_id}` to get full results."
            )
        elif current_status == TaskStatus.RUNNING:
            response_parts.append(
                "⚡ Task is still running. Check back later or use `action: results` to see partial progress."
            )
        elif current_status == TaskStatus.FAILED:
            response_parts.append("❌ Task failed. Use `action: results` to see error details and any partial output.")
        elif current_status in [TaskStatus.PENDING, TaskStatus.STARTING]:
            response_parts.append("⏳ Task is starting up. Check back in a few moments.")

        return "\n".join(response_parts)

    async def _summary(self, arguments: dict[str, Any]) -> str:
        """Aggregate multi-task summary: status counts, metrics, files touched, and compact lines."""
        task_manager = agent_manager.get_task_manager()
        all_tasks = list(task_manager.active_tasks.values())

        # Optional status filtering
        filter_status = arguments.get("filter_status", [])
        if filter_status:
            try:
                filter_statuses = [TaskStatus(s) for s in filter_status]
                all_tasks = [t for t in all_tasks if t.status in filter_statuses]
            except Exception:
                pass

        if not all_tasks:
            return "📭 No tasks to summarize. Launch tasks with `agent_async`."

        # Aggregations
        from collections import Counter
        status_counter = Counter([t.status for t in all_tasks])
        steps_total = 0
        tools_total = 0
        files_changed_total = 0
        files_union: set[str] = set()

        def parse_metrics_from_messages(msgs: list[Any] | None) -> dict:
            if not msgs:
                return {}
            for m in msgs:
                content = (
                    getattr(m, "message", None) or getattr(m, "content", None)
                    if not isinstance(m, dict)
                    else (m.get("message") or m.get("content"))
                )
                if isinstance(content, str) and content.startswith("metrics: "):
                    raw = content.replace("metrics: ", "").strip()
                    try:
                        import ast
                        obj = ast.literal_eval(raw)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        return {}
            return {}

        rows: list[str] = []
        now = datetime.now(timezone.utc)
        for task in all_tasks:
            since_started = (now - task.created_at).total_seconds()
            since_finished = None
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED]:
                since_finished = (now - task.updated_at).total_seconds()
            metrics = parse_metrics_from_messages(task.result.messages if task.result else [])
            steps_total += int(metrics.get("steps_est", 0) or 0)
            tools_total += int(metrics.get("tool_calls_est", 0) or 0)
            files_changed_total += int(metrics.get("files_changed_total", 0) or 0)
            ft = metrics.get("files_touched") or []
            if isinstance(ft, list):
                files_union.update(ft)
            short_desc = task.request.task_description[:60] + (
                "..." if len(task.request.task_description) > 60 else ""
            )
            dur = f"{since_started:.0f}s"
            if since_finished is not None:
                dur += f" | +{since_finished:.0f}s"
            rows.append(
                f"- {task.status.value.upper()} | {task.task_id} | {task.request.agent_type.value} | {short_desc} | {dur}"
            )

        # Compose output
        response_parts = [
            f"# 🧭 Agent Inbox Summary ({len(all_tasks)} tasks)",
            "",
            "## Status Counts",
        ]
        for status, cnt in sorted(status_counter.items(), key=lambda x: x[0].value):
            response_parts.append(f"- {status.value.title()}: {cnt}")

        response_parts.extend([
            "",
            "## Aggregated Metrics",
            f"- steps_est (sum): {steps_total}",
            f"- tool_calls_est (sum): {tools_total}",
            f"- files_changed_total (sum): {files_changed_total}",
        ])

        if files_union:
            response_parts.append("\n## Files Touched (union)")
            max_show = 25
            for p in list(files_union)[:max_show]:
                response_parts.append(f"\n- {p}")
            if len(files_union) > max_show:
                response_parts.append(f"\n- ... and {len(files_union) - max_show} more")

        response_parts.append("\n## Tasks")
        response_parts.extend(["", *rows])

        response_parts.extend([
            "",
            "---",
            "💡 Tip: Use `agent_inbox` with `action: results` and a `task_id` for details.",
        ])

        return "\n".join(response_parts)

    async def _get_task_results(self, task_id: str, arguments: dict[str, Any]) -> str:
        """Get complete results for a task."""
        task_manager = agent_manager.get_task_manager()
        task = await task_manager.get_task(task_id)

        if not task:
            return f"❌ Task `{task_id}` not found."

        include_messages = arguments.get("include_messages", True)
        max_msg_length = arguments.get("max_message_length", 1000)

        # Get live messages if task is still running
        messages = []
        if task.agent_port and task.status == TaskStatus.RUNNING:
            messages = await self._get_live_agent_messages(task.agent_port)
        elif task.result and task.result.messages:
            messages = task.result.messages

        status_icons = {
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.TIMEOUT: "⏰",
            TaskStatus.RUNNING: "⚡",
            TaskStatus.CANCELLED: "🚫",
        }

        icon = status_icons.get(task.status, "❓")

        response_parts = [
            f"# {icon} Task Results: {task_id}",
            f"\n**Agent**: {task.request.agent_type.value}",
            f"**Status**: {task.status.name}",
            f"**Description**: {task.request.task_description}",
        ]

        # Optional color default via env
        import os
        use_color = bool(arguments.get("color", False) or os.getenv("INBOX_COLOR") == "1")

        def colorize(s: str, code: str) -> str:
            return f"\x1b[{code}m{s}\x1b[0m" if use_color else s

        # Add timing information + 1-line status with time since
        if task.result:
            now = datetime.now(timezone.utc)
            started = task.result.started_at
            completed = task.result.completed_at
            if task.result.duration_seconds is not None:
                response_parts.append(f"**Duration**: {colorize(f'{task.result.duration_seconds:.1f}s', '35')}")
            if started:
                response_parts.append(
                    f"**Started**: {colorize(started.strftime('%Y-%m-%d %H:%M:%S UTC'), '36')} ({colorize(str(int((now-started).total_seconds()))+'s', '33')} since)"
                )
            if completed:
                response_parts.append(
                    f"**Completed**: {colorize(completed.strftime('%Y-%m-%d %H:%M:%S UTC'), '36')} ({colorize(str(int((now-completed).total_seconds()))+'s', '35')} since)"
                )

        # Compact status line
        if task.status == TaskStatus.RUNNING:
            response_parts.append("\n" + colorize("⏺ Working • live updates flowing", '33'))
        elif task.status == TaskStatus.COMPLETED:
            response_parts.append("\n" + colorize("✅ Done • awaiting review", '32'))
        elif task.status == TaskStatus.FAILED:
            response_parts.append("\n" + colorize("❌ Failed • check error + action trail", '31'))
        elif task.status in [TaskStatus.PENDING, TaskStatus.STARTING]:
            response_parts.append("\n" + colorize("🚀 Starting • provisioning agent", '36'))
        elif task.status == TaskStatus.CANCELLED:
            response_parts.append("\n" + colorize("🛑 Cancelled", '90'))

        # Add error information
        if task.result and task.result.error:
            response_parts.append(f"\n## ❌ Error\n\n{task.result.error}")

        # Add final output
        if task.result and task.result.output:
            response_parts.append(f"\n## 📄 Final Report\n\n{task.result.output}")

        # Synthesize metrics and action trail from messages
        metrics_line = None
        metrics_obj = None
        actions = []
        if messages:
            for m in messages:
                getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None) or "unknown"
                raw_content = (getattr(m, "content", None) if not isinstance(m, dict) else m.get("content"))
                if raw_content is None:
                    raw_content = getattr(m, "message", None) if not isinstance(m, dict) else m.get("message")
                content = raw_content or ""
                if isinstance(content, str) and content.startswith("metrics: "):
                    metrics_line = content.replace("metrics: ", "").strip()
                    try:
                        import ast
                        metrics_obj = ast.literal_eval(metrics_line)
                    except Exception:
                        metrics_obj = None
                elif isinstance(content, str) and content.startswith("action: "):
                    actions.append(content.replace("action: ", "").strip())
        if metrics_line:
            response_parts.append(f"\n## 📊 Metrics\n\n{metrics_line}")
        if metrics_obj and isinstance(metrics_obj, dict):
            files = metrics_obj.get("files_touched") or []
            if isinstance(files, list) and files:
                response_parts.append("\n## 📁 Files Touched")
                max_show = 15
                for p in files[:max_show]:
                    response_parts.append(f"\n- {p}")
                if len(files) > max_show:
                    response_parts.append(f"\n- ... and {len(files) - max_show} more")
        if include_messages and actions:
            response_parts.append("\n## 🔀 Action Trail")
            for i, a in enumerate(actions[-20:], 1):
                if len(a) > max_msg_length:
                    a = a[:max_msg_length] + f"... (truncated from {len(a)} chars)"
                response_parts.append(f"\n{i}. {a}")

        response_parts.append(f"\n---\n**Task ID**: {task_id}")

        return "\n".join(response_parts)

    async def _list_tasks(self, arguments: dict[str, Any]) -> str:
        """List all tasks with optional filtering."""
        task_manager = agent_manager.get_task_manager()
        filter_status = arguments.get("filter_status", [])

        # Get all active tasks (this is a simplified implementation)
        # In a real implementation, you'd query the task storage
        all_tasks = list(task_manager.active_tasks.values())

        # Apply status filter
        if filter_status:
            filter_statuses = [TaskStatus(s) for s in filter_status]
            all_tasks = [task for task in all_tasks if task.status in filter_statuses]

        if not all_tasks:
            return "📭 No tasks found matching your criteria.\n\nUse `agent_async` to launch new background tasks."

        use_color = bool(arguments.get("color", False))
        def colorize(s: str, code: str) -> str:
            return f"\x1b[{code}m{s}\x1b[0m" if use_color else s

        response_parts = [f"# 📋 Agent Task Inbox ({len(all_tasks)} tasks)\n"]


        # Optional: filter to a batch
        batch_id = arguments.get("batch_id")
        if batch_id:
            from utils.batch_registry import get_batch
            rec = get_batch(batch_id)
            if rec:
                ids = set(rec.task_ids)
                all_tasks = [t for t in all_tasks if t.task_id in ids]

        # Group tasks by status
        status_groups = {}
        for task in all_tasks:
            status = task.status
            if status not in status_groups:
                status_groups[status] = []
            status_groups[status].append(task)

        status_icons = {
            TaskStatus.PENDING: "⏳",
            TaskStatus.STARTING: "🚀",
            TaskStatus.RUNNING: "⚡",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.TIMEOUT: "⏰",
            TaskStatus.CANCELLED: "🚫",
        }

        for status, tasks in status_groups.items():
            icon = status_icons.get(status, "❓")
            response_parts.append(f"\n## {icon} {status.value.title()} ({len(tasks)})")

            for task in tasks:
                now = datetime.now(timezone.utc)
                since_started = (now - task.created_at).total_seconds()
                since_finished = None
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED]:
                    since_finished = (now - task.updated_at).total_seconds()
                # Latest action (if any) from result messages
                latest_action = None
                if task.result and task.result.messages:
                    for m in reversed(task.result.messages):
                        content = getattr(m, "message", None) or getattr(m, "content", None) if not isinstance(m, dict) else (m.get("message") or m.get("content"))
                        if isinstance(content, str) and content.startswith("action: "):
                            latest_action = content.replace("action: ", "").strip()
                            break
                short_desc = task.request.task_description[:60] + (
                    "..." if len(task.request.task_description) > 60 else ""
                )
                status_emoji = status_icons.get(task.status, "❓")
                # Colorize status emoji if requested
                if use_color:
                    color_map = {
                        TaskStatus.RUNNING: "33",  # yellow
                        TaskStatus.COMPLETED: "32",  # green
                        TaskStatus.FAILED: "31",  # red
                        TaskStatus.PENDING: "34",  # blue
                        TaskStatus.STARTING: "36",  # cyan
                        TaskStatus.TIMEOUT: "35",  # magenta
                        TaskStatus.CANCELLED: "90",  # grey
                    }
                    emoji_col = colorize(status_emoji, color_map.get(task.status, "0"))
                else:
                    emoji_col = status_emoji

                # Build duration string first
                dur_str = f"{since_started:.0f}s"
                if since_finished is not None:
                    dur_str += f" | +{since_finished:.0f}s"

                # Compose line using (possibly) colored emoji
                line = (
                    f"\n- {emoji_col} {task.task_id} | {task.request.agent_type.value} | {short_desc} | {dur_str}"
                )
                if latest_action:
                    # keep it compact
                    la = latest_action if len(latest_action) <= 80 else latest_action[:80] + "..."
                    line += f" | {la}"
                response_parts.append(line)

        response_parts.extend(
            [
                "\n---",
                "\n💡 **Tips**:",
                "- Use `agent_inbox` with a specific `task_id` to get detailed status",
                "- Use `action: results` to get complete results and conversation history",
                "- Use `action: cancel` to stop a running task",
            ]
        )

        return "\n".join(response_parts)

    async def _cancel_task(self, task_id: str) -> str:
        """Cancel a running task."""
        task_manager = agent_manager.get_task_manager()
        task = await task_manager.get_task(task_id)

        if not task:
            return f"❌ Task `{task_id}` not found."

        if task.status not in [TaskStatus.PENDING, TaskStatus.STARTING, TaskStatus.RUNNING]:
            return f"⚠️ Task `{task_id}` cannot be cancelled (status: {task.status.value})"

        # Update task status
        task.status = TaskStatus.CANCELLED
        task.updated_at = datetime.now(timezone.utc)

        # Try to stop the agent process
        if task.process_id:
            try:
                import subprocess

                subprocess.run(["kill", str(task.process_id)], timeout=5)
            except Exception as e:
                logger.debug(f"Error stopping process {task.process_id}: {e}")

        return f"🚫 Task `{task_id}` has been cancelled."

    async def _get_live_agent_status(self, port: int) -> Optional[str]:
        """Get live status from running agent."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:{port}/status", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("status")
        except Exception:
            pass
        return None

    async def _get_live_agent_messages(self, port: int) -> list[dict[str, Any]]:
        """Get live messages from running agent."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:{port}/messages", timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("messages", [])
        except Exception:
            pass
        return []
