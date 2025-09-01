"""
Batch Summary Tool

Provides an aggregated summary view for a previously launched batch
of agent tasks, identified by batch_id. Uses the in-memory batch registry.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from mcp.types import TextContent

import utils.agent_manager as agent_manager
from tools.shared.agent_models import TaskStatus
from tools.shared.base_tool import BaseTool
from utils.batch_registry import get_batch

logger = logging.getLogger(__name__)


class AgentBatchSummaryTool(BaseTool):
    """Summarize a launched batch of agent tasks by batch_id."""

    name = "agent_batch_summary"
    description = "Generate an aggregated summary for a batch using its batch_id."

    def get_annotations(self) -> dict[str, Any] | None:
        return {
            "category": "agent-orchestration",
            "tags": ["agent", "batch", "summary"],
            "destructiveHint": False,
            "readOnlyHint": True,
        }

    def requires_model(self) -> bool:
        return False

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_request_model(self):
        from pydantic import BaseModel, Field

        class BatchSummaryRequest(BaseModel):
            batch_id: str = Field(..., description="Batch ID returned by agent_batch")
            filter_status: list[str] | None = Field(None, description="Optional statuses to include")

        return BatchSummaryRequest

    def get_system_prompt(self) -> str:
        return "You summarize a batch of agent tasks identified by batch_id."

    async def prepare_prompt(self, request) -> str:
        return "Batch summary tool - coordination only."

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "batch_id": {"type": "string", "description": "Batch ID"},
                "filter_status": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "pending",
                            "starting",
                            "running",
                            "completed",
                            "failed",
                            "timeout",
                            "cancelled",
                        ],
                    },
                    "description": "Only include these statuses in the summary",
                },
            },
            "required": ["batch_id"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            batch_id = arguments["batch_id"]
            filter_status = arguments.get("filter_status") or []

            reg = get_batch(batch_id)
            if not reg:
                return [TextContent(type="text", text=f"❌ Batch `{batch_id}` not found.")]

            tm = agent_manager.get_task_manager()
            tasks = []
            for tid in reg.task_ids:
                t = await tm.get_task(tid)
                if t:
                    tasks.append(t)

            if filter_status:
                try:
                    statuses = [TaskStatus(s) for s in filter_status]
                    tasks = [t for t in tasks if t.status in statuses]
                except Exception:
                    pass

            if not tasks:
                return [TextContent(type="text", text=f"📭 No tasks recorded for batch `{batch_id}`.")]

            # Aggregate similar to agent_inbox summary
            from collections import Counter
            status_counter = Counter([t.status for t in tasks])
            steps_total = tools_total = files_changed_total = 0
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
            for task in tasks:
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

            response_parts = [
                f"# 📦 Batch Summary: {batch_id}",
                f"\nDescription: {reg.description}" if reg.description else "",
                f"Tasks: {len(tasks)}",
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

            return [TextContent(type="text", text="\n".join(response_parts))]

        except Exception as e:
            logger.error(f"Error generating batch summary: {e}")
            return [TextContent(type="text", text=f"Error generating batch summary: {str(e)}")]

