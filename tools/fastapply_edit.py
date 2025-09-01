"""
FastApply Edit tool

High-precision, provider-backed file edits with dry-run preview and optional apply.

Parameters:
- instructions (str, optional): natural language edit intent (Morph recommended)
- edits (array, optional): explicit operations for builtin provider
  Each op: {type: replace|insert|write, filepath, find?, replace?, count?, anchor?, content?, position?}
- dry_run (bool, default true): preview diff without writing
- apply (bool, default false): apply changes to disk
- task_id (str, optional): if provided, stream edit events under this task via SSE/WebSocket
- provider (str, optional): override env `ZEN_EDIT_PROVIDER`

Env:
- ZEN_EDIT_PROVIDER=builtin|morph
- MORPH_API_KEY, MORPH_BASE_URL (when provider=morph)
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from tools.models import ToolOutput
from tools.shared.base_tool import BaseTool
from utils.edit_providers import get_edit_provider
from utils.streaming_protocol import StreamMessageType, get_streaming_manager

logger = logging.getLogger(__name__)


class FastApplyRequest(BaseModel):
    instructions: str | None = Field(
        default=None,
        description="Natural language edit intent. Recommended with Morph provider.",
    )
    edits: list[dict[str, Any]] | None = Field(
        default=None,
        description="Explicit operations for builtin provider (replace|insert|write).",
    )
    dry_run: bool = Field(default=True, description="Preview diffs without writing to disk.")
    apply: bool = Field(default=False, description="Apply changes to disk if true.")
    task_id: str | None = Field(default=None, description="Stream events under this task ID if provided.")
    provider: str | None = Field(default=None, description="Override env-selected edit provider.")


class FastApplyEditTool(BaseTool):
    def get_name(self) -> str:
        return "fastapply_edit"

    def get_description(self) -> str:
        return (
            "Plan and apply precise code edits with dry-run diffs. "
            "Backed by Morph Fast Apply when configured, with builtin fallback for explicit ops."
        )

    def get_input_schema(self) -> dict[str, Any]:
        return FastApplyRequest.model_json_schema()

    def get_system_prompt(self) -> str:
        return (
            "You are a precise code-edit executor. Prefer minimal, safe changes. "
            "When generating explicit operations, use small, targeted replacements."
        )

    def requires_model(self) -> bool:
        # This tool executes provider logic, not a model call
        return False

    def get_request_model(self):
        """Return the request model class for validation."""
        return FastApplyRequest

    async def prepare_prompt(self, request) -> str:
        """This tool doesn't use AI models, so no prompt preparation needed."""
        return ""

    async def execute(self, *, arguments: dict[str, Any], **kwargs) -> ToolOutput:
        req = FastApplyRequest(**arguments)
        provider = get_edit_provider(req.provider)

        # Build plan
        plan = provider.plan_edit(
            instructions=req.instructions,
            operations=req.edits,
            context={},
        )

        # Dry-run first
        preview = provider.dry_run(plan)

        # Optionally apply
        applied = None
        if req.apply and not req.dry_run:
            applied = provider.apply(plan)

        # Streaming events if task_id provided
        if req.task_id:
            try:
                sm = get_streaming_manager()
                await sm.broadcast_message(
                    req.task_id,
                    StreamMessageType.FILE_UPDATE,
                    {"edit_plan": {"plan_id": plan.id, "provider": provider.name}},
                    agent_type="edit"
                )
                await sm.broadcast_message(
                    req.task_id,
                    StreamMessageType.FILE_UPDATE,
                    {"edit_preview": {"files": preview.get("files", []), "errors": preview.get("errors")}},
                    agent_type="edit"
                )
                if applied:
                    await sm.broadcast_message(
                        req.task_id,
                        StreamMessageType.FILE_UPDATE,
                        {"edit_applied": {"files_changed": applied.get("files_changed", 0)}},
                        agent_type="edit"
                    )
            except Exception:
                logger.debug("Streaming edit events failed", exc_info=True)

        # Format result
        summary = (
            f"Provider={provider.name} plan={plan.id} "
            f"files={len(preview.get('files', []))} applied={bool(applied)}"
        )
        output_payload = {
            "provider": provider.name,
            "plan_id": plan.id,
            "preview": preview,
            "applied": applied,
        }
        return ToolOutput(
            content=[
                {
                    "type": "text",
                    "text": summary,
                },
                {
                    "type": "json",
                    "value": output_payload,
                },
            ]
        )

