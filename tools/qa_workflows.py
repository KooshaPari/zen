"""
QA Workflows Tool - Guided smoke/regression workflows across multiple surfaces

This workflow tool orchestrates structured QA checks for:
- CLI (MCP stdio)
- API (server_http)
- Web (frontend via Playwright)
- MCP (HTTP transport)
- Desktop (Desktop Automation MCP)
- Mobile (mobile-next MCP)

It provides stepwise guidance and success criteria. It does not execute
tests itself; instead, it coordinates the agent’s actions and consolidates
findings using the standard workflow pattern.
"""

from typing import TYPE_CHECKING, Any, Optional

from pydantic import Field

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from config import TEMPERATURE_ANALYTICAL
from systemprompts import QA_WORKFLOWS_PROMPT
from tools.shared.base_models import WorkflowRequest

from .workflow.base import WorkflowTool

QA_TARGETS = [
    "cli",
    "api",
    "web",
    "mcp_stdio",
    "mcp_http",
    "desktop",
    "mobile",
]


class QAWorkflowRequest(WorkflowRequest):
    """Workflow request for QA workflows"""

    target: str = Field(..., description="QA target surface", json_schema_extra={"enum": QA_TARGETS})
    api_base: Optional[str] = Field(None, description="Base URL for API checks, e.g., http://localhost:8080")
    web_base: Optional[str] = Field(None, description="Base URL for web checks (if applicable)")
    mcp_http_base: Optional[str] = Field(None, description="Base URL for MCP HTTP transport (if applicable)")


class QAWorkflowsTool(WorkflowTool):
    """Guided QA workflow tool with per-surface playbooks and parallelization tips"""

    def get_name(self) -> str:
        return "qa_workflows"

    def get_description(self) -> str:
        return (
            "Guide agents through non-destructive QA smoke/regression workflows for CLI (stdio), API, Web, MCP (HTTP), "
            "Desktop MCP, and Mobile MCP. Provides per-surface steps, success criteria, and consolidation."
        )

    def get_system_prompt(self) -> str:
        return QA_WORKFLOWS_PROMPT

    def get_default_temperature(self) -> float:
        return TEMPERATURE_ANALYTICAL

    def get_model_category(self) -> "ToolModelCategory":
        from tools.models import ToolModelCategory

        return ToolModelCategory.ANALYTICAL

    def get_workflow_request_model(self):
        return QAWorkflowRequest

    def get_tool_fields(self) -> dict[str, dict[str, Any]]:
        return {
            "target": {
                "type": "string",
                "enum": QA_TARGETS,
                "description": "QA target surface",
            },
            "api_base": {"type": "string", "description": "Base URL for API checks"},
            "web_base": {"type": "string", "description": "Base URL for web checks"},
            "mcp_http_base": {"type": "string", "description": "Base URL for MCP HTTP checks"},
        }

    def get_required_fields(self) -> list[str]:
        return ["target", "step", "step_number", "total_steps", "next_step_required", "findings"]

    def get_required_actions(
        self, step_number: int, confidence: str, findings: str, total_steps: int, request=None
    ) -> list[str]:
        target = getattr(request, "target", "api") if request else "api"

        # Per-surface base actions
        if target == "cli":
            base = [
                "Initialize MCP stdio (initialize)",
                "List tools (tools/list), verify non-empty",
                "Optionally call a safe tool (version or listmodels)",
            ]
        elif target == "api":
            base = [
                "GET /health returns 200 and valid JSON",
                "POST /tasks creates a task and returns task_id",
                "GET /tasks/{task_id}/results completes with structured payload",
                "POST /llm/batch with batch_mode=parallel returns results summary",
            ]
        elif target == "web":
            base = [
                "Playwright: assert /health returns 200 JSON",
                "If UI present: load homepage and assert key element",
            ]
        elif target == "mcp_http":
            base = [
                "Connect to MCP HTTP server",
                "List tools, verify non-empty",
                "Get a prompt, call a safe tool",
            ]
        elif target == "mcp_stdio":
            base = [
                "Initialize stdio session",
                "List tools (tools/list)",
                "Optionally call a safe tool",
            ]
        elif target == "desktop":
            base = [
                "Ensure Desktop MCP is attached",
                "List tools via MCP",
                "Call a non-destructive info tool (e.g., window info)",
            ]
        elif target == "mobile":
            base = [
                "Ensure mobile-next MCP attached with proper transport/auth",
                "List tools",
                "Call a safe info tool (e.g., device info)",
            ]
        else:
            base = ["Define target-specific actions"]

        # Use the standard helper to adapt by step/confidence
        return self.get_standard_required_actions(step_number, confidence or "low", base)

    def requires_expert_analysis(self) -> bool:
        # Optional expert model at the end — allow, but not mandatory
        return True

    def prepare_expert_analysis_context(self, consolidated_findings) -> str:
        # Summarize QA evidence succinctly
        parts = [
            "=== QA WORKFLOWS SUMMARY ===",
            f"Total steps: {len(consolidated_findings.findings)}",
            f"Issues found: {len(consolidated_findings.issues_found)}",
            "",
            "=== KEY FINDINGS ===",
        ]
        if consolidated_findings.findings:
            parts.extend(consolidated_findings.findings[-10:])
        if consolidated_findings.issues_found:
            parts.append("\n=== ISSUES ===")
            for issue in consolidated_findings.issues_found:
                sev = issue.get("severity", "unknown").upper()
                desc = issue.get("description", "")
                parts.append(f"[{sev}] {desc}")
        parts.append("\nProvide a concise QA assessment and recommended next actions.")
        return "\n".join(parts)

