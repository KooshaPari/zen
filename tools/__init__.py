"""
Tool implementations and registry helpers for Zen MCP Server.

Exposes concrete tool classes and a lightweight registry function
`get_all_tools()` used by the HTTP server variant to discover tools
without relying on the stdio MCP server registry.
"""

from __future__ import annotations

# Tool filtering via environment (parity with stdio server)
import os
from collections.abc import Awaitable
from typing import Any, Callable, Dict, Set

from .a2a_tool import A2ATool
from .agent_async import AgentAsyncTool
from .agent_batch import AgentBatchTool
from .agent_batch_summary import AgentBatchSummaryTool
from .agent_doctor import AgentDoctorTool
from .agent_inbox import AgentInboxTool
from .agent_or_llm import AgentOrLLMTool
from .agent_registry import AgentRegistryTool
from .agent_sync import AgentSyncTool
from .analyze import AnalyzeTool
from .challenge import ChallengeTool
from .chat import ChatTool
from .codereview import CodeReviewTool
from .consensus import ConsensusTool
from .debug import DebugIssueTool
from .docgen import DocgenTool
from .fastapply_edit import FastApplyEditTool
from .listmodels import ListModelsTool
from .messaging_tool import MessagingTool
from .planner import PlannerTool
from .precommit import PrecommitTool
from .project_tool import ProjectTool
from .qa_workflows import QAWorkflowsTool
from .refactor import RefactorTool
from .secaudit import SecauditTool
from .sem_ingest import SemIngestTool
from .sem_rag_search import SemRAGSearchTool
from .testgen import TestGenTool
from .thinkdeep import ThinkDeepTool
from .tracer import TracerTool
from .universal_executor import DeployTool
from .version import VersionTool

# Tools that must remain enabled regardless of DISABLED_TOOLS
ESSENTIAL_TOOLS: set[str] = {"version", "listmodels"}


def _parse_disabled_tools_env() -> set[str]:
    """Parse DISABLED_TOOLS env var into a set of tool names (lowercase)."""
    raw = os.getenv("DISABLED_TOOLS", "").strip()
    if not raw:
        return set()
    return {t.strip().lower() for t in raw.split(",") if t.strip()}

# Temporarily disabled - needs abstract method implementations
# from .workflow_orchestrator import WorkflowOrchestratorTool

__all__ = [
    "ThinkDeepTool",
    "CodeReviewTool",
    "DebugIssueTool",
    "DocgenTool",
    "AnalyzeTool",
    "ChatTool",
    "ConsensusTool",
    "ListModelsTool",
    "PlannerTool",
    "PrecommitTool",
    "ChallengeTool",
    "RefactorTool",
    "SecauditTool",
    "QAWorkflowsTool",
    "TestGenTool",
    "TracerTool",
    "DeployTool",
    "VersionTool",
    "AgentRegistryTool",
    "AgentSyncTool",
    "AgentAsyncTool",
    "AgentInboxTool",
    "AgentOrLLMTool",
    "AgentBatchTool",
    "AgentBatchSummaryTool",
    "AgentDoctorTool",
    "FastApplyEditTool",
]


def _instantiate_tools() -> dict[str, Any]:
    """Create streamlined tool instances keyed by their registered name.

    Mirrors the stdio server's streamlined TOOLS registry - essential operational tools only.
    Deploy tool consolidates all execution capabilities under a unified interface.
    """
    # Core execution - unified interface for all capabilities
    instances = [DeployTool()]

    # Essential operational tools
    instances += [
        AgentRegistryTool(),
        AgentInboxTool(),
        ListModelsTool(),
        VersionTool(),
        AgentDoctorTool(),
    ]

    # Semantic tools (pgvector) - commented out as they're abstract classes
    # instances += [SemIngestTool(), SemRAGSearchTool()]

    # New comms/project/A2A tools: use stubs via get_all_tools() to avoid instantiating abstract classes

    # Legacy tools (optionally disabled)
    disable_legacy = os.getenv("DISABLE_LEGACY_TOOLS", "0").lower() in ("1", "true", "yes")
    if not disable_legacy:
        instances += [
            ChatTool(),
            AnalyzeTool(),
            ThinkDeepTool(),
            CodeReviewTool(),
            PlannerTool(),
            DebugIssueTool(),
            TestGenTool(),
            ConsensusTool(),
        ]

    tools = {tool.get_name(): tool for tool in instances}

    # Apply DISABLED_TOOLS filtering (keep essentials)
    disabled = _parse_disabled_tools_env()
    if disabled:
        filtered = {}
        for name, tool in tools.items():
            lname = name.lower()
            if lname in ESSENTIAL_TOOLS or lname not in disabled:
                filtered[name] = tool
        tools = filtered

    return tools


def get_all_tools() -> dict[str, dict[str, Any]]:
    """Return a mapping of tool metadata and callable wrappers.

    Shape expected by server_mcp_http:
      {
        "tool_name": {
           "description": str,
           "input_schema": { ... },
           "function": async callable(**kwargs) -> Any,
        },
        ...
      }
    """
    tools: dict[str, Any] = _instantiate_tools()
    registry: dict[str, dict[str, Any]] = {}

    for name, tool in tools.items():
        # Prepare async wrapper that always calls execute(arguments=kwargs)
        async def _runner(_tool=tool, **kwargs) -> Any:  # default arg to bind current tool
            return await _tool.execute(arguments=kwargs)

        try:
            schema = tool.get_input_schema()
        except Exception:
            # Fall back to minimal schema if generation fails
            schema = {"type": "object", "properties": {}}

        try:
            desc = tool.get_description()
        except Exception:
            desc = f"{name} tool"

        registry[name] = {
            "description": desc,
            "input_schema": schema,
            "function": _runner,
            "annotations": (tool.get_annotations() or {}),
        }

    # Add stubs for new tools when legacy set is disabled, without instantiating abstract classes
    if os.getenv("DISABLE_LEGACY_TOOLS", "0").lower() in ("1", "true", "yes"):
        async def _not_implemented(**kwargs):
            return [
                {"type": "text", "text": "This tool is not implemented in this build."}
            ]
        for nm, desc in (
            ("messaging", "Messaging utilities (stub)"),
            ("project", "Project operations (stub)"),
            ("a2a", "Agent-to-agent protocol (stub)"),
        ):
            if nm not in registry:
                registry[nm] = {
                    "description": desc,
                    "input_schema": {"type": "object", "properties": {"action": {"type": "string"}}},
                    "function": _not_implemented,
                    "annotations": {},
                }

    return registry

    # Note: return above, keep code consistent
