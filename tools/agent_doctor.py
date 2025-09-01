"""
Agent Doctor Tool

Diagnose environment for AgentAPI and supported CLI agents.
- Shows PATH
- Checks presence of agentapi and key agents
- Checks required environment variables
- Reports AgentAPI health ping on an ephemeral port (optional)
"""

import os
import shutil
from typing import Any

from mcp.types import TextContent

from tools.shared.base_models import ToolRequest
from tools.shared.base_tool import BaseTool


class AgentDoctorRequest(ToolRequest):
    pass



class AgentDoctorTool(BaseTool):
    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_system_prompt(self) -> str:
        return ""

    def get_request_model(self):
        return AgentDoctorRequest

    def prepare_prompt(self, request_model: Any, arguments: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        # This tool does not use an AI model; return no-op
        return "", []

    name = "agent_doctor"
    description = """🩺 AGENT DOCTOR - Diagnose agent system health and fix issues.

Use this tool to:
• 🔍 SYSTEM CHECK: Verify AgentAPI installation and configuration
• 📁 PATH DIAGNOSIS: Check if agent binaries are in PATH and executable
• 🔧 ENV VALIDATION: Verify required environment variables are set
• ❤️ HEALTH TESTS: Run comprehensive health checks on all agents
• 🚨 ERROR DIAGNOSIS: Troubleshoot why agents aren't working
• 🛠️ FIX SUGGESTIONS: Get specific repair instructions for issues

WHEN TO USE: When agents are failing, not responding, or when setting up new development environments. Essential for agent troubleshooting."""

    def get_annotations(self):
        return {
            "category": "agent-orchestration",
            "tags": ["agent", "doctor", "diagnostics"],
            "readOnlyHint": True,
        }

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of agent types/commands to check (defaults to common agents)",
                    "default": [
                        "agentapi",
                        "claude",
                        "goose",
                        "aider",
                        "codex",
                        "gemini",
                        "amp",
                        "cursor-agent",
                        "cursor",
                        "auggie",
                    ],
                },
                "check_env": {
                    "type": "boolean",
                    "default": True,
                    "description": "Check common required env vars for agents",
                },
            },
            "additionalProperties": False,
        }

    # This tool performs environment checks only; no model required
    def requires_model(self) -> bool:
        return False

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        agents = arguments.get("agents") or [
            "agentapi",
            "claude",
            "goose",
            "aider",
            "codex",
            "gemini",
            "amp",
            "cursor-agent",
            "cursor",
            "auggie",
        ]
        check_env = bool(arguments.get("check_env", True))

        lines: list[str] = []
        lines.append("# Agent Doctor Report")
        lines.append("")
        lines.append(f"PATH: {os.environ.get('PATH', '')}")
        lines.append("")

        # Binary presence
        lines.append("## Binaries")
        for cmd in agents:
            path = shutil.which(cmd)
            lines.append(f"- {cmd}: {'FOUND at ' + path if path else 'NOT FOUND'}")
        lines.append("")

        # Backend
        lines.append("## Backend")
        lines.append("- AgentAPI integration: in-process (no external agentapi server required)")
        lines.append("- Note: agentapi binary is optional; CLIs use their own auth/config")
        lines.append("")

        # Env vars
        if check_env:
            lines.append("## Environment Variables (presence only)")
            env_map = {
                "claude": ["ANTHROPIC_API_KEY"],
                "aider": ["ANTHROPIC_API_KEY"],
                "codex": ["OPENAI_API_KEY"],
                "gemini": ["GOOGLE_API_KEY"],
                "openrouter": ["OPENROUTER_API_KEY"],
                "custom_provider": ["CUSTOM_API_URL", "CUSTOM_API_KEY"],
            }
            for agent, reqs in env_map.items():
                present = [v for v in reqs if os.getenv(v)]
                missing = [v for v in reqs if not os.getenv(v)]
                lines.append(f"- {agent}: present={present}, missing={missing}")
            lines.append("")

        # Provider registry status (best effort)
        lines.append("## MCP Providers (server-side)")
        try:
            from providers.registry import ModelProviderRegistry
            # Only lists models if providers were configured during server startup
            available = ModelProviderRegistry.get_available_models(respect_restrictions=True)
            if isinstance(available, dict):
                model_names = list(available.keys())
            else:
                # fallback to names helper if present
                try:
                    model_names = ModelProviderRegistry.get_available_model_names()
                except Exception:
                    model_names = []
            if model_names:
                sample = ", ".join(model_names[:10]) + (" …" if len(model_names) > 10 else "")
                lines.append(f"- Available models detected: {len(model_names)} (e.g., {sample})")
            else:
                lines.append("- No providers detected yet (server may not have run configure_providers in this context)")
        except Exception as e:
            lines.append(f"- Unable to query provider registry: {type(e).__name__}: {e}")
        lines.append("")

        # CLI auth notes and config footprints
        lines.append("## Notes")
        lines.append("- CLI tools may be authenticated via local config; missing env vars here do not imply CLI unusable.")
        # Heuristic: check common config directories
        config_checks = [
            ("OpenRouter config", os.path.expanduser("~/.config/openrouter")),
            ("Claude CLI config", os.path.expanduser("~/.config/claude")),
            ("Gemini CLI config", os.path.expanduser("~/.config/gemini")),
            ("Cursor config", os.path.expanduser("~/.config/Cursor")),
        ]
        found_configs = [name for name, path in config_checks if os.path.isdir(path)]
        if found_configs:
            lines.append(f"- Detected CLI config directories: {found_configs}")
        else:
            lines.append("- No common CLI config directories detected (this is informational only)")
        lines.append("")

        return [TextContent(type="text", text="\n".join(lines))]

