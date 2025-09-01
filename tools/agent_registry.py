"""
Agent Registry Tool

This tool discovers and describes available AgentAPI-supported CLI agents
with their capabilities, helping the primary agent choose appropriate sub-agents.
"""

import logging
import shutil
import subprocess
from typing import Any, Optional

from mcp.types import TextContent

from tools.shared.agent_models import (
    AgentCapability,
    AgentDefinition,
    AgentRegistryEntry,
    AgentType,
)
from tools.shared.base_tool import BaseTool

logger = logging.getLogger(__name__)


class AgentRegistryTool(BaseTool):
    """Tool for discovering and managing available agents."""

    name = "agent_registry"
    description = """🤖 AGENT REGISTRY - Discover available AI agents and their capabilities.

Use this tool to:
• 🔍 DISCOVER AGENTS: Find all available AI agents (Claude, Aider, Continue, etc.)
• 📋 LIST CAPABILITIES: See what each agent can do (coding, analysis, file editing)
• ✅ CHECK AVAILABILITY: Verify which agents are installed and ready to use
• 🎯 AGENT SELECTION: Choose the right agent for specific tasks

WHEN TO USE: Before using agent_inbox or deploy tool with agent_type='agent', use this to discover what agents are available and their strengths."""
    def get_annotations(self) -> Optional[dict[str, Any]]:
        """Agent orchestration tool; discovery (read-only).
        """
        return {
            "category": "agent-orchestration",
            "tags": ["agent", "registry", "discovery"],
            "destructiveHint": False,
            "readOnlyHint": True,
        }

    def __init__(self):
        super().__init__()
        self._agent_definitions = self._initialize_agent_definitions()

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

        class AgentRegistryRequest(BaseModel):
            agent_type: Optional[str] = Field(None, description="Specific agent type to query (optional)")
            check_availability: bool = Field(True, description="Whether to check if agents are installed")
            include_capabilities: bool = Field(True, description="Whether to include detailed capabilities")

        return AgentRegistryRequest

    def get_system_prompt(self) -> str:
        """Return the system prompt for this tool."""
        return "You are an agent registry tool that discovers and describes available CLI agents."

    async def prepare_prompt(self, request) -> str:
        """Prepare the prompt for the agent registry tool."""
        return "Agent registry tool - no prompt preparation needed as this is a data-only tool."

    def requires_model(self) -> bool:
        """This tool doesn't require AI model access."""
        return False

    def get_input_schema(self) -> dict[str, Any]:
        """Get the input schema for the agent registry tool."""
        return {
            "type": "object",
            "properties": {
                "agent_type": {
                    "type": "string",
                    "enum": [agent.value for agent in AgentType],
                    "description": "Specific agent type to query (optional - returns all if not specified)",
                },
                "check_availability": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to check if agents are actually installed and available",
                },
                "include_capabilities": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to include detailed capability descriptions",
                },
            },
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Execute the agent registry tool."""
        agent_type = arguments.get("agent_type")
        check_availability = arguments.get("check_availability", True)
        include_capabilities = arguments.get("include_capabilities", True)

        try:
            # Get registry entries
            if agent_type:
                entries = [await self._get_agent_entry(AgentType(agent_type), check_availability)]
            else:
                entries = []
                for agent in AgentType:
                    entry = await self._get_agent_entry(agent, check_availability)
                    entries.append(entry)

            # Format response
            response = self._format_registry_response(entries, include_capabilities)

            return [TextContent(type="text", text=response)]

        except Exception as e:
            logger.error(f"Error in agent registry: {e}")
            return [TextContent(type="text", text=f"Error querying agent registry: {str(e)}")]

    async def _get_agent_entry(self, agent_type: AgentType, check_availability: bool) -> AgentRegistryEntry:
        """Get registry entry for a specific agent type."""
        definition = self._agent_definitions.get(agent_type)
        if not definition:
            # Create basic definition for unknown agent types
            definition = AgentDefinition(
                agent_type=agent_type,
                name=agent_type.value.title(),
                description=f"{agent_type.value} agent",
                command_template=agent_type.value,
            )

        entry = AgentRegistryEntry(agent_type=agent_type, definition=definition, available=True)  # Default to available

        if check_availability:
            # Check if agent is actually installed
            entry.available = await self._check_agent_availability(agent_type)
            if entry.available:
                entry.installation_path = shutil.which(self._get_agent_command(agent_type))
                entry.version = await self._get_agent_version(agent_type)

        return entry

    async def _check_agent_availability(self, agent_type: AgentType) -> bool:
        """Check if an agent is available on the system."""
        command = self._get_agent_command(agent_type)

        # Check if command exists in PATH
        if not shutil.which(command):
            return False

        # For some agents, we can do a quick version check
        try:
            result = subprocess.run([command, "--version"], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            # If version check fails, just check if executable exists
            return shutil.which(command) is not None

    async def _get_agent_version(self, agent_type: AgentType) -> Optional[str]:
        """Get version information for an agent."""
        command = self._get_agent_command(agent_type)

        try:
            result = subprocess.run([command, "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass

        return None

    def _get_agent_command(self, agent_type: AgentType) -> str:
        """Get the command to run a specific agent type."""
        commands = {
            AgentType.CLAUDE: "claude",
            AgentType.GOOSE: "goose",
            AgentType.AIDER: "aider",
            AgentType.CODEX: "codex",
            AgentType.GEMINI: "gemini",
            AgentType.AMP: "amp",
            AgentType.CURSOR_AGENT: "cursor-agent",
            AgentType.CURSOR: "cursor",
            AgentType.AUGGIE: "auggie",
        }
        return commands.get(agent_type, agent_type.value)

    def _initialize_agent_definitions(self) -> dict[AgentType, AgentDefinition]:
        """Initialize comprehensive agent definitions with capabilities."""
        return {
            AgentType.CLAUDE: AgentDefinition(
                agent_type=AgentType.CLAUDE,
                name="Claude Code",
                description="Anthropic's Claude optimized for coding tasks with advanced reasoning capabilities",
                capabilities=[
                    AgentCapability(
                        name="Code Generation",
                        description="Generate high-quality code in multiple programming languages",
                        use_cases=["Creating new features", "Writing boilerplate code", "API implementations"],
                        strengths=["Clean, readable code", "Best practices", "Documentation"],
                        limitations=["Requires clear specifications", "May need iteration for complex logic"],
                    ),
                    AgentCapability(
                        name="Code Analysis",
                        description="Analyze existing code for issues, improvements, and understanding",
                        use_cases=["Code reviews", "Bug analysis", "Architecture assessment"],
                        strengths=["Deep understanding", "Security awareness", "Performance insights"],
                        limitations=["Limited to provided context", "May miss runtime behavior"],
                    ),
                    AgentCapability(
                        name="Debugging",
                        description="Help identify and fix bugs in code",
                        use_cases=["Error investigation", "Logic debugging", "Performance issues"],
                        strengths=["Systematic approach", "Multiple debugging strategies", "Root cause analysis"],
                        limitations=["Needs error context", "Cannot run code directly"],
                    ),
                ],
                command_template="claude",
                default_args=["--allowedTools", "Bash(git*) Edit Replace"],
                required_env_vars=["ANTHROPIC_API_KEY"],
                timeout_seconds=600,
            ),
            AgentType.AIDER: AgentDefinition(
                agent_type=AgentType.AIDER,
                name="Aider",
                description="AI pair programming tool that can edit code in your local git repository",
                capabilities=[
                    AgentCapability(
                        name="Direct Code Editing",
                        description="Directly edit files in your repository with git integration",
                        use_cases=["Implementing features", "Refactoring code", "Fixing bugs"],
                        strengths=["Git integration", "Direct file editing", "Incremental changes"],
                        limitations=["Requires git repository", "May make unwanted changes"],
                    ),
                    AgentCapability(
                        name="Repository Understanding",
                        description="Understands entire codebase context for better edits",
                        use_cases=["Large refactors", "Cross-file changes", "Architecture updates"],
                        strengths=["Full repo context", "Dependency tracking", "Consistent changes"],
                        limitations=["Can be slow on large repos", "High token usage"],
                    ),
                ],
                command_template="aider",
                default_args=["--model", "sonnet"],
                required_env_vars=["ANTHROPIC_API_KEY"],
                timeout_seconds=900,
            ),
            AgentType.GOOSE: AgentDefinition(
                agent_type=AgentType.GOOSE,
                name="Goose",
                description="Developer agent that can run commands and edit files to accomplish tasks",
                capabilities=[
                    AgentCapability(
                        name="Task Automation",
                        description="Automate development tasks by running commands and editing files",
                        use_cases=["Build automation", "Testing workflows", "Deployment tasks"],
                        strengths=["Command execution", "File manipulation", "Task orchestration"],
                        limitations=["Requires careful permission management", "Can be destructive"],
                    ),
                    AgentCapability(
                        name="Environment Setup",
                        description="Set up development environments and dependencies",
                        use_cases=["Project initialization", "Dependency management", "Configuration"],
                        strengths=["System integration", "Package management", "Environment configuration"],
                        limitations=["Platform-specific", "Requires system permissions"],
                    ),
                ],
                command_template="goose",
                default_args=[],
                timeout_seconds=600,
            ),
        }
        # Add more agent definitions as needed

    def _format_registry_response(self, entries: list[AgentRegistryEntry], include_capabilities: bool) -> str:
        """Format the registry response for display."""
        if not entries:
            return "No agents found in registry."

        response_parts = ["# Available Agent Registry\n"]

        available_count = sum(1 for entry in entries if entry.available)
        total_count = len(entries)

        response_parts.append(f"**Status**: {available_count}/{total_count} agents available\n")

        for entry in entries:
            status_icon = "✅" if entry.available else "❌"
            response_parts.append(f"\n## {status_icon} {entry.definition.name} ({entry.agent_type.value})")

            response_parts.append(f"\n**Description**: {entry.definition.description}")

            if entry.available:
                if entry.installation_path:
                    response_parts.append(f"**Installation**: {entry.installation_path}")
                if entry.version:
                    response_parts.append(f"**Version**: {entry.version}")
            else:
                response_parts.append("**Status**: Not installed or not available")

            response_parts.append(f"**Command**: `{entry.definition.command_template}`")

            if entry.definition.default_args:
                response_parts.append(f"**Default Args**: {' '.join(entry.definition.default_args)}")

            if entry.definition.required_env_vars:
                response_parts.append(f"**Required Env Vars**: {', '.join(entry.definition.required_env_vars)}")

            if include_capabilities and entry.definition.capabilities:
                response_parts.append("\n**Capabilities**:")
                for cap in entry.definition.capabilities:
                    response_parts.append(f"\n### {cap.name}")
                    response_parts.append(f"{cap.description}")

                    if cap.use_cases:
                        response_parts.append(f"**Use Cases**: {', '.join(cap.use_cases)}")
                    if cap.strengths:
                        response_parts.append(f"**Strengths**: {', '.join(cap.strengths)}")
                    if cap.limitations:
                        response_parts.append(f"**Limitations**: {', '.join(cap.limitations)}")

        response_parts.append("\n---")
        response_parts.append(
            "\n**Usage**: Use the `agent_sync`, `agent_async`, or `agent_batch` tools to delegate tasks to these agents."
        )

        return "\n".join(response_parts)
