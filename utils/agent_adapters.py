"""
In-process agent adapters.

This module defines a minimal adapter registry that maps AgentType to a
callable capable of running the underlying CLI in a non-interactive way,
using the CLI's own local auth/config.

We intentionally avoid any external AgentAPI microservice.
"""
from __future__ import annotations

import os
import subprocess
from typing import Callable

from tools.shared.agent_models import AgentTaskRequest, AgentType
from utils.agent_defaults import build_effective_path_env

# Import moved to avoid circular import

# Type for adapter runner
AdapterRunner = Callable[[AgentTaskRequest], tuple[str, int, str]]


def _run_with_args(full_cmd: list[str], request: AgentTaskRequest, stdin_text: str | None = None) -> tuple[str, int, str]:
    """Helper to spawn a process with args and optional stdin text."""
    env = os.environ.copy()
    env["PATH"] = build_effective_path_env()
    cwd = request.working_directory or os.getcwd()
    try:
        os.makedirs(cwd, exist_ok=True)
    except Exception:
        pass
    try:
        proc = subprocess.Popen(
            full_cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE if stdin_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(input=stdin_text, timeout=request.timeout_seconds or 300)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            return stdout or "", 124, stderr or "timeout"
        return stdout or "", int(proc.returncode or 0), stderr or ""
    except FileNotFoundError as e:
        return "", 127, f"CLI not found: {e}"
    except Exception as e:
        return "", 1, f"Adapter error: {type(e).__name__}: {e}"


def _generic_cli_runner(request: AgentTaskRequest) -> tuple[str, int, str]:
    """Run the agent CLI generically by piping the message to stdin."""
    cmd = _get_agent_command(request.agent_type)
    full_cmd = [cmd] + (request.agent_args or [])
    return _run_with_args(full_cmd, request, stdin_text=request.message)


def _get_agent_command(agent_type: AgentType) -> str:
    """Get the CLI command for an agent type."""
    mapping = {
        AgentType.CLAUDE: "claude",
        AgentType.GOOSE: "goose",
        AgentType.AIDER: "aider",
        AgentType.CODEX: "codex",
        AgentType.GEMINI: "gemini",
        AgentType.AMP: "amp",
        AgentType.CURSOR_AGENT: "cursor-agent",
        AgentType.CURSOR: "cursor",
        AgentType.AUGGIE: "auggie",
        AgentType.CRUSH: "crush",
    }
    return mapping.get(agent_type, agent_type.value)


def _claude_runner(request: AgentTaskRequest) -> tuple[str, int, str]:
    """Configure Claude for autonomous file creation and editing."""
    args = ["--allowedTools", "Write Edit Read Bash(git*)", "--output-format", "text"] + (request.agent_args or [])
    if request.message:
        return _run_with_args([_get_agent_command(request.agent_type)] + args, request, stdin_text=request.message)
    return _run_with_args([_get_agent_command(request.agent_type)] + args + [request.message], request)


def _gemini_runner(request: AgentTaskRequest) -> tuple[str, int, str]:
    """Use Gemini's -p/--prompt for non-interactive execution."""
    cmd = _get_agent_command(request.agent_type)
    if request.message:
        args = ["-p", request.message] + (request.agent_args or [])
        return _run_with_args([cmd] + args, request)
    return _generic_cli_runner(request)


def _auggie_runner(request: AgentTaskRequest) -> tuple[str, int, str]:
    """Special handler for Auggie using --print mode with full capabilities."""
    cmd = _get_agent_command(request.agent_type)
    args = ["--print"]

    if request.working_directory:
        args.extend(["-w", request.working_directory])

    if request.agent_args:
        args.extend(request.agent_args)

    if request.message:
        return _run_with_args([cmd] + args, request, stdin_text=request.message)
    return _run_with_args([cmd] + args, request)


def _cursor_runner(request: AgentTaskRequest) -> tuple[str, int, str]:
    """Cursor runner using stdin mode."""
    args = ["-"] + (request.agent_args or []) if request.message else (request.agent_args or [])
    return _run_with_args([_get_agent_command(request.agent_type)] + args, request, stdin_text=request.message)


def _codex_runner(request: AgentTaskRequest) -> tuple[str, int, str]:
    """Use 'codex exec' for non-interactive execution."""
    cmd = _get_agent_command(request.agent_type)
    if request.message:
        full_cmd = [cmd, "exec", request.message]
        return _run_with_args(full_cmd, request)
    return _generic_cli_runner(request)


# Adapter registry mapping agent types to runners
ADAPTERS: dict[AgentType, AdapterRunner] = {
    AgentType.CLAUDE: _claude_runner,
    AgentType.GOOSE: _generic_cli_runner,
    AgentType.AIDER: _generic_cli_runner,
    AgentType.CODEX: _codex_runner,
    AgentType.GEMINI: _gemini_runner,
    AgentType.AMP: _generic_cli_runner,
    AgentType.CURSOR_AGENT: _generic_cli_runner,
    AgentType.CURSOR: _cursor_runner,
    AgentType.AUGGIE: _auggie_runner,
    AgentType.CRUSH: _generic_cli_runner,
}


def run_adapter(request: AgentTaskRequest) -> tuple[str, int, str]:
    """Run an agent adapter with enhanced communication protocol."""
    # Enhance the message with structured communication protocol
    if request.message:
        # Import here to avoid circular import
        from utils.agent_prompts import enhance_agent_message
        enhanced_request = request.model_copy(update={
            "message": enhance_agent_message(request.message, request.agent_type)
        })
    else:
        enhanced_request = request

    runner = ADAPTERS.get(enhanced_request.agent_type, _generic_cli_runner)
    return runner(enhanced_request)
