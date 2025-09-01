"""
Default behaviors for agent orchestration (working directory, env, etc.).
"""
from __future__ import annotations

import os

from tools.shared.agent_models import AgentType
from utils.client_info import get_client_friendly_name


def get_default_working_directory(
    agent_type: AgentType,
    preferred_dir: str | None = None,
    session_id: str | None = None,
    parent_session_id: str | None = None,
) -> str:
    """Resolve a safe default working directory for agent tasks.

    Priority:
    1) preferred_dir if provided and non-empty
    2) ~/agents/sessions/<session_id> (or parent_session_id) when available
    3) ~/agents/<calling-client-friendly-name> if available
    4) ~/agents/<agent_type>
    """
    if preferred_dir:
        return preferred_dir

    home = os.path.expanduser("~")
    base = os.path.join(home, "agents")

    # Prefer a stable per-conversation directory when available
    sid = session_id or parent_session_id
    if sid:
        # Prefer parent thread directory if available so orchestrator+subagents share a workspace
        try:
            from utils.conversation_memory import get_thread  # local import to avoid heavy dependency at import time

            ctx = get_thread(str(sid))
            if ctx and getattr(ctx, "parent_thread_id", None):
                sid = ctx.parent_thread_id
        except Exception:
            # Best-effort; fall back to provided session id
            pass
        safe_sid = "".join(ch for ch in str(sid) if ch.isalnum() or ch in ("-", "_"))
        return os.path.join(base, "sessions", safe_sid)

    # Otherwise, try to use calling client's friendly name (e.g., "Claude", "Gemini")
    friendly = get_client_friendly_name()
    if friendly:
        sub = friendly.strip().lower().replace(" ", "-")
        return os.path.join(base, sub)

    # Fallback to agent type
    return os.path.join(base, agent_type.value)


def build_effective_path_env() -> str:
    """Build an effective PATH string honoring ZEN_AGENT_PATHS if provided.

    We prepend ZEN_AGENT_PATHS to PATH to prefer user-specified locations.
    """
    current = os.environ.get("PATH", "")
    extra = os.environ.get("ZEN_AGENT_PATHS", "")
    if not extra:
        return current
    # Prepend and de-duplicate order-preserving
    seen = set()
    parts = []
    for p in (extra.split(":") + current.split(":")):
        if p and p not in seen:
            seen.add(p)
            parts.append(p)
    return ":".join(parts)

