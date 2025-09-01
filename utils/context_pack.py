"""
Context Pack Builder (MVP)

Assembles a compact context blob for an agent/task using:
- Project graph (project, agents, channels)
- Recent artifacts
- Unread messaging summaries

Budgeting uses a rough tokens→chars heuristic. This is a best-effort packer.
Feature flag: ZEN_CONTEXT_PACK=1 enables packing during LLM task creation.
"""
from __future__ import annotations

from typing import Any

CHARS_PER_TOKEN = 4


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def build_context_pack(
    *,
    project_id: str | None,
    agent_id: str | None,
    token_budget: int = 1024,
) -> dict[str, Any]:
    from utils.messaging_store import get_unread
    from utils.project_store import get_project, list_agents, list_artifacts

    max_chars = token_budget * CHARS_PER_TOKEN
    budget = max_chars

    ctx: dict[str, Any] = {"version": 1, "sections": []}

    # Project section
    if project_id:
        proj = get_project(project_id)
        if proj:
            section = {"type": "project", "id": project_id, "name": proj.get("name"), "owner": proj.get("owner")}
            ctx["sections"].append(section)
            budget -= len(str(section))

            # Agents list (truncated)
            agents = list_agents(project_id)
            agents_line = f"Agents: {', '.join(agents[:20])}"
            agents_line = _truncate(agents_line, min(budget, 800))
            ctx["sections"].append({"type": "agents", "text": agents_line})
            budget -= len(agents_line)

            # Recent artifacts (truncated)
            arts = list_artifacts(project_id, limit=10)
            lines: list[str] = []
            for a in arts:
                name = a.get("name") or a.get("id", "artifact")
                kind = a.get("type") or "artifact"
                lines.append(f"- {kind}: {name}")
                if sum(map(len, lines)) > 800:
                    break
            arts_text = _truncate("\n".join(lines), min(800, budget)) if lines else ""
            if arts_text:
                ctx["sections"].append({"type": "artifacts", "text": arts_text})
                budget -= len(arts_text)

    # Unread messages for the agent (summary lines)
    if agent_id and budget > 100:
        items = get_unread(agent_id, limit=10)
        if items:
            lines: list[str] = []
            for m in items:
                prefix = "[DM]" if m.get("type") == "dm" else f"[Ch:{m.get('channel_id','?')}]"
                sender = m.get("from", "?")
                body = (m.get("body") or "").strip().replace("\n", " ")
                line = f"- {prefix} {sender}: {body}"
                lines.append(line)
                if sum(map(len, lines)) > 1000:
                    break
            msg_text = _truncate("\n".join(lines), min(1000, budget))
            ctx["sections"].append({"type": "unread", "text": msg_text})
            budget -= len(msg_text)

    ctx["budget_left_chars"] = max(0, budget)
    return ctx

