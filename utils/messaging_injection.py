"""
Messaging Injection (MVP)

Builds a compact injection string of top unread messages for an agent under a
fixed token budget. Summarization is naive for now (truncate lines). Later we
can call a fast LLM to summarize threads.
"""
from __future__ import annotations

# Rough heuristic: 4 chars per token
CHARS_PER_TOKEN = 4


def build_injection(agent_id: str, unread_items: list[dict], max_tokens: int = 512) -> str:
    if not unread_items:
        return ""
    max_chars = max_tokens * CHARS_PER_TOKEN
    header = f"You have {len(unread_items)} unread messages. Showing most recent:\n"
    out: list[str] = [header]
    budget = max_chars - len(header)
    for m in unread_items:
        prefix = "[DM]" if m.get("type") == "dm" else f"[Channel:{m.get('channel_id','?')}]"
        sender = m.get("from", "?")
        body = (m.get("body") or "").strip().replace("\n", " ")
        line = f"- {prefix} from {sender}: {body}\n"
        if len(line) <= budget:
            out.append(line)
            budget -= len(line)
        else:
            out.append(line[:max(0, budget)])
            break
    return "".join(out)

