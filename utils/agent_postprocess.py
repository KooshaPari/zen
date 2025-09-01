"""
Post-processing utilities for agent CLI outputs.

Produces a compressed action trail and a final message from raw stdout/stderr.
We avoid external LLM calls; use lightweight heuristics per agent.
"""
from __future__ import annotations

import re
from typing import Any

from tools.shared.agent_models import AgentType
from utils.agent_prompts import format_agent_summary, parse_agent_output

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Known noisy prefixes to strip
NOISE_PREFIXES = [
    "Successfully migrated settings",
    "Legacy settings file detected",
    "Auggie — Augment CLI Agent.",
]

# Known noisy substrings to strip anywhere in the line
NOISE_CONTAINS = [
    "raw mode is not supported",
    "attempting to use raw mode",
    "ink warning",
    "react-stack-bottom",
    "augment.mjs",
]

ACTION_PATTERNS = [
    r"^(?:Action|Tool|Running|Execute|Command|Shell|Apply|Edit|Create|Update|Delete)[:\s]",
    r"^(?:git|npm|pnpm|bun|yarn|python|node|deno)\b",
    r"^\s*->\s*",
    r"^\s*[•\-]\s+",
]

ACTION_RE = re.compile("|".join(ACTION_PATTERNS), re.IGNORECASE)


def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def _drop_noise(lines: list[str]) -> list[str]:
    clean: list[str] = []
    for ln in lines:
        if any(ln.startswith(pfx) for pfx in NOISE_PREFIXES):
            continue
        low = ln.lower()
        if any(s in low for s in NOISE_CONTAINS):
            continue
        clean.append(ln)
    return clean


def _split_blocks(text: str) -> list[str]:
    # Split on blank lines into paragraphs
    blocks: list[str] = []
    cur: list[str] = []
    for ln in text.splitlines():
        if ln.strip() == "":
            if cur:
                blocks.append("\n".join(cur).strip())
                cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append("\n".join(cur).strip())
    return blocks


def _estimate_tokens(s: str) -> int:
    # Rough heuristic: ~4 chars per token
    return max(0, int(len(s) / 4))


def _choose_final_block(blocks: list[str]) -> str:
    if not blocks:
        return ""

    # Check if we have a multi-block code response (common with Claude)
    # Look for blocks that start with ``` or contain class/function definitions
    code_blocks = []
    for i, b in enumerate(blocks):
        if ("```" in b or "class " in b or "def " in b or
            b.strip().startswith("class ") or b.strip().startswith("def ")):
            code_blocks.append((i, b))

    # If we have multiple code blocks, combine them to form the complete response
    if len(code_blocks) >= 2:
        # Find the range of code blocks
        start_idx = code_blocks[0][0]
        end_idx = code_blocks[-1][0]

        # Combine all blocks from the first code block to the last
        combined = "\n\n".join(blocks[start_idx:end_idx+1])

        # If the combined result is substantial, return it
        if len(combined.strip()) > 200:
            return combined

    # Fallback to original logic: look at recent blocks
    candidates = list(reversed(blocks[-5:]))  # look at up to last 5 blocks
    for b in candidates:
        if "```" in b or "function " in b or "class " in b or "def " in b:
            return b
    # else pick the longest recent block
    return max(candidates, key=lambda x: len(x)) if candidates else blocks[-1]


def _file_change_counts(lines: list[str]) -> dict[str, int]:
    add = mod = delete = diffplus = diffminus = 0
    for ln in lines:
        s = ln.strip()
        low = s.lower()
        if low.startswith(("created", "added")):
            add += 1
        elif low.startswith(("modified", "updated", "changed")):
            mod += 1
        elif low.startswith(("deleted", "removed")):
            delete += 1
        if s.startswith("+++"):
            diffplus += 1
        if s.startswith("---"):
            diffminus += 1
    return {"added": add, "modified": mod, "deleted": delete, "diff_files_plus": diffplus, "diff_files_minus": diffminus}


def _extract_files(lines: list[str]) -> list[str]:
    files: list[str] = []
    for ln in lines:
        s = ln.strip()
        # diff --git a/path b/path
        if s.startswith("diff --git ") and " a/" in s and " b/" in s:
            try:
                parts = s.split()
                parts.index("a/") if "a/" in parts else -1
            except Exception:
                pass
            # Fallback: simple extraction
            try:
                a_part = s.split(" a/")[-1]
                a_path = a_part.split(" ")[0]
                if a_path:
                    files.append(a_path)
            except Exception:
                pass
        if s.startswith("+++") and ("/" in s or "\\" in s):
            p = s.replace("+++ b/", "").replace("+++ ", "").strip()
            files.append(p)
        elif s.startswith("---") and ("/" in s or "\\" in s):
            p = s.replace("--- a/", "").replace("--- ", "").strip()
            files.append(p)
        else:
            low = s.lower()
            if low.startswith(("created", "added", "modified", "updated", "changed", "deleted", "removed", "rename to")):
                # extract last token if it's a path-like
                parts = s.split()
                if parts:
                    cand = parts[-1]
                    if "/" in cand or "." in cand:
                        files.append(cand)
            # git porcelain formats
            if s.startswith("?? "):
                files.append(s[3:].strip())
            elif s.startswith("R ") and "->" in s:
                try:
                    right = s.split("->", 1)[1].strip()
                    files.append(right)
                except Exception:
                    pass
    # dedupe
    seen = set()
    out: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out[:20]


def extract_actions_and_final(agent: AgentType, stdout: str, stderr: str, input_message: str | None = None) -> tuple[list[str], str, dict[str, Any]]:
    """Extract a compressed action trail, a final message, and basic metrics.

    NEW: First tries to parse structured XML tags from agents, falls back to heuristics.
    - Removes ANSI codes and known noise lines
    - Picks lines that look like actions/steps
    - Final message: best recent substantial block; fallback to stderr tail
    - Metrics: actions_count, file change counts, stdout/stderr sizes, token estimates
    """
    out = _strip_ansi(stdout or "").strip()
    err = _strip_ansi(stderr or "").strip()

    # Try structured parsing first
    parsed_response = parse_agent_output(out)
    if parsed_response.summary or parsed_response.actions:
        # We got structured output - use it!
        actions = parsed_response.actions or []
        final_msg = format_agent_summary(out)

        # Enhanced metrics from structured response
        files_list = parsed_response.files_created + parsed_response.files_modified
        files_total = len(parsed_response.files_created) + len(parsed_response.files_modified)
        files = {
            "added": len(parsed_response.files_created),
            "modified": len(parsed_response.files_modified),
            "deleted": 0,
            "diff_files_plus": 0,
            "diff_files_minus": 0
        }

        meta: dict[str, Any] = {
            "actions_count": len(actions),
            "steps_est": len(actions),
            "tool_calls_est": len(actions),
            "stdout_chars": len(out),
            "stderr_chars": len(err),
            "files": files,
            "files_changed_total": files_total,
            "files_touched": files_list,
            "structured_response": True,
            "agent_status": parsed_response.status,
            "has_questions": len(parsed_response.questions) > 0
        }

        if input_message is not None:
            meta["tokens_in_est"] = _estimate_tokens(input_message)
        if final_msg:
            meta["tokens_out_est"] = _estimate_tokens(final_msg)
            if "tokens_in_est" in meta:
                meta["tokens_delta_est"] = meta["tokens_out_est"] - meta["tokens_in_est"]

        return actions, final_msg, meta

    # Fallback to original heuristic parsing
    out_lines = _drop_noise(out.splitlines())

    # Action lines heuristic
    action_lines: list[str] = []
    for ln in out_lines:
        if ACTION_RE.search(ln):
            action_lines.append(ln.strip())
        # Also capture JSON step markers if present
        elif ln.strip().startswith("{") and '"action"' in ln:
            action_lines.append(ln.strip())
    # De-duplicate while preserving order
    seen = set()
    actions: list[str] = []
    for ln in action_lines:
        if ln not in seen:
            seen.add(ln)
            actions.append(ln)
    # Limit length
    if len(actions) > 50:
        actions = actions[-50:]

    # Final message selection
    final_msg = ""
    blocks = _split_blocks("\n".join(out_lines))
    if blocks:
        chosen_block = _choose_final_block(blocks)
        # If chosen block is too short or doesn't seem like a complete response,
        # return empty final_msg to fall back to full stdout
        if len(chosen_block.strip()) < 100:
            final_msg = ""  # Fall back to full stdout
        else:
            final_msg = chosen_block
    elif err:
        err_lines = err.splitlines()
        final_msg = "\n".join(err_lines[-20:])

    # Metrics
    files = _file_change_counts(out_lines)
    files_list = _extract_files(out_lines)
    files_total = files.get("added", 0) + files.get("modified", 0) + files.get("deleted", 0)
    meta: dict[str, Any] = {
        "actions_count": len(actions),
        "steps_est": len(actions),
        "tool_calls_est": len(actions),
        "stdout_chars": len(out),
        "stderr_chars": len(err),
        "files": files,
        "files_changed_total": files_total,
        "files_touched": files_list,
        "structured_response": False
    }
    if input_message is not None:
        meta["tokens_in_est"] = _estimate_tokens(input_message)
    if final_msg:
        meta["tokens_out_est"] = _estimate_tokens(final_msg)
        if "tokens_in_est" in meta:
            meta["tokens_delta_est"] = meta["tokens_out_est"] - meta["tokens_in_est"]

    return actions, (final_msg or ""), meta
