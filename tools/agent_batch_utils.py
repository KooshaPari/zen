"""
Utilities for AgentBatchTool to make testing easier.

Provides a helper to format the batch launch response so tests can assert
on the same content while AgentBatchTool focuses on orchestration.
"""
from __future__ import annotations

from tools.shared.agent_models import BatchTaskRequest


def _one_line_task_status(agent: str, task_id: str, description: str, batch_id: str | None = None) -> str:
    short_desc = description[:60] + ("..." if len(description) > 60 else "")
    suffix = f" | batch: {batch_id}" if batch_id else ""
    # At launch, tasks are newly started; show a consistent one-liner
    return f"- 🆔 {task_id} | {agent} | {short_desc} | 0s{suffix}"


def format_batch_launch_response(
    batch_id: str,
    batch_request: BatchTaskRequest,
    description: str,
    launched: list[tuple[str, str, str]] | None = None,
    color: bool = False,
) -> str:
    def colorize(s: str, code: str) -> str:
        return f"\x1b[{code}m{s}\x1b[0m" if color else s

    response_parts = [
        "# 🚀 Batch Agent Tasks Launched",
        f"\nBatch ID: `{batch_id}`",
        f"Strategy: {batch_request.coordination_strategy}",
        f"Tasks: {len(batch_request.tasks)}",
        f"Max Concurrent: {batch_request.max_concurrent}",
        f"Fail Fast: {'Yes' if batch_request.fail_fast else 'No'}",
        f"Timeout: {batch_request.timeout_seconds}s",
    ]

    if description:
        response_parts.append(f"**Description**: {description}")

    response_parts.append("\n## Task Breakdown")

    for i, request in enumerate(batch_request.tasks, 1):
        response_parts.append(f"\n### Task {i}: {request.agent_type.value}")
        response_parts.append(f"**Description**: {request.task_description}")
    # Header with batch id highlighted when color enabled
    if color:
        response_parts[1] = f"\nBatch ID: `{colorize(batch_id, '36')}`"

        if request.working_directory:
            response_parts.append(f"**Directory**: {request.working_directory}")
        if request.agent_args:
            response_parts.append(f"**Args**: {' '.join(request.agent_args)}")

    if launched:
        response_parts.append("\n## Batch Tracker")
        for task_id, agent, desc in launched:
            line = _one_line_task_status(agent, task_id, desc, batch_id)
            if color:
                # Highlight agent name in cyan and task id in yellow
                parts = line.split("|")
                if len(parts) >= 3:
                    parts[0] = parts[0]  # emoji + prefix stays
                    parts[1] = f" {colorize(parts[1].strip(), '33')} "  # task id
                    parts[2] = f" {colorize(parts[2].strip(), '36')} "  # agent
                    line = "|".join(parts)
            response_parts.append("\n" + line)

    response_parts.extend(
        [
            "\n## Monitoring",
            "",
            "Your batch is now running in the background. You can:",
            "",
            "1. **Check Progress**: Use `agent_inbox` with `action: list` to see all running tasks",
            "2. **Individual Status**: Use `agent_inbox` with specific task IDs to check individual progress",
            "3. **Continue Working**: All tasks run independently in the background",
            "",
            "## Coordination Strategy",
            "",
        ]
    )

    if batch_request.coordination_strategy == "parallel":
        response_parts.extend(
            [
                f"🔄 **Parallel Execution**: Up to {batch_request.max_concurrent} tasks running simultaneously",
                "- Tasks start immediately and run concurrently",
                "- Faster overall completion time",
                "- Resource usage distributed across tasks",
            ]
        )
    else:
        response_parts.extend(
            [
                "➡️ **Sequential Execution**: Tasks run one after another",
                "- Each task waits for the previous to complete",
                "- Predictable resource usage",
                "- Useful for dependent tasks",
            ]
        )

    if batch_request.fail_fast:
        response_parts.append("\n⚡ **Fail Fast**: If any task fails, remaining tasks will be cancelled")
    else:
        response_parts.append("\n🔄 **Continue on Error**: Tasks continue even if others fail")

    response_parts.extend(
        [
            "",
            "---",
            "",
            "💡 **Tip**: Use `agent_inbox` regularly to monitor progress and collect results as tasks complete.",
        ]
    )

    return "\n".join(response_parts)
