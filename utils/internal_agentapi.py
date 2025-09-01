"""
Internal AgentAPI facade - in-process implementation.

Replaces the external agentapi microservice. Provides the same surface used by
our tools (agent_sync/agent_async/agent_batch/agent_inbox), backed by
AgentTaskManager and in-process agent adapters.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from tools.shared.agent_models import (
    AgentTask,
    AgentTaskRequest,
    AgentTaskResult,
    AgentType,
    TaskStatus,
)
from utils.agent_adapters import run_adapter
from utils.agent_manager import AgentTaskManager, get_task_manager


class InternalAgentAPI:
    def __init__(self, manager: AgentTaskManager | None = None) -> None:
        self.manager = manager or get_task_manager()

    async def start_existing_task(self, task: AgentTask) -> AgentTask:
        """Start running an already-created task in the background."""
        async def _runner(t: AgentTask) -> None:
            try:
                t.status = TaskStatus.RUNNING
                await self._store(t)
                loop = asyncio.get_event_loop()
                stdout, code, stderr = await loop.run_in_executor(None, lambda: run_adapter(t.request))
                from utils.agent_postprocess import extract_actions_and_final
                actions, final_msg, meta = extract_actions_and_final(t.request.agent_type, stdout, stderr, t.request.message)

                t.status = TaskStatus.COMPLETED if code == 0 else TaskStatus.FAILED
                completed_at = datetime.now(timezone.utc)
                duration = (completed_at - t.created_at).total_seconds()
                t.result = AgentTaskResult(
                    task_id=t.task_id,
                    agent_type=t.request.agent_type,
                    status=t.status,
                    started_at=t.created_at,
                    completed_at=completed_at,
                    duration_seconds=duration,
                    output=final_msg or stdout,
                    error=None if code == 0 else (stderr or f"exit code {code}"),
                )
                try:
                    from tools.shared.agent_models import Message
                    t.result.messages = [Message(role="system", content="action: " + a) for a in actions]
                    if meta:
                        t.result.messages.insert(0, Message(role="system", content=f"metrics: {meta}"))
                    # Add batch id(s) for cross-linking
                    try:
                        from utils.batch_registry import find_batches_for_task
                        for bid in find_batches_for_task(t.task_id):
                            t.result.messages.insert(0, Message(role="system", content=f"batch: {bid}"))
                    except Exception:
                        pass
                    # Add follow-up guidance
                    if (len(final_msg) < 200) or (t.status != TaskStatus.COMPLETED):
                        t.result.messages.append(
                            Message(
                                role="system",
                                content="followup: Provide additional details, files, or next steps to continue."
                            )
                        )
                except Exception:
                    pass
            except Exception as e:
                t.status = TaskStatus.FAILED
                t.result = AgentTaskResult(
                    task_id=t.task_id,
                    agent_type=t.request.agent_type,
                    status=t.status,
                    started_at=t.created_at,
                    completed_at=datetime.now(timezone.utc),
                    output="",
                    error=str(e),
                )
            finally:
                await self._store(t)

        task.status = TaskStatus.STARTING
        await self._store(task)
        asyncio.create_task(_runner(task))
        return task

    async def agent_sync(self, request: AgentTaskRequest) -> AgentTaskResult:
        # Special-case Auggie: run via async path and wait, to avoid raw-mode UI issues
        if request.agent_type == AgentType.AUGGIE:
            t = await self.agent_async(request)
            # Wait with polling
            deadline = (request.timeout_seconds or 300)
            start = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start < deadline:
                current = await self.manager.get_task(t.task_id)
                if current and current.result and current.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT}:
                    return current.result
                await asyncio.sleep(1)
            # Timeout fallback
            current = await self.manager.get_task(t.task_id)
            if current and current.result:
                return current.result
            # synthesize timeout result
            return AgentTaskResult(
                task_id=t.task_id,
                agent_type=request.agent_type,
                status=TaskStatus.TIMEOUT,
                started_at=current.created_at if current else datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                duration_seconds=deadline,
                output="",
                error=f"Task timed out after {deadline}s",
            )

        # Create ephemeral task (no external server)
        task = await self.manager.create_task(request)
        task.status = TaskStatus.STARTING
        await self._store(task)

        # Run adapter synchronously
        task.status = TaskStatus.RUNNING
        await self._store(task)
        stdout, code, stderr = await asyncio.get_event_loop().run_in_executor(None, lambda: run_adapter(request))

        # Complete with post-processing
        from utils.agent_postprocess import extract_actions_and_final
        actions, final_msg, meta = extract_actions_and_final(request.agent_type, stdout, stderr, request.message)

        task.status = TaskStatus.COMPLETED if code == 0 else TaskStatus.FAILED
        completed_at = datetime.now(timezone.utc)
        duration = (completed_at - task.created_at).total_seconds()
        result = AgentTaskResult(
            task_id=task.task_id,
            agent_type=request.agent_type,
            status=task.status,
            started_at=task.created_at,
            completed_at=completed_at,
            duration_seconds=duration,
            output=final_msg or stdout,
            error=None if code == 0 else (stderr or f"exit code {code}"),
        )
        # Attach compressed action trail as messages for the consumer
        try:
            from tools.shared.agent_models import Message
            # Encode action trail and key metrics
            result.messages = [Message(role="system", content="action: " + a) for a in actions]
            if meta:
                result.messages.insert(0, Message(role="system", content=f"metrics: {meta}"))
            # Add batch id(s) for cross-linking
            try:
                from utils.batch_registry import find_batches_for_task
                for bid in find_batches_for_task(task.task_id):
                    result.messages.insert(0, Message(role="system", content=f"batch: {bid}"))
            except Exception:
                pass
        except Exception:
            pass
        task.result = result
        await self._store(task)
        return result

    async def agent_async(self, request: AgentTaskRequest) -> AgentTask:
        task = await self.manager.create_task(request)
        task.status = TaskStatus.STARTING
        await self._store(task)

        async def _runner(t: AgentTask) -> None:
            try:
                t.status = TaskStatus.RUNNING
                await self._store(t)
                loop = asyncio.get_event_loop()
                stdout, code, stderr = await loop.run_in_executor(None, lambda: run_adapter(t.request))
                from utils.agent_postprocess import extract_actions_and_final
                actions, final_msg, meta = extract_actions_and_final(t.request.agent_type, stdout, stderr, t.request.message)

                t.status = TaskStatus.COMPLETED if code == 0 else TaskStatus.FAILED
                completed_at = datetime.now(timezone.utc)
                duration = (completed_at - t.created_at).total_seconds()
                t.result = AgentTaskResult(
                    task_id=t.task_id,
                    agent_type=t.request.agent_type,
                    status=t.status,
                    started_at=t.created_at,
                    completed_at=completed_at,
                    duration_seconds=duration,
                    output=final_msg or stdout,
                    error=None if code == 0 else (stderr or f"exit code {code}"),
                )
                try:
                    from tools.shared.agent_models import Message
                    t.result.messages = [Message(role="system", content="action: " + a) for a in actions]
                    if meta:
                        t.result.messages.insert(0, Message(role="system", content=f"metrics: {meta}"))
                except Exception:
                    pass
            except Exception as e:
                t.status = TaskStatus.FAILED
                t.result = AgentTaskResult(
                    task_id=t.task_id,
                    agent_type=t.request.agent_type,
                    status=t.status,
                    started_at=t.created_at,
                    completed_at=datetime.now(timezone.utc),
                    output="",
                    error=str(e),
                    exit_code=1,
                )
            finally:
                await self._store(t)

        asyncio.create_task(_runner(task))
        return task

    async def agent_inbox(self, action: str, **kwargs):
        # The existing tools already go through AgentTaskManager; reuse it
        if action == "list":
            total, items = await self.manager.list_tasks()
            return {"total": total, "items": [i.model_dump() for i in items]}
        elif action == "status":
            tid = kwargs.get("task_id")
            task = await self.manager.get_task(tid) if tid else None
            return task.model_dump() if task else None
        elif action == "results":
            tid = kwargs.get("task_id")
            task = await self.manager.get_task(tid) if tid else None
            return task.result.model_dump() if task and task.result else None
        elif action == "cancel":
            # Best-effort cancel not implemented for simple adapters
            return {"ok": False, "reason": "cancel not supported in in-process adapter yet"}
        else:
            return {"error": f"unknown action {action}"}

    async def agent_batch(self, tasks: list[AgentTaskRequest]) -> list[AgentTaskResult]:
        # Simple sequential implementation with per-task timeouts respected by adapters
        results: list[AgentTaskResult] = []
        for req in tasks:
            results.append(await self.agent_sync(req))
        return results

    async def _store(self, task: AgentTask) -> None:
        await self.manager._store_task(task)

