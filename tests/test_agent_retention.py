import asyncio

import pytest

from tools.shared.agent_models import AgentTaskRequest, AgentType, TaskStatus
from utils import agent_manager


@pytest.mark.asyncio
async def test_completed_task_retention(monkeypatch):
    # Set short retention
    monkeypatch.setenv("AGENT_TASK_RETENTION_SEC", "1")

    mgr = agent_manager.get_task_manager()

    # Bypass external checks
    import shutil as _sh
    monkeypatch.setattr(agent_manager, "shutil", _sh, raising=False)
    monkeypatch.setattr(_sh, "which", lambda cmd: "/bin/true")

    # Create task and mark as completed directly
    req = AgentTaskRequest(
        agent_type=AgentType.CLAUDE,
        task_description="retention test",
        message="hello",
        timeout_seconds=1,
    )

    task = await mgr.create_task(req)
    # Simulate completed status
    t = await mgr.get_task(task.task_id)
    t.status = TaskStatus.COMPLETED

    # Immediately cleanup should retain (age ~0)
    await mgr.cleanup_completed_tasks()
    assert task.task_id in mgr.active_tasks

    # Wait past retention
    await asyncio.sleep(1.1)
    await mgr.cleanup_completed_tasks()
    assert task.task_id not in mgr.active_tasks

