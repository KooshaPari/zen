import os

import pytest

from tools.shared.agent_models import AgentTaskRequest, AgentType
from utils import agent_manager


@pytest.mark.asyncio
async def test_agent_server_startup_error_diagnostic(monkeypatch):
    # Force diagnostic mode
    monkeypatch.setenv("AGENTAPI_DIAGNOSTIC", "1")

    mgr = agent_manager.get_task_manager()

    # Make checks pass so we go into process startup
    # Patch the module-level check functions indirectly by patching subprocess path checks
    monkeypatch.setattr(agent_manager, "shutil", __import__("shutil"), raising=False)
    import shutil as _sh
    monkeypatch.setattr(_sh, "which", lambda cmd: "/bin/true")

    # Create a request with short timeout and bogus command arg to force early exit
    req = AgentTaskRequest(
        agent_type=AgentType.CLAUDE,
        task_description="diag test",
        message="hello",
        agent_args=["--bad-flag"],
        working_directory=os.getcwd(),
        timeout_seconds=1,
    )

    task = await mgr.create_task(req)

    # Monkeypatch subprocess.Popen to simulate immediate exit
    class DummyProc:
        def __init__(self):
            self._pid = 12345
        def poll(self):
            return 1
        def communicate(self):
            return ("out", "err: failed to start")
        @property
        def pid(self):
            return self._pid

    monkeypatch.setattr(agent_manager.subprocess, "Popen", lambda *a, **k: DummyProc())

    ok = await mgr.start_task(task.task_id)
    assert not ok


@pytest.mark.asyncio
async def test_agent_inbox_list_render(monkeypatch):
    # Sanity: listing should not crash with no tasks
    mgr = agent_manager.get_task_manager()
    # No explicit assertion here — just ensure no exceptions
    await mgr.cleanup_completed_tasks()

