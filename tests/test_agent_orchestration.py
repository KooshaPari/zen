"""
Tests for Agent Orchestration Tools

This module contains comprehensive tests for all agent orchestration tools
including unit tests and integration tests with AgentAPI.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.agent_async import AgentAsyncTool
from tools.agent_batch import AgentBatchTool
from tools.agent_inbox import AgentInboxTool
from tools.agent_registry import AgentRegistryTool
from tools.agent_sync import AgentSyncTool
from tools.shared.agent_models import (
    AgentTask,
    AgentTaskRequest,
    AgentTaskResult,
    AgentType,
    TaskStatus,
)
from utils.agent_manager import AgentTaskManager


class TestAgentRegistryTool:
    """Test cases for AgentRegistryTool."""

    @pytest.fixture
    def registry_tool(self):
        return AgentRegistryTool()

    @pytest.mark.asyncio
    async def test_list_all_agents(self, registry_tool):
        """Test listing all available agents."""
        result = await registry_tool.execute({})

        assert len(result) == 1
        response_text = result[0].text
        assert "Available Agent Registry" in response_text
        assert "claude" in response_text.lower()
        assert "aider" in response_text.lower()
        assert "goose" in response_text.lower()

    @pytest.mark.asyncio
    async def test_query_specific_agent(self, registry_tool):
        """Test querying a specific agent type."""
        result = await registry_tool.execute({"agent_type": "claude"})

        assert len(result) == 1
        response_text = result[0].text
        assert "Claude Code" in response_text
        assert "Code Generation" in response_text
        assert "Code Analysis" in response_text

    @pytest.mark.asyncio
    async def test_check_availability_disabled(self, registry_tool):
        """Test with availability checking disabled."""
        result = await registry_tool.execute({"check_availability": False, "include_capabilities": False})

        assert len(result) == 1
        response_text = result[0].text
        assert "Available Agent Registry" in response_text
        # Should not contain installation paths when availability check is disabled
        assert "Installation:" not in response_text

    def test_requires_model(self, registry_tool):
        """Test that registry tool doesn't require AI model."""
        assert not registry_tool.requires_model()


class TestAgentSyncTool:
    """Test cases for AgentSyncTool."""

    @pytest.fixture
    def sync_tool(self):
        return AgentSyncTool()

    @pytest.fixture
    def mock_task_manager(self):
        manager = MagicMock(spec=AgentTaskManager)
        manager.create_task = AsyncMock()
        manager.start_task = AsyncMock()
        manager.cleanup_completed_tasks = AsyncMock()
        return manager

    @pytest.mark.asyncio
    async def test_successful_sync_execution(self, sync_tool, mock_task_manager):
        """Test successful synchronous task execution."""
        # Mock task creation and execution
        mock_task = AgentTask(
            task_id="test-task-123",
            request=AgentTaskRequest(agent_type=AgentType.CLAUDE, task_description="Test task", message="Hello agent"),
            status=TaskStatus.PENDING,
            agent_port=3284,
        )

        mock_task_manager.create_task.return_value = mock_task
        mock_task_manager.start_task.return_value = True

        with (
            patch("utils.agent_manager.get_task_manager", return_value=mock_task_manager),
            patch.object(sync_tool, "_wait_for_completion") as mock_wait,
        ):

            mock_result = AgentTaskResult(
                task_id="test-task-123",
                agent_type=AgentType.CLAUDE,
                status=TaskStatus.COMPLETED,
                messages=[{"role": "agent", "message": "Task completed successfully"}],
                output="Task completed successfully",
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                duration_seconds=10.5,
            )
            mock_wait.return_value = mock_result

            result = await sync_tool.execute(
                {"agent_type": "claude", "task_description": "Test task", "message": "Hello agent"}
            )

            assert len(result) == 1
            response_text = result[0].text
            assert "✅ Agent Task Result" in response_text
            assert "claude" in response_text
            assert "COMPLETED" in response_text
            assert "Task completed successfully" in response_text

    @pytest.mark.asyncio
    async def test_task_creation_failure(self, sync_tool, mock_task_manager):
        """Test handling of task creation failure."""
        mock_task = AgentTask(
            task_id="failed-task",
            request=AgentTaskRequest(agent_type=AgentType.CLAUDE, task_description="Test task", message="Hello agent"),
            status=TaskStatus.FAILED,
        )
        mock_task.result = AgentTaskResult(
            task_id="failed-task",
            agent_type=AgentType.CLAUDE,
            status=TaskStatus.FAILED,
            error="Failed to allocate port",
            started_at=datetime.now(timezone.utc),
        )

        mock_task_manager.create_task.return_value = mock_task

        with patch("utils.agent_manager.get_task_manager", return_value=mock_task_manager):
            result = await sync_tool.execute(
                {"agent_type": "claude", "task_description": "Test task", "message": "Hello agent"}
            )

            assert len(result) == 1
            response_text = result[0].text
            assert "❌ Agent Task Result" in response_text
            assert "FAILED" in response_text
            assert "Failed to allocate port" in response_text

    def test_input_schema_validation(self, sync_tool):
        """Test input schema validation."""
        schema = sync_tool.get_input_schema()

        assert "agent_type" in schema["properties"]
        assert "task_description" in schema["properties"]
        assert "message" in schema["properties"]
        assert schema["properties"]["agent_type"]["enum"] == [agent.value for agent in AgentType]
        assert "timeout_seconds" in schema["properties"]
        assert schema["properties"]["timeout_seconds"]["default"] == 300


class TestAgentAsyncTool:
    """Test cases for AgentAsyncTool."""

    @pytest.fixture
    def async_tool(self):
        return AgentAsyncTool()

    @pytest.fixture
    def mock_task_manager(self):
        manager = MagicMock(spec=AgentTaskManager)
        manager.create_task = AsyncMock()
        return manager

    @pytest.mark.asyncio
    async def test_successful_async_launch(self, async_tool, mock_task_manager):
        """Test successful asynchronous task launch."""
        mock_task = AgentTask(
            task_id="async-task-456",
            request=AgentTaskRequest(
                agent_type=AgentType.AIDER, task_description="Refactor code", message="Please refactor this code"
            ),
            status=TaskStatus.PENDING,
            agent_port=3285,
        )

        mock_task_manager.create_task.return_value = mock_task

        with (
            patch("utils.agent_manager.get_task_manager", return_value=mock_task_manager),
            patch("asyncio.create_task") as mock_create_task,
        ):

            result = await async_tool.execute(
                {
                    "agent_type": "aider",
                    "task_description": "Refactor code",
                    "message": "Please refactor this code",
                    "priority": "high",
                }
            )

            assert len(result) == 1
            response_text = result[0].text
            assert "🚀 Agent Task Launched" in response_text
            assert "async-task-456" in response_text
            assert "aider" in response_text
            assert "Refactor code" in response_text

            # Verify background task was created
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_launch_with_custom_timeout(self, async_tool, mock_task_manager):
        """Test async launch with custom timeout."""
        mock_task = AgentTask(
            task_id="timeout-task",
            request=AgentTaskRequest(
                agent_type=AgentType.GOOSE,
                task_description="Long running task",
                message="This will take a while",
                timeout_seconds=3600,
            ),
            status=TaskStatus.PENDING,
        )

        mock_task_manager.create_task.return_value = mock_task

        with (
            patch("utils.agent_manager.get_task_manager", return_value=mock_task_manager),
            patch("asyncio.create_task"),
        ):

            result = await async_tool.execute(
                {
                    "agent_type": "goose",
                    "task_description": "Long running task",
                    "message": "This will take a while",
                    "timeout_seconds": 3600,
                }
            )

            assert len(result) == 1
            response_text = result[0].text
            assert "3600s" in response_text


class TestAgentInboxTool:
    """Test cases for AgentInboxTool."""

    @pytest.fixture
    def inbox_tool(self):
        return AgentInboxTool()

    @pytest.fixture
    def mock_task_manager(self):
        manager = MagicMock(spec=AgentTaskManager)
        manager.get_task = AsyncMock()
        manager.active_tasks = {}
        return manager

    @pytest.mark.asyncio
    async def test_get_task_status(self, inbox_tool, mock_task_manager):
        """Test getting task status."""
        mock_task = AgentTask(
            task_id="status-task",
            request=AgentTaskRequest(
                agent_type=AgentType.CLAUDE, task_description="Status check", message="Check my status"
            ),
            status=TaskStatus.RUNNING,
            agent_port=3286,
        )

        mock_task_manager.get_task.return_value = mock_task

        with (
            patch("utils.agent_manager.get_task_manager", return_value=mock_task_manager),
            patch.object(inbox_tool, "_get_live_agent_status", return_value="running"),
        ):

            result = await inbox_tool.execute({"task_id": "status-task", "action": "status"})

            assert len(result) == 1
            response_text = result[0].text
            assert "⚡ Task Status: status-task" in response_text
            assert "RUNNING" in response_text
            assert "claude" in response_text

    @pytest.mark.asyncio
    async def test_get_task_results(self, inbox_tool, mock_task_manager):
        """Test getting complete task results."""
        mock_task = AgentTask(
            task_id="results-task",
            request=AgentTaskRequest(
                agent_type=AgentType.AIDER, task_description="Get results", message="Show me results"
            ),
            status=TaskStatus.COMPLETED,
        )
        mock_task.result = AgentTaskResult(
            task_id="results-task",
            agent_type=AgentType.AIDER,
            status=TaskStatus.COMPLETED,
            messages=[
                {"role": "user", "message": "Show me results"},
                {"role": "agent", "message": "Here are your results"},
            ],
            output="Here are your results",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            duration_seconds=25.3,
        )

        mock_task_manager.get_task.return_value = mock_task

        with patch("utils.agent_manager.get_task_manager", return_value=mock_task_manager):
            result = await inbox_tool.execute({"task_id": "results-task", "action": "results"})

            assert len(result) == 1
            response_text = result[0].text
            assert "✅ Task Results: results-task" in response_text
            assert "COMPLETED" in response_text
            assert "Here are your results" in response_text
            assert "25.3s" in response_text

    @pytest.mark.asyncio
    async def test_list_tasks(self, inbox_tool, mock_task_manager):
        """Test listing all tasks."""
        # Create mock tasks
        running_task = AgentTask(
            task_id="running-1",
            request=AgentTaskRequest(
                agent_type=AgentType.CLAUDE, task_description="Running task", message="Still working"
            ),
            status=TaskStatus.RUNNING,
        )

        completed_task = AgentTask(
            task_id="completed-1",
            request=AgentTaskRequest(agent_type=AgentType.AIDER, task_description="Completed task", message="All done"),
            status=TaskStatus.COMPLETED,
        )

        mock_task_manager.active_tasks = {"running-1": running_task, "completed-1": completed_task}

        with patch("utils.agent_manager.get_task_manager", return_value=mock_task_manager):
            result = await inbox_tool.execute({"action": "list"})

            assert len(result) == 1
            response_text = result[0].text
            assert "📋 Agent Task Inbox (2 tasks)" in response_text
            assert "⚡ Running (1)" in response_text
            assert "✅ Completed (1)" in response_text
            assert "running-1" in response_text
            assert "completed-1" in response_text


class TestAgentBatchTool:
    """Test cases for AgentBatchTool."""

    @pytest.fixture
    def batch_tool(self):
        return AgentBatchTool()

    @pytest.mark.asyncio
    async def test_parallel_batch_execution(self, batch_tool):
        """Test parallel batch execution."""
        with patch("asyncio.create_task") as mock_create_task:
            result = await batch_tool.execute(
                {
                    "tasks": [
                        {"agent_type": "claude", "task_description": "Frontend task", "message": "Create home page"},
                        {"agent_type": "aider", "task_description": "Backend task", "message": "Create API endpoints"},
                    ],
                    "coordination_strategy": "parallel",
                    "max_concurrent": 2,
                    "batch_description": "CRUD todo app development",
                }
            )

            assert len(result) == 1
            response_text = result[0].text
            assert "🚀 Batch Agent Tasks Launched" in response_text
            assert "Tasks: 2" in response_text
            assert "parallel" in response_text
            assert "Max Concurrent: 2" in response_text
            assert "Frontend task" in response_text
            assert "Backend task" in response_text

            # Verify background batch execution was started
            mock_create_task.assert_called_once()

    def test_input_schema_validation(self, batch_tool):
        """Test batch tool input schema."""
        schema = batch_tool.get_input_schema()

        assert "tasks" in schema["properties"]
        assert schema["properties"]["tasks"]["minItems"] == 2
        assert schema["properties"]["tasks"]["maxItems"] == 10
        assert "coordination_strategy" in schema["properties"]
        assert "parallel" in schema["properties"]["coordination_strategy"]["enum"]
        assert "sequential" in schema["properties"]["coordination_strategy"]["enum"]


@pytest.mark.integration
class TestAgentOrchestrationIntegration:
    """Integration tests requiring AgentAPI to be installed."""

    @pytest.mark.skipif(
        not pytest.importorskip("subprocess").run(["which", "agentapi"], capture_output=True).returncode == 0,
        reason="AgentAPI not installed",
    )
    @pytest.mark.asyncio
    async def test_full_workflow_integration(self):
        """Test complete agent orchestration workflow."""
        # This test would require actual AgentAPI installation
        # and would test the full workflow from registry to execution
        pass
