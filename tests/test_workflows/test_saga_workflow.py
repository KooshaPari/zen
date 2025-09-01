"""
Tests for Saga Workflow Implementation

Tests the saga pattern for distributed transactions including forward actions,
compensation logic, orchestration vs choreography modes, and failure handling.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.shared.agent_models import AgentType
from workflows.saga_workflow import (
    SagaStatus,
    SagaStepDefinition,
    SagaStepExecution,
    SagaStepStatus,
    SagaTransaction,
    SagaWorkflow,
    start_distributed_saga,
)


class TestSagaStepDefinition:
    """Test suite for SagaStepDefinition model."""

    def test_saga_step_definition_creation(self):
        """Test creating a saga step definition."""
        step = SagaStepDefinition(
            step_id="step1",
            name="Create Database",
            description="Create database schema",
            forward_action={
                "type": "task",
                "description": "Create database schema",
                "database": "user_db"
            },
            compensation_action={
                "type": "task",
                "description": "Drop database schema",
                "database": "user_db"
            },
            agent_type=AgentType.CLAUDE,
            timeout_minutes=15,
            retry_count=2
        )

        assert step.step_id == "step1"
        assert step.name == "Create Database"
        assert step.forward_action["type"] == "task"
        assert step.compensation_action is not None
        assert step.agent_type == AgentType.CLAUDE
        assert step.timeout_minutes == 15
        assert step.retry_count == 2
        assert step.critical is True  # default value
        assert step.dependencies == []  # default value

    def test_saga_step_definition_without_compensation(self):
        """Test saga step without compensation action."""
        step = SagaStepDefinition(
            step_id="step2",
            name="Log Event",
            description="Log audit event",
            forward_action={
                "type": "api_call",
                "url": "https://api.example.com/audit",
                "method": "POST"
            },
            agent_type=AgentType.CLAUDE,
            critical=False  # Non-critical step
        )

        assert step.compensation_action is None
        assert step.critical is False

    def test_saga_step_definition_with_dependencies(self):
        """Test saga step with dependencies."""
        step = SagaStepDefinition(
            step_id="step3",
            name="Deploy API",
            description="Deploy API service",
            forward_action={"type": "task", "description": "Deploy service"},
            agent_type=AgentType.CLAUDE,
            dependencies=["step1", "step2"],
            parallel_group="deployment"
        )

        assert step.dependencies == ["step1", "step2"]
        assert step.parallel_group == "deployment"


class TestSagaStepExecution:
    """Test suite for SagaStepExecution model."""

    def test_saga_step_execution_creation(self):
        """Test creating saga step execution."""
        step_def = SagaStepDefinition(
            step_id="test_step",
            name="Test Step",
            description="Test step execution",
            forward_action={"type": "task"},
            agent_type=AgentType.CLAUDE
        )

        execution = SagaStepExecution(
            step_id="test_step",
            definition=step_def
        )

        assert execution.step_id == "test_step"
        assert execution.definition == step_def
        assert execution.status == SagaStepStatus.PENDING
        assert execution.attempt_count == 0
        assert execution.started_at is None
        assert execution.forward_result is None

    def test_saga_step_execution_with_results(self):
        """Test saga step execution with results."""
        step_def = SagaStepDefinition(
            step_id="completed_step",
            name="Completed Step",
            description="Completed step",
            forward_action={"type": "task"},
            agent_type=AgentType.CLAUDE
        )

        execution = SagaStepExecution(
            step_id="completed_step",
            definition=step_def,
            status=SagaStepStatus.COMPLETED,
            attempt_count=1,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            forward_result={"success": True, "result": "Task completed"}
        )

        assert execution.status == SagaStepStatus.COMPLETED
        assert execution.attempt_count == 1
        assert execution.forward_result["success"] is True


class TestSagaTransaction:
    """Test suite for SagaTransaction model."""

    def test_saga_transaction_creation(self):
        """Test creating a saga transaction."""
        steps = [
            SagaStepDefinition(
                step_id="step1",
                name="Step 1",
                description="First step",
                forward_action={"type": "task"},
                agent_type=AgentType.CLAUDE
            ),
            SagaStepDefinition(
                step_id="step2",
                name="Step 2",
                description="Second step",
                forward_action={"type": "task"},
                agent_type=AgentType.CLAUDE
            )
        ]

        saga = SagaTransaction(
            saga_id="test-saga",
            name="Test Saga",
            description="Test saga transaction",
            steps=steps,
            created_at=datetime.utcnow(),
            context={"environment": "test"}
        )

        assert saga.saga_id == "test-saga"
        assert saga.name == "Test Saga"
        assert len(saga.steps) == 2
        assert saga.status == SagaStatus.PENDING
        assert saga.context["environment"] == "test"
        assert len(saga.execution_state) == 0  # Initially empty
        assert saga.compensation_order == []


class TestSagaWorkflow:
    """Test suite for SagaWorkflow."""

    @pytest.fixture
    def orchestration_workflow(self):
        """Create orchestration-based saga workflow."""
        workflow = SagaWorkflow(coordination_mode="orchestration")
        # Mock dependencies
        workflow.task_manager = MagicMock()
        workflow.storage = MagicMock()
        workflow.event_bus = MagicMock()
        workflow.event_bus.publish = AsyncMock()
        workflow.event_bus.subscribe = AsyncMock(return_value=AsyncMock())
        return workflow

    @pytest.fixture
    def choreography_workflow(self):
        """Create choreography-based saga workflow."""
        workflow = SagaWorkflow(coordination_mode="choreography")
        # Mock dependencies
        workflow.task_manager = MagicMock()
        workflow.storage = MagicMock()
        workflow.event_bus = MagicMock()
        workflow.event_bus.publish = AsyncMock()
        workflow.event_bus.subscribe = AsyncMock(return_value=AsyncMock())
        return workflow

    @pytest.fixture
    def sample_saga_definition(self):
        """Create sample saga definition."""
        return {
            "saga_id": "test-saga-123",
            "name": "Test Saga",
            "description": "Test distributed transaction",
            "steps": [
                {
                    "step_id": "create_user",
                    "name": "Create User",
                    "description": "Create user account",
                    "forward_action": {
                        "type": "task",
                        "description": "Create user in database",
                        "success_rate": 0.9
                    },
                    "compensation_action": {
                        "type": "task",
                        "description": "Delete user from database"
                    },
                    "agent_type": "CLAUDE",
                    "timeout_minutes": 10,
                    "retry_count": 2,
                    "critical": True,
                    "dependencies": [],
                    "parallel_group": None
                },
                {
                    "step_id": "send_welcome_email",
                    "name": "Send Welcome Email",
                    "description": "Send welcome email to user",
                    "forward_action": {
                        "type": "api_call",
                        "url": "https://api.example.com/email",
                        "method": "POST"
                    },
                    "compensation_action": None,
                    "agent_type": "CLAUDE",
                    "timeout_minutes": 5,
                    "retry_count": 1,
                    "critical": False,
                    "dependencies": ["create_user"],
                    "parallel_group": None
                }
            ],
            "context": {"user_id": "user123", "environment": "test"}
        }

    def test_orchestration_workflow_initialization(self, orchestration_workflow):
        """Test orchestration workflow initialization."""
        assert orchestration_workflow.coordination_mode == "orchestration"
        assert orchestration_workflow.task_manager is not None
        assert orchestration_workflow.storage is not None
        assert orchestration_workflow.event_bus is not None

    def test_choreography_workflow_initialization(self, choreography_workflow):
        """Test choreography workflow initialization."""
        assert choreography_workflow.coordination_mode == "choreography"

    @pytest.mark.asyncio
    async def test_orchestrate_success(self, orchestration_workflow, sample_saga_definition):
        """Test successful saga orchestration."""
        # Mock storage operations
        orchestration_workflow.storage.setex = MagicMock()

        # Mock successful step execution
        orchestration_workflow._execute_saga_step = AsyncMock(return_value={
            "success": True,
            "result": {"message": "Step completed successfully"}
        })

        workflow_args = {
            "saga_definition": sample_saga_definition,
            "config": {"timeout_seconds": 300}
        }

        result = await orchestration_workflow.orchestrate(workflow_args)

        assert result["saga_id"] == "test-saga-123"
        assert result["status"] == "completed"
        assert result["success_rate"] == 1.0
        assert result["completed_steps"] == 2
        assert result["total_steps"] == 2

        # Verify events were published
        assert orchestration_workflow.event_bus.publish.called

    @pytest.mark.asyncio
    async def test_orchestrate_with_compensation(self, orchestration_workflow, sample_saga_definition):
        """Test saga orchestration with compensation."""
        orchestration_workflow.storage.setex = MagicMock()

        # Mock first step success, second step failure
        def mock_execute_step(saga_transaction, step_definition, config):
            if step_definition.step_id == "create_user":
                return {"success": True, "result": {"user_id": "user123"}}
            else:
                return {"success": False, "error": "Email service unavailable"}

        orchestration_workflow._execute_saga_step = AsyncMock(side_effect=mock_execute_step)

        # Mock successful compensation
        orchestration_workflow._execute_compensation = AsyncMock(return_value={
            "success": True,
            "successful_compensations": 1,
            "total_compensations": 1
        })

        workflow_args = {
            "saga_definition": sample_saga_definition,
            "config": {}
        }

        result = await orchestration_workflow.orchestrate(workflow_args)

        assert result["saga_id"] == "test-saga-123"
        assert result["status"] == "compensated"
        assert "compensation_result" in result
        assert result["compensation_result"]["success"] is True

    @pytest.mark.asyncio
    async def test_choreography_orchestrate(self, choreography_workflow, sample_saga_definition):
        """Test choreography-based saga orchestration."""
        choreography_workflow.storage.setex = MagicMock()

        # Mock event queue for choreography
        mock_queue = AsyncMock()
        choreography_workflow.event_bus.subscribe = AsyncMock(return_value=mock_queue)

        # Mock events for completion
        completion_events = [
            {
                "saga_id": "test-saga-123",
                "event": "saga_step_completed",
                "step_id": "create_user"
            },
            {
                "saga_id": "test-saga-123",
                "event": "saga_step_completed",
                "step_id": "send_welcome_email"
            }
        ]

        mock_queue.get = AsyncMock(side_effect=completion_events)

        workflow_args = {
            "saga_definition": sample_saga_definition,
            "config": {"saga_timeout_minutes": 5}
        }

        result = await choreography_workflow.orchestrate(workflow_args)

        assert result["saga_id"] == "test-saga-123"
        assert result["status"] == "completed"
        assert result["completed_steps"] == 2

    def test_build_execution_plan(self, orchestration_workflow):
        """Test execution plan building with dependencies."""
        steps = [
            SagaStepDefinition(
                step_id="step1",
                name="Step 1",
                description="Independent step",
                forward_action={"type": "task"},
                agent_type=AgentType.CLAUDE,
                dependencies=[]
            ),
            SagaStepDefinition(
                step_id="step2",
                name="Step 2",
                description="Depends on step1",
                forward_action={"type": "task"},
                agent_type=AgentType.CLAUDE,
                dependencies=["step1"]
            ),
            SagaStepDefinition(
                step_id="step3",
                name="Step 3",
                description="Parallel with step2",
                forward_action={"type": "task"},
                agent_type=AgentType.CLAUDE,
                dependencies=["step1"],
                parallel_group="group1"
            ),
            SagaStepDefinition(
                step_id="step4",
                name="Step 4",
                description="Also parallel with step2",
                forward_action={"type": "task"},
                agent_type=AgentType.CLAUDE,
                dependencies=["step1"],
                parallel_group="group1"
            )
        ]

        execution_plan = orchestration_workflow._build_execution_plan(steps)

        # Should have 3 execution groups:
        # 1. step1 (independent)
        # 2. step2 (sequential, depends on step1)
        # 3. [step3, step4] (parallel group, both depend on step1)
        assert len(execution_plan) == 3

        # First group should be step1
        assert len(execution_plan[0]) == 1
        assert execution_plan[0][0].step_id == "step1"

        # Second group should be step2
        assert len(execution_plan[1]) == 1
        assert execution_plan[1][0].step_id == "step2"

        # Third group should be step3 and step4 (parallel)
        assert len(execution_plan[2]) == 2
        parallel_step_ids = {step.step_id for step in execution_plan[2]}
        assert parallel_step_ids == {"step3", "step4"}

    def test_build_execution_plan_circular_dependency(self, orchestration_workflow):
        """Test execution plan with circular dependency detection."""
        steps = [
            SagaStepDefinition(
                step_id="step1",
                name="Step 1",
                description="Depends on step2",
                forward_action={"type": "task"},
                agent_type=AgentType.CLAUDE,
                dependencies=["step2"]
            ),
            SagaStepDefinition(
                step_id="step2",
                name="Step 2",
                description="Depends on step1",
                forward_action={"type": "task"},
                agent_type=AgentType.CLAUDE,
                dependencies=["step1"]
            )
        ]

        with pytest.raises(Exception, match="Cannot resolve step dependencies"):
            orchestration_workflow._build_execution_plan(steps)

    @pytest.mark.asyncio
    async def test_execute_saga_step_success(self, orchestration_workflow):
        """Test successful saga step execution."""
        step_definition = SagaStepDefinition(
            step_id="test_step",
            name="Test Step",
            description="Test step",
            forward_action={
                "type": "task",
                "description": "Test task",
                "success_rate": 1.0,  # Always succeed
                "duration_seconds": 0.1
            },
            agent_type=AgentType.CLAUDE,
            retry_count=1
        )

        saga_transaction = SagaTransaction(
            saga_id="test-saga",
            name="Test Saga",
            description="Test saga",
            steps=[step_definition],
            created_at=datetime.utcnow()
        )

        # Initialize execution state
        saga_transaction.execution_state[step_definition.step_id] = SagaStepExecution(
            step_id=step_definition.step_id,
            definition=step_definition
        )

        result = await orchestration_workflow._execute_saga_step(
            saga_transaction, step_definition, {}
        )

        assert result["success"] is True
        assert "result" in result

        # Check execution state was updated
        execution = saga_transaction.execution_state[step_definition.step_id]
        assert execution.status == SagaStepStatus.COMPLETED
        assert execution.forward_result is not None
        assert step_definition.step_id in saga_transaction.compensation_order

    @pytest.mark.asyncio
    async def test_execute_saga_step_with_retry(self, orchestration_workflow):
        """Test saga step execution with retry logic."""
        step_definition = SagaStepDefinition(
            step_id="retry_step",
            name="Retry Step",
            description="Step that needs retry",
            forward_action={
                "type": "task",
                "description": "Flaky task",
                "success_rate": 0.3,  # Low success rate to trigger retries
                "duration_seconds": 0.05
            },
            agent_type=AgentType.CLAUDE,
            retry_count=3
        )

        saga_transaction = SagaTransaction(
            saga_id="retry-saga",
            name="Retry Saga",
            description="Saga with retry",
            steps=[step_definition],
            created_at=datetime.utcnow()
        )

        saga_transaction.execution_state[step_definition.step_id] = SagaStepExecution(
            step_id=step_definition.step_id,
            definition=step_definition
        )

        result = await orchestration_workflow._execute_saga_step(
            saga_transaction, step_definition, {}
        )

        # Should eventually succeed or fail after retries
        assert "success" in result

        execution = saga_transaction.execution_state[step_definition.step_id]
        assert execution.attempt_count > 0
        assert execution.status in [SagaStepStatus.COMPLETED, SagaStepStatus.FAILED]

    @pytest.mark.asyncio
    async def test_execute_step_action_task(self, orchestration_workflow):
        """Test executing task-type step action."""
        action = {
            "type": "task",
            "description": "Test task action",
            "success_rate": 1.0,
            "duration_seconds": 0.1
        }

        result = await orchestration_workflow._execute_step_action(
            action, AgentType.CLAUDE, {}, 30
        )

        assert result["success"] is True
        assert result["agent_type"] == AgentType.CLAUDE.value
        assert result["task_description"] == "Test task action"

    @pytest.mark.asyncio
    async def test_execute_step_action_api_call(self, orchestration_workflow):
        """Test executing API call-type step action."""
        action = {
            "type": "api_call",
            "url": "https://api.example.com/test",
            "method": "POST"
        }

        result = await orchestration_workflow._execute_step_action(
            action, AgentType.CLAUDE, {}, 30
        )

        assert result["success"] is True
        assert result["method"] == "POST"
        assert result["url"] == "https://api.example.com/test"

    @pytest.mark.asyncio
    async def test_execute_step_action_script(self, orchestration_workflow):
        """Test executing script-type step action."""
        action = {
            "type": "script",
            "script": "print('Hello, World!')"
        }

        result = await orchestration_workflow._execute_step_action(
            action, AgentType.CLAUDE, {}, 30
        )

        assert result["success"] is True
        assert result["script_length"] == len("print('Hello, World!')")

    @pytest.mark.asyncio
    async def test_execute_step_action_unsupported(self, orchestration_workflow):
        """Test executing unsupported action type."""
        action = {
            "type": "unsupported_type",
            "data": "test"
        }

        result = await orchestration_workflow._execute_step_action(
            action, AgentType.CLAUDE, {}, 30
        )

        assert result["success"] is False
        assert "Unsupported action type" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_compensation(self, orchestration_workflow):
        """Test compensation execution."""
        step1 = SagaStepDefinition(
            step_id="step1",
            name="Step 1",
            description="First step",
            forward_action={"type": "task"},
            compensation_action={"type": "task", "description": "Compensate step 1"},
            agent_type=AgentType.CLAUDE
        )

        step2 = SagaStepDefinition(
            step_id="step2",
            name="Step 2",
            description="Second step",
            forward_action={"type": "task"},
            compensation_action={"type": "task", "description": "Compensate step 2"},
            agent_type=AgentType.CLAUDE
        )

        saga_transaction = SagaTransaction(
            saga_id="compensation-saga",
            name="Compensation Saga",
            description="Saga needing compensation",
            steps=[step1, step2],
            created_at=datetime.utcnow(),
            compensation_order=["step2", "step1"]  # Reverse order
        )

        # Initialize execution states
        for step in [step1, step2]:
            saga_transaction.execution_state[step.step_id] = SagaStepExecution(
                step_id=step.step_id,
                definition=step,
                status=SagaStepStatus.COMPLETED,
                forward_result={"success": True}
            )

        result = await orchestration_workflow._execute_compensation(saga_transaction)

        assert result["success"] is True
        assert result["successful_compensations"] == 2
        assert result["total_compensations"] == 2

        # Check compensation results were stored
        for step_id in ["step1", "step2"]:
            execution = saga_transaction.execution_state[step_id]
            assert execution.status == SagaStepStatus.COMPENSATED
            assert execution.compensation_result is not None

    @pytest.mark.asyncio
    async def test_store_saga_transaction(self, orchestration_workflow):
        """Test saga transaction storage."""
        saga_transaction = SagaTransaction(
            saga_id="storage-test-saga",
            name="Storage Test Saga",
            description="Test saga storage",
            steps=[],
            created_at=datetime.utcnow()
        )

        await orchestration_workflow._store_saga_transaction(saga_transaction)

        # Verify storage was called
        orchestration_workflow.storage.setex.assert_called()

        call_args = orchestration_workflow.storage.setex.call_args
        key, ttl, data = call_args[0]

        assert key == f"saga_transaction:{saga_transaction.saga_id}"
        assert ttl == 24 * 60 * 60  # 24 hours

        # Verify data serialization
        import json
        saga_data = json.loads(data)
        assert saga_data["saga_id"] == "storage-test-saga"
        assert saga_data["name"] == "Storage Test Saga"

    @pytest.mark.asyncio
    async def test_get_saga_transaction(self, orchestration_workflow):
        """Test saga transaction retrieval."""
        saga_data = {
            "saga_id": "retrieve-test-saga",
            "name": "Retrieve Test Saga",
            "description": "Test retrieval",
            "steps": [],
            "status": "completed",
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "completed_at": None,
            "context": {},
            "execution_state": {},
            "compensation_order": [],
            "total_duration_seconds": None,
            "success_rate": 0.0
        }

        orchestration_workflow.storage.get = MagicMock(return_value=json.dumps(saga_data))

        saga = await orchestration_workflow.get_saga_transaction("retrieve-test-saga")

        assert saga is not None
        assert isinstance(saga, SagaTransaction)
        assert saga.saga_id == "retrieve-test-saga"
        assert saga.name == "Retrieve Test Saga"
        assert saga.status == SagaStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_get_saga_transaction_not_found(self, orchestration_workflow):
        """Test saga transaction retrieval when not found."""
        orchestration_workflow.storage.get = MagicMock(return_value=None)

        saga = await orchestration_workflow.get_saga_transaction("non-existent-saga")

        assert saga is None


class TestSagaConvenienceFunctions:
    """Test suite for saga convenience functions."""

    @pytest.mark.asyncio
    async def test_start_distributed_saga_orchestration(self):
        """Test start_distributed_saga with orchestration mode."""
        saga_definition = {
            "saga_id": "convenience-test",
            "name": "Convenience Test Saga",
            "description": "Test convenience function",
            "steps": [
                {
                    "step_id": "test_step",
                    "name": "Test Step",
                    "description": "Test step",
                    "forward_action": {"type": "task"},
                    "agent_type": "CLAUDE"
                }
            ]
        }

        with patch("workflows.saga_workflow.get_temporal_client") as mock_get_client:
            mock_client = MagicMock()
            mock_result = MagicMock()
            mock_result.model_dump = MagicMock(return_value={"status": "completed"})
            mock_client.start_workflow = AsyncMock(return_value=mock_result)
            mock_get_client.return_value = mock_client

            result = await start_distributed_saga(
                saga_definition=saga_definition,
                coordination_mode="orchestration",
                config={"timeout_seconds": 600}
            )

            assert result["status"] == "completed"
            mock_client.start_workflow.assert_called_once()

            # Verify workflow arguments
            call_args = mock_client.start_workflow.call_args
            workflow_args = call_args[1]["workflow_args"]
            assert workflow_args["saga_definition"]["saga_id"] == "convenience-test"
            assert workflow_args["config"]["timeout_seconds"] == 600

    @pytest.mark.asyncio
    async def test_start_distributed_saga_choreography(self):
        """Test start_distributed_saga with choreography mode."""
        saga_definition = {
            "saga_id": "choreography-test",
            "name": "Choreography Test Saga",
            "description": "Test choreography mode",
            "steps": []
        }

        with patch("workflows.saga_workflow.get_temporal_client") as mock_get_client:
            mock_client = MagicMock()
            mock_result = MagicMock()
            mock_result.model_dump = MagicMock(return_value={"status": "completed"})
            mock_client.start_workflow = AsyncMock(return_value=mock_result)
            mock_get_client.return_value = mock_client

            result = await start_distributed_saga(
                saga_definition=saga_definition,
                coordination_mode="choreography"
            )

            assert result["status"] == "completed"

            # Verify that SagaWorkflow was configured with choreography mode
            # The workflow class should be created with choreography mode
            # This tests the integration but not the internal workflow creation


@pytest.mark.integration
class TestSagaWorkflowIntegration:
    """Integration tests for saga workflow."""

    @pytest.mark.asyncio
    async def test_end_to_end_saga_execution(self):
        """Test complete saga execution flow."""
        pytest.skip("Requires full integration environment with Temporal and agents")

    @pytest.mark.asyncio
    async def test_distributed_transaction_with_real_services(self):
        """Test distributed transaction with real service calls."""
        pytest.skip("Requires external services for integration testing")
