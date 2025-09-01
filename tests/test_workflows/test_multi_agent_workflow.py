"""
Tests for Multi-Agent Workflow Orchestration

Tests the complex multi-agent project workflow including phase execution,
agent coordination, approval gates, and failure recovery.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.shared.agent_models import AgentType
from workflows.multi_agent_workflow import (
    AgentAssignment,
    MultiAgentProjectWorkflow,
    ProjectPhase,
    ProjectSpec,
    ProjectWorkflowState,
    start_multi_agent_project,
)


class TestProjectSpec:
    """Test suite for ProjectSpec model."""

    def test_project_spec_creation(self):
        """Test creating a valid project specification."""
        spec = ProjectSpec(
            project_id="test-project-123",
            name="Test Project",
            description="A test project for validation",
            requirements=["Feature A", "Feature B"],
            constraints={"budget": 10000, "timeline": "2 weeks"},
            success_criteria=["All features implemented", "Tests passing"],
            estimated_duration_hours=40,
            agents_required=[AgentType.CLAUDE, AgentType.AIDER]
        )

        assert spec.project_id == "test-project-123"
        assert spec.name == "Test Project"
        assert len(spec.requirements) == 2
        assert spec.constraints["budget"] == 10000
        assert len(spec.success_criteria) == 2
        assert spec.estimated_duration_hours == 40
        assert AgentType.CLAUDE in spec.agents_required
        assert spec.priority == "normal"  # default value
        assert spec.approval_gates == []  # default value

    def test_project_spec_with_approval_gates(self):
        """Test project spec with approval gates."""
        spec = ProjectSpec(
            project_id="approval-project",
            name="Project with Approvals",
            description="Project requiring approvals",
            requirements=["Secure feature"],
            constraints={},
            success_criteria=["Security validated"],
            estimated_duration_hours=20,
            agents_required=[AgentType.CLAUDE],
            approval_gates=["Backend API", "deployment"]
        )

        assert "Backend API" in spec.approval_gates
        assert "deployment" in spec.approval_gates


class TestProjectPhase:
    """Test suite for ProjectPhase model."""

    def test_project_phase_creation(self):
        """Test creating a project phase."""
        phase = ProjectPhase(
            phase_id="implement_api",
            name="Implement API",
            description="Implement REST API endpoints",
            dependencies=["design_phase"],
            estimated_duration_minutes=240,
            required_agents=[AgentType.CLAUDE],
            tasks=[{"description": "Create API endpoints", "files": ["api.py"]}],
            success_criteria=["API endpoints functional"]
        )

        assert phase.phase_id == "implement_api"
        assert phase.name == "Implement API"
        assert phase.dependencies == ["design_phase"]
        assert phase.estimated_duration_minutes == 240
        assert AgentType.CLAUDE in phase.required_agents
        assert len(phase.tasks) == 1
        assert phase.requires_approval is False  # default value

    def test_project_phase_with_approval(self):
        """Test project phase requiring approval."""
        phase = ProjectPhase(
            phase_id="deploy_prod",
            name="Deploy to Production",
            description="Deploy application to production",
            dependencies=["testing"],
            estimated_duration_minutes=60,
            required_agents=[AgentType.CLAUDE],
            tasks=[{"description": "Deploy application"}],
            success_criteria=["Application deployed"],
            requires_approval=True
        )

        assert phase.requires_approval is True


class TestAgentAssignment:
    """Test suite for AgentAssignment model."""

    def test_agent_assignment_creation(self):
        """Test creating an agent assignment."""
        assignment = AgentAssignment(
            assignment_id="assignment-123",
            agent_type=AgentType.CLAUDE,
            task_description="Implement user authentication",
            context={"project_id": "test-project"},
            estimated_duration_minutes=120,
            priority="high"
        )

        assert assignment.assignment_id == "assignment-123"
        assert assignment.agent_type == AgentType.CLAUDE
        assert assignment.task_description == "Implement user authentication"
        assert assignment.context["project_id"] == "test-project"
        assert assignment.estimated_duration_minutes == 120
        assert assignment.priority == "high"
        assert assignment.dependencies == []  # default value


class TestProjectWorkflowState:
    """Test suite for ProjectWorkflowState model."""

    def test_workflow_state_creation(self):
        """Test creating workflow state."""
        start_time = datetime.utcnow()

        state = ProjectWorkflowState(
            project_id="test-project",
            workflow_id="workflow-123",
            status="running",
            start_time=start_time
        )

        assert state.project_id == "test-project"
        assert state.workflow_id == "workflow-123"
        assert state.status == "running"
        assert state.start_time == start_time
        assert state.current_phase is None  # default
        assert state.completed_phases == []  # default
        assert state.total_agents_used == 0  # default
        assert state.success_rate == 0.0  # default


class TestMultiAgentProjectWorkflow:
    """Test suite for MultiAgentProjectWorkflow."""

    @pytest.fixture
    def workflow(self):
        """Create test workflow instance."""
        workflow = MultiAgentProjectWorkflow()
        # Mock dependencies
        workflow.task_manager = MagicMock()
        workflow.storage = MagicMock()
        workflow.event_bus = MagicMock()
        workflow.event_bus.publish = AsyncMock()
        return workflow

    @pytest.fixture
    def sample_project_spec(self):
        """Create sample project specification."""
        return ProjectSpec(
            project_id="sample-project",
            name="Sample Project",
            description="A sample project for testing",
            requirements=["User authentication", "Data storage"],
            constraints={"timeline": "1 week"},
            success_criteria=["All features working", "Tests pass"],
            estimated_duration_hours=20,
            agents_required=[AgentType.CLAUDE]
        )

    def test_workflow_initialization(self, workflow):
        """Test workflow initialization."""
        assert workflow.task_manager is not None
        assert workflow.storage is not None
        assert workflow.event_bus is not None

    @pytest.mark.asyncio
    async def test_orchestrate_success(self, workflow, sample_project_spec):
        """Test successful workflow orchestration."""
        # Setup workflow context
        workflow.setup_context("test-workflow-123")

        # Mock storage operations
        workflow.storage.setex = MagicMock()

        # Mock wait_for_approval to always return True
        workflow.wait_for_approval = AsyncMock(return_value=True)

        # Execute workflow
        workflow_args = {
            "project_spec": sample_project_spec.model_dump(),
            "config": {"approval_timeout": 3600}
        }

        result = await workflow.orchestrate(workflow_args)

        # Verify result
        assert result["status"] == "completed"
        assert result["project_id"] == "sample-project"
        assert "workflow_id" in result
        assert "execution_time_seconds" in result
        assert result["success_rate"] > 0

        # Verify events were published
        assert workflow.event_bus.publish.called

    @pytest.mark.asyncio
    async def test_orchestrate_approval_rejection(self, workflow, sample_project_spec):
        """Test workflow with approval rejection."""
        workflow.setup_context("test-workflow-123")
        workflow.storage.setex = MagicMock()

        # Mock wait_for_approval to return False (rejection)
        workflow.wait_for_approval = AsyncMock(return_value=False)

        # Add approval gate to project
        project_spec = sample_project_spec.model_dump()
        project_spec["approval_gates"] = ["Backend API"]

        workflow_args = {
            "project_spec": project_spec,
            "config": {}
        }

        result = await workflow.orchestrate(workflow_args)

        assert result["status"] == "approval_rejected"
        assert "project_state" in result

    @pytest.mark.asyncio
    async def test_orchestrate_failure_with_compensation(self, workflow, sample_project_spec):
        """Test workflow failure handling with compensation."""
        workflow.setup_context("test-workflow-123")
        workflow.storage.setex = MagicMock()

        # Mock analysis phase to fail
        with patch.object(workflow, '_execute_analysis_phase', side_effect=Exception("Analysis failed")):
            workflow_args = {
                "project_spec": sample_project_spec.model_dump(),
                "config": {}
            }

            result = await workflow.orchestrate(workflow_args)

            assert result["status"] == "failed"
            assert "error" in result
            assert "Analysis failed" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_analysis_phase(self, workflow, sample_project_spec):
        """Test analysis phase execution."""
        project_state = ProjectWorkflowState(
            project_id="test-project",
            workflow_id="workflow-123",
            status="running",
            start_time=datetime.utcnow()
        )

        result = await workflow._execute_analysis_phase(sample_project_spec, project_state)

        assert result["success"] is True
        assert "components" in result
        assert len(result["components"]) > 0
        assert "test_requirements" in result

        # Verify event was published
        workflow.event_bus.publish.assert_called()

    def test_build_analysis_prompt(self, workflow, sample_project_spec):
        """Test analysis prompt building."""
        prompt = workflow._build_analysis_prompt(sample_project_spec)

        assert "Sample Project" in prompt
        assert "User authentication" in prompt
        assert "Data storage" in prompt
        assert "timeline" in prompt
        assert "All features working" in prompt

    @pytest.mark.asyncio
    async def test_generate_project_phases(self, workflow, sample_project_spec):
        """Test project phase generation."""
        analysis_result = {
            "components": [
                {
                    "name": "Backend API",
                    "description": "REST API implementation",
                    "preferred_agent": AgentType.CLAUDE,
                    "estimated_hours": 4,
                    "implementation_task": {
                        "description": "Implement API endpoints",
                        "files": ["api/main.py"]
                    }
                }
            ],
            "test_requirements": ["Unit tests", "Integration tests"]
        }

        phases = await workflow._generate_project_phases(
            sample_project_spec,
            analysis_result,
            {}
        )

        assert len(phases) >= 1  # At least implementation phase

        # Check implementation phase
        impl_phase = next(p for p in phases if "implement_backend_api" in p.phase_id)
        assert impl_phase.name == "Implement Backend API"
        assert AgentType.CLAUDE in impl_phase.required_agents
        assert impl_phase.estimated_duration_minutes == 240  # 4 hours * 60 minutes

        # Check if testing phase was created
        testing_phases = [p for p in phases if p.phase_id == "testing"]
        if testing_phases:
            testing_phase = testing_phases[0]
            assert "testing" in testing_phase.name.lower()
            assert len(testing_phase.dependencies) > 0  # Should depend on implementation phases

    @pytest.mark.asyncio
    async def test_execute_project_phase_success(self, workflow, sample_project_spec):
        """Test successful project phase execution."""
        phase = ProjectPhase(
            phase_id="test_phase",
            name="Test Phase",
            description="Test phase execution",
            estimated_duration_minutes=60,
            required_agents=[AgentType.CLAUDE],
            tasks=[{"description": "Test task"}],
            success_criteria=["Task completed"]
        )

        project_state = ProjectWorkflowState(
            project_id="test-project",
            workflow_id="workflow-123",
            status="running",
            start_time=datetime.utcnow()
        )

        # Mock execute_agent_assignment to return success
        workflow._execute_agent_assignment = AsyncMock(return_value={
            "success": True,
            "assignment_id": "test-assignment",
            "result": "Task completed"
        })

        result = await workflow._execute_project_phase(phase, sample_project_spec, project_state)

        assert result["success"] is True
        assert result["phase_id"] == "test_phase"
        assert result["agents_used"] == 1
        assert result["success_rate"] == 1.0

        # Verify events were published
        assert workflow.event_bus.publish.call_count >= 2  # Started and completed events

    @pytest.mark.asyncio
    async def test_execute_project_phase_failure(self, workflow, sample_project_spec):
        """Test project phase execution with failures."""
        phase = ProjectPhase(
            phase_id="failing_phase",
            name="Failing Phase",
            description="Phase that will fail",
            estimated_duration_minutes=60,
            required_agents=[AgentType.CLAUDE],
            tasks=[{"description": "Failing task"}],
            success_criteria=["Task completed"]
        )

        project_state = ProjectWorkflowState(
            project_id="test-project",
            workflow_id="workflow-123",
            status="running",
            start_time=datetime.utcnow()
        )

        # Mock execute_agent_assignment to return failure
        workflow._execute_agent_assignment = AsyncMock(return_value={
            "success": False,
            "error": "Task execution failed"
        })

        result = await workflow._execute_project_phase(phase, sample_project_spec, project_state)

        assert result["success"] is False
        assert result["success_rate"] < 0.8  # Below success threshold
        assert result["failed_assignments"] == 1

    @pytest.mark.asyncio
    async def test_execute_agent_assignment_success(self, workflow):
        """Test successful agent assignment execution."""
        assignment = AgentAssignment(
            assignment_id="test-assignment",
            agent_type=AgentType.CLAUDE,
            task_description="Test task",
            context={},
            estimated_duration_minutes=30,
            priority="normal"
        )

        # Mock successful execution (simulated in the actual method)
        result = await workflow._execute_agent_assignment(assignment)

        # Note: The actual method uses random simulation
        # In real implementation, this would interface with agent task manager
        assert "success" in result
        assert result["assignment_id"] == "test-assignment"
        assert result["agent_type"] == AgentType.CLAUDE.value

    @pytest.mark.asyncio
    async def test_execute_integration_phase(self, workflow, sample_project_spec):
        """Test integration phase execution."""
        project_state = ProjectWorkflowState(
            project_id="test-project",
            workflow_id="workflow-123",
            status="running",
            start_time=datetime.utcnow(),
            completed_phases=["implement_backend_api", "implement_frontend_ui"],
            failed_phases=[]
        )

        result = await workflow._execute_integration_phase(
            sample_project_spec,
            project_state,
            {}
        )

        assert result["success"] is True
        assert result["tests_passed"] is True
        assert result["deployment_ready"] is True
        assert "validation_results" in result

    @pytest.mark.asyncio
    async def test_execute_compensation_actions(self, workflow):
        """Test compensation actions execution."""
        project_state = ProjectWorkflowState(
            project_id="test-project",
            workflow_id="workflow-123",
            status="failed",
            start_time=datetime.utcnow(),
            active_assignments={"assignment1": "task1", "assignment2": "task2"}
        )

        await workflow._execute_compensation_actions(project_state)

        # Verify compensation event was published
        workflow.event_bus.publish.assert_called()

        # Check the last published event
        last_call = workflow.event_bus.publish.call_args_list[-1]
        event_data = last_call[0][0]
        assert event_data["event"] == "project_compensation_executed"
        assert event_data["project_id"] == "test-project"
        assert event_data["cancelled_assignments"] == 2

    @pytest.mark.asyncio
    async def test_store_project_state(self, workflow):
        """Test project state storage."""
        project_state = ProjectWorkflowState(
            project_id="test-project",
            workflow_id="workflow-123",
            status="running",
            start_time=datetime.utcnow()
        )

        await workflow._store_project_state(project_state)

        # Verify storage was called
        workflow.storage.setex.assert_called()

        # Check storage call parameters
        call_args = workflow.storage.setex.call_args
        key, ttl, data = call_args[0]

        assert key == f"project_state:{project_state.project_id}"
        assert ttl == 24 * 60 * 60  # 24 hours

        # Verify data can be deserialized
        import json
        state_data = json.loads(data)
        assert state_data["project_id"] == "test-project"
        assert state_data["status"] == "running"

    @pytest.mark.asyncio
    async def test_get_project_state(self, workflow):
        """Test project state retrieval."""
        # Mock storage to return project state data
        project_data = {
            "project_id": "test-project",
            "workflow_id": "workflow-123",
            "status": "completed",
            "start_time": datetime.utcnow().isoformat(),
            "completed_phases": ["phase1", "phase2"],
            "failed_phases": [],
            "active_assignments": {},
            "completed_assignments": [],
            "failed_assignments": [],
            "total_agents_used": 2,
            "success_rate": 1.0,
            "error_log": []
        }

        workflow.storage.get = MagicMock(return_value=json.dumps(project_data))

        state = await workflow.get_project_state("test-project")

        assert state is not None
        assert isinstance(state, ProjectWorkflowState)
        assert state.project_id == "test-project"
        assert state.workflow_id == "workflow-123"
        assert state.status == "completed"
        assert len(state.completed_phases) == 2
        assert state.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_get_project_state_not_found(self, workflow):
        """Test project state retrieval when not found."""
        workflow.storage.get = MagicMock(return_value=None)

        state = await workflow.get_project_state("non-existent-project")

        assert state is None


class TestWorkflowConvenienceFunctions:
    """Test suite for workflow convenience functions."""

    @pytest.mark.asyncio
    async def test_start_multi_agent_project(self):
        """Test start_multi_agent_project convenience function."""
        project_spec = ProjectSpec(
            project_id="convenience-test",
            name="Convenience Test Project",
            description="Test convenience function",
            requirements=["Test requirement"],
            constraints={},
            success_criteria=["Test passes"],
            estimated_duration_hours=1,
            agents_required=[AgentType.CLAUDE]
        )

        with patch("workflows.multi_agent_workflow.get_temporal_client") as mock_get_client:
            mock_client = MagicMock()
            mock_result = MagicMock()
            mock_result.model_dump = MagicMock(return_value={"status": "completed"})
            mock_client.start_workflow = AsyncMock(return_value=mock_result)
            mock_get_client.return_value = mock_client

            result = await start_multi_agent_project(
                project_spec=project_spec,
                config={"test_config": True}
            )

            assert result["status"] == "completed"
            mock_client.start_workflow.assert_called_once()

            # Verify workflow arguments
            call_args = mock_client.start_workflow.call_args
            workflow_args = call_args[1]["workflow_args"]
            assert workflow_args["project_spec"]["project_id"] == "convenience-test"
            assert workflow_args["config"]["test_config"] is True


@pytest.mark.integration
class TestMultiAgentWorkflowIntegration:
    """Integration tests for multi-agent workflow."""

    @pytest.mark.asyncio
    async def test_end_to_end_project_workflow(self):
        """Test complete project workflow execution."""
        # This would require full integration environment
        pytest.skip("Requires full integration environment with Temporal and agents")

    @pytest.mark.asyncio
    async def test_real_agent_coordination(self):
        """Test real agent coordination through workflow."""
        pytest.skip("Requires agent API and task manager integration")
