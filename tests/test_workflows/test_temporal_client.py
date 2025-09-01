"""
Tests for Temporal Workflow Client

Tests the core Temporal integration, workflow execution, human approval
management, and fallback behavior when Temporal is not available.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.temporal_client import (
    ApprovalDecision,
    BaseWorkflow,
    HumanApprovalRequest,
    TemporalWorkflowClient,
    WorkflowExecutionResult,
    get_temporal_client,
)


class MockWorkflow(BaseWorkflow):
    """Mock workflow for testing."""

    async def orchestrate(self, workflow_args):
        await asyncio.sleep(0.1)  # Simulate work
        if workflow_args.get("should_fail"):
            raise Exception("Simulated workflow failure")

        return {
            "status": "completed",
            "result": "Mock workflow completed successfully",
            "args": workflow_args
        }


class TestTemporalWorkflowClient:
    """Test suite for Temporal workflow client."""

    @pytest.fixture
    def client(self):
        """Create test Temporal client."""
        return TemporalWorkflowClient(
            temporal_address="localhost:7233",
            namespace="test",
            task_queue="test-workflows"
        )

    @pytest.fixture
    def mock_temporal_client(self):
        """Mock Temporal client for testing."""
        mock_client = MagicMock()
        mock_handle = MagicMock()
        mock_handle.first_execution_run_id = "test-run-123"
        mock_handle.result = AsyncMock(return_value={"status": "completed"})
        mock_client.start_workflow = AsyncMock(return_value=mock_handle)
        return mock_client

    def test_client_initialization(self, client):
        """Test client initialization with correct parameters."""
        assert client.temporal_address == "localhost:7233"
        assert client.namespace == "test"
        assert client.task_queue == "test-workflows"
        assert client.client is None
        assert client.worker is None
        assert isinstance(client.pending_approvals, dict)

    @pytest.mark.asyncio
    async def test_connect_success(self, client):
        """Test successful connection to Temporal server."""
        with patch("utils.temporal_client.TEMPORAL_AVAILABLE", True):
            with patch("utils.temporal_client.Client.connect") as mock_connect:
                mock_connect.return_value = MagicMock()

                result = await client.connect()

                assert result is True
                assert client.client is not None
                mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure(self, client):
        """Test connection failure handling."""
        with patch("utils.temporal_client.TEMPORAL_AVAILABLE", True):
            with patch("utils.temporal_client.Client.connect", side_effect=Exception("Connection failed")):
                result = await client.connect()

                assert result is False
                assert client.client is None

    @pytest.mark.asyncio
    async def test_connect_temporal_unavailable(self, client):
        """Test behavior when Temporal SDK is not available."""
        with patch("utils.temporal_client.TEMPORAL_AVAILABLE", False):
            result = await client.connect()

            assert result is False
            assert client.client is None

    @pytest.mark.asyncio
    async def test_start_workflow_success(self, client, mock_temporal_client):
        """Test successful workflow execution."""
        client.client = mock_temporal_client

        workflow_args = {"test_param": "test_value"}
        result = await client.start_workflow(
            MockWorkflow,
            workflow_args,
            workflow_id="test-workflow-123"
        )

        assert isinstance(result, WorkflowExecutionResult)
        assert result.workflow_id == "test-workflow-123"
        assert result.status == "completed"
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.execution_time_seconds is not None

    @pytest.mark.asyncio
    async def test_start_workflow_failure(self, client, mock_temporal_client):
        """Test workflow execution failure handling."""
        # Configure mock to simulate workflow failure
        mock_handle = mock_temporal_client.start_workflow.return_value
        mock_handle.result = AsyncMock(side_effect=Exception("Workflow execution failed"))

        client.client = mock_temporal_client

        result = await client.start_workflow(
            MockWorkflow,
            {"should_fail": True}
        )

        assert isinstance(result, WorkflowExecutionResult)
        assert result.status == "failed"
        assert result.error is not None
        assert "Workflow execution failed" in result.error

    @pytest.mark.asyncio
    async def test_fallback_workflow_execution(self, client):
        """Test fallback workflow execution when Temporal is unavailable."""
        # Don't set client to simulate unavailable Temporal

        result = await client.start_workflow(
            MockWorkflow,
            {"test_param": "fallback_test"}
        )

        assert isinstance(result, WorkflowExecutionResult)
        assert result.status == "completed_fallback"
        assert "fallback mode" in result.result["message"]

    @pytest.mark.asyncio
    async def test_request_human_approval(self, client):
        """Test human approval request creation."""
        workflow_id = "test-workflow-123"

        approval_id = await client.request_human_approval(
            workflow_id=workflow_id,
            stage="deploy",
            description="Approve deployment to production",
            context={"environment": "production"},
            timeout_seconds=3600
        )

        assert approval_id is not None
        assert approval_id in client.pending_approvals

        request = client.pending_approvals[approval_id]
        assert isinstance(request, HumanApprovalRequest)
        assert request.workflow_id == workflow_id
        assert request.stage == "deploy"
        assert request.description == "Approve deployment to production"
        assert request.context["environment"] == "production"

    @pytest.mark.asyncio
    async def test_submit_approval_decision_approve(self, client):
        """Test submitting an approval decision (approve)."""
        # First create an approval request
        approval_id = await client.request_human_approval(
            workflow_id="test-workflow",
            stage="test",
            description="Test approval",
            context={}
        )

        # Submit approval decision
        success = await client.submit_approval_decision(
            approval_id=approval_id,
            approved=True,
            feedback="Looks good to proceed",
            decided_by="test_user"
        )

        assert success is True
        assert approval_id not in client.pending_approvals

        # Check decision was stored
        decision = await client.get_approval_status(approval_id)
        assert isinstance(decision, ApprovalDecision)
        assert decision.approved is True
        assert decision.feedback == "Looks good to proceed"
        assert decision.decided_by == "test_user"

    @pytest.mark.asyncio
    async def test_submit_approval_decision_reject(self, client):
        """Test submitting an approval decision (reject)."""
        # Create approval request
        approval_id = await client.request_human_approval(
            workflow_id="test-workflow",
            stage="test",
            description="Test approval",
            context={}
        )

        # Submit rejection
        success = await client.submit_approval_decision(
            approval_id=approval_id,
            approved=False,
            feedback="Issues found, cannot approve",
            decided_by="test_user"
        )

        assert success is True

        decision = await client.get_approval_status(approval_id)
        assert decision.approved is False
        assert decision.feedback == "Issues found, cannot approve"

    @pytest.mark.asyncio
    async def test_submit_approval_decision_not_found(self, client):
        """Test submitting approval decision for non-existent request."""
        success = await client.submit_approval_decision(
            approval_id="non-existent-approval",
            approved=True
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_list_pending_approvals(self, client):
        """Test listing pending approval requests."""
        # Create multiple approval requests
        await client.request_human_approval(
            workflow_id="workflow1",
            stage="stage1",
            description="First approval",
            context={}
        )

        await client.request_human_approval(
            workflow_id="workflow2",
            stage="stage2",
            description="Second approval",
            context={}
        )

        await client.request_human_approval(
            workflow_id="workflow1",
            stage="stage3",
            description="Third approval",
            context={}
        )

        # List all pending approvals
        all_pending = await client.list_pending_approvals()
        assert len(all_pending) == 3

        # List approvals for specific workflow
        workflow1_approvals = await client.list_pending_approvals("workflow1")
        assert len(workflow1_approvals) == 2

        # Verify approvals belong to correct workflow
        for approval in workflow1_approvals:
            assert approval.workflow_id == "workflow1"

    @pytest.mark.asyncio
    async def test_cancel_workflow(self, client, mock_temporal_client):
        """Test workflow cancellation."""
        client.client = mock_temporal_client

        # Mock workflow handle
        mock_handle = MagicMock()
        mock_handle.cancel = AsyncMock()
        mock_temporal_client.get_workflow_handle.return_value = mock_handle

        result = await client.cancel_workflow("test-workflow", "User cancelled")

        assert result is True
        mock_temporal_client.get_workflow_handle.assert_called_once_with("test-workflow")
        mock_handle.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_workflow_not_connected(self, client):
        """Test workflow cancellation when not connected to Temporal."""
        result = await client.cancel_workflow("test-workflow", "Test cancellation")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_workflow_status(self, client, mock_temporal_client):
        """Test getting workflow status."""
        client.client = mock_temporal_client

        mock_handle = MagicMock()
        mock_temporal_client.get_workflow_handle.return_value = mock_handle

        status = await client.get_workflow_status("test-workflow")

        assert status is not None
        assert status["workflow_id"] == "test-workflow"
        assert "status" in status
        assert "started_at" in status

    @pytest.mark.asyncio
    async def test_disconnect(self, client):
        """Test clean disconnection."""
        # Mock worker and client
        client.worker = MagicMock()
        client.worker.shutdown = MagicMock()

        await client.disconnect()

        client.worker.shutdown.assert_called_once()

    def test_get_temporal_client_singleton(self):
        """Test that get_temporal_client returns singleton instance."""
        client1 = get_temporal_client()
        client2 = get_temporal_client()

        assert client1 is client2
        assert isinstance(client1, TemporalWorkflowClient)


class TestBaseWorkflow:
    """Test suite for BaseWorkflow class."""

    @pytest.fixture
    def workflow(self):
        """Create test workflow instance."""
        return BaseWorkflow()

    def test_workflow_initialization(self, workflow):
        """Test workflow initialization."""
        assert workflow.context is None

    def test_setup_context(self, workflow):
        """Test workflow context setup."""
        workflow_id = "test-workflow-123"
        workflow.setup_context(workflow_id)

        assert workflow.context is not None
        assert workflow.context.workflow_id == workflow_id
        assert isinstance(workflow.context.signals, dict)
        assert isinstance(workflow.context.activities_completed, list)
        assert workflow.context.start_time is not None

    def test_orchestrate_not_implemented(self, workflow):
        """Test that orchestrate method raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            asyncio.run(workflow.orchestrate({}))

    @pytest.mark.asyncio
    async def test_wait_for_approval_timeout(self, workflow):
        """Test wait_for_approval with timeout."""
        workflow.setup_context("test-workflow")

        # This should timeout quickly since no approval will be submitted
        result = await asyncio.wait_for(
            workflow.wait_for_approval(
                stage="test",
                description="Test approval",
                context={},
                timeout_seconds=1  # Very short timeout
            ),
            timeout=5.0  # Test timeout
        )

        # Should return False due to timeout
        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_approval_context_not_initialized(self, workflow):
        """Test wait_for_approval without initialized context."""
        with pytest.raises(RuntimeError, match="Workflow context not initialized"):
            await workflow.wait_for_approval(
                stage="test",
                description="Test approval",
                context={}
            )


@pytest.mark.integration
class TestTemporalIntegration:
    """Integration tests requiring actual Temporal setup."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_execution(self):
        """Test complete workflow execution flow."""
        # This test would require a running Temporal server
        # Skip in unit tests, run separately in integration environment
        pytest.skip("Requires running Temporal server for integration testing")

    @pytest.mark.asyncio
    async def test_human_approval_integration(self):
        """Test human approval flow integration."""
        pytest.skip("Requires running Temporal server for integration testing")
