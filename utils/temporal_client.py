"""
Temporal Workflow Orchestration Client

This module provides integration with Temporal for long-running, fault-tolerant
multi-agent workflows with human approval gates and saga patterns.

Features:
- Temporal workflow client configuration
- Workflow execution and monitoring
- Human approval integration
- Saga pattern support for distributed transactions
- Integration with existing Redis, NATS, and Kafka infrastructure
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Optional, TypeVar
from uuid import uuid4

from pydantic import BaseModel

# Temporal dependencies (optional)
try:
    from temporalio import workflow
    from temporalio.client import Client, TLSConfig
    from temporalio.common import RetryPolicy
    from temporalio.worker import Worker

    TEMPORAL_AVAILABLE = True
except ImportError:
    # Fallback implementations when Temporal is not available
    TEMPORAL_AVAILABLE = False
    workflow = None
    Client = None
    Worker = None
    RetryPolicy = None
    TLSConfig = None

from utils.event_bus import get_event_bus
from utils.storage_backend import get_storage_backend

logger = logging.getLogger(__name__)

# Type definitions
WorkflowType = TypeVar("WorkflowType")


class WorkflowExecutionResult(BaseModel):
    """Result of a workflow execution."""
    workflow_id: str
    run_id: str
    status: str
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None


class HumanApprovalRequest(BaseModel):
    """Request for human approval in a workflow."""
    workflow_id: str
    approval_id: str
    stage: str
    description: str
    context: dict[str, Any]
    requested_at: datetime
    timeout_seconds: int
    callback_url: Optional[str] = None


class ApprovalDecision(BaseModel):
    """Human approval decision."""
    approval_id: str
    approved: bool
    feedback: Optional[str] = None
    decided_at: datetime
    decided_by: Optional[str] = None


class TemporalWorkflowClient:
    """Client for managing Temporal workflows with agent orchestration features."""

    def __init__(
        self,
        temporal_address: str = "localhost:7233",
        namespace: str = "default",
        task_queue: str = "zen-agent-workflows",
        tls_config: Optional[TLSConfig] = None
    ):
        """Initialize the Temporal workflow client."""
        self.temporal_address = temporal_address
        self.namespace = namespace
        self.task_queue = task_queue
        self.tls_config = tls_config
        self.client: Optional[Client] = None
        self.worker: Optional[Worker] = None

        # Integration components
        self.storage = get_storage_backend()
        self.event_bus = get_event_bus()

        # Approval management
        self.pending_approvals: dict[str, HumanApprovalRequest] = {}

        logger.info(f"TemporalWorkflowClient initialized - temporal_available: {TEMPORAL_AVAILABLE}")

    async def connect(self) -> bool:
        """Connect to Temporal server."""
        if not TEMPORAL_AVAILABLE:
            logger.warning("Temporal SDK not available - using fallback implementation")
            return False

        try:
            self.client = await Client.connect(
                self.temporal_address,
                namespace=self.namespace,
                tls=self.tls_config
            )
            logger.info(f"Connected to Temporal server at {self.temporal_address}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Temporal server: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Temporal server."""
        if self.worker:
            self.worker.shutdown()
        # Client doesn't need explicit cleanup
        logger.info("Disconnected from Temporal server")

    async def start_workflow(
        self,
        workflow_class: type[WorkflowType],
        workflow_args: dict[str, Any],
        workflow_id: Optional[str] = None,
        timeout_seconds: int = 3600,
        retry_policy: Optional[RetryPolicy] = None
    ) -> WorkflowExecutionResult:
        """Start a new workflow execution."""
        if not self.client:
            return await self._fallback_workflow_execution(workflow_class, workflow_args, workflow_id)

        workflow_id = workflow_id or f"zen-workflow-{uuid4()}"
        started_at = datetime.utcnow()

        try:
            # Publish workflow started event
            await self.event_bus.publish({
                "event": "workflow_started",
                "workflow_id": workflow_id,
                "workflow_class": workflow_class.__name__,
                "args": workflow_args,
                "timestamp": started_at.isoformat()
            })

            # Start workflow execution
            handle = await self.client.start_workflow(
                workflow_class.orchestrate,
                workflow_args,
                id=workflow_id,
                task_queue=self.task_queue,
                execution_timeout=timedelta(seconds=timeout_seconds),
                retry_policy=retry_policy or RetryPolicy(
                    initial_interval=timedelta(seconds=1),
                    maximum_attempts=3,
                    maximum_interval=timedelta(seconds=60)
                )
            )

            # Store workflow metadata
            await self._store_workflow_metadata(workflow_id, {
                "workflow_class": workflow_class.__name__,
                "started_at": started_at.isoformat(),
                "run_id": handle.first_execution_run_id,
                "args": workflow_args
            })

            # Wait for completion or timeout
            try:
                result = await handle.result()
                completed_at = datetime.utcnow()
                execution_time = (completed_at - started_at).total_seconds()

                # Publish completion event
                await self.event_bus.publish({
                    "event": "workflow_completed",
                    "workflow_id": workflow_id,
                    "status": "completed",
                    "execution_time_seconds": execution_time,
                    "timestamp": completed_at.isoformat()
                })

                return WorkflowExecutionResult(
                    workflow_id=workflow_id,
                    run_id=handle.first_execution_run_id,
                    status="completed",
                    result=result,
                    started_at=started_at,
                    completed_at=completed_at,
                    execution_time_seconds=execution_time
                )

            except Exception as execution_error:
                completed_at = datetime.utcnow()
                execution_time = (completed_at - started_at).total_seconds()

                # Publish error event
                await self.event_bus.publish({
                    "event": "workflow_failed",
                    "workflow_id": workflow_id,
                    "error": str(execution_error),
                    "execution_time_seconds": execution_time,
                    "timestamp": completed_at.isoformat()
                })

                return WorkflowExecutionResult(
                    workflow_id=workflow_id,
                    run_id=handle.first_execution_run_id,
                    status="failed",
                    error=str(execution_error),
                    started_at=started_at,
                    completed_at=completed_at,
                    execution_time_seconds=execution_time
                )

        except Exception as e:
            logger.error(f"Failed to start workflow {workflow_id}: {e}")
            return WorkflowExecutionResult(
                workflow_id=workflow_id,
                run_id="",
                status="failed",
                error=str(e),
                started_at=started_at
            )

    async def _fallback_workflow_execution(
        self,
        workflow_class: type[WorkflowType],
        workflow_args: dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> WorkflowExecutionResult:
        """Fallback workflow execution when Temporal is not available."""
        workflow_id = workflow_id or f"fallback-workflow-{uuid4()}"
        started_at = datetime.utcnow()

        logger.warning(f"Using fallback execution for workflow {workflow_id}")

        try:
            # Publish workflow started event
            await self.event_bus.publish({
                "event": "workflow_started_fallback",
                "workflow_id": workflow_id,
                "workflow_class": workflow_class.__name__,
                "args": workflow_args,
                "timestamp": started_at.isoformat()
            })

            # Simulate workflow execution
            await asyncio.sleep(1)  # Brief delay to simulate processing

            completed_at = datetime.utcnow()
            execution_time = (completed_at - started_at).total_seconds()

            result = {
                "status": "completed_fallback",
                "message": "Workflow executed in fallback mode (Temporal not available)",
                "args": workflow_args
            }

            # Publish completion event
            await self.event_bus.publish({
                "event": "workflow_completed_fallback",
                "workflow_id": workflow_id,
                "status": "completed_fallback",
                "execution_time_seconds": execution_time,
                "timestamp": completed_at.isoformat()
            })

            return WorkflowExecutionResult(
                workflow_id=workflow_id,
                run_id="fallback",
                status="completed_fallback",
                result=result,
                started_at=started_at,
                completed_at=completed_at,
                execution_time_seconds=execution_time
            )

        except Exception as e:
            logger.error(f"Fallback workflow execution failed: {e}")
            return WorkflowExecutionResult(
                workflow_id=workflow_id,
                run_id="fallback",
                status="failed",
                error=str(e),
                started_at=started_at
            )

    async def request_human_approval(
        self,
        workflow_id: str,
        stage: str,
        description: str,
        context: dict[str, Any],
        timeout_seconds: int = 24 * 60 * 60,  # 24 hours default
        callback_url: Optional[str] = None
    ) -> str:
        """Request human approval for a workflow stage."""
        approval_id = f"approval-{uuid4()}"

        approval_request = HumanApprovalRequest(
            workflow_id=workflow_id,
            approval_id=approval_id,
            stage=stage,
            description=description,
            context=context,
            requested_at=datetime.utcnow(),
            timeout_seconds=timeout_seconds,
            callback_url=callback_url
        )

        # Store approval request
        self.pending_approvals[approval_id] = approval_request
        await self._store_approval_request(approval_request)

        # Publish approval request event
        await self.event_bus.publish({
            "event": "human_approval_requested",
            "workflow_id": workflow_id,
            "approval_id": approval_id,
            "stage": stage,
            "description": description,
            "context": context,
            "timeout_seconds": timeout_seconds,
            "timestamp": approval_request.requested_at.isoformat()
        })

        logger.info(f"Human approval requested for workflow {workflow_id}, stage {stage}")
        return approval_id

    async def submit_approval_decision(
        self,
        approval_id: str,
        approved: bool,
        feedback: Optional[str] = None,
        decided_by: Optional[str] = None
    ) -> bool:
        """Submit a human approval decision."""
        if approval_id not in self.pending_approvals:
            logger.warning(f"Approval {approval_id} not found in pending approvals")
            return False

        approval_request = self.pending_approvals[approval_id]

        decision = ApprovalDecision(
            approval_id=approval_id,
            approved=approved,
            feedback=feedback,
            decided_at=datetime.utcnow(),
            decided_by=decided_by
        )

        # Remove from pending approvals
        del self.pending_approvals[approval_id]

        # Store decision
        await self._store_approval_decision(decision)

        # Publish decision event
        await self.event_bus.publish({
            "event": "human_approval_decided",
            "workflow_id": approval_request.workflow_id,
            "approval_id": approval_id,
            "approved": approved,
            "feedback": feedback,
            "decided_by": decided_by,
            "timestamp": decision.decided_at.isoformat()
        })

        logger.info(f"Approval decision submitted for {approval_id}: {'approved' if approved else 'rejected'}")
        return True

    async def get_approval_status(self, approval_id: str) -> Optional[ApprovalDecision]:
        """Get the status of an approval request."""
        # Check if decision was already made
        decision_key = f"approval_decision:{approval_id}"
        decision_data = self.storage.get(decision_key)
        if decision_data:
            return ApprovalDecision.model_validate_json(decision_data)

        # Check if still pending
        if approval_id in self.pending_approvals:
            return None  # Still pending

        # Not found
        logger.warning(f"Approval {approval_id} not found")
        return None

    async def list_pending_approvals(
        self,
        workflow_id: Optional[str] = None
    ) -> list[HumanApprovalRequest]:
        """List pending approval requests."""
        approvals = list(self.pending_approvals.values())

        if workflow_id:
            approvals = [a for a in approvals if a.workflow_id == workflow_id]

        return approvals

    async def cancel_workflow(self, workflow_id: str, reason: str = "User cancelled") -> bool:
        """Cancel a running workflow."""
        if not self.client:
            logger.warning(f"Cannot cancel workflow {workflow_id} - Temporal client not available")
            return False

        try:
            handle = self.client.get_workflow_handle(workflow_id)
            await handle.cancel()

            # Publish cancellation event
            await self.event_bus.publish({
                "event": "workflow_cancelled",
                "workflow_id": workflow_id,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            })

            logger.info(f"Workflow {workflow_id} cancelled: {reason}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel workflow {workflow_id}: {e}")
            return False

    async def get_workflow_status(self, workflow_id: str) -> Optional[dict[str, Any]]:
        """Get the status of a workflow."""
        if not self.client:
            return None

        try:
            self.client.get_workflow_handle(workflow_id)
            # This would return workflow status in a real implementation
            return {
                "workflow_id": workflow_id,
                "status": "running",  # Placeholder
                "started_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get workflow status for {workflow_id}: {e}")
            return None

    async def _store_workflow_metadata(self, workflow_id: str, metadata: dict[str, Any]):
        """Store workflow metadata in Redis."""
        key = f"workflow_metadata:{workflow_id}"
        self.storage.setex(key, 24 * 60 * 60, json.dumps(metadata))  # 24 hour TTL

    async def _store_approval_request(self, request: HumanApprovalRequest):
        """Store approval request in Redis."""
        key = f"approval_request:{request.approval_id}"
        self.storage.setex(
            key,
            request.timeout_seconds,
            request.model_dump_json()
        )

    async def _store_approval_decision(self, decision: ApprovalDecision):
        """Store approval decision in Redis."""
        key = f"approval_decision:{decision.approval_id}"
        self.storage.setex(key, 24 * 60 * 60, decision.model_dump_json())  # 24 hour TTL


# Global client instance
_temporal_client: Optional[TemporalWorkflowClient] = None


def get_temporal_client() -> TemporalWorkflowClient:
    """Get the global Temporal workflow client."""
    global _temporal_client
    if _temporal_client is None:
        temporal_address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
        namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
        task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "zen-agent-workflows")

        _temporal_client = TemporalWorkflowClient(
            temporal_address=temporal_address,
            namespace=namespace,
            task_queue=task_queue
        )

    return _temporal_client


# Decorators and utilities for workflow definitions
def workflow_activity(func):
    """Decorator to mark a function as a workflow activity."""
    func._is_workflow_activity = True
    return func


def workflow_signal(func):
    """Decorator to mark a function as a workflow signal handler."""
    func._is_workflow_signal = True
    return func


class WorkflowContext:
    """Context object available during workflow execution."""

    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.signals: dict[str, Any] = {}
        self.activities_completed: list[str] = []
        self.start_time = datetime.utcnow()

    def set_signal(self, signal_name: str, value: Any):
        """Set a signal value."""
        self.signals[signal_name] = value

    def get_signal(self, signal_name: str) -> Optional[Any]:
        """Get a signal value."""
        return self.signals.get(signal_name)

    def mark_activity_completed(self, activity_name: str):
        """Mark an activity as completed."""
        if activity_name not in self.activities_completed:
            self.activities_completed.append(activity_name)


# Workflow base class
class BaseWorkflow:
    """Base class for Temporal workflows."""

    def __init__(self):
        self.context: Optional[WorkflowContext] = None

    async def orchestrate(self, workflow_args: dict[str, Any]) -> dict[str, Any]:
        """Main workflow orchestration method - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement orchestrate method")

    def setup_context(self, workflow_id: str):
        """Setup workflow context."""
        self.context = WorkflowContext(workflow_id)

    async def wait_for_approval(
        self,
        stage: str,
        description: str,
        context: dict[str, Any],
        timeout_seconds: int = 24 * 60 * 60
    ) -> bool:
        """Wait for human approval during workflow execution."""
        if not self.context:
            raise RuntimeError("Workflow context not initialized")

        client = get_temporal_client()
        approval_id = await client.request_human_approval(
            workflow_id=self.context.workflow_id,
            stage=stage,
            description=description,
            context=context,
            timeout_seconds=timeout_seconds
        )

        # Wait for approval decision
        timeout_time = datetime.utcnow() + timedelta(seconds=timeout_seconds)

        while datetime.utcnow() < timeout_time:
            decision = await client.get_approval_status(approval_id)
            if decision:
                return decision.approved

            await asyncio.sleep(10)  # Check every 10 seconds

        # Timeout reached
        logger.warning(f"Approval timeout reached for {approval_id}")
        return False


logger.info("Temporal workflow client module initialized")
