"""
Workflow Orchestration Tool

This tool provides a unified interface for managing Temporal-based workflows
including multi-agent projects, human approval flows, and saga transactions.
Integrates with the existing MCP tool infrastructure.

Features:
- Start multi-agent project workflows
- Request human approvals
- Execute saga transactions
- Monitor workflow status
- Manage workflow lifecycle
"""

import logging
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from tools.shared.base_tool import BaseTool
from tools.shared.schema_builders import build_tool_schema
from utils.temporal_client import get_temporal_client
from workflows.approval_workflow import HumanApprovalWorkflow, delegate_approval, submit_approval_decision
from workflows.integrations import get_kafka_integration, get_nats_integration, get_workflow_state_manager
from workflows.multi_agent_workflow import ProjectSpec, start_multi_agent_project
from workflows.saga_workflow import start_distributed_saga
from workflows.workflow_monitor import get_workflow_monitor

logger = logging.getLogger(__name__)


class WorkflowRequest(BaseModel):
    """Request to start a new workflow."""
    workflow_type: str = Field(description="Type of workflow (multi_agent_project, approval, saga)")
    workflow_spec: dict[str, Any] = Field(description="Workflow specification")
    config: Optional[dict[str, Any]] = Field(default=None, description="Workflow configuration")


class ApprovalRequest(BaseModel):
    """Request for human approval."""
    approval_id: Optional[str] = Field(default=None, description="Existing approval ID (for decisions)")
    action: str = Field(description="Action to take (request, approve, reject, delegate)")
    workflow_id: Optional[str] = Field(default=None, description="Workflow ID for new requests")
    stage: Optional[str] = Field(default=None, description="Approval stage")
    description: Optional[str] = Field(default=None, description="Approval description")
    context: Optional[dict[str, Any]] = Field(default=None, description="Approval context")
    feedback: Optional[str] = Field(default=None, description="Approval feedback")
    approver_id: Optional[str] = Field(default=None, description="Approver user ID")
    delegate_to: Optional[str] = Field(default=None, description="User to delegate approval to")


class WorkflowStatusRequest(BaseModel):
    """Request for workflow status."""
    workflow_id: str = Field(description="Workflow ID to check status")
    include_details: bool = Field(default=False, description="Include detailed execution information")


class WorkflowMonitorRequest(BaseModel):
    """Request for workflow monitoring operations."""
    action: str = Field(description="Monitoring action (start, stop, dashboard, health_check)")
    workflow_id: Optional[str] = Field(default=None, description="Specific workflow ID")
    config: Optional[dict[str, Any]] = Field(default=None, description="Monitoring configuration")


class WorkflowOrchestratorTool(BaseTool):
    """
    Workflow Orchestration Tool

    Provides comprehensive workflow management capabilities including:
    - Multi-agent project orchestration
    - Human approval workflows
    - Saga pattern distributed transactions
    - Workflow monitoring and recovery
    """

    def __init__(self):
        super().__init__(
            name="workflow_orchestrator",
            description="Orchestrate complex workflows with multi-agent coordination, human approvals, and distributed transactions"
        )

        # Initialize clients and integrations
        self.temporal_client = get_temporal_client()
        self.workflow_monitor = get_workflow_monitor()
        self.state_manager = get_workflow_state_manager()
        self.nats_integration = get_nats_integration()
        self.kafka_integration = get_kafka_integration()

        # Track active workflows
        self.active_workflows: dict[str, dict[str, Any]] = {}

    async def execute(self, args: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Execute workflow orchestration operation."""
        try:
            operation = args.get("operation", "start_workflow")

            if operation == "start_workflow":
                return await self._start_workflow(args)
            elif operation == "approval":
                return await self._handle_approval(args)
            elif operation == "status":
                return await self._get_workflow_status(args)
            elif operation == "monitor":
                return await self._handle_monitoring(args)
            elif operation == "list_workflows":
                return await self._list_workflows(args)
            elif operation == "cancel_workflow":
                return await self._cancel_workflow(args)
            else:
                return {
                    "success": False,
                    "error": f"Unknown operation: {operation}",
                    "available_operations": [
                        "start_workflow", "approval", "status", "monitor",
                        "list_workflows", "cancel_workflow"
                    ]
                }

        except Exception as e:
            logger.error(f"Workflow orchestration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "operation": args.get("operation", "unknown")
            }

    async def _start_workflow(self, args: dict[str, Any]) -> dict[str, Any]:
        """Start a new workflow."""
        workflow_request = WorkflowRequest(**args)

        try:
            # Connect to Temporal if not already connected
            if not self.temporal_client.client:
                connected = await self.temporal_client.connect()
                if not connected:
                    logger.warning("Temporal not available, using fallback mode")

            # Route to appropriate workflow type
            if workflow_request.workflow_type == "multi_agent_project":
                result = await self._start_multi_agent_project(workflow_request)
            elif workflow_request.workflow_type == "approval":
                result = await self._start_approval_workflow(workflow_request)
            elif workflow_request.workflow_type == "saga":
                result = await self._start_saga_workflow(workflow_request)
            else:
                return {
                    "success": False,
                    "error": f"Unknown workflow type: {workflow_request.workflow_type}",
                    "supported_types": ["multi_agent_project", "approval", "saga"]
                }

            # Register with monitoring
            if result.get("success") and result.get("workflow_id"):
                await self.workflow_monitor.register_workflow(
                    workflow_id=result["workflow_id"],
                    workflow_type=workflow_request.workflow_type,
                    metadata=workflow_request.config or {}
                )

                # Track active workflow
                self.active_workflows[result["workflow_id"]] = {
                    "type": workflow_request.workflow_type,
                    "started_at": datetime.utcnow().isoformat(),
                    "spec": workflow_request.workflow_spec,
                    "config": workflow_request.config
                }

            return result

        except Exception as e:
            logger.error(f"Failed to start {workflow_request.workflow_type} workflow: {e}")
            return {
                "success": False,
                "error": str(e),
                "workflow_type": workflow_request.workflow_type
            }

    async def _start_multi_agent_project(self, request: WorkflowRequest) -> dict[str, Any]:
        """Start a multi-agent project workflow."""
        try:
            # Validate project spec
            project_spec = ProjectSpec(**request.workflow_spec)

            # Start workflow
            result = await start_multi_agent_project(
                project_spec=project_spec,
                config=request.config
            )

            return {
                "success": True,
                "workflow_type": "multi_agent_project",
                "workflow_id": result.get("workflow_id"),
                "project_id": project_spec.project_id,
                "status": result.get("status"),
                "estimated_duration_hours": project_spec.estimated_duration_hours,
                "agents_required": [agent.value for agent in project_spec.agents_required],
                "result": result
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to start multi-agent project: {str(e)}"
            }

    async def _start_approval_workflow(self, request: WorkflowRequest) -> dict[str, Any]:
        """Start a human approval workflow."""
        try:
            workflow_spec = request.workflow_spec

            # Create approval workflow
            workflow = HumanApprovalWorkflow()

            # Start workflow execution
            result = await self.temporal_client.start_workflow(
                workflow_class=workflow.__class__,
                workflow_args={
                    "approval_request": workflow_spec,
                    "config": request.config or {}
                },
                timeout_seconds=workflow_spec.get("timeout_minutes", 1440) * 60
            )

            return {
                "success": True,
                "workflow_type": "approval",
                "workflow_id": result.workflow_id,
                "approval_title": workflow_spec.get("title"),
                "required_approvers": workflow_spec.get("required_approvers", []),
                "timeout_minutes": workflow_spec.get("timeout_minutes", 1440),
                "result": result.model_dump()
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to start approval workflow: {str(e)}"
            }

    async def _start_saga_workflow(self, request: WorkflowRequest) -> dict[str, Any]:
        """Start a saga transaction workflow."""
        try:
            saga_definition = request.workflow_spec
            coordination_mode = request.config.get("coordination_mode", "orchestration")

            # Start distributed saga
            result = await start_distributed_saga(
                saga_definition=saga_definition,
                coordination_mode=coordination_mode,
                config=request.config
            )

            return {
                "success": True,
                "workflow_type": "saga",
                "workflow_id": result.get("saga_id"),
                "coordination_mode": coordination_mode,
                "total_steps": len(saga_definition.get("steps", [])),
                "result": result
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to start saga workflow: {str(e)}"
            }

    async def _handle_approval(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle approval-related operations."""
        try:
            approval_request = ApprovalRequest(**args)

            if approval_request.action == "request":
                return await self._request_approval(approval_request)
            elif approval_request.action == "approve":
                return await self._approve_request(approval_request)
            elif approval_request.action == "reject":
                return await self._reject_request(approval_request)
            elif approval_request.action == "delegate":
                return await self._delegate_approval(approval_request)
            elif approval_request.action == "list_pending":
                return await self._list_pending_approvals(approval_request)
            else:
                return {
                    "success": False,
                    "error": f"Unknown approval action: {approval_request.action}",
                    "available_actions": ["request", "approve", "reject", "delegate", "list_pending"]
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Approval operation failed: {str(e)}"
            }

    async def _request_approval(self, request: ApprovalRequest) -> dict[str, Any]:
        """Request human approval."""
        if not request.workflow_id or not request.stage or not request.description:
            return {
                "success": False,
                "error": "workflow_id, stage, and description are required for approval requests"
            }

        approval_id = await self.temporal_client.request_human_approval(
            workflow_id=request.workflow_id,
            stage=request.stage,
            description=request.description,
            context=request.context or {},
            timeout_seconds=3600  # 1 hour default
        )

        return {
            "success": True,
            "action": "approval_requested",
            "approval_id": approval_id,
            "workflow_id": request.workflow_id,
            "stage": request.stage
        }

    async def _approve_request(self, request: ApprovalRequest) -> dict[str, Any]:
        """Approve a pending request."""
        if not request.approval_id or not request.approver_id:
            return {
                "success": False,
                "error": "approval_id and approver_id are required for approval"
            }

        success = await submit_approval_decision(
            request_id=request.approval_id,
            approver_id=request.approver_id,
            decision="approve",
            feedback=request.feedback
        )

        return {
            "success": success,
            "action": "approved",
            "approval_id": request.approval_id,
            "approver_id": request.approver_id,
            "feedback": request.feedback
        }

    async def _reject_request(self, request: ApprovalRequest) -> dict[str, Any]:
        """Reject a pending request."""
        if not request.approval_id or not request.approver_id:
            return {
                "success": False,
                "error": "approval_id and approver_id are required for rejection"
            }

        success = await submit_approval_decision(
            request_id=request.approval_id,
            approver_id=request.approver_id,
            decision="reject",
            feedback=request.feedback
        )

        return {
            "success": success,
            "action": "rejected",
            "approval_id": request.approval_id,
            "approver_id": request.approver_id,
            "feedback": request.feedback
        }

    async def _delegate_approval(self, request: ApprovalRequest) -> dict[str, Any]:
        """Delegate approval to another user."""
        if not request.approval_id or not request.approver_id or not request.delegate_to:
            return {
                "success": False,
                "error": "approval_id, approver_id, and delegate_to are required for delegation"
            }

        success = await delegate_approval(
            request_id=request.approval_id,
            from_approver=request.approver_id,
            to_approver=request.delegate_to,
            reason=request.feedback
        )

        return {
            "success": success,
            "action": "delegated",
            "approval_id": request.approval_id,
            "from_approver": request.approver_id,
            "to_approver": request.delegate_to
        }

    async def _list_pending_approvals(self, request: ApprovalRequest) -> dict[str, Any]:
        """List pending approval requests."""
        pending_approvals = await self.temporal_client.list_pending_approvals(
            workflow_id=request.workflow_id
        )

        return {
            "success": True,
            "action": "list_pending",
            "workflow_id": request.workflow_id,
            "pending_approvals": [approval.model_dump() for approval in pending_approvals],
            "count": len(pending_approvals)
        }

    async def _get_workflow_status(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get workflow status and details."""
        try:
            status_request = WorkflowStatusRequest(**args)

            # Get status from Temporal
            temporal_status = await self.temporal_client.get_workflow_status(status_request.workflow_id)

            # Get monitoring data
            health_check = await self.workflow_monitor.perform_health_check(status_request.workflow_id)

            # Get workflow state if available
            workflow_state = await self.state_manager.get_workflow_state(status_request.workflow_id)

            result = {
                "success": True,
                "workflow_id": status_request.workflow_id,
                "temporal_status": temporal_status,
                "health_status": health_check.model_dump(),
                "workflow_state": workflow_state
            }

            # Add active workflow details if available
            if status_request.workflow_id in self.active_workflows:
                result["active_workflow"] = self.active_workflows[status_request.workflow_id]

            # Add detailed monitoring data if requested
            if status_request.include_details:
                dashboard_data = await self.workflow_monitor.get_workflow_dashboard_data(
                    status_request.workflow_id
                )
                result["detailed_monitoring"] = dashboard_data

            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get workflow status: {str(e)}"
            }

    async def _handle_monitoring(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle workflow monitoring operations."""
        try:
            monitor_request = WorkflowMonitorRequest(**args)

            if monitor_request.action == "start":
                await self.workflow_monitor.start_monitoring()
                return {"success": True, "action": "monitoring_started"}

            elif monitor_request.action == "stop":
                await self.workflow_monitor.stop_monitoring()
                return {"success": True, "action": "monitoring_stopped"}

            elif monitor_request.action == "dashboard":
                dashboard_data = await self.workflow_monitor.get_workflow_dashboard_data(
                    monitor_request.workflow_id
                )
                return {
                    "success": True,
                    "action": "dashboard",
                    "data": dashboard_data
                }

            elif monitor_request.action == "health_check":
                if not monitor_request.workflow_id:
                    return {
                        "success": False,
                        "error": "workflow_id required for health check"
                    }

                health_check = await self.workflow_monitor.perform_health_check(
                    monitor_request.workflow_id
                )
                return {
                    "success": True,
                    "action": "health_check",
                    "workflow_id": monitor_request.workflow_id,
                    "health": health_check.model_dump()
                }

            else:
                return {
                    "success": False,
                    "error": f"Unknown monitoring action: {monitor_request.action}",
                    "available_actions": ["start", "stop", "dashboard", "health_check"]
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Monitoring operation failed: {str(e)}"
            }

    async def _list_workflows(self, args: dict[str, Any]) -> dict[str, Any]:
        """List active workflows."""
        try:
            # Get workflows from monitoring system
            dashboard_data = await self.workflow_monitor.get_workflow_dashboard_data()

            # Combine with active workflows tracked by this tool
            active_workflows = []
            for workflow_id, details in self.active_workflows.items():
                workflow_info = {
                    "workflow_id": workflow_id,
                    "type": details["type"],
                    "started_at": details["started_at"],
                    "status": "active"
                }
                active_workflows.append(workflow_info)

            return {
                "success": True,
                "active_workflows": active_workflows,
                "monitoring_data": dashboard_data.get("workflows", []),
                "total_active": len(active_workflows),
                "monitoring_status": dashboard_data.get("monitoring_status")
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to list workflows: {str(e)}"
            }

    async def _cancel_workflow(self, args: dict[str, Any]) -> dict[str, Any]:
        """Cancel a running workflow."""
        try:
            workflow_id = args.get("workflow_id")
            reason = args.get("reason", "User requested cancellation")

            if not workflow_id:
                return {
                    "success": False,
                    "error": "workflow_id is required for cancellation"
                }

            # Cancel through Temporal
            success = await self.temporal_client.cancel_workflow(workflow_id, reason)

            # Remove from active workflows
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]

            return {
                "success": success,
                "action": "cancelled",
                "workflow_id": workflow_id,
                "reason": reason
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to cancel workflow: {str(e)}"
            }

    def get_schema(self) -> dict[str, Any]:
        """Get tool schema for MCP."""
        return build_tool_schema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "Workflow operation to perform",
                        "enum": ["start_workflow", "approval", "status", "monitor", "list_workflows", "cancel_workflow"]
                    },
                    "workflow_type": {
                        "type": "string",
                        "description": "Type of workflow to start (for start_workflow operation)",
                        "enum": ["multi_agent_project", "approval", "saga"]
                    },
                    "workflow_spec": {
                        "type": "object",
                        "description": "Workflow specification (for start_workflow operation)"
                    },
                    "config": {
                        "type": "object",
                        "description": "Workflow configuration options"
                    },
                    "approval_id": {
                        "type": "string",
                        "description": "Approval request ID (for approval operations)"
                    },
                    "action": {
                        "type": "string",
                        "description": "Specific action to take (for approval and monitor operations)"
                    },
                    "workflow_id": {
                        "type": "string",
                        "description": "Workflow ID (for status, monitoring, and cancellation operations)"
                    },
                    "approver_id": {
                        "type": "string",
                        "description": "User ID of the approver (for approval decisions)"
                    },
                    "feedback": {
                        "type": "string",
                        "description": "Feedback or reason for approval decisions"
                    },
                    "include_details": {
                        "type": "boolean",
                        "description": "Include detailed information (for status operation)",
                        "default": False
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for cancellation (for cancel_workflow operation)"
                    }
                },
                "required": ["operation"]
            }
        )


# Register tool instance
workflow_orchestrator_tool = WorkflowOrchestratorTool()

logger.info("Workflow orchestrator tool initialized")
