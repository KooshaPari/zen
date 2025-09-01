"""
Human Approval Workflow System

This module implements sophisticated human-in-the-loop approval workflows
with timeout handling, escalation policies, and integration with external
approval systems (email, Slack, webhooks, etc.).

Features:
- Multi-stage approval processes
- Timeout and escalation handling
- Approval delegation and routing
- External system integrations
- Audit trail and compliance tracking
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import aiohttp
from pydantic import BaseModel

from utils.event_bus import get_event_bus
from utils.storage_backend import get_storage_backend
from utils.temporal_client import BaseWorkflow

logger = logging.getLogger(__name__)


class ApprovalStage(str, Enum):
    """Stages in the approval process."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    ESCALATED = "escalated"
    DELEGATED = "delegated"


class ApprovalPriority(str, Enum):
    """Priority levels for approval requests."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class NotificationChannel(str, Enum):
    """Notification channels for approval requests."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    IN_APP = "in_app"


class ApprovalRule(BaseModel):
    """Rules governing approval processes."""
    rule_id: str
    name: str
    description: str
    conditions: dict[str, Any]  # Conditions that trigger this rule
    required_approvers: list[str]  # User IDs or roles
    minimum_approvals: int = 1
    timeout_minutes: int = 1440  # 24 hours default
    escalation_chain: list[str] = []  # Escalation hierarchy
    notification_channels: list[NotificationChannel] = []
    auto_approve_conditions: dict[str, Any] = {}  # Conditions for auto-approval
    delegation_allowed: bool = True


class ApprovalRequest(BaseModel):
    """Individual approval request."""
    request_id: str
    workflow_id: str
    title: str
    description: str
    context: dict[str, Any]
    priority: ApprovalPriority
    rule_id: Optional[str] = None
    required_approvers: list[str]
    minimum_approvals: int = 1
    created_at: datetime
    expires_at: datetime
    current_stage: ApprovalStage = ApprovalStage.PENDING
    approvals_received: list[dict[str, Any]] = []
    rejections_received: list[dict[str, Any]] = []
    notifications_sent: list[dict[str, Any]] = []
    escalation_level: int = 0
    auto_approved: bool = False
    delegated_to: list[str] = []


class ApprovalDecision(BaseModel):
    """Individual approval decision."""
    decision_id: str
    request_id: str
    approver_id: str
    decision: ApprovalStage  # APPROVED or REJECTED
    feedback: Optional[str] = None
    decided_at: datetime
    delegation_from: Optional[str] = None  # If decision was delegated


class EscalationPolicy(BaseModel):
    """Escalation policy for approval timeouts."""
    policy_id: str
    name: str
    stages: list[dict[str, Any]]  # Each stage defines timeout and escalation target
    max_escalations: int = 3
    final_action: str = "reject"  # reject, approve, or manual


class HumanApprovalWorkflow(BaseWorkflow):
    """
    Sophisticated human approval workflow with multi-stage approvals,
    escalation policies, and external system integrations.
    """

    def __init__(self):
        super().__init__()
        self.storage = get_storage_backend()
        self.event_bus = get_event_bus()

        # Load approval rules and policies
        self.approval_rules: dict[str, ApprovalRule] = {}
        self.escalation_policies: dict[str, EscalationPolicy] = {}

    async def orchestrate(self, workflow_args: dict[str, Any]) -> dict[str, Any]:
        """
        Main approval workflow orchestration.

        Args:
            workflow_args: Contains approval request details and configuration

        Returns:
            Dict containing approval decision and audit trail
        """
        request_data = workflow_args["approval_request"]
        config = workflow_args.get("config", {})

        # Create approval request
        approval_request = ApprovalRequest(
            request_id=request_data.get("request_id", f"approval-{uuid4()}"),
            workflow_id=self.context.workflow_id if self.context else f"approval-workflow-{uuid4()}",
            title=request_data["title"],
            description=request_data["description"],
            context=request_data.get("context", {}),
            priority=ApprovalPriority(request_data.get("priority", "normal")),
            rule_id=request_data.get("rule_id"),
            required_approvers=request_data["required_approvers"],
            minimum_approvals=request_data.get("minimum_approvals", 1),
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(
                minutes=request_data.get("timeout_minutes", 1440)
            )
        )

        try:
            logger.info(f"Starting approval workflow: {approval_request.title}")

            # Apply approval rules if specified
            if approval_request.rule_id:
                await self._apply_approval_rule(approval_request)

            # Check for auto-approval conditions
            if await self._check_auto_approval(approval_request):
                approval_request.current_stage = ApprovalStage.APPROVED
                approval_request.auto_approved = True

                await self._store_approval_request(approval_request)
                await self._publish_approval_event(approval_request, "auto_approved")

                return {
                    "status": "approved",
                    "request_id": approval_request.request_id,
                    "auto_approved": True,
                    "decided_at": datetime.utcnow().isoformat(),
                    "approval_request": approval_request.model_dump()
                }

            # Send initial notifications
            await self._send_approval_notifications(approval_request)

            # Store initial request
            await self._store_approval_request(approval_request)
            await self._publish_approval_event(approval_request, "requested")

            # Wait for approval decision with timeout and escalation
            decision_result = await self._wait_for_approval_decision(
                approval_request, config
            )

            # Process final decision
            final_result = await self._process_final_decision(
                approval_request, decision_result
            )

            return final_result

        except Exception as e:
            logger.error(f"Approval workflow failed: {e}")

            approval_request.current_stage = ApprovalStage.TIMEOUT
            await self._store_approval_request(approval_request)
            await self._publish_approval_event(approval_request, "failed", {"error": str(e)})

            return {
                "status": "failed",
                "error": str(e),
                "request_id": approval_request.request_id,
                "approval_request": approval_request.model_dump()
            }

    async def _apply_approval_rule(self, approval_request: ApprovalRequest):
        """Apply approval rule to customize request parameters."""
        if not approval_request.rule_id or approval_request.rule_id not in self.approval_rules:
            logger.warning(f"Approval rule {approval_request.rule_id} not found")
            return

        rule = self.approval_rules[approval_request.rule_id]

        # Override request parameters based on rule
        if rule.required_approvers:
            approval_request.required_approvers = rule.required_approvers

        approval_request.minimum_approvals = rule.minimum_approvals

        # Update expiration based on rule timeout
        approval_request.expires_at = approval_request.created_at + timedelta(
            minutes=rule.timeout_minutes
        )

        logger.info(f"Applied approval rule {rule.name} to request {approval_request.request_id}")

    async def _check_auto_approval(self, approval_request: ApprovalRequest) -> bool:
        """Check if request meets auto-approval conditions."""
        # Example auto-approval conditions
        auto_approve_conditions = [
            # Low priority requests under certain thresholds
            (
                approval_request.priority == ApprovalPriority.LOW and
                approval_request.context.get("estimated_cost", 0) < 100
            ),
            # Requests from trusted systems
            approval_request.context.get("source") == "trusted_automation",
            # Emergency requests with proper authorization
            (
                approval_request.priority == ApprovalPriority.EMERGENCY and
                approval_request.context.get("emergency_override") is True
            )
        ]

        return any(auto_approve_conditions)

    async def _send_approval_notifications(self, approval_request: ApprovalRequest):
        """Send approval notifications through configured channels."""
        notification_tasks = []

        for approver_id in approval_request.required_approvers:
            # Get approver's notification preferences
            approver_config = await self._get_approver_config(approver_id)

            if not approver_config:
                logger.warning(f"No configuration found for approver {approver_id}")
                continue

            # Send notifications through preferred channels
            for channel in approver_config.get("notification_channels", [NotificationChannel.EMAIL]):
                notification_tasks.append(
                    self._send_notification(
                        channel, approver_id, approval_request, approver_config
                    )
                )

        # Execute all notifications in parallel
        if notification_tasks:
            results = await asyncio.gather(*notification_tasks, return_exceptions=True)

            # Log notification results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Notification failed: {result}")
                else:
                    approval_request.notifications_sent.append({
                        "channel": notification_tasks[i].__name__ if hasattr(notification_tasks[i], '__name__') else "unknown",
                        "sent_at": datetime.utcnow().isoformat(),
                        "success": result.get("success", False)
                    })

    async def _send_notification(
        self,
        channel: NotificationChannel,
        approver_id: str,
        approval_request: ApprovalRequest,
        approver_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Send notification through specific channel."""
        try:
            if channel == NotificationChannel.EMAIL:
                return await self._send_email_notification(
                    approver_id, approval_request, approver_config
                )
            elif channel == NotificationChannel.SLACK:
                return await self._send_slack_notification(
                    approver_id, approval_request, approver_config
                )
            elif channel == NotificationChannel.WEBHOOK:
                return await self._send_webhook_notification(
                    approver_id, approval_request, approver_config
                )
            elif channel == NotificationChannel.SMS:
                return await self._send_sms_notification(
                    approver_id, approval_request, approver_config
                )
            else:
                logger.warning(f"Unsupported notification channel: {channel}")
                return {"success": False, "error": "Unsupported channel"}

        except Exception as e:
            logger.error(f"Failed to send {channel} notification to {approver_id}: {e}")
            return {"success": False, "error": str(e)}

    async def _send_email_notification(
        self,
        approver_id: str,
        approval_request: ApprovalRequest,
        approver_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Send email notification (placeholder implementation)."""
        logger.info(f"Sending email notification to {approver_id}")

        # In real implementation, integrate with email service
        email_data = {
            "to": approver_config.get("email"),
            "subject": f"Approval Required: {approval_request.title}",
            "body": f"""
            An approval request requires your attention:

            Title: {approval_request.title}
            Description: {approval_request.description}
            Priority: {approval_request.priority.value}
            Expires: {approval_request.expires_at.isoformat()}

            Request ID: {approval_request.request_id}

            Please review and approve or reject this request.
            """
        }

        # Simulate email sending
        await asyncio.sleep(0.1)
        return {"success": True, "email_data": email_data}

    async def _send_slack_notification(
        self,
        approver_id: str,
        approval_request: ApprovalRequest,
        approver_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Send Slack notification (placeholder implementation)."""
        logger.info(f"Sending Slack notification to {approver_id}")

        slack_webhook_url = approver_config.get("slack_webhook")
        if not slack_webhook_url:
            return {"success": False, "error": "No Slack webhook configured"}

        # Prepare Slack message
        slack_payload = {
            "text": f"Approval Required: {approval_request.title}",
            "attachments": [
                {
                    "color": "warning" if approval_request.priority in [ApprovalPriority.HIGH, ApprovalPriority.CRITICAL] else "good",
                    "fields": [
                        {
                            "title": "Description",
                            "value": approval_request.description,
                            "short": False
                        },
                        {
                            "title": "Priority",
                            "value": approval_request.priority.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Expires",
                            "value": approval_request.expires_at.strftime("%Y-%m-%d %H:%M UTC"),
                            "short": True
                        },
                        {
                            "title": "Request ID",
                            "value": approval_request.request_id,
                            "short": True
                        }
                    ],
                    "actions": [
                        {
                            "type": "button",
                            "text": "Approve",
                            "style": "primary",
                            "value": f"approve_{approval_request.request_id}"
                        },
                        {
                            "type": "button",
                            "text": "Reject",
                            "style": "danger",
                            "value": f"reject_{approval_request.request_id}"
                        }
                    ]
                }
            ]
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    slack_webhook_url,
                    json=slack_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return {"success": True}
                    else:
                        return {"success": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_webhook_notification(
        self,
        approver_id: str,
        approval_request: ApprovalRequest,
        approver_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Send webhook notification."""
        webhook_url = approver_config.get("webhook_url")
        if not webhook_url:
            return {"success": False, "error": "No webhook URL configured"}

        webhook_payload = {
            "event": "approval_required",
            "approver_id": approver_id,
            "request": approval_request.model_dump(),
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=webhook_payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return {"success": True}
                    else:
                        return {"success": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_sms_notification(
        self,
        approver_id: str,
        approval_request: ApprovalRequest,
        approver_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Send SMS notification (placeholder implementation)."""
        logger.info(f"Sending SMS notification to {approver_id}")

        phone_number = approver_config.get("phone_number")
        if not phone_number:
            return {"success": False, "error": "No phone number configured"}

        # In real implementation, integrate with SMS service (Twilio, AWS SNS, etc.)
        sms_message = f"Approval required: {approval_request.title}. Priority: {approval_request.priority.value}. ID: {approval_request.request_id}"

        # Simulate SMS sending
        await asyncio.sleep(0.1)
        return {"success": True, "phone_number": phone_number, "message": sms_message}

    async def _wait_for_approval_decision(
        self,
        approval_request: ApprovalRequest,
        config: dict[str, Any]
    ) -> dict[str, Any]:
        """Wait for approval decision with timeout and escalation handling."""
        check_interval = config.get("check_interval_seconds", 30)
        escalation_policy_id = config.get("escalation_policy_id")

        while datetime.utcnow() < approval_request.expires_at:
            # Check for decisions
            current_request = await self._load_approval_request(approval_request.request_id)
            if not current_request:
                return {"status": "error", "message": "Request not found"}

            # Update our reference
            approval_request = current_request

            # Check if minimum approvals received
            if len(approval_request.approvals_received) >= approval_request.minimum_approvals:
                return {
                    "status": "approved",
                    "approvals": approval_request.approvals_received,
                    "decided_at": datetime.utcnow().isoformat()
                }

            # Check for rejections
            if approval_request.rejections_received:
                return {
                    "status": "rejected",
                    "rejections": approval_request.rejections_received,
                    "decided_at": datetime.utcnow().isoformat()
                }

            # Check for escalation conditions
            if escalation_policy_id and self._should_escalate(approval_request):
                await self._handle_escalation(approval_request, escalation_policy_id)

            await asyncio.sleep(check_interval)

        # Timeout reached
        return {
            "status": "timeout",
            "expired_at": approval_request.expires_at.isoformat()
        }

    def _should_escalate(self, approval_request: ApprovalRequest) -> bool:
        """Check if request should be escalated."""
        # Example escalation conditions
        time_elapsed = datetime.utcnow() - approval_request.created_at

        # Escalate if 50% of timeout period has passed without response
        timeout_period = approval_request.expires_at - approval_request.created_at
        escalation_threshold = timeout_period * 0.5

        return time_elapsed > escalation_threshold and not approval_request.approvals_received

    async def _handle_escalation(
        self,
        approval_request: ApprovalRequest,
        escalation_policy_id: str
    ):
        """Handle approval escalation."""
        if escalation_policy_id not in self.escalation_policies:
            logger.warning(f"Escalation policy {escalation_policy_id} not found")
            return

        policy = self.escalation_policies[escalation_policy_id]

        if approval_request.escalation_level >= policy.max_escalations:
            logger.info(f"Maximum escalations reached for request {approval_request.request_id}")
            return

        approval_request.escalation_level += 1
        approval_request.current_stage = ApprovalStage.ESCALATED

        # Get next escalation stage
        if approval_request.escalation_level <= len(policy.stages):
            stage = policy.stages[approval_request.escalation_level - 1]
            escalation_targets = stage.get("escalation_targets", [])

            # Add escalation targets to required approvers
            for target in escalation_targets:
                if target not in approval_request.required_approvers:
                    approval_request.required_approvers.append(target)

            # Send escalation notifications
            await self._send_escalation_notifications(approval_request, stage)

        await self._store_approval_request(approval_request)
        await self._publish_approval_event(
            approval_request,
            "escalated",
            {"escalation_level": approval_request.escalation_level}
        )

    async def _send_escalation_notifications(
        self,
        approval_request: ApprovalRequest,
        escalation_stage: dict[str, Any]
    ):
        """Send escalation notifications."""
        escalation_targets = escalation_stage.get("escalation_targets", [])

        for target in escalation_targets:
            approver_config = await self._get_approver_config(target)
            if approver_config:
                # Send high-priority notification
                await self._send_notification(
                    NotificationChannel.EMAIL,  # Use email for escalations
                    target,
                    approval_request,
                    approver_config
                )

        logger.info(f"Sent escalation notifications for request {approval_request.request_id}")

    async def _process_final_decision(
        self,
        approval_request: ApprovalRequest,
        decision_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Process final approval decision."""
        status = decision_result["status"]

        if status == "approved":
            approval_request.current_stage = ApprovalStage.APPROVED
        elif status == "rejected":
            approval_request.current_stage = ApprovalStage.REJECTED
        elif status == "timeout":
            approval_request.current_stage = ApprovalStage.TIMEOUT

        # Store final state
        await self._store_approval_request(approval_request)
        await self._publish_approval_event(approval_request, "final_decision", decision_result)

        # Send final notifications to stakeholders
        await self._send_final_notifications(approval_request, decision_result)

        return {
            "status": status,
            "request_id": approval_request.request_id,
            "final_stage": approval_request.current_stage.value,
            "decision_details": decision_result,
            "approval_request": approval_request.model_dump()
        }

    async def _send_final_notifications(
        self,
        approval_request: ApprovalRequest,
        decision_result: dict[str, Any]
    ):
        """Send final decision notifications to all stakeholders."""
        # Notify original requester
        if approval_request.context.get("requester_id"):
            logger.info(f"Sending final decision notification for request {approval_request.request_id}")

        # Notify all participants
        all_participants = set(approval_request.required_approvers + approval_request.delegated_to)
        for _participant in all_participants:
            # In real implementation, send final decision notification
            pass

    async def _get_approver_config(self, approver_id: str) -> Optional[dict[str, Any]]:
        """Get approver configuration and preferences."""
        key = f"approver_config:{approver_id}"
        config_data = self.storage.get(key)

        if config_data:
            return json.loads(config_data)

        # Return default configuration
        return {
            "notification_channels": [NotificationChannel.EMAIL],
            "email": f"{approver_id}@example.com",  # Placeholder
            "timezone": "UTC"
        }

    async def _store_approval_request(self, approval_request: ApprovalRequest):
        """Store approval request in Redis."""
        key = f"approval_request:{approval_request.request_id}"
        ttl = max(3600, int((approval_request.expires_at - datetime.utcnow()).total_seconds()) + 3600)
        self.storage.setex(key, ttl, approval_request.model_dump_json())

    async def _load_approval_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Load approval request from Redis."""
        key = f"approval_request:{request_id}"
        data = self.storage.get(key)
        if data:
            return ApprovalRequest.model_validate_json(data)
        return None

    async def _publish_approval_event(
        self,
        approval_request: ApprovalRequest,
        event_type: str,
        extra_data: Optional[dict[str, Any]] = None
    ):
        """Publish approval event to event bus."""
        event_data = {
            "event": f"approval_{event_type}",
            "request_id": approval_request.request_id,
            "workflow_id": approval_request.workflow_id,
            "title": approval_request.title,
            "priority": approval_request.priority.value,
            "current_stage": approval_request.current_stage.value,
            "timestamp": datetime.utcnow().isoformat()
        }

        if extra_data:
            event_data.update(extra_data)

        await self.event_bus.publish(event_data)


# Approval management functions
async def submit_approval_decision(
    request_id: str,
    approver_id: str,
    decision: str,  # "approve" or "reject"
    feedback: Optional[str] = None
) -> bool:
    """Submit an approval decision."""
    storage = get_storage_backend()
    event_bus = get_event_bus()

    # Load approval request
    key = f"approval_request:{request_id}"
    data = storage.get(key)
    if not data:
        logger.warning(f"Approval request {request_id} not found")
        return False

    approval_request = ApprovalRequest.model_validate_json(data)

    # Check if approver is authorized
    if approver_id not in approval_request.required_approvers:
        logger.warning(f"Approver {approver_id} not authorized for request {request_id}")
        return False

    # Create decision record
    decision_record = {
        "approver_id": approver_id,
        "decision": decision,
        "feedback": feedback,
        "decided_at": datetime.utcnow().isoformat()
    }

    # Update approval request
    if decision.lower() == "approve":
        approval_request.approvals_received.append(decision_record)
    else:
        approval_request.rejections_received.append(decision_record)

    # Store updated request
    ttl = max(3600, int((approval_request.expires_at - datetime.utcnow()).total_seconds()) + 3600)
    storage.setex(key, ttl, approval_request.model_dump_json())

    # Publish decision event
    await event_bus.publish({
        "event": "approval_decision_submitted",
        "request_id": request_id,
        "approver_id": approver_id,
        "decision": decision,
        "timestamp": datetime.utcnow().isoformat()
    })

    logger.info(f"Approval decision submitted: {request_id} - {decision} by {approver_id}")
    return True


async def delegate_approval(
    request_id: str,
    from_approver: str,
    to_approver: str,
    reason: Optional[str] = None
) -> bool:
    """Delegate approval to another user."""
    storage = get_storage_backend()
    event_bus = get_event_bus()

    # Load approval request
    key = f"approval_request:{request_id}"
    data = storage.get(key)
    if not data:
        return False

    approval_request = ApprovalRequest.model_validate_json(data)

    # Check if from_approver is authorized
    if from_approver not in approval_request.required_approvers:
        return False

    # Add delegation
    if to_approver not in approval_request.required_approvers:
        approval_request.required_approvers.append(to_approver)

    if to_approver not in approval_request.delegated_to:
        approval_request.delegated_to.append(to_approver)

    approval_request.current_stage = ApprovalStage.DELEGATED

    # Store updated request
    ttl = max(3600, int((approval_request.expires_at - datetime.utcnow()).total_seconds()) + 3600)
    storage.setex(key, ttl, approval_request.model_dump_json())

    # Publish delegation event
    await event_bus.publish({
        "event": "approval_delegated",
        "request_id": request_id,
        "from_approver": from_approver,
        "to_approver": to_approver,
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat()
    })

    logger.info(f"Approval delegated: {request_id} from {from_approver} to {to_approver}")
    return True


logger.info("Human approval workflow module initialized")
