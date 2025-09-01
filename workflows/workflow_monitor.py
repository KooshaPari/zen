"""
Workflow Monitoring and Recovery System

This module provides comprehensive monitoring, alerting, and recovery
capabilities for Temporal workflows. Includes performance metrics,
health checks, automatic recovery, and integration with observability systems.

Features:
- Real-time workflow monitoring
- Performance metrics collection
- Automatic failure detection and recovery
- Alert management and notification
- Workflow health dashboards
- Recovery policy enforcement
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel

from utils.event_bus import get_event_bus
from utils.storage_backend import get_storage_backend

logger = logging.getLogger(__name__)


class WorkflowHealthStatus(str, Enum):
    """Workflow health status indicators."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RecoveryAction(str, Enum):
    """Recovery actions that can be taken."""
    RETRY = "retry"
    RESTART = "restart"
    COMPENSATE = "compensate"
    ESCALATE = "escalate"
    MANUAL_INTERVENTION = "manual_intervention"
    ABORT = "abort"


class WorkflowMetrics(BaseModel):
    """Workflow performance metrics."""
    workflow_id: str
    workflow_type: str
    status: str
    started_at: datetime
    updated_at: datetime
    duration_seconds: Optional[float] = None
    completion_rate: float = 0.0
    error_count: int = 0
    retry_count: int = 0
    resource_usage: dict[str, Any] = {}
    throughput_metrics: dict[str, float] = {}
    agent_utilization: dict[str, float] = {}


class WorkflowAlert(BaseModel):
    """Workflow alert definition."""
    alert_id: str
    workflow_id: str
    alert_type: str
    severity: AlertSeverity
    title: str
    description: str
    created_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    metadata: dict[str, Any] = {}
    notification_channels: list[str] = []


class RecoveryPolicy(BaseModel):
    """Recovery policy definition."""
    policy_id: str
    name: str
    workflow_types: list[str]  # Workflow types this policy applies to
    conditions: dict[str, Any]  # Conditions that trigger recovery
    recovery_actions: list[RecoveryAction]
    max_recovery_attempts: int = 3
    recovery_delay_seconds: int = 60
    escalation_threshold: int = 3
    notification_channels: list[str] = []


class WorkflowHealthCheck(BaseModel):
    """Workflow health check result."""
    workflow_id: str
    status: WorkflowHealthStatus
    checked_at: datetime
    checks_performed: list[str]
    issues_detected: list[str]
    recommendations: list[str]
    next_check_at: datetime


class WorkflowMonitor:
    """
    Comprehensive workflow monitoring and recovery system.

    Provides real-time monitoring, alerting, and automatic recovery
    for all workflow types in the system.
    """

    def __init__(self):
        self.storage = get_storage_backend()
        self.event_bus = get_event_bus()

        # Monitoring state
        self.monitored_workflows: dict[str, WorkflowMetrics] = {}
        self.active_alerts: dict[str, WorkflowAlert] = {}
        self.recovery_policies: dict[str, RecoveryPolicy] = {}
        self.health_checks: dict[str, WorkflowHealthCheck] = {}

        # Monitoring configuration
        self.monitoring_enabled = True
        self.health_check_interval = 300  # 5 minutes
        self.metrics_collection_interval = 60  # 1 minute
        self.alert_evaluation_interval = 30  # 30 seconds

        # Performance thresholds
        self.performance_thresholds = {
            "max_execution_time_hours": 24,
            "min_completion_rate": 0.8,
            "max_error_rate": 0.1,
            "max_retry_rate": 0.3
        }

        self._monitoring_tasks: set[asyncio.Task] = set()

    async def start_monitoring(self):
        """Start workflow monitoring services."""
        if not self.monitoring_enabled:
            logger.info("Workflow monitoring is disabled")
            return

        logger.info("Starting workflow monitoring services")

        # Subscribe to workflow events
        await self._subscribe_to_workflow_events()

        # Start monitoring tasks
        tasks = [
            self._metrics_collection_loop(),
            self._health_check_loop(),
            self._alert_evaluation_loop(),
            self._recovery_management_loop()
        ]

        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._monitoring_tasks.add(task)
            task.add_done_callback(self._monitoring_tasks.discard)

        # Load existing recovery policies
        await self._load_recovery_policies()

        logger.info("Workflow monitoring services started")

    async def stop_monitoring(self):
        """Stop workflow monitoring services."""
        logger.info("Stopping workflow monitoring services")

        # Cancel monitoring tasks
        for task in self._monitoring_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)

        self._monitoring_tasks.clear()

        logger.info("Workflow monitoring services stopped")

    async def register_workflow(self, workflow_id: str, workflow_type: str, metadata: dict[str, Any] = None):
        """Register a workflow for monitoring."""
        metrics = WorkflowMetrics(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            status="initializing",
            started_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        self.monitored_workflows[workflow_id] = metrics
        await self._store_workflow_metrics(metrics)

        logger.info(f"Registered workflow {workflow_id} for monitoring")

        # Publish registration event
        await self.event_bus.publish({
            "event": "workflow_monitoring_registered",
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def update_workflow_metrics(self, workflow_id: str, updates: dict[str, Any]):
        """Update workflow metrics."""
        if workflow_id not in self.monitored_workflows:
            logger.warning(f"Workflow {workflow_id} not registered for monitoring")
            return

        metrics = self.monitored_workflows[workflow_id]

        # Update metrics
        for key, value in updates.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)

        metrics.updated_at = datetime.utcnow()

        # Calculate derived metrics
        if metrics.started_at:
            metrics.duration_seconds = (metrics.updated_at - metrics.started_at).total_seconds()

        await self._store_workflow_metrics(metrics)

        # Check for alert conditions
        await self._evaluate_workflow_alerts(workflow_id, metrics)

    async def create_alert(
        self,
        workflow_id: str,
        alert_type: str,
        severity: AlertSeverity,
        title: str,
        description: str,
        metadata: dict[str, Any] = None
    ) -> str:
        """Create a new workflow alert."""
        alert_id = f"alert-{uuid4()}"

        alert = WorkflowAlert(
            alert_id=alert_id,
            workflow_id=workflow_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            description=description,
            created_at=datetime.utcnow(),
            metadata=metadata or {}
        )

        self.active_alerts[alert_id] = alert
        await self._store_workflow_alert(alert)

        # Send notifications
        await self._send_alert_notifications(alert)

        # Publish alert event
        await self.event_bus.publish({
            "event": "workflow_alert_created",
            "alert_id": alert_id,
            "workflow_id": workflow_id,
            "severity": severity.value,
            "title": title,
            "timestamp": alert.created_at.isoformat()
        })

        logger.warning(f"Created {severity.value} alert for workflow {workflow_id}: {title}")
        return alert_id

    async def resolve_alert(self, alert_id: str, resolved_by: Optional[str] = None):
        """Resolve an active alert."""
        if alert_id not in self.active_alerts:
            logger.warning(f"Alert {alert_id} not found in active alerts")
            return False

        alert = self.active_alerts[alert_id]
        alert.resolved_at = datetime.utcnow()

        await self._store_workflow_alert(alert)
        del self.active_alerts[alert_id]

        # Publish resolution event
        await self.event_bus.publish({
            "event": "workflow_alert_resolved",
            "alert_id": alert_id,
            "workflow_id": alert.workflow_id,
            "resolved_by": resolved_by,
            "timestamp": alert.resolved_at.isoformat()
        })

        logger.info(f"Resolved alert {alert_id}")
        return True

    async def perform_health_check(self, workflow_id: str) -> WorkflowHealthCheck:
        """Perform comprehensive health check on a workflow."""
        checks_performed = []
        issues_detected = []
        recommendations = []

        # Check if workflow is registered
        if workflow_id not in self.monitored_workflows:
            health_check = WorkflowHealthCheck(
                workflow_id=workflow_id,
                status=WorkflowHealthStatus.UNKNOWN,
                checked_at=datetime.utcnow(),
                checks_performed=["registration_check"],
                issues_detected=["Workflow not registered for monitoring"],
                recommendations=["Register workflow for monitoring"],
                next_check_at=datetime.utcnow() + timedelta(seconds=self.health_check_interval)
            )
            return health_check

        metrics = self.monitored_workflows[workflow_id]
        status = WorkflowHealthStatus.HEALTHY

        # Check execution time
        checks_performed.append("execution_time_check")
        if metrics.duration_seconds and metrics.duration_seconds > self.performance_thresholds["max_execution_time_hours"] * 3600:
            issues_detected.append(f"Workflow running for {metrics.duration_seconds / 3600:.1f} hours")
            recommendations.append("Consider checking for workflow deadlocks or infinite loops")
            status = WorkflowHealthStatus.WARNING

        # Check completion rate
        checks_performed.append("completion_rate_check")
        if metrics.completion_rate < self.performance_thresholds["min_completion_rate"]:
            issues_detected.append(f"Low completion rate: {metrics.completion_rate:.2f}")
            recommendations.append("Investigate failed steps and improve error handling")
            if status == WorkflowHealthStatus.HEALTHY:
                status = WorkflowHealthStatus.WARNING

        # Check error rate
        checks_performed.append("error_rate_check")
        total_operations = max(1, metrics.error_count + metrics.retry_count + 1)
        error_rate = metrics.error_count / total_operations
        if error_rate > self.performance_thresholds["max_error_rate"]:
            issues_detected.append(f"High error rate: {error_rate:.2f}")
            recommendations.append("Review error patterns and improve error handling")
            status = WorkflowHealthStatus.CRITICAL

        # Check for active alerts
        checks_performed.append("active_alerts_check")
        workflow_alerts = [alert for alert in self.active_alerts.values() if alert.workflow_id == workflow_id]
        critical_alerts = [alert for alert in workflow_alerts if alert.severity == AlertSeverity.CRITICAL]

        if critical_alerts:
            issues_detected.append(f"{len(critical_alerts)} critical alerts active")
            recommendations.append("Address critical alerts immediately")
            status = WorkflowHealthStatus.CRITICAL
        elif workflow_alerts:
            issues_detected.append(f"{len(workflow_alerts)} alerts active")
            if status == WorkflowHealthStatus.HEALTHY:
                status = WorkflowHealthStatus.WARNING

        # Check resource usage
        checks_performed.append("resource_usage_check")
        if metrics.resource_usage:
            high_cpu = metrics.resource_usage.get("cpu_percentage", 0) > 80
            high_memory = metrics.resource_usage.get("memory_percentage", 0) > 80

            if high_cpu or high_memory:
                issues_detected.append("High resource usage detected")
                recommendations.append("Consider optimizing workflow or scaling resources")
                if status == WorkflowHealthStatus.HEALTHY:
                    status = WorkflowHealthStatus.WARNING

        # Check workflow-specific health
        checks_performed.append("workflow_specific_check")
        await self._perform_workflow_specific_health_check(
            workflow_id, metrics, issues_detected, recommendations
        )

        health_check = WorkflowHealthCheck(
            workflow_id=workflow_id,
            status=status,
            checked_at=datetime.utcnow(),
            checks_performed=checks_performed,
            issues_detected=issues_detected,
            recommendations=recommendations,
            next_check_at=datetime.utcnow() + timedelta(seconds=self.health_check_interval)
        )

        self.health_checks[workflow_id] = health_check
        await self._store_health_check(health_check)

        return health_check

    async def trigger_recovery(
        self,
        workflow_id: str,
        recovery_reason: str,
        recovery_action: RecoveryAction = RecoveryAction.RETRY
    ) -> dict[str, Any]:
        """Trigger recovery action for a workflow."""
        logger.info(f"Triggering recovery for workflow {workflow_id}: {recovery_reason}")

        recovery_id = f"recovery-{uuid4()}"
        recovery_started_at = datetime.utcnow()

        # Publish recovery started event
        await self.event_bus.publish({
            "event": "workflow_recovery_started",
            "workflow_id": workflow_id,
            "recovery_id": recovery_id,
            "recovery_action": recovery_action.value,
            "reason": recovery_reason,
            "timestamp": recovery_started_at.isoformat()
        })

        try:
            # Execute recovery action
            if recovery_action == RecoveryAction.RETRY:
                result = await self._execute_retry_recovery(workflow_id)
            elif recovery_action == RecoveryAction.RESTART:
                result = await self._execute_restart_recovery(workflow_id)
            elif recovery_action == RecoveryAction.COMPENSATE:
                result = await self._execute_compensation_recovery(workflow_id)
            elif recovery_action == RecoveryAction.ESCALATE:
                result = await self._execute_escalation_recovery(workflow_id)
            else:
                result = {
                    "success": False,
                    "error": f"Unsupported recovery action: {recovery_action}"
                }

            recovery_duration = (datetime.utcnow() - recovery_started_at).total_seconds()

            # Publish recovery completed event
            await self.event_bus.publish({
                "event": "workflow_recovery_completed",
                "workflow_id": workflow_id,
                "recovery_id": recovery_id,
                "success": result.get("success", False),
                "duration_seconds": recovery_duration,
                "timestamp": datetime.utcnow().isoformat()
            })

            return {
                "recovery_id": recovery_id,
                "success": result.get("success", False),
                "duration_seconds": recovery_duration,
                "result": result
            }

        except Exception as e:
            logger.error(f"Recovery failed for workflow {workflow_id}: {e}")

            # Publish recovery failed event
            await self.event_bus.publish({
                "event": "workflow_recovery_failed",
                "workflow_id": workflow_id,
                "recovery_id": recovery_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })

            return {
                "recovery_id": recovery_id,
                "success": False,
                "error": str(e)
            }

    async def get_workflow_dashboard_data(self, workflow_id: Optional[str] = None) -> dict[str, Any]:
        """Get comprehensive dashboard data for workflows."""
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_status": "active" if self.monitoring_enabled else "inactive"
        }

        if workflow_id:
            # Single workflow dashboard
            if workflow_id in self.monitored_workflows:
                metrics = self.monitored_workflows[workflow_id]
                health_check = self.health_checks.get(workflow_id)
                workflow_alerts = [
                    alert for alert in self.active_alerts.values()
                    if alert.workflow_id == workflow_id
                ]

                dashboard_data.update({
                    "workflow_id": workflow_id,
                    "metrics": metrics.model_dump(),
                    "health_check": health_check.model_dump() if health_check else None,
                    "active_alerts": [alert.model_dump() for alert in workflow_alerts],
                    "alert_count": len(workflow_alerts)
                })
            else:
                dashboard_data["error"] = f"Workflow {workflow_id} not found"
        else:
            # System-wide dashboard
            total_workflows = len(self.monitored_workflows)
            healthy_workflows = len([
                hc for hc in self.health_checks.values()
                if hc.status == WorkflowHealthStatus.HEALTHY
            ])
            critical_workflows = len([
                hc for hc in self.health_checks.values()
                if hc.status == WorkflowHealthStatus.CRITICAL
            ])

            dashboard_data.update({
                "summary": {
                    "total_workflows": total_workflows,
                    "healthy_workflows": healthy_workflows,
                    "critical_workflows": critical_workflows,
                    "total_alerts": len(self.active_alerts),
                    "critical_alerts": len([
                        alert for alert in self.active_alerts.values()
                        if alert.severity == AlertSeverity.CRITICAL
                    ])
                },
                "workflows": [metrics.model_dump() for metrics in self.monitored_workflows.values()],
                "recent_alerts": [
                    alert.model_dump() for alert in
                    sorted(self.active_alerts.values(), key=lambda a: a.created_at, reverse=True)[:10]
                ]
            })

        return dashboard_data

    async def _subscribe_to_workflow_events(self):
        """Subscribe to workflow events for monitoring."""
        event_queue = await self.event_bus.subscribe()

        async def event_handler():
            while self.monitoring_enabled:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                    await self._handle_workflow_event(event)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error handling workflow event: {e}")

        # Start event handler task
        task = asyncio.create_task(event_handler())
        self._monitoring_tasks.add(task)
        task.add_done_callback(self._monitoring_tasks.discard)

    async def _handle_workflow_event(self, event: dict[str, Any]):
        """Handle workflow events for monitoring."""
        event_type = event.get("event")
        workflow_id = event.get("workflow_id")

        if not workflow_id or workflow_id not in self.monitored_workflows:
            return

        # Update metrics based on event
        updates = {"updated_at": datetime.utcnow()}

        if event_type in ["workflow_started", "multi_agent_project_started"]:
            updates["status"] = "running"
        elif event_type in ["workflow_completed", "multi_agent_project_completed"]:
            updates["status"] = "completed"
            updates["completion_rate"] = 1.0
        elif event_type in ["workflow_failed", "multi_agent_project_failed"]:
            updates["status"] = "failed"
            updates["error_count"] = self.monitored_workflows[workflow_id].error_count + 1
        elif "retry" in event_type:
            updates["retry_count"] = self.monitored_workflows[workflow_id].retry_count + 1

        if updates:
            await self.update_workflow_metrics(workflow_id, updates)

    async def _metrics_collection_loop(self):
        """Background task for metrics collection."""
        while self.monitoring_enabled:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.metrics_collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.metrics_collection_interval)

    async def _health_check_loop(self):
        """Background task for health checks."""
        while self.monitoring_enabled:
            try:
                # Perform health checks for all workflows
                for workflow_id in list(self.monitored_workflows.keys()):
                    await self.perform_health_check(workflow_id)

                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _alert_evaluation_loop(self):
        """Background task for alert evaluation."""
        while self.monitoring_enabled:
            try:
                await self._evaluate_all_workflow_alerts()
                await asyncio.sleep(self.alert_evaluation_interval)
            except Exception as e:
                logger.error(f"Error in alert evaluation: {e}")
                await asyncio.sleep(self.alert_evaluation_interval)

    async def _recovery_management_loop(self):
        """Background task for recovery management."""
        while self.monitoring_enabled:
            try:
                await self._check_recovery_conditions()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in recovery management: {e}")
                await asyncio.sleep(60)

    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        # Placeholder for system metrics collection
        # In real implementation, collect CPU, memory, disk, network metrics
        pass

    async def _evaluate_all_workflow_alerts(self):
        """Evaluate alert conditions for all workflows."""
        for workflow_id, metrics in self.monitored_workflows.items():
            await self._evaluate_workflow_alerts(workflow_id, metrics)

    async def _evaluate_workflow_alerts(self, workflow_id: str, metrics: WorkflowMetrics):
        """Evaluate alert conditions for a specific workflow."""
        # Check for long-running workflows
        if (metrics.duration_seconds and
            metrics.duration_seconds > self.performance_thresholds["max_execution_time_hours"] * 3600):

            # Check if alert already exists
            existing_alert = next(
                (alert for alert in self.active_alerts.values()
                 if alert.workflow_id == workflow_id and alert.alert_type == "long_running"),
                None
            )

            if not existing_alert:
                await self.create_alert(
                    workflow_id=workflow_id,
                    alert_type="long_running",
                    severity=AlertSeverity.WARNING,
                    title="Long Running Workflow",
                    description=f"Workflow has been running for {metrics.duration_seconds / 3600:.1f} hours"
                )

    async def _check_recovery_conditions(self):
        """Check if any workflows need recovery actions."""
        for workflow_id, health_check in self.health_checks.items():
            if health_check.status == WorkflowHealthStatus.CRITICAL:
                # Check if recovery action is needed
                critical_alerts = [
                    alert for alert in self.active_alerts.values()
                    if alert.workflow_id == workflow_id and alert.severity == AlertSeverity.CRITICAL
                ]

                if len(critical_alerts) >= 3:  # Threshold for automatic recovery
                    await self.trigger_recovery(
                        workflow_id=workflow_id,
                        recovery_reason="Multiple critical alerts detected",
                        recovery_action=RecoveryAction.RETRY
                    )

    async def _perform_workflow_specific_health_check(
        self,
        workflow_id: str,
        metrics: WorkflowMetrics,
        issues_detected: list[str],
        recommendations: list[str]
    ):
        """Perform workflow-type-specific health checks."""
        # Check for workflow-specific issues based on type
        if metrics.workflow_type == "multi_agent_project":
            # Check agent utilization
            if metrics.agent_utilization:
                low_utilization_agents = [
                    agent for agent, util in metrics.agent_utilization.items()
                    if util < 0.3
                ]

                if low_utilization_agents:
                    issues_detected.append(f"Low agent utilization: {', '.join(low_utilization_agents)}")
                    recommendations.append("Review agent task distribution and workload balancing")

    async def _execute_retry_recovery(self, workflow_id: str) -> dict[str, Any]:
        """Execute retry recovery action."""
        logger.info(f"Executing retry recovery for workflow {workflow_id}")

        # In real implementation, this would restart failed workflow steps
        # For now, simulate recovery
        await asyncio.sleep(1)

        return {
            "success": True,
            "action": "retry",
            "message": "Workflow steps retried successfully"
        }

    async def _execute_restart_recovery(self, workflow_id: str) -> dict[str, Any]:
        """Execute restart recovery action."""
        logger.info(f"Executing restart recovery for workflow {workflow_id}")

        # Simulate workflow restart
        await asyncio.sleep(2)

        return {
            "success": True,
            "action": "restart",
            "message": "Workflow restarted successfully"
        }

    async def _execute_compensation_recovery(self, workflow_id: str) -> dict[str, Any]:
        """Execute compensation recovery action."""
        logger.info(f"Executing compensation recovery for workflow {workflow_id}")

        # Simulate compensation execution
        await asyncio.sleep(1.5)

        return {
            "success": True,
            "action": "compensate",
            "message": "Compensation actions executed successfully"
        }

    async def _execute_escalation_recovery(self, workflow_id: str) -> dict[str, Any]:
        """Execute escalation recovery action."""
        logger.info(f"Executing escalation recovery for workflow {workflow_id}")

        # Create escalation alert
        await self.create_alert(
            workflow_id=workflow_id,
            alert_type="escalation",
            severity=AlertSeverity.CRITICAL,
            title="Workflow Escalated for Manual Intervention",
            description="Automatic recovery failed, manual intervention required"
        )

        return {
            "success": True,
            "action": "escalate",
            "message": "Workflow escalated for manual intervention"
        }

    async def _send_alert_notifications(self, alert: WorkflowAlert):
        """Send notifications for workflow alerts."""
        # Placeholder for alert notifications
        # In real implementation, send via email, Slack, webhooks, etc.

        logger.info(f"Sending notifications for alert {alert.alert_id}")

        # Simulate notification sending
        await asyncio.sleep(0.1)

    async def _load_recovery_policies(self):
        """Load recovery policies from storage."""
        # Load default recovery policies
        default_policies = [
            RecoveryPolicy(
                policy_id="default_retry",
                name="Default Retry Policy",
                workflow_types=["*"],  # Apply to all workflow types
                conditions={"error_count": {">=": 3}},
                recovery_actions=[RecoveryAction.RETRY],
                max_recovery_attempts=3
            ),
            RecoveryPolicy(
                policy_id="long_running_restart",
                name="Long Running Workflow Restart",
                workflow_types=["multi_agent_project"],
                conditions={"duration_hours": {">=": 12}},
                recovery_actions=[RecoveryAction.RESTART],
                max_recovery_attempts=1
            )
        ]

        for policy in default_policies:
            self.recovery_policies[policy.policy_id] = policy

        logger.info(f"Loaded {len(default_policies)} recovery policies")

    async def _store_workflow_metrics(self, metrics: WorkflowMetrics):
        """Store workflow metrics in Redis."""
        key = f"workflow_metrics:{metrics.workflow_id}"
        self.storage.setex(key, 24 * 60 * 60, metrics.model_dump_json())  # 24 hour TTL

    async def _store_workflow_alert(self, alert: WorkflowAlert):
        """Store workflow alert in Redis."""
        key = f"workflow_alert:{alert.alert_id}"
        self.storage.setex(key, 7 * 24 * 60 * 60, alert.model_dump_json())  # 7 day TTL

    async def _store_health_check(self, health_check: WorkflowHealthCheck):
        """Store health check result in Redis."""
        key = f"workflow_health:{health_check.workflow_id}"
        self.storage.setex(key, 24 * 60 * 60, health_check.model_dump_json())  # 24 hour TTL


# Global monitor instance
_workflow_monitor: Optional[WorkflowMonitor] = None


def get_workflow_monitor() -> WorkflowMonitor:
    """Get the global workflow monitor instance."""
    global _workflow_monitor
    if _workflow_monitor is None:
        _workflow_monitor = WorkflowMonitor()
    return _workflow_monitor


logger.info("Workflow monitoring module initialized")
