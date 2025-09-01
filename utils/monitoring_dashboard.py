"""
Real-Time Monitoring Dashboard for Enterprise Agent Orchestration

This module provides comprehensive monitoring, metrics collection, alerting,
and dashboard functionality for the enterprise agent orchestration system.
"""

import asyncio
import json
import logging
import os
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Optional

import httpx
from pydantic import BaseModel

try:
    import redis
except ImportError:
    redis = None

try:
    import websockets
except ImportError:
    websockets = None

from utils.event_bus import get_event_bus

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class MetricType(str, Enum):
    """Types of metrics collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Individual metric data point."""

    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class Alert:
    """Alert definition and state."""

    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str  # Condition expression
    threshold: float
    triggered_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    is_active: bool = False
    notification_sent: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class SystemHealth(BaseModel):
    """Overall system health status."""

    status: str  # healthy, degraded, critical, unknown
    score: float  # 0.0 - 1.0
    components: dict[str, dict[str, Any]]
    active_alerts: int
    last_updated: datetime
    uptime_seconds: float


class MonitoringDashboard:
    """Real-time monitoring dashboard for agent orchestration."""

    def __init__(self,
                 redis_client=None,
                 websocket_port: int = 8765,
                 metrics_retention_hours: int = 24,
                 alert_check_interval: int = 30):
        """Initialize monitoring dashboard."""

        self.redis_client = redis_client or self._get_redis_client()
        self.websocket_port = websocket_port
        self.metrics_retention_hours = metrics_retention_hours
        self.alert_check_interval = alert_check_interval

        # In-memory storage
        self.metrics_store: dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alerts: dict[str, Alert] = {}
        self.alert_rules: dict[str, Callable] = {}
        self.connected_clients: set[websockets.WebSocketServerProtocol] = set()

        # System state tracking
        self.component_health: dict[str, dict[str, Any]] = {
            "agent_manager": {"status": "unknown", "last_check": None},
            "rabbitmq_queue": {"status": "unknown", "last_check": None},
            "a2a_protocol": {"status": "unknown", "last_check": None},
            "redis": {"status": "unknown", "last_check": None},
            "event_bus": {"status": "unknown", "last_check": None}
        }

        self.start_time = datetime.now(timezone.utc)
        self.monitoring_tasks: list[asyncio.Task] = []

        # Setup default alert rules
        self._setup_default_alerts()

        # Subscribe to system events
        asyncio.create_task(self._subscribe_to_events())

    def _get_redis_client(self):
        """Get Redis client for metrics storage."""
        try:
            client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "3")),  # Use DB 3 for monitoring
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            client.ping()
            logger.debug("Connected to Redis for monitoring")
            return client
        except Exception as e:
            logger.warning(f"Redis not available for monitoring: {e}")
            return None

    def _setup_default_alerts(self):
        """Setup default alert rules."""

        # High task failure rate
        self.add_alert_rule(
            name="high_task_failure_rate",
            description="Task failure rate exceeds 20%",
            severity=AlertSeverity.HIGH,
            condition=lambda: self._calculate_failure_rate() > 0.20,
            threshold=0.20
        )

        # Low system throughput
        self.add_alert_rule(
            name="low_system_throughput",
            description="System throughput below 1 task/second",
            severity=AlertSeverity.MEDIUM,
            condition=lambda: self._calculate_throughput() < 1.0,
            threshold=1.0
        )

        # High queue backlog
        self.add_alert_rule(
            name="high_queue_backlog",
            description="Queue backlog exceeds 1000 tasks",
            severity=AlertSeverity.MEDIUM,
            condition=lambda: self._get_queue_backlog() > 1000,
            threshold=1000
        )

        # Component down
        self.add_alert_rule(
            name="component_unavailable",
            description="Critical component is unavailable",
            severity=AlertSeverity.CRITICAL,
            condition=lambda: self._check_critical_components(),
            threshold=1
        )

        # High memory usage
        self.add_alert_rule(
            name="high_memory_usage",
            description="Memory usage exceeds 90%",
            severity=AlertSeverity.HIGH,
            condition=lambda: self._get_memory_usage() > 0.90,
            threshold=0.90
        )

    async def _subscribe_to_events(self):
        """Subscribe to system events for real-time monitoring."""
        try:
            event_bus = get_event_bus()

            # Define event handlers
            async def handle_task_event(event_data):
                await self._process_task_event(event_data)

            async def handle_system_event(event_data):
                await self._process_system_event(event_data)

            # Subscribe to events
            await event_bus.subscribe("task_*", handle_task_event)
            await event_bus.subscribe("agent_*", handle_system_event)
            await event_bus.subscribe("rabbitmq_*", handle_system_event)
            await event_bus.subscribe("a2a_*", handle_system_event)

            logger.info("Subscribed to system events for monitoring")

        except Exception as e:
            logger.warning(f"Failed to subscribe to events: {e}")

    async def _process_task_event(self, event_data: dict[str, Any]):
        """Process task-related events."""
        event_type = event_data.get("event", "")
        timestamp = datetime.now(timezone.utc)

        if event_type == "task_created":
            await self.record_metric("tasks_created_total", 1, MetricType.COUNTER,
                                    labels={"agent": event_data.get("agent", "unknown")})

        elif event_type == "task_completed":
            await self.record_metric("tasks_completed_total", 1, MetricType.COUNTER,
                                    labels={"status": event_data.get("status", "unknown")})

            # Record execution time if available
            exec_time = event_data.get("execution_time")
            if exec_time:
                await self.record_metric("task_execution_time", exec_time, MetricType.HISTOGRAM,
                                        labels={"agent": event_data.get("agent", "unknown")})

        elif event_type == "task_failed":
            await self.record_metric("tasks_failed_total", 1, MetricType.COUNTER,
                                    labels={"error": event_data.get("error", "unknown")})

        # Broadcast to connected clients
        await self._broadcast_to_clients({
            "type": "task_event",
            "data": event_data,
            "timestamp": timestamp.isoformat()
        })

    async def _process_system_event(self, event_data: dict[str, Any]):
        """Process system-related events."""
        event_type = event_data.get("event", "")
        timestamp = datetime.now(timezone.utc)

        # Update component health based on events
        if "rabbitmq" in event_type:
            self.component_health["rabbitmq_queue"]["last_check"] = timestamp
            if "connected" in event_type:
                self.component_health["rabbitmq_queue"]["status"] = "healthy"
            elif "error" in event_type or "failed" in event_type:
                self.component_health["rabbitmq_queue"]["status"] = "unhealthy"

        elif "agent_" in event_type:
            if "registered" in event_type or "advertised" in event_type:
                self.component_health["a2a_protocol"]["status"] = "healthy"
                self.component_health["a2a_protocol"]["last_check"] = timestamp

        # Record system metrics
        if event_type == "agent_registered":
            await self.record_metric("agents_registered_total", 1, MetricType.COUNTER)

        # Broadcast to clients
        await self._broadcast_to_clients({
            "type": "system_event",
            "data": event_data,
            "timestamp": timestamp.isoformat()
        })

    async def record_metric(self,
                           name: str,
                           value: float,
                           metric_type: MetricType,
                           labels: dict[str, str] = None,
                           unit: str = ""):
        """Record a metric data point."""
        timestamp = datetime.now(timezone.utc)
        labels = labels or {}

        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=timestamp,
            labels=labels,
            unit=unit
        )

        # Store in memory
        metric_key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        self.metrics_store[metric_key].append(metric)

        # Store in Redis for persistence
        if self.redis_client:
            try:
                redis_key = f"metrics:{metric_key}:{int(timestamp.timestamp())}"
                metric_data = {
                    "name": name,
                    "value": value,
                    "type": metric_type.value,
                    "labels": json.dumps(labels),
                    "unit": unit,
                    "timestamp": timestamp.isoformat()
                }

                # Store with TTL based on retention hours
                ttl_seconds = self.metrics_retention_hours * 3600
                self.redis_client.setex(redis_key, ttl_seconds, json.dumps(metric_data))

                # Maintain time-series index
                ts_key = f"metrics_ts:{name}"
                self.redis_client.zadd(ts_key, {redis_key: timestamp.timestamp()})
                self.redis_client.expire(ts_key, ttl_seconds)

            except Exception as e:
                logger.debug(f"Failed to store metric in Redis: {e}")

        # Broadcast to connected clients
        await self._broadcast_to_clients({
            "type": "metric_update",
            "data": {
                "name": name,
                "value": value,
                "type": metric_type.value,
                "labels": labels,
                "timestamp": timestamp.isoformat()
            }
        })

    def add_alert_rule(self,
                      name: str,
                      description: str,
                      severity: AlertSeverity,
                      condition: Callable[[], bool],
                      threshold: float):
        """Add an alert rule."""
        alert_id = str(uuid.uuid4())

        alert = Alert(
            alert_id=alert_id,
            name=name,
            description=description,
            severity=severity,
            condition=str(condition),
            threshold=threshold
        )

        self.alerts[alert_id] = alert
        self.alert_rules[alert_id] = condition

        logger.info(f"Added alert rule: {name} ({severity.value})")

    async def check_alerts(self):
        """Check all alert rules and trigger/resolve alerts."""
        current_time = datetime.now(timezone.utc)

        for alert_id, condition_func in self.alert_rules.items():
            alert = self.alerts[alert_id]

            try:
                # Evaluate condition
                is_triggered = condition_func()

                if is_triggered and not alert.is_active:
                    # Trigger alert
                    alert.is_active = True
                    alert.triggered_at = current_time
                    alert.notification_sent = False

                    logger.warning(f"ALERT TRIGGERED: {alert.name} - {alert.description}")

                    # Send notification
                    await self._send_alert_notification(alert)

                    # Broadcast to clients
                    await self._broadcast_to_clients({
                        "type": "alert_triggered",
                        "data": {
                            "alert_id": alert_id,
                            "name": alert.name,
                            "description": alert.description,
                            "severity": alert.severity.value,
                            "triggered_at": alert.triggered_at.isoformat()
                        }
                    })

                elif not is_triggered and alert.is_active:
                    # Resolve alert
                    alert.is_active = False
                    alert.resolved_at = current_time

                    logger.info(f"ALERT RESOLVED: {alert.name}")

                    # Broadcast to clients
                    await self._broadcast_to_clients({
                        "type": "alert_resolved",
                        "data": {
                            "alert_id": alert_id,
                            "name": alert.name,
                            "resolved_at": alert.resolved_at.isoformat()
                        }
                    })

            except Exception as e:
                logger.error(f"Error checking alert {alert.name}: {e}")

    async def _send_alert_notification(self, alert: Alert):
        """Send alert notification (webhook, email, etc.)."""
        try:
            # Webhook notification
            webhook_url = os.getenv("ALERT_WEBHOOK_URL")
            if webhook_url:
                async with httpx.AsyncClient(timeout=10) as client:
                    payload = {
                        "alert": {
                            "id": alert.alert_id,
                            "name": alert.name,
                            "description": alert.description,
                            "severity": alert.severity.value,
                            "triggered_at": alert.triggered_at.isoformat() if alert.triggered_at else None,
                            "threshold": alert.threshold
                        },
                        "system": {
                            "hostname": os.getenv("HOSTNAME", "unknown"),
                            "service": "zen-mcp-server"
                        }
                    }

                    await client.post(webhook_url, json=payload)
                    alert.notification_sent = True
                    logger.debug(f"Sent webhook notification for alert: {alert.name}")

            # Log-based notification (always available)
            logger.critical(
                f"ALERT: {alert.name} | {alert.description} | "
                f"Severity: {alert.severity.value} | Threshold: {alert.threshold}"
            )

        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")

    async def get_system_health(self) -> SystemHealth:
        """Get overall system health status."""
        current_time = datetime.now(timezone.utc)
        uptime = (current_time - self.start_time).total_seconds()

        # Check component health
        await self._update_component_health()

        # Calculate health score
        healthy_components = sum(1 for c in self.component_health.values() if c["status"] == "healthy")
        total_components = len(self.component_health)
        health_score = healthy_components / total_components if total_components > 0 else 0.0

        # Determine overall status
        active_alerts = sum(1 for a in self.alerts.values() if a.is_active)
        critical_alerts = sum(1 for a in self.alerts.values() if a.is_active and a.severity == AlertSeverity.CRITICAL)

        if critical_alerts > 0 or health_score < 0.5:
            status = "critical"
        elif active_alerts > 0 or health_score < 0.8:
            status = "degraded"
        elif health_score >= 0.9:
            status = "healthy"
        else:
            status = "unknown"

        return SystemHealth(
            status=status,
            score=health_score,
            components=self.component_health,
            active_alerts=active_alerts,
            last_updated=current_time,
            uptime_seconds=uptime
        )

    async def _update_component_health(self):
        """Update health status of system components."""
        current_time = datetime.now(timezone.utc)

        # Check Redis
        if self.redis_client:
            try:
                self.redis_client.ping()
                self.component_health["redis"]["status"] = "healthy"
            except Exception:
                self.component_health["redis"]["status"] = "unhealthy"
        else:
            self.component_health["redis"]["status"] = "unavailable"
        self.component_health["redis"]["last_check"] = current_time

        # Check RabbitMQ (via metrics)
        try:
            from utils.rabbitmq_queue import get_queue_manager
            manager = get_queue_manager()
            if hasattr(manager, 'connection') and manager.connection:
                self.component_health["rabbitmq_queue"]["status"] = "healthy"
            else:
                self.component_health["rabbitmq_queue"]["status"] = "unknown"
        except Exception:
            self.component_health["rabbitmq_queue"]["status"] = "unhealthy"

        # Event bus is always considered healthy if we can import it
        try:
            get_event_bus()
            self.component_health["event_bus"]["status"] = "healthy"
        except Exception:
            self.component_health["event_bus"]["status"] = "unhealthy"
        self.component_health["event_bus"]["last_check"] = current_time

    async def get_metrics(self,
                         name_pattern: str = None,
                         time_range_minutes: int = 60,
                         labels: dict[str, str] = None) -> list[Metric]:
        """Get metrics matching criteria."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=time_range_minutes)

        matching_metrics = []

        for metric_key, metric_deque in self.metrics_store.items():
            # Parse metric key
            if ":" in metric_key:
                metric_name = metric_key.split(":")[0]
            else:
                metric_name = metric_key

            # Check name pattern
            if name_pattern and name_pattern not in metric_name:
                continue

            # Get metrics in time range
            for metric in metric_deque:
                if start_time <= metric.timestamp <= end_time:
                    # Check label filters
                    if labels:
                        if not all(metric.labels.get(k) == v for k, v in labels.items()):
                            continue

                    matching_metrics.append(metric)

        return sorted(matching_metrics, key=lambda m: m.timestamp)

    async def get_dashboard_data(self) -> dict[str, Any]:
        """Get comprehensive dashboard data."""
        health = await self.get_system_health()
        recent_metrics = await self.get_metrics(time_range_minutes=30)

        # Aggregate key metrics
        task_metrics = {
            "total_created": len([m for m in recent_metrics if m.name == "tasks_created_total"]),
            "total_completed": len([m for m in recent_metrics if m.name == "tasks_completed_total"]),
            "total_failed": len([m for m in recent_metrics if m.name == "tasks_failed_total"]),
            "avg_execution_time": 0.0
        }

        exec_times = [m.value for m in recent_metrics if m.name == "task_execution_time"]
        if exec_times:
            task_metrics["avg_execution_time"] = sum(exec_times) / len(exec_times)

        # Active alerts
        active_alerts = [
            {
                "id": a.alert_id,
                "name": a.name,
                "description": a.description,
                "severity": a.severity.value,
                "triggered_at": a.triggered_at.isoformat() if a.triggered_at else None
            }
            for a in self.alerts.values() if a.is_active
        ]

        return {
            "health": health.model_dump(),
            "task_metrics": task_metrics,
            "active_alerts": active_alerts,
            "component_status": self.component_health,
            "connected_clients": len(self.connected_clients),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

    # Alert condition helper methods

    def _calculate_failure_rate(self) -> float:
        """Calculate recent task failure rate."""
        recent_metrics = []
        for metric_deque in self.metrics_store.values():
            recent_metrics.extend([
                m for m in metric_deque
                if m.timestamp > datetime.now(timezone.utc) - timedelta(minutes=15)
            ])

        completed = len([m for m in recent_metrics if m.name == "tasks_completed_total"])
        failed = len([m for m in recent_metrics if m.name == "tasks_failed_total"])

        total = completed + failed
        return failed / total if total > 0 else 0.0

    def _calculate_throughput(self) -> float:
        """Calculate recent throughput (tasks/second)."""
        recent_metrics = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=5)

        for metric_deque in self.metrics_store.values():
            recent_metrics.extend([
                m for m in metric_deque
                if m.timestamp > cutoff_time and m.name in ["tasks_created_total", "tasks_completed_total"]
            ])

        if not recent_metrics:
            return 0.0

        duration_seconds = (datetime.now(timezone.utc) - cutoff_time).total_seconds()
        return len(recent_metrics) / duration_seconds if duration_seconds > 0 else 0.0

    def _get_queue_backlog(self) -> int:
        """Get current queue backlog (simulated)."""
        # In real implementation, this would query RabbitMQ management API
        try:
            from utils.rabbitmq_queue import get_queue_manager
            get_queue_manager()
            # Return simulated backlog based on recent metrics
            return len([
                m for m in self.metrics_store.get("tasks_created_total:", [])
                if m.timestamp > datetime.now(timezone.utc) - timedelta(minutes=10)
            ])
        except Exception:
            return 0

    def _check_critical_components(self) -> bool:
        """Check if any critical components are down."""
        critical_components = ["agent_manager", "event_bus"]

        for component in critical_components:
            if self.component_health.get(component, {}).get("status") == "unhealthy":
                return True

        return False

    def _get_memory_usage(self) -> float:
        """Get current memory usage (simulated)."""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            # Simulate memory usage based on metrics store size
            total_metrics = sum(len(deque) for deque in self.metrics_store.values())
            return min(total_metrics / 100000.0, 0.95)  # Cap at 95%

    # WebSocket server for real-time updates

    async def _websocket_handler(self, websocket, path):
        """Handle WebSocket connections for real-time updates."""
        logger.info(f"New WebSocket client connected from {websocket.remote_address}")
        self.connected_clients.add(websocket)

        try:
            # Send initial dashboard data
            dashboard_data = await self.get_dashboard_data()
            await websocket.send(json.dumps({
                "type": "dashboard_data",
                "data": dashboard_data
            }))

            # Keep connection alive and handle messages
            async for message in websocket:
                try:
                    request = json.loads(message)
                    await self._handle_websocket_request(websocket, request)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON"
                    }))
                except Exception as e:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.connected_clients.discard(websocket)

    async def _handle_websocket_request(self, websocket, request: dict[str, Any]):
        """Handle specific WebSocket requests."""
        request_type = request.get("type")

        if request_type == "get_metrics":
            name_pattern = request.get("name_pattern")
            time_range = request.get("time_range_minutes", 60)
            metrics = await self.get_metrics(name_pattern, time_range)

            await websocket.send(json.dumps({
                "type": "metrics_data",
                "data": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "type": m.metric_type.value,
                        "timestamp": m.timestamp.isoformat(),
                        "labels": m.labels
                    } for m in metrics
                ]
            }))

        elif request_type == "get_health":
            health = await self.get_system_health()
            await websocket.send(json.dumps({
                "type": "health_data",
                "data": health.model_dump()
            }))

        elif request_type == "acknowledge_alert":
            alert_id = request.get("alert_id")
            if alert_id in self.alerts:
                self.alerts[alert_id].notification_sent = True
                await websocket.send(json.dumps({
                    "type": "alert_acknowledged",
                    "alert_id": alert_id
                }))

    async def _broadcast_to_clients(self, message: dict[str, Any]):
        """Broadcast message to all connected WebSocket clients."""
        if not self.connected_clients:
            return

        message_str = json.dumps(message)
        disconnected_clients = set()

        for client in self.connected_clients:
            try:
                await client.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.debug(f"Failed to send message to client: {e}")
                disconnected_clients.add(client)

        # Remove disconnected clients
        self.connected_clients -= disconnected_clients

    async def start_monitoring(self):
        """Start all monitoring services."""
        logger.info("Starting monitoring dashboard...")

        # Start WebSocket server if websockets is available
        if websockets:
            try:
                websocket_server = await websockets.serve(
                    self._websocket_handler,
                    "localhost",
                    self.websocket_port
                )
                self.monitoring_tasks.append(asyncio.create_task(websocket_server.wait_closed()))
                logger.info(f"WebSocket server started on port {self.websocket_port}")
            except Exception as e:
                logger.warning(f"Failed to start WebSocket server: {e}")

        # Start alert checking loop
        async def alert_loop():
            while True:
                try:
                    await self.check_alerts()
                    await asyncio.sleep(self.alert_check_interval)
                except Exception as e:
                    logger.error(f"Alert checking error: {e}")
                    await asyncio.sleep(self.alert_check_interval)

        self.monitoring_tasks.append(asyncio.create_task(alert_loop()))

        # Start periodic health updates
        async def health_update_loop():
            while True:
                try:
                    # Update component health
                    await self._update_component_health()

                    # Broadcast health update to clients
                    health = await self.get_system_health()
                    await self._broadcast_to_clients({
                        "type": "health_update",
                        "data": health.model_dump()
                    })

                    await asyncio.sleep(60)  # Update every minute
                except Exception as e:
                    logger.error(f"Health update error: {e}")
                    await asyncio.sleep(60)

        self.monitoring_tasks.append(asyncio.create_task(health_update_loop()))

        logger.info("Monitoring dashboard started successfully")

    async def stop_monitoring(self):
        """Stop all monitoring services."""
        logger.info("Stopping monitoring dashboard...")

        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.monitoring_tasks.clear()

        # Close WebSocket connections
        if self.connected_clients:
            await asyncio.gather(
                *[client.close() for client in self.connected_clients],
                return_exceptions=True
            )
            self.connected_clients.clear()

        logger.info("Monitoring dashboard stopped")


# Global monitoring dashboard instance
_monitoring_dashboard: Optional[MonitoringDashboard] = None


def get_monitoring_dashboard() -> MonitoringDashboard:
    """Get the global monitoring dashboard instance."""
    global _monitoring_dashboard
    if _monitoring_dashboard is None:
        _monitoring_dashboard = MonitoringDashboard()
    return _monitoring_dashboard


# Example usage and HTML dashboard template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Agent Orchestration Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel { background: #2a2a2a; padding: 20px; border-radius: 8px; border: 1px solid #444; }
        .metric { margin: 10px 0; }
        .alert { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .alert.critical { background: #dc3545; }
        .alert.high { background: #fd7e14; }
        .alert.medium { background: #ffc107; color: #000; }
        .alert.low { background: #17a2b8; }
        .health.healthy { color: #28a745; }
        .health.degraded { color: #ffc107; }
        .health.critical { color: #dc3545; }
        .component { margin: 5px 0; padding: 5px; background: #333; border-radius: 4px; }
        .component.healthy { border-left: 4px solid #28a745; }
        .component.unhealthy { border-left: 4px solid #dc3545; }
        .component.unknown { border-left: 4px solid #6c757d; }
    </style>
</head>
<body>
    <h1>Enterprise Agent Orchestration Dashboard</h1>

    <div class="dashboard">
        <div class="panel">
            <h2>System Health</h2>
            <div id="health-status"></div>
            <div id="components"></div>
        </div>

        <div class="panel">
            <h2>Task Metrics</h2>
            <div id="task-metrics"></div>
        </div>

        <div class="panel">
            <h2>Active Alerts</h2>
            <div id="alerts"></div>
        </div>

        <div class="panel">
            <h2>System Info</h2>
            <div id="system-info"></div>
        </div>
    </div>

    <script>
        const ws = new WebSocket('ws://localhost:8765');

        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            handleMessage(message);
        };

        function handleMessage(message) {
            if (message.type === 'dashboard_data') {
                updateDashboard(message.data);
            } else if (message.type === 'health_update') {
                updateHealth(message.data);
            } else if (message.type === 'alert_triggered' || message.type === 'alert_resolved') {
                updateAlerts();
            }
        }

        function updateDashboard(data) {
            updateHealth(data.health);
            updateTaskMetrics(data.task_metrics);
            updateAlerts(data.active_alerts);
            updateSystemInfo(data);
        }

        function updateHealth(health) {
            document.getElementById('health-status').innerHTML = `
                <div class="health ${health.status}">
                    <strong>Status:</strong> ${health.status.toUpperCase()}<br>
                    <strong>Score:</strong> ${(health.score * 100).toFixed(1)}%<br>
                    <strong>Active Alerts:</strong> ${health.active_alerts}<br>
                    <strong>Uptime:</strong> ${Math.floor(health.uptime_seconds / 3600)}h ${Math.floor((health.uptime_seconds % 3600) / 60)}m
                </div>
            `;

            const components = Object.entries(health.components).map(([name, info]) => `
                <div class="component ${info.status}">
                    <strong>${name}:</strong> ${info.status}
                </div>
            `).join('');
            document.getElementById('components').innerHTML = components;
        }

        function updateTaskMetrics(metrics) {
            document.getElementById('task-metrics').innerHTML = `
                <div class="metric"><strong>Created:</strong> ${metrics.total_created}</div>
                <div class="metric"><strong>Completed:</strong> ${metrics.total_completed}</div>
                <div class="metric"><strong>Failed:</strong> ${metrics.total_failed}</div>
                <div class="metric"><strong>Avg Execution Time:</strong> ${metrics.avg_execution_time.toFixed(2)}s</div>
            `;
        }

        function updateAlerts(alerts = []) {
            if (alerts.length === 0) {
                document.getElementById('alerts').innerHTML = '<div class="metric">No active alerts</div>';
                return;
            }

            const alertsHtml = alerts.map(alert => `
                <div class="alert ${alert.severity}">
                    <strong>${alert.name}</strong><br>
                    ${alert.description}<br>
                    <small>Triggered: ${new Date(alert.triggered_at).toLocaleString()}</small>
                </div>
            `).join('');
            document.getElementById('alerts').innerHTML = alertsHtml;
        }

        function updateSystemInfo(data) {
            document.getElementById('system-info').innerHTML = `
                <div class="metric"><strong>Connected Clients:</strong> ${data.connected_clients}</div>
                <div class="metric"><strong>Last Updated:</strong> ${new Date(data.last_updated).toLocaleString()}</div>
            `;
        }
    </script>
</body>
</html>
"""
