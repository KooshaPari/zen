"""
NATS Real-time Metrics and Monitoring System

This module provides comprehensive monitoring and metrics collection for NATS
infrastructure with enterprise-grade observability features:
- Real-time performance metrics collection and aggregation
- Multi-dimensional metric tracking (latency, throughput, errors)
- Health monitoring with automated alerting
- Dashboard data aggregation and visualization support
- Historical metrics storage and trend analysis
- SLA monitoring and service level indicators
- Distributed tracing integration
- Custom metrics and business KPIs
"""

import asyncio
import json
import logging
import os
import statistics
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import redis
from pydantic import BaseModel, Field

from utils.agent_discovery import AgentDiscoveryService, get_discovery_service
from utils.edge_orchestrator import EdgeOrchestrator, get_edge_orchestrator
from utils.nats_communicator import NATSCommunicator, get_nats_communicator
from utils.nats_streaming import NATSStreamingManager, get_streaming_manager

logger = logging.getLogger(__name__)


class MetricDataPoint(BaseModel):
    """Individual metric data point."""

    timestamp: datetime = Field(..., description="Measurement timestamp")
    value: float = Field(..., description="Metric value")
    labels: dict[str, str] = Field(default_factory=dict, description="Metric labels/tags")


class MetricSeries(BaseModel):
    """Time series of metric data points."""

    name: str = Field(..., description="Metric name")
    description: str = Field(default="", description="Metric description")
    unit: str = Field(default="", description="Metric unit")
    metric_type: str = Field(..., description="Metric type: counter, gauge, histogram, summary")
    data_points: list[MetricDataPoint] = Field(default_factory=list, description="Data points")
    labels: dict[str, str] = Field(default_factory=dict, description="Common labels")

    def add_data_point(self, value: float, labels: Optional[dict[str, str]] = None,
                      timestamp: Optional[datetime] = None) -> None:
        """Add a data point to the series."""
        point = MetricDataPoint(
            timestamp=timestamp or datetime.now(timezone.utc),
            value=value,
            labels=labels or {}
        )
        self.data_points.append(point)

        # Maintain rolling window to prevent memory bloat
        max_points = int(os.getenv("NATS_METRICS_MAX_POINTS", "10000"))
        if len(self.data_points) > max_points:
            self.data_points = self.data_points[-max_points:]

    def get_latest_value(self) -> Optional[float]:
        """Get the most recent value."""
        return self.data_points[-1].value if self.data_points else None

    def get_average(self, duration_minutes: Optional[int] = None) -> Optional[float]:
        """Get average value over specified duration."""
        if not self.data_points:
            return None

        if duration_minutes is None:
            values = [dp.value for dp in self.data_points]
        else:
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=duration_minutes)
            values = [dp.value for dp in self.data_points if dp.timestamp >= cutoff_time]

        return statistics.mean(values) if values else None

    def get_percentile(self, percentile: float, duration_minutes: Optional[int] = None) -> Optional[float]:
        """Get percentile value over specified duration."""
        if not self.data_points:
            return None

        if duration_minutes is None:
            values = [dp.value for dp in self.data_points]
        else:
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=duration_minutes)
            values = [dp.value for dp in self.data_points if dp.timestamp >= cutoff_time]

        if not values:
            return None

        values.sort()
        k = (len(values) - 1) * (percentile / 100.0)
        f = int(k)
        c = k - f

        if f + 1 < len(values):
            return values[f] * (1 - c) + values[f + 1] * c
        else:
            return values[f]


class HealthCheckResult(BaseModel):
    """Health check result."""

    check_name: str = Field(..., description="Health check name")
    status: str = Field(..., description="Health status: healthy, degraded, unhealthy")
    message: str = Field(default="", description="Status message")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = Field(..., description="Check duration in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AlertRule(BaseModel):
    """Alert rule configuration."""

    rule_id: str = Field(..., description="Unique rule identifier")
    metric_name: str = Field(..., description="Metric name to monitor")
    condition: str = Field(..., description="Alert condition: gt, lt, eq, ne")
    threshold: float = Field(..., description="Alert threshold value")
    duration_minutes: int = Field(default=5, description="Duration before alerting")
    severity: str = Field(default="warning", description="Alert severity: info, warning, critical")
    description: str = Field(..., description="Alert description")
    labels: dict[str, str] = Field(default_factory=dict, description="Alert labels")
    enabled: bool = Field(default=True, description="Whether rule is enabled")


class Alert(BaseModel):
    """Active alert."""

    alert_id: str = Field(..., description="Unique alert identifier")
    rule: AlertRule = Field(..., description="Alert rule that triggered")
    current_value: float = Field(..., description="Current metric value")
    triggered_at: datetime = Field(..., description="When alert was triggered")
    resolved_at: Optional[datetime] = Field(None, description="When alert was resolved")
    status: str = Field(default="firing", description="Alert status: firing, resolved")
    annotations: dict[str, str] = Field(default_factory=dict, description="Alert annotations")


class NATSMonitoringSystem:
    """
    Comprehensive NATS monitoring and metrics system with real-time alerting.
    """

    def __init__(self, nats_communicator: Optional[NATSCommunicator] = None,
                 streaming_manager: Optional[NATSStreamingManager] = None,
                 discovery_service: Optional[AgentDiscoveryService] = None,
                 edge_orchestrator: Optional[EdgeOrchestrator] = None,
                 redis_client: Optional[redis.Redis] = None):
        """Initialize monitoring system."""
        self.nats = nats_communicator
        self.streaming = streaming_manager
        self.discovery = discovery_service
        self.edge = edge_orchestrator
        self.redis_client = redis_client

        # Metrics storage
        self.metrics: dict[str, MetricSeries] = {}
        self.metrics_lock = asyncio.Lock()

        # Health monitoring
        self.health_checks: dict[str, Callable] = {}
        self.health_results: dict[str, HealthCheckResult] = {}
        self.health_history: list[HealthCheckResult] = []

        # Alerting
        self.alert_rules: dict[str, AlertRule] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []

        # Performance tracking
        self.latency_tracker = deque(maxlen=10000)  # Rolling latency measurements
        self.throughput_tracker = defaultdict(lambda: deque(maxlen=1000))
        self.error_tracker = defaultdict(int)

        # Monitoring configuration
        self.monitoring_interval = 30  # seconds
        self.health_check_interval = 60  # seconds
        self.metrics_retention_hours = 24
        self.alert_check_interval = 30  # seconds

        # Background tasks
        self._metrics_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._alert_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        self._initialize_default_metrics()
        self._initialize_default_health_checks()
        self._initialize_default_alerts()

        logger.info("NATS Monitoring System initialized")

    async def start(self) -> None:
        """Start the monitoring system."""
        if not self.nats:
            self.nats = await get_nats_communicator(self.redis_client)

        if not self.streaming:
            self.streaming = await get_streaming_manager(self.nats, self.redis_client)

        if not self.discovery:
            self.discovery = await get_discovery_service(self.nats, self.redis_client)

        if not self.edge:
            self.edge = await get_edge_orchestrator(self.nats, self.discovery, self.redis_client)

        # Set up NATS subscriptions for metrics
        await self._setup_metrics_subscriptions()

        # Start background monitoring tasks
        self._metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self._health_task = asyncio.create_task(self._health_monitoring_loop())
        self._alert_task = asyncio.create_task(self._alert_processing_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("NATS Monitoring System started")

    async def stop(self) -> None:
        """Stop the monitoring system."""
        self._shutdown_event.set()

        # Cancel background tasks
        for task in [self._metrics_task, self._health_task, self._alert_task, self._cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("NATS Monitoring System stopped")

    def _initialize_default_metrics(self) -> None:
        """Initialize default metrics to track."""
        default_metrics = [
            # NATS Core Metrics
            ("nats_connections_total", "Total NATS connections", "connections", "gauge"),
            ("nats_messages_published_total", "Total messages published", "messages", "counter"),
            ("nats_messages_consumed_total", "Total messages consumed", "messages", "counter"),
            ("nats_message_latency_ms", "Message latency", "milliseconds", "histogram"),
            ("nats_bytes_published_total", "Total bytes published", "bytes", "counter"),
            ("nats_bytes_consumed_total", "Total bytes consumed", "bytes", "counter"),
            ("nats_errors_total", "Total NATS errors", "errors", "counter"),

            # JetStream Metrics
            ("jetstream_streams_total", "Total JetStream streams", "streams", "gauge"),
            ("jetstream_consumers_total", "Total JetStream consumers", "consumers", "gauge"),
            ("jetstream_messages_stored_total", "Total messages stored", "messages", "gauge"),
            ("jetstream_bytes_stored_total", "Total bytes stored", "bytes", "gauge"),
            ("jetstream_acks_total", "Total message acknowledgments", "acks", "counter"),
            ("jetstream_naks_total", "Total message negative acknowledgments", "naks", "counter"),

            # Agent Discovery Metrics
            ("discovery_services_total", "Total registered services", "services", "gauge"),
            ("discovery_requests_total", "Total discovery requests", "requests", "counter"),
            ("discovery_request_latency_ms", "Discovery request latency", "milliseconds", "histogram"),

            # Edge Orchestration Metrics
            ("edge_locations_total", "Total edge locations", "locations", "gauge"),
            ("edge_placements_total", "Total workload placements", "placements", "counter"),
            ("edge_placement_success_rate", "Placement success rate", "percentage", "gauge"),

            # System Health Metrics
            ("system_health_score", "Overall system health score", "score", "gauge"),
            ("component_health_status", "Component health status", "status", "gauge"),

            # Performance Metrics
            ("cpu_usage_percent", "CPU usage", "percentage", "gauge"),
            ("memory_usage_bytes", "Memory usage", "bytes", "gauge"),
            ("disk_usage_bytes", "Disk usage", "bytes", "gauge"),
            ("network_rx_bytes", "Network received bytes", "bytes", "counter"),
            ("network_tx_bytes", "Network transmitted bytes", "bytes", "counter"),
        ]

        for name, description, unit, metric_type in default_metrics:
            self.metrics[name] = MetricSeries(
                name=name,
                description=description,
                unit=unit,
                metric_type=metric_type
            )

    def _initialize_default_health_checks(self) -> None:
        """Initialize default health checks."""
        self.health_checks.update({
            "nats_connectivity": self._check_nats_connectivity,
            "jetstream_status": self._check_jetstream_status,
            "discovery_service": self._check_discovery_service,
            "edge_orchestrator": self._check_edge_orchestrator,
            "redis_connectivity": self._check_redis_connectivity,
            "system_resources": self._check_system_resources,
        })

    def _initialize_default_alerts(self) -> None:
        """Initialize default alert rules."""
        default_alerts = [
            AlertRule(
                rule_id="high_message_latency",
                metric_name="nats_message_latency_ms",
                condition="gt",
                threshold=1000.0,  # 1 second
                duration_minutes=5,
                severity="warning",
                description="High NATS message latency detected"
            ),
            AlertRule(
                rule_id="critical_message_latency",
                metric_name="nats_message_latency_ms",
                condition="gt",
                threshold=5000.0,  # 5 seconds
                duration_minutes=2,
                severity="critical",
                description="Critical NATS message latency detected"
            ),
            AlertRule(
                rule_id="high_error_rate",
                metric_name="nats_errors_total",
                condition="gt",
                threshold=100.0,
                duration_minutes=5,
                severity="warning",
                description="High NATS error rate detected"
            ),
            AlertRule(
                rule_id="low_health_score",
                metric_name="system_health_score",
                condition="lt",
                threshold=70.0,
                duration_minutes=3,
                severity="warning",
                description="System health score below threshold"
            ),
            AlertRule(
                rule_id="edge_placement_failure",
                metric_name="edge_placement_success_rate",
                condition="lt",
                threshold=80.0,
                duration_minutes=10,
                severity="warning",
                description="Edge placement success rate below threshold"
            ),
        ]

        for alert_rule in default_alerts:
            self.alert_rules[alert_rule.rule_id] = alert_rule

    async def _setup_metrics_subscriptions(self) -> None:
        """Set up NATS subscriptions for metrics collection."""
        if not self.nats:
            return

        # Subscribe to metrics from all components
        await self.nats.subscribe(
            "metrics.>",
            self._handle_metrics_message,
            queue_group="monitoring"
        )

        # Subscribe to performance events
        await self.nats.subscribe(
            "performance.>",
            self._handle_performance_message,
            queue_group="monitoring"
        )

        # Subscribe to health events
        await self.nats.subscribe(
            "health.>",
            self._handle_health_message,
            queue_group="monitoring"
        )

    async def record_metric(self, name: str, value: float, labels: Optional[dict[str, str]] = None,
                          timestamp: Optional[datetime] = None) -> None:
        """Record a metric value."""
        try:
            async with self.metrics_lock:
                if name not in self.metrics:
                    # Create new metric series
                    self.metrics[name] = MetricSeries(
                        name=name,
                        description=f"Custom metric: {name}",
                        unit="",
                        metric_type="gauge"
                    )

                self.metrics[name].add_data_point(value, labels, timestamp)

                # Store in Redis for persistence
                if self.redis_client:
                    try:
                        key = f"metrics:{name}"
                        metric_data = {
                            "value": value,
                            "labels": labels or {},
                            "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat()
                        }

                        # Use Redis sorted set for time-series data
                        score = time.time()
                        self.redis_client.zadd(key, {json.dumps(metric_data): score})

                        # Set expiration
                        self.redis_client.expire(key, 86400 * 7)  # 7 days

                    except Exception as e:
                        logger.debug(f"Failed to store metric in Redis: {e}")

        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")

    async def get_metric(self, name: str) -> Optional[MetricSeries]:
        """Get metric series by name."""
        async with self.metrics_lock:
            return self.metrics.get(name)

    async def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of all metrics."""
        async with self.metrics_lock:
            summary = {}

            for name, series in self.metrics.items():
                latest_value = series.get_latest_value()
                avg_5min = series.get_average(5)
                avg_1hour = series.get_average(60)

                summary[name] = {
                    "latest_value": latest_value,
                    "avg_5min": avg_5min,
                    "avg_1hour": avg_1hour,
                    "data_points_count": len(series.data_points),
                    "unit": series.unit,
                    "type": series.metric_type
                }

            return summary

    async def add_health_check(self, name: str, check_func: Callable) -> None:
        """Add a custom health check."""
        self.health_checks[name] = check_func

    async def run_health_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check."""
        if name not in self.health_checks:
            return None

        start_time = time.perf_counter()

        try:
            check_func = self.health_checks[name]
            result = await check_func()

            duration_ms = (time.perf_counter() - start_time) * 1000

            if isinstance(result, HealthCheckResult):
                result.duration_ms = duration_ms
                return result
            else:
                # Convert simple result to HealthCheckResult
                return HealthCheckResult(
                    check_name=name,
                    status="healthy" if result else "unhealthy",
                    message=str(result) if isinstance(result, str) else "",
                    duration_ms=duration_ms
                )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Health check {name} failed: {e}")

            return HealthCheckResult(
                check_name=name,
                status="unhealthy",
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms
            )

    async def get_system_health(self) -> dict[str, Any]:
        """Get overall system health status."""
        health_results = []

        # Run all health checks
        for name in self.health_checks.keys():
            result = await self.run_health_check(name)
            if result:
                health_results.append(result)
                self.health_results[name] = result

        # Calculate overall health score
        if health_results:
            healthy_count = sum(1 for r in health_results if r.status == "healthy")
            degraded_count = sum(1 for r in health_results if r.status == "degraded")

            # Health score: 100% for all healthy, 50% for degraded, 0% for unhealthy
            health_score = (healthy_count * 100 + degraded_count * 50) / len(health_results)
        else:
            health_score = 0

        # Record health score metric
        await self.record_metric("system_health_score", health_score)

        return {
            "health_score": health_score,
            "status": "healthy" if health_score >= 80 else "degraded" if health_score >= 50 else "unhealthy",
            "checks": {result.check_name: {
                "status": result.status,
                "message": result.message,
                "duration_ms": result.duration_ms,
                "timestamp": result.timestamp.isoformat()
            } for result in health_results},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.alert_rules[rule.rule_id] = rule

    async def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            return True
        return False

    async def get_active_alerts(self) -> list[Alert]:
        """Get all active alerts."""
        return [alert for alert in self.active_alerts.values() if alert.status == "firing"]

    async def get_dashboard_data(self) -> dict[str, Any]:
        """Get comprehensive dashboard data."""
        # Get metrics summary
        metrics_summary = await self.get_metrics_summary()

        # Get system health
        system_health = await self.get_system_health()

        # Get active alerts
        active_alerts = await self.get_active_alerts()

        # Get component status
        component_status = {}

        if self.nats:
            nats_metrics = await self.nats.get_performance_metrics()
            component_status["nats"] = {
                "connected": nats_metrics["connected"],
                "message_count": nats_metrics["message_count"],
                "error_count": nats_metrics["error_count"],
                "avg_latency_ms": nats_metrics["avg_latency_ms"]
            }

        if self.streaming:
            streaming_metrics = await self.streaming.get_streaming_metrics()
            component_status["jetstream"] = {
                "messages_published": streaming_metrics["messages_published"],
                "messages_consumed": streaming_metrics["messages_consumed"],
                "ack_rate": streaming_metrics["ack_rate"],
                "active_streams": streaming_metrics["active_streams"]
            }

        if self.discovery:
            discovery_status = await self.discovery.get_registry_status()
            component_status["discovery"] = {
                "total_services": discovery_status["total_services"],
                "discovery_requests": discovery_status["discovery_requests"]
            }

        if self.edge:
            edge_metrics = await self.edge.get_orchestrator_metrics()
            component_status["edge"] = {
                "successful_placements": edge_metrics["successful_placements"],
                "failed_placements": edge_metrics["failed_placements"],
                "success_rate": edge_metrics["success_rate"]
            }

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_health": system_health,
            "metrics_summary": metrics_summary,
            "component_status": component_status,
            "active_alerts": [alert.model_dump() for alert in active_alerts],
            "alert_summary": {
                "total": len(active_alerts),
                "critical": len([a for a in active_alerts if a.rule.severity == "critical"]),
                "warning": len([a for a in active_alerts if a.rule.severity == "warning"]),
                "info": len([a for a in active_alerts if a.rule.severity == "info"])
            }
        }

    # Default health check implementations
    async def _check_nats_connectivity(self) -> HealthCheckResult:
        """Check NATS connectivity."""
        if not self.nats or not self.nats.connected:
            return HealthCheckResult(
                check_name="nats_connectivity",
                status="unhealthy",
                message="NATS not connected",
                duration_ms=0
            )

        try:
            # Test connectivity with a ping-like operation
            test_subject = "health.test.nats"
            test_message = {"timestamp": datetime.now(timezone.utc).isoformat()}

            success = await self.nats.publish(test_subject, test_message)

            return HealthCheckResult(
                check_name="nats_connectivity",
                status="healthy" if success else "unhealthy",
                message="NATS connection verified" if success else "NATS publish test failed",
                duration_ms=0  # Will be set by caller
            )

        except Exception as e:
            return HealthCheckResult(
                check_name="nats_connectivity",
                status="unhealthy",
                message=f"NATS connectivity test failed: {str(e)}",
                duration_ms=0
            )

    async def _check_jetstream_status(self) -> HealthCheckResult:
        """Check JetStream status."""
        if not self.streaming or not self.streaming.js:
            return HealthCheckResult(
                check_name="jetstream_status",
                status="unhealthy",
                message="JetStream not available",
                duration_ms=0
            )

        try:
            # Get account info to verify JetStream is working
            account_info = await self.streaming.js.account_info()

            # Check if we're approaching limits
            memory_usage = account_info.memory / account_info.limits.max_memory if account_info.limits.max_memory > 0 else 0
            storage_usage = account_info.storage / account_info.limits.max_storage if account_info.limits.max_storage > 0 else 0

            status = "healthy"
            message = "JetStream operating normally"

            if memory_usage > 0.9 or storage_usage > 0.9:
                status = "degraded"
                message = f"JetStream resource usage high (memory: {memory_usage:.1%}, storage: {storage_usage:.1%})"
            elif memory_usage > 0.95 or storage_usage > 0.95:
                status = "unhealthy"
                message = f"JetStream resource usage critical (memory: {memory_usage:.1%}, storage: {storage_usage:.1%})"

            return HealthCheckResult(
                check_name="jetstream_status",
                status=status,
                message=message,
                duration_ms=0,
                metadata={
                    "memory_usage": memory_usage,
                    "storage_usage": storage_usage,
                    "streams": account_info.streams,
                    "consumers": account_info.consumers
                }
            )

        except Exception as e:
            return HealthCheckResult(
                check_name="jetstream_status",
                status="unhealthy",
                message=f"JetStream health check failed: {str(e)}",
                duration_ms=0
            )

    async def _check_discovery_service(self) -> HealthCheckResult:
        """Check agent discovery service."""
        if not self.discovery:
            return HealthCheckResult(
                check_name="discovery_service",
                status="degraded",
                message="Discovery service not available",
                duration_ms=0
            )

        try:
            registry_status = await self.discovery.get_registry_status()

            total_services = registry_status["total_services"]

            status = "healthy"
            message = f"Discovery service operational with {total_services} services"

            if total_services == 0:
                status = "degraded"
                message = "Discovery service operational but no services registered"

            return HealthCheckResult(
                check_name="discovery_service",
                status=status,
                message=message,
                duration_ms=0,
                metadata=registry_status
            )

        except Exception as e:
            return HealthCheckResult(
                check_name="discovery_service",
                status="unhealthy",
                message=f"Discovery service health check failed: {str(e)}",
                duration_ms=0
            )

    async def _check_edge_orchestrator(self) -> HealthCheckResult:
        """Check edge orchestrator."""
        if not self.edge:
            return HealthCheckResult(
                check_name="edge_orchestrator",
                status="degraded",
                message="Edge orchestrator not available",
                duration_ms=0
            )

        try:
            edge_metrics = await self.edge.get_orchestrator_metrics()

            success_rate = edge_metrics["success_rate"]
            total_locations = edge_metrics["locations"]["total_locations"]

            status = "healthy"
            message = f"Edge orchestrator operational with {total_locations} locations"

            if success_rate < 0.8:
                status = "degraded"
                message = f"Edge orchestrator success rate low: {success_rate:.1%}"
            elif success_rate < 0.5:
                status = "unhealthy"
                message = f"Edge orchestrator success rate critical: {success_rate:.1%}"

            return HealthCheckResult(
                check_name="edge_orchestrator",
                status=status,
                message=message,
                duration_ms=0,
                metadata={
                    "success_rate": success_rate,
                    "total_locations": total_locations,
                    "successful_placements": edge_metrics["successful_placements"],
                    "failed_placements": edge_metrics["failed_placements"]
                }
            )

        except Exception as e:
            return HealthCheckResult(
                check_name="edge_orchestrator",
                status="unhealthy",
                message=f"Edge orchestrator health check failed: {str(e)}",
                duration_ms=0
            )

    async def _check_redis_connectivity(self) -> HealthCheckResult:
        """Check Redis connectivity."""
        if not self.redis_client:
            return HealthCheckResult(
                check_name="redis_connectivity",
                status="degraded",
                message="Redis client not configured",
                duration_ms=0
            )

        try:
            # Test Redis with a ping
            response = self.redis_client.ping()

            if response:
                return HealthCheckResult(
                    check_name="redis_connectivity",
                    status="healthy",
                    message="Redis connection verified",
                    duration_ms=0
                )
            else:
                return HealthCheckResult(
                    check_name="redis_connectivity",
                    status="unhealthy",
                    message="Redis ping failed",
                    duration_ms=0
                )

        except Exception as e:
            return HealthCheckResult(
                check_name="redis_connectivity",
                status="unhealthy",
                message=f"Redis connectivity test failed: {str(e)}",
                duration_ms=0
            )

    async def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage."""
        try:
            import psutil

            # Get system resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Record metrics
            await self.record_metric("cpu_usage_percent", cpu_percent)
            await self.record_metric("memory_usage_bytes", memory.used)
            await self.record_metric("disk_usage_bytes", disk.used)

            # Determine status
            status = "healthy"
            issues = []

            if cpu_percent > 90:
                status = "unhealthy"
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent > 80:
                status = "degraded"
                issues.append(f"CPU usage high: {cpu_percent:.1f}%")

            if memory.percent > 95:
                status = "unhealthy"
                issues.append(f"Memory usage critical: {memory.percent:.1f}%")
            elif memory.percent > 85:
                if status == "healthy":
                    status = "degraded"
                issues.append(f"Memory usage high: {memory.percent:.1f}%")

            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 95:
                status = "unhealthy"
                issues.append(f"Disk usage critical: {disk_percent:.1f}%")
            elif disk_percent > 85:
                if status == "healthy":
                    status = "degraded"
                issues.append(f"Disk usage high: {disk_percent:.1f}%")

            message = "; ".join(issues) if issues else f"System resources normal (CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%, Disk: {disk_percent:.1f}%)"

            return HealthCheckResult(
                check_name="system_resources",
                status=status,
                message=message,
                duration_ms=0,
                metadata={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available": memory.available,
                    "disk_percent": disk_percent,
                    "disk_free": disk.free
                }
            )

        except ImportError:
            return HealthCheckResult(
                check_name="system_resources",
                status="degraded",
                message="psutil not available for system monitoring",
                duration_ms=0
            )
        except Exception as e:
            return HealthCheckResult(
                check_name="system_resources",
                status="unhealthy",
                message=f"System resource check failed: {str(e)}",
                duration_ms=0
            )

    # Message handlers
    async def _handle_metrics_message(self, data: dict[str, Any]) -> None:
        """Handle incoming metrics messages."""
        try:
            # Extract metric information from message
            metric_name = data.get("metric_name")
            metric_value = data.get("value")
            labels = data.get("labels", {})

            if metric_name is not None and metric_value is not None:
                await self.record_metric(metric_name, float(metric_value), labels)

        except Exception as e:
            logger.error(f"Failed to handle metrics message: {e}")

    async def _handle_performance_message(self, data: dict[str, Any]) -> None:
        """Handle performance measurement messages."""
        try:
            # Track latency measurements
            latency = data.get("latency_ms")
            if latency is not None:
                self.latency_tracker.append(float(latency))
                await self.record_metric("nats_message_latency_ms", float(latency))

            # Track throughput by operation type
            operation = data.get("operation", "unknown")
            timestamp = time.time()
            self.throughput_tracker[operation].append(timestamp)

        except Exception as e:
            logger.error(f"Failed to handle performance message: {e}")

    async def _handle_health_message(self, data: dict[str, Any]) -> None:
        """Handle health status messages."""
        try:
            component = data.get("component")
            status = data.get("status")

            if component and status:
                # Record component health as metric
                health_value = 1.0 if status == "healthy" else 0.5 if status == "degraded" else 0.0
                await self.record_metric("component_health_status", health_value, {"component": component})

        except Exception as e:
            logger.error(f"Failed to handle health message: {e}")

    # Background tasks
    async def _metrics_collection_loop(self) -> None:
        """Collect metrics from all components periodically."""
        while not self._shutdown_event.is_set():
            try:
                # Collect NATS metrics
                if self.nats:
                    nats_metrics = await self.nats.get_performance_metrics()
                    await self.record_metric("nats_connections_total", 1 if nats_metrics["connected"] else 0)
                    await self.record_metric("nats_messages_published_total", nats_metrics["message_count"])
                    await self.record_metric("nats_errors_total", nats_metrics["error_count"])

                # Collect JetStream metrics
                if self.streaming:
                    streaming_metrics = await self.streaming.get_streaming_metrics()
                    await self.record_metric("jetstream_messages_stored_total", streaming_metrics["messages_published"])
                    await self.record_metric("jetstream_acks_total", streaming_metrics["ack_count"])
                    await self.record_metric("jetstream_naks_total", streaming_metrics["nak_count"])

                # Collect discovery metrics
                if self.discovery:
                    discovery_status = await self.discovery.get_registry_status()
                    await self.record_metric("discovery_services_total", discovery_status["total_services"])
                    await self.record_metric("discovery_requests_total", discovery_status["discovery_requests"])

                # Collect edge metrics
                if self.edge:
                    edge_metrics = await self.edge.get_orchestrator_metrics()
                    await self.record_metric("edge_placements_total", edge_metrics["successful_placements"] + edge_metrics["failed_placements"])
                    await self.record_metric("edge_placement_success_rate", edge_metrics["success_rate"] * 100)

                # Publish metrics snapshot
                if self.nats:
                    metrics_summary = await self.get_metrics_summary()
                    await self.nats.publish("system.metrics.snapshot", metrics_summary, use_jetstream=True)

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5)

    async def _health_monitoring_loop(self) -> None:
        """Monitor system health periodically."""
        while not self._shutdown_event.is_set():
            try:
                # Get system health
                system_health = await self.get_system_health()

                # Publish health status
                if self.nats:
                    await self.nats.publish("system.health.status", system_health, use_jetstream=True)

                # Store health results history
                for result in self.health_results.values():
                    self.health_history.append(result)

                # Limit history size
                if len(self.health_history) > 1000:
                    self.health_history = self.health_history[-1000:]

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(10)

    async def _alert_processing_loop(self) -> None:
        """Process alert rules periodically."""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.now(timezone.utc)

                for rule in self.alert_rules.values():
                    if not rule.enabled:
                        continue

                    # Get metric series
                    metric = await self.get_metric(rule.metric_name)
                    if not metric:
                        continue

                    # Check if condition is met
                    current_value = metric.get_latest_value()
                    if current_value is None:
                        continue

                    condition_met = False

                    if rule.condition == "gt":
                        condition_met = current_value > rule.threshold
                    elif rule.condition == "lt":
                        condition_met = current_value < rule.threshold
                    elif rule.condition == "eq":
                        condition_met = abs(current_value - rule.threshold) < 0.001
                    elif rule.condition == "ne":
                        condition_met = abs(current_value - rule.threshold) >= 0.001

                    alert_id = f"{rule.rule_id}_{rule.metric_name}"

                    if condition_met:
                        # Check if alert already exists
                        if alert_id not in self.active_alerts:
                            # Create new alert
                            alert = Alert(
                                alert_id=alert_id,
                                rule=rule,
                                current_value=current_value,
                                triggered_at=current_time,
                                status="firing"
                            )

                            self.active_alerts[alert_id] = alert

                            # Publish alert
                            if self.nats:
                                alert_data = {
                                    "alert_id": alert_id,
                                    "rule_id": rule.rule_id,
                                    "severity": rule.severity,
                                    "description": rule.description,
                                    "current_value": current_value,
                                    "threshold": rule.threshold,
                                    "triggered_at": current_time.isoformat(),
                                    "status": "firing"
                                }
                                await self.nats.publish("system.alerts.fired", alert_data, use_jetstream=True)

                            logger.warning(f"Alert fired: {rule.description} (value: {current_value}, threshold: {rule.threshold})")

                    else:
                        # Check if we need to resolve an existing alert
                        if alert_id in self.active_alerts:
                            alert = self.active_alerts[alert_id]
                            alert.resolved_at = current_time
                            alert.status = "resolved"

                            # Move to history
                            self.alert_history.append(alert)
                            del self.active_alerts[alert_id]

                            # Publish resolution
                            if self.nats:
                                resolution_data = {
                                    "alert_id": alert_id,
                                    "rule_id": rule.rule_id,
                                    "resolved_at": current_time.isoformat(),
                                    "status": "resolved"
                                }
                                await self.nats.publish("system.alerts.resolved", resolution_data, use_jetstream=True)

                            logger.info(f"Alert resolved: {rule.description}")

                await asyncio.sleep(self.alert_check_interval)

            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(10)

    async def _cleanup_loop(self) -> None:
        """Clean up old metrics and alerts."""
        while not self._shutdown_event.is_set():
            try:
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.metrics_retention_hours)

                # Clean up old metric data points
                async with self.metrics_lock:
                    for series in self.metrics.values():
                        series.data_points = [
                            dp for dp in series.data_points
                            if dp.timestamp > cutoff_time
                        ]

                # Clean up old alerts
                self.alert_history = [
                    alert for alert in self.alert_history
                    if alert.resolved_at and alert.resolved_at > cutoff_time
                ]

                # Clean up old health results
                self.health_history = [
                    result for result in self.health_history
                    if result.timestamp > cutoff_time
                ]

                await asyncio.sleep(3600)  # Run cleanup every hour

            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(300)


# Global monitoring system instance
_monitoring_system: Optional[NATSMonitoringSystem] = None


async def get_monitoring_system(nats_communicator: Optional[NATSCommunicator] = None,
                               streaming_manager: Optional[NATSStreamingManager] = None,
                               discovery_service: Optional[AgentDiscoveryService] = None,
                               edge_orchestrator: Optional[EdgeOrchestrator] = None,
                               redis_client: Optional[redis.Redis] = None) -> NATSMonitoringSystem:
    """Get or create the global monitoring system instance."""
    global _monitoring_system

    if _monitoring_system is None:
        _monitoring_system = NATSMonitoringSystem(
            nats_communicator, streaming_manager, discovery_service, edge_orchestrator, redis_client
        )
        await _monitoring_system.start()

    return _monitoring_system


async def shutdown_monitoring_system() -> None:
    """Shutdown the global monitoring system."""
    global _monitoring_system

    if _monitoring_system:
        await _monitoring_system.stop()
        _monitoring_system = None
