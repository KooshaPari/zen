"""
Real-Time Analytics Pipeline for Agent Orchestration

This module provides real-time stream processing and analytics capabilities
for agent performance monitoring, anomaly detection, and business intelligence
using Kafka Streams and time-series analytics.

Key Features:
- Real-time agent performance analytics
- Cross-agent correlation analysis
- Anomaly detection for agent behavior
- Performance trend analysis and alerting
- Multi-agent task distribution analytics
- Time-series aggregations and windowing
- Custom metric definitions and calculations
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable

try:
    from kafka import KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False


from .kafka_events import AgentEvent, EventType, KafkaEventPublisher

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics for analytics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    RATE = "rate"
    DURATION = "duration"


class AggregationFunction(str, Enum):
    """Aggregation functions for time-series data."""

    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    MEDIAN = "median"
    P95 = "p95"
    P99 = "p99"
    STDDEV = "stddev"


@dataclass
class TimeWindow:
    """Time window configuration for analytics."""

    duration: timedelta
    slide: timedelta | None = None  # For sliding windows

    def __post_init__(self):
        if self.slide is None:
            self.slide = self.duration  # Tumbling window by default


@dataclass
class Metric:
    """Metric definition for analytics."""

    name: str
    metric_type: MetricType
    description: str
    labels: dict[str, str] = field(default_factory=dict)
    value: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "description": self.description,
            "labels": self.labels,
            "value": self.value,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PerformanceProfile:
    """Performance profile for an agent or system component."""

    agent_id: str
    response_time_avg: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    success_rate: float = 1.0
    error_rate: float = 0.0
    throughput: float = 0.0
    resource_usage: dict[str, float] = field(default_factory=dict)
    task_completion_rate: float = 1.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "response_time_avg": self.response_time_avg,
            "response_time_p95": self.response_time_p95,
            "response_time_p99": self.response_time_p99,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "throughput": self.throughput,
            "resource_usage": self.resource_usage,
            "task_completion_rate": self.task_completion_rate,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class AnomalyAlert:
    """Anomaly detection alert."""

    alert_id: str
    agent_id: str
    metric_name: str
    anomaly_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    threshold: float
    actual_value: float
    deviation: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "agent_id": self.agent_id,
            "metric_name": self.metric_name,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "description": self.description,
            "threshold": self.threshold,
            "actual_value": self.actual_value,
            "deviation": self.deviation,
            "timestamp": self.timestamp.isoformat()
        }


class TimeSeries:
    """Time series data structure with windowing and aggregation."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.data: deque = deque(maxlen=max_size)
        self._lock = asyncio.Lock()

    async def add_point(self, timestamp: datetime, value: float):
        """Add a data point to the time series."""
        async with self._lock:
            self.data.append((timestamp, value))

    async def get_range(self,
                       start_time: datetime,
                       end_time: datetime) -> list[tuple[datetime, float]]:
        """Get data points within time range."""
        async with self._lock:
            return [
                (ts, val) for ts, val in self.data
                if start_time <= ts <= end_time
            ]

    async def aggregate(self,
                       window: TimeWindow,
                       func: AggregationFunction,
                       end_time: datetime | None = None) -> list[tuple[datetime, float]]:
        """Aggregate data over time windows."""

        if not end_time:
            end_time = datetime.now(timezone.utc)

        start_time = end_time - window.duration
        data_points = await self.get_range(start_time, end_time)

        if not data_points:
            return []

        # Group data points by window
        window_groups: dict[datetime, list[float]] = defaultdict(list)

        current_window_start = start_time
        while current_window_start < end_time:
            window_end = current_window_start + window.duration

            for ts, value in data_points:
                if current_window_start <= ts < window_end:
                    window_groups[current_window_start].append(value)

            current_window_start += window.slide

        # Apply aggregation function to each window
        result = []
        for window_start, values in window_groups.items():
            if not values:
                continue

            if func == AggregationFunction.SUM:
                agg_value = sum(values)
            elif func == AggregationFunction.AVG:
                agg_value = statistics.mean(values)
            elif func == AggregationFunction.MIN:
                agg_value = min(values)
            elif func == AggregationFunction.MAX:
                agg_value = max(values)
            elif func == AggregationFunction.COUNT:
                agg_value = len(values)
            elif func == AggregationFunction.MEDIAN:
                agg_value = statistics.median(values)
            elif func == AggregationFunction.P95:
                agg_value = self._percentile(values, 0.95)
            elif func == AggregationFunction.P99:
                agg_value = self._percentile(values, 0.99)
            elif func == AggregationFunction.STDDEV:
                agg_value = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                agg_value = 0.0

            result.append((window_start, agg_value))

        return sorted(result)

    def _percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(percentile * len(sorted_values))
        if index >= len(sorted_values):
            index = len(sorted_values) - 1

        return sorted_values[index]


class AnomalyDetector:
    """Statistical anomaly detection for agent metrics."""

    def __init__(self,
                 sensitivity: float = 2.0,  # Standard deviations for threshold
                 min_data_points: int = 10):
        self.sensitivity = sensitivity
        self.min_data_points = min_data_points
        self.metric_baselines: dict[str, dict[str, float]] = {}

    async def update_baseline(self,
                            metric_name: str,
                            agent_id: str,
                            time_series: TimeSeries):
        """Update baseline statistics for a metric."""

        # Get recent data for baseline calculation
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)  # Use last hour for baseline

        data_points = await time_series.get_range(start_time, end_time)

        if len(data_points) < self.min_data_points:
            return  # Not enough data for baseline

        values = [point[1] for point in data_points]

        key = f"{agent_id}:{metric_name}"
        self.metric_baselines[key] = {
            "mean": statistics.mean(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
            "updated_at": end_time
        }

    def detect_anomaly(self,
                      metric_name: str,
                      agent_id: str,
                      current_value: float) -> AnomalyAlert | None:
        """Detect anomaly in current metric value."""

        key = f"{agent_id}:{metric_name}"
        baseline = self.metric_baselines.get(key)

        if not baseline:
            return None  # No baseline available

        mean = baseline["mean"]
        stddev = baseline["stddev"]

        if stddev == 0:
            return None  # No variance in baseline

        # Calculate z-score
        z_score = abs(current_value - mean) / stddev

        if z_score > self.sensitivity:
            # Anomaly detected
            severity = self._calculate_severity(z_score)

            return AnomalyAlert(
                alert_id=f"anomaly_{agent_id}_{metric_name}_{int(time.time())}",
                agent_id=agent_id,
                metric_name=metric_name,
                anomaly_type="statistical_outlier",
                severity=severity,
                description=f"{metric_name} value {current_value} deviates {z_score:.2f} standard deviations from baseline mean {mean:.2f}",
                threshold=mean + (self.sensitivity * stddev),
                actual_value=current_value,
                deviation=z_score
            )

        return None

    def _calculate_severity(self, z_score: float) -> str:
        """Calculate alert severity based on z-score."""
        if z_score > 5.0:
            return "critical"
        elif z_score > 3.5:
            return "high"
        elif z_score > 2.5:
            return "medium"
        else:
            return "low"


class AgentAnalyticsEngine:
    """Real-time analytics engine for agent orchestration."""

    def __init__(self,
                 kafka_consumer_config: dict[str, Any] = None,
                 event_publisher: KafkaEventPublisher | None = None):

        self.event_publisher = event_publisher
        self.consumer_config = kafka_consumer_config or {
            "bootstrap_servers": "localhost:9092",
            "group_id": "analytics-engine",
            "auto_offset_reset": "latest",
            "enable_auto_commit": True,
            "value_deserializer": lambda x: json.loads(x.decode('utf-8'))
        }

        self.consumer = None
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=5)

        # Analytics state
        self.metrics: dict[str, TimeSeries] = {}
        self.performance_profiles: dict[str, PerformanceProfile] = {}
        self.anomaly_detector = AnomalyDetector()

        # Event handlers
        self.event_processors: dict[EventType, Callable] = {
            EventType.TASK_STARTED: self._process_task_started,
            EventType.TASK_COMPLETED: self._process_task_completed,
            EventType.TASK_FAILED: self._process_task_failed,
            EventType.AGENT_HEARTBEAT: self._process_agent_heartbeat,
            EventType.PERFORMANCE_METRIC: self._process_performance_metric,
            EventType.TOOL_INVOKED: self._process_tool_invoked,
            EventType.TOOL_COMPLETED: self._process_tool_completed,
        }

        # Analytics windows
        self.windows = {
            "1min": TimeWindow(timedelta(minutes=1)),
            "5min": TimeWindow(timedelta(minutes=5)),
            "15min": TimeWindow(timedelta(minutes=15)),
            "1hour": TimeWindow(timedelta(hours=1)),
            "1day": TimeWindow(timedelta(days=1))
        }

    async def start(self):
        """Start the analytics engine."""
        if not KAFKA_AVAILABLE:
            logger.warning("Kafka not available, using mock analytics engine")
            self.is_running = True
            return

        try:
            self.consumer = KafkaConsumer(**self.consumer_config)

            # Subscribe to relevant topics
            topics = [
                "agent-events",
                "task-events",
                "workflow-events",
                "performance-events",
                "system-events"
            ]

            self.consumer.subscribe(topics)
            self.is_running = True

            # Start processing loop
            asyncio.create_task(self._processing_loop())

            logger.info("Analytics engine started")

        except Exception as e:
            logger.error(f"Failed to start analytics engine: {e}")
            raise

    async def stop(self):
        """Stop the analytics engine."""
        self.is_running = False

        if self.consumer:
            self.consumer.close()

        logger.info("Analytics engine stopped")

    async def _processing_loop(self):
        """Main event processing loop."""

        while self.is_running:
            try:
                if not KAFKA_AVAILABLE:
                    await asyncio.sleep(1)
                    continue

                # Poll for messages
                loop = asyncio.get_event_loop()
                messages = await loop.run_in_executor(
                    self.executor,
                    lambda: self.consumer.poll(timeout_ms=1000)
                )

                for _topic_partition, msgs in messages.items():
                    for message in msgs:
                        await self._process_event(message.value)

            except Exception as e:
                logger.error(f"Error in analytics processing loop: {e}")
                await asyncio.sleep(5)  # Back off on error

    async def _process_event(self, event_data: dict[str, Any]):
        """Process individual event."""

        try:
            event = AgentEvent.from_kafka_message(event_data)

            # Route to appropriate processor
            processor = self.event_processors.get(event.event_type)
            if processor:
                await processor(event)

            # Update anomaly detection baselines
            await self._update_anomaly_baselines(event)

        except Exception as e:
            logger.error(f"Error processing event: {e}")

    async def _process_task_started(self, event: AgentEvent):
        """Process task started event."""

        agent_id = event.payload.get("agent_id", "unknown")

        # Update task start count
        await self._increment_metric(f"tasks_started_{agent_id}")

        # Track task start time
        await self._set_gauge(f"task_start_time_{event.aggregate_id}", time.time())

    async def _process_task_completed(self, event: AgentEvent):
        """Process task completed event."""

        agent_id = event.payload.get("agent_id", "unknown")

        # Update completion metrics
        await self._increment_metric(f"tasks_completed_{agent_id}")

        # Calculate task duration
        start_time_key = f"task_start_time_{event.aggregate_id}"
        if start_time_key in self.metrics:
            start_time_ts = await self.metrics[start_time_key].get_range(
                datetime.now(timezone.utc) - timedelta(hours=1),
                datetime.now(timezone.utc)
            )

            if start_time_ts:
                duration = time.time() - start_time_ts[-1][1]
                await self._record_duration(f"task_duration_{agent_id}", duration)

                # Update performance profile
                await self._update_performance_profile(agent_id, duration, True)

    async def _process_task_failed(self, event: AgentEvent):
        """Process task failed event."""

        agent_id = event.payload.get("agent_id", "unknown")

        # Update failure metrics
        await self._increment_metric(f"tasks_failed_{agent_id}")

        # Update performance profile
        await self._update_performance_profile(agent_id, 0, False)

    async def _process_agent_heartbeat(self, event: AgentEvent):
        """Process agent heartbeat event."""

        agent_id = event.aggregate_id
        resource_usage = event.payload.get("resource_usage", {})

        # Record resource usage metrics
        for resource, value in resource_usage.items():
            await self._set_gauge(f"resource_{resource}_{agent_id}", float(value))

    async def _process_performance_metric(self, event: AgentEvent):
        """Process performance metric event."""

        agent_id = event.aggregate_id
        metric_name = event.payload.get("metric_name")
        metric_value = event.payload.get("metric_value")

        if metric_name and metric_value is not None:
            await self._set_gauge(f"{metric_name}_{agent_id}", float(metric_value))

    async def _process_tool_invoked(self, event: AgentEvent):
        """Process tool invocation event."""

        tool_name = event.payload.get("tool_name", "unknown")
        agent_id = event.payload.get("agent_id", "unknown")

        # Track tool usage
        await self._increment_metric(f"tool_invocations_{tool_name}")
        await self._increment_metric(f"agent_tool_invocations_{agent_id}")

        # Record invocation time
        await self._set_gauge(f"tool_invocation_time_{event.aggregate_id}", time.time())

    async def _process_tool_completed(self, event: AgentEvent):
        """Process tool completion event."""

        tool_name = event.payload.get("tool_name", "unknown")
        success = event.payload.get("success", True)

        if success:
            await self._increment_metric(f"tool_completions_{tool_name}")
        else:
            await self._increment_metric(f"tool_failures_{tool_name}")

        # Calculate tool execution duration
        invocation_time_key = f"tool_invocation_time_{event.aggregate_id}"
        if invocation_time_key in self.metrics:
            start_time_ts = await self.metrics[invocation_time_key].get_range(
                datetime.now(timezone.utc) - timedelta(hours=1),
                datetime.now(timezone.utc)
            )

            if start_time_ts:
                duration = time.time() - start_time_ts[-1][1]
                await self._record_duration(f"tool_duration_{tool_name}", duration)

    async def _increment_metric(self, metric_name: str, increment: float = 1.0):
        """Increment counter metric."""

        if metric_name not in self.metrics:
            self.metrics[metric_name] = TimeSeries()

        await self.metrics[metric_name].add_point(
            datetime.now(timezone.utc),
            increment
        )

    async def _set_gauge(self, metric_name: str, value: float):
        """Set gauge metric value."""

        if metric_name not in self.metrics:
            self.metrics[metric_name] = TimeSeries()

        await self.metrics[metric_name].add_point(
            datetime.now(timezone.utc),
            value
        )

    async def _record_duration(self, metric_name: str, duration: float):
        """Record duration metric."""

        if metric_name not in self.metrics:
            self.metrics[metric_name] = TimeSeries()

        await self.metrics[metric_name].add_point(
            datetime.now(timezone.utc),
            duration
        )

    async def _update_performance_profile(self, agent_id: str, duration: float, success: bool):
        """Update agent performance profile."""

        if agent_id not in self.performance_profiles:
            self.performance_profiles[agent_id] = PerformanceProfile(agent_id=agent_id)

        profile = self.performance_profiles[agent_id]

        # Get recent durations for statistics
        duration_metric = f"task_duration_{agent_id}"
        if duration_metric in self.metrics:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=15)  # Last 15 minutes

            recent_durations = await self.metrics[duration_metric].get_range(start_time, end_time)

            if recent_durations:
                durations = [point[1] for point in recent_durations]
                profile.response_time_avg = statistics.mean(durations)
                profile.response_time_p95 = self._percentile(durations, 0.95)
                profile.response_time_p99 = self._percentile(durations, 0.99)

        # Calculate success rate
        success_metric = f"tasks_completed_{agent_id}"
        failure_metric = f"tasks_failed_{agent_id}"

        if success_metric in self.metrics and failure_metric in self.metrics:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=15)

            success_points = await self.metrics[success_metric].get_range(start_time, end_time)
            failure_points = await self.metrics[failure_metric].get_range(start_time, end_time)

            total_successes = sum(point[1] for point in success_points)
            total_failures = sum(point[1] for point in failure_points)
            total_tasks = total_successes + total_failures

            if total_tasks > 0:
                profile.success_rate = total_successes / total_tasks
                profile.error_rate = total_failures / total_tasks
                profile.throughput = total_tasks / 15.0  # Tasks per minute

        profile.last_updated = datetime.now(timezone.utc)

    def _percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(percentile * len(sorted_values))
        if index >= len(sorted_values):
            index = len(sorted_values) - 1

        return sorted_values[index]

    async def _update_anomaly_baselines(self, event: AgentEvent):
        """Update anomaly detection baselines."""

        # Extract numeric values from event payload for anomaly detection
        numeric_fields = {}

        for key, value in event.payload.items():
            if isinstance(value, (int, float)):
                numeric_fields[key] = float(value)

        # Update baselines for numeric metrics
        for field_name, field_value in numeric_fields.items():
            metric_name = f"{event.event_type.value}_{field_name}"

            if metric_name not in self.metrics:
                self.metrics[metric_name] = TimeSeries()

            await self.metrics[metric_name].add_point(
                datetime.now(timezone.utc),
                field_value
            )

            # Update anomaly baseline
            await self.anomaly_detector.update_baseline(
                metric_name,
                event.aggregate_id,
                self.metrics[metric_name]
            )

            # Check for anomalies
            anomaly = self.anomaly_detector.detect_anomaly(
                metric_name,
                event.aggregate_id,
                field_value
            )

            if anomaly:
                await self._publish_anomaly_alert(anomaly)

    async def _publish_anomaly_alert(self, anomaly: AnomalyAlert):
        """Publish anomaly alert as event."""

        if self.event_publisher:
            from .kafka_events import AgentEvent, EventMetadata, EventType

            alert_event = AgentEvent(
                event_type=EventType.ANOMALY_DETECTED,
                aggregate_id=anomaly.agent_id,
                aggregate_type="agent",
                payload=anomaly.to_dict(),
                metadata=EventMetadata()
            )

            await self.event_publisher.publish_event(alert_event)

        logger.warning(f"Anomaly detected: {anomaly.description}")

    async def get_agent_analytics(self,
                                agent_id: str,
                                time_window: str = "1hour") -> dict[str, Any]:
        """Get comprehensive analytics for an agent."""

        if time_window not in self.windows:
            time_window = "1hour"

        window = self.windows[time_window]
        end_time = datetime.now(timezone.utc)
        start_time = end_time - window.duration

        analytics = {
            "agent_id": agent_id,
            "time_window": time_window,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "performance_profile": {},
            "metrics": {},
            "aggregations": {}
        }

        # Get performance profile
        if agent_id in self.performance_profiles:
            analytics["performance_profile"] = self.performance_profiles[agent_id].to_dict()

        # Get relevant metrics
        agent_metrics = [name for name in self.metrics.keys() if agent_id in name]

        for metric_name in agent_metrics:
            time_series = self.metrics[metric_name]
            data_points = await time_series.get_range(start_time, end_time)

            if data_points:
                values = [point[1] for point in data_points]
                analytics["metrics"][metric_name] = {
                    "count": len(values),
                    "sum": sum(values),
                    "avg": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1] if values else 0
                }

                # Add aggregations
                aggregations = {}
                for func in [AggregationFunction.AVG, AggregationFunction.MAX, AggregationFunction.COUNT]:
                    agg_data = await time_series.aggregate(window, func, end_time)
                    aggregations[func.value] = agg_data

                analytics["aggregations"][metric_name] = aggregations

        return analytics

    async def get_system_analytics(self, time_window: str = "1hour") -> dict[str, Any]:
        """Get system-wide analytics."""

        if time_window not in self.windows:
            time_window = "1hour"

        window = self.windows[time_window]
        end_time = datetime.now(timezone.utc)

        analytics = {
            "time_window": time_window,
            "end_time": end_time.isoformat(),
            "agents": {},
            "system_metrics": {},
            "top_performers": [],
            "alerts": []
        }

        # Get analytics for all agents
        agent_ids = set()
        for metric_name in self.metrics.keys():
            parts = metric_name.split('_')
            if len(parts) >= 2:
                potential_agent_id = parts[-1]
                if potential_agent_id.startswith('agent-') or len(potential_agent_id) > 8:
                    agent_ids.add(potential_agent_id)

        for agent_id in agent_ids:
            analytics["agents"][agent_id] = await self.get_agent_analytics(agent_id, time_window)

        # Calculate system metrics
        total_tasks = sum(
            profile.throughput * window.duration.total_seconds() / 60.0
            for profile in self.performance_profiles.values()
        )

        avg_response_time = statistics.mean([
            profile.response_time_avg
            for profile in self.performance_profiles.values()
        ]) if self.performance_profiles else 0

        system_success_rate = statistics.mean([
            profile.success_rate
            for profile in self.performance_profiles.values()
        ]) if self.performance_profiles else 1.0

        analytics["system_metrics"] = {
            "total_agents": len(self.performance_profiles),
            "total_tasks_completed": total_tasks,
            "average_response_time": avg_response_time,
            "system_success_rate": system_success_rate,
            "active_metrics": len(self.metrics)
        }

        # Get top performers
        top_performers = sorted(
            self.performance_profiles.values(),
            key=lambda p: (p.success_rate * p.throughput) - p.response_time_avg,
            reverse=True
        )[:5]

        analytics["top_performers"] = [p.to_dict() for p in top_performers]

        return analytics


# Global analytics engine instance
_analytics_engine: AgentAnalyticsEngine | None = None


async def get_analytics_engine() -> AgentAnalyticsEngine:
    """Get global analytics engine instance."""
    global _analytics_engine

    if _analytics_engine is None:
        from .kafka_events import get_event_publisher
        publisher = await get_event_publisher()
        _analytics_engine = AgentAnalyticsEngine(event_publisher=publisher)
        await _analytics_engine.start()

    return _analytics_engine


async def get_agent_performance_metrics(agent_id: str, time_window: str = "1hour") -> dict[str, Any]:
    """Convenience function to get agent performance metrics."""

    engine = await get_analytics_engine()
    return await engine.get_agent_analytics(agent_id, time_window)


async def get_system_performance_dashboard() -> dict[str, Any]:
    """Convenience function to get system performance dashboard."""

    engine = await get_analytics_engine()
    return await engine.get_system_analytics()
