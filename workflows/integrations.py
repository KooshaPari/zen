"""
Workflow Integration Layer

This module provides integration between Temporal workflows and the existing
Zen MCP Server infrastructure including Redis state management, NATS messaging,
and Kafka event streaming.

Features:
- Redis integration for workflow state persistence
- NATS integration for real-time workflow communication
- Kafka integration for workflow event streaming
- Agent API integration for workflow-driven agent coordination
- Event bus bridge for workflow lifecycle events
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import uuid4

# Optional NATS integration
try:
    import nats
    from nats.js import JetStreamContext
    NATS_AVAILABLE = True
except ImportError:
    NATS_AVAILABLE = False
    nats = None
    JetStreamContext = None

# Optional Kafka integration
try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    AIOKafkaProducer = None
    AIOKafkaConsumer = None

from tools.shared.agent_models import AgentTaskRequest, AgentType, TaskStatus
from utils.agent_manager import get_task_manager
from utils.event_bus import get_event_bus
from utils.storage_backend import get_storage_backend

logger = logging.getLogger(__name__)


class WorkflowStateManager:
    """
    Manages workflow state using Redis with advanced features like
    state snapshots, rollback capabilities, and cross-workflow coordination.
    """

    def __init__(self):
        self.storage = get_storage_backend()
        self.state_ttl = 7 * 24 * 60 * 60  # 7 days default TTL

    async def store_workflow_state(
        self,
        workflow_id: str,
        state: dict[str, Any],
        version: Optional[int] = None,
        ttl: Optional[int] = None
    ) -> int:
        """Store workflow state with versioning support."""
        current_version = version or await self._get_next_version(workflow_id)

        state_record = {
            "workflow_id": workflow_id,
            "state": state,
            "version": current_version,
            "timestamp": datetime.utcnow().isoformat(),
            "ttl": ttl or self.state_ttl
        }

        # Store current state
        current_key = f"workflow_state:{workflow_id}"
        self.storage.setex(current_key, ttl or self.state_ttl, json.dumps(state_record))

        # Store versioned state for history
        version_key = f"workflow_state:{workflow_id}:v{current_version}"
        self.storage.setex(version_key, ttl or self.state_ttl, json.dumps(state_record))

        # Update version counter
        version_key = f"workflow_version:{workflow_id}"
        self.storage.setex(version_key, ttl or self.state_ttl, str(current_version))

        logger.debug(f"Stored workflow state for {workflow_id}, version {current_version}")
        return current_version

    async def get_workflow_state(
        self,
        workflow_id: str,
        version: Optional[int] = None
    ) -> Optional[dict[str, Any]]:
        """Retrieve workflow state, optionally by version."""
        if version:
            key = f"workflow_state:{workflow_id}:v{version}"
        else:
            key = f"workflow_state:{workflow_id}"

        data = self.storage.get(key)
        if data:
            state_record = json.loads(data)
            return state_record["state"]

        return None

    async def create_state_snapshot(
        self,
        workflow_id: str,
        snapshot_name: str,
        description: Optional[str] = None
    ) -> bool:
        """Create a named snapshot of workflow state."""
        current_state = await self.get_workflow_state(workflow_id)
        if not current_state:
            return False

        current_version = await self._get_current_version(workflow_id)

        snapshot_record = {
            "workflow_id": workflow_id,
            "snapshot_name": snapshot_name,
            "description": description,
            "state": current_state,
            "version": current_version,
            "created_at": datetime.utcnow().isoformat()
        }

        snapshot_key = f"workflow_snapshot:{workflow_id}:{snapshot_name}"
        self.storage.setex(snapshot_key, self.state_ttl, json.dumps(snapshot_record))

        # Add to snapshots index
        index_key = f"workflow_snapshots:{workflow_id}"
        snapshots = self.storage.get(index_key)
        snapshot_list = json.loads(snapshots) if snapshots else []

        if snapshot_name not in snapshot_list:
            snapshot_list.append(snapshot_name)
            self.storage.setex(index_key, self.state_ttl, json.dumps(snapshot_list))

        logger.info(f"Created state snapshot '{snapshot_name}' for workflow {workflow_id}")
        return True

    async def restore_from_snapshot(
        self,
        workflow_id: str,
        snapshot_name: str
    ) -> bool:
        """Restore workflow state from a named snapshot."""
        snapshot_key = f"workflow_snapshot:{workflow_id}:{snapshot_name}"
        data = self.storage.get(snapshot_key)

        if not data:
            logger.warning(f"Snapshot '{snapshot_name}' not found for workflow {workflow_id}")
            return False

        snapshot_record = json.loads(data)
        restored_state = snapshot_record["state"]

        # Store as new version
        new_version = await self.store_workflow_state(workflow_id, restored_state)

        logger.info(f"Restored workflow {workflow_id} from snapshot '{snapshot_name}' as version {new_version}")
        return True

    async def _get_current_version(self, workflow_id: str) -> int:
        """Get current version number for workflow."""
        version_key = f"workflow_version:{workflow_id}"
        version_data = self.storage.get(version_key)
        return int(version_data) if version_data else 0

    async def _get_next_version(self, workflow_id: str) -> int:
        """Get next version number for workflow."""
        current_version = await self._get_current_version(workflow_id)
        return current_version + 1


class NATSWorkflowIntegration:
    """
    Integration with NATS for real-time workflow coordination and messaging.
    Provides pub/sub capabilities for workflow events and agent coordination.
    """

    def __init__(self):
        self.nc: Optional[nats.NATS] = None
        self.js: Optional[JetStreamContext] = None
        self.connected = False
        self.event_bus = get_event_bus()

    async def connect(
        self,
        servers: Optional[list[str]] = None,
        max_reconnect_attempts: int = 10
    ) -> bool:
        """Connect to NATS server."""
        if not NATS_AVAILABLE:
            logger.warning("NATS not available - workflow integration disabled")
            return False

        servers = servers or [os.getenv("NATS_URL", "nats://localhost:4222")]

        try:
            self.nc = await nats.connect(
                servers=servers,
                max_reconnect_attempts=max_reconnect_attempts,
                reconnected_cb=self._on_reconnected,
                disconnected_cb=self._on_disconnected,
                error_cb=self._on_error
            )

            # Setup JetStream for persistent messaging
            self.js = self.nc.jetstream()

            # Create workflow-specific streams
            await self._setup_workflow_streams()

            self.connected = True
            logger.info(f"Connected to NATS: {servers}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            return False

    async def disconnect(self):
        """Disconnect from NATS server."""
        if self.nc and self.connected:
            await self.nc.close()
            self.connected = False
            logger.info("Disconnected from NATS")

    async def publish_workflow_event(
        self,
        subject: str,
        event_data: dict[str, Any],
        workflow_id: Optional[str] = None
    ):
        """Publish workflow event to NATS."""
        if not self.connected or not self.nc:
            logger.warning("NATS not connected - cannot publish event")
            return

        event_payload = {
            "event_id": str(uuid4()),
            "workflow_id": workflow_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": event_data
        }

        try:
            # Publish to JetStream for persistence
            if self.js:
                await self.js.publish(
                    subject=subject,
                    payload=json.dumps(event_payload).encode()
                )
            else:
                # Fallback to core NATS
                await self.nc.publish(
                    subject=subject,
                    payload=json.dumps(event_payload).encode()
                )

            logger.debug(f"Published workflow event to {subject}")

        except Exception as e:
            logger.error(f"Failed to publish workflow event: {e}")

    async def subscribe_to_workflow_events(
        self,
        subjects: list[str],
        callback: callable,
        queue_group: Optional[str] = None
    ):
        """Subscribe to workflow events from NATS."""
        if not self.connected or not self.nc:
            logger.warning("NATS not connected - cannot subscribe")
            return

        async def message_handler(msg):
            try:
                event_data = json.loads(msg.data.decode())
                await callback(msg.subject, event_data)
            except Exception as e:
                logger.error(f"Error handling workflow event: {e}")

        for subject in subjects:
            try:
                if queue_group:
                    await self.nc.subscribe(subject, queue=queue_group, cb=message_handler)
                else:
                    await self.nc.subscribe(subject, cb=message_handler)

                logger.info(f"Subscribed to workflow events on {subject}")

            except Exception as e:
                logger.error(f"Failed to subscribe to {subject}: {e}")

    async def coordinate_multi_agent_task(
        self,
        task_id: str,
        agent_assignments: list[dict[str, Any]],
        coordination_type: str = "parallel"
    ) -> dict[str, Any]:
        """Coordinate multi-agent task execution via NATS."""

        # Publish task assignments
        for assignment in agent_assignments:
            agent_subject = f"workflow.agent.{assignment['agent_type']}"

            await self.publish_workflow_event(
                subject=agent_subject,
                event_data={
                    "event": "agent_task_assigned",
                    "task_id": task_id,
                    "assignment": assignment,
                    "coordination_type": coordination_type
                },
                workflow_id=assignment.get("workflow_id")
            )

        # Wait for completion signals
        completed_agents = set()
        results = {}

        def completion_handler(subject: str, event_data: dict[str, Any]):
            if event_data["data"].get("task_id") == task_id:
                agent_id = event_data["data"].get("agent_id")
                if agent_id:
                    completed_agents.add(agent_id)
                    results[agent_id] = event_data["data"]

        # Subscribe to completion events
        await self.subscribe_to_workflow_events(
            subjects=["workflow.agent.*.completed"],
            callback=completion_handler
        )

        # Wait for all agents to complete or timeout
        timeout_seconds = 300  # 5 minutes
        start_time = datetime.utcnow()

        while (
            len(completed_agents) < len(agent_assignments) and
            (datetime.utcnow() - start_time).total_seconds() < timeout_seconds
        ):
            await asyncio.sleep(1)

        return {
            "task_id": task_id,
            "coordination_type": coordination_type,
            "completed_agents": len(completed_agents),
            "total_agents": len(agent_assignments),
            "results": results,
            "timed_out": len(completed_agents) < len(agent_assignments)
        }

    async def _setup_workflow_streams(self):
        """Setup NATS JetStream streams for workflow events."""
        if not self.js:
            return

        try:
            # Workflow lifecycle stream
            await self.js.add_stream(
                name="workflow-lifecycle",
                subjects=["workflow.lifecycle.*"],
                retention="limits",
                max_msgs=10000,
                max_age=7 * 24 * 60 * 60  # 7 days
            )

            # Agent coordination stream
            await self.js.add_stream(
                name="workflow-coordination",
                subjects=["workflow.agent.*", "workflow.coordination.*"],
                retention="limits",
                max_msgs=50000,
                max_age=24 * 60 * 60  # 24 hours
            )

            logger.info("Setup NATS JetStream streams for workflows")

        except Exception as e:
            logger.warning(f"Failed to setup NATS streams: {e}")

    async def _on_reconnected(self):
        """Handle NATS reconnection."""
        logger.info("Reconnected to NATS")
        self.connected = True

    async def _on_disconnected(self):
        """Handle NATS disconnection."""
        logger.warning("Disconnected from NATS")
        self.connected = False

    async def _on_error(self, error):
        """Handle NATS errors."""
        logger.error(f"NATS error: {error}")


class KafkaWorkflowIntegration:
    """
    Integration with Kafka for workflow event streaming and analytics.
    Provides event sourcing capabilities and real-time workflow monitoring.
    """

    def __init__(self):
        self.producer: Optional[AIOKafkaProducer] = None
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.connected = False
        self.event_bus = get_event_bus()

    async def connect(
        self,
        bootstrap_servers: Optional[list[str]] = None,
        producer_config: Optional[dict[str, Any]] = None,
        consumer_config: Optional[dict[str, Any]] = None
    ) -> bool:
        """Connect to Kafka cluster."""
        if not KAFKA_AVAILABLE:
            logger.warning("Kafka not available - workflow event streaming disabled")
            return False

        bootstrap_servers = bootstrap_servers or [
            os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        ]

        try:
            # Setup producer
            producer_config = producer_config or {
                "value_serializer": lambda v: json.dumps(v).encode('utf-8'),
                "key_serializer": lambda k: k.encode('utf-8') if k else None
            }

            self.producer = AIOKafkaProducer(
                bootstrap_servers=bootstrap_servers,
                **producer_config
            )
            await self.producer.start()

            # Setup consumer
            consumer_config = consumer_config or {
                "value_deserializer": lambda v: json.loads(v.decode('utf-8')),
                "key_deserializer": lambda k: k.decode('utf-8') if k else None,
                "group_id": "workflow-integration"
            }

            self.consumer = AIOKafkaConsumer(
                bootstrap_servers=bootstrap_servers,
                **consumer_config
            )
            await self.consumer.start()

            self.connected = True
            logger.info(f"Connected to Kafka: {bootstrap_servers}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Kafka cluster."""
        if self.producer:
            await self.producer.stop()
        if self.consumer:
            await self.consumer.stop()

        self.connected = False
        logger.info("Disconnected from Kafka")

    async def stream_workflow_event(
        self,
        topic: str,
        event_data: dict[str, Any],
        workflow_id: Optional[str] = None,
        partition_key: Optional[str] = None
    ):
        """Stream workflow event to Kafka topic."""
        if not self.connected or not self.producer:
            logger.warning("Kafka not connected - cannot stream event")
            return

        event_payload = {
            "event_id": str(uuid4()),
            "workflow_id": workflow_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": event_data
        }

        try:
            # Use workflow_id as partition key for ordering guarantees
            key = partition_key or workflow_id

            await self.producer.send(
                topic=topic,
                key=key,
                value=event_payload
            )

            logger.debug(f"Streamed workflow event to Kafka topic {topic}")

        except Exception as e:
            logger.error(f"Failed to stream workflow event: {e}")

    async def consume_workflow_events(
        self,
        topics: list[str],
        callback: callable
    ):
        """Consume workflow events from Kafka topics."""
        if not self.connected or not self.consumer:
            logger.warning("Kafka not connected - cannot consume events")
            return

        try:
            # Subscribe to topics
            self.consumer.subscribe(topics)

            logger.info(f"Subscribed to Kafka topics: {topics}")

            # Start consuming
            async for message in self.consumer:
                try:
                    await callback(message.topic, message.key, message.value)
                except Exception as e:
                    logger.error(f"Error processing Kafka message: {e}")

        except Exception as e:
            logger.error(f"Error consuming Kafka events: {e}")

    async def create_workflow_analytics_stream(
        self,
        workflow_id: str,
        analytics_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Create real-time analytics stream for workflow events."""
        # This would integrate with Kafka Streams for real-time processing
        # For now, return configuration for analytics pipeline

        analytics_stream = {
            "stream_id": f"workflow-analytics-{workflow_id}",
            "workflow_id": workflow_id,
            "input_topics": [f"workflow-events-{workflow_id}"],
            "output_topics": [f"workflow-metrics-{workflow_id}"],
            "processing_config": analytics_config,
            "created_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Created analytics stream for workflow {workflow_id}")
        return analytics_stream


class WorkflowAgentBridge:
    """
    Bridge between workflows and the existing agent management system.
    Enables workflows to orchestrate agents while maintaining compatibility
    with the existing agent infrastructure.
    """

    def __init__(self):
        self.task_manager = get_task_manager()
        self.event_bus = get_event_bus()
        self.storage = get_storage_backend()

        # Track workflow-agent task mappings
        self.workflow_agent_tasks: dict[str, list[str]] = {}

    async def create_workflow_agent_task(
        self,
        workflow_id: str,
        agent_type: AgentType,
        task_description: str,
        context: dict[str, Any],
        timeout_minutes: int = 30
    ) -> Optional[str]:
        """Create agent task as part of workflow execution."""
        try:
            # Create agent task request
            task_request = AgentTaskRequest(
                agent_type=agent_type,
                task_description=task_description,
                message=self._build_agent_message(task_description, context),
                working_directory=context.get("working_directory", "/tmp"),
                agent_args=context.get("agent_args", []),
                env_vars=context.get("env_vars", {}),
                timeout_minutes=timeout_minutes
            )

            # Create task through agent manager
            agent_task = await self.task_manager.create_task(task_request)

            if agent_task:
                # Track workflow-agent mapping
                if workflow_id not in self.workflow_agent_tasks:
                    self.workflow_agent_tasks[workflow_id] = []
                self.workflow_agent_tasks[workflow_id].append(agent_task.task_id)

                # Store mapping in Redis
                await self._store_workflow_agent_mapping(workflow_id, agent_task.task_id)

                # Start task execution
                success = await self.task_manager.start_task(agent_task.task_id)

                if success:
                    logger.info(f"Created and started agent task {agent_task.task_id} for workflow {workflow_id}")
                    return agent_task.task_id
                else:
                    logger.error(f"Failed to start agent task {agent_task.task_id}")
                    return None

        except Exception as e:
            logger.error(f"Failed to create workflow agent task: {e}")
            return None

    async def wait_for_agent_task_completion(
        self,
        task_id: str,
        timeout_seconds: int = 1800  # 30 minutes
    ) -> dict[str, Any]:
        """Wait for agent task completion with timeout."""
        start_time = datetime.utcnow()
        timeout_time = start_time + timedelta(seconds=timeout_seconds)

        while datetime.utcnow() < timeout_time:
            task = await self.task_manager.get_task(task_id)

            if not task:
                return {
                    "success": False,
                    "error": f"Agent task {task_id} not found"
                }

            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED]:
                # Task finished
                result = {
                    "success": task.status == TaskStatus.COMPLETED,
                    "status": task.status.value,
                    "task_id": task_id,
                    "duration_seconds": (datetime.utcnow() - task.created_at).total_seconds()
                }

                if task.result:
                    result["result"] = task.result.model_dump() if hasattr(task.result, 'model_dump') else task.result

                if task.status != TaskStatus.COMPLETED:
                    result["error"] = getattr(task.result, "error", f"Task failed with status: {task.status.value}")

                return result

            # Wait before checking again
            await asyncio.sleep(10)

        # Timeout reached
        return {
            "success": False,
            "error": f"Agent task {task_id} timed out after {timeout_seconds} seconds",
            "status": "timeout"
        }

    async def cancel_workflow_agent_tasks(self, workflow_id: str) -> dict[str, Any]:
        """Cancel all agent tasks associated with a workflow."""
        if workflow_id not in self.workflow_agent_tasks:
            return {"cancelled_tasks": 0, "errors": []}

        task_ids = self.workflow_agent_tasks[workflow_id]
        cancelled_count = 0
        errors = []

        for task_id in task_ids:
            try:
                # In real implementation, this would cancel the agent task
                # For now, just log the cancellation
                logger.info(f"Cancelling agent task {task_id} for workflow {workflow_id}")
                cancelled_count += 1
            except Exception as e:
                errors.append(f"Failed to cancel task {task_id}: {str(e)}")

        # Clean up mapping
        del self.workflow_agent_tasks[workflow_id]
        await self._remove_workflow_agent_mapping(workflow_id)

        return {
            "cancelled_tasks": cancelled_count,
            "total_tasks": len(task_ids),
            "errors": errors
        }

    def _build_agent_message(self, task_description: str, context: dict[str, Any]) -> str:
        """Build agent message from task description and context."""
        message_parts = [task_description]

        if context.get("files"):
            message_parts.append(f"\nFiles to work with: {', '.join(context['files'])}")

        if context.get("requirements"):
            message_parts.append(f"\nRequirements: {', '.join(context['requirements'])}")

        if context.get("constraints"):
            message_parts.append(f"\nConstraints: {json.dumps(context['constraints'], indent=2)}")

        return "\n".join(message_parts)

    async def _store_workflow_agent_mapping(self, workflow_id: str, task_id: str):
        """Store workflow-agent task mapping in Redis."""
        key = f"workflow_agents:{workflow_id}"
        existing_data = self.storage.get(key)
        task_ids = json.loads(existing_data) if existing_data else []

        if task_id not in task_ids:
            task_ids.append(task_id)
            self.storage.setex(key, 24 * 60 * 60, json.dumps(task_ids))  # 24 hour TTL

    async def _remove_workflow_agent_mapping(self, workflow_id: str):
        """Remove workflow-agent task mapping from Redis."""
        key = f"workflow_agents:{workflow_id}"
        # Set very short TTL to effectively delete
        self.storage.setex(key, 1, "[]")


# Global integration instances
_state_manager: Optional[WorkflowStateManager] = None
_nats_integration: Optional[NATSWorkflowIntegration] = None
_kafka_integration: Optional[KafkaWorkflowIntegration] = None
_agent_bridge: Optional[WorkflowAgentBridge] = None


def get_workflow_state_manager() -> WorkflowStateManager:
    """Get the global workflow state manager."""
    global _state_manager
    if _state_manager is None:
        _state_manager = WorkflowStateManager()
    return _state_manager


def get_nats_integration() -> NATSWorkflowIntegration:
    """Get the global NATS workflow integration."""
    global _nats_integration
    if _nats_integration is None:
        _nats_integration = NATSWorkflowIntegration()
    return _nats_integration


def get_kafka_integration() -> KafkaWorkflowIntegration:
    """Get the global Kafka workflow integration."""
    global _kafka_integration
    if _kafka_integration is None:
        _kafka_integration = KafkaWorkflowIntegration()
    return _kafka_integration


def get_workflow_agent_bridge() -> WorkflowAgentBridge:
    """Get the global workflow-agent bridge."""
    global _agent_bridge
    if _agent_bridge is None:
        _agent_bridge = WorkflowAgentBridge()
    return _agent_bridge


logger.info("Workflow integrations module initialized")
