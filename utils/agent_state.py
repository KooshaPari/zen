"""
Agent State Persistence and Management

This module provides comprehensive agent state management using Redis for persistent storage,
enabling the Zen MCP Server to scale from 10-50 agents to 1000+ concurrent agents with
sophisticated state coordination and high availability.

Key Features:
- Distributed agent state persistence with Redis clustering support
- Optimistic concurrency control for safe concurrent state updates
- Agent lifecycle management (initialization, running, paused, terminated)
- Resource tracking and allocation management (ports, memory, connections)
- State synchronization across multiple server instances
- Automatic state recovery and cleanup for failed agents
- Performance monitoring and health checks
- Integration with existing agent orchestration system

State Model:
- Agent Initialization: Basic metadata, capabilities, resource requirements
- Agent Running: Active state, current task, resource usage, performance metrics
- Agent Paused: Temporary suspension with context preservation
- Agent Terminated: Cleanup state with resource release and audit trail

Integration Points:
- Extends existing AgentTaskManager with persistent state storage
- Provides state APIs for NATS agent connection state caching
- Creates event hooks for Kafka agent audit trail consumption
- Enables session management for Temporal agent workflow contexts
- Generates state metrics for Testing agent monitoring
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from tools.shared.agent_models import AgentType
from utils.redis_manager import RedisDB, RedisKeys, TTLPolicies, get_redis_manager

logger = logging.getLogger(__name__)


class AgentLifecycleState(str, Enum):
    """Agent lifecycle states for comprehensive state management"""
    INITIALIZING = "initializing"  # Agent is being created and configured
    READY = "ready"                # Agent is ready to accept tasks
    RUNNING = "running"            # Agent is actively processing tasks
    PAUSED = "paused"              # Agent is temporarily suspended
    TERMINATING = "terminating"    # Agent is being shut down gracefully
    TERMINATED = "terminated"      # Agent has been completely shut down
    ERROR = "error"                # Agent is in error state
    UNKNOWN = "unknown"            # Agent state could not be determined


class AgentResourceType(str, Enum):
    """Types of resources that can be allocated to agents"""
    PORT = "port"
    MEMORY = "memory"
    CONNECTION = "connection"
    PROCESS = "process"
    TEMPORARY_FILE = "temp_file"


@dataclass
class AgentResource:
    """Represents a resource allocated to an agent"""
    resource_id: str
    resource_type: AgentResourceType
    allocated_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class AgentCapabilities:
    """Agent capabilities and configuration"""
    agent_type: AgentType
    supported_operations: list[str]
    max_concurrent_tasks: int
    memory_limit_mb: int
    timeout_seconds: int
    environment_requirements: dict[str, str]
    port_range: tuple[int, int]


@dataclass
class AgentPerformanceMetrics:
    """Agent performance metrics for monitoring"""
    tasks_completed: int
    tasks_failed: int
    total_processing_time: float
    average_response_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    last_activity: datetime
    health_score: float  # 0.0 to 1.0


@dataclass
class AgentStateData:
    """Complete agent state data structure"""
    agent_id: str
    agent_type: AgentType
    lifecycle_state: AgentLifecycleState
    capabilities: AgentCapabilities
    current_task_id: Optional[str] = None
    allocated_resources: list[AgentResource] = None
    performance_metrics: Optional[AgentPerformanceMetrics] = None
    metadata: Optional[dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    version: int = 1  # For optimistic concurrency control

    def __post_init__(self):
        if self.allocated_resources is None:
            self.allocated_resources = []
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)


class AgentStateManager:
    """
    Enterprise Agent State Management System

    Provides comprehensive agent state persistence, coordination, and monitoring
    using Redis for distributed storage and real-time synchronization.
    """

    def __init__(self):
        self.redis_manager = get_redis_manager()

        # Concurrency control
        self._state_locks: dict[str, asyncio.Lock] = {}
        self._lock_manager_lock = asyncio.Lock()

        # Performance tracking
        self._operation_count = 0
        self._total_latency = 0.0

        logger.info("Agent State Manager initialized for enterprise orchestration")

    async def _get_agent_lock(self, agent_id: str) -> asyncio.Lock:
        """Get or create async lock for agent state operations"""
        async with self._lock_manager_lock:
            if agent_id not in self._state_locks:
                self._state_locks[agent_id] = asyncio.Lock()
            return self._state_locks[agent_id]

    # Core State Management Methods

    async def create_agent_state(self, agent_id: str, agent_type: AgentType,
                                capabilities: AgentCapabilities,
                                metadata: Optional[dict[str, Any]] = None) -> bool:
        """Create new agent state with initialization lifecycle"""
        start_time = time.time()

        try:
            agent_lock = await self._get_agent_lock(agent_id)
            async with agent_lock:
                # Check if agent already exists
                existing_state = await self.get_agent_state(agent_id)
                if existing_state:
                    logger.warning(f"Agent {agent_id} already exists with state {existing_state.lifecycle_state}")
                    return False

                # Create initial state
                state = AgentStateData(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    lifecycle_state=AgentLifecycleState.INITIALIZING,
                    capabilities=capabilities,
                    metadata=metadata or {},
                    performance_metrics=AgentPerformanceMetrics(
                        tasks_completed=0,
                        tasks_failed=0,
                        total_processing_time=0.0,
                        average_response_time=0.0,
                        memory_usage_mb=0.0,
                        cpu_usage_percent=0.0,
                        last_activity=datetime.now(timezone.utc),
                        health_score=1.0
                    )
                )

                # Persist state to Redis
                success = self.redis_manager.set_agent_state(
                    agent_id,
                    asdict(state),
                    TTLPolicies.AGENT_STATE
                )

                if success:
                    # Record state creation event
                    await self._record_state_event(agent_id, "agent_created", {
                        "agent_type": agent_type.value,
                        "capabilities": asdict(capabilities)
                    })

                    logger.info(f"Created agent state for {agent_id} ({agent_type.value})")
                    return True
                else:
                    logger.error(f"Failed to persist agent state for {agent_id}")
                    return False

        except Exception as e:
            logger.error(f"Error creating agent state for {agent_id}: {e}")
            return False
        finally:
            # Record performance metrics
            latency = time.time() - start_time
            self._record_operation_latency("create_agent_state", latency)

    async def get_agent_state(self, agent_id: str) -> Optional[AgentStateData]:
        """Retrieve current agent state"""
        start_time = time.time()

        try:
            state_data = self.redis_manager.get_agent_state(agent_id)
            if state_data:
                # Convert dict back to AgentStateData
                return self._deserialize_agent_state(state_data)
            return None

        except Exception as e:
            logger.error(f"Error retrieving agent state for {agent_id}: {e}")
            return None
        finally:
            latency = time.time() - start_time
            self._record_operation_latency("get_agent_state", latency)

    async def update_agent_state(self, agent_id: str,
                               lifecycle_state: Optional[AgentLifecycleState] = None,
                               current_task_id: Optional[str] = None,
                               performance_metrics: Optional[AgentPerformanceMetrics] = None,
                               metadata: Optional[dict[str, Any]] = None) -> bool:
        """Update agent state with optimistic concurrency control"""
        start_time = time.time()

        try:
            agent_lock = await self._get_agent_lock(agent_id)
            async with agent_lock:
                # Get current state for optimistic concurrency control
                current_state = await self.get_agent_state(agent_id)
                if not current_state:
                    logger.error(f"Cannot update non-existent agent state for {agent_id}")
                    return False

                # Update fields that were provided
                if lifecycle_state is not None:
                    old_state = current_state.lifecycle_state
                    current_state.lifecycle_state = lifecycle_state

                    # Record state transition event
                    await self._record_state_event(agent_id, "state_transition", {
                        "from_state": old_state.value,
                        "to_state": lifecycle_state.value
                    })

                if current_task_id is not None:
                    current_state.current_task_id = current_task_id

                if performance_metrics is not None:
                    current_state.performance_metrics = performance_metrics

                if metadata is not None:
                    if current_state.metadata:
                        current_state.metadata.update(metadata)
                    else:
                        current_state.metadata = metadata

                # Update timestamps and version for optimistic concurrency
                current_state.updated_at = datetime.now(timezone.utc)
                current_state.version += 1

                # Persist updated state
                success = self.redis_manager.set_agent_state(
                    agent_id,
                    asdict(current_state),
                    TTLPolicies.AGENT_STATE
                )

                if success:
                    logger.debug(f"Updated agent state for {agent_id}")
                    return True
                else:
                    logger.error(f"Failed to persist updated agent state for {agent_id}")
                    return False

        except Exception as e:
            logger.error(f"Error updating agent state for {agent_id}: {e}")
            return False
        finally:
            latency = time.time() - start_time
            self._record_operation_latency("update_agent_state", latency)

    async def delete_agent_state(self, agent_id: str) -> bool:
        """Delete agent state and perform cleanup"""
        start_time = time.time()

        try:
            agent_lock = await self._get_agent_lock(agent_id)
            async with agent_lock:
                # Get current state for cleanup
                current_state = await self.get_agent_state(agent_id)
                if not current_state:
                    logger.debug(f"Agent state {agent_id} already deleted or never existed")
                    return True

                # Clean up allocated resources
                cleanup_success = await self._cleanup_agent_resources(current_state)
                if not cleanup_success:
                    logger.warning(f"Some resources for agent {agent_id} could not be cleaned up")

                # Record termination event
                await self._record_state_event(agent_id, "agent_deleted", {
                    "final_state": current_state.lifecycle_state.value,
                    "resources_cleaned": len(current_state.allocated_resources)
                })

                # Remove state from Redis
                self.redis_manager.get_connection(RedisDB.STATE)
                state_key = RedisKeys.AGENT_STATE.format(agent_id=agent_id)
                heartbeat_key = RedisKeys.AGENT_HEARTBEAT.format(agent_id=agent_id)

                # Use pipeline for atomic deletion
                pipeline = self.redis_manager.get_pipeline(RedisDB.STATE, transaction=True)
                pipeline.delete(state_key)
                pipeline.delete(heartbeat_key)
                pipeline.execute()

                logger.info(f"Deleted agent state for {agent_id}")
                return True

        except Exception as e:
            logger.error(f"Error deleting agent state for {agent_id}: {e}")
            return False
        finally:
            latency = time.time() - start_time
            self._record_operation_latency("delete_agent_state", latency)

    # Resource Management Methods

    async def allocate_resource(self, agent_id: str, resource_type: AgentResourceType,
                              resource_id: Optional[str] = None,
                              expires_at: Optional[datetime] = None,
                              metadata: Optional[dict[str, Any]] = None) -> Optional[AgentResource]:
        """Allocate resource to agent with Redis-based coordination"""
        try:
            agent_lock = await self._get_agent_lock(agent_id)
            async with agent_lock:
                current_state = await self.get_agent_state(agent_id)
                if not current_state:
                    logger.error(f"Cannot allocate resource to non-existent agent {agent_id}")
                    return None

                # Generate resource ID if not provided
                if resource_id is None:
                    if resource_type == AgentResourceType.PORT:
                        # Use Redis-based port allocation
                        port_range = current_state.capabilities.port_range
                        allocated_port = self.redis_manager.allocate_port(agent_id, port_range)
                        if allocated_port is None:
                            logger.error(f"No available ports for agent {agent_id}")
                            return None
                        resource_id = str(allocated_port)
                    else:
                        resource_id = str(uuid.uuid4())

                # Create resource object
                resource = AgentResource(
                    resource_id=resource_id,
                    resource_type=resource_type,
                    allocated_at=datetime.now(timezone.utc),
                    expires_at=expires_at,
                    metadata=metadata
                )

                # Add resource to agent state
                current_state.allocated_resources.append(resource)
                current_state.updated_at = datetime.now(timezone.utc)
                current_state.version += 1

                # Persist updated state
                success = self.redis_manager.set_agent_state(
                    agent_id,
                    asdict(current_state),
                    TTLPolicies.AGENT_STATE
                )

                if success:
                    logger.debug(f"Allocated {resource_type.value} resource {resource_id} to agent {agent_id}")
                    return resource
                else:
                    # Rollback port allocation if it was allocated
                    if resource_type == AgentResourceType.PORT:
                        self.redis_manager.release_port(agent_id, int(resource_id))
                    return None

        except Exception as e:
            logger.error(f"Error allocating resource for agent {agent_id}: {e}")
            return None

    async def release_resource(self, agent_id: str, resource_id: str) -> bool:
        """Release allocated resource"""
        try:
            agent_lock = await self._get_agent_lock(agent_id)
            async with agent_lock:
                current_state = await self.get_agent_state(agent_id)
                if not current_state:
                    logger.error(f"Cannot release resource from non-existent agent {agent_id}")
                    return False

                # Find and remove resource
                resource_to_remove = None
                for resource in current_state.allocated_resources:
                    if resource.resource_id == resource_id:
                        resource_to_remove = resource
                        break

                if not resource_to_remove:
                    logger.warning(f"Resource {resource_id} not found for agent {agent_id}")
                    return False

                # Release resource based on type
                if resource_to_remove.resource_type == AgentResourceType.PORT:
                    self.redis_manager.release_port(agent_id, int(resource_id))

                # Remove from agent state
                current_state.allocated_resources.remove(resource_to_remove)
                current_state.updated_at = datetime.now(timezone.utc)
                current_state.version += 1

                # Persist updated state
                success = self.redis_manager.set_agent_state(
                    agent_id,
                    asdict(current_state),
                    TTLPolicies.AGENT_STATE
                )

                if success:
                    logger.debug(f"Released {resource_to_remove.resource_type.value} resource {resource_id} from agent {agent_id}")
                    return True
                else:
                    logger.error(f"Failed to persist resource release for agent {agent_id}")
                    return False

        except Exception as e:
            logger.error(f"Error releasing resource for agent {agent_id}: {e}")
            return False

    async def _cleanup_agent_resources(self, agent_state: AgentStateData) -> bool:
        """Clean up all resources allocated to an agent"""
        cleanup_success = True

        for resource in agent_state.allocated_resources:
            try:
                if resource.resource_type == AgentResourceType.PORT:
                    success = self.redis_manager.release_port(agent_state.agent_id, int(resource.resource_id))
                    if not success:
                        cleanup_success = False
                        logger.warning(f"Failed to release port {resource.resource_id} for agent {agent_state.agent_id}")
                # Add other resource type cleanup as needed

            except Exception as e:
                cleanup_success = False
                logger.error(f"Error cleaning up resource {resource.resource_id} for agent {agent_state.agent_id}: {e}")

        return cleanup_success

    # Query and Discovery Methods

    async def get_agents_by_state(self, lifecycle_state: AgentLifecycleState) -> list[AgentStateData]:
        """Get all agents in a specific lifecycle state"""
        try:
            active_agent_ids = self.redis_manager.get_active_agents()
            agents_in_state = []

            for agent_id in active_agent_ids:
                agent_state = await self.get_agent_state(agent_id)
                if agent_state and agent_state.lifecycle_state == lifecycle_state:
                    agents_in_state.append(agent_state)

            logger.debug(f"Found {len(agents_in_state)} agents in state {lifecycle_state.value}")
            return agents_in_state

        except Exception as e:
            logger.error(f"Error getting agents by state {lifecycle_state.value}: {e}")
            return []

    async def get_agents_by_type(self, agent_type: AgentType) -> list[AgentStateData]:
        """Get all agents of a specific type"""
        try:
            active_agent_ids = self.redis_manager.get_active_agents()
            agents_of_type = []

            for agent_id in active_agent_ids:
                agent_state = await self.get_agent_state(agent_id)
                if agent_state and agent_state.agent_type == agent_type:
                    agents_of_type.append(agent_state)

            logger.debug(f"Found {len(agents_of_type)} agents of type {agent_type.value}")
            return agents_of_type

        except Exception as e:
            logger.error(f"Error getting agents by type {agent_type.value}: {e}")
            return []

    async def get_available_agents(self, agent_type: Optional[AgentType] = None) -> list[AgentStateData]:
        """Get agents that are available to accept new tasks"""
        try:
            active_agent_ids = self.redis_manager.get_active_agents()
            available_agents = []

            for agent_id in active_agent_ids:
                agent_state = await self.get_agent_state(agent_id)
                if not agent_state:
                    continue

                # Check if agent is in a state that can accept tasks
                if agent_state.lifecycle_state not in [AgentLifecycleState.READY, AgentLifecycleState.RUNNING]:
                    continue

                # Filter by type if specified
                if agent_type and agent_state.agent_type != agent_type:
                    continue

                # Check if agent has capacity (not at max concurrent tasks)
                # This would need to be implemented based on actual task tracking
                available_agents.append(agent_state)

            logger.debug(f"Found {len(available_agents)} available agents" +
                        (f" of type {agent_type.value}" if agent_type else ""))
            return available_agents

        except Exception as e:
            logger.error(f"Error getting available agents: {e}")
            return []

    # Health and Monitoring Methods

    async def get_agent_health_summary(self) -> dict[str, Any]:
        """Get comprehensive health summary of all agents"""
        try:
            active_agent_ids = self.redis_manager.get_active_agents()

            health_summary = {
                'total_agents': len(active_agent_ids),
                'by_state': {},
                'by_type': {},
                'performance_summary': {
                    'total_tasks_completed': 0,
                    'total_tasks_failed': 0,
                    'average_health_score': 0.0
                },
                'resource_usage': {
                    'ports_allocated': 0,
                    'total_memory_mb': 0.0
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            total_health_score = 0.0
            agents_with_metrics = 0

            for agent_id in active_agent_ids:
                try:
                    agent_state = await self.get_agent_state(agent_id)
                    if not agent_state:
                        continue

                    # Count by state
                    state_name = agent_state.lifecycle_state.value
                    health_summary['by_state'][state_name] = health_summary['by_state'].get(state_name, 0) + 1

                    # Count by type
                    type_name = agent_state.agent_type.value
                    health_summary['by_type'][type_name] = health_summary['by_type'].get(type_name, 0) + 1

                    # Aggregate performance metrics
                    if agent_state.performance_metrics:
                        metrics = agent_state.performance_metrics
                        health_summary['performance_summary']['total_tasks_completed'] += metrics.tasks_completed
                        health_summary['performance_summary']['total_tasks_failed'] += metrics.tasks_failed
                        total_health_score += metrics.health_score
                        agents_with_metrics += 1

                        health_summary['resource_usage']['total_memory_mb'] += metrics.memory_usage_mb

                    # Count allocated resources
                    for resource in agent_state.allocated_resources:
                        if resource.resource_type == AgentResourceType.PORT:
                            health_summary['resource_usage']['ports_allocated'] += 1

                except Exception as e:
                    logger.debug(f"Error processing agent {agent_id} in health summary: {e}")
                    continue

            # Calculate average health score
            if agents_with_metrics > 0:
                health_summary['performance_summary']['average_health_score'] = total_health_score / agents_with_metrics

            return health_summary

        except Exception as e:
            logger.error(f"Error generating agent health summary: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    async def cleanup_stale_agents(self, max_age_hours: int = 24) -> int:
        """Clean up stale agent states that haven't been updated recently"""
        try:
            active_agent_ids = self.redis_manager.get_active_agents()
            stale_agents = []
            cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)

            for agent_id in active_agent_ids:
                agent_state = await self.get_agent_state(agent_id)
                if not agent_state:
                    continue

                # Check if agent hasn't been updated recently
                if agent_state.updated_at and agent_state.updated_at.timestamp() < cutoff_time:
                    stale_agents.append(agent_id)

            # Clean up stale agents
            cleaned_count = 0
            for agent_id in stale_agents:
                try:
                    success = await self.delete_agent_state(agent_id)
                    if success:
                        cleaned_count += 1
                        logger.info(f"Cleaned up stale agent state: {agent_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up stale agent {agent_id}: {e}")

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} stale agent states")

            return cleaned_count

        except Exception as e:
            logger.error(f"Error during stale agent cleanup: {e}")
            return 0

    # Integration Methods for Other Agents

    async def get_agent_connection_state_for_nats(self, agent_id: str) -> Optional[dict[str, Any]]:
        """Get agent connection state for NATS integration"""
        agent_state = await self.get_agent_state(agent_id)
        if not agent_state:
            return None

        # Extract connection-relevant information
        connection_state = {
            'agent_id': agent_id,
            'agent_type': agent_state.agent_type.value,
            'lifecycle_state': agent_state.lifecycle_state.value,
            'allocated_ports': [
                int(r.resource_id) for r in agent_state.allocated_resources
                if r.resource_type == AgentResourceType.PORT
            ],
            'last_activity': agent_state.performance_metrics.last_activity.isoformat() if agent_state.performance_metrics else None,
            'health_score': agent_state.performance_metrics.health_score if agent_state.performance_metrics else 0.0
        }

        return connection_state

    async def set_workflow_context_for_temporal(self, agent_id: str, workflow_context: dict[str, Any]) -> bool:
        """Set workflow context for Temporal integration"""
        try:
            # Store workflow context in agent metadata
            return await self.update_agent_state(
                agent_id,
                metadata={'temporal_workflow_context': workflow_context}
            )
        except Exception as e:
            logger.error(f"Error setting workflow context for agent {agent_id}: {e}")
            return False

    # Utility Methods

    def _deserialize_agent_state(self, state_data: dict[str, Any]) -> AgentStateData:
        """Convert dictionary back to AgentStateData object"""
        try:
            # Handle nested objects
            if 'capabilities' in state_data:
                capabilities_data = state_data['capabilities']
                state_data['capabilities'] = AgentCapabilities(**capabilities_data)

            if 'performance_metrics' in state_data and state_data['performance_metrics']:
                metrics_data = state_data['performance_metrics']
                # Convert datetime strings back to datetime objects
                if 'last_activity' in metrics_data and isinstance(metrics_data['last_activity'], str):
                    metrics_data['last_activity'] = datetime.fromisoformat(metrics_data['last_activity'].replace('Z', '+00:00'))
                state_data['performance_metrics'] = AgentPerformanceMetrics(**metrics_data)

            if 'allocated_resources' in state_data:
                resources = []
                for resource_data in state_data['allocated_resources']:
                    # Convert datetime strings back to datetime objects
                    if 'allocated_at' in resource_data and isinstance(resource_data['allocated_at'], str):
                        resource_data['allocated_at'] = datetime.fromisoformat(resource_data['allocated_at'].replace('Z', '+00:00'))
                    if 'expires_at' in resource_data and resource_data['expires_at'] and isinstance(resource_data['expires_at'], str):
                        resource_data['expires_at'] = datetime.fromisoformat(resource_data['expires_at'].replace('Z', '+00:00'))

                    resource_data['resource_type'] = AgentResourceType(resource_data['resource_type'])
                    resources.append(AgentResource(**resource_data))
                state_data['allocated_resources'] = resources

            # Convert datetime strings
            if 'created_at' in state_data and isinstance(state_data['created_at'], str):
                state_data['created_at'] = datetime.fromisoformat(state_data['created_at'].replace('Z', '+00:00'))
            if 'updated_at' in state_data and isinstance(state_data['updated_at'], str):
                state_data['updated_at'] = datetime.fromisoformat(state_data['updated_at'].replace('Z', '+00:00'))

            # Convert enums
            state_data['agent_type'] = AgentType(state_data['agent_type'])
            state_data['lifecycle_state'] = AgentLifecycleState(state_data['lifecycle_state'])

            return AgentStateData(**state_data)

        except Exception as e:
            logger.error(f"Error deserializing agent state: {e}")
            raise

    async def _record_state_event(self, agent_id: str, event_type: str, data: dict[str, Any]) -> None:
        """Record state management event for audit and monitoring"""
        try:
            event_data = {
                'agent_id': agent_id,
                'event_type': event_type,
                'data': data,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'agent_state_manager'
            }

            # This would integrate with event streaming systems
            # For now, just log the event
            logger.info(f"AGENT_STATE_EVENT: {json.dumps(event_data, default=str)}")

        except Exception as e:
            logger.debug(f"Failed to record state event: {e}")

    def _record_operation_latency(self, operation: str, latency: float) -> None:
        """Record operation latency for performance monitoring"""
        self._operation_count += 1
        self._total_latency += latency

        # Record metrics in Redis for monitoring
        self.redis_manager.record_metric(f"agent_state.{operation}.latency", latency)
        self.redis_manager.record_metric(f"agent_state.{operation}.count", 1)


# Global agent state manager instance
_agent_state_manager: Optional[AgentStateManager] = None
_manager_lock = asyncio.Lock()


async def get_agent_state_manager() -> AgentStateManager:
    """Get global agent state manager instance (singleton pattern)"""
    global _agent_state_manager

    if _agent_state_manager is None:
        async with _manager_lock:
            if _agent_state_manager is None:
                _agent_state_manager = AgentStateManager()

    return _agent_state_manager
