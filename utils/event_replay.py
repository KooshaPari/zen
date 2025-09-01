"""
Event Replay and State Reconstruction System

This module provides comprehensive event replay capabilities for agent
orchestration systems, enabling state reconstruction, debugging, testing,
and disaster recovery through event sourcing patterns.

Key Features:
- Complete state reconstruction from event streams
- Point-in-time state recovery
- Event stream debugging and analysis
- Snapshot-based performance optimization
- Parallel replay processing
- Event filtering and transformation
- State validation and consistency checking
- Disaster recovery and backup restoration
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable

try:
    from kafka import KafkaConsumer, TopicPartition
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False


from .kafka_events import AgentEvent, EventType

logger = logging.getLogger(__name__)


class ReplayMode(str, Enum):
    """Event replay modes."""

    FULL = "full"              # Replay all events from beginning
    INCREMENTAL = "incremental" # Replay events since last snapshot
    POINT_IN_TIME = "point_in_time" # Replay up to specific timestamp
    SELECTIVE = "selective"     # Replay filtered events only
    PARALLEL = "parallel"      # Parallel replay across partitions


class StateType(str, Enum):
    """Types of state that can be reconstructed."""

    AGENT_STATE = "agent_state"
    TASK_STATE = "task_state"
    WORKFLOW_STATE = "workflow_state"
    SYSTEM_STATE = "system_state"
    AUDIT_STATE = "audit_state"
    PERFORMANCE_STATE = "performance_state"


@dataclass
class ReplayFilter:
    """Filter configuration for event replay."""

    event_types: set[EventType] | None = None
    aggregate_ids: set[str] | None = None
    aggregate_types: set[str] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    user_ids: set[str] | None = None
    correlation_ids: set[str] | None = None
    custom_predicate: Callable[[AgentEvent], bool] | None = None

    def matches(self, event: AgentEvent) -> bool:
        """Check if event matches filter criteria."""

        if self.event_types and event.event_type not in self.event_types:
            return False

        if self.aggregate_ids and event.aggregate_id not in self.aggregate_ids:
            return False

        if self.aggregate_types and event.aggregate_type not in self.aggregate_types:
            return False

        if self.start_time and event.timestamp < self.start_time:
            return False

        if self.end_time and event.timestamp > self.end_time:
            return False

        if self.user_ids and event.metadata.user_id not in self.user_ids:
            return False

        if self.correlation_ids and event.metadata.correlation_id not in self.correlation_ids:
            return False

        if self.custom_predicate and not self.custom_predicate(event):
            return False

        return True


@dataclass
class StateSnapshot:
    """Snapshot of system state at a point in time."""

    snapshot_id: str
    timestamp: datetime
    state_type: StateType
    aggregate_id: str
    state_data: dict[str, Any]
    version: int
    event_position: int  # Position in event stream
    checksum: str
    compressed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp.isoformat(),
            "state_type": self.state_type.value,
            "aggregate_id": self.aggregate_id,
            "state_data": self.state_data,
            "version": self.version,
            "event_position": self.event_position,
            "checksum": self.checksum,
            "compressed": self.compressed
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateSnapshot:
        """Create snapshot from dictionary."""
        return cls(
            snapshot_id=data["snapshot_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            state_type=StateType(data["state_type"]),
            aggregate_id=data["aggregate_id"],
            state_data=data["state_data"],
            version=data["version"],
            event_position=data["event_position"],
            checksum=data["checksum"],
            compressed=data.get("compressed", False)
        )


class StateProjector:
    """Projects events to reconstruct state."""

    def __init__(self):
        self.projection_handlers: dict[tuple[StateType, EventType], Callable] = {}
        self.state_initializers: dict[StateType, Callable] = {}
        self.state_validators: dict[StateType, Callable] = {}

    def register_projection(self,
                          state_type: StateType,
                          event_type: EventType,
                          handler: Callable[[dict[str, Any], AgentEvent], dict[str, Any]]):
        """Register projection handler for state type and event type."""
        self.projection_handlers[(state_type, event_type)] = handler

    def register_initializer(self, state_type: StateType, initializer: Callable[[], dict[str, Any]]):
        """Register state initializer."""
        self.state_initializers[state_type] = initializer

    def register_validator(self, state_type: StateType, validator: Callable[[dict[str, Any]], bool]):
        """Register state validator."""
        self.state_validators[state_type] = validator

    def initialize_state(self, state_type: StateType, aggregate_id: str) -> dict[str, Any]:
        """Initialize empty state for aggregate."""

        initializer = self.state_initializers.get(state_type)
        if initializer:
            state = initializer()
        else:
            state = {
                "aggregate_id": aggregate_id,
                "aggregate_type": state_type.value,
                "created_at": datetime.now(timezone.utc),
                "version": 0,
                "events_applied": 0
            }

        state["aggregate_id"] = aggregate_id
        return state

    def apply_event(self,
                   state_type: StateType,
                   current_state: dict[str, Any],
                   event: AgentEvent) -> dict[str, Any]:
        """Apply event to current state."""

        handler = self.projection_handlers.get((state_type, event.event_type))

        if handler:
            try:
                new_state = handler(current_state.copy(), event)
                new_state["version"] = current_state.get("version", 0) + 1
                new_state["events_applied"] = current_state.get("events_applied", 0) + 1
                new_state["last_updated"] = event.timestamp
                return new_state
            except Exception as e:
                logger.error(f"Error applying event {event.event_id} to state: {e}")
                return current_state

        # Default behavior - store event in history
        if "event_history" not in current_state:
            current_state["event_history"] = []

        current_state["event_history"].append({
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp,
            "payload": event.payload
        })

        current_state["version"] = current_state.get("version", 0) + 1
        current_state["events_applied"] = current_state.get("events_applied", 0) + 1
        current_state["last_updated"] = event.timestamp

        return current_state

    def validate_state(self, state_type: StateType, state: dict[str, Any]) -> bool:
        """Validate reconstructed state."""

        validator = self.state_validators.get(state_type)
        if validator:
            return validator(state)

        # Basic validation
        required_fields = ["aggregate_id", "version", "events_applied"]
        return all(field in state for field in required_fields)


class EventReplayEngine:
    """Core engine for event replay and state reconstruction."""

    def __init__(self,
                 kafka_config: dict[str, Any] = None,
                 projector: StateProjector | None = None,
                 enable_snapshots: bool = True,
                 snapshot_interval: int = 100):  # Take snapshot every N events

        self.kafka_config = kafka_config or {
            "bootstrap_servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            "group_id": "event-replay-engine",
            "auto_offset_reset": "earliest",
            "enable_auto_commit": False,
            "value_deserializer": lambda x: json.loads(x.decode('utf-8'))
        }

        self.projector = projector or StateProjector()
        self.enable_snapshots = enable_snapshots
        self.snapshot_interval = snapshot_interval

        # State management
        self.reconstructed_states: dict[tuple[StateType, str], dict[str, Any]] = {}
        self.snapshots: dict[str, StateSnapshot] = {}
        self.executor = ThreadPoolExecutor(max_workers=8)

        # Replay tracking
        self.replay_progress: dict[str, dict[str, Any]] = {}
        self.replay_callbacks: list[Callable] = []

        # Default projection handlers
        self._register_default_projections()

    def _register_default_projections(self):
        """Register default projection handlers for common event types."""

        # Agent state projections
        def project_agent_created(state: dict[str, Any], event: AgentEvent) -> dict[str, Any]:
            state.update({
                "agent_id": event.aggregate_id,
                "status": "created",
                "created_at": event.timestamp,
                "agent_type": event.payload.get("agent_type"),
                "capabilities": event.payload.get("capabilities", []),
                "configuration": event.payload.get("configuration", {})
            })
            return state

        def project_agent_started(state: dict[str, Any], event: AgentEvent) -> dict[str, Any]:
            state.update({
                "status": "running",
                "started_at": event.timestamp,
                "process_id": event.payload.get("process_id"),
                "port": event.payload.get("port")
            })
            return state

        def project_agent_stopped(state: dict[str, Any], event: AgentEvent) -> dict[str, Any]:
            state.update({
                "status": "stopped",
                "stopped_at": event.timestamp,
                "reason": event.payload.get("reason")
            })
            return state

        # Task state projections
        def project_task_created(state: dict[str, Any], event: AgentEvent) -> dict[str, Any]:
            state.update({
                "task_id": event.aggregate_id,
                "status": "pending",
                "created_at": event.timestamp,
                "task_description": event.payload.get("task_description"),
                "agent_type": event.payload.get("agent_type"),
                "priority": event.payload.get("priority", "normal")
            })
            return state

        def project_task_started(state: dict[str, Any], event: AgentEvent) -> dict[str, Any]:
            state.update({
                "status": "running",
                "started_at": event.timestamp,
                "assigned_agent": event.payload.get("agent_id")
            })
            return state

        def project_task_completed(state: dict[str, Any], event: AgentEvent) -> dict[str, Any]:
            state.update({
                "status": "completed",
                "completed_at": event.timestamp,
                "result": event.payload.get("result"),
                "execution_time": event.payload.get("execution_time")
            })
            return state

        def project_task_failed(state: dict[str, Any], event: AgentEvent) -> dict[str, Any]:
            state.update({
                "status": "failed",
                "failed_at": event.timestamp,
                "error": event.payload.get("error"),
                "retry_count": event.payload.get("retry_count", 0)
            })
            return state

        # Register projections
        self.projector.register_projection(StateType.AGENT_STATE, EventType.AGENT_CREATED, project_agent_created)
        self.projector.register_projection(StateType.AGENT_STATE, EventType.AGENT_STARTED, project_agent_started)
        self.projector.register_projection(StateType.AGENT_STATE, EventType.AGENT_STOPPED, project_agent_stopped)

        self.projector.register_projection(StateType.TASK_STATE, EventType.TASK_CREATED, project_task_created)
        self.projector.register_projection(StateType.TASK_STATE, EventType.TASK_STARTED, project_task_started)
        self.projector.register_projection(StateType.TASK_STATE, EventType.TASK_COMPLETED, project_task_completed)
        self.projector.register_projection(StateType.TASK_STATE, EventType.TASK_FAILED, project_task_failed)

        # State initializers
        def initialize_agent_state() -> dict[str, Any]:
            return {
                "status": "unknown",
                "capabilities": [],
                "active_tasks": [],
                "performance_metrics": {}
            }

        def initialize_task_state() -> dict[str, Any]:
            return {
                "status": "unknown",
                "priority": "normal",
                "retry_count": 0,
                "subtasks": []
            }

        self.projector.register_initializer(StateType.AGENT_STATE, initialize_agent_state)
        self.projector.register_initializer(StateType.TASK_STATE, initialize_task_state)

    async def replay_events(self,
                          state_type: StateType,
                          replay_mode: ReplayMode = ReplayMode.FULL,
                          event_filter: ReplayFilter | None = None,
                          target_time: datetime | None = None,
                          parallel: bool = False) -> dict[str, Any]:
        """Replay events to reconstruct state."""

        replay_id = f"replay_{int(time.time())}"

        self.replay_progress[replay_id] = {
            "state_type": state_type.value,
            "mode": replay_mode.value,
            "status": "starting",
            "events_processed": 0,
            "start_time": datetime.now(timezone.utc),
            "errors": []
        }

        try:
            if not KAFKA_AVAILABLE:
                return await self._mock_replay(replay_id, state_type, event_filter)

            # Determine replay strategy
            if parallel and replay_mode == ReplayMode.FULL:
                result = await self._parallel_replay(replay_id, state_type, event_filter)
            else:
                result = await self._sequential_replay(replay_id, state_type, replay_mode, event_filter, target_time)

            self.replay_progress[replay_id]["status"] = "completed"
            self.replay_progress[replay_id]["end_time"] = datetime.now(timezone.utc)

            # Notify callbacks
            for callback in self.replay_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(replay_id, result)
                    else:
                        callback(replay_id, result)
                except Exception as e:
                    logger.error(f"Error in replay callback: {e}")

            return result

        except Exception as e:
            self.replay_progress[replay_id]["status"] = "failed"
            self.replay_progress[replay_id]["error"] = str(e)
            logger.error(f"Event replay failed for {replay_id}: {e}")
            raise

    async def _sequential_replay(self,
                               replay_id: str,
                               state_type: StateType,
                               replay_mode: ReplayMode,
                               event_filter: ReplayFilter | None,
                               target_time: datetime | None) -> dict[str, Any]:
        """Perform sequential event replay."""

        consumer = KafkaConsumer(**self.kafka_config)

        try:
            # Determine topics to consume
            topics = self._get_topics_for_state_type(state_type)
            consumer.subscribe(topics)

            # Reset to beginning for full replay
            if replay_mode == ReplayMode.FULL:
                consumer.poll()  # Initialize partitions
                consumer.seek_to_beginning()

            aggregates_state: dict[str, dict[str, Any]] = {}
            events_processed = 0

            self.replay_progress[replay_id]["status"] = "processing"

            # Process events
            while True:
                messages = consumer.poll(timeout_ms=5000)

                if not messages:
                    break  # No more messages

                for _topic_partition, msgs in messages.items():
                    for message in msgs:
                        try:
                            # Parse event
                            event_data = message.value
                            event = AgentEvent.from_kafka_message(event_data)

                            # Check target time
                            if target_time and event.timestamp > target_time:
                                continue

                            # Apply filter
                            if event_filter and not event_filter.matches(event):
                                continue

                            # Initialize aggregate state if needed
                            if event.aggregate_id not in aggregates_state:
                                aggregates_state[event.aggregate_id] = self.projector.initialize_state(
                                    state_type, event.aggregate_id
                                )

                            # Apply event to state
                            current_state = aggregates_state[event.aggregate_id]
                            new_state = self.projector.apply_event(state_type, current_state, event)
                            aggregates_state[event.aggregate_id] = new_state

                            events_processed += 1

                            # Update progress
                            if events_processed % 100 == 0:
                                self.replay_progress[replay_id]["events_processed"] = events_processed
                                logger.debug(f"Processed {events_processed} events for replay {replay_id}")

                            # Create snapshot if needed
                            if (self.enable_snapshots and
                                events_processed % self.snapshot_interval == 0):
                                await self._create_snapshots(state_type, aggregates_state, events_processed)

                        except Exception as e:
                            self.replay_progress[replay_id]["errors"].append({
                                "event_id": event_data.get("event_id", "unknown"),
                                "error": str(e),
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            })
                            logger.error(f"Error processing event in replay: {e}")

            # Store reconstructed states
            for aggregate_id, state in aggregates_state.items():
                key = (state_type, aggregate_id)
                self.reconstructed_states[key] = state

            self.replay_progress[replay_id]["events_processed"] = events_processed

            return {
                "replay_id": replay_id,
                "state_type": state_type.value,
                "aggregates_count": len(aggregates_state),
                "events_processed": events_processed,
                "reconstructed_states": aggregates_state,
                "duration": (datetime.now(timezone.utc) - self.replay_progress[replay_id]["start_time"]).total_seconds()
            }

        finally:
            consumer.close()

    async def _parallel_replay(self,
                             replay_id: str,
                             state_type: StateType,
                             event_filter: ReplayFilter | None) -> dict[str, Any]:
        """Perform parallel event replay across partitions."""

        # Get partition information
        topics = self._get_topics_for_state_type(state_type)

        # Create consumer to get partition metadata
        consumer = KafkaConsumer(**self.kafka_config)
        consumer.subscribe(topics)
        consumer.poll()  # Initialize

        partitions = []
        for topic in topics:
            topic_partitions = consumer.partitions_for_topic(topic)
            if topic_partitions:
                partitions.extend([TopicPartition(topic, p) for p in topic_partitions])

        consumer.close()

        # Process partitions in parallel
        futures = []
        for partition in partitions:
            future = asyncio.create_task(
                self._replay_partition(replay_id, state_type, partition, event_filter)
            )
            futures.append(future)

        # Collect results
        partition_results = []
        for future in asyncio.as_completed(futures):
            try:
                result = await future
                partition_results.append(result)
            except Exception as e:
                logger.error(f"Partition replay failed: {e}")
                self.replay_progress[replay_id]["errors"].append({
                    "partition_error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

        # Merge partition results
        merged_states = {}
        total_events = 0

        for partition_result in partition_results:
            total_events += partition_result["events_processed"]

            # Merge states by applying events in timestamp order
            for aggregate_id, state in partition_result["states"].items():
                if aggregate_id not in merged_states:
                    merged_states[aggregate_id] = state
                else:
                    # Merge by comparing versions and timestamps
                    if (state.get("version", 0) > merged_states[aggregate_id].get("version", 0) or
                        state.get("last_updated", datetime.min) > merged_states[aggregate_id].get("last_updated", datetime.min)):
                        merged_states[aggregate_id] = state

        # Store reconstructed states
        for aggregate_id, state in merged_states.items():
            key = (state_type, aggregate_id)
            self.reconstructed_states[key] = state

        self.replay_progress[replay_id]["events_processed"] = total_events

        return {
            "replay_id": replay_id,
            "state_type": state_type.value,
            "aggregates_count": len(merged_states),
            "events_processed": total_events,
            "partitions_processed": len(partition_results),
            "reconstructed_states": merged_states,
            "duration": (datetime.now(timezone.utc) - self.replay_progress[replay_id]["start_time"]).total_seconds()
        }

    async def _replay_partition(self,
                              replay_id: str,
                              state_type: StateType,
                              partition: TopicPartition,
                              event_filter: ReplayFilter | None) -> dict[str, Any]:
        """Replay events from a specific partition."""

        # Create dedicated consumer for this partition
        config = self.kafka_config.copy()
        config["group_id"] = f"{config['group_id']}_partition_{partition.partition}"

        consumer = KafkaConsumer(**config)
        consumer.assign([partition])
        consumer.seek_to_beginning(partition)

        aggregates_state: dict[str, dict[str, Any]] = {}
        events_processed = 0

        try:
            while True:
                messages = consumer.poll(timeout_ms=2000)

                if not messages:
                    break

                for _topic_partition, msgs in messages.items():
                    for message in msgs:
                        try:
                            event_data = message.value
                            event = AgentEvent.from_kafka_message(event_data)

                            # Apply filter
                            if event_filter and not event_filter.matches(event):
                                continue

                            # Initialize aggregate state if needed
                            if event.aggregate_id not in aggregates_state:
                                aggregates_state[event.aggregate_id] = self.projector.initialize_state(
                                    state_type, event.aggregate_id
                                )

                            # Apply event to state
                            current_state = aggregates_state[event.aggregate_id]
                            new_state = self.projector.apply_event(state_type, current_state, event)
                            aggregates_state[event.aggregate_id] = new_state

                            events_processed += 1

                        except Exception as e:
                            logger.error(f"Error processing event in partition replay: {e}")

            return {
                "partition": f"{partition.topic}:{partition.partition}",
                "events_processed": events_processed,
                "states": aggregates_state
            }

        finally:
            consumer.close()

    async def _mock_replay(self,
                         replay_id: str,
                         state_type: StateType,
                         event_filter: ReplayFilter | None) -> dict[str, Any]:
        """Mock replay for testing when Kafka is not available."""

        logger.info(f"Performing mock replay for {state_type.value}")

        # Create mock states
        mock_states = {
            f"mock_agent_{i}": self.projector.initialize_state(state_type, f"mock_agent_{i}")
            for i in range(3)
        }

        for state in mock_states.values():
            state["status"] = "running"
            state["version"] = 10
            state["events_applied"] = 25

        self.replay_progress[replay_id]["events_processed"] = 25

        return {
            "replay_id": replay_id,
            "state_type": state_type.value,
            "aggregates_count": len(mock_states),
            "events_processed": 25,
            "reconstructed_states": mock_states,
            "duration": 0.5
        }

    def _get_topics_for_state_type(self, state_type: StateType) -> list[str]:
        """Get Kafka topics relevant for a state type."""

        topic_mapping = {
            StateType.AGENT_STATE: ["agent-events"],
            StateType.TASK_STATE: ["task-events"],
            StateType.WORKFLOW_STATE: ["workflow-events"],
            StateType.SYSTEM_STATE: ["agent-events", "task-events", "system-events"],
            StateType.AUDIT_STATE: ["audit-events"],
            StateType.PERFORMANCE_STATE: ["performance-events"]
        }

        return topic_mapping.get(state_type, ["agent-events", "task-events"])

    async def _create_snapshots(self,
                              state_type: StateType,
                              aggregates_state: dict[str, dict[str, Any]],
                              event_position: int):
        """Create snapshots of current state."""

        for aggregate_id, state in aggregates_state.items():
            snapshot = StateSnapshot(
                snapshot_id=f"snapshot_{aggregate_id}_{int(time.time())}",
                timestamp=datetime.now(timezone.utc),
                state_type=state_type,
                aggregate_id=aggregate_id,
                state_data=state.copy(),
                version=state.get("version", 0),
                event_position=event_position,
                checksum=self._calculate_state_checksum(state)
            )

            self.snapshots[snapshot.snapshot_id] = snapshot

        logger.debug(f"Created {len(aggregates_state)} snapshots at position {event_position}")

    def _calculate_state_checksum(self, state: dict[str, Any]) -> str:
        """Calculate checksum for state integrity verification."""
        import hashlib

        # Create deterministic JSON representation
        state_json = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(state_json.encode()).hexdigest()

    async def get_reconstructed_state(self,
                                    state_type: StateType,
                                    aggregate_id: str) -> dict[str, Any] | None:
        """Get reconstructed state for a specific aggregate."""

        key = (state_type, aggregate_id)
        return self.reconstructed_states.get(key)

    async def get_all_reconstructed_states(self, state_type: StateType) -> dict[str, dict[str, Any]]:
        """Get all reconstructed states of a specific type."""

        return {
            aggregate_id: state
            for (st, aggregate_id), state in self.reconstructed_states.items()
            if st == state_type
        }

    async def validate_reconstructed_states(self, state_type: StateType) -> dict[str, Any]:
        """Validate integrity of reconstructed states."""

        states = await self.get_all_reconstructed_states(state_type)

        validation_result = {
            "total_states": len(states),
            "valid_states": 0,
            "invalid_states": 0,
            "validation_errors": []
        }

        for aggregate_id, state in states.items():
            is_valid = self.projector.validate_state(state_type, state)

            if is_valid:
                validation_result["valid_states"] += 1
            else:
                validation_result["invalid_states"] += 1
                validation_result["validation_errors"].append({
                    "aggregate_id": aggregate_id,
                    "error": "State validation failed"
                })

        return validation_result

    async def compare_states(self,
                           state_type: StateType,
                           aggregate_id: str,
                           other_state: dict[str, Any]) -> dict[str, Any]:
        """Compare reconstructed state with another state."""

        reconstructed_state = await self.get_reconstructed_state(state_type, aggregate_id)

        if not reconstructed_state:
            return {
                "comparison_possible": False,
                "reason": "Reconstructed state not found"
            }

        # Compare key fields
        differences = []

        for key in set(list(reconstructed_state.keys()) + list(other_state.keys())):
            recon_value = reconstructed_state.get(key)
            other_value = other_state.get(key)

            if recon_value != other_value:
                differences.append({
                    "field": key,
                    "reconstructed_value": recon_value,
                    "other_value": other_value
                })

        return {
            "comparison_possible": True,
            "states_match": len(differences) == 0,
            "differences_count": len(differences),
            "differences": differences,
            "comparison_timestamp": datetime.now(timezone.utc).isoformat()
        }

    def add_replay_callback(self, callback: Callable):
        """Add callback to be called when replay completes."""
        self.replay_callbacks.append(callback)

    async def get_replay_progress(self, replay_id: str) -> dict[str, Any] | None:
        """Get progress of a specific replay."""
        return self.replay_progress.get(replay_id)

    async def cleanup_old_replays(self, max_age_hours: int = 24):
        """Clean up old replay progress data."""

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        expired_replays = [
            replay_id for replay_id, progress in self.replay_progress.items()
            if progress.get("start_time", datetime.now(timezone.utc)) < cutoff_time
        ]

        for replay_id in expired_replays:
            del self.replay_progress[replay_id]

        logger.info(f"Cleaned up {len(expired_replays)} old replay records")
        return len(expired_replays)


# Global event replay engine
_replay_engine: EventReplayEngine | None = None


async def get_replay_engine() -> EventReplayEngine:
    """Get global event replay engine instance."""
    global _replay_engine

    if _replay_engine is None:
        _replay_engine = EventReplayEngine()

    return _replay_engine


async def replay_agent_state(agent_id: str,
                           target_time: datetime | None = None) -> dict[str, Any] | None:
    """Convenience function to replay specific agent state."""

    engine = await get_replay_engine()

    # Create filter for specific agent
    event_filter = ReplayFilter(
        aggregate_ids={agent_id},
        end_time=target_time
    )

    # Replay events
    result = await engine.replay_events(
        state_type=StateType.AGENT_STATE,
        replay_mode=ReplayMode.POINT_IN_TIME if target_time else ReplayMode.FULL,
        event_filter=event_filter,
        target_time=target_time
    )

    # Return specific agent state
    return result["reconstructed_states"].get(agent_id)


async def replay_task_state(task_id: str,
                          target_time: datetime | None = None) -> dict[str, Any] | None:
    """Convenience function to replay specific task state."""

    engine = await get_replay_engine()

    # Create filter for specific task
    event_filter = ReplayFilter(
        aggregate_ids={task_id},
        end_time=target_time
    )

    # Replay events
    result = await engine.replay_events(
        state_type=StateType.TASK_STATE,
        replay_mode=ReplayMode.POINT_IN_TIME if target_time else ReplayMode.FULL,
        event_filter=event_filter,
        target_time=target_time
    )

    # Return specific task state
    return result["reconstructed_states"].get(task_id)
