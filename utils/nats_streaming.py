"""
NATS JetStream Integration for Persistent Agent Messaging

This module provides advanced JetStream capabilities for reliable, persistent
messaging in agent orchestration. Features include:
- Persistent message streams with configurable retention
- Message acknowledgment and delivery guarantees
- Stream replay and time-travel capabilities
- Message deduplication and exactly-once delivery
- Consumer groups for scalable message processing
- Stream mirroring and backup strategies
- Performance optimization for high-throughput scenarios
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional, Union

try:
    import nats
    from nats.aio.client import Client as NATS
    from nats.js import JetStreamContext
    from nats.js.api import (
        AckPolicy,
        ConsumerConfig,
        DeliverPolicy,
        DiscardPolicy,
        MirrorConfig,
        ReplayPolicy,
        RetentionPolicy,
        SourceConfig,
        StorageType,
        StreamConfig,
        StreamInfo,
        StreamState,
    )
    from nats.js.errors import BadRequestError, NotFoundError
except ImportError:
    # Mock imports for systems where NATS is not available
    nats = None

import redis
from pydantic import BaseModel, Field

from utils.nats_communicator import NATSCommunicator, get_nats_communicator

logger = logging.getLogger(__name__)


class StreamDefinition(BaseModel):
    """JetStream stream definition with advanced configuration."""

    name: str = Field(..., description="Stream name")
    subjects: list[str] = Field(..., description="Subject patterns for the stream")
    description: str = Field(default="", description="Stream description")

    # Retention and storage
    retention_policy: str = Field(default="limits", description="Retention policy: limits, interest, workqueue")
    storage_type: str = Field(default="file", description="Storage type: file, memory")
    max_msgs: int = Field(default=1000000, description="Maximum number of messages")
    max_bytes: int = Field(default=1024*1024*1024, description="Maximum bytes (1GB)")
    max_age_seconds: int = Field(default=86400, description="Maximum message age (24h)")
    max_msg_size: int = Field(default=1048576, description="Maximum message size (1MB)")

    # Replication and durability
    replicas: int = Field(default=1, description="Number of replicas")
    discard_policy: str = Field(default="old", description="Discard policy: old, new")

    # Performance optimizations
    duplicate_window_seconds: int = Field(default=120, description="Duplicate detection window")
    allow_rollup_headers: bool = Field(default=False, description="Allow rollup headers")
    deny_delete: bool = Field(default=False, description="Deny message deletion")
    deny_purge: bool = Field(default=False, description="Deny stream purging")

    # Mirroring and sources
    mirror_stream: Optional[str] = Field(None, description="Mirror source stream")
    sources: list[str] = Field(default_factory=list, description="Source streams")


class ConsumerDefinition(BaseModel):
    """JetStream consumer definition with advanced configuration."""

    durable_name: str = Field(..., description="Consumer durable name")
    stream_name: str = Field(..., description="Stream name")
    description: str = Field(default="", description="Consumer description")

    # Delivery options
    deliver_policy: str = Field(default="all", description="Deliver policy: all, last, new, by_start_sequence, by_start_time")
    deliver_subject: Optional[str] = Field(None, description="Delivery subject for push consumers")
    ack_policy: str = Field(default="explicit", description="Ack policy: none, all, explicit")
    replay_policy: str = Field(default="instant", description="Replay policy: instant, original")

    # Filtering and selection
    filter_subject: Optional[str] = Field(None, description="Filter subject for consumer")
    filter_subjects: list[str] = Field(default_factory=list, description="Multiple filter subjects")

    # Flow control and backpressure
    max_deliver: int = Field(default=1, description="Maximum delivery attempts")
    max_ack_pending: int = Field(default=1000, description="Maximum unacknowledged messages")
    max_waiting: int = Field(default=512, description="Maximum waiting pull requests")
    max_batch: int = Field(default=100, description="Maximum batch size for pulls")

    # Timing and timeouts
    ack_wait_seconds: int = Field(default=30, description="Ack wait timeout")
    idle_heartbeat_seconds: int = Field(default=5, description="Idle heartbeat interval")
    inactive_threshold_seconds: int = Field(default=300, description="Inactive consumer threshold")

    # Advanced options
    flow_control: bool = Field(default=True, description="Enable flow control")
    headers_only: bool = Field(default=False, description="Deliver headers only")
    sample_freq: Optional[str] = Field(None, description="Sample frequency for metrics")


class MessageMetadata(BaseModel):
    """Enhanced metadata for JetStream messages."""

    stream_name: str = Field(..., description="Stream name")
    sequence: int = Field(..., description="Stream sequence number")
    timestamp: datetime = Field(..., description="Message timestamp")
    delivered_count: int = Field(default=1, description="Delivery attempt count")

    # Message properties
    subject: str = Field(..., description="Message subject")
    reply_to: Optional[str] = Field(None, description="Reply subject")
    headers: dict[str, str] = Field(default_factory=dict, description="Message headers")

    # Duplicate detection
    msg_id: Optional[str] = Field(None, description="Message ID for deduplication")

    # Consumer info
    consumer_name: Optional[str] = Field(None, description="Consuming consumer name")
    ack_floor_sequence: Optional[int] = Field(None, description="Ack floor sequence")


class NATSStreamingManager:
    """
    Advanced JetStream manager for persistent messaging with enterprise features.
    """

    def __init__(self, nats_communicator: Optional[NATSCommunicator] = None,
                 redis_client: Optional[redis.Redis] = None):
        """Initialize JetStream manager."""
        self.nats = nats_communicator
        self.redis_client = redis_client

        # JetStream context
        self.js: Optional[JetStreamContext] = None

        # Stream and consumer management
        self.streams: dict[str, StreamDefinition] = {}
        self.consumers: dict[str, ConsumerDefinition] = {}
        self.active_subscriptions: dict[str, Any] = {}

        # Performance metrics
        self.messages_published = 0
        self.messages_consumed = 0
        self.ack_count = 0
        self.nak_count = 0
        self.duplicate_count = 0

        # Background tasks
        self._metrics_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        logger.info("NATS Streaming Manager initialized")

    async def initialize(self) -> bool:
        """Initialize JetStream context and default streams."""
        if not self.nats:
            self.nats = await get_nats_communicator(self.redis_client)

        if not self.nats.connected:
            await self.nats.connect()

        try:
            self.js = self.nats.nc.jetstream()

            # Create default streams
            await self._create_default_streams()

            # Start background tasks
            self._metrics_task = asyncio.create_task(self._metrics_loop())
            self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            logger.info("NATS Streaming Manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"JetStream initialization failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown streaming manager and cleanup resources."""
        self._shutdown_event.set()

        # Cancel background tasks
        for task in [self._metrics_task, self._health_monitor_task, self._cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close active subscriptions
        for subscription in self.active_subscriptions.values():
            try:
                await subscription.unsubscribe()
            except Exception:
                pass

        self.active_subscriptions.clear()
        logger.info("NATS Streaming Manager shutdown completed")

    async def _create_default_streams(self) -> None:
        """Create default streams for agent orchestration."""
        default_streams = [
            # High-throughput command stream
            StreamDefinition(
                name="AGENT_COMMANDS",
                subjects=["agents.commands.>", "agents.batch.>"],
                description="Agent command stream with workqueue retention",
                retention_policy="workqueue",
                storage_type="file",
                max_msgs=100000,
                max_bytes=500*1024*1024,  # 500MB
                max_age_seconds=3600,  # 1 hour
                replicas=3,
                duplicate_window_seconds=60
            ),

            # Response and result stream
            StreamDefinition(
                name="AGENT_RESPONSES",
                subjects=["agents.responses.>", "agents.results.>"],
                description="Agent response stream with interest retention",
                retention_policy="interest",
                storage_type="file",
                max_msgs=50000,
                max_bytes=1024*1024*1024,  # 1GB
                max_age_seconds=7200,  # 2 hours
                replicas=3,
                duplicate_window_seconds=120
            ),

            # High-frequency status stream
            StreamDefinition(
                name="AGENT_STATUS",
                subjects=["agents.status.>", "agents.heartbeat.>", "agents.metrics.>"],
                description="Agent status stream with limits retention",
                retention_policy="limits",
                storage_type="memory",  # Fast access for status
                max_msgs=1000000,
                max_bytes=100*1024*1024,  # 100MB
                max_age_seconds=300,  # 5 minutes
                replicas=1,  # Status can be ephemeral
                duplicate_window_seconds=30
            ),

            # Event and audit stream
            StreamDefinition(
                name="SYSTEM_EVENTS",
                subjects=["system.>", "audit.>", "orchestration.>"],
                description="System event stream for auditing and monitoring",
                retention_policy="limits",
                storage_type="file",
                max_msgs=1000000,
                max_bytes=2*1024*1024*1024,  # 2GB
                max_age_seconds=86400*7,  # 7 days
                replicas=3,
                duplicate_window_seconds=300,
                deny_delete=True,  # Audit trail protection
                deny_purge=True
            ),

            # Performance metrics stream
            StreamDefinition(
                name="PERFORMANCE_METRICS",
                subjects=["metrics.>", "performance.>", "telemetry.>"],
                description="Performance metrics stream",
                retention_policy="limits",
                storage_type="file",
                max_msgs=10000000,
                max_bytes=5*1024*1024*1024,  # 5GB
                max_age_seconds=86400*30,  # 30 days
                replicas=2,
                duplicate_window_seconds=60
            ),

            # Edge computing stream
            StreamDefinition(
                name="EDGE_ORCHESTRATION",
                subjects=["edge.>", "placement.>", "workload.>"],
                description="Edge orchestration stream",
                retention_policy="interest",
                storage_type="file",
                max_msgs=100000,
                max_bytes=1024*1024*1024,  # 1GB
                max_age_seconds=86400,  # 24 hours
                replicas=3,
                duplicate_window_seconds=120
            )
        ]

        for stream_def in default_streams:
            await self.create_or_update_stream(stream_def)

    async def create_or_update_stream(self, stream_def: StreamDefinition) -> bool:
        """Create or update a JetStream stream."""
        if not self.js:
            logger.error("JetStream not initialized")
            return False

        try:
            # Convert to NATS stream config
            storage_type = StorageType.FILE if stream_def.storage_type == "file" else StorageType.MEMORY
            retention = RetentionPolicy.LIMITS
            if stream_def.retention_policy == "interest":
                retention = RetentionPolicy.INTEREST
            elif stream_def.retention_policy == "workqueue":
                retention = RetentionPolicy.WORK_QUEUE

            discard = DiscardPolicy.OLD if stream_def.discard_policy == "old" else DiscardPolicy.NEW

            config = StreamConfig(
                name=stream_def.name,
                subjects=stream_def.subjects,
                retention=retention,
                max_msgs=stream_def.max_msgs,
                max_bytes=stream_def.max_bytes,
                max_age=stream_def.max_age_seconds,
                max_msg_size=stream_def.max_msg_size,
                storage=storage_type,
                replicas=stream_def.replicas,
                discard=discard,
                duplicate_window=timedelta(seconds=stream_def.duplicate_window_seconds),
                allow_rollup_hdrs=stream_def.allow_rollup_headers,
                deny_delete=stream_def.deny_delete,
                deny_purge=stream_def.deny_purge,
                description=stream_def.description
            )

            # Add mirroring if configured
            if stream_def.mirror_stream:
                config.mirror = MirrorConfig(name=stream_def.mirror_stream)

            # Add sources if configured
            if stream_def.sources:
                config.sources = [SourceConfig(name=source) for source in stream_def.sources]

            # Try to get existing stream
            try:
                await self.js.stream_info(stream_def.name)
                # Update existing stream
                await self.js.update_stream(config)
                logger.info(f"Updated JetStream stream: {stream_def.name}")
            except NotFoundError:
                # Create new stream
                await self.js.add_stream(config)
                logger.info(f"Created JetStream stream: {stream_def.name}")

            # Store stream definition
            self.streams[stream_def.name] = stream_def

            # Store in Redis for persistence
            if self.redis_client:
                try:
                    key = f"jetstream:stream:{stream_def.name}"
                    self.redis_client.setex(key, 86400, stream_def.model_dump_json())
                except Exception as e:
                    logger.warning(f"Failed to store stream definition in Redis: {e}")

            return True

        except Exception as e:
            logger.error(f"Failed to create/update stream {stream_def.name}: {e}")
            return False

    async def create_consumer(self, consumer_def: ConsumerDefinition) -> bool:
        """Create a JetStream consumer with advanced configuration."""
        if not self.js:
            logger.error("JetStream not initialized")
            return False

        try:
            # Convert delivery policy
            deliver_policy = DeliverPolicy.ALL
            if consumer_def.deliver_policy == "last":
                deliver_policy = DeliverPolicy.LAST
            elif consumer_def.deliver_policy == "new":
                deliver_policy = DeliverPolicy.NEW
            elif consumer_def.deliver_policy == "by_start_sequence":
                deliver_policy = DeliverPolicy.BY_START_SEQUENCE
            elif consumer_def.deliver_policy == "by_start_time":
                deliver_policy = DeliverPolicy.BY_START_TIME

            # Convert ack policy
            ack_policy = AckPolicy.EXPLICIT
            if consumer_def.ack_policy == "none":
                ack_policy = AckPolicy.NONE
            elif consumer_def.ack_policy == "all":
                ack_policy = AckPolicy.ALL

            # Convert replay policy
            replay_policy = ReplayPolicy.INSTANT
            if consumer_def.replay_policy == "original":
                replay_policy = ReplayPolicy.ORIGINAL

            # Build consumer config
            config = ConsumerConfig(
                durable_name=consumer_def.durable_name,
                deliver_policy=deliver_policy,
                ack_policy=ack_policy,
                replay_policy=replay_policy,
                max_deliver=consumer_def.max_deliver,
                max_ack_pending=consumer_def.max_ack_pending,
                max_waiting=consumer_def.max_waiting,
                max_batch=consumer_def.max_batch,
                ack_wait=timedelta(seconds=consumer_def.ack_wait_seconds),
                idle_heartbeat=timedelta(seconds=consumer_def.idle_heartbeat_seconds) if consumer_def.idle_heartbeat_seconds > 0 else None,
                inactive_threshold=timedelta(seconds=consumer_def.inactive_threshold_seconds),
                flow_control=consumer_def.flow_control,
                headers_only=consumer_def.headers_only,
                deliver_subject=consumer_def.deliver_subject,
                description=consumer_def.description
            )

            # Add filter subjects
            if consumer_def.filter_subject:
                config.filter_subject = consumer_def.filter_subject
            elif consumer_def.filter_subjects:
                config.filter_subjects = consumer_def.filter_subjects

            # Add sampling
            if consumer_def.sample_freq:
                config.sample_freq = consumer_def.sample_freq

            # Create consumer
            await self.js.add_consumer(consumer_def.stream_name, config)

            # Store consumer definition
            consumer_key = f"{consumer_def.stream_name}:{consumer_def.durable_name}"
            self.consumers[consumer_key] = consumer_def

            logger.info(f"Created JetStream consumer: {consumer_def.durable_name} on stream {consumer_def.stream_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create consumer {consumer_def.durable_name}: {e}")
            return False

    async def publish_persistent(self, subject: str, data: Union[dict[str, Any], bytes, str],
                               headers: Optional[dict[str, str]] = None,
                               msg_id: Optional[str] = None,
                               expected_stream: Optional[str] = None,
                               timeout: float = 10.0) -> bool:
        """
        Publish message with JetStream persistence and delivery guarantees.

        Args:
            subject: Subject to publish to
            data: Message data
            headers: Optional headers
            msg_id: Optional message ID for deduplication
            expected_stream: Expected stream name for validation
            timeout: Publish timeout

        Returns:
            Success status
        """
        if not self.js:
            logger.error("JetStream not initialized")
            return False

        try:
            # Convert data to bytes
            if isinstance(data, dict):
                payload = json.dumps(data).encode()
            elif isinstance(data, str):
                payload = data.encode()
            else:
                payload = data

            # Prepare headers
            publish_headers = {}
            if headers:
                publish_headers.update(headers)

            # Add deduplication ID
            if msg_id:
                publish_headers["Nats-Msg-Id"] = msg_id

            # Add timestamp
            publish_headers["Nats-Timestamp"] = datetime.now(timezone.utc).isoformat()

            # Publish with acknowledgment
            ack = await asyncio.wait_for(
                self.js.publish(subject, payload, headers=publish_headers, stream=expected_stream),
                timeout=timeout
            )

            # Validate acknowledgment
            if ack.stream and ack.seq:
                self.messages_published += 1
                logger.debug(f"Published message to {subject} (stream: {ack.stream}, seq: {ack.seq})")

                # Check for duplicates
                if ack.duplicate:
                    self.duplicate_count += 1
                    logger.debug(f"Duplicate message detected for {subject}")

                return True
            else:
                logger.error(f"Invalid acknowledgment for {subject}")
                return False

        except asyncio.TimeoutError:
            logger.warning(f"Publish timeout for {subject}")
            return False
        except Exception as e:
            logger.error(f"Publish error for {subject}: {e}")
            return False

    async def subscribe_persistent(self, stream_name: str, consumer_name: str,
                                 message_handler: Callable[[dict[str, Any], MessageMetadata], bool],
                                 manual_ack: bool = True) -> bool:
        """
        Subscribe to JetStream with persistent consumer and message acknowledgment.

        Args:
            stream_name: Stream name to subscribe to
            consumer_name: Consumer name
            message_handler: Handler function that returns True if message processed successfully
            manual_ack: Whether to manually acknowledge messages

        Returns:
            Success status
        """
        if not self.js:
            logger.error("JetStream not initialized")
            return False

        try:
            # Get consumer info
            await self.js.consumer_info(stream_name, consumer_name)

            async def process_message(msg):
                try:
                    start_time = time.perf_counter()

                    # Parse message data
                    try:
                        data = json.loads(msg.data.decode())
                    except Exception:
                        data = {"raw_data": msg.data.decode()}

                    # Extract metadata
                    metadata = MessageMetadata(
                        stream_name=msg.metadata.stream,
                        sequence=msg.metadata.sequence.stream,
                        timestamp=msg.metadata.timestamp,
                        delivered_count=msg.metadata.num_delivered,
                        subject=msg.subject,
                        reply_to=msg.reply,
                        headers=dict(msg.headers) if msg.headers else {}
                    )

                    # Call message handler
                    try:
                        success = await asyncio.get_event_loop().run_in_executor(
                            None, message_handler, data, metadata
                        )
                    except Exception as e:
                        logger.error(f"Message handler error: {e}")
                        success = False

                    # Handle acknowledgment
                    if manual_ack:
                        if success:
                            await msg.ack()
                            self.ack_count += 1
                            self.messages_consumed += 1
                        else:
                            await msg.nak()
                            self.nak_count += 1
                            logger.warning(f"Message processing failed, sending NAK: {msg.subject}")
                    else:
                        self.messages_consumed += 1

                    # Update performance metrics
                    process_time_ms = (time.perf_counter() - start_time) * 1000
                    logger.debug(f"Processed message in {process_time_ms:.2f}ms")

                except Exception as e:
                    logger.error(f"Message processing error: {e}")
                    if manual_ack:
                        try:
                            await msg.nak()
                            self.nak_count += 1
                        except Exception:
                            pass

            # Create subscription
            subscription = await self.js.subscribe(
                None,  # Will use consumer's filter subjects
                cb=process_message,
                stream=stream_name,
                durable=consumer_name,
                manual_ack=manual_ack
            )

            # Store subscription
            sub_key = f"{stream_name}:{consumer_name}"
            self.active_subscriptions[sub_key] = subscription

            logger.info(f"Subscribed to stream {stream_name} with consumer {consumer_name}")
            return True

        except Exception as e:
            logger.error(f"Subscribe error for {stream_name}/{consumer_name}: {e}")
            return False

    async def pull_messages(self, stream_name: str, consumer_name: str,
                          batch_size: int = 100, timeout: float = 30.0) -> list[dict[str, Any]]:
        """
        Pull messages from JetStream consumer.

        Args:
            stream_name: Stream name
            consumer_name: Consumer name
            batch_size: Number of messages to pull
            timeout: Pull timeout

        Returns:
            List of messages
        """
        if not self.js:
            logger.error("JetStream not initialized")
            return []

        try:
            messages = []

            # Create pull subscription
            psub = await self.js.pull_subscribe(
                None,  # Use consumer's filter
                durable=consumer_name,
                stream=stream_name
            )

            # Pull messages
            msgs = await psub.fetch(batch_size, timeout=timeout)

            for msg in msgs:
                try:
                    # Parse message data
                    try:
                        data = json.loads(msg.data.decode())
                    except Exception:
                        data = {"raw_data": msg.data.decode()}

                    # Add metadata
                    data["_metadata"] = {
                        "stream": msg.metadata.stream,
                        "sequence": msg.metadata.sequence.stream,
                        "timestamp": msg.metadata.timestamp.isoformat(),
                        "delivered_count": msg.metadata.num_delivered,
                        "subject": msg.subject
                    }

                    messages.append(data)

                    # Acknowledge message
                    await msg.ack()
                    self.ack_count += 1
                    self.messages_consumed += 1

                except Exception as e:
                    logger.error(f"Error processing pulled message: {e}")
                    try:
                        await msg.nak()
                        self.nak_count += 1
                    except Exception:
                        pass

            return messages

        except Exception as e:
            logger.error(f"Pull messages error: {e}")
            return []

    async def replay_messages(self, stream_name: str, start_time: datetime,
                            end_time: Optional[datetime] = None,
                            subject_filter: Optional[str] = None) -> AsyncIterator[dict[str, Any]]:
        """
        Replay messages from stream within time range.

        Args:
            stream_name: Stream name
            start_time: Start time for replay
            end_time: End time for replay (None = now)
            subject_filter: Optional subject filter

        Yields:
            Message data
        """
        if not self.js:
            logger.error("JetStream not initialized")
            return

        try:
            # Create temporary consumer for replay
            consumer_name = f"replay_{int(time.time())}"

            config = ConsumerConfig(
                durable_name=consumer_name,
                deliver_policy=DeliverPolicy.BY_START_TIME,
                opt_start_time=start_time,
                ack_policy=AckPolicy.EXPLICIT,
                replay_policy=ReplayPolicy.INSTANT,
                filter_subject=subject_filter
            )

            await self.js.add_consumer(stream_name, config)

            try:
                # Pull all messages in range
                psub = await self.js.pull_subscribe(
                    None,
                    durable=consumer_name,
                    stream=stream_name
                )

                end_ts = end_time or datetime.now(timezone.utc)

                while True:
                    try:
                        msgs = await psub.fetch(100, timeout=5.0)
                        if not msgs:
                            break

                        for msg in msgs:
                            # Check if we've passed end time
                            if msg.metadata.timestamp > end_ts:
                                return

                            try:
                                # Parse message data
                                try:
                                    data = json.loads(msg.data.decode())
                                except:
                                    data = {"raw_data": msg.data.decode()}

                                # Add metadata
                                data["_metadata"] = {
                                    "stream": msg.metadata.stream,
                                    "sequence": msg.metadata.sequence.stream,
                                    "timestamp": msg.metadata.timestamp.isoformat(),
                                    "subject": msg.subject
                                }

                                yield data

                                # Acknowledge message
                                await msg.ack()

                            except Exception as e:
                                logger.error(f"Error processing replay message: {e}")

                    except asyncio.TimeoutError:
                        # No more messages
                        break

            finally:
                # Clean up temporary consumer
                try:
                    await self.js.delete_consumer(stream_name, consumer_name)
                except:
                    pass

        except Exception as e:
            logger.error(f"Message replay error: {e}")

    async def get_stream_info(self, stream_name: str) -> Optional[dict[str, Any]]:
        """Get detailed stream information and statistics."""
        if not self.js:
            return None

        try:
            info = await self.js.stream_info(stream_name)

            return {
                "name": info.config.name,
                "subjects": info.config.subjects,
                "retention": info.config.retention.value,
                "storage": info.config.storage.value,
                "replicas": info.config.replicas,
                "messages": info.state.messages,
                "bytes": info.state.bytes,
                "first_seq": info.state.first_seq,
                "last_seq": info.state.last_seq,
                "first_ts": info.state.first_ts.isoformat() if info.state.first_ts else None,
                "last_ts": info.state.last_ts.isoformat() if info.state.last_ts else None,
                "consumers": info.state.consumer_count,
                "created": info.created.isoformat() if info.created else None
            }

        except Exception as e:
            logger.error(f"Failed to get stream info for {stream_name}: {e}")
            return None

    async def get_consumer_info(self, stream_name: str, consumer_name: str) -> Optional[dict[str, Any]]:
        """Get detailed consumer information and statistics."""
        if not self.js:
            return None

        try:
            info = await self.js.consumer_info(stream_name, consumer_name)

            return {
                "name": info.name,
                "stream_name": info.stream_name,
                "created": info.created.isoformat() if info.created else None,
                "delivered": {
                    "consumer_seq": info.delivered.consumer_seq,
                    "stream_seq": info.delivered.stream_seq,
                    "last_active": info.delivered.last_active.isoformat() if info.delivered.last_active else None
                },
                "ack_floor": {
                    "consumer_seq": info.ack_floor.consumer_seq,
                    "stream_seq": info.ack_floor.stream_seq,
                    "last_active": info.ack_floor.last_active.isoformat() if info.ack_floor.last_active else None
                },
                "num_ack_pending": info.num_ack_pending,
                "num_redelivered": info.num_redelivered,
                "num_waiting": info.num_waiting,
                "num_pending": info.num_pending
            }

        except Exception as e:
            logger.error(f"Failed to get consumer info for {stream_name}/{consumer_name}: {e}")
            return None

    # Background tasks
    async def _metrics_loop(self) -> None:
        """Publish streaming metrics periodically."""
        while not self._shutdown_event.is_set():
            try:
                metrics = {
                    "messages_published": self.messages_published,
                    "messages_consumed": self.messages_consumed,
                    "ack_count": self.ack_count,
                    "nak_count": self.nak_count,
                    "duplicate_count": self.duplicate_count,
                    "active_streams": len(self.streams),
                    "active_consumers": len(self.consumers),
                    "active_subscriptions": len(self.active_subscriptions),
                    "ack_rate": self.ack_count / max(self.ack_count + self.nak_count, 1),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

                # Add stream-specific metrics
                stream_metrics = {}
                for stream_name in self.streams.keys():
                    stream_info = await self.get_stream_info(stream_name)
                    if stream_info:
                        stream_metrics[stream_name] = stream_info

                metrics["streams"] = stream_metrics

                # Publish metrics
                if self.nats:
                    await self.publish_persistent(
                        "metrics.jetstream",
                        metrics,
                        expected_stream="PERFORMANCE_METRICS"
                    )

                await asyncio.sleep(60)  # Every minute

            except Exception as e:
                logger.error(f"Metrics loop error: {e}")
                await asyncio.sleep(5)

    async def _health_monitor_loop(self) -> None:
        """Monitor stream and consumer health."""
        while not self._shutdown_event.is_set():
            try:
                unhealthy_streams = []
                unhealthy_consumers = []

                # Check stream health
                for stream_name in self.streams.keys():
                    try:
                        info = await self.js.stream_info(stream_name)

                        # Check for issues
                        if info.state.messages > info.config.max_msgs * 0.9:
                            logger.warning(f"Stream {stream_name} approaching message limit")

                        if info.state.bytes > info.config.max_bytes * 0.9:
                            logger.warning(f"Stream {stream_name} approaching size limit")

                    except Exception as e:
                        unhealthy_streams.append(stream_name)
                        logger.error(f"Stream health check failed for {stream_name}: {e}")

                # Check consumer health
                for consumer_key, consumer_def in self.consumers.items():
                    stream_name, consumer_name = consumer_key.split(":", 1)
                    try:
                        info = await self.js.consumer_info(stream_name, consumer_name)

                        # Check for high pending acks
                        if info.num_ack_pending > consumer_def.max_ack_pending * 0.9:
                            logger.warning(f"Consumer {consumer_name} has high pending acks: {info.num_ack_pending}")

                        # Check for redeliveries
                        if info.num_redelivered > 0:
                            logger.warning(f"Consumer {consumer_name} has redelivered messages: {info.num_redelivered}")

                    except Exception as e:
                        unhealthy_consumers.append(consumer_key)
                        logger.error(f"Consumer health check failed for {consumer_key}: {e}")

                # Publish health alerts if needed
                if unhealthy_streams or unhealthy_consumers:
                    health_alert = {
                        "event": "jetstream_health_alert",
                        "unhealthy_streams": unhealthy_streams,
                        "unhealthy_consumers": unhealthy_consumers,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }

                    if self.nats:
                        await self.publish_persistent(
                            "system.health.jetstream",
                            health_alert,
                            expected_stream="SYSTEM_EVENTS"
                        )

                await asyncio.sleep(300)  # Every 5 minutes

            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(30)

    async def _cleanup_loop(self) -> None:
        """Clean up old streams and consumers."""
        while not self._shutdown_event.is_set():
            try:
                # This would implement cleanup logic for:
                # - Old temporary consumers
                # - Streams that have exceeded retention
                # - Inactive consumers

                await asyncio.sleep(3600)  # Every hour

            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(300)

    async def get_streaming_metrics(self) -> dict[str, Any]:
        """Get comprehensive streaming metrics."""
        metrics = {
            "messages_published": self.messages_published,
            "messages_consumed": self.messages_consumed,
            "ack_count": self.ack_count,
            "nak_count": self.nak_count,
            "duplicate_count": self.duplicate_count,
            "active_streams": len(self.streams),
            "active_consumers": len(self.consumers),
            "active_subscriptions": len(self.active_subscriptions),
            "ack_rate": self.ack_count / max(self.ack_count + self.nak_count, 1),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Add JetStream account info if available
        if self.js:
            try:
                account_info = await self.js.account_info()
                metrics["account_info"] = {
                    "memory": account_info.memory,
                    "storage": account_info.storage,
                    "streams": account_info.streams,
                    "consumers": account_info.consumers,
                    "limits": {
                        "max_memory": account_info.limits.max_memory,
                        "max_storage": account_info.limits.max_storage,
                        "max_streams": account_info.limits.max_streams,
                        "max_consumers": account_info.limits.max_consumers
                    }
                }
            except Exception as e:
                logger.debug(f"Failed to get account info: {e}")

        return metrics


# Global streaming manager instance
_streaming_manager: Optional[NATSStreamingManager] = None


async def get_streaming_manager(nats_communicator: Optional[NATSCommunicator] = None,
                               redis_client: Optional[redis.Redis] = None) -> NATSStreamingManager:
    """Get or create the global streaming manager instance."""
    global _streaming_manager

    if _streaming_manager is None:
        _streaming_manager = NATSStreamingManager(nats_communicator, redis_client)
        await _streaming_manager.initialize()

    return _streaming_manager


async def shutdown_streaming_manager() -> None:
    """Shutdown the global streaming manager."""
    global _streaming_manager

    if _streaming_manager:
        await _streaming_manager.shutdown()
        _streaming_manager = None
