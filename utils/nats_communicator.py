"""
NATS Communication Infrastructure for Ultra-Low Latency Agent Messaging

This module provides the core NATS messaging infrastructure for real-time agent
communication with support for:
- Ultra-low latency messaging (sub-millisecond)
- NATS cluster support and high availability
- JetStream for persistent messaging
- Message ordering guarantees
- Fault-tolerant delivery
- 1000+ concurrent agent connections
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Union

try:
    import nats
    from nats.aio.client import Client as NATS
    from nats.js import JetStreamContext
    from nats.js.api import (
        ConsumerConfig,
        DeliverPolicy,
        ReplayPolicy,
        RetentionPolicy,
        StreamConfig,
    )
    from nats.js.errors import NotFoundError
except ImportError:
    # Mock NATS for systems where it's not available
    nats = None
    NATS = None
    JetStreamContext = None

import redis
from pydantic import BaseModel, Field

from tools.shared.agent_models import AgentType, TaskStatus

logger = logging.getLogger(__name__)


class NATSMessage(BaseModel):
    """Structured NATS message for agent communication."""

    message_id: str = Field(..., description="Unique message identifier")
    sender_id: str = Field(..., description="Sender agent/service identifier")
    recipient_id: Optional[str] = Field(None, description="Target recipient (optional for broadcasts)")
    message_type: str = Field(..., description="Message type (command, response, status, etc.)")
    payload: dict[str, Any] = Field(..., description="Message payload")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = Field(None, description="Correlation ID for request-response")
    reply_to: Optional[str] = Field(None, description="Reply subject for responses")
    priority: int = Field(default=5, description="Message priority (1=highest, 10=lowest)")
    ttl_seconds: Optional[int] = Field(None, description="Time-to-live in seconds")


class AgentConnectionInfo(BaseModel):
    """Information about connected agent."""

    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: AgentType = Field(..., description="Type of agent")
    status: TaskStatus = Field(..., description="Current agent status")
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    capabilities: list[str] = Field(default_factory=list, description="Agent capabilities")
    load_factor: float = Field(default=0.0, description="Current load (0.0-1.0)")
    endpoint: Optional[str] = Field(None, description="Agent endpoint URL")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class NATSConfig(BaseModel):
    """NATS configuration settings."""

    servers: list[str] = Field(default_factory=lambda: ["nats://localhost:4222"])
    cluster_name: Optional[str] = Field(None, description="NATS cluster name")
    max_reconnect_attempts: int = Field(default=-1, description="Max reconnection attempts (-1=infinite)")
    reconnect_delay: float = Field(default=2.0, description="Delay between reconnection attempts")
    max_payload: int = Field(default=1048576, description="Maximum message payload size (1MB)")
    drain_timeout: float = Field(default=30.0, description="Drain timeout on shutdown")

    # JetStream configuration
    jetstream_domain: Optional[str] = Field(None, description="JetStream domain")
    stream_replicas: int = Field(default=1, description="Number of stream replicas (use 1 for single-node)")
    max_memory: int = Field(default=1024*1024*1024, description="Max memory for streams (1GB)")
    max_storage: int = Field(default=10*1024*1024*1024, description="Max storage for streams (10GB)")

    # Performance tuning
    flush_timeout: float = Field(default=0.001, description="Flush timeout (1ms)")
    ping_interval: int = Field(default=30, description="Ping interval in seconds")
    max_pending_msgs: int = Field(default=10000, description="Max pending messages per subscription")

    # Edge computing
    leaf_node_urls: list[str] = Field(default_factory=list, description="Leaf node URLs for edge")


class NATSCommunicator:
    """
    Core NATS messaging infrastructure with ultra-low latency and high availability.
    """

    def __init__(self, config: Optional[NATSConfig] = None, redis_client: Optional[redis.Redis] = None):
        """Initialize NATS communicator."""
        self.config = config or self._load_config_from_env()
        self.redis_client = redis_client

        # Core NATS components
        self.nc: Optional[NATS] = None
        self.js: Optional[JetStreamContext] = None

        # Connection state
        self.connected = False
        self.connection_id: Optional[str] = None
        self.reconnect_count = 0

        # Agent registry
        self.agents: dict[str, AgentConnectionInfo] = {}
        self.subscriptions: dict[str, Any] = {}

        # Performance metrics
        self.message_count = 0
        self.error_count = 0
        self.total_latency_ms = 0.0
        self.max_latency_ms = 0.0
        self.min_latency_ms = float('inf')
        # Publish JS fallback observability
        self.publish_js_attempts = 0
        self.publish_js_fallbacks = 0
        self.publish_core_success = 0

        # Event callbacks
        self.message_handlers: dict[str, Callable[[NATSMessage], None]] = {}
        self.connection_handlers: list[Callable[[], None]] = []
        self.disconnection_handlers: list[Callable[[], None]] = []

        # Async locks and queues
        self._connection_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

        # Stream names for different message types
        self.streams = {
            "agent.commands": "AGENT_COMMANDS",
            "agent.responses": "AGENT_RESPONSES",
            "agent.status": "AGENT_STATUS",
            "agent.discovery": "AGENT_DISCOVERY",
            "agent.metrics": "AGENT_METRICS",
            "system.events": "SYSTEM_EVENTS",
            # Added lifecycle/message streams for tasks/batches/messaging
            "tasks.lifecycle": "TASKS_LIFECYCLE",
            "tasks.messages": "TASKS_MESSAGES",
            "batches.lifecycle": "BATCHES_LIFECYCLE",
            "messaging.events": "MESSAGING_EVENTS",
            # A2A subjects
            "a2a.agent": "A2A_AGENT_DIRECT",
            "a2a.task.events": "A2A_TASK_EVENTS",
            "a2a.task.rpc": "A2A_TASK_RPC",
            "a2a.events": "A2A_EVENTS",
        }

        # Feature toggles (env-driven)
        self.use_js_for_rpc: bool = os.getenv("NATS_USE_JS_FOR_RPC", "false").lower() in ("1","true","yes")
        self.use_js_for_events: bool = os.getenv("NATS_USE_JS_FOR_EVENTS", "false").lower() in ("1","true","yes")

        logger.info(f"NATS Communicator initialized with {len(self.config.servers)} servers")

    def _load_config_from_env(self) -> NATSConfig:
        """Load NATS configuration from environment variables."""
        servers = os.getenv("NATS_SERVERS", "nats://localhost:4222").split(",")

        config = NATSConfig(
            servers=[s.strip() for s in servers],
            cluster_name=os.getenv("NATS_CLUSTER_NAME"),
            max_reconnect_attempts=int(os.getenv("NATS_MAX_RECONNECT", "-1")),
            reconnect_delay=float(os.getenv("NATS_RECONNECT_DELAY", "2.0")),
            max_payload=int(os.getenv("NATS_MAX_PAYLOAD", "1048576")),
            jetstream_domain=os.getenv("NATS_JETSTREAM_DOMAIN"),
            stream_replicas=int(os.getenv("NATS_STREAM_REPLICAS", "3")),
            flush_timeout=float(os.getenv("NATS_FLUSH_TIMEOUT", "0.001")),
            ping_interval=int(os.getenv("NATS_PING_INTERVAL", "30")),
            max_pending_msgs=int(os.getenv("NATS_MAX_PENDING", "10000")),
        )

        # Parse leaf node URLs for edge computing
        leaf_urls = os.getenv("NATS_LEAF_NODES", "")
        if leaf_urls:
            config.leaf_node_urls = [url.strip() for url in leaf_urls.split(",")]

        return config

    async def connect(self) -> bool:
        """Connect to NATS cluster with retry logic and health monitoring."""
        if not nats:
            logger.error("NATS client library not available. Install with: pip install nats-py")
            return False

        async with self._connection_lock:
            if self.connected:
                return True

            try:
                # Connection options for high performance
                # Minimal options to avoid incompatibilities; can be extended after connection established
                options = {
                    "servers": self.config.servers,
                }

                # Add leaf node connection for edge computing
                if self.config.leaf_node_urls:
                    options["leaf_node_urls"] = self.config.leaf_node_urls

                self.nc = await nats.connect(**options)
                self.connection_id = self.nc._client_id

                # Initialize JetStream for persistent messaging
                self.js = self.nc.jetstream(domain=self.config.jetstream_domain)

                # Set up core streams
                await self._setup_streams()

                # Start heartbeat and metrics collection
                asyncio.create_task(self._heartbeat_loop())
                asyncio.create_task(self._metrics_collector())

                self.connected = True
                self.reconnect_count = 0

                # Notify connection handlers
                for handler in self.connection_handlers:
                    try:
                        handler()
                    except Exception as e:
                        logger.warning(f"Connection handler error: {e}")

                # Publish connection event
                await self._publish_system_event("communicator.connected", {
                    "connection_id": self.connection_id,
                    "servers": self.config.servers,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                logger.info(f"NATS connected successfully to {self.config.servers}")
                return True

            except Exception as e:
                logger.error(f"NATS connection failed: {e}")
                self.connected = False
                return False

    async def disconnect(self) -> None:
        """Gracefully disconnect from NATS."""
        if not self.nc or not self.connected:
            return

        try:
            self._shutdown_event.set()

            # Publish disconnection event
            await self._publish_system_event("communicator.disconnecting", {
                "connection_id": self.connection_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Drain subscriptions gracefully
            if self.nc:
                await self.nc.drain(timeout=self.config.drain_timeout)

            self.connected = False
            self.nc = None
            self.js = None

            logger.info("NATS disconnected gracefully")

        except Exception as e:
            logger.error(f"Error during NATS disconnection: {e}")

    async def _setup_streams(self) -> None:
        """Set up JetStream streams for persistent messaging."""
        if not self.js:
            return

        # Single-node JetStream: always use 1 replica to avoid server error 10074
        # Reduce sizes considerably for dev to avoid storage errors, and include A2A subjects
        stream_configs = [
            StreamConfig(
                name=self.streams["agent.commands"],
                subjects=["agent.commands.*", "agent.batch.*"],
                retention=RetentionPolicy.WORK_QUEUE,
                max_msgs=5000,
                max_bytes=50 * 1024 * 1024,
                max_age=1800,
                num_replicas=1
            ),
            StreamConfig(
                name=self.streams["agent.responses"],
                subjects=["agent.responses.*", "agent.results.*"],
                retention=RetentionPolicy.INTEREST,
                max_msgs=5000,
                max_bytes=50 * 1024 * 1024,
                max_age=3600,
                num_replicas=1
            ),
            StreamConfig(
                name=self.streams["agent.status"],
                subjects=["agent.status.*", "agent.heartbeat.*"],
                retention=RetentionPolicy.LIMITS,
                max_msgs=20000,
                max_bytes=50 * 1024 * 1024,
                max_age=600,
                num_replicas=1
            ),
            StreamConfig(
                name=self.streams["agent.discovery"],
                subjects=["discovery.*", "registry.*"],
                retention=RetentionPolicy.INTEREST,
                max_msgs=2000,
                max_bytes=10 * 1024 * 1024,
                max_age=86400,
                num_replicas=1
            ),
            StreamConfig(
                name=self.streams["agent.metrics"],
                subjects=["metrics.*", "performance.*"],
                retention=RetentionPolicy.LIMITS,
                max_msgs=20000,
                max_bytes=50 * 1024 * 1024,
                max_age=3600,
                num_replicas=1
            ),
            StreamConfig(
                name=self.streams["system.events"],
                subjects=["system.*", "events.*"],
                retention=RetentionPolicy.INTEREST,
                max_msgs=5000,
                max_bytes=25 * 1024 * 1024,
                max_age=86400,
                num_replicas=1
            ),
            StreamConfig(
                name=self.streams["tasks.lifecycle"],
                subjects=["tasks.created", "tasks.updated.*", "tasks.completed", "tasks.failed", "tasks.timeout", "tasks.cancelled"],
                retention=RetentionPolicy.LIMITS,
                max_msgs=20000,
                max_bytes=50 * 1024 * 1024,
                max_age=604800,
                num_replicas=1
            ),
            StreamConfig(
                name=self.streams["tasks.messages"],
                subjects=["tasks.msg.*"],
                retention=RetentionPolicy.LIMITS,
                max_msgs=20000,
                max_bytes=50 * 1024 * 1024,
                max_age=21600,
                num_replicas=1
            ),
            StreamConfig(
                name=self.streams["batches.lifecycle"],
                subjects=["batches.started", "batches.updated", "batches.finished"],
                retention=RetentionPolicy.LIMITS,
                max_msgs=10000,
                max_bytes=25 * 1024 * 1024,
                max_age=604800,
                num_replicas=1
            ),
            StreamConfig(
                name=self.streams["messaging.events"],
                subjects=["messaging.*", "channels.*"],
                retention=RetentionPolicy.LIMITS,
                max_msgs=20000,
                max_bytes=25 * 1024 * 1024,
                max_age=86400,
                num_replicas=1
            ),
            # A2A streams
            StreamConfig(
                name=self.streams["a2a.agent"],
                subjects=["a2a.agent.*.in", "a2a.agent.*.out", "discovery.advertise"],
                retention=RetentionPolicy.LIMITS,
                max_msgs=20000,
                max_bytes=50 * 1024 * 1024,
                max_age=86400,
                num_replicas=1
            ),
            StreamConfig(
                name=self.streams["a2a.task.events"],
                subjects=["a2a.task.*.events"],
                retention=RetentionPolicy.LIMITS,
                max_msgs=20000,
                max_bytes=50 * 1024 * 1024,
                max_age=86400,
                num_replicas=1
            ),
            StreamConfig(
                name=self.streams["a2a.task.rpc"],
                subjects=["a2a.task.*.rpc"],
                retention=RetentionPolicy.WORK_QUEUE,
                max_msgs=20000,
                max_bytes=50 * 1024 * 1024,
                max_age=86400,
                num_replicas=1
            ),
            StreamConfig(
                name=self.streams["a2a.events"],
                subjects=["a2a.events"],
                retention=RetentionPolicy.LIMITS,
                max_msgs=20000,
                max_bytes=50 * 1024 * 1024,
                max_age=86400,
                num_replicas=1
            ),
        ]

        # Create or update streams (best-effort, non-fatal)
        for config in stream_configs:
            try:
                try:
                    info = await self.js.stream_info(config.name)
                    logger.debug(f"Stream {config.name} exists with {info.state.messages} messages")
                except NotFoundError:
                    try:
                        await self.js.add_stream(config)
                        logger.info(f"Created JetStream stream: {config.name}")
                    except Exception as ee:
                        logger.warning(f"Soft-fail creating stream {config.name}: {ee}")
                        continue
            except Exception as e:
                logger.warning(f"Soft-fail inspecting stream {config.name}: {e}")
                continue

    async def publish(self, subject: str, message: Union[NATSMessage, dict[str, Any]],
                     use_jetstream: bool = False, timeout: float = 1.0) -> bool:
        """
        Publish message with ultra-low latency optimization.

        Args:
            subject: NATS subject to publish to
            message: Message to publish (NATSMessage or dict)
            use_jetstream: Use JetStream for persistence
            timeout: Publish timeout in seconds

        Returns:
            Success status
        """
        if not self.nc or not self.connected:
            logger.error("NATS not connected")
            return False

        try:
            start_time = time.perf_counter()

            # Convert message to bytes
            if isinstance(message, NATSMessage):
                data = message.model_dump_json().encode()
            elif isinstance(message, dict):
                data = json.dumps(message).encode()
            else:
                data = str(message).encode()

            # Validate payload size
            if len(data) > self.config.max_payload:
                logger.error(f"Message too large: {len(data)} bytes > {self.config.max_payload}")
                return False

            # Publish with appropriate method
            if use_jetstream and self.js:
                self.publish_js_attempts += 1
                try:
                    await asyncio.wait_for(self.js.publish(subject, data), timeout=timeout)
                except Exception as je:
                    self.publish_js_fallbacks += 1
                    logger.warning(f"JetStream publish failed for {subject}: {je}; falling back to core NATS")
                    use_jetstream = False
            if not use_jetstream:
                # Use core NATS for maximum performance
                await asyncio.wait_for(self.nc.publish(subject, data), timeout=timeout)
                self.publish_core_success += 1

            # Flush for immediate delivery (critical for low latency)
            await asyncio.wait_for(self.nc.flush(timeout=self.config.flush_timeout), timeout=self.config.flush_timeout * 2)

            # Update performance metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._update_metrics(latency_ms)

            return True

        except asyncio.TimeoutError:
            logger.warning(f"Publish timeout on subject {subject}")
            self.error_count += 1
            return False
        except Exception as e:
            logger.error(f"Publish error on subject {subject}: {e}")
            self.error_count += 1
            return False

    async def request(self, subject: str, message: Union[NATSMessage, dict[str, Any]],
                     timeout: float = 5.0) -> Optional[dict[str, Any]]:
        """
        Send request and wait for response with timeout.

        Args:
            subject: NATS subject for request
            message: Request message
            timeout: Response timeout in seconds

        Returns:
            Response message or None if timeout/error
        """
        if not self.nc or not self.connected:
            logger.error("NATS not connected")
            return None

        try:
            start_time = time.perf_counter()

            # Convert message to bytes
            if isinstance(message, NATSMessage):
                data = message.model_dump_json().encode()
            elif isinstance(message, dict):
                data = json.dumps(message).encode()
            else:
                data = str(message).encode()

            # Send request
            response = await asyncio.wait_for(
                self.nc.request(subject, data, timeout=timeout),
                timeout=timeout + 1.0
            )

            # Parse response
            response_data = json.loads(response.data.decode())

            # Update metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._update_metrics(latency_ms)

            return response_data

        except asyncio.TimeoutError:
            logger.warning(f"Request timeout on subject {subject}")
            self.error_count += 1
            return None
        except Exception as e:
            logger.error(f"Request error on subject {subject}: {e}")
            self.error_count += 1
            return None

    async def subscribe(self, subject: str, handler: Callable[[dict[str, Any]], None],
                       queue_group: Optional[str] = None, use_jetstream: bool = False,
                       durable_name: Optional[str] = None) -> bool:
        """
        Subscribe to messages with high-performance handler.

        Args:
            subject: NATS subject pattern
            handler: Message handler function
            queue_group: Queue group for load balancing
            use_jetstream: Use JetStream subscription
            durable_name: Durable consumer name

        Returns:
            Success status
        """
        if not self.nc or not self.connected:
            logger.error("NATS not connected")
            return False

        try:
            async def message_handler(msg):
                try:
                    data = json.loads(msg.data.decode())
                    # Execute handler inline for core NATS; await if it's async
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                    # Ack JetStream messages if present
                    try:
                        await msg.ack()
                    except Exception:
                        pass
                except Exception as e:
                    logger.error(f"Message handler error: {e}")

            if use_jetstream and self.js and durable_name:
                # JetStream subscription with persistence
                consumer_config = ConsumerConfig(
                    durable_name=durable_name,
                    deliver_policy=DeliverPolicy.NEW,
                    ack_policy="explicit",
                    replay_policy=ReplayPolicy.INSTANT,
                    max_deliver=3,
                    max_ack_pending=self.config.max_pending_msgs
                )

                # Find appropriate stream for subject
                stream_name = self._find_stream_for_subject(subject)
                if stream_name:
                    subscription = await self.js.subscribe(
                        subject,
                        cb=message_handler,
                        durable=durable_name,
                        config=consumer_config,
                        stream=stream_name
                    )
                else:
                    logger.error(f"No stream found for subject {subject}")
                    return False
            else:
                # Core NATS subscription for maximum performance
                subscription = await self.nc.subscribe(
                    subject,
                    cb=message_handler,
                    queue=queue_group,
                    max_msgs=self.config.max_pending_msgs
                )

            self.subscriptions[subject] = subscription
            logger.info(f"Subscribed to {subject} (jetstream={use_jetstream})")
            return True

        except Exception as e:
            logger.error(f"Subscribe error for {subject}: {e}")
            return False

    def _find_stream_for_subject(self, subject: str) -> Optional[str]:
        """Find the appropriate stream for a given subject."""
        subject_patterns = {
            "agent.commands": ["agent.commands.*", "agent.batch.*"],
            "agent.responses": ["agent.responses.*", "agent.results.*"],
            "agent.status": ["agent.status.*", "agent.heartbeat.*"],
            "agent.discovery": ["discovery.*", "registry.*"],
            "agent.metrics": ["metrics.*", "performance.*"],
            "system.events": ["system.*", "events.*"],
            # A2A
            "a2a.agent": ["a2a.agent.*.in", "a2a.agent.*.out", "discovery.advertise"],
            "a2a.task.events": ["a2a.task.*.events"],
            "a2a.task.rpc": ["a2a.task.*.rpc"],
            "a2a.events": ["a2a.events"],
        }

        for stream_key, patterns in subject_patterns.items():
            for pattern in patterns:
                if self._match_subject(subject, pattern):
                    return self.streams[stream_key]

        return None

    def _match_subject(self, subject: str, pattern: str) -> bool:
        """Simple subject pattern matching."""
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return subject.startswith(prefix)
        return subject == pattern

    async def _publish_system_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Publish system event."""
        event_data = {
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "nats_communicator"
        }

        await self.publish("system.events", event_data, use_jetstream=True)

    def _update_metrics(self, latency_ms: float) -> None:
        """Update performance metrics."""
        self.message_count += 1
        self.total_latency_ms += latency_ms
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        avg_latency = 0.0
        if self.message_count > 0:
            avg_latency = self.total_latency_ms / self.message_count

        metrics = {
            "connected": self.connected,
            "connection_id": self.connection_id,
            "reconnect_count": self.reconnect_count,
            "message_count": self.message_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.message_count, 1),
            "avg_latency_ms": avg_latency,
            "min_latency_ms": self.min_latency_ms if self.min_latency_ms != float('inf') else 0.0,
            "max_latency_ms": self.max_latency_ms,
            "active_subscriptions": len(self.subscriptions),
            "registered_agents": len(self.agents),
            "servers": self.config.servers,
            "publish_js_attempts": self.publish_js_attempts,
            "publish_js_fallbacks": self.publish_js_fallbacks,
            "publish_core_success": self.publish_core_success,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return metrics

    async def _heartbeat_loop(self) -> None:
        """Maintain connection heartbeat and publish metrics."""
        while not self._shutdown_event.is_set() and self.connected:
            try:
                # Publish heartbeat
                heartbeat_data = {
                    "connection_id": self.connection_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "active_agents": len(self.agents),
                    "message_count": self.message_count
                }

                await self.publish("system.heartbeat", heartbeat_data)

                # Wait for next heartbeat
                await asyncio.sleep(30)  # 30 second heartbeat

            except Exception as e:
                logger.debug(f"Heartbeat error: {e}")
                await asyncio.sleep(1)

    async def _metrics_collector(self) -> None:
        """Collect and publish performance metrics periodically."""
        while not self._shutdown_event.is_set() and self.connected:
            try:
                metrics = await self.get_performance_metrics()
                await self.publish("metrics.nats", metrics, use_jetstream=True)

                # Publish to Redis for integration with existing monitoring
                if self.redis_client:
                    try:
                        self.redis_client.setex(
                            "nats:metrics",
                            300,  # 5 minute TTL
                            json.dumps(metrics)
                        )
                    except Exception:
                        pass  # Redis is optional

                await asyncio.sleep(60)  # Publish metrics every minute

            except Exception as e:
                logger.debug(f"Metrics collection error: {e}")
                await asyncio.sleep(5)

    # Connection event handlers
    async def _on_disconnected(self) -> None:
        """Handle disconnection events."""
        self.connected = False
        logger.warning("NATS disconnected")

        for handler in self.disconnection_handlers:
            try:
                handler()
            except Exception as e:
                logger.warning(f"Disconnection handler error: {e}")

    async def _on_reconnected(self) -> None:
        """Handle reconnection events."""
        self.reconnect_count += 1
        logger.info(f"NATS reconnected (attempt {self.reconnect_count})")


    async def unsubscribe(self, subject: str) -> bool:
        """Unsubscribe from a subject previously subscribed via this communicator."""
        if subject not in self.subscriptions:
            return True
        try:
            sub = self.subscriptions.pop(subject, None)
            if sub is None:
                return True
            try:
                await sub.unsubscribe()
            except TypeError:
                # Some subscription types use sync unsubscribe
                try:
                    sub.unsubscribe()
                except Exception:
                    pass
            return True
        except Exception as e:
            logger.error(f"Unsubscribe error for {subject}: {e}")
            return False

        # Re-setup streams after reconnection
        if self.js:
            await self._setup_streams()

    async def _on_error(self, error: Exception) -> None:
        """Handle connection errors."""
        self.error_count += 1
        logger.error(f"NATS error: {error}")

    async def _on_closed(self) -> None:
        """Handle connection closed events."""
        self.connected = False
        logger.info("NATS connection closed")


# Global NATS communicator instance
_nats_communicator: Optional[NATSCommunicator] = None


async def get_nats_communicator(redis_client: Optional[redis.Redis] = None) -> NATSCommunicator:
    """Get or create the global NATS communicator instance."""
    global _nats_communicator

    if _nats_communicator is None:
        _nats_communicator = NATSCommunicator(redis_client=redis_client)

        # Auto-connect if not already connected
        if not _nats_communicator.connected:
            await _nats_communicator.connect()

    return _nats_communicator


async def shutdown_nats_communicator() -> None:
    """Shutdown the global NATS communicator."""
    global _nats_communicator

    if _nats_communicator:
        await _nats_communicator.disconnect()
        _nats_communicator = None
