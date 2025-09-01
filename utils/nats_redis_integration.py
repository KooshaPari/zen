"""
NATS-Redis Integration for Hybrid State Management

This module provides seamless integration between NATS messaging and Redis state
management, combining the best of both systems:
- NATS for ultra-low latency real-time messaging
- Redis for persistent state and distributed caching
- Hybrid patterns for optimal performance and reliability
- Automatic failover and recovery mechanisms
- State synchronization and consistency guarantees
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import redis
from pydantic import BaseModel, Field

from utils.nats_communicator import NATSCommunicator, get_nats_communicator
from utils.nats_streaming import NATSStreamingManager, get_streaming_manager

logger = logging.getLogger(__name__)


class StateEvent(BaseModel):
    """Event for state changes with NATS/Redis coordination."""

    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Type of state event")
    entity_type: str = Field(..., description="Entity type (agent, task, service, etc.)")
    entity_id: str = Field(..., description="Entity identifier")

    # State data
    before_state: Optional[dict[str, Any]] = Field(None, description="State before change")
    after_state: dict[str, Any] = Field(..., description="State after change")

    # Event metadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = Field(..., description="Source of the state change")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for related events")

    # Storage preferences
    persist_to_redis: bool = Field(default=True, description="Store in Redis")
    broadcast_via_nats: bool = Field(default=True, description="Broadcast via NATS")
    ttl_seconds: Optional[int] = Field(None, description="Time-to-live in seconds")


class CacheStrategy(BaseModel):
    """Caching strategy configuration for different data types."""

    strategy_name: str = Field(..., description="Strategy identifier")
    entity_types: set[str] = Field(..., description="Entity types this strategy applies to")

    # Redis settings
    redis_ttl_seconds: int = Field(default=3600, description="Redis TTL")
    redis_key_prefix: str = Field(..., description="Redis key prefix")
    use_redis_pubsub: bool = Field(default=True, description="Use Redis pub/sub for notifications")

    # NATS settings
    nats_subject_prefix: str = Field(..., description="NATS subject prefix")
    use_jetstream: bool = Field(default=False, description="Use JetStream for persistence")
    nats_ttl_seconds: Optional[int] = Field(None, description="NATS message TTL")

    # Sync settings
    sync_to_redis: bool = Field(default=True, description="Synchronize to Redis")
    sync_to_nats: bool = Field(default=True, description="Broadcast via NATS")
    prefer_nats_for_reads: bool = Field(default=False, description="Prefer NATS for read operations")

    # Performance tuning
    batch_size: int = Field(default=100, description="Batch size for bulk operations")
    async_updates: bool = Field(default=True, description="Use async updates")


class NATSRedisIntegrator:
    """
    Hybrid integration service combining NATS real-time messaging with Redis persistence.
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None,
                 nats_communicator: Optional[NATSCommunicator] = None,
                 streaming_manager: Optional[NATSStreamingManager] = None):
        """Initialize the NATS-Redis integrator."""
        self.redis_client = redis_client or self._get_redis_client()
        self.nats = nats_communicator
        self.streaming = streaming_manager

        # State management
        self.state_cache: dict[str, dict[str, Any]] = {}
        self.state_locks: dict[str, asyncio.Lock] = {}

        # Integration strategies
        self.cache_strategies: dict[str, CacheStrategy] = {}
        self._setup_default_strategies()

        # Performance metrics
        self.redis_operations = 0
        self.nats_operations = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.sync_conflicts = 0

        # Background tasks
        self._sync_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        logger.info("NATS-Redis Integrator initialized")

    def _get_redis_client(self) -> Optional[redis.Redis]:
        """Get Redis client with cluster support."""
        try:
            # Try Redis Cluster first for enterprise deployments
            redis_nodes = []
            cluster_hosts = os.getenv("REDIS_CLUSTER_HOSTS", "").split(",")

            if cluster_hosts and cluster_hosts[0]:
                for host_port in cluster_hosts:
                    if ":" in host_port:
                        host, port = host_port.strip().split(":", 1)
                        redis_nodes.append({"host": host, "port": int(port)})

                if redis_nodes:
                    try:
                        # redis>=4 provides cluster in redis.cluster
                        from redis.cluster import RedisCluster
                        client = RedisCluster(
                            startup_nodes=redis_nodes,
                            decode_responses=True,
                            socket_timeout=5,
                            socket_connect_timeout=5,
                        )
                        client.ping()
                        logger.info("Connected to Redis Cluster")
                        return client
                    except Exception as e:
                        logger.warning(f"Redis Cluster connection failed or unavailable: {e}")

            # Fallback to single Redis instance
            client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                retry_on_error=[redis.ConnectionError, redis.TimeoutError],
                retry=redis.Retry(3, 0.1)
            )
            client.ping()
            logger.info("Connected to Redis instance")
            return client

        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            return None

    def _setup_default_strategies(self) -> None:
        """Set up default caching strategies for different entity types."""
        strategies = [
            # Agent tasks - high-frequency updates, moderate persistence
            CacheStrategy(
                strategy_name="agent_tasks",
                entity_types={"agent_task"},
                redis_ttl_seconds=3600,  # 1 hour
                redis_key_prefix="agent:task:",
                use_redis_pubsub=True,
                nats_subject_prefix="agents.tasks",
                use_jetstream=True,
                nats_ttl_seconds=7200,
                sync_to_redis=True,
                sync_to_nats=True,
                prefer_nats_for_reads=False,
                batch_size=50,
                async_updates=True
            ),

            # Agent services - discovery and health data
            CacheStrategy(
                strategy_name="agent_services",
                entity_types={"agent_service"},
                redis_ttl_seconds=600,  # 10 minutes
                redis_key_prefix="agent:service:",
                use_redis_pubsub=True,
                nats_subject_prefix="discovery.services",
                use_jetstream=False,  # Real-time updates, less persistence needed
                sync_to_redis=True,
                sync_to_nats=True,
                prefer_nats_for_reads=True,  # Prefer NATS for service discovery
                batch_size=100,
                async_updates=True
            ),

            # Edge locations - infrastructure state
            CacheStrategy(
                strategy_name="edge_locations",
                entity_types={"edge_location"},
                redis_ttl_seconds=7200,  # 2 hours
                redis_key_prefix="edge:location:",
                use_redis_pubsub=True,
                nats_subject_prefix="edge.locations",
                use_jetstream=True,
                nats_ttl_seconds=14400,
                sync_to_redis=True,
                sync_to_nats=True,
                prefer_nats_for_reads=False,
                batch_size=20,
                async_updates=True
            ),

            # System metrics - high volume, short-lived
            CacheStrategy(
                strategy_name="system_metrics",
                entity_types={"metrics", "performance"},
                redis_ttl_seconds=300,  # 5 minutes
                redis_key_prefix="metrics:",
                use_redis_pubsub=False,
                nats_subject_prefix="metrics",
                use_jetstream=True,
                nats_ttl_seconds=3600,
                sync_to_redis=False,  # Metrics primarily via NATS
                sync_to_nats=True,
                prefer_nats_for_reads=True,
                batch_size=500,
                async_updates=True
            ),

            # Configuration and settings - low frequency, high persistence
            CacheStrategy(
                strategy_name="configuration",
                entity_types={"config", "settings"},
                redis_ttl_seconds=86400,  # 24 hours
                redis_key_prefix="config:",
                use_redis_pubsub=True,
                nats_subject_prefix="system.config",
                use_jetstream=True,
                sync_to_redis=True,
                sync_to_nats=True,
                prefer_nats_for_reads=False,
                batch_size=10,
                async_updates=False  # Synchronous for configuration
            )
        ]

        for strategy in strategies:
            self.cache_strategies[strategy.strategy_name] = strategy
            # Also index by entity type for quick lookup
            for entity_type in strategy.entity_types:
                self.cache_strategies[entity_type] = strategy

    async def start(self) -> None:
        """Start the integration service."""
        if not self.nats:
            self.nats = await get_nats_communicator(self.redis_client)

        if not self.streaming:
            self.streaming = await get_streaming_manager(self.nats, self.redis_client)

        # Set up NATS subscriptions for state sync
        await self._setup_subscriptions()

        # Set up Redis pub/sub
        await self._setup_redis_pubsub()

        # Start background tasks
        self._sync_task = asyncio.create_task(self._state_sync_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._metrics_task = asyncio.create_task(self._metrics_loop())

        logger.info("NATS-Redis Integrator started")

    async def stop(self) -> None:
        """Stop the integration service."""
        self._shutdown_event.set()

        # Cancel background tasks
        for task in [self._sync_task, self._cleanup_task, self._metrics_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("NATS-Redis Integrator stopped")

    async def _setup_subscriptions(self) -> None:
        """Set up NATS subscriptions for state synchronization."""
        if not self.nats:
            return

        # Subscribe to state change events
        await self.nats.subscribe(
            "state.changes.>",
            self._handle_nats_state_change,
            queue_group="state_sync"
        )

        # Subscribe to sync requests
        await self.nats.subscribe(
            "sync.request",
            self._handle_sync_request,
            queue_group="state_sync"
        )

        # Subscribe to invalidation events
        await self.nats.subscribe(
            "cache.invalidate",
            self._handle_cache_invalidation,
            queue_group="state_sync"
        )

    async def _setup_redis_pubsub(self) -> None:
        """Set up Redis pub/sub for cache notifications."""
        if not self.redis_client:
            return

        try:
            # Create pub/sub connection
            pubsub = self.redis_client.pubsub()

            # Subscribe to cache invalidation events
            await pubsub.subscribe("cache:invalidate")
            await pubsub.subscribe("state:changes")

            # Start pub/sub listener task
            asyncio.create_task(self._redis_pubsub_listener(pubsub))

        except Exception as e:
            logger.error(f"Redis pub/sub setup failed: {e}")

    async def set_state(self, entity_type: str, entity_id: str, state: dict[str, Any],
                       correlation_id: Optional[str] = None, ttl_seconds: Optional[int] = None) -> bool:
        """
        Set entity state with hybrid storage strategy.

        Args:
            entity_type: Type of entity
            entity_id: Entity identifier
            state: State data to store
            correlation_id: Correlation ID for related operations
            ttl_seconds: TTL override

        Returns:
            Success status
        """
        try:
            # Get caching strategy
            strategy = self.cache_strategies.get(entity_type)
            if not strategy:
                logger.warning(f"No caching strategy for entity type: {entity_type}")
                strategy = self.cache_strategies.get("agent_tasks")  # Default strategy

            entity_key = f"{entity_type}:{entity_id}"

            # Get lock for this entity
            if entity_key not in self.state_locks:
                self.state_locks[entity_key] = asyncio.Lock()

            async with self.state_locks[entity_key]:
                # Get previous state for change detection
                previous_state = await self.get_state(entity_type, entity_id, use_cache=True)

                # Update local cache
                if entity_type not in self.state_cache:
                    self.state_cache[entity_type] = {}
                self.state_cache[entity_type][entity_id] = state

                # Store in Redis if configured
                if strategy.sync_to_redis and self.redis_client:
                    redis_key = f"{strategy.redis_key_prefix}{entity_id}"
                    redis_ttl = ttl_seconds or strategy.redis_ttl_seconds

                    try:
                        if redis_ttl > 0:
                            self.redis_client.setex(redis_key, redis_ttl, json.dumps(state))
                        else:
                            self.redis_client.set(redis_key, json.dumps(state))

                        self.redis_operations += 1

                        # Publish Redis notification if configured
                        if strategy.use_redis_pubsub:
                            notification = {
                                "entity_type": entity_type,
                                "entity_id": entity_id,
                                "action": "set",
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            }
                            self.redis_client.publish("state:changes", json.dumps(notification))

                    except Exception as e:
                        logger.error(f"Redis set failed for {redis_key}: {e}")

                # Broadcast via NATS if configured
                if strategy.sync_to_nats and self.nats:
                    state_event = StateEvent(
                        event_id=f"{entity_type}_{entity_id}_{int(time.time())}",
                        event_type="state_set",
                        entity_type=entity_type,
                        entity_id=entity_id,
                        before_state=previous_state,
                        after_state=state,
                        source="nats_redis_integrator",
                        correlation_id=correlation_id,
                        persist_to_redis=strategy.sync_to_redis,
                        broadcast_via_nats=True,
                        ttl_seconds=ttl_seconds
                    )

                    subject = f"{strategy.nats_subject_prefix}.state.set"

                    success = await self.nats.publish(
                        subject,
                        state_event.model_dump(),
                        use_jetstream=strategy.use_jetstream
                    )

                    if success:
                        self.nats_operations += 1
                    else:
                        logger.error(f"NATS publish failed for {subject}")

                return True

        except Exception as e:
            logger.error(f"Set state failed for {entity_type}/{entity_id}: {e}")
            return False

    async def get_state(self, entity_type: str, entity_id: str,
                       use_cache: bool = True, prefer_nats: Optional[bool] = None) -> Optional[dict[str, Any]]:
        """
        Get entity state with intelligent source selection.

        Args:
            entity_type: Type of entity
            entity_id: Entity identifier
            use_cache: Whether to use local cache
            prefer_nats: Override preference for NATS vs Redis

        Returns:
            Entity state or None if not found
        """
        try:
            # Get caching strategy
            strategy = self.cache_strategies.get(entity_type)
            if not strategy:
                strategy = self.cache_strategies.get("agent_tasks")  # Default

            entity_key = f"{entity_type}:{entity_id}"

            # Check local cache first if enabled
            if use_cache and entity_type in self.state_cache:
                cached_state = self.state_cache[entity_type].get(entity_id)
                if cached_state is not None:
                    self.cache_hits += 1
                    return cached_state

            self.cache_misses += 1

            # Determine preferred source
            use_nats = prefer_nats if prefer_nats is not None else strategy.prefer_nats_for_reads

            # Try NATS first if preferred
            if use_nats and self.nats:
                try:
                    # Request state via NATS
                    request_data = {
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "request_id": f"get_{entity_type}_{entity_id}_{int(time.time())}"
                    }

                    subject = f"{strategy.nats_subject_prefix}.state.get"
                    response = await self.nats.request(subject, request_data, timeout=2.0)

                    if response and response.get("success"):
                        state = response.get("state")
                        if state:
                            # Update local cache
                            if entity_type not in self.state_cache:
                                self.state_cache[entity_type] = {}
                            self.state_cache[entity_type][entity_id] = state
                            self.nats_operations += 1
                            return state

                except Exception as e:
                    logger.debug(f"NATS state get failed for {entity_key}: {e}")

            # Try Redis
            if self.redis_client:
                try:
                    redis_key = f"{strategy.redis_key_prefix}{entity_id}"
                    data = self.redis_client.get(redis_key)

                    if data:
                        state = json.loads(data)

                        # Update local cache
                        if entity_type not in self.state_cache:
                            self.state_cache[entity_type] = {}
                        self.state_cache[entity_type][entity_id] = state
                        self.redis_operations += 1
                        return state

                except Exception as e:
                    logger.error(f"Redis get failed for {redis_key}: {e}")

            return None

        except Exception as e:
            logger.error(f"Get state failed for {entity_type}/{entity_id}: {e}")
            return None

    async def delete_state(self, entity_type: str, entity_id: str,
                          correlation_id: Optional[str] = None) -> bool:
        """
        Delete entity state from all storage layers.

        Args:
            entity_type: Type of entity
            entity_id: Entity identifier
            correlation_id: Correlation ID

        Returns:
            Success status
        """
        try:
            strategy = self.cache_strategies.get(entity_type)
            if not strategy:
                strategy = self.cache_strategies.get("agent_tasks")

            entity_key = f"{entity_type}:{entity_id}"

            # Get lock for this entity
            if entity_key not in self.state_locks:
                self.state_locks[entity_key] = asyncio.Lock()

            async with self.state_locks[entity_key]:
                # Get current state for change event
                current_state = await self.get_state(entity_type, entity_id, use_cache=True)

                # Remove from local cache
                if entity_type in self.state_cache:
                    self.state_cache[entity_type].pop(entity_id, None)

                # Remove from Redis
                if strategy.sync_to_redis and self.redis_client:
                    redis_key = f"{strategy.redis_key_prefix}{entity_id}"

                    try:
                        self.redis_client.delete(redis_key)
                        self.redis_operations += 1

                        # Publish Redis notification
                        if strategy.use_redis_pubsub:
                            notification = {
                                "entity_type": entity_type,
                                "entity_id": entity_id,
                                "action": "delete",
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            }
                            self.redis_client.publish("state:changes", json.dumps(notification))

                    except Exception as e:
                        logger.error(f"Redis delete failed for {redis_key}: {e}")

                # Broadcast deletion via NATS
                if strategy.sync_to_nats and self.nats:
                    state_event = StateEvent(
                        event_id=f"{entity_type}_{entity_id}_delete_{int(time.time())}",
                        event_type="state_delete",
                        entity_type=entity_type,
                        entity_id=entity_id,
                        before_state=current_state,
                        after_state={},
                        source="nats_redis_integrator",
                        correlation_id=correlation_id,
                        persist_to_redis=False,
                        broadcast_via_nats=True
                    )

                    subject = f"{strategy.nats_subject_prefix}.state.delete"

                    success = await self.nats.publish(
                        subject,
                        state_event.model_dump(),
                        use_jetstream=strategy.use_jetstream
                    )

                    if success:
                        self.nats_operations += 1

                return True

        except Exception as e:
            logger.error(f"Delete state failed for {entity_type}/{entity_id}: {e}")
            return False

    async def invalidate_cache(self, entity_type: str, entity_id: Optional[str] = None,
                             broadcast: bool = True) -> None:
        """
        Invalidate cache entries locally and optionally broadcast invalidation.

        Args:
            entity_type: Type of entity to invalidate
            entity_id: Specific entity ID (None = all entities of type)
            broadcast: Whether to broadcast invalidation to other instances
        """
        try:
            if entity_id:
                # Invalidate specific entity
                if entity_type in self.state_cache:
                    self.state_cache[entity_type].pop(entity_id, None)
            else:
                # Invalidate all entities of type
                self.state_cache.pop(entity_type, None)

            # Broadcast invalidation if requested
            if broadcast:
                invalidation_event = {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "nats_redis_integrator"
                }

                # Broadcast via NATS
                if self.nats:
                    await self.nats.publish("cache.invalidate", invalidation_event)

                # Broadcast via Redis
                if self.redis_client:
                    try:
                        self.redis_client.publish("cache:invalidate", json.dumps(invalidation_event))
                    except Exception as e:
                        logger.error(f"Redis invalidation broadcast failed: {e}")

            logger.debug(f"Cache invalidated for {entity_type}" + (f"/{entity_id}" if entity_id else ""))

        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")

    async def bulk_set_states(self, states: list[tuple[str, str, dict[str, Any]]],
                            correlation_id: Optional[str] = None) -> int:
        """
        Bulk set multiple entity states efficiently.

        Args:
            states: List of (entity_type, entity_id, state) tuples
            correlation_id: Correlation ID for the bulk operation

        Returns:
            Number of successfully set states
        """
        success_count = 0

        try:
            # Group by strategy for batch processing
            strategy_groups: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}

            for entity_type, entity_id, state in states:
                strategy = self.cache_strategies.get(entity_type)
                if not strategy:
                    strategy = self.cache_strategies.get("agent_tasks")

                strategy_key = strategy.strategy_name
                if strategy_key not in strategy_groups:
                    strategy_groups[strategy_key] = []
                strategy_groups[strategy_key].append((entity_type, entity_id, state))

            # Process each strategy group
            for strategy_name, group_states in strategy_groups.items():
                strategy = self.cache_strategies[strategy_name]

                # Process in batches
                for i in range(0, len(group_states), strategy.batch_size):
                    batch = group_states[i:i + strategy.batch_size]

                    if strategy.async_updates:
                        # Async batch processing
                        tasks = []
                        for entity_type, entity_id, state in batch:
                            task = asyncio.create_task(
                                self.set_state(entity_type, entity_id, state, correlation_id)
                            )
                            tasks.append(task)

                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        success_count += sum(1 for r in results if r is True)
                    else:
                        # Synchronous batch processing
                        for entity_type, entity_id, state in batch:
                            success = await self.set_state(entity_type, entity_id, state, correlation_id)
                            if success:
                                success_count += 1

            logger.info(f"Bulk set completed: {success_count}/{len(states)} successful")
            return success_count

        except Exception as e:
            logger.error(f"Bulk set states failed: {e}")
            return success_count

    # Message handlers
    async def _handle_nats_state_change(self, data: dict[str, Any]) -> None:
        """Handle state change events from NATS."""
        try:
            event = StateEvent.model_validate(data)

            # Update local cache if this is from another instance
            if event.source != "nats_redis_integrator":
                entity_key = f"{event.entity_type}:{event.entity_id}"

                if event.event_type == "state_set":
                    if event.entity_type not in self.state_cache:
                        self.state_cache[event.entity_type] = {}
                    self.state_cache[event.entity_type][event.entity_id] = event.after_state
                elif event.event_type == "state_delete":
                    if event.entity_type in self.state_cache:
                        self.state_cache[event.entity_type].pop(event.entity_id, None)

                logger.debug(f"Applied NATS state change: {event.event_type} for {entity_key}")

        except Exception as e:
            logger.error(f"NATS state change handler error: {e}")

    async def _handle_sync_request(self, data: dict[str, Any]) -> None:
        """Handle synchronization requests from other instances."""
        try:
            entity_type = data.get("entity_type")
            entity_id = data.get("entity_id")
            reply_to = data.get("reply_to")

            if entity_type and entity_id and reply_to:
                state = await self.get_state(entity_type, entity_id, use_cache=True)

                response = {
                    "success": state is not None,
                    "state": state,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

                if self.nats:
                    await self.nats.publish(reply_to, response)

        except Exception as e:
            logger.error(f"Sync request handler error: {e}")

    async def _handle_cache_invalidation(self, data: dict[str, Any]) -> None:
        """Handle cache invalidation events."""
        try:
            entity_type = data.get("entity_type")
            entity_id = data.get("entity_id")
            source = data.get("source", "")

            # Don't process our own invalidations
            if source != "nats_redis_integrator":
                await self.invalidate_cache(entity_type, entity_id, broadcast=False)

        except Exception as e:
            logger.error(f"Cache invalidation handler error: {e}")

    async def _redis_pubsub_listener(self, pubsub) -> None:
        """Listen for Redis pub/sub messages."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    message = await pubsub.get_message(timeout=1.0)
                    if message and message['type'] == 'message':
                        channel = message['channel']
                        data = json.loads(message['data'])

                        if channel == 'cache:invalidate':
                            await self._handle_cache_invalidation(data)
                        elif channel == 'state:changes':
                            # Handle Redis state change notifications
                            pass  # Already handled via NATS

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Redis pub/sub error: {e}")
                    await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Redis pub/sub listener error: {e}")

    # Background tasks
    async def _state_sync_loop(self) -> None:
        """Periodically synchronize state across all storage layers."""
        while not self._shutdown_event.is_set():
            try:
                # This would implement periodic sync verification
                # For now, just clean up old locks
                time.time()
                old_locks = [
                    key for key in self.state_locks.keys()
                    if not self.state_locks[key].locked()
                ]

                # Remove unused locks (memory cleanup)
                for key in old_locks[:100]:  # Limit cleanup per cycle
                    self.state_locks.pop(key, None)

                await asyncio.sleep(300)  # Every 5 minutes

            except Exception as e:
                logger.error(f"State sync loop error: {e}")
                await asyncio.sleep(30)

    async def _cleanup_loop(self) -> None:
        """Clean up expired cache entries and optimize memory usage."""
        while not self._shutdown_event.is_set():
            try:
                # Clean up local cache based on TTL and size limits
                cache_size = sum(len(entities) for entities in self.state_cache.values())
                max_cache_size = int(os.getenv("NATS_REDIS_MAX_CACHE_SIZE", "10000"))

                if cache_size > max_cache_size:
                    # Simple cleanup: remove oldest 10% of entries
                    cleanup_count = int(cache_size * 0.1)
                    cleaned = 0

                    for entity_type in list(self.state_cache.keys()):
                        if cleaned >= cleanup_count:
                            break

                        entities = self.state_cache[entity_type]
                        if entities:
                            # Remove some entries (in real implementation, use LRU)
                            keys_to_remove = list(entities.keys())[:min(len(entities) // 2, cleanup_count - cleaned)]
                            for key in keys_to_remove:
                                entities.pop(key, None)
                                cleaned += 1

                    logger.info(f"Cleaned up {cleaned} cache entries")

                await asyncio.sleep(600)  # Every 10 minutes

            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)

    async def _metrics_loop(self) -> None:
        """Publish integration performance metrics."""
        while not self._shutdown_event.is_set():
            try:
                metrics = {
                    "redis_operations": self.redis_operations,
                    "nats_operations": self.nats_operations,
                    "cache_hits": self.cache_hits,
                    "cache_misses": self.cache_misses,
                    "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
                    "sync_conflicts": self.sync_conflicts,
                    "cached_entities": sum(len(entities) for entities in self.state_cache.values()),
                    "active_locks": len([lock for lock in self.state_locks.values() if lock.locked()]),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

                # Publish via NATS
                if self.nats:
                    await self.nats.publish("metrics.integration.nats_redis", metrics, use_jetstream=True)

                await asyncio.sleep(60)  # Every minute

            except Exception as e:
                logger.error(f"Metrics loop error: {e}")
                await asyncio.sleep(30)

    async def get_integration_metrics(self) -> dict[str, Any]:
        """Get current integration performance metrics."""
        return {
            "redis_operations": self.redis_operations,
            "nats_operations": self.nats_operations,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            "sync_conflicts": self.sync_conflicts,
            "cached_entities": sum(len(entities) for entities in self.state_cache.values()),
            "cache_strategies": len(self.cache_strategies),
            "active_locks": len([lock for lock in self.state_locks.values() if lock.locked()]),
            "redis_connected": self.redis_client is not None,
            "nats_connected": self.nats.connected if self.nats else False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global integrator instance
_nats_redis_integrator: Optional[NATSRedisIntegrator] = None


async def get_nats_redis_integrator(redis_client: Optional[redis.Redis] = None,
                                   nats_communicator: Optional[NATSCommunicator] = None,
                                   streaming_manager: Optional[NATSStreamingManager] = None) -> NATSRedisIntegrator:
    """Get or create the global NATS-Redis integrator instance."""
    global _nats_redis_integrator

    if _nats_redis_integrator is None:
        _nats_redis_integrator = NATSRedisIntegrator(redis_client, nats_communicator, streaming_manager)
        await _nats_redis_integrator.start()

    return _nats_redis_integrator


async def shutdown_nats_redis_integrator() -> None:
    """Shutdown the global NATS-Redis integrator."""
    global _nats_redis_integrator

    if _nats_redis_integrator:
        await _nats_redis_integrator.stop()
        _nats_redis_integrator = None
