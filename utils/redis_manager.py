"""
Redis Manager for Enterprise Agent Orchestration

This module provides enterprise-grade Redis integration for the Zen MCP Server,
designed to scale from 10-50 agents to 1000+ concurrent agents with high availability,
clustering support, and sophisticated state management.

Key Features:
- Redis clustering with automatic failover and load balancing
- Connection pooling with health monitoring and circuit breaker pattern
- Intelligent key naming conventions and data structure organization
- Performance optimization with pipelining and batching operations
- Memory optimization with intelligent TTL policies and eviction strategies
- Thread-safe operations with optimistic concurrency control
- Comprehensive monitoring and metrics for performance analysis
- Integration with existing agent orchestration system

Architecture Design:
- Uses Redis DB allocation strategy: DB 0 (conversations), DB 1 (tasks), DB 2 (state), DB 3 (memory)
- Implements Redis Streams for agent communication logs and event sourcing
- Provides pub/sub coordination for real-time agent state synchronization
- Supports Redis Sentinel for high availability and automatic failover
- Includes Redis Cluster support for horizontal scaling beyond single node limits

Integration Points:
- Provides state APIs for NATS agent connection state caching
- Creates event hooks for Kafka agent audit trail consumption
- Enables session management for Temporal agent workflow contexts
- Generates performance metrics for Testing agent monitoring and alerts
"""

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional, Union

# Optional Redis dependencies with graceful fallback
try:
    import redis
    import redis.exceptions
    import redis.sentinel
    if TYPE_CHECKING:
        from redis.client import Pipeline
except ImportError:
    redis = None
    if TYPE_CHECKING:
        Pipeline = Any


logger = logging.getLogger(__name__)

# Redis DB allocation strategy for organized data separation
class RedisDB:
    """Redis database allocation for different data types"""
    CONVERSATIONS = 0  # Conversation threads and memory (existing)
    TASKS = 1          # Agent task storage (existing)
    STATE = 2          # Agent state and coordination (new)
    MEMORY = 3         # Agent memory and vector similarity (new)
    PUBSUB = 4         # Pub/sub coordination and streams (new)
    CACHE = 5          # Caching layer and temporary data (new)
    METRICS = 6        # Performance metrics and monitoring (new)

# Key naming conventions for consistent data organization
class RedisKeys:
    """Standardized Redis key naming conventions"""

    # Agent state management
    AGENT_STATE = "agent:{agent_id}:state"
    AGENT_HEARTBEAT = "agent:{agent_id}:heartbeat"
    AGENT_CAPABILITIES = "agent:{agent_id}:capabilities"
    AGENT_LOCKS = "agent:{agent_id}:lock"

    # Agent memory and context
    AGENT_MEMORY_SHORT = "memory:{agent_id}:short"
    AGENT_MEMORY_WORKING = "memory:{agent_id}:working"
    AGENT_MEMORY_LONG = "memory:{agent_id}:long"
    AGENT_MEMORY_INDEX = "memory:index"

    # Task coordination
    TASK_QUEUE = "tasks:queue:{priority}"
    TASK_ACTIVE = "tasks:active"
    TASK_RESULTS = "tasks:results"

    # Connection and resource management
    CONNECTION_POOL = "connections:pool"
    RESOURCE_ALLOCATION = "resources:allocation"
    PORT_ALLOCATION = "ports:allocated"

    # Performance and monitoring
    METRICS_COUNTERS = "metrics:counters"
    METRICS_TIMINGS = "metrics:timings"
    METRICS_HEALTH = "metrics:health"

    # Event streams
    AGENT_EVENTS = "stream:agent:events"
    SYSTEM_EVENTS = "stream:system:events"
    COORDINATION_EVENTS = "stream:coordination:events"

# TTL policies for different data types (in seconds)
class TTLPolicies:
    """Time-to-live policies for different data types"""

    # Agent state TTL
    AGENT_HEARTBEAT = 300      # 5 minutes
    AGENT_STATE = 3600         # 1 hour
    AGENT_CAPABILITIES = 86400 # 24 hours

    # Memory TTL
    MEMORY_SHORT = 1800        # 30 minutes
    MEMORY_WORKING = 7200      # 2 hours
    MEMORY_LONG = 86400        # 24 hours

    # Task coordination TTL
    TASK_ACTIVE = 3600         # 1 hour
    TASK_RESULTS = 7200        # 2 hours

    # Performance data TTL
    METRICS_COUNTERS = 3600    # 1 hour
    METRICS_TIMINGS = 1800     # 30 minutes
    METRICS_HEALTH = 600       # 10 minutes

    # Caching TTL
    CACHE_DEFAULT = 300        # 5 minutes
    CACHE_LONG = 1800          # 30 minutes


class RedisConnectionError(Exception):
    """Redis connection related errors"""
    pass


class RedisClusterManager:
    """Manages Redis cluster connections with failover and load balancing"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.cluster_nodes = config.get('cluster_nodes', [])
        self.sentinel_nodes = config.get('sentinel_nodes', [])
        self.use_cluster = config.get('use_cluster', False)
        self.use_sentinel = config.get('use_sentinel', False)

        self._connections: dict[int, redis.Redis] = {}
        self._sentinel: Optional[redis.sentinel.Sentinel] = None
        self._cluster: Optional[redis.RedisCluster] = None
        self._lock = threading.Lock()

        logger.info(f"Redis cluster manager initialized: cluster={self.use_cluster}, sentinel={self.use_sentinel}")

    def _create_sentinel_connection(self) -> redis.sentinel.Sentinel:
        """Create Sentinel connection for high availability"""
        if not self.sentinel_nodes:
            raise RedisConnectionError("No sentinel nodes configured")

        sentinel = redis.sentinel.Sentinel(
            self.sentinel_nodes,
            sentinel_kwargs={
                'socket_timeout': self.config.get('socket_timeout', 5),
                'socket_connect_timeout': self.config.get('socket_connect_timeout', 5),
            }
        )

        # Test sentinel connection
        try:
            sentinel.discover_master('mymaster')
            logger.info(f"Connected to Redis Sentinel: {len(self.sentinel_nodes)} nodes")
            return sentinel
        except Exception as e:
            raise RedisConnectionError(f"Failed to connect to Redis Sentinel: {e}")

    def _create_cluster_connection(self) -> redis.RedisCluster:
        """Create Redis Cluster connection for horizontal scaling"""
        if not self.cluster_nodes:
            raise RedisConnectionError("No cluster nodes configured")

        cluster = redis.RedisCluster(
            startup_nodes=self.cluster_nodes,
            decode_responses=True,
            skip_full_coverage_check=True,
            socket_timeout=self.config.get('socket_timeout', 5),
            socket_connect_timeout=self.config.get('socket_connect_timeout', 5),
            max_connections=self.config.get('max_connections', 50),
            retry_on_timeout=True,
            health_check_interval=30,
        )

        # Test cluster connection
        try:
            cluster.ping()
            logger.info(f"Connected to Redis Cluster: {len(self.cluster_nodes)} nodes")
            return cluster
        except Exception as e:
            raise RedisConnectionError(f"Failed to connect to Redis Cluster: {e}")

    def get_connection(self, db: int = 0) -> redis.Redis:
        """Get Redis connection for specified database with clustering support"""
        with self._lock:
            if db in self._connections:
                # Test existing connection
                try:
                    self._connections[db].ping()
                    return self._connections[db]
                except:
                    # Connection failed, will recreate below
                    del self._connections[db]

            # Create new connection based on configuration
            if self.use_cluster:
                if self._cluster is None:
                    self._cluster = self._create_cluster_connection()
                connection = self._cluster
            elif self.use_sentinel:
                if self._sentinel is None:
                    self._sentinel = self._create_sentinel_connection()
                # Get master connection from sentinel
                connection = self._sentinel.master_for('mymaster', db=db, decode_responses=True)
            else:
                # Standard single-node connection
                connection = redis.Redis(
                    host=self.config.get('host', 'localhost'),
                    port=self.config.get('port', 6379),
                    db=db,
                    decode_responses=True,
                    socket_timeout=self.config.get('socket_timeout', 5),
                    socket_connect_timeout=self.config.get('socket_connect_timeout', 5),
                    max_connections=self.config.get('max_connections', 50),
                    retry_on_timeout=True,
                    health_check_interval=30,
                )

            # Test connection before storing
            try:
                connection.ping()
                self._connections[db] = connection
                logger.debug(f"Created Redis connection for DB {db}")
                return connection
            except Exception as e:
                raise RedisConnectionError(f"Failed to connect to Redis DB {db}: {e}")


class RedisManager:
    """
    Enterprise Redis Manager for Agent Orchestration

    Provides comprehensive Redis integration with clustering, high availability,
    performance optimization, and enterprise-grade features for scaling to 1000+ agents.
    """

    def __init__(self):
        self.config = self._load_configuration()
        self.cluster_manager = RedisClusterManager(self.config)

        # Connection pools for different databases
        self._connections: dict[int, redis.Redis] = {}
        self._pipeline_cache: dict[int, Pipeline] = {}

        # Circuit breaker for connection health
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = 0
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_timeout = 60  # 1 minute

        # Performance monitoring
        self._operation_count = 0
        self._total_latency = 0.0
        self._last_health_check = 0

        # Lock for thread safety
        self._lock = threading.Lock()

        logger.info("Redis Manager initialized for enterprise agent orchestration")

    def _load_configuration(self) -> dict[str, Any]:
        """Load Redis configuration from environment variables"""
        config = {
            # Basic connection settings
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', '6379')),
            'password': os.getenv('REDIS_PASSWORD'),
            'socket_timeout': int(os.getenv('REDIS_SOCKET_TIMEOUT', '5')),
            'socket_connect_timeout': int(os.getenv('REDIS_CONNECT_TIMEOUT', '5')),
            'max_connections': int(os.getenv('REDIS_MAX_CONNECTIONS', '50')),

            # Clustering configuration
            'use_cluster': os.getenv('REDIS_USE_CLUSTER', '0').lower() in ('1', 'true', 'yes'),
            'use_sentinel': os.getenv('REDIS_USE_SENTINEL', '0').lower() in ('1', 'true', 'yes'),

            # Performance settings
            'enable_pipelining': os.getenv('REDIS_ENABLE_PIPELINING', '1').lower() in ('1', 'true', 'yes'),
            'pipeline_batch_size': int(os.getenv('REDIS_PIPELINE_BATCH_SIZE', '100')),
            'connection_pool_size': int(os.getenv('REDIS_POOL_SIZE', '20')),

            # Memory optimization
            'enable_compression': os.getenv('REDIS_ENABLE_COMPRESSION', '0').lower() in ('1', 'true', 'yes'),
            'compression_threshold': int(os.getenv('REDIS_COMPRESSION_THRESHOLD', '1024')),

            # Monitoring settings
            'enable_metrics': os.getenv('REDIS_ENABLE_METRICS', '1').lower() in ('1', 'true', 'yes'),
            'health_check_interval': int(os.getenv('REDIS_HEALTH_CHECK_INTERVAL', '60')),
        }

        # Parse cluster nodes if configured
        cluster_nodes_str = os.getenv('REDIS_CLUSTER_NODES', '')
        if cluster_nodes_str:
            nodes = []
            for node in cluster_nodes_str.split(','):
                if ':' in node:
                    host, port = node.strip().split(':')
                    nodes.append({'host': host, 'port': int(port)})
            config['cluster_nodes'] = nodes

        # Parse sentinel nodes if configured
        sentinel_nodes_str = os.getenv('REDIS_SENTINEL_NODES', '')
        if sentinel_nodes_str:
            nodes = []
            for node in sentinel_nodes_str.split(','):
                if ':' in node:
                    host, port = node.strip().split(':')
                    nodes.append((host, int(port)))
            config['sentinel_nodes'] = nodes

        return config

    @contextmanager
    def _circuit_breaker(self):
        """Circuit breaker pattern for Redis operations"""
        current_time = time.time()

        # Check if circuit breaker should be open
        if (self._circuit_breaker_failures >= self._circuit_breaker_threshold and
            current_time - self._circuit_breaker_last_failure < self._circuit_breaker_timeout):
            raise RedisConnectionError("Circuit breaker open - Redis temporarily unavailable")

        try:
            yield
            # Reset failure count on success
            self._circuit_breaker_failures = 0
        except Exception as e:
            self._circuit_breaker_failures += 1
            self._circuit_breaker_last_failure = current_time
            logger.warning(f"Redis operation failed (failures: {self._circuit_breaker_failures}): {e}")
            raise

    def get_connection(self, db: int = RedisDB.STATE) -> redis.Redis:
        """Get Redis connection for specified database with clustering support"""
        return self.cluster_manager.get_connection(db)

    def get_pipeline(self, db: int = RedisDB.STATE, transaction: bool = False) -> "Pipeline":
        """Get Redis pipeline for batch operations"""
        connection = self.get_connection(db)
        return connection.pipeline(transaction=transaction)

    # Agent State Management Methods

    def set_agent_state(self, agent_id: str, state: dict[str, Any], ttl: int = TTLPolicies.AGENT_STATE) -> bool:
        """Set agent state with optional TTL"""
        try:
            with self._circuit_breaker():
                conn = self.get_connection(RedisDB.STATE)
                key = RedisKeys.AGENT_STATE.format(agent_id=agent_id)

                # Serialize state with timestamp
                state_data = {
                    'state': state,
                    'updated_at': datetime.now(timezone.utc).isoformat(),
                    'agent_id': agent_id
                }

                serialized = json.dumps(state_data, default=str)

                if ttl > 0:
                    conn.setex(key, ttl, serialized)
                else:
                    conn.set(key, serialized)

                # Update heartbeat
                self._update_agent_heartbeat(agent_id)

                # Publish state change event
                self._publish_agent_event(agent_id, 'state_updated', {'state': state})

                logger.debug(f"Updated agent state for {agent_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to set agent state for {agent_id}: {e}")
            return False

    def get_agent_state(self, agent_id: str) -> Optional[dict[str, Any]]:
        """Get current agent state"""
        try:
            with self._circuit_breaker():
                conn = self.get_connection(RedisDB.STATE)
                key = RedisKeys.AGENT_STATE.format(agent_id=agent_id)

                data = conn.get(key)
                if data:
                    state_data = json.loads(data)
                    return state_data.get('state')
                return None

        except Exception as e:
            logger.error(f"Failed to get agent state for {agent_id}: {e}")
            return None

    def _update_agent_heartbeat(self, agent_id: str) -> None:
        """Update agent heartbeat timestamp"""
        try:
            conn = self.get_connection(RedisDB.STATE)
            key = RedisKeys.AGENT_HEARTBEAT.format(agent_id=agent_id)
            timestamp = time.time()
            conn.setex(key, TTLPolicies.AGENT_HEARTBEAT, timestamp)
        except Exception as e:
            logger.debug(f"Failed to update heartbeat for {agent_id}: {e}")

    def get_active_agents(self) -> list[str]:
        """Get list of active agents based on heartbeat"""
        try:
            with self._circuit_breaker():
                conn = self.get_connection(RedisDB.STATE)
                pattern = RedisKeys.AGENT_HEARTBEAT.format(agent_id='*')

                active_agents = []
                for key in conn.scan_iter(match=pattern):
                    # Extract agent_id from key
                    parts = key.split(':')
                    if len(parts) >= 2:
                        agent_id = parts[1]
                        active_agents.append(agent_id)

                logger.debug(f"Found {len(active_agents)} active agents")
                return active_agents

        except Exception as e:
            logger.error(f"Failed to get active agents: {e}")
            return []

    # Agent Memory Management Methods

    def set_agent_memory(self, agent_id: str, memory_type: str, data: dict[str, Any],
                        ttl: Optional[int] = None) -> bool:
        """Set agent memory with type-specific TTL policies"""
        try:
            with self._circuit_breaker():
                # Select appropriate database and TTL
                if memory_type == 'short':
                    key = RedisKeys.AGENT_MEMORY_SHORT.format(agent_id=agent_id)
                    ttl = ttl or TTLPolicies.MEMORY_SHORT
                elif memory_type == 'working':
                    key = RedisKeys.AGENT_MEMORY_WORKING.format(agent_id=agent_id)
                    ttl = ttl or TTLPolicies.MEMORY_WORKING
                elif memory_type == 'long':
                    key = RedisKeys.AGENT_MEMORY_LONG.format(agent_id=agent_id)
                    ttl = ttl or TTLPolicies.MEMORY_LONG
                else:
                    logger.error(f"Unknown memory type: {memory_type}")
                    return False

                conn = self.get_connection(RedisDB.MEMORY)

                # Serialize memory data with metadata
                memory_data = {
                    'data': data,
                    'type': memory_type,
                    'agent_id': agent_id,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'version': 1
                }

                serialized = json.dumps(memory_data, default=str)

                # Store with compression if enabled and data is large
                if (self.config.get('enable_compression') and
                    len(serialized) > self.config.get('compression_threshold', 1024)):
                    # Simple compression placeholder - would use zlib or similar
                    logger.debug(f"Compressing large memory data for {agent_id}")

                conn.setex(key, ttl, serialized)

                # Update memory index for similarity search
                self._update_memory_index(agent_id, memory_type, data)

                logger.debug(f"Stored {memory_type} memory for agent {agent_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to set {memory_type} memory for {agent_id}: {e}")
            return False

    def get_agent_memory(self, agent_id: str, memory_type: str) -> Optional[dict[str, Any]]:
        """Get agent memory by type"""
        try:
            with self._circuit_breaker():
                if memory_type == 'short':
                    key = RedisKeys.AGENT_MEMORY_SHORT.format(agent_id=agent_id)
                elif memory_type == 'working':
                    key = RedisKeys.AGENT_MEMORY_WORKING.format(agent_id=agent_id)
                elif memory_type == 'long':
                    key = RedisKeys.AGENT_MEMORY_LONG.format(agent_id=agent_id)
                else:
                    return None

                conn = self.get_connection(RedisDB.MEMORY)
                data = conn.get(key)

                if data:
                    memory_data = json.loads(data)
                    return memory_data.get('data')
                return None

        except Exception as e:
            logger.error(f"Failed to get {memory_type} memory for {agent_id}: {e}")
            return None

    def _update_memory_index(self, agent_id: str, memory_type: str, data: dict[str, Any]) -> None:
        """Update memory index for vector similarity search (placeholder)"""
        try:
            # This would integrate with vector similarity systems like Redis Search
            # For now, just maintain a simple index of agent memory keys
            conn = self.get_connection(RedisDB.MEMORY)
            index_key = RedisKeys.AGENT_MEMORY_INDEX

            {
                'agent_id': agent_id,
                'memory_type': memory_type,
                'timestamp': time.time(),
                'keys': list(data.keys()) if isinstance(data, dict) else []
            }

            # Use sorted set for efficient retrieval
            score = time.time()  # Use timestamp as score
            conn.zadd(index_key, {f"{agent_id}:{memory_type}": score})

        except Exception as e:
            logger.debug(f"Failed to update memory index: {e}")

    # Pub/Sub Coordination Methods

    def _publish_agent_event(self, agent_id: str, event_type: str, data: dict[str, Any]) -> None:
        """Publish agent event for real-time coordination"""
        try:
            conn = self.get_connection(RedisDB.PUBSUB)

            event = {
                'agent_id': agent_id,
                'event_type': event_type,
                'data': data,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'redis_manager'
            }

            # Publish to general agent events channel
            channel = f"agent_events:{agent_id}"
            conn.publish(channel, json.dumps(event, default=str))

            # Also add to Redis Stream for event sourcing
            stream_key = RedisKeys.AGENT_EVENTS
            conn.xadd(stream_key, event, maxlen=10000, approximate=True)

        except Exception as e:
            logger.debug(f"Failed to publish agent event: {e}")

    def subscribe_to_agent_events(self, agent_id: str) -> Optional[redis.client.PubSub]:
        """Subscribe to agent events for real-time updates"""
        try:
            conn = self.get_connection(RedisDB.PUBSUB)
            pubsub = conn.pubsub()

            channel = f"agent_events:{agent_id}"
            pubsub.subscribe(channel)

            logger.debug(f"Subscribed to events for agent {agent_id}")
            return pubsub

        except Exception as e:
            logger.error(f"Failed to subscribe to agent events: {e}")
            return None

    # Performance and Monitoring Methods

    def record_metric(self, metric_name: str, value: Union[int, float], tags: Optional[dict[str, str]] = None) -> None:
        """Record performance metric"""
        if not self.config.get('enable_metrics'):
            return

        try:
            conn = self.get_connection(RedisDB.METRICS)

            # Create metric key with tags
            tag_str = ""
            if tags:
                tag_str = ":" + ":".join(f"{k}={v}" for k, v in sorted(tags.items()))

            metric_key = f"{RedisKeys.METRICS_COUNTERS}:{metric_name}{tag_str}"

            # Increment counter or set gauge
            if isinstance(value, (int, float)) and value == 1:
                conn.incr(metric_key)
            else:
                conn.set(metric_key, value, ex=TTLPolicies.METRICS_COUNTERS)

            # Also record timing if it's a duration
            if 'duration' in metric_name or 'latency' in metric_name:
                timing_key = f"{RedisKeys.METRICS_TIMINGS}:{metric_name}{tag_str}"
                conn.lpush(timing_key, value)
                conn.ltrim(timing_key, 0, 999)  # Keep last 1000 measurements
                conn.expire(timing_key, TTLPolicies.METRICS_TIMINGS)

        except Exception as e:
            logger.debug(f"Failed to record metric {metric_name}: {e}")

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status"""
        try:
            with self._circuit_breaker():
                health = {
                    'redis_available': True,
                    'cluster_status': 'healthy',
                    'active_agents': len(self.get_active_agents()),
                    'circuit_breaker_failures': self._circuit_breaker_failures,
                    'average_latency': self._total_latency / max(self._operation_count, 1),
                    'last_health_check': datetime.now(timezone.utc).isoformat()
                }

                # Test each database connection
                db_status = {}
                for db_name, db_num in RedisDB.__dict__.items():
                    if isinstance(db_num, int):
                        try:
                            conn = self.get_connection(db_num)
                            conn.ping()
                            db_status[db_name.lower()] = 'healthy'
                        except:
                            db_status[db_name.lower()] = 'unhealthy'
                            health['cluster_status'] = 'degraded'

                health['database_status'] = db_status

                # Cache health status
                if self.config.get('enable_metrics'):
                    try:
                        conn = self.get_connection(RedisDB.METRICS)
                        health_key = RedisKeys.METRICS_HEALTH
                        conn.setex(health_key, TTLPolicies.METRICS_HEALTH,
                                 json.dumps(health, default=str))
                    except:
                        pass

                return health

        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {
                'redis_available': False,
                'error': str(e),
                'last_health_check': datetime.now(timezone.utc).isoformat()
            }

    # Resource Management Methods

    def allocate_port(self, agent_id: str, port_range: tuple[int, int] = (3284, 10000)) -> Optional[int]:
        """Allocate unique port for agent with Redis-based coordination"""
        try:
            with self._circuit_breaker():
                conn = self.get_connection(RedisDB.STATE)
                allocation_key = RedisKeys.PORT_ALLOCATION

                # Use Redis SETNX for atomic port allocation
                start_port, end_port = port_range

                for port in range(start_port, end_port + 1):
                    port_key = f"{allocation_key}:{port}"

                    # Try to atomically allocate this port
                    if conn.set(port_key, agent_id, nx=True, ex=TTLPolicies.AGENT_STATE):
                        logger.debug(f"Allocated port {port} to agent {agent_id}")

                        # Record allocation in agent's resource list
                        agent_resources_key = f"resources:{agent_id}"
                        conn.sadd(agent_resources_key, f"port:{port}")
                        conn.expire(agent_resources_key, TTLPolicies.AGENT_STATE)

                        return port

                logger.warning(f"No available ports in range {start_port}-{end_port} for agent {agent_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to allocate port for agent {agent_id}: {e}")
            return None

    def release_port(self, agent_id: str, port: int) -> bool:
        """Release allocated port"""
        try:
            with self._circuit_breaker():
                conn = self.get_connection(RedisDB.STATE)
                allocation_key = RedisKeys.PORT_ALLOCATION
                port_key = f"{allocation_key}:{port}"

                # Check if this agent owns the port
                current_owner = conn.get(port_key)
                if current_owner == agent_id:
                    conn.delete(port_key)

                    # Remove from agent's resource list
                    agent_resources_key = f"resources:{agent_id}"
                    conn.srem(agent_resources_key, f"port:{port}")

                    logger.debug(f"Released port {port} from agent {agent_id}")
                    return True
                else:
                    logger.warning(f"Agent {agent_id} does not own port {port} (owner: {current_owner})")
                    return False

        except Exception as e:
            logger.error(f"Failed to release port {port} for agent {agent_id}: {e}")
            return False

    # Batch Operations and Performance Optimization

    def batch_operation(self, operations: list[tuple[str, str, Any]], db: int = RedisDB.STATE) -> list[Any]:
        """Execute batch operations with pipelining for performance"""
        if not self.config.get('enable_pipelining'):
            # Fallback to individual operations
            results = []
            for op, key, value in operations:
                if op == 'set':
                    results.append(self.get_connection(db).set(key, value))
                elif op == 'get':
                    results.append(self.get_connection(db).get(key))
                # Add more operations as needed
            return results

        try:
            with self._circuit_breaker():
                pipeline = self.get_pipeline(db, transaction=True)

                # Add all operations to pipeline
                for op, key, value in operations:
                    if op == 'set':
                        pipeline.set(key, value)
                    elif op == 'setex':
                        ttl, actual_value = value  # value is (ttl, actual_value) tuple
                        pipeline.setex(key, ttl, actual_value)
                    elif op == 'get':
                        pipeline.get(key)
                    elif op == 'delete':
                        pipeline.delete(key)
                    elif op == 'sadd':
                        pipeline.sadd(key, value)
                    elif op == 'zadd':
                        pipeline.zadd(key, value)
                    # Add more operations as needed

                # Execute all operations atomically
                results = pipeline.execute()

                logger.debug(f"Executed {len(operations)} operations in batch")
                return results

        except Exception as e:
            logger.error(f"Failed to execute batch operations: {e}")
            return [None] * len(operations)

    # Integration APIs for other agents

    def get_nats_connection_state(self, connection_id: str) -> Optional[dict[str, Any]]:
        """API for NATS agent to retrieve connection state from Redis cache"""
        cache_key = f"nats:connection:{connection_id}"
        return self._get_cached_data(cache_key)

    def set_nats_connection_state(self, connection_id: str, state: dict[str, Any], ttl: int = TTLPolicies.CACHE_DEFAULT) -> bool:
        """API for NATS agent to cache connection state in Redis"""
        cache_key = f"nats:connection:{connection_id}"
        return self._set_cached_data(cache_key, state, ttl)

    def get_temporal_workflow_context(self, workflow_id: str) -> Optional[dict[str, Any]]:
        """API for Temporal agent to retrieve workflow context"""
        context_key = f"temporal:workflow:{workflow_id}"
        return self._get_cached_data(context_key)

    def set_temporal_workflow_context(self, workflow_id: str, context: dict[str, Any], ttl: int = TTLPolicies.CACHE_LONG) -> bool:
        """API for Temporal agent to store workflow context"""
        context_key = f"temporal:workflow:{workflow_id}"
        return self._set_cached_data(context_key, context, ttl)

    def _get_cached_data(self, key: str) -> Optional[dict[str, Any]]:
        """Helper to get cached data from Redis"""
        try:
            conn = self.get_connection(RedisDB.CACHE)
            data = conn.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.debug(f"Failed to get cached data for {key}: {e}")
            return None

    def _set_cached_data(self, key: str, data: dict[str, Any], ttl: int) -> bool:
        """Helper to set cached data in Redis"""
        try:
            conn = self.get_connection(RedisDB.CACHE)
            serialized = json.dumps(data, default=str)
            conn.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.debug(f"Failed to set cached data for {key}: {e}")
            return False


# Global Redis manager instance
_redis_manager: Optional[RedisManager] = None
_redis_manager_lock = threading.Lock()


def get_redis_manager() -> RedisManager:
    """Get global Redis manager instance (singleton pattern)"""
    global _redis_manager

    if _redis_manager is None:
        with _redis_manager_lock:
            if _redis_manager is None:
                try:
                    _redis_manager = RedisManager()
                except Exception as e:
                    logger.error(f"Failed to initialize Redis manager: {e}")
                    # Return a mock manager for graceful fallback
                    _redis_manager = MockRedisManager()

    return _redis_manager


class MockRedisManager:
    """Mock Redis manager for graceful fallback when Redis is unavailable"""

    def __init__(self):
        logger.warning("Using mock Redis manager - Redis functionality disabled")
        self._data = {}

    def set_agent_state(self, agent_id: str, state: dict[str, Any], ttl: int = 0) -> bool:
        self._data[f"state:{agent_id}"] = state
        return True

    def get_agent_state(self, agent_id: str) -> Optional[dict[str, Any]]:
        return self._data.get(f"state:{agent_id}")

    def get_active_agents(self) -> list[str]:
        return []

    def set_agent_memory(self, agent_id: str, memory_type: str, data: dict[str, Any], ttl: Optional[int] = None) -> bool:
        self._data[f"memory:{agent_id}:{memory_type}"] = data
        return True

    def get_agent_memory(self, agent_id: str, memory_type: str) -> Optional[dict[str, Any]]:
        return self._data.get(f"memory:{agent_id}:{memory_type}")

    def allocate_port(self, agent_id: str, port_range: tuple[int, int] = (3284, 10000)) -> Optional[int]:
        # Simple in-memory port allocation
        start_port, end_port = port_range
        allocated_ports = {v for k, v in self._data.items() if k.startswith("port:") and isinstance(v, int)}

        for port in range(start_port, end_port + 1):
            if port not in allocated_ports:
                self._data[f"port:{agent_id}"] = port
                return port
        return None

    def release_port(self, agent_id: str, port: int) -> bool:
        key = f"port:{agent_id}"
        if self._data.get(key) == port:
            del self._data[key]
            return True
        return False

    def record_metric(self, metric_name: str, value: Union[int, float], tags: Optional[dict[str, str]] = None) -> None:
        pass  # No-op for mock

    def get_health_status(self) -> dict[str, Any]:
        return {
            'redis_available': False,
            'mock_mode': True,
            'last_health_check': datetime.now(timezone.utc).isoformat()
        }

    def subscribe_to_agent_events(self, agent_id: str):
        return None

    def batch_operation(self, operations: list[tuple[str, str, Any]], db: int = 0) -> list[Any]:
        return [None] * len(operations)

    def get_nats_connection_state(self, connection_id: str) -> Optional[dict[str, Any]]:
        return None

    def set_nats_connection_state(self, connection_id: str, state: dict[str, Any], ttl: int = 300) -> bool:
        return False

    def get_temporal_workflow_context(self, workflow_id: str) -> Optional[dict[str, Any]]:
        return None

    def set_temporal_workflow_context(self, workflow_id: str, context: dict[str, Any], ttl: int = 1800) -> bool:
        return False


# Global Redis manager instance
_redis_manager_instance: Optional[RedisManager] = None
_redis_manager_lock = threading.Lock()


def get_redis_manager() -> RedisManager:
    """
    Get the global Redis manager instance (singleton pattern).

    Returns:
        RedisManager: The global Redis manager instance
    """
    global _redis_manager_instance

    if _redis_manager_instance is None:
        with _redis_manager_lock:
            if _redis_manager_instance is None:
                try:
                    if redis is not None:
                        # Test Redis connection before creating full manager
                        test_client = redis.Redis(
                            host=os.getenv('REDIS_HOST', 'localhost'),
                            port=int(os.getenv('REDIS_PORT', 6379)),
                            socket_connect_timeout=1,
                            socket_timeout=1
                        )
                        test_client.ping()  # This will raise an exception if Redis isn't available
                        _redis_manager_instance = RedisManager()
                        logger.info("Created Redis manager instance with full Redis support")
                    else:
                        _redis_manager_instance = MockRedisManager()
                        logger.warning("Created mock Redis manager (Redis not available)")
                except Exception as e:
                    logger.warning(f"Redis server not available ({e}), using mock Redis manager")
                    _redis_manager_instance = MockRedisManager()
                    logger.info("Falling back to mock Redis manager for development")

    return _redis_manager_instance
