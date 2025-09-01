# Redis Integration Architecture for Zen MCP Server

## Executive Summary

This document outlines the comprehensive Redis integration architecture designed to scale the Zen MCP Server agent orchestration system from 10-50 concurrent agents to 1000+ agents with enterprise-grade performance and reliability.

## Current State Analysis

The existing system includes:
- **AgentTaskManager**: Basic Redis integration for task persistence
- **In-memory storage**: Primary state in Python dictionaries
- **Simple port pooling**: 100-port range (3284-3384)
- **Basic event bus**: In-process pub/sub
- **Optional Redis**: Backup storage with 1-hour TTL

### Identified Limitations
- **Memory bottlenecks**: In-memory state won't scale to 1000+ agents
- **Single-point failure**: No clustering or high availability
- **Limited coordination**: No real-time agent coordination
- **Basic caching**: No performance optimization layers
- **Port exhaustion**: 100-port limit insufficient for 1000+ agents

## Proposed Redis Architecture

### 1. Redis Clustering Strategy

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Redis Master  │    │   Redis Master  │    │   Redis Master  │
│     Shard 0     │    │     Shard 1     │    │     Shard 2     │
│                 │    │                 │    │                 │
│ Agent State     │    │ Agent Memory    │    │ Coordination    │
│ Task Queues     │    │ Sessions        │    │ Events/Pub/Sub  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
┌─────────▼───────┐    ┌─────────▼───────┐    ┌─────────▼───────┐
│  Redis Replica  │    │  Redis Replica  │    │  Redis Replica  │
│    (Read-Only)  │    │    (Read-Only)  │    │    (Read-Only)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2. Data Structure Design

#### Key Naming Conventions
```
# Agent State
agent:state:{agent_id}                    # Agent configuration and status
agent:memory:{agent_id}:short            # Short-term memory (15min TTL)
agent:memory:{agent_id}:working          # Working memory (1hr TTL)
agent:memory:{agent_id}:long             # Long-term memory (24hr TTL)

# Task Management
task:{task_id}                           # Task metadata and status
task:queue:{priority}                    # Priority-based task queues
task:result:{task_id}                    # Task results and outputs
task:metrics:{agent_id}                  # Performance metrics

# Coordination
coord:locks:{resource_id}                # Distributed locks
coord:sessions:{session_id}              # Agent sessions
coord:events                             # Event stream
coord:heartbeat:{agent_id}               # Agent health monitoring

# Indexes
idx:agents:active                        # Active agent set
idx:tasks:by_status:{status}             # Tasks by status
idx:tasks:by_created                     # Tasks by creation time
idx:agents:by_type:{type}                # Agents by type
```

#### Memory Optimization Strategy
```
Memory Tier 1: Hot Data (Redis Memory)
├── Active agent states (current tasks)
├── Short-term memory (15min TTL)
├── Task queues and results
└── Real-time coordination data

Memory Tier 2: Warm Data (Redis with compression)
├── Working memory (1hr TTL)
├── Recent task history
├── Session data
└── Performance metrics

Memory Tier 3: Cold Data (Redis with aggressive TTL)
├── Long-term memory (24hr TTL)
├── Historical metrics
├── Archived task results
└── Audit trails
```

### 3. Agent State Management

#### Agent State Schema
```python
{
    "agent_id": "uuid",
    "agent_type": "claude|aider|goose|...",
    "status": "pending|starting|running|idle|error|terminated",
    "current_task_id": "uuid",
    "port": 3284,
    "process_id": 12345,
    "started_at": "2025-01-15T10:30:00Z",
    "last_heartbeat": "2025-01-15T10:35:00Z",
    "memory_usage": 256,  # MB
    "cpu_usage": 15.5,    # %
    "task_count": 42,
    "success_rate": 0.95,
    "average_response_time": 1.2,  # seconds
    "capabilities": ["code_generation", "file_editing"],
    "configuration": {
        "max_memory": 512,
        "timeout_seconds": 300,
        "retry_count": 3
    }
}
```

#### Agent Memory Management
```python
# Short-term memory (15min TTL)
{
    "conversation_context": "recent interactions",
    "active_files": ["file1.py", "file2.js"],
    "temporary_variables": {"key": "value"}
}

# Working memory (1hr TTL)
{
    "task_history": [{"task_id": "...", "summary": "..."}],
    "learned_patterns": ["pattern1", "pattern2"],
    "context_embeddings": [0.1, 0.2, 0.3, ...]
}

# Long-term memory (24hr TTL)
{
    "expertise_areas": ["python", "react", "databases"],
    "performance_history": {"avg_time": 2.1, "success_rate": 0.98},
    "preference_settings": {"style": "functional", "verbosity": "medium"}
}
```

### 4. Performance Optimization

#### Multi-Level Caching Strategy
```
L1 Cache: In-Process Memory
├── Recently accessed agent states
├── Active task metadata
├── Frequently used configurations
└── Hot coordination data

L2 Cache: Redis Memory
├── Agent state snapshots
├── Task result cache
├── Session data
└── Index lookups

L3 Cache: Redis with Compression
├── Historical data
├── Archived results
├── Performance metrics
└── Audit logs
```

#### Connection Pooling and Optimization
```python
# Redis connection configuration
REDIS_POOL_CONFIG = {
    "max_connections": 100,
    "retry_on_timeout": True,
    "socket_keepalive": True,
    "socket_keepalive_options": {},
    "health_check_interval": 30,
    "connection_pool_class": "redis.connection.BlockingConnectionPool"
}

# Cluster configuration
REDIS_CLUSTER_CONFIG = {
    "startup_nodes": [
        {"host": "redis-node-1", "port": 7000},
        {"host": "redis-node-2", "port": 7001},
        {"host": "redis-node-3", "port": 7002}
    ],
    "decode_responses": True,
    "skip_full_coverage_check": True,
    "max_connections_per_node": 50
}
```

### 5. Real-Time Coordination

#### Pub/Sub Channels
```
# Agent lifecycle events
channel:agent:started
channel:agent:stopped
channel:agent:error
channel:agent:heartbeat

# Task coordination
channel:task:assigned
channel:task:completed
channel:task:failed
channel:task:priority_changed

# System coordination
channel:system:scale_up
channel:system:scale_down
channel:system:maintenance
channel:system:emergency_stop

# Inter-agent communication
channel:agent:{agent_id}:message
channel:broadcast:all_agents
channel:broadcast:{agent_type}
```

#### Redis Streams for Event Logging
```
# Event stream structure
XADD events:agent_lifecycle * 
    event_type "agent_started"
    agent_id "uuid"
    agent_type "claude"
    timestamp "1642248600000"
    metadata "{\"port\": 3284, \"pid\": 12345}"

XADD events:task_execution *
    event_type "task_completed"
    task_id "uuid"
    agent_id "uuid"
    duration_ms "1250"
    status "success"
    result_size "4096"
```

### 6. Concurrency and Locking

#### Optimistic Locking Strategy
```python
# Version-based optimistic locking
def update_agent_state(agent_id: str, updates: dict) -> bool:
    with redis_client.pipeline() as pipe:
        pipe.watch(f"agent:state:{agent_id}")
        current_state = pipe.get(f"agent:state:{agent_id}")
        
        if current_state:
            state_data = json.loads(current_state)
            current_version = state_data.get("version", 0)
            
            # Apply updates
            state_data.update(updates)
            state_data["version"] = current_version + 1
            state_data["updated_at"] = datetime.utcnow().isoformat()
            
            # Atomic update
            pipe.multi()
            pipe.set(f"agent:state:{agent_id}", json.dumps(state_data))
            pipe.execute()
            return True
        return False
```

#### Distributed Locking for Critical Resources
```python
# Redis distributed lock implementation
def acquire_resource_lock(resource_id: str, timeout: int = 30) -> bool:
    lock_key = f"coord:locks:{resource_id}"
    lock_value = f"{uuid.uuid4()}:{time.time()}"
    
    # SET with NX and EX for atomic lock acquisition
    result = redis_client.set(
        lock_key, 
        lock_value, 
        nx=True, 
        ex=timeout
    )
    return result is True
```

### 7. High Availability and Failover

#### Health Monitoring
```python
# Automated health checks
async def monitor_redis_health():
    for node in redis_cluster_nodes:
        try:
            response_time = await ping_redis_node(node)
            if response_time > HEALTH_THRESHOLD:
                await trigger_failover(node)
        except Exception as e:
            await handle_node_failure(node, e)
```

#### Automatic Failover Strategy
```
Primary Failure Scenario:
1. Detect primary node failure (health check timeout)
2. Promote replica to primary (Redis Sentinel)
3. Update client connection pools
4. Migrate active agent states
5. Resume operations on new primary
6. Log failover event for audit

Data Consistency:
- Use Redis Sentinel for automatic failover
- Implement eventual consistency for non-critical data
- Use strong consistency for task states and locks
- Maintain transaction logs for data recovery
```

## Integration APIs for Other Agents

### 1. NATS Agent Integration
```python
# Connection state caching API
class RedisNATSIntegration:
    def cache_connection_state(self, connection_id: str, state: dict):
        """Cache NATS connection state for fast reconnection"""
        redis_client.setex(
            f"nats:connection:{connection_id}",
            300,  # 5min TTL
            json.dumps(state)
        )
    
    def get_cached_connection(self, connection_id: str) -> Optional[dict]:
        """Retrieve cached connection state"""
        data = redis_client.get(f"nats:connection:{connection_id}")
        return json.loads(data) if data else None
```

### 2. Kafka Agent Integration
```python
# Event distribution API
class RedisKafkaIntegration:
    def publish_to_kafka_queue(self, event: dict):
        """Queue events for Kafka distribution"""
        redis_client.lpush("kafka:event_queue", json.dumps(event))
    
    def get_audit_trail(self, agent_id: str) -> List[dict]:
        """Get agent audit trail for Kafka logging"""
        events = redis_client.xrange(
            "events:agent_lifecycle",
            min="-",
            max="+",
            count=1000
        )
        return [self._parse_stream_event(e) for e in events]
```

### 3. Temporal Agent Integration
```python
# Workflow context API
class RedisTemporalIntegration:
    def store_workflow_context(self, workflow_id: str, context: dict):
        """Store workflow context for Temporal orchestration"""
        redis_client.hset(
            f"temporal:workflow:{workflow_id}",
            "context",
            json.dumps(context)
        )
        redis_client.expire(f"temporal:workflow:{workflow_id}", 3600)
    
    def get_agent_session_data(self, session_id: str) -> dict:
        """Get agent session data for workflow coordination"""
        return redis_client.hgetall(f"coord:sessions:{session_id}")
```

### 4. Testing Agent Integration
```python
# Performance monitoring API
class RedisTestingIntegration:
    def get_performance_metrics(self, time_range: tuple) -> dict:
        """Get performance metrics for testing analysis"""
        start_time, end_time = time_range
        metrics = {}
        
        # Agent performance metrics
        agent_keys = redis_client.keys("task:metrics:*")
        for key in agent_keys:
            agent_id = key.split(":")[-1]
            agent_metrics = redis_client.hgetall(key)
            metrics[agent_id] = agent_metrics
        
        return metrics
    
    def create_test_snapshot(self, test_id: str):
        """Create system state snapshot for testing"""
        snapshot = {
            "active_agents": redis_client.scard("idx:agents:active"),
            "pending_tasks": redis_client.llen("task:queue:normal"),
            "memory_usage": self._get_redis_memory_usage(),
            "timestamp": datetime.utcnow().isoformat()
        }
        redis_client.setex(
            f"test:snapshot:{test_id}",
            1800,  # 30min TTL
            json.dumps(snapshot)
        )
```

## Implementation Phases

### Phase 1: Core Redis Infrastructure (Week 1-2)
- [ ] Implement `utils/redis_manager.py` with clustering support
- [ ] Create connection pooling and health monitoring
- [ ] Set up basic state management with TTL policies
- [ ] Implement optimistic locking mechanisms

### Phase 2: Agent State Management (Week 2-3)
- [ ] Implement `utils/agent_state.py` with comprehensive state management
- [ ] Create memory tier management (short/working/long term)
- [ ] Add agent health monitoring and heartbeat system
- [ ] Implement performance metrics collection

### Phase 3: Advanced Features (Week 3-4)
- [ ] Implement `utils/agent_memory.py` with vector similarity
- [ ] Add Redis pub/sub coordination system
- [ ] Create Redis Streams for event logging
- [ ] Implement multi-level caching strategy

### Phase 4: Integration and Testing (Week 4-5)
- [ ] Integrate with existing AgentTaskManager
- [ ] Create APIs for other agents (NATS, Kafka, Temporal, Testing)
- [ ] Comprehensive unit and integration testing
- [ ] Performance benchmarking and optimization

## Success Metrics

### Performance Targets
- **Concurrent agents**: 1000+ simultaneous agents
- **State retrieval**: < 1 second for any agent state
- **Memory efficiency**: < 1GB Redis memory per 100 agents
- **Availability**: 99.9% uptime with automatic failover
- **Throughput**: 10,000+ operations per second per Redis node

### Monitoring and Alerting
- Redis cluster health and performance
- Agent state consistency and integrity
- Memory usage and optimization effectiveness
- Task queue depth and processing latency
- Inter-agent coordination response times

This architecture provides a robust foundation for scaling the Zen MCP Server to enterprise levels while maintaining performance, reliability, and integration capabilities for other agent systems.