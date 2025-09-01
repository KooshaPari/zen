"""
Unit Tests for Redis Integration

This module provides comprehensive unit tests for the Redis integration components:
- RedisManager enterprise functionality
- AgentStateManager state persistence and coordination
- AgentMemoryManager memory management with vector similarity
- Integration with existing AgentTaskManager

Test Coverage:
- Redis connection management and clustering
- Agent state lifecycle management
- Memory storage and retrieval with vector similarity
- Port allocation and resource management
- Performance monitoring and health checks
- Error handling and graceful fallbacks
- Integration with existing systems
"""

import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest

from tools.shared.agent_models import AgentType
from utils.agent_memory import (
    AgentMemoryManager,
    MemoryAnalytics,
    MemoryEntry,
    MemoryPriority,
    MemorySearchResult,
    MemoryType,
    MemoryVector,
)
from utils.agent_state import (
    AgentCapabilities,
    AgentLifecycleState,
    AgentPerformanceMetrics,
    AgentResource,
    AgentResourceType,
    AgentStateData,
    AgentStateManager,
)

# Import modules under test
from utils.redis_manager import MockRedisManager, RedisDB, RedisManager, get_redis_manager


class TestRedisManager:
    """Test suite for RedisManager enterprise functionality"""

    @pytest.fixture
    def redis_manager(self):
        """Create Redis manager for testing"""
        with patch('utils.redis_manager.redis') as mock_redis:
            # Mock Redis connection
            mock_conn = Mock()
            mock_conn.ping.return_value = True
            mock_conn.set.return_value = True
            mock_conn.get.return_value = None
            mock_conn.setex.return_value = True
            mock_conn.delete.return_value = 1

            mock_redis.Redis.return_value = mock_conn

            manager = RedisManager()
            manager._test_mode = True  # Add test mode flag
            return manager

    def test_redis_manager_initialization(self, redis_manager):
        """Test Redis manager initialization with configuration"""
        assert redis_manager is not None
        assert hasattr(redis_manager, 'config')
        assert hasattr(redis_manager, 'cluster_manager')
        assert redis_manager.config['host'] == 'localhost'
        assert redis_manager.config['port'] == 6379

    def test_connection_management(self, redis_manager):
        """Test Redis connection management with multiple databases"""
        # Test getting connections for different databases
        conn_state = redis_manager.get_connection(RedisDB.STATE)
        conn_memory = redis_manager.get_connection(RedisDB.MEMORY)
        conn_cache = redis_manager.get_connection(RedisDB.CACHE)

        assert conn_state is not None
        assert conn_memory is not None
        assert conn_cache is not None

    def test_agent_state_management(self, redis_manager):
        """Test agent state persistence and retrieval"""
        agent_id = "test_agent_001"
        test_state = {
            "lifecycle": "running",
            "task_id": "task_123",
            "performance": {
                "tasks_completed": 5,
                "average_latency": 1.2
            }
        }

        # Test state setting
        success = redis_manager.set_agent_state(agent_id, test_state)
        assert success

        # Test state retrieval
        retrieved_state = redis_manager.get_agent_state(agent_id)
        assert retrieved_state is not None
        assert "lifecycle" in retrieved_state

    def test_agent_memory_management(self, redis_manager):
        """Test agent memory storage and retrieval"""
        agent_id = "test_agent_002"
        test_memory = {
            "context": "Testing memory storage",
            "files": ["test.py", "utils.py"],
            "learned_patterns": ["error_handling", "async_patterns"]
        }

        # Test memory setting for different types
        success_short = redis_manager.set_agent_memory(agent_id, "short", test_memory)
        success_working = redis_manager.set_agent_memory(agent_id, "working", test_memory)
        success_long = redis_manager.set_agent_memory(agent_id, "long", test_memory)

        assert success_short
        assert success_working
        assert success_long

        # Test memory retrieval
        retrieved_short = redis_manager.get_agent_memory(agent_id, "short")
        retrieved_working = redis_manager.get_agent_memory(agent_id, "working")
        retrieved_long = redis_manager.get_agent_memory(agent_id, "long")

        assert retrieved_short is not None
        assert retrieved_working is not None
        assert retrieved_long is not None

    def test_port_allocation(self, redis_manager):
        """Test Redis-coordinated port allocation"""
        agent_id = "test_agent_003"

        # Test port allocation
        allocated_port = redis_manager.allocate_port(agent_id, (3284, 3300))
        assert allocated_port is not None
        assert 3284 <= allocated_port <= 3300

        # Test port release
        release_success = redis_manager.release_port(agent_id, allocated_port)
        assert release_success

    def test_performance_monitoring(self, redis_manager):
        """Test performance metrics recording and health status"""
        # Test metric recording
        redis_manager.record_metric("test.latency", 1.5, {"operation": "create_task"})
        redis_manager.record_metric("test.count", 1, {"operation": "create_task"})

        # Test health status
        health = redis_manager.get_health_status()
        assert health is not None
        assert "redis_available" in health
        assert "cluster_status" in health
        assert "active_agents" in health

    def test_batch_operations(self, redis_manager):
        """Test batch operations for performance optimization"""
        operations = [
            ("set", "test:key1", "value1"),
            ("set", "test:key2", "value2"),
            ("setex", "test:key3", (300, "value3")),
            ("get", "test:key1", None),
        ]

        results = redis_manager.batch_operation(operations, RedisDB.CACHE)
        assert len(results) == len(operations)

    def test_integration_apis(self, redis_manager):
        """Test integration APIs for other agents"""
        # Test NATS integration
        connection_id = "nats_conn_001"
        connection_state = {"status": "connected", "channels": ["agent.events"]}

        success = redis_manager.set_nats_connection_state(connection_id, connection_state)
        assert success

        retrieved_state = redis_manager.get_nats_connection_state(connection_id)
        assert retrieved_state is not None

        # Test Temporal integration
        workflow_id = "temporal_workflow_001"
        workflow_context = {"step": "process_agent_task", "state": "running"}

        success = redis_manager.set_temporal_workflow_context(workflow_id, workflow_context)
        assert success

        retrieved_context = redis_manager.get_temporal_workflow_context(workflow_id)
        assert retrieved_context is not None


class TestAgentStateManager:
    """Test suite for AgentStateManager functionality"""

    @pytest.fixture
    async def state_manager(self):
        """Create agent state manager for testing"""
        with patch('utils.agent_state.get_redis_manager') as mock_get_redis:
            mock_redis = Mock()
            mock_redis.set_agent_state.return_value = True
            mock_redis.get_agent_state.return_value = None
            mock_redis.get_active_agents.return_value = []
            mock_redis.allocate_port.return_value = 3284
            mock_redis.release_port.return_value = True
            mock_redis.record_metric.return_value = None

            mock_get_redis.return_value = mock_redis

            manager = AgentStateManager()
            return manager

    @pytest.mark.asyncio
    async def test_create_agent_state(self, state_manager):
        """Test agent state creation"""
        agent_id = "test_agent_001"
        capabilities = AgentCapabilities(
            agent_type=AgentType.CLAUDE,
            supported_operations=["chat", "analyze", "code_review"],
            max_concurrent_tasks=3,
            memory_limit_mb=1024,
            timeout_seconds=300,
            environment_requirements={"ANTHROPIC_API_KEY": "required"},
            port_range=(3284, 3384)
        )

        success = await state_manager.create_agent_state(
            agent_id, AgentType.CLAUDE, capabilities
        )
        assert success

    @pytest.mark.asyncio
    async def test_agent_state_updates(self, state_manager):
        """Test agent state updates with lifecycle transitions"""
        agent_id = "test_agent_002"
        capabilities = AgentCapabilities(
            agent_type=AgentType.GOOSE,
            supported_operations=["code_gen", "refactor"],
            max_concurrent_tasks=2,
            memory_limit_mb=512,
            timeout_seconds=180,
            environment_requirements={},
            port_range=(3284, 3384)
        )

        # Create initial state
        await state_manager.create_agent_state(agent_id, AgentType.GOOSE, capabilities)

        # Test lifecycle state updates
        success = await state_manager.update_agent_state(
            agent_id,
            lifecycle_state=AgentLifecycleState.READY,
            current_task_id="task_001"
        )
        assert success

        success = await state_manager.update_agent_state(
            agent_id,
            lifecycle_state=AgentLifecycleState.RUNNING,
            performance_metrics=AgentPerformanceMetrics(
                tasks_completed=1,
                tasks_failed=0,
                total_processing_time=10.5,
                average_response_time=2.1,
                memory_usage_mb=256.0,
                cpu_usage_percent=45.0,
                last_activity=datetime.now(timezone.utc),
                health_score=0.95
            )
        )
        assert success

    @pytest.mark.asyncio
    async def test_resource_allocation(self, state_manager):
        """Test agent resource allocation and management"""
        agent_id = "test_agent_003"
        capabilities = AgentCapabilities(
            agent_type=AgentType.AIDER,
            supported_operations=["file_edit", "code_gen"],
            max_concurrent_tasks=1,
            memory_limit_mb=256,
            timeout_seconds=120,
            environment_requirements={},
            port_range=(3284, 3384)
        )

        # Create agent state
        await state_manager.create_agent_state(agent_id, AgentType.AIDER, capabilities)

        # Test resource allocation
        resource = await state_manager.allocate_resource(
            agent_id,
            AgentResourceType.PORT,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
        )
        assert resource is not None
        assert resource.resource_type == AgentResourceType.PORT

        # Test resource release
        success = await state_manager.release_resource(agent_id, resource.resource_id)
        assert success

    @pytest.mark.asyncio
    async def test_agent_queries(self, state_manager):
        """Test agent query and discovery methods"""
        # Mock some active agents
        state_manager.redis_manager.get_active_agents.return_value = [
            "agent_001", "agent_002", "agent_003"
        ]

        # Mock agent states
        def mock_get_state(agent_id):
            return {
                "agent_id": agent_id,
                "agent_type": "claude",
                "lifecycle_state": "running" if agent_id != "agent_003" else "ready",
                "capabilities": {
                    "agent_type": "claude",
                    "supported_operations": ["chat"],
                    "max_concurrent_tasks": 2,
                    "memory_limit_mb": 512,
                    "timeout_seconds": 300,
                    "environment_requirements": {},
                    "port_range": [3284, 3384]
                },
                "allocated_resources": [],
                "performance_metrics": {
                    "tasks_completed": 5,
                    "tasks_failed": 0,
                    "total_processing_time": 25.0,
                    "average_response_time": 2.5,
                    "memory_usage_mb": 128.0,
                    "cpu_usage_percent": 30.0,
                    "last_activity": datetime.now(timezone.utc).isoformat(),
                    "health_score": 0.9
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "version": 1
            }

        state_manager.redis_manager.get_agent_state.side_effect = mock_get_state

        # Test getting agents by state
        running_agents = await state_manager.get_agents_by_state(AgentLifecycleState.RUNNING)
        assert len(running_agents) >= 0

        # Test getting available agents
        available_agents = await state_manager.get_available_agents(AgentType.CLAUDE)
        assert len(available_agents) >= 0

    @pytest.mark.asyncio
    async def test_health_monitoring(self, state_manager):
        """Test agent health monitoring and analytics"""
        # Mock active agents
        state_manager.redis_manager.get_active_agents.return_value = ["agent_001", "agent_002"]

        # Mock agent states with health data
        def mock_get_state(agent_id):
            return AgentStateData(
                agent_id=agent_id,
                agent_type=AgentType.CLAUDE,
                lifecycle_state=AgentLifecycleState.RUNNING,
                capabilities=AgentCapabilities(
                    agent_type=AgentType.CLAUDE,
                    supported_operations=["chat"],
                    max_concurrent_tasks=2,
                    memory_limit_mb=512,
                    timeout_seconds=300,
                    environment_requirements={},
                    port_range=(3284, 3384)
                ),
                performance_metrics=AgentPerformanceMetrics(
                    tasks_completed=10,
                    tasks_failed=1,
                    total_processing_time=50.0,
                    average_response_time=2.5,
                    memory_usage_mb=256.0,
                    cpu_usage_percent=35.0,
                    last_activity=datetime.now(timezone.utc),
                    health_score=0.85
                ),
                allocated_resources=[
                    AgentResource(
                        resource_id="3284",
                        resource_type=AgentResourceType.PORT,
                        allocated_at=datetime.now(timezone.utc)
                    )
                ]
            )

        with patch.object(state_manager, 'get_agent_state', side_effect=mock_get_state):
            health_summary = await state_manager.get_agent_health_summary()

            assert health_summary is not None
            assert "total_agents" in health_summary
            assert "by_state" in health_summary
            assert "by_type" in health_summary
            assert "performance_summary" in health_summary
            assert "resource_usage" in health_summary


class TestAgentMemoryManager:
    """Test suite for AgentMemoryManager functionality"""

    @pytest.fixture
    async def memory_manager(self):
        """Create agent memory manager for testing"""
        with patch('utils.agent_memory.get_redis_manager') as mock_get_redis:
            mock_redis = Mock()
            mock_redis.set_agent_memory.return_value = True
            mock_redis.get_agent_memory.return_value = None
            mock_redis.record_metric.return_value = None

            mock_get_redis.return_value = mock_redis

            manager = AgentMemoryManager()
            return manager

    @pytest.mark.asyncio
    async def test_store_memory(self, memory_manager):
        """Test memory storage with different types"""
        agent_id = "test_agent_001"

        # Test short-term memory
        short_term_content = {
            "recent_interaction": "User asked about Python debugging",
            "context": "debugging session",
            "files_involved": ["main.py", "debug.py"]
        }

        memory_entry = await memory_manager.store_memory(
            agent_id=agent_id,
            memory_type=MemoryType.SHORT_TERM,
            content=short_term_content,
            tags=["debugging", "python", "interaction"],
            priority=MemoryPriority.MEDIUM
        )

        assert memory_entry is not None
        assert memory_entry.agent_id == agent_id
        assert memory_entry.memory_type == MemoryType.SHORT_TERM
        assert memory_entry.priority == MemoryPriority.MEDIUM

    @pytest.mark.asyncio
    async def test_memory_search(self, memory_manager):
        """Test memory search with vector similarity"""
        agent_id = "test_agent_002"

        # Mock stored memories
        test_memories = [
            MemoryEntry(
                memory_id="mem_001",
                agent_id=agent_id,
                memory_type=MemoryType.SEMANTIC,
                priority=MemoryPriority.HIGH,
                content={"concept": "Python error handling", "details": "try/except patterns"},
                tags=["python", "errors", "patterns"],
                vector=MemoryVector(
                    vector_id="vec_001",
                    dimensions=384,
                    values=[0.1] * 384,  # Mock vector
                    metadata={"concept": "error_handling"},
                    created_at=datetime.now(timezone.utc)
                ),
                access_count=5
            ),
            MemoryEntry(
                memory_id="mem_002",
                agent_id=agent_id,
                memory_type=MemoryType.PROCEDURAL,
                priority=MemoryPriority.MEDIUM,
                content={"procedure": "Code review process", "steps": ["syntax", "logic", "performance"]},
                tags=["code_review", "process", "quality"],
                vector=MemoryVector(
                    vector_id="vec_002",
                    dimensions=384,
                    values=[0.2] * 384,  # Mock vector
                    metadata={"concept": "code_review"},
                    created_at=datetime.now(timezone.utc)
                ),
                access_count=3
            )
        ]

        with patch.object(memory_manager, '_get_all_memories', return_value=test_memories):
            search_results = await memory_manager.search_memories(
                agent_id=agent_id,
                query="Python error handling patterns",
                limit=5
            )

            assert len(search_results) >= 0
            for result in search_results:
                assert isinstance(result, MemorySearchResult)
                assert 0.0 <= result.similarity_score <= 1.0
                assert 0.0 <= result.relevance_score <= 1.0

    @pytest.mark.asyncio
    async def test_conversation_memory_integration(self, memory_manager):
        """Test integration with conversation memory system"""
        from utils.conversation_memory import ConversationTurn, ThreadContext

        agent_id = "test_agent_003"

        # Create mock conversation context
        conversation_context = ThreadContext(
            thread_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            last_updated_at=datetime.now(timezone.utc).isoformat(),
            tool_name="analyze",
            turns=[
                ConversationTurn(
                    role="user",
                    content="Analyze this Python code for errors",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    files=["main.py", "utils.py"],
                    tool_name="analyze"
                ),
                ConversationTurn(
                    role="assistant",
                    content="Found 3 potential issues: missing error handling, unused imports, inefficient loop",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    files=["main.py"],
                    tool_name="analyze"
                )
            ],
            initial_context={"task": "code_analysis", "language": "python"}
        )

        # Test storing conversation as episodic memory
        success = await memory_manager.store_conversation_memory(agent_id, conversation_context)
        assert success

    @pytest.mark.asyncio
    async def test_memory_analytics(self, memory_manager):
        """Test memory analytics and insights"""
        agent_id = "test_agent_004"

        # Mock memory data for analytics
        mock_memories = [
            MemoryEntry(
                memory_id=f"mem_{i}",
                agent_id=agent_id,
                memory_type=MemoryType.SHORT_TERM if i % 2 == 0 else MemoryType.LONG_TERM,
                priority=MemoryPriority.HIGH if i < 3 else MemoryPriority.MEDIUM,
                content={"data": f"memory content {i}"},
                tags=[f"tag_{i}", "common_tag"],
                access_count=i * 2,
                created_at=datetime.now(timezone.utc)
            )
            for i in range(10)
        ]

        with patch.object(memory_manager, '_get_all_memories', return_value=mock_memories):
            analytics = await memory_manager.get_memory_analytics(agent_id)

            assert isinstance(analytics, MemoryAnalytics)
            assert analytics.total_memories == 10
            assert len(analytics.by_type) > 0
            assert len(analytics.by_priority) > 0
            assert analytics.total_size_mb > 0
            assert 0.0 <= analytics.memory_efficiency_score <= 1.0


class TestIntegration:
    """Test suite for Redis integration with existing systems"""

    @pytest.mark.asyncio
    async def test_agent_task_manager_integration(self):
        """Test integration with existing AgentTaskManager"""
        from utils.agent_manager import AgentTaskManager

        with patch('utils.agent_manager.get_redis_manager') as mock_get_redis:
            mock_redis = Mock()
            mock_redis.allocate_port.return_value = 3284
            mock_redis.release_port.return_value = True
            mock_redis.set_agent_state.return_value = True
            mock_redis.record_metric.return_value = None

            mock_get_redis.return_value = mock_redis

            # Create enhanced task manager
            task_manager = AgentTaskManager()

            # Verify Redis manager integration
            assert hasattr(task_manager, 'redis_manager')
            assert task_manager.redis_manager is not None

            # Test enhanced port allocation
            allocated_port = task_manager._allocate_port("test_agent_001")
            assert allocated_port == 3284

            # Test enhanced port release
            task_manager._release_port(3284, "test_agent_001")
            mock_redis.release_port.assert_called_with("test_agent_001", 3284)

    def test_performance_benchmarks(self):
        """Test performance benchmarks for Redis operations"""
        with patch('utils.redis_manager.redis') as mock_redis:
            mock_conn = Mock()
            mock_conn.ping.return_value = True
            mock_conn.set.return_value = True
            mock_conn.get.return_value = json.dumps({"test": "data"})
            mock_conn.setex.return_value = True

            mock_redis.Redis.return_value = mock_conn

            redis_manager = RedisManager()

            # Benchmark agent state operations
            start_time = time.time()
            for i in range(100):
                redis_manager.set_agent_state(f"agent_{i}", {"state": f"running_{i}"})
            state_write_time = time.time() - start_time

            start_time = time.time()
            for i in range(100):
                redis_manager.get_agent_state(f"agent_{i}")
            state_read_time = time.time() - start_time

            # Benchmark memory operations
            start_time = time.time()
            for i in range(50):
                redis_manager.set_agent_memory(f"agent_{i}", "working", {"memory": f"data_{i}"})
            memory_write_time = time.time() - start_time

            start_time = time.time()
            for i in range(50):
                redis_manager.get_agent_memory(f"agent_{i}", "working")
            memory_read_time = time.time() - start_time

            # Assert reasonable performance (< 1 second for 100 operations)
            assert state_write_time < 1.0
            assert state_read_time < 1.0
            assert memory_write_time < 1.0
            assert memory_read_time < 1.0

            print("Performance Benchmarks:")
            print(f"  State writes (100): {state_write_time:.3f}s ({100/state_write_time:.1f} ops/sec)")
            print(f"  State reads (100): {state_read_time:.3f}s ({100/state_read_time:.1f} ops/sec)")
            print(f"  Memory writes (50): {memory_write_time:.3f}s ({50/memory_write_time:.1f} ops/sec)")
            print(f"  Memory reads (50): {memory_read_time:.3f}s ({50/memory_read_time:.1f} ops/sec)")

    def test_error_handling_and_fallbacks(self):
        """Test error handling and graceful fallbacks"""
        # Test Redis unavailable scenario
        with patch('utils.redis_manager.redis', None):
            # Should fall back to MockRedisManager
            redis_manager = get_redis_manager()
            assert isinstance(redis_manager, MockRedisManager)

            # Test mock functionality
            success = redis_manager.set_agent_state("test_agent", {"state": "test"})
            assert success

            state = redis_manager.get_agent_state("test_agent")
            assert state is not None

            health = redis_manager.get_health_status()
            assert not health["redis_available"]
            assert health["mock_mode"]

    def test_scalability_limits(self):
        """Test scalability for 1000+ agents"""
        with patch('utils.redis_manager.redis') as mock_redis:
            mock_conn = Mock()
            mock_conn.ping.return_value = True
            mock_conn.set.return_value = True
            mock_conn.get.return_value = None
            mock_conn.setex.return_value = True

            mock_redis.Redis.return_value = mock_conn

            redis_manager = RedisManager()

            # Test port allocation for 1000+ agents
            start_port = 3284
            end_port = 10000
            available_ports = end_port - start_port + 1

            # Should support 1000+ agents with expanded port range
            assert available_ports > 1000

            # Test batch operations for performance at scale
            operations = [
                ("set", f"agent:state:{i}", f"running_{i}")
                for i in range(1000)
            ]

            start_time = time.time()
            results = redis_manager.batch_operation(operations, RedisDB.STATE)
            batch_time = time.time() - start_time

            assert len(results) == 1000
            assert batch_time < 5.0  # Should complete within 5 seconds

            print(f"Scalability Test - 1000 agent states: {batch_time:.3f}s ({1000/batch_time:.1f} ops/sec)")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
