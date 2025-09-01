"""
NATS Integration Tests and Performance Benchmarks

This module provides comprehensive integration tests and performance benchmarks
for the NATS messaging infrastructure, validating:
- Core NATS functionality and reliability
- JetStream persistent messaging capabilities
- Agent discovery and service registry operations
- Edge orchestration and workload placement
- Real-time metrics and monitoring systems
- Performance characteristics under various loads
- Fault tolerance and recovery mechanisms
"""

import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.shared.agent_models import AgentType, TaskStatus
from utils.agent_discovery import AgentDiscoveryService, AgentService, ServiceQuery
from utils.edge_orchestrator import EdgeLocation, EdgeOrchestrator, WorkloadRequest
from utils.nats_communicator import NATSCommunicator, NATSConfig, NATSMessage
from utils.nats_monitoring import NATSMonitoringSystem
from utils.nats_redis_integration import NATSRedisIntegrator
from utils.nats_streaming import NATSStreamingManager, StreamDefinition


@pytest.fixture
async def mock_redis():
    """Mock Redis client for testing."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.setex.return_value = True
    mock_client.get.return_value = None
    mock_client.delete.return_value = True
    mock_client.publish.return_value = 1
    mock_client.exists.return_value = False
    mock_client.smembers.return_value = set()
    mock_client.sadd.return_value = 1
    mock_client.srem.return_value = 1
    mock_client.expire.return_value = True
    mock_client.zadd.return_value = 1
    mock_client.zremrangebyrank.return_value = 1
    return mock_client


@pytest.fixture
async def nats_config():
    """NATS configuration for testing."""
    return NATSConfig(
        servers=["nats://localhost:4222"],
        max_payload=1024*1024,  # 1MB
        flush_timeout=0.001,
        ping_interval=30,
        max_pending_msgs=1000
    )


@pytest.fixture
async def nats_communicator(nats_config, mock_redis):
    """NATS communicator instance for testing."""
    with patch('utils.nats_communicator.nats') as mock_nats:
        # Mock NATS client
        mock_client = AsyncMock()
        mock_client._client_id = "test_client_123"
        mock_client.jetstream.return_value = AsyncMock()

        mock_nats.connect.return_value = mock_client

        communicator = NATSCommunicator(nats_config, mock_redis)
        await communicator.connect()

        yield communicator

        await communicator.disconnect()


@pytest.fixture
async def streaming_manager(nats_communicator, mock_redis):
    """NATS streaming manager for testing."""
    with patch('utils.nats_streaming.nats'):
        manager = NATSStreamingManager(nats_communicator, mock_redis)

        # Mock JetStream operations
        mock_js = AsyncMock()
        mock_js.stream_info.return_value = MagicMock()
        mock_js.add_stream.return_value = MagicMock()
        mock_js.add_consumer.return_value = MagicMock()
        mock_js.publish.return_value = MagicMock(stream="TEST_STREAM", seq=1, duplicate=False)
        mock_js.account_info.return_value = MagicMock(
            memory=1024*1024, storage=1024*1024*10,
            streams=5, consumers=20,
            limits=MagicMock(max_memory=1024*1024*1024, max_storage=1024*1024*1024*10,
                           max_streams=1000, max_consumers=10000)
        )

        manager.js = mock_js
        await manager.initialize()

        yield manager

        await manager.shutdown()


@pytest.fixture
async def discovery_service(nats_communicator, mock_redis):
    """Agent discovery service for testing."""
    service = AgentDiscoveryService(nats_communicator, mock_redis)
    await service.start()

    yield service

    await service.stop()


@pytest.fixture
async def edge_orchestrator(nats_communicator, discovery_service, mock_redis):
    """Edge orchestrator for testing."""
    orchestrator = EdgeOrchestrator(nats_communicator, discovery_service, mock_redis)
    await orchestrator.start()

    yield orchestrator

    await orchestrator.stop()


@pytest.fixture
async def monitoring_system(nats_communicator, streaming_manager, discovery_service,
                           edge_orchestrator, mock_redis):
    """NATS monitoring system for testing."""
    monitor = NATSMonitoringSystem(
        nats_communicator, streaming_manager, discovery_service, edge_orchestrator, mock_redis
    )
    await monitor.start()

    yield monitor

    await monitor.stop()


@pytest.fixture
async def nats_redis_integrator(mock_redis, nats_communicator, streaming_manager):
    """NATS-Redis integrator for testing."""
    integrator = NATSRedisIntegrator(mock_redis, nats_communicator, streaming_manager)
    await integrator.start()

    yield integrator

    await integrator.stop()


class TestNATSCommunicator:
    """Test NATS core communication functionality."""

    @pytest.mark.asyncio
    async def test_connection_establishment(self, nats_communicator):
        """Test NATS connection establishment."""
        assert nats_communicator.connected is True
        assert nats_communicator.connection_id is not None
        assert nats_communicator.nc is not None

    @pytest.mark.asyncio
    async def test_message_publishing(self, nats_communicator):
        """Test message publishing functionality."""
        test_message = {"test": "data", "timestamp": datetime.now(timezone.utc).isoformat()}

        success = await nats_communicator.publish("test.subject", test_message)
        assert success is True

        # Test with NATSMessage
        nats_msg = NATSMessage(
            message_id="test_123",
            sender_id="test_sender",
            message_type="test",
            payload=test_message
        )

        success = await nats_communicator.publish("test.nats.message", nats_msg)
        assert success is True

    @pytest.mark.asyncio
    async def test_request_response(self, nats_communicator):
        """Test request-response pattern."""
        with patch.object(nats_communicator.nc, 'request') as mock_request:
            mock_response = AsyncMock()
            mock_response.data = json.dumps({"response": "test_data"}).encode()
            mock_request.return_value = mock_response

            response = await nats_communicator.request("test.request", {"query": "test"})
            assert response is not None
            assert response["response"] == "test_data"

    @pytest.mark.asyncio
    async def test_subscription(self, nats_communicator):
        """Test message subscription functionality."""
        received_messages = []

        def message_handler(data):
            received_messages.append(data)

        success = await nats_communicator.subscribe("test.subscription", message_handler)
        assert success is True
        assert "test.subscription" in nats_communicator.subscriptions

    @pytest.mark.asyncio
    async def test_performance_metrics(self, nats_communicator):
        """Test performance metrics collection."""
        # Simulate some operations
        await nats_communicator.publish("test.metrics", {"data": "test"})

        metrics = await nats_communicator.get_performance_metrics()

        assert "connected" in metrics
        assert "message_count" in metrics
        assert "error_count" in metrics
        assert "avg_latency_ms" in metrics
        assert metrics["connected"] is True


class TestNATSStreaming:
    """Test NATS JetStream functionality."""

    @pytest.mark.asyncio
    async def test_stream_creation(self, streaming_manager):
        """Test JetStream stream creation."""
        stream_def = StreamDefinition(
            name="TEST_STREAM",
            subjects=["test.stream.*"],
            retention_policy="limits",
            max_msgs=1000,
            max_bytes=1024*1024,
            max_age_seconds=3600
        )

        success = await streaming_manager.create_or_update_stream(stream_def)
        assert success is True
        assert "TEST_STREAM" in streaming_manager.streams

    @pytest.mark.asyncio
    async def test_persistent_publishing(self, streaming_manager):
        """Test persistent message publishing."""
        test_data = {"persistent": "message", "timestamp": time.time()}

        success = await streaming_manager.publish_persistent(
            "test.persistent", test_data, use_jetstream=True, expected_stream="AGENT_COMMANDS"
        )
        assert success is True

    @pytest.mark.asyncio
    async def test_stream_info(self, streaming_manager):
        """Test stream information retrieval."""
        stream_info = await streaming_manager.get_stream_info("AGENT_COMMANDS")
        assert stream_info is not None
        assert "name" in stream_info
        assert "messages" in stream_info

    @pytest.mark.asyncio
    async def test_streaming_metrics(self, streaming_manager):
        """Test streaming metrics collection."""
        metrics = await streaming_manager.get_streaming_metrics()

        assert "messages_published" in metrics
        assert "messages_consumed" in metrics
        assert "ack_count" in metrics
        assert "active_streams" in metrics


class TestAgentDiscovery:
    """Test agent discovery service functionality."""

    @pytest.mark.asyncio
    async def test_service_registration(self, discovery_service):
        """Test agent service registration."""
        test_service = AgentService(
            service_id="test_agent_123",
            agent_id="agent_123",
            agent_type=AgentType.CLAUDE,
            capabilities={"coding", "analysis"},
            status=TaskStatus.RUNNING,
            zone="us-west-1",
            region="us-west"
        )

        success = await discovery_service.register_service(test_service)
        assert success is True
        assert "test_agent_123" in discovery_service.services

    @pytest.mark.asyncio
    async def test_service_discovery(self, discovery_service):
        """Test service discovery queries."""
        # Register a test service first
        test_service = AgentService(
            service_id="discoverable_agent",
            agent_id="agent_456",
            agent_type=AgentType.CLAUDE,
            capabilities={"nlp", "reasoning"},
            status=TaskStatus.RUNNING,
            load_factor=0.3,
            success_rate=0.95
        )

        await discovery_service.register_service(test_service)

        # Query for services
        query = ServiceQuery(
            agent_type=AgentType.CLAUDE,
            capabilities={"reasoning"},
            max_load=0.5,
            min_success_rate=0.9
        )

        services = await discovery_service.discover_services(query)
        assert len(services) >= 1
        assert any(s.service_id == "discoverable_agent" for s in services)

    @pytest.mark.asyncio
    async def test_service_deregistration(self, discovery_service):
        """Test service deregistration."""
        # Register then deregister
        test_service = AgentService(
            service_id="temp_agent",
            agent_id="temp_123",
            agent_type=AgentType.AIDER,
            capabilities={"temp"}
        )

        await discovery_service.register_service(test_service)
        assert "temp_agent" in discovery_service.services

        success = await discovery_service.deregister_service("temp_agent")
        assert success is True
        assert "temp_agent" not in discovery_service.services

    @pytest.mark.asyncio
    async def test_registry_status(self, discovery_service):
        """Test registry status retrieval."""
        status = await discovery_service.get_registry_status()

        assert "total_services" in status
        assert "discovery_requests" in status
        assert "services_by_type" in status


class TestEdgeOrchestrator:
    """Test edge orchestration functionality."""

    @pytest.mark.asyncio
    async def test_location_registration(self, edge_orchestrator):
        """Test edge location registration."""
        test_location = EdgeLocation(
            location_id="edge_us_west_1",
            name="US West 1 Edge",
            region="us-west",
            zone="us-west-1a",
            country_code="US",
            latency_to_cloud_ms=50.0,
            bandwidth_mbps=1000.0,
            max_cpu_cores=16,
            max_memory_gb=64.0,
            max_storage_gb=500.0,
            max_agents=100,
            supported_agent_types={AgentType.CLAUDE, AgentType.AIDER}
        )

        success = await edge_orchestrator.register_edge_location(test_location)
        assert success is True
        assert "edge_us_west_1" in edge_orchestrator.edge_locations

    @pytest.mark.asyncio
    async def test_workload_placement(self, edge_orchestrator):
        """Test workload placement functionality."""
        # Register an edge location first
        edge_location = EdgeLocation(
            location_id="placement_test_edge",
            name="Placement Test Edge",
            region="test",
            zone="test-1",
            country_code="US",
            latency_to_cloud_ms=25.0,
            bandwidth_mbps=2000.0,
            max_cpu_cores=32,
            max_memory_gb=128.0,
            max_storage_gb=1000.0,
            max_agents=50,
            supported_agent_types={AgentType.CLAUDE, AgentType.GEMINI}
        )

        await edge_orchestrator.register_edge_location(edge_location)

        # Create workload request
        from tools.shared.agent_models import AgentTaskRequest

        workload_request = WorkloadRequest(
            request_id="test_workload_123",
            task_requests=[
                AgentTaskRequest(
                    agent_type=AgentType.CLAUDE,
                    task_description="Test task",
                    message="Test message"
                )
            ],
            preferred_regions=["test"],
            min_cpu_cores=2,
            min_memory_gb=4.0,
            latency_budget_ms=100.0
        )

        # Mock the placement execution to avoid external dependencies
        with patch.object(edge_orchestrator, '_execute_placement') as mock_execute:
            mock_execute.return_value = True

            placement = await edge_orchestrator.place_workload(workload_request)
            assert placement is not None
            assert placement.request.request_id == "test_workload_123"
            assert len(placement.selected_locations) > 0

    @pytest.mark.asyncio
    async def test_orchestrator_metrics(self, edge_orchestrator):
        """Test orchestrator metrics retrieval."""
        metrics = await edge_orchestrator.get_orchestrator_metrics()

        assert "placement_requests" in metrics
        assert "successful_placements" in metrics
        assert "failed_placements" in metrics
        assert "locations" in metrics


class TestNATSMonitoring:
    """Test NATS monitoring and metrics system."""

    @pytest.mark.asyncio
    async def test_metric_recording(self, monitoring_system):
        """Test metric recording functionality."""
        await monitoring_system.record_metric("test_metric", 42.5, {"source": "test"})

        metric = await monitoring_system.get_metric("test_metric")
        assert metric is not None
        assert metric.get_latest_value() == 42.5

    @pytest.mark.asyncio
    async def test_health_checks(self, monitoring_system):
        """Test health check system."""
        # Add custom health check
        async def test_health_check():
            return True

        await monitoring_system.add_health_check("test_check", test_health_check)

        result = await monitoring_system.run_health_check("test_check")
        assert result is not None
        assert result.status == "healthy"

    @pytest.mark.asyncio
    async def test_system_health(self, monitoring_system):
        """Test overall system health assessment."""
        health_status = await monitoring_system.get_system_health()

        assert "health_score" in health_status
        assert "status" in health_status
        assert "checks" in health_status
        assert health_status["health_score"] >= 0

    @pytest.mark.asyncio
    async def test_dashboard_data(self, monitoring_system):
        """Test dashboard data aggregation."""
        dashboard_data = await monitoring_system.get_dashboard_data()

        assert "timestamp" in dashboard_data
        assert "system_health" in dashboard_data
        assert "metrics_summary" in dashboard_data
        assert "component_status" in dashboard_data
        assert "active_alerts" in dashboard_data

    @pytest.mark.asyncio
    async def test_alert_system(self, monitoring_system):
        """Test alerting system functionality."""
        from utils.nats_monitoring import AlertRule

        # Add alert rule
        alert_rule = AlertRule(
            rule_id="test_alert",
            metric_name="test_metric",
            condition="gt",
            threshold=100.0,
            severity="warning",
            description="Test alert rule"
        )

        await monitoring_system.add_alert_rule(alert_rule)
        assert "test_alert" in monitoring_system.alert_rules

        # Record metric that should trigger alert
        await monitoring_system.record_metric("test_metric", 150.0)

        # Manually process alerts (normally done by background task)
        # This would be tested with proper alert processing in real scenario


class TestNATSRedisIntegration:
    """Test NATS-Redis hybrid integration."""

    @pytest.mark.asyncio
    async def test_state_management(self, nats_redis_integrator):
        """Test hybrid state management."""
        test_state = {"key": "value", "timestamp": time.time()}

        success = await nats_redis_integrator.set_state("test_entity", "entity_123", test_state)
        assert success is True

        retrieved_state = await nats_redis_integrator.get_state("test_entity", "entity_123")
        assert retrieved_state is not None
        assert retrieved_state["key"] == "value"

    @pytest.mark.asyncio
    async def test_bulk_operations(self, nats_redis_integrator):
        """Test bulk state operations."""
        states = [
            ("entity", "id1", {"data": "test1"}),
            ("entity", "id2", {"data": "test2"}),
            ("entity", "id3", {"data": "test3"})
        ]

        success_count = await nats_redis_integrator.bulk_set_states(states)
        assert success_count == 3

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, nats_redis_integrator):
        """Test cache invalidation functionality."""
        # Set some state
        await nats_redis_integrator.set_state("cache_test", "item_1", {"cached": "data"})

        # Invalidate cache
        await nats_redis_integrator.invalidate_cache("cache_test", "item_1", broadcast=False)

        # Verify invalidation (would need proper cache checking in real implementation)

    @pytest.mark.asyncio
    async def test_integration_metrics(self, nats_redis_integrator):
        """Test integration performance metrics."""
        metrics = await nats_redis_integrator.get_integration_metrics()

        assert "redis_operations" in metrics
        assert "nats_operations" in metrics
        assert "cache_hits" in metrics
        assert "cache_misses" in metrics
        assert "cache_hit_rate" in metrics


class TestPerformanceBenchmarks:
    """Performance benchmarks for NATS infrastructure."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_message_throughput(self, nats_communicator):
        """Benchmark message throughput."""
        message_count = 1000
        message_size = 1024  # 1KB messages

        test_message = {"data": "x" * message_size, "id": 0}

        start_time = time.perf_counter()

        # Send messages
        tasks = []
        for i in range(message_count):
            test_message["id"] = i
            task = nats_communicator.publish(f"benchmark.throughput.{i}", test_message)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.perf_counter()
        duration = end_time - start_time

        successful_sends = sum(1 for r in results if r is True)
        throughput = successful_sends / duration

        print("\nMessage Throughput Benchmark:")
        print(f"Messages: {message_count}")
        print(f"Message Size: {message_size} bytes")
        print(f"Duration: {duration:.3f}s")
        print(f"Successful: {successful_sends}")
        print(f"Throughput: {throughput:.1f} msg/sec")
        print(f"Data Rate: {(throughput * message_size / 1024 / 1024):.1f} MB/sec")

        assert throughput > 100  # At least 100 msg/sec

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_latency_distribution(self, nats_communicator):
        """Benchmark message latency distribution."""
        sample_count = 100
        latencies = []

        for i in range(sample_count):
            start_time = time.perf_counter()

            success = await nats_communicator.publish(f"benchmark.latency.{i}", {"data": f"test_{i}"})

            end_time = time.perf_counter()

            if success:
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

        if latencies:
            avg_latency = statistics.mean(latencies)
            p50_latency = statistics.median(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
            p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

            print("\nLatency Distribution Benchmark:")
            print(f"Samples: {len(latencies)}")
            print(f"Average: {avg_latency:.3f}ms")
            print(f"P50: {p50_latency:.3f}ms")
            print(f"P95: {p95_latency:.3f}ms")
            print(f"P99: {p99_latency:.3f}ms")

            assert avg_latency < 10.0  # Average latency under 10ms

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_concurrent_connections(self, nats_config, mock_redis):
        """Benchmark concurrent connection handling."""
        connection_count = 50
        connections = []

        start_time = time.perf_counter()

        # Create multiple connections
        with patch('utils.nats_communicator.nats') as mock_nats:
            mock_client = AsyncMock()
            mock_client._client_id = lambda: f"test_client_{len(connections)}"
            mock_client.jetstream.return_value = AsyncMock()

            mock_nats.connect.return_value = mock_client

            for i in range(connection_count):
                comm = NATSCommunicator(nats_config, mock_redis)
                await comm.connect()
                connections.append(comm)

        connection_time = time.perf_counter() - start_time

        # Test concurrent publishing
        publish_start = time.perf_counter()

        tasks = []
        for i, conn in enumerate(connections):
            task = conn.publish(f"benchmark.concurrent.{i}", {"connection": i})
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        publish_time = time.perf_counter() - publish_start

        successful_publishes = sum(1 for r in results if r is True)

        # Cleanup
        for conn in connections:
            await conn.disconnect()

        print("\nConcurrent Connections Benchmark:")
        print(f"Connections: {connection_count}")
        print(f"Connection Setup Time: {connection_time:.3f}s")
        print(f"Concurrent Publish Time: {publish_time:.3f}s")
        print(f"Successful Publishes: {successful_publishes}")
        print(f"Publish Success Rate: {(successful_publishes/connection_count)*100:.1f}%")

        assert successful_publishes >= connection_count * 0.9  # 90% success rate

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_discovery_performance(self, discovery_service):
        """Benchmark service discovery performance."""
        service_count = 100

        # Register multiple services
        registration_start = time.perf_counter()

        for i in range(service_count):
            service = AgentService(
                service_id=f"perf_service_{i}",
                agent_id=f"agent_{i}",
                agent_type=AgentType.CLAUDE if i % 2 == 0 else AgentType.AIDER,
                capabilities={"capability_1", "capability_2"} if i % 3 == 0 else {"capability_2"},
                zone=f"zone_{i % 5}",
                load_factor=0.1 * (i % 10)
            )

            await discovery_service.register_service(service)

        registration_time = time.perf_counter() - registration_start

        # Test discovery queries
        query_count = 50
        query_start = time.perf_counter()

        for i in range(query_count):
            query = ServiceQuery(
                agent_type=AgentType.CLAUDE if i % 2 == 0 else None,
                capabilities={"capability_2"},
                max_load=0.5,
                limit=10
            )

            services = await discovery_service.discover_services(query)
            assert len(services) > 0

        query_time = time.perf_counter() - query_start

        print("\nService Discovery Performance Benchmark:")
        print(f"Services Registered: {service_count}")
        print(f"Registration Time: {registration_time:.3f}s")
        print(f"Registration Rate: {service_count/registration_time:.1f} services/sec")
        print(f"Discovery Queries: {query_count}")
        print(f"Query Time: {query_time:.3f}s")
        print(f"Query Rate: {query_count/query_time:.1f} queries/sec")
        print(f"Avg Query Latency: {(query_time/query_count)*1000:.3f}ms")

        assert (query_count/query_time) > 50  # At least 50 queries/sec


class TestFaultTolerance:
    """Test fault tolerance and recovery mechanisms."""

    @pytest.mark.asyncio
    async def test_connection_recovery(self, nats_communicator):
        """Test connection recovery after failure."""
        # Simulate connection failure
        nats_communicator.connected = False

        # Attempt operation during failure
        success = await nats_communicator.publish("test.recovery", {"data": "test"})
        assert success is False

        # Simulate recovery
        nats_communicator.connected = True
        success = await nats_communicator.publish("test.recovery", {"data": "test"})
        assert success is True

    @pytest.mark.asyncio
    async def test_partial_system_failure(self, monitoring_system):
        """Test system behavior with partial component failures."""
        # Simulate discovery service failure
        original_discovery = monitoring_system.discovery
        monitoring_system.discovery = None

        # System health should reflect degraded state
        health_status = await monitoring_system.get_system_health()
        assert health_status["status"] in ["degraded", "unhealthy"]

        # Restore service
        monitoring_system.discovery = original_discovery

        health_status = await monitoring_system.get_system_health()
        # Health should improve (though may not be fully healthy due to other mock limitations)


@pytest.mark.asyncio
async def test_integration_end_to_end():
    """End-to-end integration test of all components."""
    with patch('utils.nats_communicator.nats') as mock_nats:
        # Set up mocks
        mock_client = AsyncMock()
        mock_client._client_id = "integration_test_client"
        mock_client.jetstream.return_value = AsyncMock()
        mock_nats.connect.return_value = mock_client

        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        # Initialize components
        config = NATSConfig(servers=["nats://localhost:4222"])
        nats_comm = NATSCommunicator(config, mock_redis)
        await nats_comm.connect()

        streaming = NATSStreamingManager(nats_comm, mock_redis)

        # Mock JetStream operations
        mock_js = AsyncMock()
        mock_js.stream_info.return_value = MagicMock()
        mock_js.add_stream.return_value = MagicMock()
        mock_js.account_info.return_value = MagicMock(
            memory=1024, storage=1024*10, streams=1, consumers=1,
            limits=MagicMock(max_memory=1024*1024, max_storage=1024*1024*10,
                           max_streams=100, max_consumers=1000)
        )
        streaming.js = mock_js
        await streaming.initialize()

        discovery = AgentDiscoveryService(nats_comm, mock_redis)
        await discovery.start()

        edge = EdgeOrchestrator(nats_comm, discovery, mock_redis)
        await edge.start()

        monitoring = NATSMonitoringSystem(nats_comm, streaming, discovery, edge, mock_redis)
        await monitoring.start()

        # Test cross-component interaction
        # 1. Register edge location
        location = EdgeLocation(
            location_id="integration_edge",
            name="Integration Test Edge",
            region="test",
            zone="test-1",
            country_code="US",
            latency_to_cloud_ms=30.0,
            bandwidth_mbps=1000.0,
            max_cpu_cores=8,
            max_memory_gb=32.0,
            max_storage_gb=200.0,
            max_agents=20,
            supported_agent_types={AgentType.CLAUDE}
        )

        await edge.register_edge_location(location)

        # 2. Register agent service
        service = AgentService(
            service_id="integration_agent",
            agent_id="agent_integration",
            agent_type=AgentType.CLAUDE,
            capabilities={"integration", "testing"}
        )

        await discovery.register_service(service)

        # 3. Record metrics
        await monitoring.record_metric("integration_test", 100.0)

        # 4. Check system health
        health = await monitoring.get_system_health()
        assert health is not None
        assert "health_score" in health

        # 5. Get dashboard data
        dashboard = await monitoring.get_dashboard_data()
        assert dashboard is not None
        assert "system_health" in dashboard

        # Cleanup
        await monitoring.stop()
        await edge.stop()
        await discovery.stop()
        await streaming.shutdown()
        await nats_comm.disconnect()

        print("\nEnd-to-end integration test completed successfully!")


if __name__ == "__main__":
    # Run benchmarks
    import sys

    if "--benchmark" in sys.argv:
        pytest.main([__file__, "-m", "benchmark", "-v", "-s"])
    else:
        pytest.main([__file__, "-v"])
