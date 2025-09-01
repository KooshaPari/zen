"""
Integration Tests for A2A Protocol

Tests the Agent-to-Agent communication protocol including discovery,
capability advertising, task delegation, and cross-platform interoperability.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from utils.a2a_protocol import A2AMessage, A2AMessageType, A2AProtocolManager, AgentCapability, AgentCard


@pytest.fixture
async def a2a_manager():
    """Create A2A protocol manager for testing."""
    manager = A2AProtocolManager(
        agent_id="test-agent-123",
        registry_endpoints=["http://localhost:8080"]
    )

    # Initialize with test capabilities
    capabilities = [
        AgentCapability(
            name="code_analysis",
            description="Analyze code for issues and improvements",
            category="analysis",
            input_schema={"type": "object", "properties": {"code": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"issues": {"type": "array"}}},
            max_concurrent=3,
            timeout_seconds=120
        ),
        AgentCapability(
            name="test_generation",
            description="Generate unit tests for code",
            category="testing",
            input_schema={"type": "object", "properties": {"source_code": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"test_code": {"type": "string"}}},
            max_concurrent=2,
            timeout_seconds=300
        )
    ]

    await manager.initialize_agent_card(
        name="Test Agent",
        version="1.0.0",
        endpoint_url="http://localhost:3284",
        capabilities=capabilities,
        organization="TestOrg"
    )

    return manager


@pytest.fixture
async def another_a2a_manager():
    """Create a second A2A protocol manager for testing interactions."""
    manager = A2AProtocolManager(
        agent_id="peer-agent-456",
        registry_endpoints=["http://localhost:8080"]
    )

    capabilities = [
        AgentCapability(
            name="code_refactor",
            description="Refactor code for better maintainability",
            category="refactoring",
            input_schema={"type": "object", "properties": {"code": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"refactored_code": {"type": "string"}}},
            max_concurrent=1,
            timeout_seconds=600
        )
    ]

    await manager.initialize_agent_card(
        name="Peer Agent",
        version="2.1.0",
        endpoint_url="http://localhost:3285",
        capabilities=capabilities,
        organization="TestOrg"
    )

    return manager


class TestA2AProtocolBasics:
    """Test basic A2A protocol functionality."""

    @pytest.mark.asyncio
    async def test_agent_card_initialization(self, a2a_manager):
        """Test agent card initialization."""
        assert a2a_manager.my_card is not None
        assert a2a_manager.my_card.agent_id == "test-agent-123"
        assert a2a_manager.my_card.name == "Test Agent"
        assert a2a_manager.my_card.version == "1.0.0"
        assert a2a_manager.my_card.organization == "TestOrg"
        assert len(a2a_manager.my_card.capabilities) == 2

        # Check capabilities
        cap_names = [cap.name for cap in a2a_manager.my_card.capabilities]
        assert "code_analysis" in cap_names
        assert "test_generation" in cap_names

    @pytest.mark.asyncio
    async def test_local_registry_storage(self, a2a_manager):
        """Test that agent card is stored in local registry."""
        assert a2a_manager.agent_id in a2a_manager.local_registry
        stored_card = a2a_manager.local_registry[a2a_manager.agent_id]
        assert stored_card.name == "Test Agent"
        assert stored_card.organization == "TestOrg"

    @pytest.mark.asyncio
    async def test_capability_filtering(self, a2a_manager):
        """Test capability filtering functionality."""
        agent_card = a2a_manager.my_card

        # Test matches
        assert a2a_manager._matches_discovery_filter(agent_card, "code", None)
        assert a2a_manager._matches_discovery_filter(agent_card, "analysis", None)
        assert a2a_manager._matches_discovery_filter(agent_card, "test", None)
        assert a2a_manager._matches_discovery_filter(agent_card, None, "TestOrg")

        # Test non-matches
        assert not a2a_manager._matches_discovery_filter(agent_card, "nonexistent", None)
        assert not a2a_manager._matches_discovery_filter(agent_card, None, "WrongOrg")


class TestA2ADiscovery:
    """Test agent discovery functionality."""

    @pytest.mark.asyncio
    async def test_local_discovery(self, a2a_manager, another_a2a_manager):
        """Test discovering agents from local registry."""
        # Add peer agent to local registry
        a2a_manager.local_registry[another_a2a_manager.agent_id] = another_a2a_manager.my_card

        # Discover all agents
        agents = await a2a_manager.discover_agents()

        assert len(agents) >= 2
        agent_ids = [agent.agent_id for agent in agents]
        assert "test-agent-123" in agent_ids
        assert "peer-agent-456" in agent_ids

    @pytest.mark.asyncio
    async def test_filtered_discovery(self, a2a_manager, another_a2a_manager):
        """Test filtered agent discovery."""
        # Add peer agent to local registry
        a2a_manager.local_registry[another_a2a_manager.agent_id] = another_a2a_manager.my_card

        # Test capability filtering
        agents = await a2a_manager.discover_agents(capability_filter="refactor")
        agent_ids = [agent.agent_id for agent in agents]
        assert "peer-agent-456" in agent_ids
        assert "test-agent-123" not in agent_ids

        # Test organization filtering
        agents = await a2a_manager.discover_agents(organization_filter="TestOrg")
        assert len(agents) >= 2  # Both agents are in TestOrg

    @pytest.mark.asyncio
    async def test_max_results_limit(self, a2a_manager):
        """Test max results limiting in discovery."""
        # Add multiple agents to registry
        for i in range(10):
            fake_agent = AgentCard(
                agent_id=f"fake-agent-{i}",
                name=f"Fake Agent {i}",
                version="1.0.0",
                endpoint_url=f"http://localhost:328{4+i}",
                capabilities=[],
                last_seen=datetime.now(timezone.utc)
            )
            a2a_manager.local_registry[fake_agent.agent_id] = fake_agent

        # Test limiting
        agents = await a2a_manager.discover_agents(max_results=5)
        assert len(agents) <= 5


class TestA2AMessaging:
    """Test A2A message handling."""

    @pytest.mark.asyncio
    async def test_discover_message_handling(self, a2a_manager):
        """Test handling of discovery messages."""
        # Create discovery message
        message = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id="requester-agent",
            message_type=A2AMessageType.DISCOVER,
            timestamp=datetime.now(timezone.utc),
            payload={
                "capability_filter": "analysis",
                "max_results": 10
            }
        )

        # Handle message
        response = await a2a_manager._handle_discover(message)

        assert response is not None
        assert response.message_type == A2AMessageType.CAPABILITY_RESPONSE
        assert response.receiver_id == "requester-agent"

        # Check payload contains matching agents
        agents = response.payload.get("agents", [])
        assert len(agents) >= 1  # Should find test agent with analysis capability

    @pytest.mark.asyncio
    async def test_advertise_message_handling(self, a2a_manager):
        """Test handling of advertisement messages."""
        # Create new agent card
        new_agent_card = AgentCard(
            agent_id="advertised-agent",
            name="Advertised Agent",
            version="1.0.0",
            endpoint_url="http://localhost:3290",
            capabilities=[],
            last_seen=datetime.now(timezone.utc)
        )

        # Create advertisement message
        message = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id="advertised-agent",
            message_type=A2AMessageType.ADVERTISE,
            timestamp=datetime.now(timezone.utc),
            payload={"agent_card": new_agent_card.model_dump()}
        )

        # Handle message
        await a2a_manager._handle_advertise(message)

        # Check agent was registered
        assert "advertised-agent" in a2a_manager.local_registry
        registered_agent = a2a_manager.local_registry["advertised-agent"]
        assert registered_agent.name == "Advertised Agent"

    @pytest.mark.asyncio
    async def test_capability_request_handling(self, a2a_manager):
        """Test handling of capability requests."""
        message = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id="requester-agent",
            message_type=A2AMessageType.CAPABILITY_REQUEST,
            timestamp=datetime.now(timezone.utc),
            payload={}
        )

        response = await a2a_manager._handle_capability_request(message)

        assert response is not None
        assert response.message_type == A2AMessageType.CAPABILITY_RESPONSE
        assert response.receiver_id == "requester-agent"

        # Check our agent card is in response
        agent_card_data = response.payload.get("agent_card")
        assert agent_card_data is not None
        assert agent_card_data["agent_id"] == "test-agent-123"

    @pytest.mark.asyncio
    async def test_task_request_handling(self, a2a_manager):
        """Test handling of task requests."""
        message = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id="requester-agent",
            message_type=A2AMessageType.TASK_REQUEST,
            timestamp=datetime.now(timezone.utc),
            correlation_id="test-correlation-123",
            payload={
                "capability_name": "code_analysis",
                "task_data": {"code": "def hello(): pass"}
            }
        )

        response = await a2a_manager._handle_task_request(message)

        assert response is not None
        assert response.message_type == A2AMessageType.TASK_RESPONSE
        assert response.receiver_id == "requester-agent"
        assert response.correlation_id == "test-correlation-123"

        # Check response payload
        task_id = response.payload.get("task_id")
        status = response.payload.get("status")
        assert task_id is not None
        assert status == "accepted"

    @pytest.mark.asyncio
    async def test_heartbeat_handling(self, a2a_manager):
        """Test heartbeat message handling."""
        # Add agent to registry first
        test_agent = AgentCard(
            agent_id="heartbeat-agent",
            name="Heartbeat Agent",
            version="1.0.0",
            endpoint_url="http://localhost:3291",
            capabilities=[],
            last_seen=datetime.now(timezone.utc)
        )
        a2a_manager.local_registry["heartbeat-agent"] = test_agent

        message = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id="heartbeat-agent",
            message_type=A2AMessageType.HEARTBEAT,
            timestamp=datetime.now(timezone.utc),
            payload={}
        )

        response = await a2a_manager._handle_heartbeat(message)

        assert response is not None
        assert response.message_type == A2AMessageType.HEARTBEAT
        assert response.receiver_id == "heartbeat-agent"
        assert response.payload.get("status") == "alive"

        # Check last seen was updated
        updated_agent = a2a_manager.local_registry["heartbeat-agent"]
        assert updated_agent.last_seen > test_agent.last_seen

    @pytest.mark.asyncio
    async def test_message_ttl_expiration(self, a2a_manager):
        """Test that expired messages are rejected."""
        # Create expired message
        expired_time = datetime.now(timezone.utc).replace(year=2020)  # Old timestamp
        message_data = {
            "message_id": str(uuid.uuid4()),
            "sender_id": "test-sender",
            "message_type": "discover",
            "timestamp": expired_time.isoformat(),
            "payload": {},
            "ttl_seconds": 60
        }

        response = await a2a_manager.handle_incoming_message(message_data)
        assert response is None  # Should be rejected due to expiration


class TestA2AIntegration:
    """Test A2A protocol integration scenarios."""

    @pytest.mark.asyncio
    async def test_agent_to_agent_communication(self, a2a_manager, another_a2a_manager):
        """Test full agent-to-agent communication flow."""
        # Setup: Register peer agent in first manager's registry
        a2a_manager.local_registry[another_a2a_manager.agent_id] = another_a2a_manager.my_card

        with patch.object(a2a_manager, '_send_message_to_agent') as mock_send:
            # Mock successful task response
            mock_response = A2AMessage(
                message_id=str(uuid.uuid4()),
                sender_id=another_a2a_manager.agent_id,
                receiver_id=a2a_manager.agent_id,
                message_type=A2AMessageType.TASK_RESPONSE,
                timestamp=datetime.now(timezone.utc),
                payload={
                    "task_id": "response-task-123",
                    "status": "completed",
                    "result": {"refactored_code": "def hello_world(): pass"}
                }
            )
            mock_send.return_value = mock_response

            # Send task request
            result = await a2a_manager.send_task_request(
                target_agent_id=another_a2a_manager.agent_id,
                capability_name="code_refactor",
                task_data={"code": "def hello(): pass"}
            )

            # Verify communication
            assert mock_send.called
            assert result["status"] == "completed"
            assert "refactored_code" in result["result"]

    @pytest.mark.asyncio
    async def test_redis_integration(self, a2a_manager):
        """Test Redis integration for agent registry."""
        # Skip if Redis is not available
        if not a2a_manager.redis_client:
            pytest.skip("Redis not available for testing")

        # Advertise capabilities (should store in Redis)
        await a2a_manager.advertise_capabilities()

        # Verify stored in Redis
        key = f"a2a_agent:{a2a_manager.agent_id}"
        stored_data = a2a_manager.redis_client.get(key)
        assert stored_data is not None

        # Verify can be loaded back
        loaded_card = AgentCard.model_validate_json(stored_data)
        assert loaded_card.agent_id == a2a_manager.agent_id
        assert loaded_card.name == "Test Agent"

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, a2a_manager):
        """Test concurrent A2A operations."""
        # Create multiple concurrent discovery requests
        tasks = []
        for _i in range(10):
            task = asyncio.create_task(
                a2a_manager.discover_agents(capability_filter="analysis")
            )
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_error_handling(self, a2a_manager):
        """Test error handling in A2A operations."""
        # Test task request to non-existent agent
        with pytest.raises(ValueError, match="not found"):
            await a2a_manager.send_task_request(
                target_agent_id="nonexistent-agent",
                capability_name="some_capability",
                task_data={}
            )

        # Test invalid message handling
        invalid_message_data = {
            "invalid": "message",
            "structure": True
        }

        result = await a2a_manager.handle_incoming_message(invalid_message_data)
        assert result is None  # Should handle gracefully


@pytest.mark.integration
class TestA2ARemoteIntegration:
    """Integration tests requiring external services."""

    @pytest.mark.skipif(True, reason="Requires external registry service")
    @pytest.mark.asyncio
    async def test_remote_registry_integration(self, a2a_manager):
        """Test integration with remote registry service."""
        # This would test actual HTTP communication with registry
        # Skip for now as it requires external service
        pass

    @pytest.mark.asyncio
    async def test_performance_with_large_registry(self, a2a_manager):
        """Test performance with large numbers of registered agents."""
        # Create many fake agents
        for i in range(1000):
            fake_agent = AgentCard(
                agent_id=f"perf-agent-{i}",
                name=f"Performance Agent {i}",
                version="1.0.0",
                endpoint_url=f"http://localhost:{3300+i}",
                capabilities=[
                    AgentCapability(
                        name=f"capability_{i}",
                        description=f"Test capability {i}",
                        category="testing",
                        input_schema={},
                        output_schema={}
                    )
                ],
                last_seen=datetime.now(timezone.utc)
            )
            a2a_manager.local_registry[fake_agent.agent_id] = fake_agent

        # Time discovery operation
        import time
        start_time = time.time()

        agents = await a2a_manager.discover_agents(capability_filter="capability")

        end_time = time.time()
        discovery_time = end_time - start_time

        # Should complete within reasonable time
        assert discovery_time < 5.0  # 5 seconds max
        assert len(agents) <= 50  # Respects max_results default
