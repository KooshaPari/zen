"""
Pytest configuration for integration tests.

Provides fixtures and configuration for comprehensive integration testing
of the enterprise agent orchestration system.
"""

import asyncio
import os
from collections.abc import Generator

import pytest


# Configure pytest for async tests
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def setup_test_environment():
    """Setup test environment for integration tests."""

    # Set test environment variables
    os.environ["ZEN_TEST_MODE"] = "1"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["REDIS_HOST"] = os.getenv("REDIS_HOST", "localhost")
    os.environ["RABBITMQ_URL"] = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")

    # Initialize test data directory
    os.makedirs("/tmp/zen_integration_tests", exist_ok=True)

    yield

    # Cleanup after all tests
    try:
        import shutil
        shutil.rmtree("/tmp/zen_integration_tests", ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
async def redis_client():
    """Provide Redis client for testing."""
    try:
        import redis
        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=15,  # Use test database
            decode_responses=True
        )
        client.ping()
        yield client

        # Cleanup test data
        client.flushdb()

    except Exception:
        pytest.skip("Redis not available for testing")


@pytest.fixture
async def rabbitmq_available():
    """Check if RabbitMQ is available for testing."""
    try:
        import aio_pika
        connection = await aio_pika.connect_robust(
            os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
        )
        await connection.close()
        return True
    except Exception:
        pytest.skip("RabbitMQ not available for testing")


@pytest.fixture
def sample_agent_request():
    """Provide sample agent request for testing."""
    from tools.shared.agent_models import AgentTaskRequest, AgentType

    return AgentTaskRequest(
        agent_type=AgentType.CLAUDE,
        task_description="Integration test task",
        message="This is a test task for integration testing",
        working_directory="/tmp/zen_integration_tests",
        timeout_seconds=120,
        env_vars={"TEST_MODE": "1"}
    )


# Markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (may require external services)"
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow running (may take several minutes)"
    )
    config.addinivalue_line(
        "markers",
        "requires_redis: marks tests that require Redis to be running"
    )
    config.addinivalue_line(
        "markers",
        "requires_rabbitmq: marks tests that require RabbitMQ to be running"
    )
    config.addinivalue_line(
        "markers",
        "requires_agents: marks tests that require AgentAPI agents to be installed"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers appropriately."""

    # Add integration marker to all tests in integration directory
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

    # Skip certain tests based on environment
    skip_integration = pytest.mark.skip(reason="Integration tests disabled")
    skip_slow = pytest.mark.skip(reason="Slow tests disabled")

    for item in items:
        if "integration" in item.keywords:
            if config.getoption("--no-integration", default=False):
                item.add_marker(skip_integration)

        if "slow" in item.keywords:
            if config.getoption("--no-slow", default=False):
                item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--no-integration",
        action="store_true",
        default=False,
        help="Skip integration tests"
    )
    parser.addoption(
        "--no-slow",
        action="store_true",
        default=False,
        help="Skip slow tests"
    )
    parser.addoption(
        "--integration-only",
        action="store_true",
        default=False,
        help="Run only integration tests"
    )


# Async test utilities
class AsyncTestHelper:
    """Helper utilities for async testing."""

    @staticmethod
    async def wait_for_condition(condition, timeout=10.0, interval=0.1):
        """Wait for a condition to become true."""
        import time
        start_time = time.time()

        while time.time() - start_time < timeout:
            if await condition() if asyncio.iscoroutinefunction(condition) else condition():
                return True
            await asyncio.sleep(interval)

        return False

    @staticmethod
    async def simulate_delay(min_seconds=0.1, max_seconds=0.5):
        """Simulate realistic processing delay."""
        import random
        delay = random.uniform(min_seconds, max_seconds)
        await asyncio.sleep(delay)


@pytest.fixture
def async_helper():
    """Provide async test helper."""
    return AsyncTestHelper()
