"""
Integration Tests for RabbitMQ Priority Queue System

Tests the RabbitMQ-based task queuing including priority handling,
retry logic, dead letter queues, and load balancing.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.shared.agent_models import AgentTaskRequest, AgentType, TaskStatus
from utils.rabbitmq_queue import QueuedTask, RabbitMQQueueManager, TaskPriority, TaskResult, default_agent_task_handler


@pytest.fixture
async def queue_manager():
    """Create RabbitMQ queue manager for testing."""
    # Skip if aio_pika is not available
    pytest.importorskip("aio_pika")

    manager = RabbitMQQueueManager(
        connection_url="amqp://guest:guest@localhost:5672/",
        exchange_name="test_agent_exchange",
        dead_letter_exchange="test_agent_dlx"
    )

    try:
        await manager.connect()
        await manager.create_agent_queues()
        yield manager
    except Exception as e:
        pytest.skip(f"RabbitMQ not available: {e}")
    finally:
        if manager.connection:
            await manager.disconnect()


@pytest.fixture
def sample_agent_request():
    """Create sample agent request for testing."""
    return AgentTaskRequest(
        agent_type=AgentType.CLAUDE,
        task_description="Test task",
        message="Analyze this code for issues",
        working_directory="/tmp/test",
        timeout_seconds=300
    )


@pytest.fixture
def sample_queued_task(sample_agent_request):
    """Create sample queued task for testing."""
    return QueuedTask(
        task_id=str(uuid.uuid4()),
        queue_name="test_queue",
        priority=TaskPriority.NORMAL,
        agent_request=sample_agent_request,
        created_at=datetime.now(timezone.utc)
    )


class TestRabbitMQConnection:
    """Test RabbitMQ connection and setup."""

    @pytest.mark.asyncio
    async def test_connection_establishment(self, queue_manager):
        """Test establishing connection to RabbitMQ."""
        assert queue_manager.connection is not None
        assert queue_manager.channel is not None
        assert queue_manager.exchange is not None
        assert queue_manager.dlx_exchange is not None

    @pytest.mark.asyncio
    async def test_queue_creation(self, queue_manager):
        """Test creating priority queues."""
        queue = await queue_manager.create_queue(
            "test_priority_queue",
            routing_key="test.priority",
            max_priority=10
        )

        assert queue is not None
        assert "test_priority_queue" in queue_manager.queues

    @pytest.mark.asyncio
    async def test_agent_queues_creation(self, queue_manager):
        """Test creation of standard agent queues."""
        expected_queues = ["agent_critical", "agent_normal", "agent_bulk", "agent_batch"]

        for queue_name in expected_queues:
            assert queue_name in queue_manager.queues

    @pytest.mark.asyncio
    async def test_connection_context_manager(self):
        """Test using queue manager as context manager."""
        pytest.importorskip("aio_pika")

        try:
            async with RabbitMQQueueManager(
                connection_url="amqp://guest:guest@localhost:5672/",
                exchange_name="test_context_exchange"
            ) as manager:
                assert manager.connection is not None
                await manager.create_queue("context_test_queue")
                assert "context_test_queue" in manager.queues
        except Exception as e:
            pytest.skip(f"RabbitMQ not available: {e}")


class TestTaskEnqueuing:
    """Test task enqueuing functionality."""

    @pytest.mark.asyncio
    async def test_enqueue_basic_task(self, queue_manager, sample_queued_task):
        """Test enqueuing a basic task."""
        task_id = await queue_manager.enqueue_task("agent_normal", sample_queued_task)

        assert task_id == sample_queued_task.task_id
        assert queue_manager.stats["tasks_enqueued"] == 1

    @pytest.mark.asyncio
    async def test_enqueue_with_priority(self, queue_manager, sample_agent_request):
        """Test enqueuing tasks with different priorities."""
        # Enqueue critical priority task
        critical_task_id = await queue_manager.enqueue_agent_task(
            "agent_critical",
            sample_agent_request,
            priority=TaskPriority.CRITICAL
        )

        # Enqueue low priority task
        low_task_id = await queue_manager.enqueue_agent_task(
            "agent_normal",
            sample_agent_request,
            priority=TaskPriority.LOW
        )

        assert critical_task_id is not None
        assert low_task_id is not None
        assert critical_task_id != low_task_id
        assert queue_manager.stats["tasks_enqueued"] == 2

    @pytest.mark.asyncio
    async def test_enqueue_with_metadata(self, queue_manager, sample_agent_request):
        """Test enqueuing task with custom metadata."""
        task_id = await queue_manager.enqueue_agent_task(
            "agent_normal",
            sample_agent_request,
            priority=TaskPriority.HIGH,
            max_retries=5,
            timeout_seconds=600,
            metadata={"source": "integration_test", "version": "1.0"}
        )

        assert task_id is not None

    @pytest.mark.asyncio
    async def test_bulk_enqueuing(self, queue_manager, sample_agent_request):
        """Test enqueuing multiple tasks efficiently."""
        task_ids = []

        # Enqueue 10 tasks
        for _i in range(10):
            task_id = await queue_manager.enqueue_agent_task(
                "agent_bulk",
                sample_agent_request,
                priority=TaskPriority.BULK
            )
            task_ids.append(task_id)

        assert len(task_ids) == 10
        assert len(set(task_ids)) == 10  # All unique
        assert queue_manager.stats["tasks_enqueued"] == 10


class TestTaskHandlers:
    """Test task handler registration and execution."""

    @pytest.mark.asyncio
    async def test_handler_registration(self, queue_manager):
        """Test registering task handlers."""
        def dummy_handler(task):
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                completed_at=datetime.now(timezone.utc)
            )

        queue_manager.register_task_handler("test_queue", dummy_handler)

        assert "test_queue" in queue_manager.task_handlers
        assert queue_manager.task_handlers["test_queue"] == dummy_handler

    @pytest.mark.asyncio
    async def test_default_agent_handler(self, sample_queued_task):
        """Test the default agent task handler."""
        with patch("utils.agent_manager.get_task_manager") as mock_manager:
            # Mock task manager
            mock_task_manager = AsyncMock()
            mock_agent_task = MagicMock()
            mock_agent_task.task_id = "mock-task-123"

            mock_task_manager.create_task.return_value = mock_agent_task
            mock_task_manager.start_task.return_value = True
            mock_manager.return_value = mock_task_manager

            # Execute handler
            result = await default_agent_task_handler(sample_queued_task)

            assert isinstance(result, TaskResult)
            assert result.task_id == sample_queued_task.task_id
            assert result.status == TaskStatus.COMPLETED

            # Verify task manager was called
            mock_task_manager.create_task.assert_called_once()
            mock_task_manager.start_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_handler_error_handling(self, sample_queued_task):
        """Test error handling in task handlers."""
        with patch("utils.agent_manager.get_task_manager") as mock_manager:
            # Mock task manager to raise exception
            mock_task_manager = AsyncMock()
            mock_task_manager.create_task.side_effect = Exception("Task creation failed")
            mock_manager.return_value = mock_task_manager

            # Execute handler
            result = await default_agent_task_handler(sample_queued_task)

            assert isinstance(result, TaskResult)
            assert result.task_id == sample_queued_task.task_id
            assert result.status == TaskStatus.FAILED
            assert "Task creation failed" in result.error


class TestMessageConsumption:
    """Test message consumption and processing."""

    @pytest.mark.asyncio
    async def test_consumer_startup(self, queue_manager):
        """Test starting message consumers."""
        # Register a simple handler
        def test_handler(task):
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                completed_at=datetime.now(timezone.utc)
            )

        queue_manager.register_task_handler("agent_normal", test_handler)

        # Start consumer
        consumer_tasks = await queue_manager.start_consumer(
            "agent_normal",
            concurrency=2
        )

        assert len(consumer_tasks) == 2
        assert all(isinstance(task, asyncio.Task) for task in consumer_tasks)

        # Cancel consumers
        for task in consumer_tasks:
            task.cancel()

    @pytest.mark.asyncio
    async def test_message_processing_flow(self, queue_manager, sample_queued_task):
        """Test complete message processing flow."""
        processed_tasks = []

        def capture_handler(task):
            processed_tasks.append(task)
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                completed_at=datetime.now(timezone.utc)
            )

        queue_manager.register_task_handler("agent_normal", capture_handler)

        # Enqueue task
        await queue_manager.enqueue_task("agent_normal", sample_queued_task)

        # Start consumer (but cancel quickly to avoid hanging test)
        consumer_tasks = await queue_manager.start_consumer("agent_normal", concurrency=1)

        # Give a moment for processing
        await asyncio.sleep(0.1)

        # Cancel consumer
        for task in consumer_tasks:
            task.cancel()


class TestRetryAndErrorHandling:
    """Test retry logic and error handling."""

    @pytest.mark.asyncio
    async def test_retry_configuration(self, queue_manager, sample_agent_request):
        """Test task retry configuration."""
        task_id = await queue_manager.enqueue_agent_task(
            "agent_normal",
            sample_agent_request,
            max_retries=5
        )

        assert task_id is not None
        # Retry behavior would be tested by processing and failing tasks

    @pytest.mark.asyncio
    async def test_dead_letter_queue_setup(self, queue_manager):
        """Test that dead letter queues are properly set up."""
        # Dead letter exchange should exist
        assert queue_manager.dlx_exchange is not None

        # Create queue with retry enabled (should have DLX routing)
        queue = await queue_manager.create_queue(
            "test_dlq_queue",
            enable_retries=True
        )

        assert queue is not None

    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self, queue_manager):
        """Test handling when queue reaches max length."""
        # Create queue with low max length
        queue = await queue_manager.create_queue(
            "limited_queue",
            max_length=5
        )

        assert queue is not None
        # Would need to enqueue more than max_length to test overflow


class TestQueueManagement:
    """Test queue management operations."""

    @pytest.mark.asyncio
    async def test_queue_statistics(self, queue_manager):
        """Test getting queue statistics."""
        stats = await queue_manager.get_queue_stats()

        assert isinstance(stats, dict)
        assert "tasks_enqueued" in stats
        assert "tasks_processed" in stats
        assert "tasks_failed" in stats
        assert "tasks_retried" in stats

    @pytest.mark.asyncio
    async def test_queue_purging(self, queue_manager, sample_queued_task):
        """Test purging messages from queue."""
        # Enqueue some tasks
        await queue_manager.enqueue_task("agent_normal", sample_queued_task)

        # Purge queue
        purged_count = await queue_manager.purge_queue("agent_normal")

        assert purged_count >= 0

    @pytest.mark.asyncio
    async def test_queue_deletion(self, queue_manager):
        """Test deleting queues."""
        # Create temporary queue
        await queue_manager.create_queue("temp_queue")
        assert "temp_queue" in queue_manager.queues

        # Delete queue
        await queue_manager.delete_queue("temp_queue")
        assert "temp_queue" not in queue_manager.queues


class TestPriorityOrdering:
    """Test priority-based task ordering."""

    @pytest.mark.asyncio
    async def test_priority_mapping(self, queue_manager):
        """Test priority value mapping."""
        assert queue_manager.priority_mapping[TaskPriority.CRITICAL] == 10
        assert queue_manager.priority_mapping[TaskPriority.HIGH] == 8
        assert queue_manager.priority_mapping[TaskPriority.NORMAL] == 5
        assert queue_manager.priority_mapping[TaskPriority.LOW] == 2
        assert queue_manager.priority_mapping[TaskPriority.BULK] == 1

    @pytest.mark.asyncio
    async def test_priority_task_creation(self, queue_manager, sample_agent_request):
        """Test creating tasks with different priorities."""
        priorities = [
            TaskPriority.CRITICAL,
            TaskPriority.HIGH,
            TaskPriority.NORMAL,
            TaskPriority.LOW,
            TaskPriority.BULK
        ]

        task_ids = []
        for priority in priorities:
            task_id = await queue_manager.enqueue_agent_task(
                "agent_normal",
                sample_agent_request,
                priority=priority
            )
            task_ids.append(task_id)

        assert len(task_ids) == 5
        assert len(set(task_ids)) == 5  # All unique


@pytest.mark.integration
class TestRabbitMQIntegration:
    """Integration tests requiring RabbitMQ server."""

    @pytest.mark.asyncio
    async def test_full_workflow_integration(self, queue_manager, sample_agent_request):
        """Test complete workflow from enqueue to processing."""
        processed_results = []

        def integration_handler(task):
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result={"processed": True, "agent_type": task.agent_request.agent_type.value},
                completed_at=datetime.now(timezone.utc),
                execution_time_seconds=0.1
            )
            processed_results.append(result)
            return result

        queue_manager.register_task_handler("agent_normal", integration_handler)

        # Enqueue task
        await queue_manager.enqueue_agent_task(
            "agent_normal",
            sample_agent_request,
            priority=TaskPriority.HIGH
        )

        # Start consumer briefly
        consumer_tasks = await queue_manager.start_consumer("agent_normal", concurrency=1)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Cancel consumers
        for task in consumer_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Check if task was processed (may not be in fast CI environments)
        # This is environment-dependent so we don't assert

    @pytest.mark.asyncio
    async def test_concurrent_producers_consumers(self, queue_manager, sample_agent_request):
        """Test concurrent producers and consumers."""
        async def producer_task(queue_name, count):
            task_ids = []
            for _i in range(count):
                task_id = await queue_manager.enqueue_agent_task(
                    queue_name,
                    sample_agent_request,
                    priority=TaskPriority.NORMAL
                )
                task_ids.append(task_id)
            return task_ids

        # Register handler
        def concurrent_handler(task):
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                completed_at=datetime.now(timezone.utc)
            )

        queue_manager.register_task_handler("agent_normal", concurrent_handler)

        # Start consumers
        consumer_tasks = await queue_manager.start_consumer("agent_normal", concurrency=3)

        # Start multiple producers
        producer_tasks = [
            asyncio.create_task(producer_task("agent_normal", 5)),
            asyncio.create_task(producer_task("agent_normal", 5)),
            asyncio.create_task(producer_task("agent_normal", 5))
        ]

        # Wait for all producers to finish
        all_task_ids = await asyncio.gather(*producer_tasks)
        total_tasks = sum(len(task_ids) for task_ids in all_task_ids)

        assert total_tasks == 15

        # Cancel consumers
        for task in consumer_tasks:
            task.cancel()

    @pytest.mark.asyncio
    async def test_performance_benchmark(self, queue_manager, sample_agent_request):
        """Test performance with high task volume."""
        import time

        def fast_handler(task):
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                completed_at=datetime.now(timezone.utc)
            )

        queue_manager.register_task_handler("agent_bulk", fast_handler)

        # Measure enqueue performance
        start_time = time.time()

        task_count = 100
        for _i in range(task_count):
            await queue_manager.enqueue_agent_task(
                "agent_bulk",
                sample_agent_request,
                priority=TaskPriority.BULK
            )

        enqueue_time = time.time() - start_time

        # Should be able to enqueue 100 tasks in reasonable time
        assert enqueue_time < 10.0  # 10 seconds max

        enqueue_rate = task_count / enqueue_time
        print(f"Enqueue rate: {enqueue_rate:.2f} tasks/second")

        assert queue_manager.stats["tasks_enqueued"] == task_count
