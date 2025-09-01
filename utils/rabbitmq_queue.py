from __future__ import annotations
"""
RabbitMQ Priority Task Queuing System

This module provides enterprise-grade task queuing using RabbitMQ with support for
priority queues, retry logic, dead letter queues, and load balancing.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel

try:
    import aio_pika
    from aio_pika import DeliveryMode, ExchangeType, Message
except ImportError:
    aio_pika = None
    Message = None
    DeliveryMode = None
    ExchangeType = None

from tools.shared.agent_models import AgentTaskRequest, TaskStatus
from utils.event_bus import get_event_bus

logger = logging.getLogger(__name__)


class TaskPriority(str, Enum):
    """Task priority levels."""

    CRITICAL = "critical"    # Priority 10 (highest)
    HIGH = "high"           # Priority 8
    NORMAL = "normal"       # Priority 5 (default)
    LOW = "low"            # Priority 2
    BULK = "bulk"          # Priority 1 (lowest)


class QueuedTask(BaseModel):
    """Queued task with metadata."""

    task_id: str
    queue_name: str
    priority: TaskPriority = TaskPriority.NORMAL
    agent_request: AgentTaskRequest
    created_at: datetime
    scheduled_at: Optional[datetime] = None  # For delayed tasks
    retry_count: int = 0
    max_retries: int = 3
    retry_delay_seconds: int = 30
    timeout_seconds: int = 300
    correlation_id: Optional[str] = None
    metadata: dict[str, Any] = {}


class TaskResult(BaseModel):
    """Task execution result."""

    task_id: str
    status: TaskStatus
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    completed_at: datetime
    retry_count: int = 0


class RabbitMQQueueManager:
    """Manages RabbitMQ-based priority task queuing with enterprise features."""

    def __init__(self,
                 connection_url: str = None,
                 exchange_name: str = "zen_agent_exchange",
                 dead_letter_exchange: str = "zen_agent_dlx"):
        """Initialize RabbitMQ queue manager."""

        if aio_pika is None:
            raise ImportError("aio-pika is required for RabbitMQ support. Install with: pip install aio-pika")

        self.connection_url = connection_url or os.getenv(
            "RABBITMQ_URL",
            "amqp://guest:guest@localhost:5672/"
        )
        self.exchange_name = exchange_name
        self.dead_letter_exchange = dead_letter_exchange

        # Connection and channel
        self.connection: Optional[aio_pika.Connection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.exchange: Optional[aio_pika.Exchange] = None
        self.dlx_exchange: Optional[aio_pika.Exchange] = None

        # Task handlers by queue name
        self.task_handlers: dict[str, Callable] = {}

        # Queues by name for monitoring
        self.queues: dict[str, aio_pika.Queue] = {}

        # Consumer tasks
        self.consumer_tasks: list[asyncio.Task] = []

        # Statistics
        self.stats = {
            "tasks_enqueued": 0,
            "tasks_processed": 0,
            "tasks_failed": 0,
            "tasks_retried": 0,
            "current_queue_size": 0
        }

        # Priority mapping
        self.priority_mapping = {
            TaskPriority.CRITICAL: 10,
            TaskPriority.HIGH: 8,
            TaskPriority.NORMAL: 5,
            TaskPriority.LOW: 2,
            TaskPriority.BULK: 1
        }

    async def connect(self):
        """Establish connection to RabbitMQ."""
        try:
            self.connection = await aio_pika.connect_robust(
                self.connection_url,
                client_properties={"connection_name": "zen-agent-queue-manager"}
            )
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=10)  # Process up to 10 tasks concurrently

            # Create main exchange
            self.exchange = await self.channel.declare_exchange(
                self.exchange_name,
                ExchangeType.TOPIC,
                durable=True
            )

            # Create dead letter exchange
            self.dlx_exchange = await self.channel.declare_exchange(
                self.dead_letter_exchange,
                ExchangeType.TOPIC,
                durable=True
            )

            # Create dead letter queue
            dlq = await self.channel.declare_queue(
                f"{self.exchange_name}_dead_letters",
                durable=True,
                arguments={"x-message-ttl": 86400000}  # 24 hours TTL
            )
            await dlq.bind(self.dlx_exchange, "#")

            logger.info("Connected to RabbitMQ successfully")

            # Publish connection event
            try:
                await get_event_bus().publish({
                    "event": "rabbitmq_connected",
                    "exchange": self.exchange_name,
                    "connection_url": self.connection_url.replace(":guest@", ":***@")
                })
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    async def disconnect(self):
        """Disconnect from RabbitMQ."""
        # Cancel consumer tasks
        for task in self.consumer_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.consumer_tasks.clear()

        # Close connection
        if self.connection:
            await self.connection.close()
            self.connection = None
            self.channel = None
            self.exchange = None
            self.dlx_exchange = None

        logger.info("Disconnected from RabbitMQ")

    async def create_queue(self,
                          queue_name: str,
                          routing_key: str = None,
                          max_priority: int = 10,
                          enable_retries: bool = True,
                          max_length: int = None) -> aio_pika.Queue:
        """Create a priority queue with optional configuration."""

        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")

        routing_key = routing_key or queue_name

        # Queue arguments
        arguments = {
            "x-max-priority": max_priority,
        }

        # Add dead letter routing for retries
        if enable_retries:
            arguments.update({
                "x-dead-letter-exchange": self.dead_letter_exchange,
                "x-dead-letter-routing-key": f"retry.{routing_key}"
            })

        # Add max length if specified (prevents queue overflow)
        if max_length:
            arguments["x-max-length"] = max_length

        # Create queue
        queue = await self.channel.declare_queue(
            queue_name,
            durable=True,
            arguments=arguments
        )

        # Bind to exchange
        await queue.bind(self.exchange, routing_key)

        # Create retry queue if retries are enabled
        if enable_retries:
            retry_queue = await self.channel.declare_queue(
                f"{queue_name}_retry",
                durable=True,
                arguments={
                    "x-dead-letter-exchange": self.exchange_name,
                    "x-dead-letter-routing-key": routing_key,
                    "x-message-ttl": 30000  # 30 second retry delay
                }
            )
            await retry_queue.bind(self.dlx_exchange, f"retry.{routing_key}")

        self.queues[queue_name] = queue
        logger.info(f"Created priority queue: {queue_name}")
        return queue

    async def enqueue_task(self,
                          queue_name: str,
                          task: QueuedTask,
                          routing_key: str = None) -> str:
        """Enqueue a task with priority."""

        if not self.exchange:
            raise RuntimeError("Not connected to RabbitMQ")

        routing_key = routing_key or queue_name

        # Serialize task
        task_payload = task.model_dump_json()

        # Create message with priority
        priority = self.priority_mapping.get(task.priority, 5)

        message = Message(
            task_payload.encode(),
            delivery_mode=DeliveryMode.PERSISTENT,
            priority=priority,
            message_id=task.task_id,
            correlation_id=task.correlation_id or str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            headers={
                "task_id": task.task_id,
                "priority": task.priority.value,
                "retry_count": task.retry_count,
                "max_retries": task.max_retries
            }
        )

        # Handle delayed tasks
        if task.scheduled_at:
            delay_ms = int((task.scheduled_at - datetime.now(timezone.utc)).total_seconds() * 1000)
            if delay_ms > 0:
                message.headers["x-delay"] = delay_ms

        # Publish message
        await self.exchange.publish(message, routing_key=routing_key)

        self.stats["tasks_enqueued"] += 1
        logger.debug(f"Enqueued task {task.task_id} to {queue_name} with priority {task.priority}")

        # Publish event
        try:
            await get_event_bus().publish({
                "event": "task_enqueued",
                "task_id": task.task_id,
                "queue": queue_name,
                "priority": task.priority.value
            })
        except Exception:
            pass

        return task.task_id

    async def enqueue_agent_task(self,
                                queue_name: str,
                                agent_request: AgentTaskRequest,
                                priority: TaskPriority = TaskPriority.NORMAL,
                                max_retries: int = 3,
                                timeout_seconds: int = 300,
                                **kwargs) -> str:
        """Convenience method to enqueue an agent task."""

        task = QueuedTask(
            task_id=str(uuid.uuid4()),
            queue_name=queue_name,
            priority=priority,
            agent_request=agent_request,
            created_at=datetime.now(timezone.utc),
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            **kwargs
        )

        return await self.enqueue_task(queue_name, task)

    def register_task_handler(self, queue_name: str, handler: Callable[[QueuedTask], TaskResult]):
        """Register a task handler for a specific queue."""
        self.task_handlers[queue_name] = handler
        logger.info(f"Registered task handler for queue: {queue_name}")

    async def start_consumer(self,
                            queue_name: str,
                            handler: Callable = None,
                            concurrency: int = 1) -> list[asyncio.Task]:
        """Start consuming tasks from a queue."""

        if queue_name not in self.queues:
            raise ValueError(f"Queue {queue_name} not found. Create it first.")

        queue = self.queues[queue_name]
        handler = handler or self.task_handlers.get(queue_name)

        if not handler:
            raise ValueError(f"No handler registered for queue {queue_name}")

        # Start multiple consumer tasks for concurrency
        consumer_tasks = []
        for i in range(concurrency):
            task = asyncio.create_task(
                self._consume_messages(queue, handler, f"{queue_name}_consumer_{i}")
            )
            consumer_tasks.append(task)

        self.consumer_tasks.extend(consumer_tasks)
        logger.info(f"Started {concurrency} consumers for queue {queue_name}")

        return consumer_tasks

    async def _consume_messages(self,
                               queue: aio_pika.Queue,
                               handler: Callable,
                               consumer_name: str):
        """Consume messages from a queue."""

        async def process_message(message: aio_pika.IncomingMessage):
            """Process a single message."""
            start_time = datetime.now(timezone.utc)
            task_result = None

            try:
                # Parse task
                task_data = json.loads(message.body.decode())
                queued_task = QueuedTask.model_validate(task_data)

                logger.debug(f"Processing task {queued_task.task_id} (retry {queued_task.retry_count})")

                # Execute handler
                task_result = await asyncio.get_event_loop().run_in_executor(
                    None, handler, queued_task
                )

                # Calculate execution time
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                task_result.execution_time_seconds = execution_time

                # Acknowledge message on success
                await message.ack()

                self.stats["tasks_processed"] += 1
                logger.info(f"Successfully processed task {queued_task.task_id} in {execution_time:.2f}s")

                # Publish success event
                try:
                    await get_event_bus().publish({
                        "event": "task_completed",
                        "task_id": queued_task.task_id,
                        "status": task_result.status.value,
                        "execution_time": execution_time
                    })
                except Exception:
                    pass

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)

                # Check retry logic
                retry_count = message.headers.get("retry_count", 0)
                max_retries = message.headers.get("max_retries", 3)

                if retry_count < max_retries:
                    # Reject with requeue for retry
                    await message.reject(requeue=False)  # Let DLX handle retry
                    self.stats["tasks_retried"] += 1
                    logger.info(f"Task will be retried (attempt {retry_count + 1}/{max_retries})")
                else:
                    # Max retries reached, acknowledge and send to DLQ
                    await message.ack()
                    self.stats["tasks_failed"] += 1
                    logger.error(f"Task failed after {max_retries} retries, sending to dead letter queue")

                    # Publish failure event
                    try:
                        await get_event_bus().publish({
                            "event": "task_failed",
                            "task_id": getattr(queued_task, 'task_id', 'unknown'),
                            "error": str(e),
                            "retry_count": retry_count
                        })
                    except Exception:
                        pass

        # Start consuming
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                await process_message(message)

    async def get_queue_stats(self, queue_name: str = None) -> dict[str, Any]:
        """Get queue statistics."""
        stats = self.stats.copy()

        if queue_name and queue_name in self.queues:
            try:
                # Get queue-specific info (requires RabbitMQ Management API)
                # For now, return basic stats
                stats["queue_name"] = queue_name
            except Exception:
                pass

        return stats

    async def purge_queue(self, queue_name: str) -> int:
        """Purge all messages from a queue."""
        if queue_name not in self.queues:
            raise ValueError(f"Queue {queue_name} not found")

        queue = self.queues[queue_name]
        result = await queue.purge()

        logger.info(f"Purged {result} messages from queue {queue_name}")
        return result

    async def delete_queue(self, queue_name: str):
        """Delete a queue."""
        if queue_name not in self.queues:
            raise ValueError(f"Queue {queue_name} not found")

        queue = self.queues[queue_name]
        await queue.delete()
        del self.queues[queue_name]

        logger.info(f"Deleted queue {queue_name}")

    async def create_agent_queues(self):
        """Create standard agent queues with appropriate configurations."""

        # High-priority queue for critical tasks
        await self.create_queue(
            "agent_critical",
            routing_key="agent.critical",
            max_priority=10,
            max_length=1000
        )

        # Normal priority queue for standard tasks
        await self.create_queue(
            "agent_normal",
            routing_key="agent.normal",
            max_priority=10,
            max_length=5000
        )

        # Low priority queue for bulk/background tasks
        await self.create_queue(
            "agent_bulk",
            routing_key="agent.bulk",
            max_priority=10,
            max_length=10000
        )

        # Queue for batch processing
        await self.create_queue(
            "agent_batch",
            routing_key="agent.batch",
            max_priority=10,
            max_length=2000
        )

        logger.info("Created standard agent queues")

    # Context manager support
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


# Global queue manager instance
_queue_manager: Optional[RabbitMQQueueManager] = None


def get_queue_manager() -> RabbitMQQueueManager:
    """Get the global queue manager instance.

    Returns a real RabbitMQ manager when available, otherwise falls back to an
    in-memory stand-in when ZEN_RABBITMQ_FAKE=1 is set or aio_pika is missing.
    """
    global _queue_manager
    if _queue_manager is None:
        use_fake = os.getenv("ZEN_RABBITMQ_FAKE", "0").lower() in ("1", "true", "yes")
        if use_fake or aio_pika is None:
            try:
                # Lazy define here to avoid top-level import noise
                class InMemoryQueue:
                    def __init__(self, name: str):
                        self.name = name

                class InMemoryQueueManager:
                    def __init__(self, exchange_name: str = "zen_agent_exchange"):
                        self.exchange_name = exchange_name
                        self.queues: dict[str, InMemoryQueue] = {}
                        self.task_handlers: dict[str, Callable] = {}
                        self.stats = {
                            "tasks_enqueued": 0,
                            "tasks_processed": 0,
                            "tasks_failed": 0,
                            "tasks_retried": 0,
                            "current_queue_size": 0,
                        }

                    async def __aenter__(self):
                        return self

                    async def __aexit__(self, exc_type, exc, tb):
                        return False

                    async def connect(self):
                        return True

                    async def disconnect(self):
                        return True

                    async def create_queue(self, queue_name: str, routing_key: str = None, max_priority: int = 10,
                                           enable_retries: bool = True, max_length: int = None) -> InMemoryQueue:
                        q = InMemoryQueue(queue_name)
                        self.queues[queue_name] = q
                        return q

                    async def create_agent_queues(self):
                        for qn in ("agent_critical", "agent_normal", "agent_bulk", "agent_batch"):
                            await self.create_queue(qn)
                        return True

                    def register_task_handler(self, queue_name: str, handler: Callable):
                        self.task_handlers[queue_name] = handler

                    async def enqueue_task(self, queue_name: str, task: QueuedTask, routing_key: str = None) -> str:
                        self.stats["tasks_enqueued"] += 1
                        self.stats["current_queue_size"] += 1
                        handler = self.task_handlers.get(queue_name)
                        if handler:
                            try:
                                res = handler(task)
                                if asyncio.iscoroutine(res):
                                    await res
                                self.stats["tasks_processed"] += 1
                            except Exception:
                                self.stats["tasks_failed"] += 1
                        return task.task_id

                    async def enqueue_agent_task(self, queue_name: str, agent_request: AgentTaskRequest,
                                                 priority: TaskPriority = TaskPriority.NORMAL,
                                                 max_retries: int = 3,
                                                 timeout_seconds: int = 300,
                                                 metadata: dict[str, Any] | None = None) -> str:
                        t = QueuedTask(
                            task_id=str(uuid.uuid4()),
                            queue_name=queue_name,
                            priority=priority,
                            agent_request=agent_request,
                            created_at=datetime.now(timezone.utc),
                            max_retries=max_retries,
                            timeout_seconds=timeout_seconds,
                            metadata=metadata or {},
                        )
                        return await self.enqueue_task(queue_name, t)

                _queue_manager = InMemoryQueueManager()
            except Exception:
                # Fallback to raising if even fake cannot be created
                _queue_manager = RabbitMQQueueManager()
        else:
            _queue_manager = RabbitMQQueueManager()
    return _queue_manager


# Example task handler
async def default_agent_task_handler(queued_task: QueuedTask) -> TaskResult:
    """Default handler for agent tasks."""

    # Import here to avoid circular imports
    from utils.agent_manager import get_task_manager

    try:
        # Create agent task
        task_manager = get_task_manager()
        agent_task = await task_manager.create_task(queued_task.agent_request)

        # Start task
        success = await task_manager.start_task(agent_task.task_id)

        if success:
            # Wait for completion (simplified - real implementation would be more sophisticated)
            # This would integrate with the existing task monitoring system
            return TaskResult(
                task_id=queued_task.task_id,
                status=TaskStatus.COMPLETED,
                result={"message": "Task completed successfully"},
                completed_at=datetime.now(timezone.utc)
            )
        else:
            return TaskResult(
                task_id=queued_task.task_id,
                status=TaskStatus.FAILED,
                error="Failed to start agent task",
                completed_at=datetime.now(timezone.utc)
            )

    except Exception as e:
        return TaskResult(
            task_id=queued_task.task_id,
            status=TaskStatus.FAILED,
            error=str(e),
            completed_at=datetime.now(timezone.utc)
        )
