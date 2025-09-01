"""
Agent Task Management System

This module provides the core infrastructure for managing agent tasks,
including task lifecycle, status tracking, and result storage.
"""

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tools.shared.agent_models import AgentTask, AgentTaskRequest, AgentTaskResult, AgentType

import httpx

# Optional Redis dependency
try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None

# Moved to function level to avoid circular imports
# from tools.shared.agent_models import AgentTask, AgentTaskRequest, AgentTaskResult, AgentType, TaskStatus

def _get_agent_models():
    """Import agent models to avoid circular import at module level."""
    from tools.shared.agent_models import AgentTask, AgentTaskRequest, AgentTaskResult, AgentType, TaskStatus
    return AgentTask, AgentTaskRequest, AgentTaskResult, AgentType, TaskStatus

# Import at function level to avoid circular import
# from utils.agent_defaults import build_effective_path_env
from utils.streaming_protocol import StreamingResponseParser, StreamMessageType, get_streaming_manager  # noqa: E402

DIAGNOSTIC = os.getenv("AGENTAPI_DIAGNOSTIC", "0") == "1"

logger = logging.getLogger(__name__)


class AgentTaskManager:
    """Manages agent task lifecycle and execution."""

    def __init__(self, redis_client: Optional[object] = None):
        """Initialize the agent task manager with enhanced Redis integration and streaming support."""
        # Enhanced Redis integration
        try:
            from utils.redis_manager import get_redis_manager
            self.redis_manager = get_redis_manager()
            logger.info("Agent Task Manager initialized with enterprise Redis integration")
        except ImportError:
            logger.warning("Redis manager not available, falling back to basic Redis client")
            self.redis_manager = None

        # Always ensure redis_client is available for backward compatibility
        self.redis_client = redis_client or self._get_redis_client()

        self.active_tasks: dict[str, AgentTask] = {}

        # Enhanced port management - expanded range for enterprise scale
        self.port_pool: set[int] = set(range(3284, 10000))  # Expanded range for 1000+ agents
        self.used_ports: set[int] = set()
        self.retention_seconds: int = int(os.getenv("AGENT_TASK_RETENTION_SEC", os.getenv("ZEN_AGENT_TASK_RETENTION_SEC", "3600")))  # default 1 hour

        # Streaming support
        self.streaming_manager = get_streaming_manager()
        self.streaming_parser = StreamingResponseParser(self.streaming_manager)

        # Optional TaskStore facade over Redis
        self.task_store = None
        if self.redis_client:
            try:
                from utils.task_store import TaskStore
                self.task_store = TaskStore(self.redis_client)
            except Exception:
                logger.debug("TaskStore not available", exc_info=True)

        # Register a global stream tap to persist all broadcast messages into Redis Streams
        try:
            from utils.streaming_protocol import StreamMessageType, register_message_tap
            async def _tap(task_id: str, mtype: StreamMessageType, content: dict) -> None:
                await self._append_task_message(task_id, mtype.value, content)
            register_message_tap(_tap)
        except Exception:
            logger.debug("Failed to register message tap", exc_info=True)

        self._cleanup_lock = asyncio.Lock()
        # Concurrency and queue/backpressure
        self._max_concurrent_tasks: int = int(os.getenv("MAX_CONCURRENT_TASKS", "24"))
        self._queue_max: int = int(os.getenv("TASK_QUEUE_MAX", "200"))
        self._sem = asyncio.Semaphore(self._max_concurrent_tasks)
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self._queue_max)
        self._dequeuer_task: Optional[asyncio.Task] = None


    def _get_redis_client(self):
        """Get Redis client for task storage, honoring ZEN_STORAGE env flag."""
        try:
            if os.getenv("ZEN_STORAGE", os.getenv("ZEN_STORAGE_MODE", "memory")).lower() != "redis":
                # Explicitly disabled; use in-memory
                logger.info("ZEN_STORAGE!=redis; using in-memory task storage")
                return None
            # Try to connect to Redis for persistent storage
            client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "1")),  # Use DB 1 for agent tasks
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            client.ping()  # Test connection
            logger.debug("Connected to Redis for agent task storage")
            return client
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory storage: {e}")
            return None

    def _allocate_port(self, agent_id: Optional[str] = None) -> Optional[int]:
        """Allocate an available port for AgentAPI with Redis coordination."""
        # Use Redis-based port allocation if available
        if self.redis_manager and agent_id:
            allocated_port = self.redis_manager.allocate_port(agent_id, (3284, 10000))
            if allocated_port:
                self.used_ports.add(allocated_port)
                logger.debug(f"Allocated port {allocated_port} for agent {agent_id} via Redis")
                return allocated_port

        # Fallback to local port pool allocation
        available_ports = self.port_pool - self.used_ports
        if not available_ports:
            logger.error("No available ports for AgentAPI")
            return None

        port = min(available_ports)
        self.used_ports.add(port)
        logger.debug(f"Allocated port {port} for AgentAPI (local pool)")
        return port

    def _release_port(self, port: int, agent_id: Optional[str] = None) -> None:
        """Release a port back to the pool with Redis coordination."""
        # Use Redis-based port release if available
        if self.redis_manager and agent_id:
            success = self.redis_manager.release_port(agent_id, port)
            if success:
                logger.debug(f"Released port {port} for agent {agent_id} via Redis")
            else:
                logger.warning(f"Failed to release port {port} via Redis for agent {agent_id}")

        # Always update local port pool
        if port in self.used_ports:
            self.used_ports.remove(port)
            logger.debug(f"Released port {port} from local pool")

    async def create_task(self, request: "AgentTaskRequest") -> "AgentTask":
        """Create a new agent task."""
        AgentTask, AgentTaskRequest, AgentTaskResult, AgentType, TaskStatus = _get_agent_models()
        task = AgentTask(request=request)

        # Allocate port for AgentAPI

        async def _publish_lifecycle(self, event: str, payload: dict) -> None:
            """Publish lifecycle event via unified EventBus adapter."""
            try:
                from utils.event_bus_adapter import get_event_publisher
                await get_event_publisher().publish_lifecycle(event, payload)
            except Exception:
                logger.debug("Lifecycle publish via adapter failed", exc_info=True)


        port = self._allocate_port()
        # Append creation message to stream
        await self._append_task_message(task.task_id, "created", {
            "agent": request.agent_type.value,
            "description": request.task_description,
            "port": task.agent_port,
        })

        if not port:
            task.status = TaskStatus.FAILED
            if task.result:
                task.result.error = "No available ports for AgentAPI"
            return task

        task.agent_port = port

        # Store task
        self.active_tasks[task.task_id] = task
        # Publish event & log JSON for created task
        try:
            await self._publish_lifecycle("task_created", {
                "task_id": task.task_id,
                "agent": request.agent_type.value,
                "status": task.status.value,
                "description": request.task_description,
                "port": task.agent_port,
            })
            logger.info(
                "TASK_EVENT %s",
                json.dumps({
                    "event": "task_created",
                    "task_id": task.task_id,
                    "agent": request.agent_type.value,
                    "status": task.status.value,
                    "description": request.task_description,
                    "port": task.agent_port,
                }),
            )

            # Send streaming update
            await self.streaming_manager.broadcast_message(
                task.task_id,
                StreamMessageType.STATUS_UPDATE,
                {
                    "status": task.status.value,
                    "description": request.task_description,
                    "port": task.agent_port,
                    "message": "Task created and ready to start",
                },
                request.agent_type.value,
            )
            # Enqueue task for execution respecting concurrency/backpressure
            try:
                self._queue.put_nowait(task.task_id)
            except asyncio.QueueFull:
                logger.warning("Task queue full; rejecting task %s", task.task_id)
            else:
                if not self._dequeuer_task or self._dequeuer_task.done():
                    self._dequeuer_task = asyncio.create_task(self._dequeue_loop())
        except Exception:
            logger.debug("Event publish failed for task_created", exc_info=True)

        await self._store_task(task)

        logger.info(f"Created agent task {task.task_id} for {request.agent_type} on port {port}")
        return task

    async def start_task(self, task_id: str) -> bool:
        """Start executing an agent task."""
        # Import runtime models to avoid circular imports
        AgentTask, AgentTaskRequest, AgentTaskResult, AgentType, TaskStatus = _get_agent_models()
        task = await self.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return False

        if task.status != TaskStatus.PENDING:
            logger.warning(f"Task {task_id} is not in pending state: {task.status}")
            return False

        try:
            # Update task status
            task.status = TaskStatus.STARTING
            task.updated_at = datetime.now(timezone.utc)
            await self._store_task(task)

            # Start AgentAPI server for this task
            success = await self._start_agent_server(task)
            if not success:
                task.status = TaskStatus.FAILED
                if not task.result:
                    task.result = AgentTaskResult(
                        task_id=task_id,
                        agent_type=task.request.agent_type,
                        status=TaskStatus.FAILED,
                        started_at=datetime.now(timezone.utc),
                        error="Failed to start AgentAPI server",
                    )
                await self._store_task(task)
                return False

            # Send initial message to agent
            success = await self._send_message_to_agent(task, task.request.message)
            if not success:
                task.status = TaskStatus.FAILED
                if task.result:
                    task.result.error = "Failed to send message to agent"
            # Append running status
            await self._append_task_message(task_id, "status", {"status": "running", "port": task.agent_port})

            await self._store_task(task)
            return False

            task.status = TaskStatus.RUNNING
            task.updated_at = datetime.now(timezone.utc)
            await self._store_task(task)

            # Send streaming update for running status
            await self.streaming_manager.broadcast_message(
                task_id,
                StreamMessageType.STATUS_UPDATE,
                {
                    "status": "running",
                    "message": "Agent task started successfully",
                    "port": task.agent_port
                },
                task.request.agent_type.value
            )

            logger.info(f"Started agent task {task_id}")
            return True

        except Exception as e:
            logger.error(f"Error starting task {task_id}: {e}")
            task.status = TaskStatus.FAILED
            # Publish status RUNNING after successful start/message
            try:
                from utils.event_bus_adapter import get_event_publisher
                await get_event_publisher().publish_lifecycle("task_updated", {"task_id": task_id, "status": TaskStatus.RUNNING.value})
                logger.info("TASK_EVENT %s", json.dumps({"event": "task_updated", "task_id": task_id, "status": TaskStatus.RUNNING.value}))
            except Exception:
                logger.debug("Event publish failed for RUNNING", exc_info=True)

            if not task.result:
                task.result = AgentTaskResult(
                    task_id=task_id,
                    agent_type=task.request.agent_type,
                    status=TaskStatus.FAILED,
                    started_at=datetime.now(timezone.utc),
                    error=str(e),
                )
            await self._store_task(task)
            return False

    async def process_streaming_response(
        self,
        task_id: str,
        chunk: str,
        is_final: bool = False
    ) -> Optional["AgentTaskResult"]:
        """Process streaming response chunk from an agent."""
        task = await self.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found for streaming response")
            return None

        try:
            # Process the streaming chunk
            parsed_response = await self.streaming_parser.process_streaming_chunk(
                task_id,
                chunk,
                task.request.agent_type.value,
                is_final
            )

            # If final response, update task result
            if is_final and parsed_response:
                AgentTask, AgentTaskRequest, AgentTaskResult, AgentType, TaskStatus = _get_agent_models()

                # Determine final status
                if parsed_response.status == "failed":
                    final_status = TaskStatus.FAILED
                    # A2A: publish status update for task events (best-effort)
                    try:
                        from utils.a2a import publish_task_event
                        await publish_task_event(task_id, "completed", {"status": final_status.value})
                    except Exception:
                        logger.debug("A2A publish failed", exc_info=True)
                elif parsed_response.status == "needs_input":
                    final_status = TaskStatus.PENDING  # Waiting for input
                else:
                    final_status = TaskStatus.COMPLETED

                # Create or update task result
                task.result = AgentTaskResult(
                    task_id=task_id,
                    agent_type=task.request.agent_type,
                    status=final_status,
                    started_at=task.created_at,
                    completed_at=datetime.now(timezone.utc)
                )

                # Append completion event
                await self._append_task_message(task_id, "completed", {
                    "status": final_status.value,
                    "summary": parsed_response.summary,
                    "files_created": parsed_response.files_created,
                    "files_modified": parsed_response.files_modified
                })

                # Publish terminal lifecycle events (NATS optional)
                try:
                    await self._publish_lifecycle(
                        "task_updated",
                        {"task_id": task_id, "status": final_status.value}
                    )
                    if final_status == TaskStatus.COMPLETED:
                        await self._publish_lifecycle("task_completed", {"task_id": task_id})
                    elif final_status == TaskStatus.FAILED:
                        await self._publish_lifecycle("task_failed", {"task_id": task_id})
                except Exception:
                    logger.debug("Lifecycle publish error", exc_info=True)

                task.result.output = parsed_response.raw_output
                task.result.summary = parsed_response.summary
                task.result.files_created = parsed_response.files_created
                task.result.files_modified = parsed_response.files_modified
                task.result.error = None if final_status != TaskStatus.FAILED else "Task failed"

                task.status = final_status
                task.updated_at = datetime.now(timezone.utc)
                await self._store_task(task)

                logger.info(f"Task {task_id} completed with status: {final_status}")
                return task.result

        except Exception as e:
            logger.error(f"Error processing streaming response for task {task_id}: {e}")

        return None

    async def _start_agent_server(self, task: "AgentTask") -> bool:
        """Start AgentAPI server for a task with robust error handling."""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                # Check if agentapi is available
                if not self._check_agentapi_available():
                    raise Exception("AgentAPI not found in PATH. Please install AgentAPI first.")

                # Check if agent command is available
                agent_cmd = self._get_agent_command(task.request.agent_type)
                if not self._check_agent_command_available(agent_cmd):
                    raise Exception(
                        f"Agent command '{agent_cmd}' not found. Please install {task.request.agent_type.value} first."
                    )

                # Build command to start AgentAPI server
                cmd = [
                    "agentapi",
                    "server",
                    "--port",
                    str(task.agent_port),
                    "--type",
                    task.request.agent_type.value,
                    "--",
                ]

                # Add agent command and args
                cmd.append(agent_cmd)
                cmd.extend(task.request.agent_args)

                # Set up environment with validation and effective PATH
                from utils.agent_defaults import build_effective_path_env
                env = os.environ.copy()
                env.update(task.request.env_vars)
                env["PATH"] = build_effective_path_env()

                # Validate required environment variables
                required_vars = self._get_required_env_vars(task.request.agent_type)
                missing_vars = [var for var in required_vars if not env.get(var)]
                if missing_vars:
                    raise Exception(f"Missing required environment variables: {', '.join(missing_vars)}")

                # Ensure working directory exists
                try:
                    os.makedirs(task.request.working_directory, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to create working directory {task.request.working_directory}: {e}")

                # Start the process with timeout
                # Stream stdout/stderr lines asynchronously
                async def _stream_pipe(pipe, kind: str):
                    try:
                        if not pipe:
                            return
                        loop = asyncio.get_event_loop()
                        while True:
                            line = await loop.run_in_executor(None, pipe.readline)
                            if not line:
                                break
                            text = line.rstrip("\n")
                            if text:
                                await self._append_task_message(task.task_id, kind, {"line": text})
                    except Exception:
                        logger.debug("pipe stream error", exc_info=True)

                process = subprocess.Popen(
                    cmd,
                    env=env,
                    cwd=task.request.working_directory,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                task.process_id = process.pid
                logger.debug(f"Started agent server process {process.pid} on port {task.agent_port}")

                # Wait for server to start with exponential backoff
                startup_timeout = 30  # 30 seconds max startup time
                check_interval = 1
                elapsed = 0

                while elapsed < startup_timeout:
                    await asyncio.sleep(check_interval)
                    elapsed += check_interval

                    # Check if process is still running
                    if process.poll() is not None:
                        stdout, stderr = process.communicate()
                        detail = f"Stdout: {stdout}\nStderr: {stderr}" if DIAGNOSTIC else "(run with AGENTAPI_DIAGNOSTIC=1 for details)"
                        raise Exception(f"Agent server process exited early. {detail}")

                    # Check if server is responding
                    if await self._check_agent_health(task.agent_port):
                        logger.info(f"Agent server started successfully on port {task.agent_port}")
                        return True

                    # Exponential backoff for health checks
                    check_interval = min(check_interval * 1.5, 5)

                # Startup timeout reached
                self._cleanup_failed_process(process)
                raise Exception(f"Agent server failed to start within {startup_timeout} seconds")

            except Exception as e:
                logger.warning(f"Agent server start attempt {attempt + 1}/{max_retries} failed: {e}")

                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    logger.error(f"Failed to start agent server after {max_retries} attempts: {e}")
                    return False

        return False

    def _get_agent_command(self, agent_type: "AgentType") -> str:
        """Get the command to start a specific agent type."""
        AgentTask, AgentTaskRequest, AgentTaskResult, AgentType, TaskStatus = _get_agent_models()
        commands = {
            AgentType.CLAUDE: "claude",
            AgentType.GOOSE: "goose",
            AgentType.AIDER: "aider",
            AgentType.CODEX: "codex",
            AgentType.GEMINI: "gemini",
            AgentType.AMP: "amp",
            AgentType.CURSOR_AGENT: "cursor-agent",
            AgentType.CURSOR: "cursor",
            AgentType.AUGGIE: "auggie",
        }
        return commands.get(agent_type, agent_type.value)

    def _check_agentapi_available(self) -> bool:
        """Check if AgentAPI is available in PATH, honoring ZEN_AGENT_PATHS."""
        import os
        import shutil

        # Temporarily augment PATH
        from utils.agent_defaults import build_effective_path_env
        original = os.environ.get("PATH", "")
        os.environ["PATH"] = build_effective_path_env()
        try:
            return shutil.which("agentapi") is not None
        finally:
            os.environ["PATH"] = original

    def _check_agent_command_available(self, command: str) -> bool:
        """Check if agent command is available in PATH, honoring ZEN_AGENT_PATHS."""
        import os
        import shutil

        from utils.agent_defaults import build_effective_path_env

        original = os.environ.get("PATH", "")
        os.environ["PATH"] = build_effective_path_env()
        try:
            return shutil.which(command) is not None
        finally:
            os.environ["PATH"] = original

    def _get_required_env_vars(self, agent_type: "AgentType") -> list[str]:
        """Get required environment variables for an agent type."""
        AgentTask, AgentTaskRequest, AgentTaskResult, AgentType, TaskStatus = _get_agent_models()
        required_vars = {
            AgentType.CLAUDE: ["ANTHROPIC_API_KEY"],
            AgentType.AIDER: ["ANTHROPIC_API_KEY"],  # or other model API keys
            AgentType.CODEX: ["OPENAI_API_KEY"],
            AgentType.GEMINI: ["GOOGLE_API_KEY"],
        }
        return required_vars.get(agent_type, [])

    def _cleanup_failed_process(self, process: subprocess.Popen) -> None:
        """Clean up a failed process."""
        try:
            if process.poll() is None:  # Process is still running
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        except Exception as e:
            logger.debug(f"Error cleaning up failed process: {e}")

    async def _check_agent_health(self, port: int) -> bool:
        """Check if agent server is healthy with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                    response = await client.get(f"http://localhost:{port}/status")
                    if response.status_code == 200:
                        data = response.json()
                        if DIAGNOSTIC:
                            logger.debug(f"Agent health payload: {data}")
                        # Validate response structure
                        if "status" in data and data["status"] in ["stable", "running"]:
                            return True
                    logger.debug(f"Health check attempt {attempt + 1}: HTTP {response.status_code}")
            except httpx.TimeoutException:
                logger.debug(f"Health check attempt {attempt + 1}: Timeout")
            except httpx.ConnectError:
                logger.debug(f"Health check attempt {attempt + 1}: Connection refused")
            except Exception as e:
                logger.debug(f"Health check attempt {attempt + 1}: {type(e).__name__}: {e}")

            if attempt < max_retries - 1:
                await asyncio.sleep(1)  # Brief delay between retries

        return False

    async def _send_message_to_agent(self, task: "AgentTask", message: str) -> bool:
        """Send a message to the agent with retry logic and validation."""
        if not message.strip():
            logger.error("Cannot send empty message to agent")
            return False

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                    # First check if agent is ready to receive messages
                    status_response = await client.get(f"http://localhost:{task.agent_port}/status")
                    if status_response.status_code != 200:
                        raise Exception(f"Agent status check failed: HTTP {status_response.status_code}")

                    status_data = status_response.json()
                    if status_data.get("status") == "running":
                        logger.warning(f"Agent is busy (status: running), attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(5)  # Wait longer if agent is busy
                            continue

                    # Send the message
                    response = await client.post(
                        f"http://localhost:{task.agent_port}/message",
                        json={"content": message, "type": "user"},
                    )

                    if response.status_code == 200:
                        logger.debug(f"Message sent successfully to agent on port {task.agent_port}")
                        return True
                    else:
                        logger.warning(f"Message send attempt {attempt + 1}: HTTP {response.status_code}")

            except httpx.TimeoutException:
                logger.warning(f"Message send attempt {attempt + 1}: Timeout")
            except httpx.ConnectError:
                logger.warning(f"Message send attempt {attempt + 1}: Connection refused")
            except Exception as e:
                logger.warning(f"Message send attempt {attempt + 1}: {type(e).__name__}: {e}")

            if attempt < max_retries - 1:
                await asyncio.sleep(2 * (attempt + 1))  # Exponential backoff

        logger.error(f"Failed to send message to agent after {max_retries} attempts")
        return False

    async def get_task(self, task_id: str) -> Optional["AgentTask"]:
        """Get a task by ID from memory or Redis (if enabled)."""
        # Check active tasks first
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        # Try loading from Redis (new/legacy keys)
        task = await self._load_task(task_id)
        if task:
            return task
        return None

    async def list_tasks(
        self,
        filter_agent: Optional[str] = None,
        filter_status: Optional[set[str]] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        sort: str = "created_at:desc",
        page: int = 1,
        limit: int = 50,
    ) -> tuple[int, list["AgentTask"]]:
        """List tasks from active memory plus Redis (if available).

        Returns (total, items) where items is the current page slice.
        Note: Redis results are limited by TTL set in _store_task.
        """
        # Start with active tasks
        tasks: list[AgentTask] = list(self.active_tasks.values())

        # Pull from Redis if available
        if self.redis_client:
            try:
                # Prefer ZSET index if present for efficient time-ordered listing
                zkey_new = "tasks:by_created_at"
                zkey_legacy = "agent_tasks_by_created_at"
                zkey = zkey_new if self.redis_client.exists(zkey_new) else zkey_legacy
                use_zset = False
                try:
                    use_zset = bool(self.redis_client.exists(zkey))
                except Exception:
                    use_zset = False
                seen = {t.task_id for t in tasks}
                if use_zset:
                    # Convert time filters to score ranges
                    min_score = created_after.timestamp() if created_after else "-inf"
                    max_score = created_before.timestamp() if created_before else "+inf"
                    # Sort direction
                    ids: list[str]
                    if sort == "created_at:asc":
                        ids = self.redis_client.zrangebyscore(zkey, min_score, max_score)  # type: ignore[arg-type]
                    else:
                        ids = self.redis_client.zrevrangebyscore(zkey, max_score, min_score)  # type: ignore[arg-type]
                    for tid in ids:
                        try:
                            if tid in seen:
                                continue
                            # Try new key then legacy
                            data = self.redis_client.get(f"task:{tid}") or self.redis_client.get(f"agent_task:{tid}")
                            if not data:
                                continue
                            task = AgentTask.model_validate_json(data)
                            tasks.append(task)
                            seen.add(tid)
                        except Exception:
                            continue
                else:
                    # Fallback: SCAN all keys (new and legacy)
                    cursor = 0
                    patterns = ["task:*", "agent_task:*"]
                    for pattern in patterns:
                        cursor = 0
                        while True:
                            cursor, keys = self.redis_client.scan(cursor=cursor, match=pattern, count=100)
                            for key in keys:
                                try:
                                    data = self.redis_client.get(key)
                                    if not data:
                                        continue
                                    task = AgentTask.model_validate_json(data)
                                    if task.task_id in seen:
                                        continue
                                    tasks.append(task)
                                    seen.add(task.task_id)
                                except Exception:
                                    continue
                            if cursor == 0:
                                break
            except Exception:
                # Ignore Redis listing errors
                pass

        # Apply created_at range filters first
        if created_after:
            tasks = [t for t in tasks if t.created_at >= created_after]
        if created_before:
            tasks = [t for t in tasks if t.created_at <= created_before]

        # Apply agent/status filters
        def include(t: "AgentTask") -> bool:
            if filter_agent and t.request.agent_type.value != filter_agent:
                return False
            if filter_status and t.status.value not in filter_status:
                return False
            return True

        filtered = [t for t in tasks if include(t)]

        # Sorting
        if sort == "created_at:asc":
            filtered.sort(key=lambda t: t.created_at)
        elif sort == "status":
            filtered.sort(key=lambda t: t.status.value)
        else:
            # default created_at:desc
            filtered.sort(key=lambda t: t.created_at, reverse=True)

        total = len(filtered)
        page = max(1, page)
        limit = max(1, min(200, limit))
        start = (page - 1) * limit
        end = start + limit
        return total, filtered[start:end]

    async def _append_task_message(self, task_id: str, event: str, data: dict) -> None:
        """Append a task message to Redis Stream if enabled, best-effort."""
        if not self.redis_client:
            return
        try:
            if self.task_store:
                await self.task_store.append_task_message(task_id, event, data)
                return
            stream_key = f"task:{task_id}:messages"
            fields = {
                "ts": int(datetime.now(timezone.utc).timestamp() * 1000),
                "event": event,
                "data": json.dumps(data),
            }
            # XADD + XTRIM MAXLEN ~1000 by default
            maxlen = int(os.getenv("TASK_MESSAGES_MAXLEN", "1000"))
            self.redis_client.xadd(stream_key, fields, maxlen=maxlen, approximate=True)  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Failed to append task message", exc_info=True)

    async def _store_task(self, task: "AgentTask") -> None:
        # Publish generic task_updated event when persisting
        try:
            from utils.event_bus_adapter import get_event_publisher
            await get_event_publisher().publish_lifecycle("task_updated", {
                "task_id": task.task_id,
                "status": task.status.value,
            })
            logger.info("TASK_EVENT %s", json.dumps({"event": "task_updated", "task_id": task.task_id, "status": task.status.value}))
        except Exception:
            logger.debug("Event publish failed for task_updated", exc_info=True)

        """Store task in persistent storage."""
        retention = int(os.getenv("AGENT_TASK_RETENTION_SEC", os.getenv("ZEN_AGENT_TASK_RETENTION_SEC", "3600")))
        if self.redis_client:
            try:
                if self.task_store:
                    await self.task_store.store_task(task, retention=retention)
                else:
                    key = f"task:{task.task_id}"
                    # Inject schema_version into blob without mutating model schema
                    data = json.loads(task.model_dump_json())
                    data["schema_version"] = 1
                    payload = json.dumps(data)
                    # Write main record with TTL
                    self.redis_client.setex(key, retention, payload)
                    # Maintain ZSET index by updated_at (prefer for inbox ordering)
                    try:
                        score = (task.updated_at or task.created_at).timestamp()
                        self.redis_client.zadd("inbox:status:" + task.status.value, {task.task_id: score})
                        # Also keep a created_at index for list view
                        self.redis_client.zadd("tasks:by_created_at", {task.task_id: task.created_at.timestamp()})
                        # Optionally trim to last N (keep memory bounded)
                        max_keep = int(os.getenv("TASK_INDEX_MAX", "5000"))
                        current = self.redis_client.zcard("tasks:by_created_at")
                        if current and int(current) > max_keep:
                            self.redis_client.zremrangebyrank("tasks:by_created_at", 0, int(current) - max_keep - 1)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Failed to store task in Redis: {e}")

    async def _load_task(self, task_id: str) -> Optional["AgentTask"]:
        """Load task from persistent storage (new + legacy keys)."""
        AgentTask, AgentTaskRequest, AgentTaskResult, AgentType, TaskStatus = _get_agent_models()
        if self.redis_client:
            try:
                if self.task_store:
                    data = await self.task_store.load_task_json(task_id)
                    if data:
                        return AgentTask.model_validate_json(data)
                else:
                    for key in (f"task:{task_id}", f"agent_task:{task_id}"):
                        data = self.redis_client.get(key)
                        if data:
                            return AgentTask.model_validate_json(data)
            except Exception as e:
                logger.warning(f"Failed to load task from Redis: {e}")
        return None

    async def cleanup_completed_tasks(self) -> None:
        """Clean up completed tasks and release resources."""
        # Import runtime models for status checks
        AgentTask, AgentTaskRequest, AgentTaskResult, AgentType, TaskStatus = _get_agent_models()
        async with self._cleanup_lock:
            completed_tasks = []
            for task_id, task in self.active_tasks.items():
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED]:
                    completed_tasks.append(task_id)

            now = datetime.now(timezone.utc)
            for task_id in completed_tasks:
                task = self.active_tasks.get(task_id)
                if not task:
                    continue
                # Retain task for retention_seconds before purging from memory
                age = (now - (task.updated_at or task.created_at)).total_seconds()
                if age < self.retention_seconds:
                    continue
                task = self.active_tasks.pop(task_id)
                if task.agent_port:
                    self._release_port(task.agent_port)
                logger.debug(f"Cleaned up completed task {task_id}")

    async def _dequeue_loop(self) -> None:
        """Background loop to start queued tasks respecting concurrency limits."""
        while True:
            try:
                task_id = await self._queue.get()
                async with self._sem:
                    try:
                        await self.start_task(task_id)
                    finally:
                        self._queue.task_done()
            except asyncio.CancelledError:  # graceful exit on shutdown
                break
            except Exception as e:
                logger.error(f"Dequeuer error: {e}")
                await asyncio.sleep(0.5)


# Global task manager instance
_task_manager: Optional[AgentTaskManager] = None


def get_task_manager() -> AgentTaskManager:
    """Get the global task manager instance, refreshing dynamic settings from env."""
    global _task_manager
    if _task_manager is None:
        _task_manager = AgentTaskManager()
    else:
        # Refresh dynamic retention from environment for tests and live tuning
        try:
            _task_manager.retention_seconds = int(os.getenv("AGENT_TASK_RETENTION_SEC", str(_task_manager.retention_seconds)))
        except Exception:
            pass
        return _task_manager

    async def cancel_task(self, task_id: str) -> bool:
        """Best-effort cancel for a running or queued task.

        Marks task as CANCELLED if not in a terminal state. Attempts to release resources.
        """
        AgentTask, AgentTaskRequest, AgentTaskResult, AgentType, TaskStatus = _get_agent_models()
        task = await self.get_task(task_id)
        if not task:
            # Try to load from Redis to update persisted state
            task = await self._load_task(task_id)
            if not task:
                return False
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED]:
            return True  # already terminal

        # Mark cancelled
        task.status = TaskStatus.CANCELLED
        task.updated_at = datetime.now(timezone.utc)
        if not task.result:
            task.result = AgentTaskResult(
                task_id=task.task_id,
                agent_type=task.request.agent_type,
                status=TaskStatus.CANCELLED,
                started_at=task.created_at,
                completed_at=datetime.now(timezone.utc),
                output="",
                error="cancelled",
            )

        # Broadcast streaming update
        try:
            await self.streaming_manager.broadcast_message(
                task.task_id,
                StreamMessageType.STATUS_UPDATE,
                {"status": TaskStatus.CANCELLED.value, "message": "Task cancelled"},
                task.request.agent_type.value,
            )
        except Exception:
            pass

        await self._store_task(task)
        # Release port if allocated
        if task.agent_port:
            self._release_port(task.agent_port)
        return True
