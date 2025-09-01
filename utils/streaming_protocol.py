"""
Streaming Communication Protocol for Real-time Agent Updates.

This module provides WebSocket and Server-Sent Events (SSE) support for real-time
communication between agents and the orchestration system, enabling progressive
status updates, streaming responses, and bidirectional communication.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

# Optional WebSocket dependency
# Optional message tap callback for global side effects (e.g., persistence)
_message_tap = None  # type: Optional[callable]

def register_message_tap(callback):
    global _message_tap
    _message_tap = callback

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    websockets = None
    WebSocketServerProtocol = None
    WEBSOCKETS_AVAILABLE = False

# Optional FastAPI/Starlette for SSE
try:
    from starlette.responses import StreamingResponse
    SSE_AVAILABLE = True
except ImportError:
    StreamingResponse = None
    SSE_AVAILABLE = False

from utils.agent_prompts import AgentResponse, AgentResponseParser  # noqa: E402

logger = logging.getLogger(__name__)


class StreamMessageType(Enum):
    """Types of streaming messages."""
    STATUS_UPDATE = "status_update"
    PROGRESS_UPDATE = "progress_update"
    ACTIVITY_UPDATE = "activity_update"
    ACTION_UPDATE = "action_update"
    FILE_UPDATE = "file_update"
    QUESTION_UPDATE = "question_update"
    WARNING = "warning"
    ERROR = "error"
    COMPLETION = "completion"
    HEARTBEAT = "heartbeat"


@dataclass
class StreamMessage:
    """Individual streaming message."""
    id: str
    type: StreamMessageType
    timestamp: str
    task_id: str
    agent_type: str
    content: dict[str, Any]
    sequence: int = 0

    def to_json(self) -> str:
        """Convert to JSON string for transmission."""
        return json.dumps(asdict(self))

    def to_sse_format(self) -> str:
        """Convert to Server-Sent Events format."""
        return f"id: {self.id}\nevent: {self.type.value}\ndata: {json.dumps(self.content)}\n\n"


class StreamingManager:
    """Manages streaming connections and message distribution."""

    def __init__(self):
        self.websocket_connections: dict[str, WebSocketServerProtocol] = {}
        self.sse_connections: dict[str, asyncio.Queue] = {}
        self.task_subscribers: dict[str, set[str]] = {}  # task_id -> set of connection_ids
        self.global_subscribers: set[str] = set()  # connection_ids receiving all task events
        self.connection_metadata: dict[str, dict[str, Any]] = {}
        self.message_sequence: dict[str, int] = {}  # task_id -> sequence counter
        # Ring buffer of recent messages per task for lightweight catch-up
        self.recent_messages: dict[str, list[StreamMessage]] = {}
        self.recent_capacity: int = 50  # keep last 50 per task


    async def register_websocket_connection(
        self,
        connection_id: str,
        websocket: WebSocketServerProtocol,
        task_ids: Optional[list[str]] = None
    ) -> None:
        """Register a new WebSocket connection."""
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError("WebSockets not available - install websockets package")

        self.websocket_connections[connection_id] = websocket
        self.connection_metadata[connection_id] = {
            "type": "websocket",
            "connected_at": datetime.now(timezone.utc).isoformat(),
            "task_ids": task_ids or []
        }

        # Subscribe to tasks or global
        if task_ids:
            for task_id in task_ids:
                if task_id not in self.task_subscribers:
                    self.task_subscribers[task_id] = set()
                self.task_subscribers[task_id].add(connection_id)
        else:
            # Global subscription to all task events
            self.global_subscribers.add(connection_id)

        logger.info(f"WebSocket connection {connection_id} registered for tasks: {task_ids}")

    async def register_sse_connection(
        self,
        connection_id: str,
        task_ids: Optional[list[str]] = None
    ) -> asyncio.Queue:
        """Register a new SSE connection and return its message queue."""
        # Aiohttp-based SSE does not require starlette; proceed regardless
        queue = asyncio.Queue()
        self.sse_connections[connection_id] = queue
        self.connection_metadata[connection_id] = {
            "type": "sse",
            "connected_at": datetime.now(timezone.utc).isoformat(),
            "task_ids": task_ids or []
        }

        # Subscribe to tasks or global
        if task_ids:
            for task_id in task_ids:
                if task_id not in self.task_subscribers:
                    self.task_subscribers[task_id] = set()
                self.task_subscribers[task_id].add(connection_id)
        else:
            self.global_subscribers.add(connection_id)

        logger.info(f"SSE connection {connection_id} registered for tasks: {task_ids}")
        return queue

    async def unregister_connection(self, connection_id: str) -> None:
        """Unregister a connection."""
        # Remove from all task subscriptions
        for _task_id, subscribers in self.task_subscribers.items():
            subscribers.discard(connection_id)
        # Remove from global subscribers
        self.global_subscribers.discard(connection_id)

        # Clean up empty subscriptions
        empty_tasks = [task_id for task_id, subs in self.task_subscribers.items() if not subs]
        for task_id in empty_tasks:
            del self.task_subscribers[task_id]

        # Remove connection
        self.websocket_connections.pop(connection_id, None)
        self.sse_connections.pop(connection_id, None)
        self.connection_metadata.pop(connection_id, None)

        logger.info(f"Connection {connection_id} unregistered")

    async def broadcast_message(
        self,
        task_id: str,
        message_type: StreamMessageType,
        content: dict[str, Any],
        agent_type: str = "unknown"
    ) -> None:
        """Broadcast a message to all subscribers of a task."""
        has_task_subs = task_id in self.task_subscribers and bool(self.task_subscribers[task_id])
        has_global_subs = bool(self.global_subscribers)
        if not has_task_subs and not has_global_subs:
            return

        # Get next sequence number for this task
        if task_id not in self.message_sequence:
            self.message_sequence[task_id] = 0
        self.message_sequence[task_id] += 1

        # Create stream message
        message = StreamMessage(
            id=str(uuid.uuid4()),
            type=message_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            task_id=task_id,
            agent_type=agent_type,
            content=content,
            sequence=self.message_sequence[task_id]
        )

        # Optional global tap (best-effort; don't block path)
        try:
            if _message_tap:
                await _message_tap(task_id, message_type, message.content)
        except Exception:
            logger.debug("message_tap failed", exc_info=True)

        # Send to all subscribers
        subscribers = self.task_subscribers.get(task_id, set()).copy()
        for connection_id in subscribers:
            try:
                if connection_id in self.websocket_connections:
                    await self._send_websocket_message(connection_id, message)
                elif connection_id in self.sse_connections:
                    await self._send_sse_message(connection_id, message)
            except Exception as e:
                logger.warning(f"Failed to send message to connection {connection_id}: {e}")
                await self.unregister_connection(connection_id)
        globals_copy = self.global_subscribers.copy()
        for connection_id in globals_copy:
            try:
                if connection_id in self.websocket_connections:
                    await self._send_websocket_message(connection_id, message)
                elif connection_id in self.sse_connections:
                    await self._send_sse_message(connection_id, message)
            except Exception as e:
                logger.warning(f"Failed to send message to global connection {connection_id}: {e}")
                await self.unregister_connection(connection_id)

    async def _send_websocket_message(self, connection_id: str, message: StreamMessage) -> None:
        """Send message via WebSocket."""
        websocket = self.websocket_connections[connection_id]
        await websocket.send(message.to_json())

        # Save to ring buffer for catch-up
        if message.task_id not in self.recent_messages:
            self.recent_messages[message.task_id] = []
        buf = self.recent_messages[message.task_id]
        buf.append(message)
        if len(buf) > self.recent_capacity:
            del buf[: len(buf) - self.recent_capacity]

    async def _send_sse_message(self, connection_id: str, message: StreamMessage) -> None:
        """Send message via SSE queue."""
        queue = self.sse_connections[connection_id]
        await queue.put(message)
        # Also mirror to ring buffer
        if message.task_id not in self.recent_messages:
            self.recent_messages[message.task_id] = []
        buf = self.recent_messages[message.task_id]
        buf.append(message)
        if len(buf) > self.recent_capacity:
            del buf[: len(buf) - self.recent_capacity]

    async def get_recent_messages(self, task_id: str) -> list[StreamMessage]:
        return list(self.recent_messages.get(task_id, []))


    async def send_heartbeat(self, task_id: str) -> None:
        """Send heartbeat to keep connections alive."""
        await self.broadcast_message(
            task_id,
            StreamMessageType.HEARTBEAT,
            {"timestamp": datetime.now(timezone.utc).isoformat()}
        )

    def get_connection_stats(self) -> dict[str, Any]:
        """Get statistics about active connections."""
        return {
            "websocket_connections": len(self.websocket_connections),
            "sse_connections": len(self.sse_connections),
            "total_connections": len(self.websocket_connections) + len(self.sse_connections),
            "active_tasks": len(self.task_subscribers),
            "connections_by_task": {
                task_id: len(subs) for task_id, subs in self.task_subscribers.items()
            },
        }

    def get_buffer_stats(self) -> dict[str, Any]:
        try:
            # approximate: average buffer size and total tracked tasks
            sizes = [len(v) for v in self.recent_messages.values()] or [0]
            avg = sum(sizes) / max(1, len(sizes))
            return {
                "tracked_tasks": len(self.recent_messages),
                "avg_buffer": avg,
                "capacity": self.recent_capacity,
            }
        except Exception:
            return {"tracked_tasks": 0, "avg_buffer": 0, "capacity": self.recent_capacity}


class StreamingResponseParser:
    """Parses streaming agent responses for progressive updates."""

    def __init__(self, streaming_manager: StreamingManager):
        self.streaming_manager = streaming_manager
        self.partial_responses: dict[str, str] = {}  # task_id -> accumulated response

    async def process_streaming_chunk(
        self,
        task_id: str,
        chunk: str,
        agent_type: str = "unknown",
        is_final: bool = False
    ) -> Optional[AgentResponse]:
        """Process a streaming chunk from an agent."""
        # Accumulate response
        if task_id not in self.partial_responses:
            self.partial_responses[task_id] = ""
        self.partial_responses[task_id] += chunk

        # Try to parse any complete XML tags
        await self._extract_and_broadcast_updates(task_id, agent_type)

        # If final chunk, parse complete response
        if is_final:
            complete_response = self.partial_responses.pop(task_id, "")
            if complete_response:
                parsed_response = AgentResponseParser.parse_response(complete_response)

                # Send completion message
                await self.streaming_manager.broadcast_message(
                    task_id,
                    StreamMessageType.COMPLETION,
                    {
                        "status": parsed_response.status,
                        "summary": parsed_response.summary,
                        "files_created": parsed_response.files_created,
                        "files_modified": parsed_response.files_modified,
                        "questions": parsed_response.questions
                    },
                    agent_type
                )

                return parsed_response

        return None

    async def _extract_and_broadcast_updates(self, task_id: str, agent_type: str) -> None:
        """Extract and broadcast real-time updates from partial response."""
        content = self.partial_responses[task_id]

        # Look for complete XML tags and broadcast updates
        updates = [
            ("STATUS", StreamMessageType.STATUS_UPDATE, r'<STATUS>(.*?)</STATUS>'),
            ("PROGRESS", StreamMessageType.PROGRESS_UPDATE, r'<PROGRESS>(.*?)</PROGRESS>'),
            ("CURRENT_ACTIVITY", StreamMessageType.ACTIVITY_UPDATE, r'<CURRENT_ACTIVITY>(.*?)</CURRENT_ACTIVITY>'),
            ("ACTIONS_COMPLETED", StreamMessageType.ACTION_UPDATE, r'<ACTIONS_COMPLETED>(.*?)</ACTIONS_COMPLETED>'),
            ("ACTIONS_IN_PROGRESS", StreamMessageType.ACTION_UPDATE, r'<ACTIONS_IN_PROGRESS>(.*?)</ACTIONS_IN_PROGRESS>'),
            ("FILES_CREATED", StreamMessageType.FILE_UPDATE, r'<FILES_CREATED>(.*?)</FILES_CREATED>'),
            ("FILES_MODIFIED", StreamMessageType.FILE_UPDATE, r'<FILES_MODIFIED>(.*?)</FILES_MODIFIED>'),
            ("WARNINGS", StreamMessageType.WARNING, r'<WARNINGS>(.*?)</WARNINGS>'),
        ]

        import re
        for tag_name, message_type, pattern in updates:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                # Use the last (most recent) match
                latest_match = matches[-1].strip()
                if latest_match:
                    await self.streaming_manager.broadcast_message(
                        task_id,
                        message_type,
                        {
                            "tag": tag_name,
                            "content": latest_match,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        },
                        agent_type
                    )


class InputTransformationPipeline:
    """Transforms user input into enhanced agent instructions with intent recognition."""

    def __init__(self):
        self.intent_patterns = {
            "implement_feature": [
                r"add\s+.*(?:feature|functionality)",
                r"implement\s+.*",
                r"create\s+.*(?:component|module|function)",
                r"build\s+.*"
            ],
            "fix_bug": [
                r"fix\s+.*(?:bug|error|issue|problem)",
                r"resolve\s+.*(?:issue|error)",
                r"debug\s+.*",
                r"troubleshoot\s+.*"
            ],
            "refactor_code": [
                r"refactor\s+.*",
                r"improve\s+.*(?:code|structure)",
                r"optimize\s+.*",
                r"clean\s+up\s+.*"
            ],
            "analyze_code": [
                r"analyze\s+.*",
                r"review\s+.*(?:code|implementation)",
                r"examine\s+.*",
                r"understand\s+.*"
            ],
            "test_code": [
                r"test\s+.*",
                r"add\s+.*tests",
                r"write\s+.*(?:test|spec)",
                r"validate\s+.*"
            ],
            "document_code": [
                r"document\s+.*",
                r"add\s+.*(?:documentation|comments)",
                r"write\s+.*(?:docs|readme)",
                r"explain\s+.*"
            ]
        }

        self.complexity_indicators = {
            "simple": ["simple", "basic", "quick", "small", "minor"],
            "medium": ["moderate", "standard", "normal", "typical"],
            "complex": ["complex", "advanced", "comprehensive", "large", "major", "full"]
        }

        self.priority_indicators = {
            "urgent": ["urgent", "asap", "immediately", "critical", "emergency"],
            "high": ["high", "important", "priority", "soon"],
            "medium": ["medium", "normal", "standard"],
            "low": ["low", "when possible", "eventually", "later"]
        }

    def transform_input(self, user_input: str, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Transform user input into enhanced agent instructions."""

        # Extract intent
        intent = self._extract_intent(user_input)

        # Extract complexity and priority signals
        complexity = self._extract_complexity(user_input)
        priority = self._extract_priority(user_input)

        # Extract constraints and preferences
        constraints = self._extract_constraints(user_input)
        preferences = self._extract_preferences(user_input)

        # Generate success criteria
        success_criteria = self._generate_success_criteria(user_input, intent)

        # Create enhanced input structure
        enhanced_input = {
            "original_message": user_input,
            "intent": intent,
            "complexity": complexity,
            "priority": priority,
            "context": context or {},
            "constraints": constraints,
            "preferences": preferences,
            "success_criteria": success_criteria,
            "implied_tasks": self._generate_implied_tasks(user_input, intent),
            "enhanced_prompt": self._build_enhanced_prompt(user_input, intent, complexity, priority)
        }

        return enhanced_input

    def _extract_intent(self, user_input: str) -> str:
        """Extract primary intent from user input."""
        import re
        user_input_lower = user_input.lower()

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    return intent

        return "general_task"

    def _extract_complexity(self, user_input: str) -> str:
        """Extract complexity level from user input."""
        user_input_lower = user_input.lower()

        for level, indicators in self.complexity_indicators.items():
            if any(indicator in user_input_lower for indicator in indicators):
                return level

        # Default complexity based on content length and technical terms
        if len(user_input) > 200 or any(term in user_input_lower for term in ["comprehensive", "enterprise", "scalable"]):
            return "complex"
        elif len(user_input) < 50:
            return "simple"
        else:
            return "medium"

    def _extract_priority(self, user_input: str) -> str:
        """Extract priority level from user input."""
        user_input_lower = user_input.lower()

        for level, indicators in self.priority_indicators.items():
            if any(indicator in user_input_lower for indicator in indicators):
                return level

        return "medium"

    def _extract_constraints(self, user_input: str) -> list[str]:
        """Extract constraints from user input."""
        import re
        constraints = []

        # Look for constraint patterns
        constraint_patterns = [
            r"don't\s+.*",
            r"avoid\s+.*",
            r"without\s+.*",
            r"must not\s+.*",
            r"cannot\s+.*"
        ]

        for pattern in constraint_patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            constraints.extend([match.strip() for match in matches])

        return constraints

    def _extract_preferences(self, user_input: str) -> list[str]:
        """Extract preferences from user input."""
        import re
        preferences = []

        # Look for preference patterns
        preference_patterns = [
            r"prefer\s+.*",
            r"would like\s+.*",
            r"should use\s+.*",
            r"ideally\s+.*"
        ]

        for pattern in preference_patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            preferences.extend([match.strip() for match in matches])

        return preferences

    def _generate_success_criteria(self, user_input: str, intent: str) -> list[str]:
        """Generate success criteria based on input and intent."""
        criteria = []

        # Intent-based criteria
        if intent == "implement_feature":
            criteria.extend([
                "Feature is implemented and functional",
                "Code follows project standards",
                "Tests pass"
            ])
        elif intent == "fix_bug":
            criteria.extend([
                "Bug is resolved",
                "No regression introduced",
                "Tests validate the fix"
            ])
        elif intent == "refactor_code":
            criteria.extend([
                "Code is cleaner and more maintainable",
                "Functionality remains unchanged",
                "Performance is maintained or improved"
            ])
        elif intent == "test_code":
            criteria.extend([
                "Comprehensive tests are written",
                "All tests pass",
                "Good test coverage achieved"
            ])

        # Add general criteria
        criteria.append("Task completed successfully")

        return criteria

    def _generate_implied_tasks(self, user_input: str, intent: str) -> list[str]:
        """Generate implied tasks based on input and intent."""
        tasks = []

        if intent == "implement_feature":
            tasks.extend([
                "Analyze requirements",
                "Design solution",
                "Implement code",
                "Add tests",
                "Update documentation"
            ])
        elif intent == "fix_bug":
            tasks.extend([
                "Reproduce the issue",
                "Identify root cause",
                "Implement fix",
                "Test the fix",
                "Verify no regression"
            ])
        elif intent == "refactor_code":
            tasks.extend([
                "Analyze current code structure",
                "Plan refactoring approach",
                "Refactor in small increments",
                "Run tests after each change",
                "Verify functionality"
            ])

        return tasks

    def _build_enhanced_prompt(self, original: str, intent: str, complexity: str, priority: str) -> str:
        """Build enhanced prompt with context and guidance."""
        prompt_parts = [
            f"Intent: {intent}",
            f"Complexity: {complexity}",
            f"Priority: {priority}",
            "",
            "Task Requirements:",
            original,
            "",
            "Please provide structured updates using the XML communication protocol."
        ]

        return "\n".join(prompt_parts)


# Global streaming manager instance
_streaming_manager: Optional[StreamingManager] = None


def get_streaming_manager() -> StreamingManager:
    """Get the global streaming manager instance."""
    global _streaming_manager
    if _streaming_manager is None:
        _streaming_manager = StreamingManager()
    return _streaming_manager


async def create_sse_stream(task_id: str) -> AsyncGenerator[str, None]:
    """Create a Server-Sent Events stream for a task."""
    if not SSE_AVAILABLE:
        raise RuntimeError("SSE not available - install starlette/fastapi package")

    connection_id = str(uuid.uuid4())
    streaming_manager = get_streaming_manager()

    try:
        # Register SSE connection
        queue = await streaming_manager.register_sse_connection(connection_id, [task_id])

        # Send initial connection message
        initial_message = StreamMessage(
            id=str(uuid.uuid4()),
            type=StreamMessageType.STATUS_UPDATE,
            timestamp=datetime.now(timezone.utc).isoformat(),
            task_id=task_id,
            agent_type="system",
            content={"status": "connected", "message": "Stream started"}
        )
        yield initial_message.to_sse_format()

        # Stream messages from queue
        while True:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield message.to_sse_format()

                # Break on completion
                if message.type == StreamMessageType.COMPLETION:
                    break

            except asyncio.TimeoutError:
                # Send heartbeat
                heartbeat = StreamMessage(
                    id=str(uuid.uuid4()),
                    type=StreamMessageType.HEARTBEAT,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    task_id=task_id,
                    agent_type="system",
                    content={"status": "alive"}
                )
                yield heartbeat.to_sse_format()

    except asyncio.CancelledError:
        # Client disconnected
        pass
    except Exception as e:
        # Send error message
        error_message = StreamMessage(
            id=str(uuid.uuid4()),
            type=StreamMessageType.ERROR,
            timestamp=datetime.now(timezone.utc).isoformat(),
            task_id=task_id,
            agent_type="system",
            content={"error": str(e)}
        )
        yield error_message.to_sse_format()
    finally:
        # Unregister connection
        await streaming_manager.unregister_connection(connection_id)


async def handle_websocket_connection(websocket: WebSocketServerProtocol, task_ids: list[str]) -> None:
    """Handle a WebSocket connection for streaming updates."""
    if not WEBSOCKETS_AVAILABLE:
        raise RuntimeError("WebSockets not available - install websockets package")

    connection_id = str(uuid.uuid4())
    streaming_manager = get_streaming_manager()

    try:
        # Register WebSocket connection
        await streaming_manager.register_websocket_connection(connection_id, websocket, task_ids)

        # Send initial connection message
        await websocket.send(json.dumps({
            "type": "connection",
            "connection_id": connection_id,
            "task_ids": task_ids,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))

        # Keep connection alive and handle incoming messages
        async for message in websocket:
            try:
                data = json.loads(message)
                # Handle client messages (ping/pong, subscriptions, etc.)
                if data.get("type") == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))
                elif data.get("type") == "subscribe":
                    # Add subscription to new task
                    new_task_id = data.get("task_id")
                    if new_task_id:
                        if new_task_id not in streaming_manager.task_subscribers:
                            streaming_manager.task_subscribers[new_task_id] = set()
                        streaming_manager.task_subscribers[new_task_id].add(connection_id)
            except json.JSONDecodeError:
                # Invalid message, ignore
                pass

    except websockets.exceptions.ConnectionClosed:
        # Client disconnected
        pass
    except Exception as e:
        logger.error(f"WebSocket error for connection {connection_id}: {e}")
    finally:
        # Unregister connection
        await streaming_manager.unregister_connection(connection_id)
