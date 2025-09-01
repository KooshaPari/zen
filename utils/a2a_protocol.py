"""
Agent-to-Agent (A2A) Protocol Implementation

This module provides the core infrastructure for Agent-to-Agent communication,
enabling cross-platform interoperability, agent discovery, and capability advertising.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import httpx
from pydantic import BaseModel

try:
    import redis
except ImportError:
    redis = None


logger = logging.getLogger(__name__)


class A2AMessageType(str, Enum):
    """Types of A2A protocol messages."""

    DISCOVER = "discover"
    ADVERTISE = "advertise"
    CAPABILITY_REQUEST = "capability_request"
    CAPABILITY_RESPONSE = "capability_response"
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_STATUS = "task_status"
    CHAT_REQUEST = "chat_request"
    CHAT_RESPONSE = "chat_response"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class AgentCapability(BaseModel):
    """Agent capability descriptor."""

    name: str
    description: str
    category: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    max_concurrent: int = 1
    timeout_seconds: int = 300
    cost_estimate: float = 0.0  # Estimated cost/complexity
    quality_rating: float = 1.0  # Agent's self-assessed quality for this capability


class AgentCard(BaseModel):
    """Agent identification and capability card."""

    agent_id: str
    name: str
    version: str
    organization: Optional[str] = None
    endpoint_url: str
    supported_protocols: list[str] = ["http", "websocket"]
    capabilities: list[AgentCapability]
    max_concurrent_tasks: int = 10
    availability_schedule: Optional[dict[str, Any]] = None  # Future: scheduling support
    authentication_required: bool = False
    cost_model: str = "free"  # free, per_task, per_minute, etc.
    last_seen: datetime
    health_status: str = "healthy"  # healthy, degraded, unavailable
    metadata: dict[str, Any] = {}


class A2AMessage(BaseModel):
    """A2A protocol message structure."""

    message_id: str
    sender_id: str
    receiver_id: Optional[str] = None  # None for broadcasts
    message_type: A2AMessageType
    timestamp: datetime
    payload: dict[str, Any]
    correlation_id: Optional[str] = None  # For request/response correlation
    ttl_seconds: int = 3600  # Time to live
    priority: int = 5  # 1=highest, 10=lowest
    reply_to: Optional[str] = None  # Endpoint for replies


class A2AProtocolManager:
    """Manages A2A protocol operations including discovery, communication, and registry."""

    def __init__(self,
                 agent_id: str = None,
                 registry_endpoints: list[str] = None,
                 redis_client = None):
        """Initialize A2A protocol manager."""
        self.agent_id = agent_id or f"zen-agent-{uuid.uuid4().hex[:8]}"
        self.registry_endpoints = registry_endpoints or [
            "http://localhost:8080",  # Local registry
            "https://a2a-registry.example.com"  # Global registry (example)
        ]

        self.redis_client = redis_client or self._get_redis_client()
        self.local_registry: dict[str, AgentCard] = {}
        self.message_handlers: dict[A2AMessageType, callable] = {}
        self.pending_requests: dict[str, asyncio.Event] = {}
        self.response_cache: dict[str, A2AMessage] = {}

        # NATS integration
        self.loop = asyncio.get_event_loop()
        self.nats = None
        self._nats_started = False
        self.use_nats = os.getenv("A2A_ENABLED", "0").lower() in ("1", "true", "yes") and os.getenv("ZEN_EVENT_BUS", os.getenv("ZEN_EVENTS", "inline")).lower() == "nats"
        self._pending_futures: dict[str, asyncio.Future] = {}

        # Agent card for this instance
        self.my_card: Optional[AgentCard] = None
        self._setup_default_handlers()
        # Counters
        self.counters: dict[str, int] = {
            "messages_in": 0,
            "messages_out": 0,
            "requests": 0,
            "responses": 0,
            "timeouts": 0,
        }


    def _get_redis_client(self):
        """Get Redis client for distributed registry."""
        try:
            client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "2")),  # Use DB 2 for A2A protocol
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            client.ping()
            logger.debug("Connected to Redis for A2A protocol")
            return client
        except Exception as e:
            logger.warning(f"Redis not available for A2A protocol: {e}")
            return None

    async def _ensure_nats(self):
        if self._nats_started or not self.use_nats:
            return
        try:
            from utils.nats_communicator import get_nats_communicator
            self.nats = await get_nats_communicator(None)
            self._nats_started = True
            # Subscriptions for in/out
            await self.nats.subscribe("discovery.advertise", self._on_nats_discovery)
            await self.nats.subscribe(f"a2a.agent.{self.agent_id}.in", self._on_nats_in)
            await self.nats.subscribe(f"a2a.agent.{self.agent_id}.out", self._on_nats_out)
            await self.nats.subscribe("a2a.task.*.events", self._on_nats_event, use_jetstream=True, durable_name=f"a2a-{self.agent_id}-events")
            logger.info("A2A NATS subscriptions active")
        except Exception as e:
            logger.warning(f"A2A NATS init failed: {e}")

    def _complete_future(self, corr_id: str, msg: dict[str, Any]):
        fut = self._pending_futures.pop(corr_id, None)
        if fut and not fut.done():
            fut.set_result(msg)

    async def _on_nats_in(self, data: dict[str, Any]):
        # Incoming direct message; if handler returns a response message, publish to sender out
        try:
            self.counters["messages_in"] += 1
            resp = await self.handle_incoming_message(data)
            if resp and self.use_nats and self.nats:
                try:
                    sender = data.get("sender_id")
                    if sender:
                        await self.nats.publish(f"a2a.agent.{sender}.out", resp, use_jetstream=True)
                        self.counters["messages_out"] += 1
                        self.counters["responses"] += 1
                except Exception as e:
                    logger.debug(f"A2A NATS reply failed: {e}")
        except Exception as e:
            logger.error(f"A2A NATS in handler error: {e}")

    async def _on_nats_out(self, data: dict[str, Any]):
        # Responses destined to this agent (optional usage)
        try:
            corr = data.get("correlation_id")
            if corr:
                payload = data.get("payload", data)
                self._complete_future(corr, payload)
        except Exception as e:
            logger.error(f"A2A NATS out handler error: {e}")

    async def _on_nats_discovery(self, data: dict[str, Any]):
        try:
            card = data.get("agent_card")
            if card:
                agent = AgentCard.model_validate(card)
                self.local_registry[agent.agent_id] = agent
        except Exception as e:
            logger.debug(f"A2A discovery ingest failed: {e}")

    async def _on_nats_event(self, data: dict[str, Any]):
        # Task or general events
        try:
            corr = data.get("correlation_id")
            if corr:
                self._complete_future(corr, data)
        except Exception as e:
            logger.error(f"A2A NATS event handler error: {e}")

    def _setup_default_handlers(self):
        """Setup default message handlers."""
        self.message_handlers[A2AMessageType.DISCOVER] = self._handle_discover
        self.message_handlers[A2AMessageType.ADVERTISE] = self._handle_advertise
        self.message_handlers[A2AMessageType.CAPABILITY_REQUEST] = self._handle_capability_request
        self.message_handlers[A2AMessageType.CAPABILITY_RESPONSE] = self._handle_capability_response
        self.message_handlers[A2AMessageType.CHAT_REQUEST] = self._handle_chat_request
        self.message_handlers[A2AMessageType.CHAT_RESPONSE] = self._handle_chat_response
        self.message_handlers[A2AMessageType.TASK_REQUEST] = self._handle_task_request
        self.message_handlers[A2AMessageType.TASK_RESPONSE] = self._handle_task_response
        self.message_handlers[A2AMessageType.TASK_STATUS] = self._handle_task_status
        self.message_handlers[A2AMessageType.HEARTBEAT] = self._handle_heartbeat
        self.message_handlers[A2AMessageType.ERROR] = self._handle_error

    async def initialize_agent_card(self,
                                    name: str,
                                    version: str,
                                    endpoint_url: str,
                                    capabilities: list[AgentCapability],
                                    organization: str = None,
                                    **kwargs):
        """Initialize this agent's card for advertising."""
        self.my_card = AgentCard(
            agent_id=self.agent_id,
            name=name,
            version=version,
            organization=organization,
            endpoint_url=endpoint_url,
            capabilities=capabilities,
            last_seen=datetime.now(timezone.utc),
            **kwargs
        )

        # Register in local registry
        self.local_registry[self.agent_id] = self.my_card

        # Advertise to remote registries
        await self.advertise_capabilities()

        logger.info(f"Initialized A2A agent card: {name} ({self.agent_id})")

    async def discover_agents(self,
                              capability_filter: str = None,
                              organization_filter: str = None,
                              max_results: int = 50) -> list[AgentCard]:
        """Discover available agents with optional filtering."""

        # Send discovery request
        message = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            message_type=A2AMessageType.DISCOVER,
            timestamp=datetime.now(timezone.utc),
            payload={
                "capability_filter": capability_filter,
                "organization_filter": organization_filter,
                "max_results": max_results
            }
        )

        discovered_agents = []

        # Check local registry first
        for agent_card in self.local_registry.values():
            if self._matches_discovery_filter(agent_card, capability_filter, organization_filter):
                discovered_agents.append(agent_card)

        # Query Redis registry
        if self.redis_client:
            try:
                redis_agents = await self._discover_from_redis(capability_filter, organization_filter, max_results)
                discovered_agents.extend(redis_agents)
            except Exception as e:
                logger.warning(f"Redis discovery failed: {e}")

        # Query remote registries
        for registry_url in self.registry_endpoints:
            try:
                remote_agents = await self._discover_from_remote(registry_url, message)
                discovered_agents.extend(remote_agents)
            except Exception as e:
                logger.debug(f"Remote discovery failed for {registry_url}: {e}")

        # Deduplicate by agent_id and return top results
        seen_ids = set()
        unique_agents = []
        for agent in discovered_agents:
            if agent.agent_id not in seen_ids:
                seen_ids.add(agent.agent_id)
                unique_agents.append(agent)
                if len(unique_agents) >= max_results:
                    break

        logger.info(f"Discovered {len(unique_agents)} agents")
        return unique_agents

    def _matches_discovery_filter(self, agent_card: AgentCard, capability_filter: str, organization_filter: str) -> bool:
        """Check if agent card matches discovery filters."""
        if organization_filter and agent_card.organization != organization_filter:
            return False

        if capability_filter:
            capability_names = [cap.name.lower() for cap in agent_card.capabilities]
            if not any(capability_filter.lower() in name for name in capability_names):
                return False

        return True

    async def _discover_from_redis(self, capability_filter: str, organization_filter: str, max_results: int) -> list[AgentCard]:
        """Discover agents from Redis registry."""
        agents = []
        try:
            # Scan for agent cards
            cursor = 0
            pattern = "a2a_agent:*"
            while True:
                cursor, keys = self.redis_client.scan(cursor=cursor, match=pattern, count=100)
                for key in keys:
                    try:
                        data = self.redis_client.get(key)
                        if data:
                            agent_card = AgentCard.model_validate_json(data)
                            if self._matches_discovery_filter(agent_card, capability_filter, organization_filter):
                                agents.append(agent_card)
                                if len(agents) >= max_results:
                                    return agents
                    except Exception:
                        continue
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning(f"Redis agent discovery error: {e}")

        return agents

    async def _discover_from_remote(self, registry_url: str, message: A2AMessage) -> list[AgentCard]:
        """Discover agents from remote registry."""
        agents = []
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                response = await client.post(
                    f"{registry_url}/a2a/discover",
                    json=message.model_dump(mode="json"),
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    data = response.json()
                    for agent_data in data.get("agents", []):
                        try:
                            agent_card = AgentCard.model_validate(agent_data)
                            agents.append(agent_card)
                        except Exception:
                            continue
        except Exception as e:
            logger.debug(f"Remote discovery error for {registry_url}: {e}")

        return agents

    async def advertise_capabilities(self):
        """Advertise this agent's capabilities to registries."""
        if not self.my_card:
            logger.warning("No agent card to advertise")
            return

        # Update last seen timestamp
        self.my_card.last_seen = datetime.now(timezone.utc)

        # Attempt to publish agent card over NATS discovery channel as well
        await self._ensure_nats()
        if self.use_nats and self.nats:
            try:
                await self.nats.publish("discovery.advertise", {
                    "agent_card": self.my_card.model_dump(),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }, use_jetstream=True)
            except Exception as e:
                logger.debug(f"A2A NATS advertise failed, continue with other paths: {e}")

        # Update last seen timestamp
        self.my_card.last_seen = datetime.now(timezone.utc)

        # Store in Redis registry
        if self.redis_client:
            try:
                key = f"a2a_agent:{self.agent_id}"
                payload = self.my_card.model_dump_json()
                # Set TTL for automatic cleanup of stale agents
                self.redis_client.setex(key, 3600, payload)  # 1 hour TTL
                logger.debug("Advertised capabilities to Redis registry")
            except Exception as e:
                logger.warning(f"Failed to advertise to Redis: {e}")

        # Advertise to remote registries
        message = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            message_type=A2AMessageType.ADVERTISE,
            timestamp=datetime.now(timezone.utc),
            payload={"agent_card": self.my_card.model_dump()}
        )

        for registry_url in self.registry_endpoints:
            try:
                await self._advertise_to_remote(registry_url, message)
            except Exception as e:
                logger.debug(f"Remote advertisement failed for {registry_url}: {e}")

        # Publish event
        try:
            from utils.event_bus_adapter import get_event_publisher
            await get_event_publisher().publish_messaging("agent_advertised", {
                "agent_id": self.agent_id,
                "capabilities": len(self.my_card.capabilities)
            })
        except Exception:
            pass

    async def _advertise_to_remote(self, registry_url: str, message: A2AMessage):
        """Advertise to a remote registry."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                response = await client.post(
                    f"{registry_url}/a2a/advertise",
                    json=message.model_dump(mode="json"),
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    logger.debug(f"Successfully advertised to {registry_url}")
                else:
                    logger.warning(f"Advertisement failed to {registry_url}: HTTP {response.status_code}")
        except Exception as e:
            logger.debug(f"Advertisement error to {registry_url}: {e}")

    async def send_task_request(self,
                                target_agent_id: str,
                                capability_name: str,
                                task_data: dict[str, Any],
                                timeout_seconds: int = 300) -> dict[str, Any]:
        """Send a task request to another agent."""

        # Find target agent
        target_agent = self.local_registry.get(target_agent_id)
        if not target_agent:
            # Try to discover the agent
            agents = await self.discover_agents()
            target_agent = next((a for a in agents if a.agent_id == target_agent_id), None)

        if not target_agent:
            raise ValueError(f"Target agent {target_agent_id} not found")

        # Create task request message
        message = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=target_agent_id,
            message_type=A2AMessageType.TASK_REQUEST,
            timestamp=datetime.now(timezone.utc),
            correlation_id=str(uuid.uuid4()),
            payload={
                "capability_name": capability_name,
                "task_data": task_data,
                "timeout_seconds": timeout_seconds
            }
        )

        # Prefer NATS for low-latency delivery
        await self._ensure_nats()
        if self.use_nats and self.nats:
            subj = f"a2a.agent.{target_agent_id}.in"
            try:
                # Fire-and-wait pattern via correlation id
                fut: asyncio.Future = self.loop.create_future()
                self._pending_futures[message.correlation_id] = fut
                await self.nats.publish(subj, message.model_dump(mode="json"), use_jetstream=True)
                self.counters["messages_out"] += 1
                self.counters["requests"] += 1
                try:
                    ack = await asyncio.wait_for(fut, timeout=timeout_seconds)
                    # If the target replied over NATS with a proper A2AMessage dict, accept it as response
                    if isinstance(ack, dict) and ack.get("message_type") == A2AMessageType.TASK_RESPONSE:
                        return ack.get("payload", {})
                except asyncio.TimeoutError:
                    self.counters["timeouts"] += 1
                    logger.debug("A2A NATS request timed out, trying HTTP fallback")
            except Exception as e:
                logger.debug(f"A2A NATS publish error: {e}, trying HTTP fallback")

        # HTTP fallback (and primary response path)
        response = await self._send_message_to_agent(target_agent, message)

        if response and response.message_type == A2AMessageType.TASK_RESPONSE:
            return response.payload
        else:
            raise RuntimeError("Task request failed or timed out")

    async def _send_message_to_agent(self, target_agent: AgentCard, message: A2AMessage) -> Optional[A2AMessage]:
        """Send a message directly to an agent."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                response = await client.post(
                    f"{target_agent.endpoint_url}/a2a/message",
                    json=message.model_dump(mode="json"),
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    response_data = response.json()
                    return A2AMessage.model_validate(response_data)
                else:
                    logger.warning(f"Message send failed: HTTP {response.status_code}")
                    return None

        except Exception as e:
            logger.warning(f"Failed to send message to {target_agent.agent_id}: {e}")
            return None

    async def handle_incoming_message(self, message_data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Handle incoming A2A protocol message."""
        try:
            message = A2AMessage.model_validate(message_data)

            # ACL: optional sender allowlist
            allow = os.getenv("A2A_ALLOWED_SENDERS", "").strip()
            if allow:
                allowed = {x.strip() for x in allow.split(",") if x.strip()}
                if message.sender_id not in allowed:
                    logger.warning(f"A2A ACL deny: sender {message.sender_id} not in allowlist")
                    err = A2AMessage(
                        message_id=str(uuid.uuid4()),
                        sender_id=self.agent_id,
                        receiver_id=message.sender_id,
                        message_type=A2AMessageType.ERROR,
                        timestamp=datetime.now(timezone.utc),
                        payload={"error": "forbidden", "reason": "sender_not_allowed"},
                        correlation_id=message.correlation_id,
                    )
                    return err.model_dump(mode="json")

            # ACL: optional message type allowlist
            allowed_types = os.getenv("A2A_ALLOWED_TYPES", "").strip()
            if allowed_types:
                allowed_set = {x.strip().lower() for x in allowed_types.split(",") if x.strip()}
                if str(message.message_type.value).lower() not in allowed_set:
                    logger.warning(f"A2A ACL deny type: {message.message_type}")
                    err = A2AMessage(
                        message_id=str(uuid.uuid4()),
                        sender_id=self.agent_id,
                        receiver_id=message.sender_id,
                        message_type=A2AMessageType.ERROR,
                        timestamp=datetime.now(timezone.utc),
                        payload={"error": "forbidden", "reason": "type_not_allowed"},
                        correlation_id=message.correlation_id,
                    )
                    return err.model_dump(mode="json")

            # Check TTL
            age = (datetime.now(timezone.utc) - message.timestamp).total_seconds()
            if age > message.ttl_seconds:
                logger.debug(f"Message {message.message_id} expired (age: {age}s)")
                return None

            # Route to appropriate handler
            handler = self.message_handlers.get(message.message_type)
            if handler:
                response = await handler(message)
                return response.model_dump(mode="json") if response else None
            else:
                logger.warning(f"No handler for message type: {message.message_type}")
                return None

        except Exception as e:
            logger.error(f"Error handling incoming message: {e}")
            return None

    # Message Handlers

    async def _handle_discover(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle discovery request."""
        payload = message.payload
        capability_filter = payload.get("capability_filter")
        organization_filter = payload.get("organization_filter")
        max_results = payload.get("max_results", 50)

        # Find matching agents in local registry
        matching_agents = []
        for agent_card in self.local_registry.values():
            if self._matches_discovery_filter(agent_card, capability_filter, organization_filter):
                matching_agents.append(agent_card.model_dump())
                if len(matching_agents) >= max_results:
                    break

        # Return response
        return A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type=A2AMessageType.CAPABILITY_RESPONSE,
            timestamp=datetime.now(timezone.utc),
            correlation_id=message.correlation_id,
            payload={"agents": matching_agents}
        )

    async def _handle_advertise(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle agent advertisement."""
        try:
            agent_data = message.payload.get("agent_card")
            if agent_data:
                agent_card = AgentCard.model_validate(agent_data)
                self.local_registry[agent_card.agent_id] = agent_card
                logger.info(f"Registered agent {agent_card.name} ({agent_card.agent_id})")

                # Publish event
                try:
                    from utils.event_bus_adapter import get_event_publisher
                    await get_event_publisher().publish_messaging("agent_registered", {
                        "agent_id": agent_card.agent_id,
                        "name": agent_card.name,
                        "capabilities": len(agent_card.capabilities)
                    })
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Error handling advertisement: {e}")

        return None  # No response needed for advertisements

    async def _handle_capability_request(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle capability request."""
        if not self.my_card:
            return None

        return A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type=A2AMessageType.CAPABILITY_RESPONSE,
            timestamp=datetime.now(timezone.utc),
            correlation_id=message.correlation_id,
            payload={"agent_card": self.my_card.model_dump()}
        )

    async def _handle_chat_request(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Auto-ACK chat: echo back and ok=true; can be overridden by registering a custom handler."""
        payload = message.payload or {}
        text = payload.get("text")
        meta = payload.get("metadata")
        return A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type=A2AMessageType.CHAT_RESPONSE,
            timestamp=datetime.now(timezone.utc),
            correlation_id=message.correlation_id,
            payload={"ok": True, "echo": text, "metadata": meta}
        )

    async def _handle_chat_response(self, message: A2AMessage) -> Optional[A2AMessage]:
        if message.correlation_id:
            self.response_cache[message.correlation_id] = message
            if message.correlation_id in self.pending_requests:
                self.pending_requests[message.correlation_id].set()
        # Also complete future if waiting via NATS path
        if message.correlation_id:
            fut = self._pending_futures.pop(message.correlation_id, None)
            if fut and not fut.done():
                fut.set_result(message.payload)
        return None

    async def _handle_capability_response(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle capability response."""
        # Update local registry with received capabilities
        agents_data = message.payload.get("agents", [])
        if not agents_data and "agent_card" in message.payload:
            agents_data = [message.payload["agent_card"]]

        for agent_data in agents_data:
            try:
                agent_card = AgentCard.model_validate(agent_data)
                self.local_registry[agent_card.agent_id] = agent_card
            except Exception as e:
                logger.warning(f"Invalid agent card received: {e}")

        # Wake up any waiting discovery requests
        if message.correlation_id and message.correlation_id in self.pending_requests:
            self.response_cache[message.correlation_id] = message
            self.pending_requests[message.correlation_id].set()

        return None

    async def _handle_task_request(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle incoming task request."""
        # This would integrate with the existing agent task system
        # For now, return a simple response
        return A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type=A2AMessageType.TASK_RESPONSE,
            timestamp=datetime.now(timezone.utc),
            correlation_id=message.correlation_id,
            payload={
                "task_id": str(uuid.uuid4()),
                "status": "accepted",
                "message": "Task request received and queued for processing"
            }
        )

    async def _handle_task_response(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle task response."""
        # Cache response for waiting requests
        if message.correlation_id:
            self.response_cache[message.correlation_id] = message
            if message.correlation_id in self.pending_requests:
                self.pending_requests[message.correlation_id].set()

        return None

    async def _handle_task_status(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle task status update."""
        # Log status update
        payload = message.payload
        logger.info(f"Task status update from {message.sender_id}: {payload}")

        # Publish event
        try:
            from utils.event_bus_adapter import get_event_publisher
            await get_event_publisher().publish_messaging("a2a_task_status_update", {
                "sender_id": message.sender_id,
                "task_id": payload.get("task_id"),
                "status": payload.get("status")
            })
        except Exception:
            pass

        return None

    async def _handle_heartbeat(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle heartbeat message."""
        # Update agent's last seen timestamp
        if message.sender_id in self.local_registry:
            self.local_registry[message.sender_id].last_seen = datetime.now(timezone.utc)
    # Helper APIs
    async def send_capability_request(self, to_agent: str) -> Optional[dict[str, Any]]:
        A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=to_agent,
            message_type=A2AMessageType.CAPABILITY_REQUEST,
            timestamp=datetime.now(timezone.utc),
            correlation_id=str(uuid.uuid4()),
            payload={},
        )

    async def chat_send(self, to_agent: str, text: str, metadata: Optional[dict[str, Any]] = None, timeout_seconds: Optional[int] = None) -> Optional[dict[str, Any]]:
        """Send a chat message to another agent. If timeout_seconds is provided, await a response (blocking)."""
        msg = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=to_agent,
            message_type=A2AMessageType.CHAT_REQUEST,
            timestamp=datetime.now(timezone.utc),
            correlation_id=str(uuid.uuid4()) if timeout_seconds else None,
            payload={"text": text, "metadata": metadata or {}},
        )
        await self._ensure_nats()
        # Loopback optimization: if chatting to self, handle directly to guarantee response
        if to_agent == self.agent_id:
            resp = await self._handle_chat_request(msg)
            return resp.payload if resp else None
        if self.use_nats and self.nats:
            try:
                subj = f"a2a.agent.{to_agent}.in"
                fut = None
                if timeout_seconds:
                    fut = self.loop.create_future()
                    self._pending_futures[msg.correlation_id] = fut  # type: ignore[index]
                await self.nats.publish(subj, msg.model_dump(mode="json"), use_jetstream=True)
                self.counters["messages_out"] += 1
                if not timeout_seconds:
                    return None
                # Blocking path: await response payload (from _handle_chat_response or out-channel completion)
                ack = await asyncio.wait_for(fut, timeout=timeout_seconds)  # type: ignore[arg-type]
                return ack
            except Exception as e:
                logger.debug(f"chat_send NATS failed: {e}")
        # No NATS path or failure: non-blocking returns None, blocking returns None on timeout/failure
        if timeout_seconds:
            return None
        return None

        await self._ensure_nats()
        if self.use_nats and self.nats:
            try:
                fut: asyncio.Future = self.loop.create_future()
                self._pending_futures[msg.correlation_id] = fut
                await self.nats.publish(f"a2a.agent.{to_agent}.in", msg.model_dump(mode="json"), use_jetstream=True)
                self.counters["messages_out"] += 1
                self.counters["requests"] += 1
                ack = await asyncio.wait_for(fut, timeout=10)
                return ack
            except Exception as e:
                logger.debug(f"capability_request NATS failed: {e}")
        # HTTP fallback is left to _send_message_to_agent if needed
        return None

    async def send_status_update(self, task_id: str, status: str, to_agent: Optional[str] = None) -> None:
        msg = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=to_agent,
            message_type=A2AMessageType.TASK_STATUS,
            timestamp=datetime.now(timezone.utc),
            payload={"task_id": task_id, "status": status},
        )
        await self._ensure_nats()
        if self.use_nats and self.nats and to_agent:
            try:
                await self.nats.publish(f"a2a.agent.{to_agent}.out", msg.model_dump(mode="json"), use_jetstream=True)
                self.counters["messages_out"] += 1
            except Exception:
                pass

    async def broadcast_event(self, intent: str, payload: dict[str, Any]) -> None:
        await self._ensure_nats()
        if self.use_nats and self.nats:
            try:
                # Prefer JetStream for durability; communicator will fallback to core NATS if needed
                await self.nats.publish("a2a.events", {
                    "spec": "a2a/1",
                    "type": "event",
                    "from": self.agent_id,
                    "intent": intent,
                    "payload": payload,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }, use_jetstream=True)
                self.counters["messages_out"] += 1
            except Exception:
                pass

    async def rpc_task(self, task_id: str, method: str, params: dict[str, Any], timeout_seconds: int = 30) -> Optional[dict[str, Any]]:
        await self._ensure_nats()
        if not (self.use_nats and self.nats):
            return None
        corr = str(uuid.uuid4())
        fut: asyncio.Future = self.loop.create_future()
        self._pending_futures[corr] = fut
        try:
            await self.nats.publish(f"a2a.task.{task_id}.rpc", {
                "spec": "a2a/1",
                "type": "rpc_request",
                "from": self.agent_id,
                "method": method,
                "params": params,
                "correlation_id": corr,
            }, use_jetstream=True)
            self.counters["messages_out"] += 1
            self.counters["requests"] += 1
            ack = await asyncio.wait_for(fut, timeout=timeout_seconds)
            return ack
        except Exception:
            self.counters["timeouts"] += 1
            return None


        # Return heartbeat acknowledgment
        return A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type=A2AMessageType.HEARTBEAT,
            timestamp=datetime.now(timezone.utc),
            payload={"status": "alive"}
        )

    async def _handle_error(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle error message."""
        logger.error(f"A2A error from {message.sender_id}: {message.payload}")
        return None

    async def start_heartbeat(self, interval_seconds: int = 60):
        """Start periodic heartbeat to maintain agent presence."""
        while True:
            try:
                await self.advertise_capabilities()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(interval_seconds)


# Global A2A protocol manager instance
_a2a_manager: Optional[A2AProtocolManager] = None


def get_a2a_manager() -> A2AProtocolManager:
    """Get the global A2A protocol manager instance."""
    global _a2a_manager
    if _a2a_manager is None:
        _a2a_manager = A2AProtocolManager()
    return _a2a_manager
