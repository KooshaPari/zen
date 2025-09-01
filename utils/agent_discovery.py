"""
Agent Service Discovery System using NATS

This module provides comprehensive service discovery capabilities for distributed
agents using NATS as the messaging backbone. Features include:
- Real-time agent registration and deregistration
- Health monitoring and automatic cleanup
- Load-aware agent selection
- Geographic/edge-aware routing
- Capability-based service matching
- High-availability service mesh patterns
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Optional

import redis
from pydantic import BaseModel, Field

from tools.shared.agent_models import AgentType, TaskStatus
from utils.nats_communicator import (
    NATSCommunicator,
    get_nats_communicator,
)

logger = logging.getLogger(__name__)


class ServiceEndpoint(BaseModel):
    """Service endpoint information."""

    endpoint_id: str = Field(..., description="Unique endpoint identifier")
    url: str = Field(..., description="Endpoint URL")
    protocol: str = Field(default="http", description="Protocol (http, https, grpc)")
    health_check_path: str = Field(default="/status", description="Health check path")
    weight: int = Field(default=100, description="Load balancing weight")
    zone: Optional[str] = Field(None, description="Geographic zone/region")
    tags: set[str] = Field(default_factory=set, description="Service tags")


class AgentService(BaseModel):
    """Complete agent service definition."""

    service_id: str = Field(..., description="Unique service identifier")
    agent_id: str = Field(..., description="Agent instance identifier")
    agent_type: AgentType = Field(..., description="Type of agent")
    capabilities: set[str] = Field(default_factory=set, description="Agent capabilities")
    endpoints: list[ServiceEndpoint] = Field(default_factory=list, description="Service endpoints")

    # Status and health
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current service status")
    health_status: str = Field(default="unknown", description="Health status")
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Performance metrics
    load_factor: float = Field(default=0.0, description="Current load (0.0-1.0)")
    response_time_ms: float = Field(default=0.0, description="Average response time")
    success_rate: float = Field(default=1.0, description="Success rate (0.0-1.0)")

    # Location and routing
    zone: Optional[str] = Field(None, description="Geographic zone")
    region: Optional[str] = Field(None, description="Geographic region")
    edge_location: bool = Field(default=False, description="Whether this is an edge location")

    # Metadata
    version: Optional[str] = Field(None, description="Agent version")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: int = Field(default=300, description="TTL for registration (5 minutes)")


class ServiceQuery(BaseModel):
    """Query for service discovery."""

    agent_type: Optional[AgentType] = Field(None, description="Required agent type")
    capabilities: set[str] = Field(default_factory=set, description="Required capabilities")
    tags: set[str] = Field(default_factory=set, description="Required tags")
    zone: Optional[str] = Field(None, description="Preferred zone")
    region: Optional[str] = Field(None, description="Preferred region")
    edge_only: bool = Field(default=False, description="Only return edge locations")
    max_load: float = Field(default=0.8, description="Maximum acceptable load")
    min_success_rate: float = Field(default=0.9, description="Minimum success rate")
    max_response_time_ms: float = Field(default=1000.0, description="Maximum response time")
    limit: int = Field(default=10, description="Maximum results to return")


class AgentDiscoveryService:
    """
    NATS-based agent discovery service with high availability and performance.
    """

    def __init__(self, nats_communicator: Optional[NATSCommunicator] = None,
                 redis_client: Optional[redis.Redis] = None):
        """Initialize agent discovery service."""
        self.nats = nats_communicator
        self.redis_client = redis_client

        # Service registry
        self.services: dict[str, AgentService] = {}
        self.service_lock = asyncio.Lock()

        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.health_timeout = 5  # seconds
        self.max_missed_heartbeats = 3

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._metrics_publisher_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Performance metrics
        self.discovery_requests = 0
        self.registration_count = 0
        self.deregistration_count = 0

        logger.info("Agent Discovery Service initialized")

    async def start(self) -> None:
        """Start the discovery service."""
        if not self.nats:
            self.nats = await get_nats_communicator(self.redis_client)

        if not self.nats.connected:
            await self.nats.connect()

        # Set up NATS subscriptions
        await self._setup_subscriptions()

        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_services())
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self._metrics_publisher_task = asyncio.create_task(self._publish_metrics_loop())

        logger.info("Agent Discovery Service started")

    async def stop(self) -> None:
        """Stop the discovery service."""
        self._shutdown_event.set()

        # Cancel background tasks
        for task in [self._cleanup_task, self._health_monitor_task, self._metrics_publisher_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("Agent Discovery Service stopped")

    async def _setup_subscriptions(self) -> None:
        """Set up NATS subscriptions for discovery events."""
        if not self.nats:
            return

        # Service registration/deregistration
        await self.nats.subscribe(
            "discovery.register",
            self._handle_register_request,
            queue_group="discovery",
            use_jetstream=True,
            durable_name="discovery_register"
        )

        await self.nats.subscribe(
            "discovery.deregister",
            self._handle_deregister_request,
            queue_group="discovery",
            use_jetstream=True,
            durable_name="discovery_deregister"
        )

        # Service queries
        await self.nats.subscribe(
            "discovery.query",
            self._handle_discovery_query,
            queue_group="discovery",
            use_jetstream=True,
            durable_name="discovery_query"
        )

        # Heartbeats
        await self.nats.subscribe(
            "discovery.heartbeat",
            self._handle_heartbeat,
            queue_group="discovery"
        )

        # Health status updates
        await self.nats.subscribe(
            "discovery.health",
            self._handle_health_update,
            queue_group="discovery"
        )

    async def register_service(self, service: AgentService) -> bool:
        """
        Register an agent service.

        Args:
            service: Agent service to register

        Returns:
            Success status
        """
        try:
            async with self.service_lock:
                # Validate service
                if not service.service_id or not service.agent_id:
                    logger.error("Service ID and Agent ID are required")
                    return False

                # Set registration timestamp
                service.registered_at = datetime.now(timezone.utc)
                service.last_heartbeat = service.registered_at

                # Store in local registry
                self.services[service.service_id] = service

                # Store in Redis for persistence
                if self.redis_client:
                    try:
                        key = f"agent_service:{service.service_id}"
                        self.redis_client.setex(
                            key,
                            service.ttl_seconds,
                            service.model_dump_json()
                        )

                        # Add to zone/region indexes
                        if service.zone:
                            self.redis_client.sadd(f"services:zone:{service.zone}", service.service_id)
                            self.redis_client.expire(f"services:zone:{service.zone}", service.ttl_seconds)

                        if service.region:
                            self.redis_client.sadd(f"services:region:{service.region}", service.service_id)
                            self.redis_client.expire(f"services:region:{service.region}", service.ttl_seconds)

                        # Add to capability indexes
                        for capability in service.capabilities:
                            self.redis_client.sadd(f"services:capability:{capability}", service.service_id)
                            self.redis_client.expire(f"services:capability:{capability}", service.ttl_seconds)

                    except Exception as e:
                        logger.warning(f"Failed to store service in Redis: {e}")

                # Publish registration event
                if self.nats:
                    registration_event = {
                        "event": "service_registered",
                        "service_id": service.service_id,
                        "agent_id": service.agent_id,
                        "agent_type": service.agent_type.value,
                        "capabilities": list(service.capabilities),
                        "zone": service.zone,
                        "region": service.region,
                        "endpoints": [ep.model_dump() for ep in service.endpoints],
                        "timestamp": service.registered_at.isoformat()
                    }

                    await self.nats.publish("registry.registered", registration_event, use_jetstream=True)

                self.registration_count += 1
                logger.info(f"Registered service {service.service_id} for agent {service.agent_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to register service {service.service_id}: {e}")
            return False

    async def deregister_service(self, service_id: str) -> bool:
        """
        Deregister an agent service.

        Args:
            service_id: ID of service to deregister

        Returns:
            Success status
        """
        try:
            async with self.service_lock:
                service = self.services.pop(service_id, None)
                if not service:
                    logger.warning(f"Service {service_id} not found for deregistration")
                    return False

                # Remove from Redis
                if self.redis_client:
                    try:
                        # Remove main record
                        self.redis_client.delete(f"agent_service:{service_id}")

                        # Remove from indexes
                        if service.zone:
                            self.redis_client.srem(f"services:zone:{service.zone}", service_id)
                        if service.region:
                            self.redis_client.srem(f"services:region:{service.region}", service_id)

                        for capability in service.capabilities:
                            self.redis_client.srem(f"services:capability:{capability}", service_id)

                    except Exception as e:
                        logger.warning(f"Failed to remove service from Redis: {e}")

                # Publish deregistration event
                if self.nats:
                    deregistration_event = {
                        "event": "service_deregistered",
                        "service_id": service_id,
                        "agent_id": service.agent_id,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }

                    await self.nats.publish("registry.deregistered", deregistration_event, use_jetstream=True)

                self.deregistration_count += 1
                logger.info(f"Deregistered service {service_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to deregister service {service_id}: {e}")
            return False

    async def discover_services(self, query: ServiceQuery) -> list[AgentService]:
        """
        Discover services matching query criteria with intelligent ranking.

        Args:
            query: Service discovery query

        Returns:
            List of matching services, ranked by suitability
        """
        try:
            self.discovery_requests += 1
            start_time = time.perf_counter()

            # Get candidate services
            candidates = []

            async with self.service_lock:
                for service in self.services.values():
                    if self._matches_query(service, query):
                        candidates.append(service)

            # Load additional candidates from Redis if available
            if self.redis_client and len(candidates) < query.limit:
                candidates.extend(await self._load_candidates_from_redis(query))

            # Rank and sort candidates
            ranked_services = self._rank_services(candidates, query)

            # Apply limit
            result = ranked_services[:query.limit]

            query_time_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Discovery query completed in {query_time_ms:.2f}ms, found {len(result)} services")

            return result

        except Exception as e:
            logger.error(f"Service discovery error: {e}")
            return []

    def _matches_query(self, service: AgentService, query: ServiceQuery) -> bool:
        """Check if service matches query criteria."""
        # Check agent type
        if query.agent_type and service.agent_type != query.agent_type:
            return False

        # Check capabilities
        if query.capabilities and not query.capabilities.issubset(service.capabilities):
            return False

        # Check tags (if service has endpoint tags)
        if query.tags:
            service_tags = set()
            for endpoint in service.endpoints:
                service_tags.update(endpoint.tags)
            if not query.tags.issubset(service_tags):
                return False

        # Check health and performance criteria
        if service.load_factor > query.max_load:
            return False

        if service.success_rate < query.min_success_rate:
            return False

        if service.response_time_ms > query.max_response_time_ms:
            return False

        # Check geographic preferences
        if query.zone and service.zone != query.zone:
            return False

        if query.region and service.region != query.region:
            return False

        if query.edge_only and not service.edge_location:
            return False

        # Check service health
        now = datetime.now(timezone.utc)
        heartbeat_age = (now - service.last_heartbeat).total_seconds()
        if heartbeat_age > self.health_check_interval * self.max_missed_heartbeats:
            return False  # Service is likely unhealthy

        return True

    async def _load_candidates_from_redis(self, query: ServiceQuery) -> list[AgentService]:
        """Load additional service candidates from Redis indexes."""
        if not self.redis_client:
            return []

        try:
            candidate_ids = set()

            # Query by zone
            if query.zone:
                zone_ids = self.redis_client.smembers(f"services:zone:{query.zone}")
                candidate_ids.update(zone_ids)

            # Query by region
            if query.region:
                region_ids = self.redis_client.smembers(f"services:region:{query.region}")
                candidate_ids.update(region_ids)

            # Query by capabilities
            for capability in query.capabilities:
                cap_ids = self.redis_client.smembers(f"services:capability:{capability}")
                if candidate_ids:
                    candidate_ids.intersection_update(cap_ids)
                else:
                    candidate_ids.update(cap_ids)

            # Load service details
            candidates = []
            for service_id in candidate_ids:
                try:
                    data = self.redis_client.get(f"agent_service:{service_id}")
                    if data:
                        service = AgentService.model_validate_json(data)
                        candidates.append(service)
                except Exception as e:
                    logger.debug(f"Failed to load service {service_id} from Redis: {e}")

            return candidates

        except Exception as e:
            logger.warning(f"Failed to load candidates from Redis: {e}")
            return []

    def _rank_services(self, services: list[AgentService], query: ServiceQuery) -> list[AgentService]:
        """Rank services by suitability score."""
        scored_services = []

        for service in services:
            score = self._calculate_service_score(service, query)
            scored_services.append((score, service))

        # Sort by score (higher is better)
        scored_services.sort(key=lambda x: x[0], reverse=True)

        return [service for _, service in scored_services]

    def _calculate_service_score(self, service: AgentService, query: ServiceQuery) -> float:
        """Calculate suitability score for a service."""
        score = 100.0  # Base score

        # Performance factors (higher is better)
        score += (1.0 - service.load_factor) * 20  # Lower load is better
        score += service.success_rate * 30  # Higher success rate is better
        score += max(0, (1000 - service.response_time_ms) / 1000) * 20  # Lower response time is better

        # Geographic preference bonus
        if query.zone and service.zone == query.zone:
            score += 15
        if query.region and service.region == query.region:
            score += 10
        if query.edge_only and service.edge_location:
            score += 25

        # Capability matching bonus
        if query.capabilities:
            capability_overlap = len(query.capabilities.intersection(service.capabilities))
            score += capability_overlap * 5

        # Health penalty
        now = datetime.now(timezone.utc)
        heartbeat_age = (now - service.last_heartbeat).total_seconds()
        if heartbeat_age > self.health_check_interval:
            score -= heartbeat_age / self.health_check_interval * 10

        # Randomization for load balancing (small factor)
        score += random.uniform(-2, 2)

        return max(0.0, score)

    async def update_service_health(self, service_id: str, health_status: str,
                                  metrics: Optional[dict[str, Any]] = None) -> bool:
        """Update service health status and metrics."""
        try:
            async with self.service_lock:
                service = self.services.get(service_id)
                if not service:
                    return False

                service.health_status = health_status
                service.last_heartbeat = datetime.now(timezone.utc)

                # Update performance metrics if provided
                if metrics:
                    service.load_factor = metrics.get("load_factor", service.load_factor)
                    service.response_time_ms = metrics.get("response_time_ms", service.response_time_ms)
                    service.success_rate = metrics.get("success_rate", service.success_rate)

                # Update in Redis
                if self.redis_client:
                    try:
                        key = f"agent_service:{service_id}"
                        self.redis_client.setex(
                            key,
                            service.ttl_seconds,
                            service.model_dump_json()
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update service in Redis: {e}")

                return True

        except Exception as e:
            logger.error(f"Failed to update service health {service_id}: {e}")
            return False

    # NATS message handlers
    async def _handle_register_request(self, data: dict[str, Any]) -> None:
        """Handle service registration request."""
        try:
            service_data = data.get("service", {})
            service = AgentService.model_validate(service_data)
            await self.register_service(service)
        except Exception as e:
            logger.error(f"Register request handler error: {e}")

    async def _handle_deregister_request(self, data: dict[str, Any]) -> None:
        """Handle service deregistration request."""
        try:
            service_id = data.get("service_id")
            if service_id:
                await self.deregister_service(service_id)
        except Exception as e:
            logger.error(f"Deregister request handler error: {e}")

    async def _handle_discovery_query(self, data: dict[str, Any]) -> None:
        """Handle service discovery query."""
        try:
            query_data = data.get("query", {})
            query = ServiceQuery.model_validate(query_data)

            services = await self.discover_services(query)

            # Send response if reply_to is specified
            reply_to = data.get("reply_to")
            if reply_to and self.nats:
                response = {
                    "services": [service.model_dump() for service in services],
                    "count": len(services),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

                await self.nats.publish(reply_to, response)

        except Exception as e:
            logger.error(f"Discovery query handler error: {e}")

    async def _handle_heartbeat(self, data: dict[str, Any]) -> None:
        """Handle service heartbeat."""
        try:
            service_id = data.get("service_id")
            metrics = data.get("metrics", {})

            if service_id:
                await self.update_service_health(service_id, "healthy", metrics)

        except Exception as e:
            logger.error(f"Heartbeat handler error: {e}")

    async def _handle_health_update(self, data: dict[str, Any]) -> None:
        """Handle service health status update."""
        try:
            service_id = data.get("service_id")
            health_status = data.get("health_status", "unknown")
            metrics = data.get("metrics")

            if service_id:
                await self.update_service_health(service_id, health_status, metrics)

        except Exception as e:
            logger.error(f"Health update handler error: {e}")

    # Background tasks
    async def _cleanup_expired_services(self) -> None:
        """Clean up expired services."""
        while not self._shutdown_event.is_set():
            try:
                now = datetime.now(timezone.utc)
                expired_services = []

                async with self.service_lock:
                    for service_id, service in self.services.items():
                        # Check TTL expiration
                        age = (now - service.registered_at).total_seconds()
                        if age > service.ttl_seconds:
                            expired_services.append(service_id)
                            continue

                        # Check heartbeat expiration
                        heartbeat_age = (now - service.last_heartbeat).total_seconds()
                        if heartbeat_age > self.health_check_interval * self.max_missed_heartbeats:
                            expired_services.append(service_id)

                # Remove expired services
                for service_id in expired_services:
                    await self.deregister_service(service_id)
                    logger.info(f"Removed expired service {service_id}")

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(5)

    async def _health_monitor_loop(self) -> None:
        """Monitor service health proactively."""
        while not self._shutdown_event.is_set():
            try:
                services_to_check = []

                async with self.service_lock:
                    for service in self.services.values():
                        if service.endpoints:
                            services_to_check.append(service)

                # Check health for services with endpoints
                for service in services_to_check:
                    asyncio.create_task(self._check_service_health(service))

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)

    async def _check_service_health(self, service: AgentService) -> None:
        """Check health of a specific service."""
        try:
            # This would implement actual health checks to service endpoints
            # For now, we'll just update the heartbeat if service is responding

            # In a real implementation, you would:
            # 1. Make HTTP requests to health check endpoints
            # 2. Check response times and status codes
            # 3. Update service metrics based on responses

            # Simulate health check for now
            await asyncio.sleep(0.1)  # Simulate network delay

            # Update metrics based on simulated check
            await self.update_service_health(
                service.service_id,
                "healthy",
                {
                    "response_time_ms": random.uniform(10, 100),
                    "success_rate": random.uniform(0.95, 1.0),
                    "load_factor": random.uniform(0.0, 0.7)
                }
            )

        except Exception as e:
            logger.debug(f"Health check failed for {service.service_id}: {e}")
            await self.update_service_health(service.service_id, "unhealthy")

    async def _publish_metrics_loop(self) -> None:
        """Publish discovery service metrics."""
        while not self._shutdown_event.is_set():
            try:
                metrics = {
                    "active_services": len(self.services),
                    "discovery_requests": self.discovery_requests,
                    "registration_count": self.registration_count,
                    "deregistration_count": self.deregistration_count,
                    "services_by_type": {},
                    "services_by_zone": {},
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

                # Calculate service distribution
                async with self.service_lock:
                    for service in self.services.values():
                        # By type
                        type_key = service.agent_type.value
                        metrics["services_by_type"][type_key] = metrics["services_by_type"].get(type_key, 0) + 1

                        # By zone
                        if service.zone:
                            zone_key = service.zone
                            metrics["services_by_zone"][zone_key] = metrics["services_by_zone"].get(zone_key, 0) + 1

                # Publish metrics
                if self.nats:
                    await self.nats.publish("metrics.discovery", metrics, use_jetstream=True)

                await asyncio.sleep(60)  # Every minute

            except Exception as e:
                logger.error(f"Metrics publishing error: {e}")
                await asyncio.sleep(5)

    async def get_registry_status(self) -> dict[str, Any]:
        """Get current registry status and statistics."""
        async with self.service_lock:
            services_by_type = {}
            services_by_status = {}
            services_by_zone = {}

            for service in self.services.values():
                # By type
                type_key = service.agent_type.value
                services_by_type[type_key] = services_by_type.get(type_key, 0) + 1

                # By status
                status_key = service.status.value
                services_by_status[status_key] = services_by_status.get(status_key, 0) + 1

                # By zone
                if service.zone:
                    zone_key = service.zone
                    services_by_zone[zone_key] = services_by_zone.get(zone_key, 0) + 1

            return {
                "total_services": len(self.services),
                "discovery_requests": self.discovery_requests,
                "registration_count": self.registration_count,
                "deregistration_count": self.deregistration_count,
                "services_by_type": services_by_type,
                "services_by_status": services_by_status,
                "services_by_zone": services_by_zone,
                "health_check_interval": self.health_check_interval,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


# Global discovery service instance
_discovery_service: Optional[AgentDiscoveryService] = None


async def get_discovery_service(nats_communicator: Optional[NATSCommunicator] = None,
                               redis_client: Optional[redis.Redis] = None) -> AgentDiscoveryService:
    """Get or create the global discovery service instance."""
    global _discovery_service

    if _discovery_service is None:
        _discovery_service = AgentDiscoveryService(nats_communicator, redis_client)
        await _discovery_service.start()

    return _discovery_service


async def shutdown_discovery_service() -> None:
    """Shutdown the global discovery service."""
    global _discovery_service

    if _discovery_service:
        await _discovery_service.stop()
        _discovery_service = None
