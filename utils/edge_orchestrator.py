"""
Edge Computing Orchestrator for Distributed Agent Workloads

This module provides comprehensive edge computing orchestration capabilities using
NATS leaf nodes and intelligent workload distribution. Features include:
- Geographic workload routing and placement
- Edge node auto-discovery and health monitoring
- Intelligent load balancing across edge locations
- Fault-tolerant edge failover patterns
- Network latency optimization
- Edge resource utilization monitoring
- Hierarchical orchestration (cloud -> edge -> devices)
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import redis
from pydantic import BaseModel, Field

from tools.shared.agent_models import AgentTaskRequest, AgentType, TaskStatus
from utils.agent_discovery import AgentDiscoveryService
from utils.nats_communicator import NATSCommunicator, get_nats_communicator

logger = logging.getLogger(__name__)


class EdgeLocation(BaseModel):
    """Edge computing location definition."""

    location_id: str = Field(..., description="Unique edge location identifier")
    name: str = Field(..., description="Human-readable location name")
    region: str = Field(..., description="Geographic region")
    zone: str = Field(..., description="Availability zone")
    country_code: str = Field(..., description="ISO country code")

    # Network properties
    latency_to_cloud_ms: float = Field(..., description="Latency to cloud region")
    bandwidth_mbps: float = Field(..., description="Available bandwidth")

    # Resource capacity
    max_cpu_cores: int = Field(..., description="Maximum CPU cores available")
    max_memory_gb: float = Field(..., description="Maximum memory in GB")
    max_storage_gb: float = Field(..., description="Maximum storage in GB")
    max_agents: int = Field(..., description="Maximum concurrent agents")

    # Current utilization
    used_cpu_cores: int = Field(default=0, description="Currently used CPU cores")
    used_memory_gb: float = Field(default=0.0, description="Currently used memory")
    used_storage_gb: float = Field(default=0.0, description="Currently used storage")
    active_agents: int = Field(default=0, description="Currently active agents")

    # Health and status
    status: str = Field(default="healthy", description="Edge location status")
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    nats_leaf_url: Optional[str] = Field(None, description="NATS leaf node URL")

    # Capabilities and constraints
    supported_agent_types: set[AgentType] = Field(default_factory=set, description="Supported agent types")
    gpu_available: bool = Field(default=False, description="GPU resources available")
    specialized_hardware: list[str] = Field(default_factory=list, description="Special hardware capabilities")

    # Metadata
    cost_per_hour: float = Field(default=0.0, description="Cost per hour for usage")
    priority: int = Field(default=100, description="Location priority (lower = higher priority)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class WorkloadRequest(BaseModel):
    """Request for workload placement and execution."""

    request_id: str = Field(..., description="Unique request identifier")
    task_requests: list[AgentTaskRequest] = Field(..., description="List of tasks to execute")

    # Placement preferences
    preferred_regions: list[str] = Field(default_factory=list, description="Preferred regions")
    preferred_zones: list[str] = Field(default_factory=list, description="Preferred zones")
    required_capabilities: set[str] = Field(default_factory=set, description="Required capabilities")

    # Resource requirements
    min_cpu_cores: int = Field(default=1, description="Minimum CPU cores needed")
    min_memory_gb: float = Field(default=1.0, description="Minimum memory needed")
    estimated_duration_seconds: int = Field(default=300, description="Estimated execution time")

    # Placement constraints
    latency_budget_ms: float = Field(default=500.0, description="Maximum acceptable latency")
    cost_budget_per_hour: float = Field(default=10.0, description="Maximum cost per hour")
    require_gpu: bool = Field(default=False, description="GPU requirement")
    edge_only: bool = Field(default=False, description="Must run on edge (not cloud)")

    # Execution options
    fail_fast: bool = Field(default=False, description="Fail entire workload on first error")
    max_retries: int = Field(default=2, description="Maximum retry attempts")
    retry_backoff_seconds: float = Field(default=5.0, description="Retry backoff multiplier")

    # Metadata
    priority: int = Field(default=100, description="Request priority")
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkloadPlacement(BaseModel):
    """Workload placement decision."""

    placement_id: str = Field(..., description="Unique placement identifier")
    request: WorkloadRequest = Field(..., description="Original workload request")

    # Placement decisions
    selected_locations: list[EdgeLocation] = Field(..., description="Selected edge locations")
    task_assignments: dict[str, str] = Field(..., description="Task ID -> Location ID mapping")

    # Placement metrics
    total_estimated_cost: float = Field(..., description="Total estimated cost")
    max_expected_latency_ms: float = Field(..., description="Maximum expected latency")
    load_balance_score: float = Field(..., description="Load balancing effectiveness score")

    # Execution tracking
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Placement status")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = Field(None, description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")

    # Results
    completed_tasks: int = Field(default=0, description="Number of completed tasks")
    failed_tasks: int = Field(default=0, description="Number of failed tasks")
    actual_cost: float = Field(default=0.0, description="Actual execution cost")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class EdgeOrchestrator:
    """
    Edge computing orchestrator for intelligent workload placement and execution.
    """

    def __init__(self, nats_communicator: Optional[NATSCommunicator] = None,
                 discovery_service: Optional[AgentDiscoveryService] = None,
                 redis_client: Optional[redis.Redis] = None):
        """Initialize edge orchestrator."""
        self.nats = nats_communicator
        self.discovery = discovery_service
        self.redis_client = redis_client

        # Edge location registry
        self.edge_locations: dict[str, EdgeLocation] = {}
        self.location_lock = asyncio.Lock()

        # Active workload tracking
        self.active_placements: dict[str, WorkloadPlacement] = {}
        self.placement_lock = asyncio.Lock()

        # Orchestration settings
        self.placement_timeout_seconds = 30
        self.health_check_interval = 60  # seconds
        self.resource_update_interval = 30  # seconds

        # Background tasks
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._resource_monitor_task: Optional[asyncio.Task] = None
        self._placement_gc_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Performance metrics
        self.placement_requests = 0
        self.successful_placements = 0
        self.failed_placements = 0
        self.total_tasks_orchestrated = 0

        logger.info("Edge Orchestrator initialized")

    async def start(self) -> None:
        """Start the edge orchestrator service."""
        if not self.nats:
            self.nats = await get_nats_communicator(self.redis_client)

        if not self.nats.connected:
            await self.nats.connect()

        # Set up NATS subscriptions
        await self._setup_subscriptions()

        # Discover existing edge locations
        await self._discover_edge_locations()

        # Start background tasks
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self._resource_monitor_task = asyncio.create_task(self._resource_monitor_loop())
        self._placement_gc_task = asyncio.create_task(self._placement_gc_loop())

        logger.info("Edge Orchestrator started")

    async def stop(self) -> None:
        """Stop the edge orchestrator service."""
        self._shutdown_event.set()

        # Cancel background tasks
        for task in [self._health_monitor_task, self._resource_monitor_task, self._placement_gc_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("Edge Orchestrator stopped")

    async def _setup_subscriptions(self) -> None:
        """Set up NATS subscriptions for orchestration events."""
        if not self.nats:
            return

        # Workload placement requests
        await self.nats.subscribe(
            "edge.placement.request",
            self._handle_placement_request,
            queue_group="orchestrator",
            use_jetstream=True,
            durable_name="edge_placement_requests"
        )

        # Edge location registration
        await self.nats.subscribe(
            "edge.location.register",
            self._handle_location_registration,
            queue_group="orchestrator",
            use_jetstream=True,
            durable_name="edge_location_register"
        )

        # Edge location health updates
        await self.nats.subscribe(
            "edge.location.health",
            self._handle_location_health,
            queue_group="orchestrator"
        )

        # Resource utilization updates
        await self.nats.subscribe(
            "edge.resources.update",
            self._handle_resource_update,
            queue_group="orchestrator"
        )

        # Task completion notifications
        await self.nats.subscribe(
            "edge.task.completed",
            self._handle_task_completion,
            queue_group="orchestrator"
        )

    async def register_edge_location(self, location: EdgeLocation) -> bool:
        """
        Register a new edge computing location.

        Args:
            location: Edge location to register

        Returns:
            Success status
        """
        try:
            async with self.location_lock:
                # Validate location
                if not location.location_id or not location.region:
                    logger.error("Location ID and region are required")
                    return False

                # Set registration timestamp
                location.last_heartbeat = datetime.now(timezone.utc)

                # Store in local registry
                self.edge_locations[location.location_id] = location

                # Store in Redis for persistence
                if self.redis_client:
                    try:
                        key = f"edge_location:{location.location_id}"
                        self.redis_client.setex(key, 3600, location.model_dump_json())  # 1 hour TTL

                        # Add to region/zone indexes
                        self.redis_client.sadd(f"edge:region:{location.region}", location.location_id)
                        self.redis_client.sadd(f"edge:zone:{location.zone}", location.location_id)

                        # Set TTL on indexes
                        self.redis_client.expire(f"edge:region:{location.region}", 3600)
                        self.redis_client.expire(f"edge:zone:{location.zone}", 3600)

                    except Exception as e:
                        logger.warning(f"Failed to store edge location in Redis: {e}")

                # Publish registration event
                if self.nats:
                    registration_event = {
                        "event": "edge_location_registered",
                        "location_id": location.location_id,
                        "name": location.name,
                        "region": location.region,
                        "zone": location.zone,
                        "capabilities": {
                            "max_agents": location.max_agents,
                            "max_cpu_cores": location.max_cpu_cores,
                            "max_memory_gb": location.max_memory_gb,
                            "gpu_available": location.gpu_available,
                            "supported_agent_types": [t.value for t in location.supported_agent_types]
                        },
                        "timestamp": location.last_heartbeat.isoformat()
                    }

                    await self.nats.publish("edge.registered", registration_event, use_jetstream=True)

                logger.info(f"Registered edge location {location.location_id} in {location.region}/{location.zone}")
                return True

        except Exception as e:
            logger.error(f"Failed to register edge location {location.location_id}: {e}")
            return False

    async def place_workload(self, workload: WorkloadRequest) -> Optional[WorkloadPlacement]:
        """
        Intelligent workload placement across edge locations.

        Args:
            workload: Workload placement request

        Returns:
            Workload placement decision or None if placement failed
        """
        try:
            self.placement_requests += 1
            start_time = time.perf_counter()

            logger.info(f"Processing workload placement request {workload.request_id}")

            # Find candidate edge locations
            candidates = await self._find_candidate_locations(workload)
            if not candidates:
                logger.warning(f"No suitable edge locations found for workload {workload.request_id}")
                self.failed_placements += 1
                return None

            # Optimize placement using intelligent algorithms
            placement = await self._optimize_placement(workload, candidates)
            if not placement:
                logger.error(f"Placement optimization failed for workload {workload.request_id}")
                self.failed_placements += 1
                return None

            # Reserve resources and execute placement
            success = await self._execute_placement(placement)
            if not success:
                logger.error(f"Placement execution failed for workload {workload.request_id}")
                self.failed_placements += 1
                return None

            # Track active placement
            async with self.placement_lock:
                self.active_placements[placement.placement_id] = placement

            self.successful_placements += 1
            self.total_tasks_orchestrated += len(workload.task_requests)

            placement_time_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"Workload {workload.request_id} placed successfully in {placement_time_ms:.2f}ms")

            return placement

        except Exception as e:
            logger.error(f"Workload placement error for {workload.request_id}: {e}")
            self.failed_placements += 1
            return None

    async def _find_candidate_locations(self, workload: WorkloadRequest) -> list[EdgeLocation]:
        """Find edge locations that can satisfy workload requirements."""
        candidates = []

        async with self.location_lock:
            for location in self.edge_locations.values():
                if self._location_meets_requirements(location, workload):
                    candidates.append(location)

        # Load additional candidates from Redis if needed
        if len(candidates) < 5 and self.redis_client:
            candidates.extend(await self._load_candidates_from_redis(workload))

        return candidates

    def _location_meets_requirements(self, location: EdgeLocation, workload: WorkloadRequest) -> bool:
        """Check if edge location meets workload requirements."""
        # Health check
        if location.status != "healthy":
            return False

        # Resource availability check
        available_cpu = location.max_cpu_cores - location.used_cpu_cores
        available_memory = location.max_memory_gb - location.used_memory_gb
        available_agents = location.max_agents - location.active_agents

        if (available_cpu < workload.min_cpu_cores or
            available_memory < workload.min_memory_gb or
            available_agents < len(workload.task_requests)):
            return False

        # Agent type support check
        required_types = {req.agent_type for req in workload.task_requests}
        if not required_types.issubset(location.supported_agent_types):
            return False

        # GPU requirement check
        if workload.require_gpu and not location.gpu_available:
            return False

        # Geographic preferences
        if (workload.preferred_regions and
            location.region not in workload.preferred_regions):
            return False

        if (workload.preferred_zones and
            location.zone not in workload.preferred_zones):
            return False

        # Latency budget check
        if location.latency_to_cloud_ms > workload.latency_budget_ms:
            return False

        # Cost budget check (approximate)
        estimated_cost = location.cost_per_hour * (workload.estimated_duration_seconds / 3600)
        if estimated_cost > workload.cost_budget_per_hour:
            return False

        return True

    async def _load_candidates_from_redis(self, workload: WorkloadRequest) -> list[EdgeLocation]:
        """Load additional location candidates from Redis."""
        if not self.redis_client:
            return []

        try:
            candidates = []
            candidate_ids = set()

            # Query by preferred regions
            for region in workload.preferred_regions:
                region_ids = self.redis_client.smembers(f"edge:region:{region}")
                candidate_ids.update(region_ids)

            # Query by preferred zones
            for zone in workload.preferred_zones:
                zone_ids = self.redis_client.smembers(f"edge:zone:{zone}")
                candidate_ids.update(zone_ids)

            # Load location details
            for location_id in candidate_ids:
                if location_id in self.edge_locations:
                    continue  # Already loaded

                try:
                    data = self.redis_client.get(f"edge_location:{location_id}")
                    if data:
                        location = EdgeLocation.model_validate_json(data)
                        if self._location_meets_requirements(location, workload):
                            candidates.append(location)
                except Exception as e:
                    logger.debug(f"Failed to load edge location {location_id} from Redis: {e}")

            return candidates

        except Exception as e:
            logger.warning(f"Failed to load edge location candidates from Redis: {e}")
            return []

    async def _optimize_placement(self, workload: WorkloadRequest,
                                candidates: list[EdgeLocation]) -> Optional[WorkloadPlacement]:
        """Optimize workload placement using intelligent algorithms."""
        try:
            # Score and rank locations
            scored_locations = []
            for location in candidates:
                score = self._calculate_location_score(location, workload)
                scored_locations.append((score, location))

            # Sort by score (higher is better)
            scored_locations.sort(key=lambda x: x[0], reverse=True)

            # Select optimal locations for placement
            selected_locations = []
            task_assignments = {}
            total_cost = 0.0
            max_latency = 0.0

            # For now, use simple round-robin assignment
            # In production, this would use more sophisticated algorithms:
            # - Bin packing for resource optimization
            # - Graph algorithms for network topology awareness
            # - Machine learning for historical performance optimization

            task_index = 0
            for i, _task_request in enumerate(workload.task_requests):
                if not scored_locations:
                    logger.error("No available locations for task placement")
                    return None

                # Select location with round-robin
                location_index = task_index % len(scored_locations)
                _, selected_location = scored_locations[location_index]

                if selected_location not in selected_locations:
                    selected_locations.append(selected_location)

                # Create task ID and assign to location
                task_id = f"task_{workload.request_id}_{i}"
                task_assignments[task_id] = selected_location.location_id

                # Calculate cost and latency
                task_cost = (selected_location.cost_per_hour *
                           workload.estimated_duration_seconds / 3600)
                total_cost += task_cost
                max_latency = max(max_latency, selected_location.latency_to_cloud_ms)

                task_index += 1

            # Calculate load balance score
            location_task_counts = {}
            for location_id in task_assignments.values():
                location_task_counts[location_id] = location_task_counts.get(location_id, 0) + 1

            # Simple load balance score (lower variance = better balance)
            if len(location_task_counts) > 1:
                counts = list(location_task_counts.values())
                avg_count = sum(counts) / len(counts)
                variance = sum((c - avg_count) ** 2 for c in counts) / len(counts)
                load_balance_score = 100.0 / (1.0 + variance)
            else:
                load_balance_score = 100.0

            # Create placement
            placement = WorkloadPlacement(
                placement_id=f"placement_{workload.request_id}_{int(time.time())}",
                request=workload,
                selected_locations=selected_locations,
                task_assignments=task_assignments,
                total_estimated_cost=total_cost,
                max_expected_latency_ms=max_latency,
                load_balance_score=load_balance_score
            )

            return placement

        except Exception as e:
            logger.error(f"Placement optimization error: {e}")
            return None

    def _calculate_location_score(self, location: EdgeLocation, workload: WorkloadRequest) -> float:
        """Calculate suitability score for a location."""
        score = 100.0  # Base score

        # Resource availability bonus
        cpu_availability = (location.max_cpu_cores - location.used_cpu_cores) / location.max_cpu_cores
        memory_availability = (location.max_memory_gb - location.used_memory_gb) / location.max_memory_gb
        agent_availability = (location.max_agents - location.active_agents) / location.max_agents

        score += cpu_availability * 20
        score += memory_availability * 20
        score += agent_availability * 15

        # Latency bonus (lower is better)
        latency_score = max(0, (1000 - location.latency_to_cloud_ms) / 1000) * 25
        score += latency_score

        # Cost efficiency bonus (lower cost is better)
        if location.cost_per_hour > 0:
            cost_score = max(0, (10.0 - location.cost_per_hour) / 10.0) * 15
            score += cost_score

        # Priority bonus
        priority_score = max(0, (200 - location.priority) / 200) * 10
        score += priority_score

        # Geographic preference bonus
        if workload.preferred_regions and location.region in workload.preferred_regions:
            score += 15
        if workload.preferred_zones and location.zone in workload.preferred_zones:
            score += 10

        # GPU bonus if required
        if workload.require_gpu and location.gpu_available:
            score += 20

        # Health penalty
        now = datetime.now(timezone.utc)
        heartbeat_age = (now - location.last_heartbeat).total_seconds()
        if heartbeat_age > self.health_check_interval:
            score -= heartbeat_age / self.health_check_interval * 15

        return max(0.0, score)

    async def _execute_placement(self, placement: WorkloadPlacement) -> bool:
        """Execute the workload placement by submitting tasks to edge locations."""
        try:
            placement.status = TaskStatus.STARTING
            placement.started_at = datetime.now(timezone.utc)

            # Reserve resources on selected locations
            for location in placement.selected_locations:
                await self._reserve_location_resources(location, placement)

            # Submit tasks to their assigned locations
            submitted_tasks = []
            for task_id, location_id in placement.task_assignments.items():
                location = next(
                    (loc for loc in placement.selected_locations if loc.location_id == location_id),
                    None
                )
                if not location:
                    logger.error(f"Location {location_id} not found for task {task_id}")
                    continue

                # Get corresponding task request
                task_index = int(task_id.split('_')[-1])
                if task_index >= len(placement.request.task_requests):
                    logger.error(f"Task index {task_index} out of range")
                    continue

                task_request = placement.request.task_requests[task_index]

                # Submit task to edge location
                success = await self._submit_task_to_edge(task_id, task_request, location, placement)
                if success:
                    submitted_tasks.append(task_id)
                else:
                    logger.error(f"Failed to submit task {task_id} to {location_id}")
                    # Could implement fallback placement here

            if not submitted_tasks:
                logger.error(f"No tasks successfully submitted for placement {placement.placement_id}")
                return False

            placement.status = TaskStatus.RUNNING

            # Publish placement event
            if self.nats:
                placement_event = {
                    "event": "workload_placed",
                    "placement_id": placement.placement_id,
                    "request_id": placement.request.request_id,
                    "locations": [loc.location_id for loc in placement.selected_locations],
                    "tasks_submitted": len(submitted_tasks),
                    "estimated_cost": placement.total_estimated_cost,
                    "timestamp": placement.started_at.isoformat()
                }

                await self.nats.publish("edge.placed", placement_event, use_jetstream=True)

            return True

        except Exception as e:
            logger.error(f"Placement execution error: {e}")
            placement.status = TaskStatus.FAILED
            placement.error_message = str(e)
            return False

    async def _reserve_location_resources(self, location: EdgeLocation, placement: WorkloadPlacement) -> None:
        """Reserve resources on an edge location for the placement."""
        try:
            # Calculate resource requirements for this location
            tasks_for_location = sum(
                1 for loc_id in placement.task_assignments.values()
                if loc_id == location.location_id
            )

            # Update resource utilization
            async with self.location_lock:
                location.used_cpu_cores += placement.request.min_cpu_cores * tasks_for_location
                location.used_memory_gb += placement.request.min_memory_gb * tasks_for_location
                location.active_agents += tasks_for_location

            # Update in Redis
            if self.redis_client:
                try:
                    key = f"edge_location:{location.location_id}"
                    self.redis_client.setex(key, 3600, location.model_dump_json())
                except Exception as e:
                    logger.warning(f"Failed to update location resources in Redis: {e}")

            logger.debug(f"Reserved resources on {location.location_id} for {tasks_for_location} tasks")

        except Exception as e:
            logger.error(f"Resource reservation error for {location.location_id}: {e}")

    async def _submit_task_to_edge(self, task_id: str, task_request: AgentTaskRequest,
                                 location: EdgeLocation, placement: WorkloadPlacement) -> bool:
        """Submit a task to a specific edge location."""
        try:
            # Create task submission message
            task_message = {
                "task_id": task_id,
                "placement_id": placement.placement_id,
                "agent_type": task_request.agent_type.value,
                "task_description": task_request.task_description,
                "message": task_request.message,
                "agent_args": task_request.agent_args,
                "env_vars": task_request.env_vars,
                "timeout_seconds": task_request.timeout_seconds,
                "working_directory": task_request.working_directory,
                "files": task_request.files,
                "location_id": location.location_id,
                "priority": placement.request.priority,
                "submitted_at": datetime.now(timezone.utc).isoformat()
            }

            # Submit to edge-specific subject
            subject = f"edge.{location.location_id}.task.submit"

            if self.nats:
                success = await self.nats.publish(subject, task_message, use_jetstream=True)
                if success:
                    logger.debug(f"Submitted task {task_id} to edge {location.location_id}")
                    return True
                else:
                    logger.error(f"Failed to publish task {task_id} to {subject}")
                    return False
            else:
                logger.error("NATS communicator not available")
                return False

        except Exception as e:
            logger.error(f"Task submission error for {task_id}: {e}")
            return False

    # NATS message handlers
    async def _handle_placement_request(self, data: dict[str, Any]) -> None:
        """Handle workload placement request."""
        try:
            workload_data = data.get("workload", {})
            workload = WorkloadRequest.model_validate(workload_data)

            placement = await self.place_workload(workload)

            # Send response if reply_to is specified
            reply_to = data.get("reply_to")
            if reply_to and self.nats:
                if placement:
                    response = {
                        "success": True,
                        "placement": placement.model_dump(),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                else:
                    response = {
                        "success": False,
                        "error": "Placement failed",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }

                await self.nats.publish(reply_to, response)

        except Exception as e:
            logger.error(f"Placement request handler error: {e}")

    async def _handle_location_registration(self, data: dict[str, Any]) -> None:
        """Handle edge location registration."""
        try:
            location_data = data.get("location", {})
            location = EdgeLocation.model_validate(location_data)
            await self.register_edge_location(location)
        except Exception as e:
            logger.error(f"Location registration handler error: {e}")

    async def _handle_location_health(self, data: dict[str, Any]) -> None:
        """Handle edge location health update."""
        try:
            location_id = data.get("location_id")
            status = data.get("status", "healthy")

            if location_id:
                async with self.location_lock:
                    location = self.edge_locations.get(location_id)
                    if location:
                        location.status = status
                        location.last_heartbeat = datetime.now(timezone.utc)

        except Exception as e:
            logger.error(f"Location health handler error: {e}")

    async def _handle_resource_update(self, data: dict[str, Any]) -> None:
        """Handle edge location resource utilization update."""
        try:
            location_id = data.get("location_id")
            resource_usage = data.get("resources", {})

            if location_id:
                async with self.location_lock:
                    location = self.edge_locations.get(location_id)
                    if location:
                        location.used_cpu_cores = resource_usage.get("used_cpu_cores", location.used_cpu_cores)
                        location.used_memory_gb = resource_usage.get("used_memory_gb", location.used_memory_gb)
                        location.used_storage_gb = resource_usage.get("used_storage_gb", location.used_storage_gb)
                        location.active_agents = resource_usage.get("active_agents", location.active_agents)

        except Exception as e:
            logger.error(f"Resource update handler error: {e}")

    async def _handle_task_completion(self, data: dict[str, Any]) -> None:
        """Handle task completion notification."""
        try:
            data.get("task_id")
            placement_id = data.get("placement_id")
            status = data.get("status", "completed")

            if placement_id:
                async with self.placement_lock:
                    placement = self.active_placements.get(placement_id)
                    if placement:
                        if status == "completed":
                            placement.completed_tasks += 1
                        else:
                            placement.failed_tasks += 1

                        # Check if all tasks are done
                        total_tasks = len(placement.task_assignments)
                        finished_tasks = placement.completed_tasks + placement.failed_tasks

                        if finished_tasks >= total_tasks:
                            placement.completed_at = datetime.now(timezone.utc)

                            if placement.failed_tasks == 0:
                                placement.status = TaskStatus.COMPLETED
                            elif placement.request.fail_fast or placement.completed_tasks == 0:
                                placement.status = TaskStatus.FAILED
                            else:
                                placement.status = TaskStatus.COMPLETED  # Partial success

                            # Publish completion event
                            if self.nats:
                                completion_event = {
                                    "event": "workload_completed",
                                    "placement_id": placement_id,
                                    "status": placement.status.value,
                                    "completed_tasks": placement.completed_tasks,
                                    "failed_tasks": placement.failed_tasks,
                                    "duration_seconds": (placement.completed_at - placement.started_at).total_seconds() if placement.started_at else 0,
                                    "timestamp": placement.completed_at.isoformat()
                                }

                                await self.nats.publish("edge.completed", completion_event, use_jetstream=True)

        except Exception as e:
            logger.error(f"Task completion handler error: {e}")

    async def _discover_edge_locations(self) -> None:
        """Discover existing edge locations from Redis and NATS."""
        try:
            if self.redis_client:
                # Load from Redis
                cursor = 0
                pattern = "edge_location:*"

                while True:
                    cursor, keys = self.redis_client.scan(cursor=cursor, match=pattern, count=100)
                    for key in keys:
                        try:
                            data = self.redis_client.get(key)
                            if data:
                                location = EdgeLocation.model_validate_json(data)
                                async with self.location_lock:
                                    self.edge_locations[location.location_id] = location
                        except Exception as e:
                            logger.debug(f"Failed to load edge location from {key}: {e}")

                    if cursor == 0:
                        break

            logger.info(f"Discovered {len(self.edge_locations)} edge locations")

        except Exception as e:
            logger.warning(f"Edge location discovery error: {e}")

    # Background tasks
    async def _health_monitor_loop(self) -> None:
        """Monitor edge location health."""
        while not self._shutdown_event.is_set():
            try:
                now = datetime.now(timezone.utc)
                unhealthy_locations = []

                async with self.location_lock:
                    for location_id, location in self.edge_locations.items():
                        heartbeat_age = (now - location.last_heartbeat).total_seconds()

                        if heartbeat_age > self.health_check_interval * 2:
                            if location.status == "healthy":
                                location.status = "unhealthy"
                                unhealthy_locations.append(location_id)
                                logger.warning(f"Edge location {location_id} marked unhealthy (no heartbeat for {heartbeat_age:.1f}s)")

                # Publish health alerts
                if unhealthy_locations and self.nats:
                    health_alert = {
                        "event": "edge_locations_unhealthy",
                        "locations": unhealthy_locations,
                        "timestamp": now.isoformat()
                    }

                    await self.nats.publish("edge.health.alert", health_alert, use_jetstream=True)

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)

    async def _resource_monitor_loop(self) -> None:
        """Monitor resource utilization across edge locations."""
        while not self._shutdown_event.is_set():
            try:
                # Collect resource utilization data
                resource_data = []

                async with self.location_lock:
                    for location in self.edge_locations.values():
                        utilization = {
                            "location_id": location.location_id,
                            "region": location.region,
                            "zone": location.zone,
                            "cpu_utilization": location.used_cpu_cores / max(location.max_cpu_cores, 1),
                            "memory_utilization": location.used_memory_gb / max(location.max_memory_gb, 1),
                            "storage_utilization": location.used_storage_gb / max(location.max_storage_gb, 1),
                            "agent_utilization": location.active_agents / max(location.max_agents, 1),
                            "status": location.status,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                        resource_data.append(utilization)

                # Publish resource metrics
                if resource_data and self.nats:
                    metrics_event = {
                        "event": "edge_resource_metrics",
                        "locations": resource_data,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }

                    await self.nats.publish("metrics.edge.resources", metrics_event, use_jetstream=True)

                await asyncio.sleep(self.resource_update_interval)

            except Exception as e:
                logger.error(f"Resource monitor error: {e}")
                await asyncio.sleep(5)

    async def _placement_gc_loop(self) -> None:
        """Garbage collect completed placements."""
        while not self._shutdown_event.is_set():
            try:
                now = datetime.now(timezone.utc)
                expired_placements = []

                async with self.placement_lock:
                    for placement_id, placement in self.active_placements.items():
                        if placement.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                            if placement.completed_at:
                                age = (now - placement.completed_at).total_seconds()
                                if age > 3600:  # Keep for 1 hour after completion
                                    expired_placements.append(placement_id)

                # Remove expired placements
                for placement_id in expired_placements:
                    async with self.placement_lock:
                        placement = self.active_placements.pop(placement_id, None)
                        if placement:
                            logger.debug(f"Garbage collected placement {placement_id}")

                await asyncio.sleep(300)  # Run every 5 minutes

            except Exception as e:
                logger.error(f"Placement GC error: {e}")
                await asyncio.sleep(30)

    async def get_orchestrator_metrics(self) -> dict[str, Any]:
        """Get orchestrator performance metrics."""
        async with self.location_lock:
            location_stats = {
                "total_locations": len(self.edge_locations),
                "healthy_locations": sum(1 for loc in self.edge_locations.values() if loc.status == "healthy"),
                "total_capacity": {
                    "cpu_cores": sum(loc.max_cpu_cores for loc in self.edge_locations.values()),
                    "memory_gb": sum(loc.max_memory_gb for loc in self.edge_locations.values()),
                    "agents": sum(loc.max_agents for loc in self.edge_locations.values())
                },
                "total_utilization": {
                    "cpu_cores": sum(loc.used_cpu_cores for loc in self.edge_locations.values()),
                    "memory_gb": sum(loc.used_memory_gb for loc in self.edge_locations.values()),
                    "agents": sum(loc.active_agents for loc in self.edge_locations.values())
                }
            }

        async with self.placement_lock:
            placement_stats = {
                "active_placements": len(self.active_placements),
                "running_placements": sum(1 for p in self.active_placements.values() if p.status == TaskStatus.RUNNING),
                "completed_placements": sum(1 for p in self.active_placements.values() if p.status == TaskStatus.COMPLETED),
                "failed_placements": sum(1 for p in self.active_placements.values() if p.status == TaskStatus.FAILED)
            }

        return {
            "placement_requests": self.placement_requests,
            "successful_placements": self.successful_placements,
            "failed_placements": self.failed_placements,
            "success_rate": self.successful_placements / max(self.placement_requests, 1),
            "total_tasks_orchestrated": self.total_tasks_orchestrated,
            "locations": location_stats,
            "placements": placement_stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global edge orchestrator instance
_edge_orchestrator: Optional[EdgeOrchestrator] = None


async def get_edge_orchestrator(nats_communicator: Optional[NATSCommunicator] = None,
                               discovery_service: Optional[AgentDiscoveryService] = None,
                               redis_client: Optional[redis.Redis] = None) -> EdgeOrchestrator:
    """Get or create the global edge orchestrator instance."""
    global _edge_orchestrator

    if _edge_orchestrator is None:
        _edge_orchestrator = EdgeOrchestrator(nats_communicator, discovery_service, redis_client)
        await _edge_orchestrator.start()

    return _edge_orchestrator


async def shutdown_edge_orchestrator() -> None:
    """Shutdown the global edge orchestrator."""
    global _edge_orchestrator

    if _edge_orchestrator:
        await _edge_orchestrator.stop()
        _edge_orchestrator = None
