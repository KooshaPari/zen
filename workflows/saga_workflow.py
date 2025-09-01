"""
Saga Pattern Implementation for Distributed Transactions

This module implements the saga pattern for managing distributed transactions
across multiple agents and services with automatic compensation on failures.
Provides both orchestration and choreography-based saga patterns.

Features:
- Forward and compensation action definitions
- Orchestration-based saga coordination
- Choreography-based event-driven sagas
- Partial rollback and recovery
- Transaction state persistence
- Integration with multi-agent workflows
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel

from tools.shared.agent_models import AgentType
from utils.agent_manager import get_task_manager
from utils.event_bus import get_event_bus
from utils.storage_backend import get_storage_backend
from utils.temporal_client import BaseWorkflow

logger = logging.getLogger(__name__)


class SagaStepStatus(str, Enum):
    """Status of individual saga steps."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"
    COMPENSATION_FAILED = "compensation_failed"
    SKIPPED = "skipped"


class SagaStatus(str, Enum):
    """Overall saga transaction status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    COMPENSATION_FAILED = "compensation_failed"
    ABORTED = "aborted"


class SagaStepDefinition(BaseModel):
    """Definition of a single saga step."""
    step_id: str
    name: str
    description: str
    forward_action: dict[str, Any]  # Action to execute
    compensation_action: Optional[dict[str, Any]] = None  # Action to compensate
    agent_type: AgentType  # Agent responsible for this step
    timeout_minutes: int = 30
    retry_count: int = 3
    critical: bool = True  # If false, failure won't trigger compensation
    dependencies: list[str] = []  # Step IDs this step depends on
    parallel_group: Optional[str] = None  # Group for parallel execution


class SagaStepExecution(BaseModel):
    """Runtime state of a saga step execution."""
    step_id: str
    definition: SagaStepDefinition
    status: SagaStepStatus = SagaStepStatus.PENDING
    attempt_count: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    forward_result: Optional[dict[str, Any]] = None
    compensation_result: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    agent_task_id: Optional[str] = None


class SagaTransaction(BaseModel):
    """Complete saga transaction definition and state."""
    saga_id: str
    name: str
    description: str
    steps: list[SagaStepDefinition]
    status: SagaStatus = SagaStatus.PENDING
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    context: dict[str, Any] = {}  # Shared transaction context
    execution_state: dict[str, SagaStepExecution] = {}
    compensation_order: list[str] = []  # Order for compensation execution
    total_duration_seconds: Optional[float] = None
    success_rate: float = 0.0


class SagaWorkflow(BaseWorkflow):
    """
    Saga pattern implementation for distributed transaction management.

    Supports both orchestration-based coordination where a central coordinator
    manages the transaction flow, and choreography-based coordination using
    event-driven communication between services.
    """

    def __init__(self, coordination_mode: str = "orchestration"):
        super().__init__()
        self.coordination_mode = coordination_mode  # "orchestration" or "choreography"
        self.task_manager = get_task_manager()
        self.storage = get_storage_backend()
        self.event_bus = get_event_bus()

    async def orchestrate(self, workflow_args: dict[str, Any]) -> dict[str, Any]:
        """
        Main saga orchestration logic.

        Args:
            workflow_args: Contains saga definition and configuration

        Returns:
            Dict containing saga execution results and compensation details
        """
        saga_definition = workflow_args["saga_definition"]
        config = workflow_args.get("config", {})

        # Create saga transaction
        saga_transaction = SagaTransaction(
            saga_id=saga_definition.get("saga_id", f"saga-{uuid4()}"),
            name=saga_definition["name"],
            description=saga_definition.get("description", ""),
            steps=[SagaStepDefinition(**step) for step in saga_definition["steps"]],
            created_at=datetime.utcnow(),
            context=saga_definition.get("context", {})
        )

        # Initialize step executions
        for step_def in saga_transaction.steps:
            saga_transaction.execution_state[step_def.step_id] = SagaStepExecution(
                step_id=step_def.step_id,
                definition=step_def
            )

        try:
            logger.info(f"Starting saga transaction: {saga_transaction.name}")

            saga_transaction.status = SagaStatus.RUNNING
            saga_transaction.started_at = datetime.utcnow()
            await self._store_saga_transaction(saga_transaction)

            # Publish saga started event
            await self.event_bus.publish({
                "event": "saga_started",
                "saga_id": saga_transaction.saga_id,
                "name": saga_transaction.name,
                "steps_count": len(saga_transaction.steps),
                "coordination_mode": self.coordination_mode,
                "timestamp": saga_transaction.started_at.isoformat()
            })

            # Execute saga based on coordination mode
            if self.coordination_mode == "orchestration":
                result = await self._execute_orchestration_saga(saga_transaction, config)
            else:
                result = await self._execute_choreography_saga(saga_transaction, config)

            # Calculate final metrics
            saga_transaction.completed_at = datetime.utcnow()
            saga_transaction.total_duration_seconds = (
                saga_transaction.completed_at - saga_transaction.started_at
            ).total_seconds()

            completed_steps = len([
                step for step in saga_transaction.execution_state.values()
                if step.status == SagaStepStatus.COMPLETED
            ])
            saga_transaction.success_rate = completed_steps / len(saga_transaction.steps)

            await self._store_saga_transaction(saga_transaction)

            # Publish completion event
            await self.event_bus.publish({
                "event": "saga_completed",
                "saga_id": saga_transaction.saga_id,
                "status": saga_transaction.status.value,
                "success_rate": saga_transaction.success_rate,
                "duration_seconds": saga_transaction.total_duration_seconds,
                "timestamp": saga_transaction.completed_at.isoformat()
            })

            return {
                "saga_id": saga_transaction.saga_id,
                "status": saga_transaction.status.value,
                "success_rate": saga_transaction.success_rate,
                "duration_seconds": saga_transaction.total_duration_seconds,
                "completed_steps": completed_steps,
                "total_steps": len(saga_transaction.steps),
                "saga_transaction": saga_transaction.model_dump(),
                **result
            }

        except Exception as e:
            logger.error(f"Saga transaction failed: {e}")

            saga_transaction.status = SagaStatus.FAILED
            saga_transaction.completed_at = datetime.utcnow()
            await self._store_saga_transaction(saga_transaction)

            # Publish failure event
            await self.event_bus.publish({
                "event": "saga_failed",
                "saga_id": saga_transaction.saga_id,
                "error": str(e),
                "timestamp": saga_transaction.completed_at.isoformat() if saga_transaction.completed_at else datetime.utcnow().isoformat()
            })

            return {
                "saga_id": saga_transaction.saga_id,
                "status": "failed",
                "error": str(e),
                "saga_transaction": saga_transaction.model_dump()
            }

    async def _execute_orchestration_saga(
        self,
        saga_transaction: SagaTransaction,
        config: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute saga using orchestration pattern (central coordinator)."""
        logger.info(f"Executing orchestration saga: {saga_transaction.saga_id}")

        try:
            # Build execution plan with dependency resolution
            execution_plan = self._build_execution_plan(saga_transaction.steps)

            # Execute steps according to plan
            for execution_group in execution_plan:
                # Execute steps in current group (can be parallel)
                if len(execution_group) == 1:
                    # Single step - execute sequentially
                    step_result = await self._execute_saga_step(
                        saga_transaction, execution_group[0], config
                    )
                    if not step_result["success"] and execution_group[0].critical:
                        raise Exception(f"Critical step {execution_group[0].step_id} failed: {step_result.get('error')}")
                else:
                    # Multiple steps - execute in parallel
                    parallel_results = await asyncio.gather(
                        *[self._execute_saga_step(saga_transaction, step, config) for step in execution_group],
                        return_exceptions=True
                    )

                    # Check results
                    for i, result in enumerate(parallel_results):
                        step = execution_group[i]
                        if isinstance(result, Exception) or (not result.get("success") and step.critical):
                            raise Exception(f"Critical parallel step {step.step_id} failed")

                # Update transaction state
                await self._store_saga_transaction(saga_transaction)

            # All steps completed successfully
            saga_transaction.status = SagaStatus.COMPLETED
            return {
                "success": True,
                "message": "All saga steps completed successfully",
                "execution_plan_groups": len(execution_plan)
            }

        except Exception as e:
            logger.error(f"Orchestration saga failed: {e}")

            # Execute compensation
            compensation_result = await self._execute_compensation(saga_transaction)

            saga_transaction.status = (
                SagaStatus.COMPENSATED if compensation_result["success"]
                else SagaStatus.COMPENSATION_FAILED
            )

            return {
                "success": False,
                "error": str(e),
                "compensation_result": compensation_result
            }

    async def _execute_choreography_saga(
        self,
        saga_transaction: SagaTransaction,
        config: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute saga using choreography pattern (event-driven)."""
        logger.info(f"Executing choreography saga: {saga_transaction.saga_id}")

        # In choreography mode, each service is responsible for listening to events
        # and triggering the next step. This is a simplified implementation.

        try:
            # Subscribe to saga events
            event_queue = await self.event_bus.subscribe()

            # Trigger first step(s)
            initial_steps = [step for step in saga_transaction.steps if not step.dependencies]

            for step in initial_steps:
                await self._trigger_choreography_step(saga_transaction, step)

            # Wait for all steps to complete via events
            completed_steps = set()
            failed_steps = set()
            timeout = datetime.utcnow() + timedelta(minutes=config.get("saga_timeout_minutes", 60))

            while len(completed_steps) + len(failed_steps) < len(saga_transaction.steps):
                if datetime.utcnow() > timeout:
                    raise Exception("Saga timeout reached")

                try:
                    # Wait for next event
                    event = await asyncio.wait_for(event_queue.get(), timeout=30.0)

                    if event.get("saga_id") != saga_transaction.saga_id:
                        continue

                    event_type = event.get("event")
                    step_id = event.get("step_id")

                    if event_type == "saga_step_completed":
                        completed_steps.add(step_id)
                        # Trigger dependent steps
                        await self._trigger_dependent_steps(saga_transaction, step_id)
                    elif event_type == "saga_step_failed":
                        failed_steps.add(step_id)
                        # Trigger compensation if critical step failed
                        step_execution = saga_transaction.execution_state.get(step_id)
                        if step_execution and step_execution.definition.critical:
                            raise Exception(f"Critical choreography step {step_id} failed")

                except asyncio.TimeoutError:
                    # Check for progress
                    logger.debug("No saga events received, continuing to wait...")

            # Cleanup subscription
            await self.event_bus.unsubscribe(event_queue)

            saga_transaction.status = SagaStatus.COMPLETED
            return {
                "success": True,
                "message": "Choreography saga completed successfully",
                "completed_steps": len(completed_steps),
                "failed_steps": len(failed_steps)
            }

        except Exception as e:
            logger.error(f"Choreography saga failed: {e}")

            # Execute compensation
            compensation_result = await self._execute_compensation(saga_transaction)

            saga_transaction.status = (
                SagaStatus.COMPENSATED if compensation_result["success"]
                else SagaStatus.COMPENSATION_FAILED
            )

            return {
                "success": False,
                "error": str(e),
                "compensation_result": compensation_result
            }

    def _build_execution_plan(self, steps: list[SagaStepDefinition]) -> list[list[SagaStepDefinition]]:
        """Build execution plan respecting dependencies and parallel groups."""
        execution_plan = []
        executed_steps = set()
        remaining_steps = {step.step_id: step for step in steps}

        while remaining_steps:
            # Find steps that can be executed (dependencies satisfied)
            ready_steps = []

            for _step_id, step in remaining_steps.items():
                if all(dep in executed_steps for dep in step.dependencies):
                    ready_steps.append(step)

            if not ready_steps:
                # Circular dependency or other issue
                raise Exception("Cannot resolve step dependencies - possible circular dependency")

            # Group by parallel execution groups
            parallel_groups = {}
            sequential_steps = []

            for step in ready_steps:
                if step.parallel_group:
                    if step.parallel_group not in parallel_groups:
                        parallel_groups[step.parallel_group] = []
                    parallel_groups[step.parallel_group].append(step)
                else:
                    sequential_steps.append(step)

            # Add parallel groups to execution plan
            for group_steps in parallel_groups.values():
                execution_plan.append(group_steps)

            # Add sequential steps individually
            for step in sequential_steps:
                execution_plan.append([step])

            # Mark steps as executed
            for step in ready_steps:
                executed_steps.add(step.step_id)
                del remaining_steps[step.step_id]

        return execution_plan

    async def _execute_saga_step(
        self,
        saga_transaction: SagaTransaction,
        step_definition: SagaStepDefinition,
        config: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a single saga step with retry logic."""
        step_execution = saga_transaction.execution_state[step_definition.step_id]

        logger.info(f"Executing saga step: {step_definition.name}")

        step_execution.status = SagaStepStatus.RUNNING
        step_execution.started_at = datetime.utcnow()

        # Publish step started event
        await self.event_bus.publish({
            "event": "saga_step_started",
            "saga_id": saga_transaction.saga_id,
            "step_id": step_definition.step_id,
            "step_name": step_definition.name,
            "agent_type": step_definition.agent_type.value,
            "timestamp": step_execution.started_at.isoformat()
        })

        max_attempts = step_definition.retry_count + 1

        for attempt in range(max_attempts):
            step_execution.attempt_count = attempt + 1

            try:
                # Execute forward action
                forward_result = await self._execute_step_action(
                    step_definition.forward_action,
                    step_definition.agent_type,
                    saga_transaction.context,
                    step_definition.timeout_minutes
                )

                if forward_result.get("success"):
                    step_execution.status = SagaStepStatus.COMPLETED
                    step_execution.completed_at = datetime.utcnow()
                    step_execution.forward_result = forward_result

                    # Add to compensation order
                    if step_definition.compensation_action:
                        saga_transaction.compensation_order.insert(0, step_definition.step_id)

                    # Publish step completed event
                    await self.event_bus.publish({
                        "event": "saga_step_completed",
                        "saga_id": saga_transaction.saga_id,
                        "step_id": step_definition.step_id,
                        "result": forward_result,
                        "timestamp": step_execution.completed_at.isoformat()
                    })

                    return {"success": True, "result": forward_result}
                else:
                    # Step failed, check if we should retry
                    error_message = forward_result.get("error", "Step execution failed")

                    if attempt < max_attempts - 1:
                        logger.warning(f"Step {step_definition.step_id} failed (attempt {attempt + 1}/{max_attempts}): {error_message}")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        # Max attempts reached
                        step_execution.status = SagaStepStatus.FAILED
                        step_execution.completed_at = datetime.utcnow()
                        step_execution.error_message = error_message

                        # Publish step failed event
                        await self.event_bus.publish({
                            "event": "saga_step_failed",
                            "saga_id": saga_transaction.saga_id,
                            "step_id": step_definition.step_id,
                            "error": error_message,
                            "attempts": max_attempts,
                            "timestamp": step_execution.completed_at.isoformat()
                        })

                        return {"success": False, "error": error_message}

            except Exception as e:
                error_message = str(e)

                if attempt < max_attempts - 1:
                    logger.warning(f"Step {step_definition.step_id} exception (attempt {attempt + 1}/{max_attempts}): {error_message}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    step_execution.status = SagaStepStatus.FAILED
                    step_execution.completed_at = datetime.utcnow()
                    step_execution.error_message = error_message

                    # Publish step failed event
                    await self.event_bus.publish({
                        "event": "saga_step_failed",
                        "saga_id": saga_transaction.saga_id,
                        "step_id": step_definition.step_id,
                        "error": error_message,
                        "timestamp": step_execution.completed_at.isoformat()
                    })

                    return {"success": False, "error": error_message}

        return {"success": False, "error": "Unexpected end of retry loop"}

    async def _execute_step_action(
        self,
        action: dict[str, Any],
        agent_type: AgentType,
        context: dict[str, Any],
        timeout_minutes: int
    ) -> dict[str, Any]:
        """Execute a step action using the appropriate agent."""
        action_type = action.get("type", "task")

        if action_type == "task":
            # Execute as agent task
            return await self._execute_agent_task_action(action, agent_type, context, timeout_minutes)
        elif action_type == "api_call":
            # Execute as API call
            return await self._execute_api_call_action(action, context)
        elif action_type == "script":
            # Execute as script
            return await self._execute_script_action(action, context)
        else:
            return {"success": False, "error": f"Unsupported action type: {action_type}"}

    async def _execute_agent_task_action(
        self,
        action: dict[str, Any],
        agent_type: AgentType,
        context: dict[str, Any],
        timeout_minutes: int
    ) -> dict[str, Any]:
        """Execute action using agent task manager."""
        try:
            # For now, simulate agent task execution
            # In real implementation, this would create and execute an agent task

            task_description = action.get("description", "Saga step execution")

            logger.debug(f"Executing agent task: {task_description}")

            # Simulate work
            await asyncio.sleep(action.get("duration_seconds", 2))

            # Simulate success/failure based on action configuration
            success_rate = action.get("success_rate", 0.9)
            import random
            success = random.random() < success_rate

            if success:
                return {
                    "success": True,
                    "agent_type": agent_type.value,
                    "task_description": task_description,
                    "result": "Task completed successfully",
                    "execution_time_seconds": action.get("duration_seconds", 2)
                }
            else:
                return {
                    "success": False,
                    "error": "Simulated task failure",
                    "agent_type": agent_type.value
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_api_call_action(
        self,
        action: dict[str, Any],
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute action as HTTP API call."""
        # Placeholder implementation for API calls
        # In real implementation, this would make actual HTTP requests

        url = action.get("url")
        method = action.get("method", "POST")

        logger.debug(f"Executing API call: {method} {url}")

        # Simulate API call
        await asyncio.sleep(1)

        return {
            "success": True,
            "method": method,
            "url": url,
            "result": "API call completed successfully"
        }

    async def _execute_script_action(
        self,
        action: dict[str, Any],
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute action as script."""
        # Placeholder implementation for script execution
        # In real implementation, this would execute actual scripts safely

        script = action.get("script", "")

        logger.debug("Executing script action")

        # Simulate script execution
        await asyncio.sleep(0.5)

        return {
            "success": True,
            "script_length": len(script),
            "result": "Script executed successfully"
        }

    async def _execute_compensation(self, saga_transaction: SagaTransaction) -> dict[str, Any]:
        """Execute compensation actions for failed saga."""
        logger.info(f"Starting compensation for saga {saga_transaction.saga_id}")

        saga_transaction.status = SagaStatus.COMPENSATING
        await self._store_saga_transaction(saga_transaction)

        # Publish compensation started event
        await self.event_bus.publish({
            "event": "saga_compensation_started",
            "saga_id": saga_transaction.saga_id,
            "compensation_steps": len(saga_transaction.compensation_order),
            "timestamp": datetime.utcnow().isoformat()
        })

        compensation_results = []
        successful_compensations = 0

        # Execute compensation actions in reverse order
        for step_id in saga_transaction.compensation_order:
            step_execution = saga_transaction.execution_state[step_id]

            if step_execution.definition.compensation_action:
                try:
                    logger.info(f"Compensating step: {step_execution.definition.name}")

                    compensation_result = await self._execute_step_action(
                        step_execution.definition.compensation_action,
                        step_execution.definition.agent_type,
                        saga_transaction.context,
                        step_execution.definition.timeout_minutes
                    )

                    if compensation_result.get("success"):
                        step_execution.status = SagaStepStatus.COMPENSATED
                        successful_compensations += 1
                    else:
                        step_execution.status = SagaStepStatus.COMPENSATION_FAILED

                    step_execution.compensation_result = compensation_result
                    compensation_results.append({
                        "step_id": step_id,
                        "success": compensation_result.get("success", False),
                        "result": compensation_result
                    })

                except Exception as e:
                    logger.error(f"Compensation failed for step {step_id}: {e}")
                    step_execution.status = SagaStepStatus.COMPENSATION_FAILED
                    compensation_results.append({
                        "step_id": step_id,
                        "success": False,
                        "error": str(e)
                    })

        compensation_success = successful_compensations == len(saga_transaction.compensation_order)

        # Publish compensation completed event
        await self.event_bus.publish({
            "event": "saga_compensation_completed",
            "saga_id": saga_transaction.saga_id,
            "success": compensation_success,
            "successful_compensations": successful_compensations,
            "total_compensations": len(saga_transaction.compensation_order),
            "timestamp": datetime.utcnow().isoformat()
        })

        return {
            "success": compensation_success,
            "successful_compensations": successful_compensations,
            "total_compensations": len(saga_transaction.compensation_order),
            "compensation_results": compensation_results
        }

    async def _trigger_choreography_step(
        self,
        saga_transaction: SagaTransaction,
        step_definition: SagaStepDefinition
    ):
        """Trigger a step in choreography mode."""
        # In choreography mode, we publish an event that the responsible service should handle
        await self.event_bus.publish({
            "event": "saga_step_trigger",
            "saga_id": saga_transaction.saga_id,
            "step_id": step_definition.step_id,
            "step_definition": step_definition.model_dump(),
            "context": saga_transaction.context,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def _trigger_dependent_steps(self, saga_transaction: SagaTransaction, completed_step_id: str):
        """Trigger steps that depend on the completed step."""
        for step_def in saga_transaction.steps:
            if completed_step_id in step_def.dependencies:
                # Check if all dependencies are satisfied
                step_execution = saga_transaction.execution_state[step_def.step_id]
                all_deps_satisfied = all(
                    saga_transaction.execution_state[dep].status == SagaStepStatus.COMPLETED
                    for dep in step_def.dependencies
                )

                if all_deps_satisfied and step_execution.status == SagaStepStatus.PENDING:
                    await self._trigger_choreography_step(saga_transaction, step_def)

    async def _store_saga_transaction(self, saga_transaction: SagaTransaction):
        """Store saga transaction state in Redis."""
        key = f"saga_transaction:{saga_transaction.saga_id}"
        ttl = 24 * 60 * 60  # 24 hours
        self.storage.setex(key, ttl, saga_transaction.model_dump_json())

    async def get_saga_transaction(self, saga_id: str) -> Optional[SagaTransaction]:
        """Retrieve saga transaction from Redis."""
        key = f"saga_transaction:{saga_id}"
        data = self.storage.get(key)
        if data:
            return SagaTransaction.model_validate_json(data)
        return None


# Convenience functions for saga execution
async def start_distributed_saga(
    saga_definition: dict[str, Any],
    coordination_mode: str = "orchestration",
    config: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Start a distributed saga transaction."""
    from utils.temporal_client import get_temporal_client

    client = get_temporal_client()
    workflow = SagaWorkflow(coordination_mode=coordination_mode)

    result = await client.start_workflow(
        workflow_class=workflow.__class__,
        workflow_args={
            "saga_definition": saga_definition,
            "config": config or {}
        },
        workflow_id=f"saga-{saga_definition.get('saga_id', uuid4())}",
        timeout_seconds=config.get("timeout_seconds", 3600) if config else 3600
    )

    return result.model_dump()


logger.info("Saga workflow module initialized")
