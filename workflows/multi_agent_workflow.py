"""
Multi-Agent Project Workflow

This module implements complex, long-running workflows that coordinate multiple agents
across different phases of project development with automatic fault tolerance,
compensation actions, and human oversight integration.

Features:
- Multi-phase project orchestration
- Parallel agent coordination
- Automatic retry and compensation
- Progress tracking and monitoring
- Integration with existing agent infrastructure
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel

from tools.shared.agent_models import AgentType
from utils.agent_manager import get_task_manager
from utils.event_bus import get_event_bus
from utils.storage_backend import get_storage_backend
from utils.temporal_client import BaseWorkflow

logger = logging.getLogger(__name__)


class ProjectSpec(BaseModel):
    """Project specification for multi-agent workflow."""
    project_id: str
    name: str
    description: str
    requirements: list[str]
    constraints: dict[str, Any]
    success_criteria: list[str]
    estimated_duration_hours: int
    priority: str = "normal"  # low, normal, high, critical
    agents_required: list[AgentType]
    approval_gates: list[str] = []  # Stages requiring human approval


class ProjectPhase(BaseModel):
    """Individual phase within a project workflow."""
    phase_id: str
    name: str
    description: str
    dependencies: list[str] = []  # Phase IDs this phase depends on
    estimated_duration_minutes: int
    required_agents: list[AgentType]
    tasks: list[dict[str, Any]]
    success_criteria: list[str]
    requires_approval: bool = False


class AgentAssignment(BaseModel):
    """Assignment of a task to a specific agent."""
    assignment_id: str
    agent_type: AgentType
    task_description: str
    context: dict[str, Any]
    estimated_duration_minutes: int
    priority: str
    dependencies: list[str] = []
    retry_policy: dict[str, Any] = {}


class ProjectWorkflowState(BaseModel):
    """Current state of a project workflow."""
    project_id: str
    workflow_id: str
    status: str  # initializing, running, paused, completed, failed, cancelled
    current_phase: Optional[str] = None
    completed_phases: list[str] = []
    failed_phases: list[str] = []
    active_assignments: dict[str, str] = {}  # assignment_id -> agent_task_id
    completed_assignments: list[str] = []
    failed_assignments: list[str] = []
    start_time: datetime
    end_time: Optional[datetime] = None
    total_agents_used: int = 0
    success_rate: float = 0.0
    error_log: list[dict[str, Any]] = []


class MultiAgentProjectWorkflow(BaseWorkflow):
    """
    Complex multi-agent project orchestration workflow.

    This workflow manages the entire lifecycle of a project, coordinating
    multiple agents across different phases with fault tolerance and human
    approval gates where needed.
    """

    def __init__(self):
        super().__init__()
        self.task_manager = get_task_manager()
        self.storage = get_storage_backend()
        self.event_bus = get_event_bus()

    async def orchestrate(self, workflow_args: dict[str, Any]) -> dict[str, Any]:
        """
        Main orchestration logic for multi-agent project workflows.

        Args:
            workflow_args: Contains 'project_spec' and optional 'config'

        Returns:
            Dict containing project results, metrics, and artifacts
        """
        project_spec = ProjectSpec(**workflow_args["project_spec"])
        config = workflow_args.get("config", {})

        workflow_id = self.context.workflow_id if self.context else f"project-{uuid4()}"

        # Initialize project state
        project_state = ProjectWorkflowState(
            project_id=project_spec.project_id,
            workflow_id=workflow_id,
            status="initializing",
            start_time=datetime.utcnow()
        )

        try:
            logger.info(f"Starting multi-agent project workflow: {project_spec.name}")

            # Phase 1: Project Analysis & Planning
            project_state.status = "running"
            await self._store_project_state(project_state)

            analysis_result = await self._execute_analysis_phase(project_spec, project_state)
            if not analysis_result["success"]:
                raise Exception(f"Analysis phase failed: {analysis_result.get('error')}")

            project_state.completed_phases.append("analysis")
            project_state.current_phase = "planning"
            await self._store_project_state(project_state)

            # Generate detailed project plan
            project_phases = await self._generate_project_phases(
                project_spec, analysis_result, config
            )

            # Phase 2: Execute Project Phases
            for phase in project_phases:
                if phase.requires_approval:
                    # Human approval gate
                    approved = await self.wait_for_approval(
                        stage=f"phase_{phase.phase_id}",
                        description=f"Approve execution of {phase.name}",
                        context={
                            "phase": phase.model_dump(),
                            "project_state": project_state.model_dump()
                        },
                        timeout_seconds=config.get("approval_timeout", 24 * 60 * 60)
                    )

                    if not approved:
                        project_state.status = "approval_rejected"
                        project_state.end_time = datetime.utcnow()
                        await self._store_project_state(project_state)

                        return {
                            "status": "approval_rejected",
                            "phase": phase.phase_id,
                            "project_state": project_state.model_dump()
                        }

                # Execute phase
                project_state.current_phase = phase.phase_id
                await self._store_project_state(project_state)

                phase_result = await self._execute_project_phase(
                    phase, project_spec, project_state
                )

                if phase_result["success"]:
                    project_state.completed_phases.append(phase.phase_id)
                    project_state.total_agents_used += phase_result.get("agents_used", 0)
                else:
                    project_state.failed_phases.append(phase.phase_id)
                    project_state.error_log.append({
                        "phase": phase.phase_id,
                        "error": phase_result.get("error"),
                        "timestamp": datetime.utcnow().isoformat()
                    })

                    # Decide if we should continue or fail the workflow
                    if phase_result.get("critical", True):
                        raise Exception(f"Critical phase {phase.phase_id} failed: {phase_result.get('error')}")

                await self._store_project_state(project_state)

            # Phase 3: Final Integration & Validation
            integration_result = await self._execute_integration_phase(
                project_spec, project_state, config
            )

            if integration_result["success"]:
                project_state.completed_phases.append("integration")
                project_state.status = "completed"
                project_state.success_rate = len(project_state.completed_phases) / (
                    len(project_state.completed_phases) + len(project_state.failed_phases)
                )
            else:
                project_state.status = "integration_failed"
                project_state.error_log.append({
                    "phase": "integration",
                    "error": integration_result.get("error"),
                    "timestamp": datetime.utcnow().isoformat()
                })

            project_state.end_time = datetime.utcnow()
            project_state.current_phase = None
            await self._store_project_state(project_state)

            # Publish workflow completion event
            await self.event_bus.publish({
                "event": "multi_agent_project_completed",
                "project_id": project_spec.project_id,
                "workflow_id": workflow_id,
                "status": project_state.status,
                "execution_time_seconds": (
                    project_state.end_time - project_state.start_time
                ).total_seconds(),
                "agents_used": project_state.total_agents_used,
                "success_rate": project_state.success_rate,
                "timestamp": project_state.end_time.isoformat()
            })

            return {
                "status": project_state.status,
                "project_id": project_spec.project_id,
                "workflow_id": workflow_id,
                "completed_phases": project_state.completed_phases,
                "failed_phases": project_state.failed_phases,
                "agents_used": project_state.total_agents_used,
                "success_rate": project_state.success_rate,
                "execution_time_seconds": (
                    project_state.end_time - project_state.start_time
                ).total_seconds(),
                "integration_result": integration_result,
                "project_state": project_state.model_dump()
            }

        except Exception as e:
            # Handle workflow failure with compensation
            logger.error(f"Multi-agent project workflow failed: {e}")

            project_state.status = "failed"
            project_state.end_time = datetime.utcnow()
            project_state.error_log.append({
                "phase": "workflow",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            await self._store_project_state(project_state)

            # Execute compensation actions
            await self._execute_compensation_actions(project_state)

            # Publish failure event
            await self.event_bus.publish({
                "event": "multi_agent_project_failed",
                "project_id": project_spec.project_id,
                "workflow_id": workflow_id,
                "error": str(e),
                "project_state": project_state.model_dump(),
                "timestamp": project_state.end_time.isoformat()
            })

            return {
                "status": "failed",
                "error": str(e),
                "project_id": project_spec.project_id,
                "workflow_id": workflow_id,
                "project_state": project_state.model_dump()
            }

    async def _execute_analysis_phase(
        self, project_spec: ProjectSpec, project_state: ProjectWorkflowState
    ) -> dict[str, Any]:
        """Execute project analysis phase using analysis agent."""
        try:
            logger.info(f"Executing analysis phase for project {project_spec.project_id}")

            # Create analysis task for analyze tool
            self._build_analysis_prompt(project_spec)

            # For now, simulate analysis result
            # In real implementation, this would delegate to the analyze tool
            await asyncio.sleep(2)  # Simulate analysis time

            analysis_result = {
                "success": True,
                "components": [
                    {
                        "name": "Backend API",
                        "description": "REST API implementation",
                        "preferred_agent": AgentType.CLAUDE,
                        "estimated_hours": 4,
                        "implementation_task": {
                            "description": "Implement REST API endpoints",
                            "files": ["api/main.py", "api/models.py"],
                            "requirements": project_spec.requirements
                        }
                    },
                    {
                        "name": "Frontend UI",
                        "description": "User interface implementation",
                        "preferred_agent": AgentType.CLAUDE,
                        "estimated_hours": 3,
                        "implementation_task": {
                            "description": "Implement user interface",
                            "files": ["ui/index.html", "ui/app.js"],
                            "requirements": project_spec.requirements
                        }
                    },
                    {
                        "name": "Database Schema",
                        "description": "Database design and migration",
                        "preferred_agent": AgentType.CLAUDE,
                        "estimated_hours": 2,
                        "implementation_task": {
                            "description": "Design and implement database schema",
                            "files": ["db/schema.sql", "db/migrations.sql"],
                            "requirements": project_spec.requirements
                        }
                    }
                ],
                "test_requirements": [
                    "Unit tests for all API endpoints",
                    "Integration tests for database operations",
                    "UI automation tests for key workflows"
                ],
                "deployment_strategy": "containerized",
                "risk_factors": [
                    "Complex data relationships",
                    "Performance requirements",
                    "Security considerations"
                ]
            }

            # Publish analysis completion event
            await self.event_bus.publish({
                "event": "project_analysis_completed",
                "project_id": project_spec.project_id,
                "workflow_id": project_state.workflow_id,
                "components_identified": len(analysis_result["components"]),
                "timestamp": datetime.utcnow().isoformat()
            })

            return analysis_result

        except Exception as e:
            logger.error(f"Analysis phase failed: {e}")
            return {"success": False, "error": str(e)}

    def _build_analysis_prompt(self, project_spec: ProjectSpec) -> str:
        """Build analysis prompt for the analyze tool."""
        return f"""
Analyze the following project for multi-agent implementation:

Project: {project_spec.name}
Description: {project_spec.description}

Requirements:
{chr(10).join(f"- {req}" for req in project_spec.requirements)}

Constraints:
{json.dumps(project_spec.constraints, indent=2)}

Success Criteria:
{chr(10).join(f"- {criteria}" for criteria in project_spec.success_criteria)}

Please provide:
1. Component breakdown suitable for different agents
2. Implementation strategy and timeline
3. Risk assessment and mitigation strategies
4. Testing and validation requirements
5. Deployment considerations
"""

    async def _generate_project_phases(
        self,
        project_spec: ProjectSpec,
        analysis_result: dict[str, Any],
        config: dict[str, Any]
    ) -> list[ProjectPhase]:
        """Generate detailed project phases based on analysis."""
        phases = []

        # Implementation phase for each component
        for _i, component in enumerate(analysis_result["components"]):
            phase = ProjectPhase(
                phase_id=f"implement_{component['name'].lower().replace(' ', '_')}",
                name=f"Implement {component['name']}",
                description=component["description"],
                dependencies=[],  # Could be configured based on component dependencies
                estimated_duration_minutes=component["estimated_hours"] * 60,
                required_agents=[component["preferred_agent"]],
                tasks=[component["implementation_task"]],
                success_criteria=[f"{component['name']} successfully implemented"],
                requires_approval=component["name"] in project_spec.approval_gates
            )
            phases.append(phase)

        # Testing phase
        if analysis_result.get("test_requirements"):
            testing_phase = ProjectPhase(
                phase_id="testing",
                name="Testing & Validation",
                description="Execute comprehensive testing suite",
                dependencies=[p.phase_id for p in phases],  # Depends on all implementation phases
                estimated_duration_minutes=120,  # 2 hours
                required_agents=[AgentType.CLAUDE],  # Use Claude for test generation
                tasks=[{
                    "description": "Generate and execute comprehensive tests",
                    "test_requirements": analysis_result["test_requirements"]
                }],
                success_criteria=["All tests pass", "Code coverage >= 80%"],
                requires_approval="testing" in project_spec.approval_gates
            )
            phases.append(testing_phase)

        return phases

    async def _execute_project_phase(
        self,
        phase: ProjectPhase,
        project_spec: ProjectSpec,
        project_state: ProjectWorkflowState
    ) -> dict[str, Any]:
        """Execute a single project phase."""
        try:
            logger.info(f"Executing phase {phase.name} for project {project_spec.project_id}")

            # Publish phase started event
            await self.event_bus.publish({
                "event": "project_phase_started",
                "project_id": project_spec.project_id,
                "workflow_id": project_state.workflow_id,
                "phase_id": phase.phase_id,
                "phase_name": phase.name,
                "timestamp": datetime.utcnow().isoformat()
            })

            phase_start_time = datetime.utcnow()
            assignments_created = []
            agents_used = 0

            # Create agent assignments for phase tasks
            for task in phase.tasks:
                for agent_type in phase.required_agents:
                    assignment = AgentAssignment(
                        assignment_id=f"assignment-{uuid4()}",
                        agent_type=agent_type,
                        task_description=task.get("description", phase.description),
                        context={
                            "project_spec": project_spec.model_dump(),
                            "phase": phase.model_dump(),
                            "task": task
                        },
                        estimated_duration_minutes=phase.estimated_duration_minutes // len(phase.required_agents),
                        priority=project_spec.priority
                    )
                    assignments_created.append(assignment)

            # Execute assignments in parallel where possible
            assignment_results = await asyncio.gather(
                *[self._execute_agent_assignment(assignment) for assignment in assignments_created],
                return_exceptions=True
            )

            # Process results
            successful_assignments = 0
            failed_assignments = 0

            for i, result in enumerate(assignment_results):
                if isinstance(result, Exception):
                    failed_assignments += 1
                    logger.error(f"Assignment {assignments_created[i].assignment_id} failed: {result}")
                elif result.get("success"):
                    successful_assignments += 1
                    agents_used += 1
                else:
                    failed_assignments += 1
                    logger.warning(f"Assignment {assignments_created[i].assignment_id} completed unsuccessfully")

            phase_duration = (datetime.utcnow() - phase_start_time).total_seconds()

            # Determine phase success
            success_rate = successful_assignments / len(assignments_created) if assignments_created else 0
            phase_success = success_rate >= 0.8  # 80% success threshold

            # Publish phase completed event
            await self.event_bus.publish({
                "event": "project_phase_completed",
                "project_id": project_spec.project_id,
                "workflow_id": project_state.workflow_id,
                "phase_id": phase.phase_id,
                "success": phase_success,
                "duration_seconds": phase_duration,
                "agents_used": agents_used,
                "success_rate": success_rate,
                "timestamp": datetime.utcnow().isoformat()
            })

            return {
                "success": phase_success,
                "phase_id": phase.phase_id,
                "duration_seconds": phase_duration,
                "agents_used": agents_used,
                "success_rate": success_rate,
                "successful_assignments": successful_assignments,
                "failed_assignments": failed_assignments,
                "assignment_results": assignment_results
            }

        except Exception as e:
            logger.error(f"Phase {phase.phase_id} execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "phase_id": phase.phase_id,
                "critical": True
            }

    async def _execute_agent_assignment(self, assignment: AgentAssignment) -> dict[str, Any]:
        """Execute a single agent assignment."""
        try:
            logger.debug(f"Executing agent assignment {assignment.assignment_id}")

            # For now, simulate agent task execution
            # In real implementation, this would use the agent task manager
            await asyncio.sleep(1)  # Simulate work

            # Simulate success/failure based on task complexity
            import random
            success = random.random() > 0.1  # 90% success rate

            if success:
                return {
                    "success": True,
                    "assignment_id": assignment.assignment_id,
                    "agent_type": assignment.agent_type.value,
                    "result": "Task completed successfully",
                    "duration_seconds": 60  # Simulated duration
                }
            else:
                return {
                    "success": False,
                    "assignment_id": assignment.assignment_id,
                    "agent_type": assignment.agent_type.value,
                    "error": "Simulated task failure",
                    "duration_seconds": 30
                }

        except Exception as e:
            logger.error(f"Agent assignment {assignment.assignment_id} failed: {e}")
            return {
                "success": False,
                "assignment_id": assignment.assignment_id,
                "error": str(e)
            }

    async def _execute_integration_phase(
        self,
        project_spec: ProjectSpec,
        project_state: ProjectWorkflowState,
        config: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute final integration and validation phase."""
        try:
            logger.info(f"Executing integration phase for project {project_spec.project_id}")

            # Simulate integration testing
            await asyncio.sleep(2)

            # Check if all critical phases completed
            critical_phases_completed = all(
                phase in project_state.completed_phases
                for phase in ["implement_backend_api", "implement_frontend_ui"]  # Example critical phases
            )

            integration_success = (
                critical_phases_completed and
                len(project_state.failed_phases) <= 1  # Allow 1 non-critical failure
            )

            return {
                "success": integration_success,
                "tests_passed": integration_success,
                "critical_phases_completed": critical_phases_completed,
                "deployment_ready": integration_success,
                "validation_results": {
                    "completed_phases": len(project_state.completed_phases),
                    "failed_phases": len(project_state.failed_phases),
                    "overall_success_rate": project_state.success_rate
                }
            }

        except Exception as e:
            logger.error(f"Integration phase failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _execute_compensation_actions(self, project_state: ProjectWorkflowState):
        """Execute compensation actions when workflow fails."""
        logger.info(f"Executing compensation actions for project {project_state.project_id}")

        # Cancel any active agent tasks
        for assignment_id, agent_task_id in project_state.active_assignments.items():
            try:
                # In real implementation, cancel the agent task
                logger.info(f"Cancelling agent task {agent_task_id} for assignment {assignment_id}")
            except Exception as e:
                logger.warning(f"Failed to cancel agent task {agent_task_id}: {e}")

        # Cleanup temporary resources
        try:
            # Cleanup project workspace, temporary files, etc.
            logger.info("Cleaning up project resources")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

        # Publish compensation event
        await self.event_bus.publish({
            "event": "project_compensation_executed",
            "project_id": project_state.project_id,
            "workflow_id": project_state.workflow_id,
            "cancelled_assignments": len(project_state.active_assignments),
            "timestamp": datetime.utcnow().isoformat()
        })

    async def _store_project_state(self, project_state: ProjectWorkflowState):
        """Store project state in Redis."""
        key = f"project_state:{project_state.project_id}"
        self.storage.setex(key, 24 * 60 * 60, project_state.model_dump_json())  # 24 hour TTL

    async def get_project_state(self, project_id: str) -> Optional[ProjectWorkflowState]:
        """Retrieve project state from Redis."""
        key = f"project_state:{project_id}"
        data = self.storage.get(key)
        if data:
            return ProjectWorkflowState.model_validate_json(data)
        return None


# Convenience functions for workflow execution
async def start_multi_agent_project(
    project_spec: ProjectSpec,
    config: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Start a multi-agent project workflow."""
    from utils.temporal_client import get_temporal_client

    client = get_temporal_client()
    workflow = MultiAgentProjectWorkflow()

    result = await client.start_workflow(
        workflow_class=workflow.__class__,
        workflow_args={
            "project_spec": project_spec.model_dump(),
            "config": config or {}
        },
        workflow_id=f"project-{project_spec.project_id}-{uuid4()}",
        timeout_seconds=project_spec.estimated_duration_hours * 3600 + 3600  # Add 1 hour buffer
    )

    return result.model_dump()


logger.info("Multi-agent project workflow module initialized")
