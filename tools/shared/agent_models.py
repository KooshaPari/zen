"""
Agent Orchestration Data Models

This module defines the core data models for agent orchestration capabilities,
including agent definitions, task management, and result handling.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """Supported agent types from AgentAPI and LLM models."""

    # External AgentAPI executables
    CLAUDE = "claude"
    GOOSE = "goose"
    AIDER = "aider"
    CODEX = "codex"
    GEMINI = "gemini"
    AMP = "amp"
    CURSOR_AGENT = "cursor-agent"
    CURSOR = "cursor"
    AUGGIE = "auggie"
    CRUSH = "crush"
    CUSTOM = "custom"

    # Direct LLM model execution (via providers)
    LLM = "llm"

    # Sophisticated LangGraph workflow orchestration with real LLM models
    AEGIS = "aegis"


class AgentCapability(BaseModel):
    """Describes a specific capability of an agent."""

    name: str = Field(..., description="Name of the capability")
    description: str = Field(..., description="Description of what this capability does")
    use_cases: list[str] = Field(default_factory=list, description="Common use cases for this capability")
    strengths: list[str] = Field(default_factory=list, description="What this agent excels at")
    limitations: list[str] = Field(default_factory=list, description="Known limitations or constraints")


class AgentDefinition(BaseModel):
    """Complete definition of an available agent."""

    agent_type: AgentType = Field(..., description="Type of agent")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Detailed description of the agent")
    capabilities: list[AgentCapability] = Field(default_factory=list, description="Agent capabilities")
    command_template: str = Field(..., description="Command template for starting this agent")
    default_args: list[str] = Field(default_factory=list, description="Default arguments")
    required_env_vars: list[str] = Field(default_factory=list, description="Required environment variables")
    optional_env_vars: list[str] = Field(default_factory=list, description="Optional environment variables")
    port_range: tuple[int, int] = Field(default=(3284, 3384), description="Port range for AgentAPI server")
    timeout_seconds: int = Field(default=300, description="Default timeout for operations")


class TaskStatus(str, Enum):
    """Status of an agent task."""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class AgentTaskRequest(BaseModel):
    """Request to execute a task with an agent or LLM model."""

    agent_type: AgentType = Field(..., description="Type of agent to use")
    task_description: str = Field(..., description="Description of the task to perform")
    message: str = Field(..., description="Message to send to the agent")

    # Agent-specific parameters
    agent_args: list[str] = Field(default_factory=list, description="Additional agent arguments")
    env_vars: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    timeout_seconds: Optional[int] = Field(None, description="Task timeout override")
    working_directory: Optional[str] = Field(None, description="Working directory for the agent")
    files: list[str] = Field(default_factory=list, description="Files to make available to the agent")

    # LLM-specific parameters (when agent_type=LLM)
    model: Optional[str] = Field(None, description="Model name for LLM agent type")
    system_prompt: Optional[str] = Field(None, description="System prompt for LLM agent type")
    temperature: Optional[float] = Field(None, description="Temperature for LLM agent type")
    max_tokens: Optional[int] = Field(None, description="Max tokens for LLM agent type")

    # Workflow parameters
    workflow: Optional[str] = Field(None, description="Workflow type (analyze, codereview, refactor, etc.)")
    workflow_params: dict[str, Any] = Field(default_factory=dict, description="Workflow-specific parameters")



class ContentPart(BaseModel):
    type: str = Field(..., description="Part type: text|image_url|tool_call|tool_result|file|file_inline")
    text: Optional[str] = Field(None, description="Text content for type=text")
    image_url: Optional[str] = Field(None, description="Image URL for type=image_url")
    tool_name: Optional[str] = Field(None, description="Tool name for type=tool_call")
    tool_args: Optional[dict[str, Any]] = Field(None, description="Arguments for tool_call")
    tool_result: Optional[dict[str, Any]] = Field(None, description="Structured result for type=tool_result")
    file_name: Optional[str] = Field(None, description="File name for type=file or file_inline")
    file_url: Optional[str] = Field(None, description="File URL for type=file")
    file_bytes_b64: Optional[str] = Field(None, description="Base64 for inline file content (type=file_inline)")


class Message(BaseModel):
    """Structured message in a conversation history.

    We allow either `content` or `message` for backward compatibility.
    `time` is ISO8601 if present.
    """

    role: str = Field(..., description="Message role: system|user|assistant|tool")
    content: Any | list[ContentPart] | None = Field(None, description="Message content; string or structured parts")
    message: Optional[str] = Field(None, description="Legacy text field; prefer `content`")
    time: Optional[str] = Field(None, description="Timestamp in ISO8601, if available")

class AgentTaskResult(BaseModel):
    """Result of an agent task execution."""

    task_id: str = Field(..., description="Unique task identifier")
    agent_type: AgentType = Field(..., description="Type of agent used")
    status: TaskStatus = Field(..., description="Final status of the task")
    messages: list[Message] = Field(default_factory=list, description="Conversation messages")
    output: str = Field(default="", description="Final output from the agent")
    error: Optional[str] = Field(None, description="Error message if task failed")
    started_at: datetime = Field(..., description="When the task started")
    completed_at: Optional[datetime] = Field(None, description="When the task completed")
    duration_seconds: Optional[float] = Field(None, description="Task duration in seconds")
    agent_port: Optional[int] = Field(None, description="Port used for AgentAPI communication")


class AgentTask(BaseModel):
    """Complete agent task with request, status, and result."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique task identifier")
    request: AgentTaskRequest = Field(..., description="Original task request")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    result: Optional[AgentTaskResult] = Field(None, description="Task result when completed")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Last update timestamp"
    )
    agent_port: Optional[int] = Field(None, description="Port assigned for AgentAPI communication")
    process_id: Optional[int] = Field(None, description="Process ID of the agent")


class BatchTaskRequest(BaseModel):
    """Request to execute multiple tasks in parallel."""

    tasks: list[AgentTaskRequest] = Field(..., description="List of tasks to execute")
    coordination_strategy: str = Field(default="parallel", description="How to coordinate tasks (parallel, sequential)")
    max_concurrent: int = Field(default=5, description="Maximum concurrent tasks")
    timeout_seconds: Optional[int] = Field(None, description="Overall batch timeout")
    fail_fast: bool = Field(default=False, description="Stop on first failure")


class BatchTaskResult(BaseModel):
    """Result of a batch task execution."""

    batch_id: str = Field(..., description="Unique batch identifier")
    tasks: list[AgentTask] = Field(..., description="Individual task results")
    status: TaskStatus = Field(..., description="Overall batch status")
    started_at: datetime = Field(..., description="When the batch started")
    completed_at: Optional[datetime] = Field(None, description="When the batch completed")
    duration_seconds: Optional[float] = Field(None, description="Batch duration in seconds")
    successful_count: int = Field(default=0, description="Number of successful tasks")
    failed_count: int = Field(default=0, description="Number of failed tasks")


class AgentRegistryEntry(BaseModel):
    """Entry in the agent registry."""

    agent_type: AgentType = Field(..., description="Type of agent")
    definition: AgentDefinition = Field(..., description="Agent definition")
    available: bool = Field(default=True, description="Whether agent is currently available")
    last_checked: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Last availability check"
    )
    installation_path: Optional[str] = Field(None, description="Path to agent executable")
    version: Optional[str] = Field(None, description="Agent version if available")


# Type aliases for convenience
AgentTaskId = str
BatchTaskId = str
AgentPort = int
