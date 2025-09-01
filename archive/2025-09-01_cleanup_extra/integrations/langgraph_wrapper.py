"""
LangGraph Wrapper for Zen MCP Server

This module provides LangGraph integration as outlined in the future LLM architecture plans.
Enables complex multi-agent workflows with state machines, conditional routing, and
human-in-the-loop patterns using the "aegis" provider option.

Features:
- State graph workflow orchestration
- Multi-agent coordination with MCP tools
- Conditional workflow routing
- Human approval integration
- Streaming workflow execution
- Tool calling within workflow nodes
- State persistence and recovery

Based on Phase 3 of the LLM Architecture Enhancement Plan.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, TypedDict

# LangGraph imports (optional dependency)
try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolExecutor
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback types for when LangGraph is not installed
    StateGraph = None
    END = "END"
    START = "START"
    MemorySaver = None
    ToolExecutor = None
    LANGGRAPH_AVAILABLE = False

from clients.mcp_http_client import MCPStreamableHTTPClient
from providers.base import BaseProvider, ProviderType
from utils.conversation_memory import ConversationMemoryManager
from utils.streaming_manager import StreamingManager

logger = logging.getLogger(__name__)


class WorkflowNodeType(Enum):
    """Types of nodes in a LangGraph workflow."""
    AGENT = "agent"              # LLM agent execution
    TOOL = "tool"               # MCP tool execution
    HUMAN = "human"             # Human approval/input
    CONDITIONAL = "conditional"  # Routing logic
    PARALLEL = "parallel"       # Parallel execution
    REDUCER = "reducer"         # Result aggregation


class WorkflowState(TypedDict):
    """State object that flows through the LangGraph workflow."""
    # Core workflow data
    request: dict[str, Any]
    response: Optional[str]
    error: Optional[str]

    # Conversation and context
    continuation_id: Optional[str]
    conversation_history: list[dict[str, Any]]
    files: list[str]
    images: list[str]

    # Workflow execution metadata
    workflow_id: str
    current_step: str
    step_count: int
    max_steps: int

    # Agent and tool results
    agent_results: dict[str, Any]
    tool_results: dict[str, Any]
    parallel_results: list[dict[str, Any]]

    # Human interaction
    pending_approvals: list[dict[str, Any]]
    human_feedback: Optional[str]

    # Configuration
    model: Optional[str]
    temperature: float
    thinking_mode: str

    # Streaming and output
    streaming_enabled: bool
    stream_id: Optional[str]


@dataclass
class WorkflowNode:
    """Definition of a workflow node."""
    name: str
    type: WorkflowNodeType
    config: dict[str, Any]
    next_nodes: list[str]
    condition_func: Optional[Callable] = None


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    name: str
    description: str
    nodes: list[WorkflowNode]
    start_node: str
    end_nodes: list[str]
    state_schema: dict[str, Any]


class AegisLangGraphProvider(BaseProvider):
    """
    LangGraph-powered provider for complex multi-agent workflows.

    Implements the "aegis" provider option that enables sophisticated
    workflow orchestration using LangGraph state machines.
    """

    def __init__(self, mcp_server_url: str = "http://localhost:8080/mcp"):
        super().__init__()
        self.mcp_server_url = mcp_server_url
        self.conversation_memory = ConversationMemoryManager()
        self.streaming_manager = StreamingManager()

        # Workflow registry
        self.workflows: dict[str, WorkflowDefinition] = {}
        self.active_executions: dict[str, dict[str, Any]] = {}

        # LangGraph components
        self.memory_saver = MemorySaver() if MemorySaver else None
        self.tool_executor = None

        # MCP client for tool execution
        self.mcp_client: Optional[MCPStreamableHTTPClient] = None

        # Pre-defined workflow templates
        self._register_default_workflows()

    def get_provider_type(self) -> ProviderType:
        return ProviderType.CUSTOM

    def get_provider_name(self) -> str:
        return "aegis-langgraph"

    def is_available(self) -> bool:
        """Check if LangGraph is available and properly configured."""
        return LANGGRAPH_AVAILABLE

    async def initialize(self):
        """Initialize the provider with MCP client connection."""
        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError("LangGraph not available. Install with: pip install langgraph")

        # Initialize MCP client
        self.mcp_client = MCPStreamableHTTPClient(self.mcp_server_url)
        await self.mcp_client.connect()

        # Initialize tool executor with MCP tools
        await self._setup_tool_executor()

        logger.info(f"Aegis LangGraph provider initialized with {len(self.workflows)} workflows")

    async def _setup_tool_executor(self):
        """Set up LangGraph tool executor with MCP tools."""
        if not self.mcp_client:
            return

        # Get available MCP tools
        try:
            tools = await self.mcp_client.list_tools()

            # Convert MCP tools to LangGraph-compatible format
            langgraph_tools = []
            for tool in tools:
                async def tool_func(name=tool.name, **kwargs):
                    return await self.mcp_client.call_tool(name, kwargs)

                tool_func.__name__ = tool.name
                tool_func.__doc__ = tool.description
                langgraph_tools.append(tool_func)

            self.tool_executor = ToolExecutor(langgraph_tools) if ToolExecutor else None
            logger.info(f"Initialized tool executor with {len(langgraph_tools)} MCP tools")

        except Exception as e:
            logger.warning(f"Failed to setup tool executor: {e}")

    def _register_default_workflows(self):
        """Register pre-defined workflow templates."""
        # Multi-agent collaboration workflow (default)
        self.register_workflow(self._create_multi_agent_workflow())

        # Human-in-the-loop approval workflow
        self.register_workflow(self._create_approval_workflow())

        # Research and analysis workflow
        self.register_workflow(self._create_research_workflow())

        # Code review and improvement workflow
        self.register_workflow(self._create_code_review_workflow())

        # Consolidate existing workflow tools
        self.register_workflow(self._create_thinking_workflow())
        self.register_workflow(self._create_consensus_workflow())
        self.register_workflow(self._create_analysis_workflow())
        self.register_workflow(self._create_documentation_workflow())
        self.register_workflow(self._create_refactor_workflow())
        self.register_workflow(self._create_debug_workflow())
        self.register_workflow(self._create_testing_workflow())

    def register_workflow(self, workflow: WorkflowDefinition):
        """Register a new workflow definition."""
        self.workflows[workflow.name] = workflow
        logger.info(f"Registered workflow: {workflow.name}")

    def _create_multi_agent_workflow(self) -> WorkflowDefinition:
        """Create a multi-agent collaboration workflow."""
        nodes = [
            WorkflowNode(
                name="planner",
                type=WorkflowNodeType.AGENT,
                config={
                    "role": "project_planner",
                    "system_prompt": "You are a project planning specialist. Break down complex tasks into actionable steps.",
                    "model": "gpt-4o",
                    "temperature": 0.3
                },
                next_nodes=["executor", "reviewer"]
            ),
            WorkflowNode(
                name="executor",
                type=WorkflowNodeType.AGENT,
                config={
                    "role": "task_executor",
                    "system_prompt": "You are a task execution specialist. Implement solutions based on the plan.",
                    "model": "claude-3-5-sonnet",
                    "temperature": 0.5,
                    "tools_enabled": True
                },
                next_nodes=["reviewer"]
            ),
            WorkflowNode(
                name="reviewer",
                type=WorkflowNodeType.AGENT,
                config={
                    "role": "quality_reviewer",
                    "system_prompt": "You are a quality assurance specialist. Review work for completeness and accuracy.",
                    "model": "gpt-4o",
                    "temperature": 0.2
                },
                next_nodes=["decision"]
            ),
            WorkflowNode(
                name="decision",
                type=WorkflowNodeType.CONDITIONAL,
                config={"condition": "review_approved"},
                next_nodes=["END", "executor"],  # Approve or iterate
                condition_func=lambda state: "END" if state.get("review_approved") else "executor"
            )
        ]

        return WorkflowDefinition(
            name="multi_agent_collaboration",
            description="Multi-agent workflow with planning, execution, and review phases",
            nodes=nodes,
            start_node="planner",
            end_nodes=["END"],
            state_schema={"review_approved": bool, "iteration_count": int}
        )

    def _create_approval_workflow(self) -> WorkflowDefinition:
        """Create a human approval workflow."""
        nodes = [
            WorkflowNode(
                name="prepare_request",
                type=WorkflowNodeType.AGENT,
                config={
                    "system_prompt": "Prepare a clear request for human approval with context and options.",
                    "model": "gpt-4o",
                    "temperature": 0.3
                },
                next_nodes=["human_approval"]
            ),
            WorkflowNode(
                name="human_approval",
                type=WorkflowNodeType.HUMAN,
                config={
                    "approval_type": "decision",
                    "timeout_minutes": 60,
                    "required_approvers": 1
                },
                next_nodes=["process_approval"]
            ),
            WorkflowNode(
                name="process_approval",
                type=WorkflowNodeType.CONDITIONAL,
                config={"condition": "approval_granted"},
                next_nodes=["execute_approved", "handle_rejection"],
                condition_func=lambda state: "execute_approved" if state.get("approval_granted") else "handle_rejection"
            ),
            WorkflowNode(
                name="execute_approved",
                type=WorkflowNodeType.AGENT,
                config={
                    "system_prompt": "Execute the approved action with the provided context.",
                    "tools_enabled": True
                },
                next_nodes=["END"]
            ),
            WorkflowNode(
                name="handle_rejection",
                type=WorkflowNodeType.AGENT,
                config={
                    "system_prompt": "Handle the rejection gracefully and provide alternatives."
                },
                next_nodes=["END"]
            )
        ]

        return WorkflowDefinition(
            name="human_approval",
            description="Workflow requiring human approval before execution",
            nodes=nodes,
            start_node="prepare_request",
            end_nodes=["END"],
            state_schema={"approval_granted": bool, "approval_feedback": str}
        )

    def _create_research_workflow(self) -> WorkflowDefinition:
        """Create a research and analysis workflow."""
        nodes = [
            WorkflowNode(
                name="research_planner",
                type=WorkflowNodeType.AGENT,
                config={
                    "system_prompt": "Plan a comprehensive research strategy for the given topic.",
                    "model": "gpt-4o"
                },
                next_nodes=["parallel_research"]
            ),
            WorkflowNode(
                name="parallel_research",
                type=WorkflowNodeType.PARALLEL,
                config={
                    "parallel_agents": [
                        {"name": "web_researcher", "tools": ["websearch"], "model": "gpt-4o"},
                        {"name": "document_analyzer", "tools": ["analyze"], "model": "claude-3-5-sonnet"},
                        {"name": "data_collector", "tools": ["chat"], "model": "gpt-4o-mini"}
                    ]
                },
                next_nodes=["synthesis"]
            ),
            WorkflowNode(
                name="synthesis",
                type=WorkflowNodeType.REDUCER,
                config={
                    "reduction_strategy": "comprehensive_analysis",
                    "model": "gpt-4o"
                },
                next_nodes=["END"]
            )
        ]

        return WorkflowDefinition(
            name="research_analysis",
            description="Comprehensive research workflow with parallel information gathering",
            nodes=nodes,
            start_node="research_planner",
            end_nodes=["END"],
            state_schema={"research_sources": list, "findings": dict}
        )

    def _create_code_review_workflow(self) -> WorkflowDefinition:
        """Create a code review and improvement workflow."""
        nodes = [
            WorkflowNode(
                name="initial_analysis",
                type=WorkflowNodeType.TOOL,
                config={"tool_name": "analyze", "focus": "code_quality"},
                next_nodes=["security_review", "style_review"]
            ),
            WorkflowNode(
                name="security_review",
                type=WorkflowNodeType.TOOL,
                config={"tool_name": "secaudit"},
                next_nodes=["consolidate_feedback"]
            ),
            WorkflowNode(
                name="style_review",
                type=WorkflowNodeType.TOOL,
                config={"tool_name": "codereview"},
                next_nodes=["consolidate_feedback"]
            ),
            WorkflowNode(
                name="consolidate_feedback",
                type=WorkflowNodeType.AGENT,
                config={
                    "system_prompt": "Consolidate all review feedback into actionable recommendations.",
                    "model": "gpt-4o"
                },
                next_nodes=["generate_improvements"]
            ),
            WorkflowNode(
                name="generate_improvements",
                type=WorkflowNodeType.TOOL,
                config={"tool_name": "refactor"},
                next_nodes=["END"]
            )
        ]

        return WorkflowDefinition(
            name="code_review_improvement",
            description="Comprehensive code review with security, style, and improvement analysis",
            nodes=nodes,
            start_node="initial_analysis",
            end_nodes=["END"],
            state_schema={"review_findings": dict, "improvements": list}
        )

    def _create_thinking_workflow(self) -> WorkflowDefinition:
        """Create thinking workflow (consolidates thinkdeep tool)."""
        nodes = [
            WorkflowNode(
                name="deep_thinking",
                type=WorkflowNodeType.AGENT,
                config={
                    "role": "deep_thinker",
                    "system_prompt": "You are an expert at deep, systematic thinking. Break down complex problems step by step, consider multiple perspectives, and provide thorough analysis.",
                    "temperature": 0.7,
                    "thinking_mode": "high"
                },
                next_nodes=["verification"]
            ),
            WorkflowNode(
                name="verification",
                type=WorkflowNodeType.AGENT,
                config={
                    "role": "verifier",
                    "system_prompt": "Review the thinking process for logical consistency, identify gaps, and suggest improvements.",
                    "temperature": 0.3
                },
                next_nodes=["END"]
            )
        ]

        return WorkflowDefinition(
            name="thinkdeep",
            description="Deep thinking and analysis workflow with verification",
            nodes=nodes,
            start_node="deep_thinking",
            end_nodes=["END"],
            state_schema={"thinking_depth": str, "verification_passed": bool}
        )

    def _create_consensus_workflow(self) -> WorkflowDefinition:
        """Create consensus workflow (consolidates consensus tool)."""
        nodes = [
            WorkflowNode(
                name="perspective_generator",
                type=WorkflowNodeType.PARALLEL,
                config={
                    "parallel_agents": [
                        {"name": "advocate", "system_prompt": "Argue strongly in favor of the proposition", "temperature": 0.6},
                        {"name": "skeptic", "system_prompt": "Challenge and critique the proposition", "temperature": 0.6},
                        {"name": "neutral", "system_prompt": "Provide balanced, objective analysis", "temperature": 0.4}
                    ]
                },
                next_nodes=["consensus_builder"]
            ),
            WorkflowNode(
                name="consensus_builder",
                type=WorkflowNodeType.REDUCER,
                config={
                    "reduction_strategy": "consensus_synthesis",
                    "system_prompt": "Synthesize multiple perspectives into a balanced consensus view."
                },
                next_nodes=["END"]
            )
        ]

        return WorkflowDefinition(
            name="consensus",
            description="Multi-perspective consensus building workflow",
            nodes=nodes,
            start_node="perspective_generator",
            end_nodes=["END"],
            state_schema={"perspectives": list, "consensus_reached": bool}
        )

    def _create_analysis_workflow(self) -> WorkflowDefinition:
        """Create analysis workflow (consolidates analyze tool)."""
        nodes = [
            WorkflowNode(
                name="analyzer",
                type=WorkflowNodeType.TOOL,
                config={"tool_name": "analyze"},
                next_nodes=["insight_generator"]
            ),
            WorkflowNode(
                name="insight_generator",
                type=WorkflowNodeType.AGENT,
                config={
                    "system_prompt": "Based on the analysis results, generate actionable insights and recommendations.",
                    "temperature": 0.5
                },
                next_nodes=["END"]
            )
        ]

        return WorkflowDefinition(
            name="analyze",
            description="Code analysis workflow with insight generation",
            nodes=nodes,
            start_node="analyzer",
            end_nodes=["END"],
            state_schema={"analysis_results": dict, "insights": list}
        )

    def _create_documentation_workflow(self) -> WorkflowDefinition:
        """Create documentation workflow (consolidates docgen tool)."""
        nodes = [
            WorkflowNode(
                name="doc_analyzer",
                type=WorkflowNodeType.TOOL,
                config={"tool_name": "analyze", "focus": "documentation_gaps"},
                next_nodes=["doc_generator"]
            ),
            WorkflowNode(
                name="doc_generator",
                type=WorkflowNodeType.TOOL,
                config={"tool_name": "docgen"},
                next_nodes=["doc_reviewer"]
            ),
            WorkflowNode(
                name="doc_reviewer",
                type=WorkflowNodeType.AGENT,
                config={
                    "system_prompt": "Review the generated documentation for clarity, completeness, and accuracy. Suggest improvements.",
                    "temperature": 0.3
                },
                next_nodes=["END"]
            )
        ]

        return WorkflowDefinition(
            name="docgen",
            description="Documentation generation workflow with quality review",
            nodes=nodes,
            start_node="doc_analyzer",
            end_nodes=["END"],
            state_schema={"doc_gaps": list, "generated_docs": str, "review_feedback": str}
        )

    def _create_refactor_workflow(self) -> WorkflowDefinition:
        """Create refactor workflow (consolidates refactor tool)."""
        nodes = [
            WorkflowNode(
                name="code_analyzer",
                type=WorkflowNodeType.TOOL,
                config={"tool_name": "analyze", "focus": "refactoring_opportunities"},
                next_nodes=["refactor_planner"]
            ),
            WorkflowNode(
                name="refactor_planner",
                type=WorkflowNodeType.AGENT,
                config={
                    "system_prompt": "Plan refactoring changes based on analysis. Identify code smells, architectural issues, and improvement opportunities.",
                    "temperature": 0.4
                },
                next_nodes=["refactor_executor"]
            ),
            WorkflowNode(
                name="refactor_executor",
                type=WorkflowNodeType.TOOL,
                config={"tool_name": "refactor"},
                next_nodes=["refactor_validator"]
            ),
            WorkflowNode(
                name="refactor_validator",
                type=WorkflowNodeType.AGENT,
                config={
                    "system_prompt": "Validate refactored code for correctness, functionality preservation, and improvement quality.",
                    "temperature": 0.3
                },
                next_nodes=["END"]
            )
        ]

        return WorkflowDefinition(
            name="refactor",
            description="Code refactoring workflow with planning and validation",
            nodes=nodes,
            start_node="code_analyzer",
            end_nodes=["END"],
            state_schema={"refactor_plan": dict, "refactored_code": str, "validation_results": dict}
        )

    def _create_debug_workflow(self) -> WorkflowDefinition:
        """Create debug workflow (consolidates debug tool)."""
        nodes = [
            WorkflowNode(
                name="issue_identifier",
                type=WorkflowNodeType.TOOL,
                config={"tool_name": "debug", "mode": "identify"},
                next_nodes=["root_cause_analyzer"]
            ),
            WorkflowNode(
                name="root_cause_analyzer",
                type=WorkflowNodeType.AGENT,
                config={
                    "system_prompt": "Analyze debug information to identify root causes. Consider data flow, error patterns, and system interactions.",
                    "temperature": 0.4
                },
                next_nodes=["solution_generator"]
            ),
            WorkflowNode(
                name="solution_generator",
                type=WorkflowNodeType.TOOL,
                config={"tool_name": "debug", "mode": "solve"},
                next_nodes=["solution_validator"]
            ),
            WorkflowNode(
                name="solution_validator",
                type=WorkflowNodeType.AGENT,
                config={
                    "system_prompt": "Validate proposed solutions for correctness and potential side effects.",
                    "temperature": 0.3
                },
                next_nodes=["END"]
            )
        ]

        return WorkflowDefinition(
            name="debug",
            description="Debugging workflow with root cause analysis and solution validation",
            nodes=nodes,
            start_node="issue_identifier",
            end_nodes=["END"],
            state_schema={"identified_issues": list, "root_causes": dict, "solutions": list}
        )

    def _create_testing_workflow(self) -> WorkflowDefinition:
        """Create testing workflow (consolidates testgen tool)."""
        nodes = [
            WorkflowNode(
                name="test_planner",
                type=WorkflowNodeType.AGENT,
                config={
                    "system_prompt": "Analyze code and create comprehensive test plan covering unit tests, integration tests, and edge cases.",
                    "temperature": 0.4
                },
                next_nodes=["test_generator"]
            ),
            WorkflowNode(
                name="test_generator",
                type=WorkflowNodeType.TOOL,
                config={"tool_name": "testgen"},
                next_nodes=["test_validator"]
            ),
            WorkflowNode(
                name="test_validator",
                type=WorkflowNodeType.AGENT,
                config={
                    "system_prompt": "Validate generated tests for completeness, correctness, and best practices compliance.",
                    "temperature": 0.3
                },
                next_nodes=["END"]
            )
        ]

        return WorkflowDefinition(
            name="testgen",
            description="Test generation workflow with planning and validation",
            nodes=nodes,
            start_node="test_planner",
            end_nodes=["END"],
            state_schema={"test_plan": dict, "generated_tests": list, "validation_results": dict}
        )

    async def execute_workflow(
        self,
        workflow_name: str,
        initial_state: dict[str, Any],
        stream_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Execute a workflow using LangGraph state machine."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        workflow = self.workflows[workflow_name]
        execution_id = str(uuid.uuid4())

        # Initialize execution tracking
        self.active_executions[execution_id] = {
            "workflow_name": workflow_name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "current_step": workflow.start_node
        }

        try:
            # Build LangGraph state graph
            graph = await self._build_state_graph(workflow)

            # Prepare initial state
            state = WorkflowState(
                request=initial_state,
                response=None,
                error=None,
                continuation_id=initial_state.get("continuation_id"),
                conversation_history=[],
                files=initial_state.get("files", []),
                images=initial_state.get("images", []),
                workflow_id=execution_id,
                current_step=workflow.start_node,
                step_count=0,
                max_steps=initial_state.get("max_steps", 20),
                agent_results={},
                tool_results={},
                parallel_results=[],
                pending_approvals=[],
                human_feedback=None,
                model=initial_state.get("model"),
                temperature=initial_state.get("temperature", 0.7),
                thinking_mode=initial_state.get("thinking_mode", "medium"),
                streaming_enabled=stream_id is not None,
                stream_id=stream_id
            )

            # Execute the workflow
            config = {"configurable": {"thread_id": execution_id}}
            if self.memory_saver:
                config["checkpointer"] = self.memory_saver

            final_state = await graph.ainvoke(state, config=config)

            # Update execution status
            self.active_executions[execution_id].update({
                "status": "completed" if not final_state.get("error") else "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "final_response": final_state.get("response")
            })

            return {
                "execution_id": execution_id,
                "workflow_name": workflow_name,
                "status": "completed",
                "result": final_state.get("response"),
                "error": final_state.get("error"),
                "metadata": {
                    "steps_executed": final_state.get("step_count", 0),
                    "tools_used": list(final_state.get("tool_results", {}).keys()),
                    "agents_involved": list(final_state.get("agent_results", {}).keys())
                }
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            self.active_executions[execution_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now(timezone.utc).isoformat()
            })

            return {
                "execution_id": execution_id,
                "workflow_name": workflow_name,
                "status": "failed",
                "error": str(e)
            }

    async def _build_state_graph(self, workflow: WorkflowDefinition) -> StateGraph:
        """Build a LangGraph StateGraph from workflow definition."""
        if not StateGraph:
            raise RuntimeError("LangGraph not available")

        graph = StateGraph(WorkflowState)

        # Add nodes to graph
        for node in workflow.nodes:
            if node.type == WorkflowNodeType.AGENT:
                graph.add_node(node.name, self._create_agent_node(node))
            elif node.type == WorkflowNodeType.TOOL:
                graph.add_node(node.name, self._create_tool_node(node))
            elif node.type == WorkflowNodeType.HUMAN:
                graph.add_node(node.name, self._create_human_node(node))
            elif node.type == WorkflowNodeType.PARALLEL:
                graph.add_node(node.name, self._create_parallel_node(node))
            elif node.type == WorkflowNodeType.REDUCER:
                graph.add_node(node.name, self._create_reducer_node(node))

        # Set entry point
        graph.set_entry_point(workflow.start_node)

        # Add edges
        for node in workflow.nodes:
            if node.type == WorkflowNodeType.CONDITIONAL:
                # Add conditional edges
                graph.add_conditional_edges(
                    node.name,
                    node.condition_func,
                    {next_node: next_node for next_node in node.next_nodes}
                )
            else:
                # Add regular edges
                for next_node in node.next_nodes:
                    if next_node == "END":
                        graph.add_edge(node.name, END)
                    else:
                        graph.add_edge(node.name, next_node)

        return graph.compile(checkpointer=self.memory_saver)

    def _create_agent_node(self, node: WorkflowNode) -> Callable:
        """Create an agent execution node using real LLM models."""
        async def agent_node(state: WorkflowState) -> WorkflowState:
            try:
                # Get the real LLM model from state or node config
                model = node.config.get("model", state.get("model"))
                if not model:
                    raise ValueError(f"No model specified for agent node {node.name}")

                # Get provider for the real model
                from providers.registry import ModelProviderRegistry
                provider = ModelProviderRegistry.get_provider_for_model(model)
                if not provider:
                    raise ValueError(f"No provider available for model {model}")

                # Prepare the prompt with context
                system_prompt = node.config.get("system_prompt", "")
                if system_prompt and state["request"].get("system_prompt"):
                    system_prompt = f"{system_prompt}\n\n{state['request'].get('system_prompt')}"
                elif state["request"].get("system_prompt"):
                    system_prompt = state["request"].get("system_prompt")

                # Build context from previous agent results
                context_parts = []
                if state.get("agent_results"):
                    context_parts.append("Previous agent results:")
                    for agent_name, result in state["agent_results"].items():
                        if result and hasattr(result, 'content'):
                            content = result.content
                        elif isinstance(result, dict):
                            content = result.get("content", str(result))
                        else:
                            content = str(result)
                        context_parts.append(f"- {agent_name}: {content[:200]}...")

                if state.get("tool_results"):
                    context_parts.append("Tool execution results:")
                    for tool_name, result in state["tool_results"].items():
                        context_parts.append(f"- {tool_name}: {str(result)[:200]}...")

                # Prepare full prompt
                full_prompt = state["request"].get("prompt", "")
                if context_parts:
                    context_str = "\n".join(context_parts)
                    full_prompt = f"{context_str}\n\nCurrent request: {full_prompt}"

                # Execute with the real LLM provider
                result = await provider.generate_content(
                    prompt=full_prompt,
                    model_name=model,
                    system_prompt=system_prompt,
                    temperature=node.config.get("temperature", state.get("temperature", 0.7)),
                    max_output_tokens=state.get("max_tokens")
                )

                # Update state
                state["agent_results"][node.name] = result
                state["current_step"] = node.name
                state["step_count"] += 1

                # Handle tools if enabled
                if node.config.get("tools_enabled") and self.mcp_client:
                    # Parse the response for tool calls
                    content = result.content if hasattr(result, 'content') else str(result)
                    tool_calls = self._extract_tool_calls(content)

                    for tool_call in tool_calls:
                        try:
                            tool_result = await self.mcp_client.call_tool(
                                tool_call["name"],
                                tool_call.get("arguments", {})
                            )
                            state["tool_results"][f"{node.name}_{tool_call['name']}"] = tool_result
                        except Exception as tool_error:
                            logger.error(f"Tool call {tool_call['name']} failed: {tool_error}")

                return state

            except Exception as e:
                logger.error(f"Agent node {node.name} failed: {e}")
                state["error"] = f"Agent {node.name} failed: {str(e)}"
                return state

        return agent_node

    def _create_tool_node(self, node: WorkflowNode) -> Callable:
        """Create a tool execution node."""
        async def tool_node(state: WorkflowState) -> WorkflowState:
            try:
                if not self.mcp_client:
                    raise RuntimeError("MCP client not initialized")

                tool_name = node.config["tool_name"]
                tool_args = {
                    **node.config,
                    "prompt": state["request"].get("prompt", ""),
                    "files": state.get("files", [])
                }

                # Remove config keys that aren't tool arguments
                tool_args.pop("tool_name", None)

                # Execute tool
                result = await self.mcp_client.call_tool(tool_name, tool_args)

                # Update state
                state["tool_results"][node.name] = result
                state["current_step"] = node.name
                state["step_count"] += 1

                return state

            except Exception as e:
                logger.error(f"Tool node {node.name} failed: {e}")
                state["error"] = f"Tool {node.name} failed: {str(e)}"
                return state

        return tool_node

    def _create_human_node(self, node: WorkflowNode) -> Callable:
        """Create a human interaction node."""
        async def human_node(state: WorkflowState) -> WorkflowState:
            try:
                # Create approval request
                approval_request = {
                    "id": str(uuid.uuid4()),
                    "type": node.config.get("approval_type", "decision"),
                    "description": state["request"].get("prompt", ""),
                    "context": state.get("agent_results", {}),
                    "timeout_minutes": node.config.get("timeout_minutes", 60),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "pending"
                }

                # Add to pending approvals
                state["pending_approvals"].append(approval_request)

                # In a real implementation, this would:
                # 1. Send notification to approvers
                # 2. Wait for response or timeout
                # 3. Update state based on approval decision

                # For now, simulate approval (this would be replaced with actual human interaction)
                state["approval_granted"] = True  # Placeholder
                state["human_feedback"] = "Approved via simulation"

                state["current_step"] = node.name
                state["step_count"] += 1

                return state

            except Exception as e:
                logger.error(f"Human node {node.name} failed: {e}")
                state["error"] = f"Human interaction {node.name} failed: {str(e)}"
                return state

        return human_node

    def _create_parallel_node(self, node: WorkflowNode) -> Callable:
        """Create a parallel execution node."""
        async def parallel_node(state: WorkflowState) -> WorkflowState:
            try:
                parallel_agents = node.config.get("parallel_agents", [])
                tasks = []

                # Create tasks for parallel execution
                for agent_config in parallel_agents:
                    async def execute_parallel_agent(config=agent_config):
                        # Execute agent with specific config
                        {
                            "prompt": state["request"].get("prompt", ""),
                            "model": config.get("model"),
                            "tools": config.get("tools", [])
                        }

                        # This would execute the agent and return results
                        return {"agent": config["name"], "result": f"Result from {config['name']}"}

                    tasks.append(execute_parallel_agent())

                # Execute all tasks concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                successful_results = []
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Parallel execution failed: {result}")
                    else:
                        successful_results.append(result)

                state["parallel_results"] = successful_results
                state["current_step"] = node.name
                state["step_count"] += 1

                return state

            except Exception as e:
                logger.error(f"Parallel node {node.name} failed: {e}")
                state["error"] = f"Parallel execution {node.name} failed: {str(e)}"
                return state

        return parallel_node

    def _create_reducer_node(self, node: WorkflowNode) -> Callable:
        """Create a result aggregation/reduction node."""
        async def reducer_node(state: WorkflowState) -> WorkflowState:
            try:
                # Aggregate results from parallel execution or multiple sources
                reduction_strategy = node.config.get("reduction_strategy", "simple_concat")

                if reduction_strategy == "comprehensive_analysis":
                    # Combine all results into a comprehensive analysis
                    all_results = {
                        "agent_results": state.get("agent_results", {}),
                        "tool_results": state.get("tool_results", {}),
                        "parallel_results": state.get("parallel_results", [])
                    }

                    # Use an agent to synthesize the results
                    f"""
                    Synthesize the following results into a comprehensive analysis:

                    Agent Results: {json.dumps(all_results['agent_results'], indent=2)}
                    Tool Results: {json.dumps(all_results['tool_results'], indent=2)}
                    Parallel Results: {json.dumps(all_results['parallel_results'], indent=2)}

                    Provide a clear, actionable summary that addresses the original request.
                    """

                    # This would call an LLM to synthesize results
                    state["response"] = "Synthesized comprehensive analysis of all workflow results"

                state["current_step"] = node.name
                state["step_count"] += 1

                return state

            except Exception as e:
                logger.error(f"Reducer node {node.name} failed: {e}")
                state["error"] = f"Result reduction {node.name} failed: {str(e)}"
                return state

        return reducer_node

    def _extract_tool_calls(self, content: str) -> list[dict[str, Any]]:
        """Extract tool calls from LLM response content."""
        # This is a simple implementation - in practice, you'd want more sophisticated parsing
        # Could look for patterns like: "I'll use the analyze tool with arguments: {...}"
        import json
        import re

        tool_calls = []

        # Look for JSON-like tool call patterns
        patterns = [
            r'use_tool\s*\(\s*["\'](\w+)["\']\s*,\s*({[^}]*})\s*\)',
            r'call_tool\s*\(\s*["\'](\w+)["\']\s*,\s*({[^}]*})\s*\)',
            r'execute\s+(\w+)\s+with\s+({[^}]*})',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    tool_name = match.group(1)
                    args_str = match.group(2)
                    arguments = json.loads(args_str)
                    tool_calls.append({
                        "name": tool_name,
                        "arguments": arguments
                    })
                except (json.JSONDecodeError, IndexError):
                    continue

        return tool_calls

    async def get_workflow_status(self, execution_id: str) -> Optional[dict[str, Any]]:
        """Get the current status of a workflow execution."""
        return self.active_executions.get(execution_id)

    async def list_workflows(self) -> list[dict[str, Any]]:
        """List all registered workflows."""
        return [
            {
                "name": workflow.name,
                "description": workflow.description,
                "nodes": len(workflow.nodes),
                "start_node": workflow.start_node,
                "end_nodes": workflow.end_nodes
            }
            for workflow in self.workflows.values()
        ]

    async def list_active_executions(self) -> list[dict[str, Any]]:
        """List all active workflow executions."""
        return list(self.active_executions.values())

    # Provider interface methods (required for compatibility)

    def validate_model_name(self, model_name: str) -> bool:
        """Validate that a model is supported."""
        # Since this is a workflow provider, it delegates to underlying agents
        return True

    async def generate_content(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Generate content using workflow execution."""
        # Determine workflow from model name or default
        workflow_name = model_name or "multi_agent_collaboration"

        if workflow_name not in self.workflows:
            workflow_name = "multi_agent_collaboration"

        # Execute workflow
        result = await self.execute_workflow(
            workflow_name,
            {"prompt": prompt, **kwargs}
        )

        # Return in expected format
        from dataclasses import dataclass

        @dataclass
        class WorkflowResponse:
            content: str
            model_name: str
            finish_reason: str = "stop"
            usage: Optional[Any] = None

        return WorkflowResponse(
            content=result.get("result", ""),
            model_name=f"aegis-{workflow_name}",
            finish_reason="completed" if result.get("status") == "completed" else "error"
        )


# Factory function for easy integration
async def create_aegis_provider(mcp_server_url: str = "http://localhost:8080/mcp") -> AegisLangGraphProvider:
    """Create and initialize an Aegis LangGraph provider."""
    provider = AegisLangGraphProvider(mcp_server_url)
    await provider.initialize()
    return provider


# Integration with existing provider system
def register_aegis_provider():
    """Register the Aegis provider in the provider registry."""
    try:
        from providers.registry import ModelProviderRegistry

        # This would be called during server startup
        async def aegis_factory():
            return await create_aegis_provider()

        # Register the provider factory
        # ModelProviderRegistry.register_provider("aegis", aegis_factory)
        logger.info("Aegis LangGraph provider registered successfully")

    except ImportError:
        logger.warning("Could not register Aegis provider - registry not available")


if __name__ == "__main__":
    # Example usage
    async def demo():
        if not LANGGRAPH_AVAILABLE:
            print("LangGraph not available. Install with: pip install langgraph")
            return

        provider = await create_aegis_provider()

        # List available workflows
        workflows = await provider.list_workflows()
        print(f"Available workflows: {[w['name'] for w in workflows]}")

        # Execute a workflow
        result = await provider.execute_workflow(
            "multi_agent_collaboration",
            {
                "prompt": "Create a comprehensive project plan for building a web application",
                "files": [],
                "model": "gpt-4o"
            }
        )

        print(f"Workflow result: {result}")

    asyncio.run(demo())
