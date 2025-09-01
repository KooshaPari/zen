"""
Universal Executor Tool

A consolidated tool that provides all Zen MCP Server capabilities through a single interface
with comprehensive parameters for feature selection. This tool unifies:

- LLM direct execution (all providers)
- Agent orchestration (async/sync/batch modes)
- Workflow execution (multi-step, approval, saga patterns)
- Tool execution (all existing MCP tools)
- Streaming and batch processing
- Conversation memory and continuation
- Multi-modal support (text, images, files)

Features controlled via parameters:
- execution_mode: sync, async, batch, streaming
- agent_type: llm, agent, aegis, workflow, tool
- provider_preference: throughput, cost, quality
- memory_mode: stateless, stateful, continuation
- output_format: text, json, structured, stream
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from providers.registry import ModelProviderRegistry as ProviderRegistry
from tools.shared.base_tool import BaseTool
from utils.agent_manager import get_task_manager
from utils.conversation_memory import add_turn, build_conversation_history, get_thread
from utils.streaming_protocol import get_streaming_manager

logger = logging.getLogger(__name__)


class UniversalExecutionRequest(BaseModel):
    """Universal execution request supporting all features."""

    # Core execution parameters
    prompt: str = Field(description="Primary prompt/instruction for execution")
    agent_type: str = Field(
        default="llm",
        description="Execution type: 'llm' (direct model), 'agent' (orchestrated), 'aegis' (LangGraph workflow), 'workflow' (multi-step), 'tool' (MCP tool call)"
    )
    execution_mode: str = Field(
        default="sync",
        description="Execution mode: 'sync' (immediate), 'async' (background), 'batch' (multiple), 'streaming' (real-time)"
    )

    # Model and provider configuration
    model: Optional[str] = Field(
        default=None,
        description="Specific model to use (auto-selected if not provided)"
    )
    provider_preference: str = Field(
        default="throughput",
        description="Provider selection criteria: 'throughput' (fastest), 'cost' (cheapest), 'quality' (best results)"
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Response creativity (0.0-1.0, auto-selected based on task if not provided)"
    )

    # Memory and conversation
    memory_mode: str = Field(
        default="stateless",
        description="Memory handling: 'stateless' (no context), 'stateful' (maintain context), 'continuation' (specific conversation)"
    )
    continuation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID for multi-turn interactions (auto-generated if memory_mode='stateful')"
    )

    # Context and input
    files: Optional[list[str]] = Field(
        default_factory=list,
        description="File paths for context (full absolute paths)"
    )
    images: Optional[list[str]] = Field(
        default_factory=list,
        description="Image paths or base64 data for visual context"
    )
    context: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional context data for execution"
    )

    # Workflow and agent parameters
    workflow_type: Optional[str] = Field(
        default=None,
        description="Workflow type for agent_type='aegis' or 'workflow': 'multi_agent_collaboration', 'human_approval', 'research_analysis', 'code_review_improvement', 'thinkdeep', 'consensus', 'analyze', 'docgen', 'refactor', 'debug', 'testgen'"
    )
    workflow_spec: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Workflow specification for complex workflows"
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="Specific tool name for agent_type='tool'"
    )
    tool_arguments: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Arguments for tool execution"
    )

    # Batch and async parameters
    batch_items: Optional[list[dict[str, Any]]] = Field(
        default_factory=list,
        description="Multiple items for batch execution"
    )
    batch_mode: str = Field(
        default="parallel",
        description="Batch processing: 'parallel' (concurrent), 'sequential' (ordered), 'adaptive' (dynamic)"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for async completion notifications"
    )

    # Output configuration
    output_format: str = Field(
        default="text",
        description="Output format: 'text' (plain), 'json' (structured), 'stream' (chunked), 'xml' (tagged)"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum response length (auto-determined if not provided)"
    )
    stream_mode: bool = Field(
        default=False,
        description="Enable streaming output (overrides output_format if True)"
    )

    # Advanced features
    thinking_mode: str = Field(
        default="medium",
        description="Reasoning depth: 'minimal', 'low', 'medium', 'high', 'max'"
    )
    use_websearch: bool = Field(
        default=False,
        description="Enable web search for current information"
    )
    enable_tools: bool = Field(
        default=False,
        description="Allow LLM to call MCP tools during execution (agent loop mode)"
    )
    quality_checks: bool = Field(
        default=False,
        description="Run quality validation on outputs"
    )


class DeployTool(BaseTool):
    """
    Deploy Tool - Unified Execution Interface

    A consolidated deployment interface for all Zen MCP Server execution capabilities.
    Replaces individual tools with a comprehensive parameter-driven approach.

    Supported execution modes:
    - LLM: Direct model execution with conversation memory
    - Agent: Orchestrated agent workflows with tool access
    - Aegis: LangGraph-powered multi-agent workflows (consolidates thinkdeep, consensus, analyze, docgen, refactor, debug, testgen)
    - Workflow: Complex multi-step processes (approvals, sagas, projects)
    - Tool: Direct MCP tool execution
    - Batch: Multiple parallel executions
    - Streaming: Real-time progressive responses
    """

    def __init__(self):
        super().__init__()

        # Initialize core managers
        self.agent_manager = get_task_manager()
        self.provider_registry = ProviderRegistry()

        # Track active executions
        self.active_executions: dict[str, dict[str, Any]] = {}

        # Performance metrics
        self.execution_stats = {
            "total_requests": 0,
            "avg_response_time": 0.0,
            "success_rate": 0.0,
            "by_agent_type": {},
            "by_execution_mode": {}
        }

    def get_name(self) -> str:
        """Return the unique name identifier for this tool."""
        return "deploy"

    def get_description(self) -> str:
        """Return a detailed description of what this tool does."""
        return """Deploy - Universal execution interface for ALL development tasks. Use this tool for:

🎯 CHAT & COLLABORATION: Interactive discussions, brainstorming, Q&A sessions (agent_type='llm')
🧠 DEEP ANALYSIS: Multi-step reasoning, investigation, research (agent_type='aegis', workflow_type='thinkdeep')
📋 TASK PLANNING: Break down complex projects into actionable steps (agent_type='aegis', workflow_type='planner')
🤝 CONSENSUS BUILDING: Multi-model analysis for important decisions (agent_type='aegis', workflow_type='consensus')
🔍 CODE REVIEW: Systematic code analysis and improvement suggestions (agent_type='aegis', workflow_type='codereview')
📊 FILE ANALYSIS: Analyze any file type - code, docs, data (agent_type='aegis', workflow_type='analyze')
🛠️ REFACTORING: Code improvement and restructuring guidance (agent_type='aegis', workflow_type='refactor')
🧪 TEST GENERATION: Create comprehensive test suites (agent_type='aegis', workflow_type='testgen')
🔒 SECURITY AUDIT: Security vulnerability analysis (agent_type='aegis', workflow_type='secaudit')
📚 DOCUMENTATION: Generate docs, comments, guides (agent_type='aegis', workflow_type='docgen')
🔄 PRE-COMMIT: Validation before code commits (agent_type='aegis', workflow_type='precommit')
🎯 ANY MCP TOOL: Execute any individual tool (agent_type='tool', tool_name='specific_tool')

EXECUTION MODES: sync (immediate), async (background), batch (multiple), streaming (real-time)
MEMORY: stateless, stateful (auto-continue), continuation (specific thread)
PROVIDERS: throughput (fastest), cost (cheapest), quality (best results)"""

    def get_input_schema(self) -> dict[str, Any]:
        """Return the JSON Schema that defines this tool's parameters."""
        return self.get_schema()

    def get_system_prompt(self) -> str:
        """Return the system prompt that configures the AI model's behavior."""
        return "You are a universal executor capable of handling any type of task through multiple execution modes and agent types."

    def get_request_model(self):
        """Return the Pydantic model class used for validating requests."""
        return UniversalExecutionRequest

    async def prepare_prompt(self, request) -> str:
        """Prepare the complete prompt for the AI model."""
        if hasattr(request, 'prompt'):
            return request.prompt
        return str(request)

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute universal request with comprehensive feature support."""
        execution_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        try:
            # Parse and validate request
            request = UniversalExecutionRequest(**arguments)

            # Update stats
            self.execution_stats["total_requests"] += 1
            self.execution_stats["by_agent_type"][request.agent_type] = \
                self.execution_stats["by_agent_type"].get(request.agent_type, 0) + 1
            self.execution_stats["by_execution_mode"][request.execution_mode] = \
                self.execution_stats["by_execution_mode"].get(request.execution_mode, 0) + 1

            # Track execution
            self.active_executions[execution_id] = {
                "request": request.model_dump(),
                "started_at": start_time.isoformat(),
                "status": "processing"
            }

            # Route to appropriate execution method
            if request.execution_mode == "async":
                return await self._execute_async(execution_id, request)
            elif request.execution_mode == "batch":
                return await self._execute_batch(execution_id, request)
            elif request.execution_mode == "streaming" or request.stream_mode:
                return await self._execute_streaming(execution_id, request)
            else:  # sync
                return await self._execute_sync(execution_id, request)

        except Exception as e:
            logger.error(f"Universal execution failed: {e}")
            return {
                "success": False,
                "execution_id": execution_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
        finally:
            # Clean up
            if execution_id in self.active_executions:
                duration = (datetime.utcnow() - start_time).total_seconds()
                self.active_executions[execution_id]["duration"] = duration
                self.active_executions[execution_id]["status"] = "completed"

    async def _execute_sync(self, execution_id: str, request: UniversalExecutionRequest) -> dict[str, Any]:
        """Execute synchronous request."""
        try:
            # Route by agent type
            if request.agent_type == "llm":
                result = await self._execute_llm(request)
            elif request.agent_type == "agent":
                result = await self._execute_agent(request)
            elif request.agent_type == "aegis":
                result = await self._execute_aegis(request)
            elif request.agent_type == "workflow":
                result = await self._execute_workflow(request)
            elif request.agent_type == "tool":
                result = await self._execute_tool(request)
            else:
                return {
                    "success": False,
                    "error": f"Unknown agent_type: {request.agent_type}",
                    "supported_types": ["llm", "agent", "aegis", "workflow", "tool"]
                }

            # Handle memory persistence
            if request.memory_mode in ["stateful", "continuation"]:
                await self._save_conversation_turn(request, result)

            # Apply output formatting
            formatted_result = await self._format_output(result, request)

            return {
                "success": True,
                "execution_id": execution_id,
                "agent_type": request.agent_type,
                "execution_mode": "sync",
                "result": formatted_result,
                "metadata": {
                    "model_used": result.get("model_info", {}).get("name"),
                    "provider_used": result.get("provider_info", {}).get("name"),
                    "tokens_used": result.get("usage", {}).get("total_tokens"),
                    "continuation_id": request.continuation_id
                }
            }

        except Exception as e:
            logger.error(f"Sync execution failed for {execution_id}: {e}")
            return {
                "success": False,
                "execution_id": execution_id,
                "error": str(e)
            }

    async def _execute_async(self, execution_id: str, request: UniversalExecutionRequest) -> dict[str, Any]:
        """Execute asynchronous request."""
        # Start background task
        asyncio.create_task(self._process_async_request(execution_id, request))

        return {
            "success": True,
            "execution_id": execution_id,
            "status": "processing",
            "execution_mode": "async",
            "callback_url": request.callback_url,
            "check_status_url": f"/tasks/{execution_id}/status"
        }

    async def _execute_batch(self, execution_id: str, request: UniversalExecutionRequest) -> dict[str, Any]:
        """Execute batch request."""
        if not request.batch_items:
            return {
                "success": False,
                "error": "batch_items required for batch execution"
            }

        try:
            # Create individual requests
            batch_requests = []
            for item in request.batch_items:
                # Inherit base request properties
                item_request = request.model_copy()
                # Override with item-specific properties
                for key, value in item.items():
                    setattr(item_request, key, value)
                batch_requests.append(item_request)

            # Execute based on batch mode
            if request.batch_mode == "parallel":
                results = await asyncio.gather(*[
                    self._execute_single_item(req) for req in batch_requests
                ], return_exceptions=True)
            elif request.batch_mode == "sequential":
                results = []
                for req in batch_requests:
                    result = await self._execute_single_item(req)
                    results.append(result)
            else:  # adaptive
                results = await self._execute_adaptive_batch(batch_requests)

            # Process results
            success_count = sum(1 for r in results if isinstance(r, dict) and r.get("success"))

            return {
                "success": True,
                "execution_id": execution_id,
                "execution_mode": "batch",
                "batch_mode": request.batch_mode,
                "total_items": len(request.batch_items),
                "successful_items": success_count,
                "failed_items": len(results) - success_count,
                "results": results
            }

        except Exception as e:
            logger.error(f"Batch execution failed for {execution_id}: {e}")
            return {
                "success": False,
                "execution_id": execution_id,
                "error": str(e)
            }

    async def _execute_streaming(self, execution_id: str, request: UniversalExecutionRequest) -> dict[str, Any]:
        """Execute streaming request."""
        try:
            # Initialize streaming session
            streaming_manager = get_streaming_manager()
            stream_id = await streaming_manager.create_stream(execution_id)

            # Start streaming in background
            asyncio.create_task(self._process_streaming_request(stream_id, request))

            return {
                "success": True,
                "execution_id": execution_id,
                "stream_id": stream_id,
                "execution_mode": "streaming",
                "stream_url": f"/streams/{stream_id}",
                "websocket_url": f"/ws/streams/{stream_id}"
            }

        except Exception as e:
            logger.error(f"Streaming execution failed for {execution_id}: {e}")
            return {
                "success": False,
                "execution_id": execution_id,
                "error": str(e)
            }

    async def _execute_llm(self, request: UniversalExecutionRequest) -> dict[str, Any]:
        """Execute direct LLM request."""
        # Select optimal provider
        provider = await self._select_provider(request)

        # Prepare messages with context
        messages = await self._prepare_messages(request)

        # Execute with provider
        if request.enable_tools:
            # Agent loop mode - LLM can call tools
            result = await self._execute_llm_with_tools(provider, messages, request)
        else:
            # Direct execution
            response = await provider.execute(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                thinking_mode=request.thinking_mode
            )
            result = {
                "content": response.content,
                "model_info": response.model_info,
                "provider_info": response.provider_info,
                "usage": response.usage
            }

        return result

    async def _execute_agent(self, request: UniversalExecutionRequest) -> dict[str, Any]:
        """Execute agent orchestration request."""
        # Use existing agent orchestration system
        agent_request = {
            "prompt": request.prompt,
            "files": request.files,
            "images": request.images,
            "model": request.model,
            "temperature": request.temperature,
            "continuation_id": request.continuation_id
        }

        result = await self.agent_manager.execute_agent_request(agent_request)
        return result

    async def _execute_aegis(self, request: UniversalExecutionRequest) -> dict[str, Any]:
        """Execute Aegis LangGraph workflow request."""
        try:
            # Check LangGraph availability
            from integrations.langgraph_wrapper import LANGGRAPH_AVAILABLE
            if not LANGGRAPH_AVAILABLE:
                return {
                    "success": False,
                    "error": "LangGraph not available. Install with: pip install langgraph"
                }

            # Import and create Aegis provider
            from integrations.langgraph_wrapper import create_aegis_provider
            aegis_provider = await create_aegis_provider()

            # Extract workflow name from request
            workflow_name = request.workflow_type or "multi_agent_collaboration"

            # Validate required model parameter
            if not request.model:
                return {
                    "success": False,
                    "error": "Model parameter required for Aegis workflows"
                }

            # Execute workflow
            result = await aegis_provider.execute_workflow(
                workflow_name=workflow_name,
                initial_state={
                    "request": {"prompt": request.prompt},
                    "model": request.model,  # Real LLM model to use
                    "system_prompt": request.context.get("system_prompt"),
                    "temperature": request.temperature or 0.7,
                    "max_tokens": request.context.get("max_tokens"),
                    "files": request.files or [],
                    "images": request.images or [],
                    "continuation_id": request.continuation_id,
                    "thinking_mode": request.context.get("thinking_mode", "medium"),
                    "use_websearch": request.context.get("use_websearch", False)
                }
            )

            return {
                "success": True,
                "result": result.get("result", ""),
                "workflow": workflow_name,
                "model": request.model,
                "execution_metadata": result.get("metadata", {}),
                "workflow_stats": {
                    "execution_id": result.get("execution_id"),
                    "steps_executed": result.get("metadata", {}).get("steps_executed", 0),
                    "tools_used": result.get("metadata", {}).get("tools_used", []),
                    "agents_involved": result.get("metadata", {}).get("agents_involved", [])
                }
            }

        except Exception as e:
            logger.error(f"Aegis execution failed: {e}")
            return {
                "success": False,
                "error": f"Aegis workflow execution failed: {str(e)}"
            }

    async def _execute_workflow(self, request: UniversalExecutionRequest) -> dict[str, Any]:
        """Execute workflow request."""
        if not request.workflow_type:
            return {
                "success": False,
                "error": "workflow_type required for workflow execution"
            }

        # Import workflow orchestrator
        from tools.workflow_orchestrator import WorkflowOrchestratorTool

        workflow_tool = WorkflowOrchestratorTool()
        workflow_args = {
            "operation": "start_workflow",
            "workflow_type": request.workflow_type,
            "workflow_spec": request.workflow_spec,
            "config": request.context
        }

        return await workflow_tool.execute(workflow_args)

    async def _execute_tool(self, request: UniversalExecutionRequest) -> dict[str, Any]:
        """Execute specific MCP tool."""
        if not request.tool_name:
            return {
                "success": False,
                "error": "tool_name required for tool execution"
            }

        # Get tool from registry
        from tools import get_tool_by_name

        tool = get_tool_by_name(request.tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{request.tool_name}' not found"
            }

        # Execute tool
        return await tool.execute(request.tool_arguments)

    async def _select_provider(self, request: UniversalExecutionRequest):
        """Select optimal provider based on preferences."""
        # Check if Aegis workflow is requested via model name
        model = request.model
        if model and (model == "aegis" or model.startswith("aegis-")):
            try:
                from providers.aegis import AegisProvider
                return AegisProvider()
            except ImportError:
                logger.warning("Aegis provider requested but LangGraph not available")
                # Fall through to default provider selection

        # Standard provider selection
        if request.provider_preference == "throughput":
            return self.provider_registry.get_fastest_provider()
        elif request.provider_preference == "cost":
            return self.provider_registry.get_cheapest_provider()
        elif request.provider_preference == "quality":
            return self.provider_registry.get_best_quality_provider()
        else:
            return self.provider_registry.get_default_provider()

    async def _prepare_messages(self, request: UniversalExecutionRequest) -> list[dict[str, Any]]:
        """Prepare message history with context."""
        messages = []

        # Load conversation history if continuing
        if request.continuation_id and request.memory_mode == "continuation":
            context = get_thread(request.continuation_id)
            if context:
                history_text, _ = build_conversation_history(context)
                if history_text:
                    messages.append({"role": "system", "content": f"Previous conversation:\n{history_text}"})


        # Add file context
        if request.files:
            from utils.file_handler import FileHandler
            file_handler = FileHandler()
            for file_path in request.files:
                content = await file_handler.read_file(file_path)
                messages.append({
                    "role": "system",
                    "content": f"File: {file_path}\n\n{content}"
                })

        # Add image context
        if request.images:
            from utils.image_handler import ImageHandler
            image_handler = ImageHandler()
            for image in request.images:
                image_content = await image_handler.process_image(image)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Image context: {image}"},
                        {"type": "image_url", "image_url": {"url": image_content}}
                    ]
                })

        # Add main prompt
        messages.append({
            "role": "user",
            "content": request.prompt
        })

        return messages

    async def _save_conversation_turn(self, request: UniversalExecutionRequest, result: dict[str, Any]):
        """Save conversation turn to memory."""
        if not request.continuation_id and request.memory_mode == "stateful":
            request.continuation_id = str(uuid.uuid4())

        if request.continuation_id:
            add_turn(
                thread_id=request.continuation_id,
                role="assistant",
                content=result.get("content", str(result))
            )

    async def _format_output(self, result: dict[str, Any], request: UniversalExecutionRequest) -> Any:
        """Format output according to requested format."""
        content = result.get("content", result)

        if request.output_format == "json":
            try:
                return json.loads(content)
            except:
                return {"content": content}
        elif request.output_format == "xml":
            return f"<response>{content}</response>"
        elif request.output_format == "structured":
            return {
                "content": content,
                "metadata": result.get("metadata", {}),
                "usage": result.get("usage", {}),
                "model_info": result.get("model_info", {})
            }
        else:  # text
            return content

    async def _process_async_request(self, execution_id: str, request: UniversalExecutionRequest):
        """Process async request in background."""
        try:
            result = await self._execute_sync(execution_id, request)
            # Store result for later retrieval
            self.active_executions[execution_id]["result"] = result
            self.active_executions[execution_id]["status"] = "completed"
        except Exception as e:
            self.active_executions[execution_id]["error"] = str(e)
            self.active_executions[execution_id]["status"] = "failed"

    async def _process_streaming_request(self, stream_id: str, request: UniversalExecutionRequest):
        """Process streaming request with progressive updates."""
        try:
            streaming_manager = get_streaming_manager()
            # For now, just execute sync and stream the final result
            result = await self._execute_sync(stream_id, request)
            await streaming_manager.send_message(stream_id, "completed", result)
        except Exception as e:
            streaming_manager = get_streaming_manager()
            await streaming_manager.send_message(stream_id, "error", {"error": str(e)})

    async def _execute_single_item(self, request: UniversalExecutionRequest) -> dict[str, Any]:
        """Execute a single batch item."""
        execution_id = str(uuid.uuid4())
        return await self._execute_sync(execution_id, request)

    async def _execute_adaptive_batch(self, requests: list[UniversalExecutionRequest]) -> list[dict[str, Any]]:
        """Execute batch with adaptive concurrency."""
        # For now, use simple parallel execution
        return await asyncio.gather(*[
            self._execute_single_item(req) for req in requests
        ], return_exceptions=True)

    async def _execute_llm_with_tools(self, provider, messages: list[dict[str, Any]], request: UniversalExecutionRequest) -> dict[str, Any]:
        """Execute LLM with tool calling capability."""
        # For now, just execute without tools
        response = await provider.execute(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            thinking_mode=request.thinking_mode
        )
        return {
            "content": response.content,
            "model_info": response.model_info,
            "provider_info": response.provider_info,
            "usage": response.usage
        }

    def get_schema(self) -> dict[str, Any]:
        """Get comprehensive tool schema."""
        return {
            "type": "object",
            "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Primary prompt/instruction for execution"
                    },
                    "agent_type": {
                        "type": "string",
                        "enum": ["llm", "agent", "aegis", "workflow", "tool"],
                        "description": "Execution type: 'llm' (direct model), 'agent' (orchestrated), 'aegis' (LangGraph workflow), 'workflow' (multi-step), 'tool' (MCP tool)",
                        "default": "llm"
                    },
                    "execution_mode": {
                        "type": "string",
                        "enum": ["sync", "async", "batch", "streaming"],
                        "description": "Execution mode: 'sync' (immediate), 'async' (background), 'batch' (multiple), 'streaming' (real-time)",
                        "default": "sync"
                    },
                    "model": {
                        "type": "string",
                        "description": "Specific model to use (auto-selected if not provided)"
                    },
                    "provider_preference": {
                        "type": "string",
                        "enum": ["throughput", "cost", "quality"],
                        "description": "Provider selection criteria",
                        "default": "throughput"
                    },
                    "temperature": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Response creativity (0.0-1.0)"
                    },
                    "memory_mode": {
                        "type": "string",
                        "enum": ["stateless", "stateful", "continuation"],
                        "description": "Memory handling mode",
                        "default": "stateless"
                    },
                    "continuation_id": {
                        "type": "string",
                        "description": "Conversation ID for multi-turn interactions"
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File paths for context (full absolute paths)"
                    },
                    "images": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Image paths or base64 data for visual context"
                    },
                    "workflow_type": {
                        "type": "string",
                        "enum": ["multi_agent_collaboration", "human_approval", "research_analysis", "code_review_improvement", "thinkdeep", "consensus", "analyze", "docgen", "refactor", "debug", "testgen"],
                        "description": "Workflow type for aegis or workflow execution (includes consolidated existing tools)"
                    },
                    "workflow_spec": {
                        "type": "object",
                        "description": "Workflow specification for complex workflows"
                    },
                    "tool_name": {
                        "type": "string",
                        "description": "Specific tool name for tool execution"
                    },
                    "tool_arguments": {
                        "type": "object",
                        "description": "Arguments for tool execution"
                    },
                    "batch_items": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Multiple items for batch execution"
                    },
                    "batch_mode": {
                        "type": "string",
                        "enum": ["parallel", "sequential", "adaptive"],
                        "description": "Batch processing mode",
                        "default": "parallel"
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["text", "json", "stream", "xml", "structured"],
                        "description": "Output format",
                        "default": "text"
                    },
                    "stream_mode": {
                        "type": "boolean",
                        "description": "Enable streaming output",
                        "default": False
                    },
                    "thinking_mode": {
                        "type": "string",
                        "enum": ["minimal", "low", "medium", "high", "max"],
                        "description": "Reasoning depth",
                        "default": "medium"
                    },
                    "use_websearch": {
                        "type": "boolean",
                        "description": "Enable web search for current information",
                        "default": False
                    },
                    "enable_tools": {
                        "type": "boolean",
                        "description": "Allow LLM to call MCP tools during execution",
                        "default": False
                    },
                    "quality_checks": {
                        "type": "boolean",
                        "description": "Run quality validation on outputs",
                        "default": False
                    }
            },
            "required": ["prompt"]
        }


# Register tool instance
deploy_tool = DeployTool()

logger.info("Deploy tool initialized - consolidating all server capabilities under unified interface")
