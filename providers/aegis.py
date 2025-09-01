"""
Aegis Provider - LangGraph Workflow Integration

This provider enables the "aegis" option in model selection, providing access to
sophisticated LangGraph-powered multi-agent workflows. When a user specifies
"aegis" as the model, the system routes to complex workflow orchestration instead
of simple LLM execution.

Usage:
- Model name: "aegis" or "aegis-{workflow_name}"
- Automatically routes to appropriate LangGraph workflows
- Supports streaming, conversation memory, and tool integration
- Provides human-in-the-loop capabilities

Integration with Universal Executor:
- agent_type: "llm"
- model: "aegis" or "aegis-multi_agent_collaboration"
- execution_mode: "sync", "async", "streaming"
- Additional parameters passed to workflow configuration
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from integrations.langgraph_wrapper import LANGGRAPH_AVAILABLE, AegisLangGraphProvider

from providers.base import BaseProvider, ProviderType

logger = logging.getLogger(__name__)


@dataclass
class AegisUsage:
    """Usage tracking for Aegis workflows."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    workflow_steps: int = 0
    tools_executed: int = 0
    agents_involved: int = 0
    execution_time_seconds: float = 0.0
    cost_per_1m_input: float = 0.0  # Free for now
    cost_per_1m_output: float = 0.0  # Free for now
    estimated_cost: float = 0.0


@dataclass
class AegisResponse:
    """Response from Aegis workflow execution."""
    content: str
    model_name: str
    finish_reason: str = "completed"
    usage: Optional[AegisUsage] = None
    execution_metadata: Optional[dict[str, Any]] = None


class AegisProvider(BaseProvider):
    """
    Aegis Provider for LangGraph workflow orchestration.

    Provides the "aegis" model option that routes to sophisticated
    multi-agent workflows instead of single LLM execution.
    """

    def __init__(self):
        super().__init__()
        self.langgraph_provider: Optional[AegisLangGraphProvider] = None
        self.mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp")
        self.initialization_lock = asyncio.Lock()
        self.initialized = False

    def get_provider_type(self) -> ProviderType:
        return ProviderType.CUSTOM

    def get_provider_name(self) -> str:
        return "aegis"

    def is_available(self) -> bool:
        """Check if Aegis provider is available."""
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available for Aegis provider")
            return False
        return True

    def validate_model_name(self, model_name: str) -> bool:
        """Validate Aegis model names."""
        if not self.is_available():
            return False

        # Accept "aegis" or "aegis-{workflow_name}"
        if model_name == "aegis":
            return True
        if model_name.startswith("aegis-"):
            # Extract workflow name and validate
            workflow_name = model_name[6:]  # Remove "aegis-" prefix
            return workflow_name in [
                "multi_agent_collaboration",
                "human_approval",
                "research_analysis",
                "code_review_improvement"
            ]
        return False

    async def _ensure_initialized(self):
        """Ensure the LangGraph provider is initialized."""
        if self.initialized:
            return

        async with self.initialization_lock:
            if self.initialized:
                return

            try:
                from integrations.langgraph_wrapper import create_aegis_provider
                self.langgraph_provider = await create_aegis_provider(self.mcp_server_url)
                self.initialized = True
                logger.info("Aegis LangGraph provider initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Aegis provider: {e}")
                raise

    def _extract_workflow_name(self, model_name: str) -> str:
        """Extract workflow name from model name."""
        if model_name == "aegis":
            return "multi_agent_collaboration"  # Default workflow
        elif model_name.startswith("aegis-"):
            return model_name[6:]  # Remove "aegis-" prefix
        else:
            return "multi_agent_collaboration"

    async def generate_content(
        self,
        prompt: str,
        model_name: str = "aegis",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        thinking_mode: str = "medium",
        files: Optional[list[str]] = None,
        images: Optional[list[str]] = None,
        continuation_id: Optional[str] = None,
        stream_mode: bool = False,
        **kwargs
    ) -> AegisResponse:
        """
        Generate content using Aegis LangGraph workflows.

        Args:
            prompt: The user prompt/request
            model_name: Aegis model name (aegis or aegis-{workflow})
            system_prompt: System prompt for workflow context
            temperature: Temperature for LLM agents in workflow
            max_output_tokens: Token limit (passed to workflow)
            thinking_mode: Reasoning depth for agents
            files: File paths for context
            images: Image paths for context
            continuation_id: Conversation continuation ID
            stream_mode: Enable streaming (creates stream_id)
            **kwargs: Additional workflow parameters

        Returns:
            AegisResponse with workflow execution results
        """
        await self._ensure_initialized()

        if not self.langgraph_provider:
            raise RuntimeError("Aegis LangGraph provider not initialized")

        # Extract workflow name from model
        workflow_name = self._extract_workflow_name(model_name)

        # Prepare workflow parameters
        workflow_params = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "thinking_mode": thinking_mode,
            "files": files or [],
            "images": images or [],
            "continuation_id": continuation_id,
            "model": kwargs.get("underlying_model", "gpt-4o"),  # Default model for agents
            **kwargs  # Pass through additional parameters
        }

        # Handle streaming
        stream_id = None
        if stream_mode:
            stream_id = f"aegis-{workflow_name}-{id(self)}"
            # Initialize streaming (this would integrate with StreamingManager)

        try:
            # Execute the workflow
            execution_start = asyncio.get_event_loop().time()

            result = await self.langgraph_provider.execute_workflow(
                workflow_name=workflow_name,
                initial_state=workflow_params,
                stream_id=stream_id
            )

            execution_time = asyncio.get_event_loop().time() - execution_start

            # Extract results
            content = result.get("result", "Workflow completed successfully")
            if result.get("error"):
                content = f"Workflow failed: {result['error']}"

            # Build usage tracking
            metadata = result.get("metadata", {})
            usage = AegisUsage(
                workflow_steps=metadata.get("steps_executed", 0),
                tools_executed=len(metadata.get("tools_used", [])),
                agents_involved=len(metadata.get("agents_involved", [])),
                execution_time_seconds=execution_time,
                # Token counting would require integration with underlying LLM calls
                input_tokens=len(prompt.split()) * 2,  # Rough estimate
                output_tokens=len(content.split()) * 2,  # Rough estimate
            )
            usage.total_tokens = usage.input_tokens + usage.output_tokens

            return AegisResponse(
                content=content,
                model_name=f"aegis-{workflow_name}",
                finish_reason="completed" if result.get("status") == "completed" else "error",
                usage=usage,
                execution_metadata={
                    "workflow_name": workflow_name,
                    "execution_id": result.get("execution_id"),
                    "execution_time": execution_time,
                    "stream_id": stream_id,
                    **metadata
                }
            )

        except Exception as e:
            logger.error(f"Aegis workflow execution failed: {e}")

            # Return error response
            error_usage = AegisUsage(
                input_tokens=len(prompt.split()) * 2,
                output_tokens=0,
                execution_time_seconds=asyncio.get_event_loop().time() - execution_start if 'execution_start' in locals() else 0
            )

            return AegisResponse(
                content=f"Workflow execution failed: {str(e)}",
                model_name=f"aegis-{workflow_name}",
                finish_reason="error",
                usage=error_usage,
                execution_metadata={
                    "workflow_name": workflow_name,
                    "error": str(e),
                    "stream_id": stream_id
                }
            )

    async def list_available_workflows(self) -> list[dict[str, Any]]:
        """List all available Aegis workflows."""
        await self._ensure_initialized()

        if not self.langgraph_provider:
            return []

        return await self.langgraph_provider.list_workflows()

    async def get_workflow_status(self, execution_id: str) -> Optional[dict[str, Any]]:
        """Get status of a running workflow."""
        await self._ensure_initialized()

        if not self.langgraph_provider:
            return None

        return await self.langgraph_provider.get_workflow_status(execution_id)

    async def list_active_executions(self) -> list[dict[str, Any]]:
        """List all active workflow executions."""
        await self._ensure_initialized()

        if not self.langgraph_provider:
            return []

        return await self.langgraph_provider.list_active_executions()

    def get_supported_models(self) -> list[str]:
        """Get list of supported Aegis model names."""
        if not self.is_available():
            return []

        return [
            "aegis",
            "aegis-multi_agent_collaboration",
            "aegis-human_approval",
            "aegis-research_analysis",
            "aegis-code_review_improvement"
        ]

    def get_model_info(self, model_name: str) -> Optional[dict[str, Any]]:
        """Get information about an Aegis model/workflow."""
        if not self.validate_model_name(model_name):
            return None

        workflow_name = self._extract_workflow_name(model_name)

        workflow_descriptions = {
            "multi_agent_collaboration": {
                "description": "Multi-agent workflow with planning, execution, and review phases",
                "agents": ["planner", "executor", "reviewer"],
                "capabilities": ["planning", "execution", "quality_review", "iteration"],
                "best_for": "Complex projects requiring multiple perspectives and iterative refinement"
            },
            "human_approval": {
                "description": "Workflow requiring human approval before execution",
                "agents": ["request_preparer", "approval_processor"],
                "capabilities": ["human_interaction", "approval_flow", "decision_routing"],
                "best_for": "Sensitive operations requiring human oversight"
            },
            "research_analysis": {
                "description": "Comprehensive research workflow with parallel information gathering",
                "agents": ["research_planner", "web_researcher", "document_analyzer", "data_collector", "synthesizer"],
                "capabilities": ["web_search", "document_analysis", "parallel_processing", "synthesis"],
                "best_for": "Research projects requiring comprehensive information gathering"
            },
            "code_review_improvement": {
                "description": "Comprehensive code review with security, style, and improvement analysis",
                "agents": ["analyzer", "security_reviewer", "style_reviewer", "improvement_generator"],
                "capabilities": ["code_analysis", "security_audit", "style_review", "refactoring"],
                "best_for": "Code quality assurance and improvement"
            }
        }

        workflow_info = workflow_descriptions.get(workflow_name, {})

        return {
            "model_name": model_name,
            "workflow_name": workflow_name,
            "provider": "aegis",
            "type": "workflow",
            "cost_per_1m_input": 0.0,  # Free for now
            "cost_per_1m_output": 0.0,  # Free for now
            **workflow_info
        }


# Convenience functions for integration

def is_aegis_model(model_name: str) -> bool:
    """Check if a model name is an Aegis workflow model."""
    return model_name == "aegis" or model_name.startswith("aegis-")


async def execute_aegis_workflow(
    workflow_name: str,
    prompt: str,
    **kwargs
) -> AegisResponse:
    """Execute an Aegis workflow directly."""
    provider = AegisProvider()
    return await provider.generate_content(
        prompt=prompt,
        model_name=f"aegis-{workflow_name}" if not workflow_name.startswith("aegis") else workflow_name,
        **kwargs
    )


# Integration with provider registry
def register_aegis_provider():
    """Register Aegis provider in the model registry."""
    try:
        from providers.registry import ModelProviderRegistry

        provider = AegisProvider()

        # Register all supported models
        for model_name in provider.get_supported_models():
            ModelProviderRegistry.register_model(model_name, provider)

        logger.info(f"Registered Aegis provider with {len(provider.get_supported_models())} workflow models")

    except ImportError:
        logger.warning("Could not register Aegis provider - registry not available")
    except Exception as e:
        logger.error(f"Failed to register Aegis provider: {e}")


# Initialize on import if available
if LANGGRAPH_AVAILABLE:
    try:
        register_aegis_provider()
    except Exception as e:
        logger.warning(f"Aegis provider registration failed: {e}")
else:
    logger.info("LangGraph not available - Aegis provider disabled")


if __name__ == "__main__":
    # Demo usage
    async def demo():
        if not LANGGRAPH_AVAILABLE:
            print("LangGraph not available. Install with: pip install langgraph")
            return

        provider = AegisProvider()

        print("Available Aegis models:")
        for model in provider.get_supported_models():
            info = provider.get_model_info(model)
            print(f"  {model}: {info.get('description', 'No description')}")

        print("\nExecuting workflow...")

        result = await provider.generate_content(
            prompt="Create a project plan for a new web application with user authentication, database design, and API development",
            model_name="aegis-multi_agent_collaboration",
            temperature=0.3
        )

        print(f"Result: {result.content}")
        print(f"Usage: {result.usage}")
        print(f"Metadata: {result.execution_metadata}")

    asyncio.run(demo())
