# Aegis LangGraph Integration

This document describes the Aegis provider integration with LangGraph for sophisticated multi-agent workflow orchestration in the Zen MCP Server.

## Overview

The Aegis provider enables the "aegis" model option, which routes requests to LangGraph-powered workflows instead of single LLM execution. This provides:

- **Multi-agent coordination** with specialized roles
- **Human-in-the-loop** approval workflows  
- **Parallel processing** for complex research tasks
- **Conditional routing** based on intermediate results
- **State management** with persistence and recovery
- **Tool integration** with all existing MCP tools

## Installation

### Prerequisites

1. **Python 3.9+** with the Zen MCP Server installed
2. **LangGraph dependencies**:
   ```bash
   pip install -r requirements-langgraph.txt
   ```

### Verify Installation

```bash
python -c "from providers.aegis import AegisProvider; print('Aegis provider available')"
```

## Usage

### Basic Usage

#### Universal Executor Tool

```json
{
  "agent_type": "llm",
  "model": "aegis",
  "prompt": "Create a comprehensive project plan for a new web application",
  "execution_mode": "sync",
  "memory_mode": "stateful"
}
```

#### HTTP API

```bash
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "llm", 
    "model": "aegis-multi_agent_collaboration",
    "prompt": "Design and implement a user authentication system",
    "files": ["src/auth.py", "docs/requirements.md"],
    "temperature": 0.3
  }'
```

### Available Workflows

#### 1. Multi-Agent Collaboration (`aegis-multi_agent_collaboration`)

**Best for**: Complex projects requiring multiple perspectives

**Agents**:
- **Planner**: Breaks down tasks into actionable steps
- **Executor**: Implements solutions with tool access
- **Reviewer**: Quality assurance and feedback
- **Decision**: Conditional routing for iteration

**Example**:
```json
{
  "model": "aegis-multi_agent_collaboration",
  "prompt": "Build a REST API with authentication, database, and documentation",
  "temperature": 0.3,
  "max_tokens": 4000
}
```

#### 2. Human Approval Workflow (`aegis-human_approval`)

**Best for**: Sensitive operations requiring human oversight

**Process**:
1. Prepare approval request with context
2. Request human approval (configurable timeout)
3. Process approval/rejection
4. Execute approved actions or handle rejection

**Example**:
```json
{
  "model": "aegis-human_approval", 
  "prompt": "Deploy the new authentication system to production",
  "workflow_params": {
    "approval_timeout_minutes": 120,
    "required_approvers": 2
  }
}
```

#### 3. Research Analysis (`aegis-research_analysis`)

**Best for**: Comprehensive information gathering and synthesis

**Agents**:
- **Research Planner**: Develops research strategy
- **Parallel Researchers**: Web search, document analysis, data collection
- **Synthesizer**: Combines findings into comprehensive analysis

**Example**:
```json
{
  "model": "aegis-research_analysis",
  "prompt": "Research current best practices for microservices architecture",
  "use_websearch": true,
  "files": ["docs/architecture.md"]
}
```

#### 4. Code Review Improvement (`aegis-code_review_improvement`)

**Best for**: Code quality assurance and enhancement

**Tools Used**:
- `analyze`: Initial code quality analysis
- `secaudit`: Security vulnerability assessment  
- `codereview`: Style and best practices review
- `refactor`: Generate improvements

**Example**:
```json
{
  "model": "aegis-code_review_improvement",
  "prompt": "Review and improve this authentication module",
  "files": ["src/auth/login.py", "src/auth/middleware.py"]
}
```

### Advanced Features

#### Streaming Support

```json
{
  "model": "aegis",
  "prompt": "Your request...",
  "stream_mode": true
}
```

Response includes `stream_url` and `websocket_url` for real-time updates.

#### Conversation Memory

```json
{
  "model": "aegis-multi_agent_collaboration",
  "prompt": "Continue working on the API design",
  "continuation_id": "existing-conversation-uuid",
  "memory_mode": "continuation"
}
```

#### Batch Processing

```json
{
  "batch_items": [
    {
      "model": "aegis-code_review_improvement",
      "prompt": "Review authentication module",
      "files": ["src/auth.py"]
    },
    {
      "model": "aegis-research_analysis", 
      "prompt": "Research OAuth 2.0 best practices"
    }
  ],
  "batch_mode": "parallel"
}
```

## Configuration

### Environment Variables

```bash
# MCP server URL for tool integration (default: http://localhost:8080/mcp)
export MCP_SERVER_URL="http://localhost:8080/mcp"

# Enable LangSmith tracing (optional)
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-langsmith-key"

# Redis for state persistence (optional)
export REDIS_URL="redis://localhost:6379"
```

### Workflow Customization

Create custom workflows by extending the `AegisLangGraphProvider`:

```python
from integrations.langgraph_wrapper import AegisLangGraphProvider, WorkflowDefinition, WorkflowNode

class CustomAegisProvider(AegisLangGraphProvider):
    def _register_default_workflows(self):
        super()._register_default_workflows()
        
        # Add custom workflow
        custom_workflow = WorkflowDefinition(
            name="custom_workflow",
            description="Custom business logic workflow",
            nodes=[...],  # Define your nodes
            start_node="start",
            end_nodes=["END"]
        )
        
        self.register_workflow(custom_workflow)
```

## API Reference

### Model Names

| Model Name | Workflow | Description |
|-----------|----------|-------------|
| `aegis` | Multi-agent collaboration | Default workflow (alias) |
| `aegis-multi_agent_collaboration` | Multi-agent collaboration | Planning, execution, review cycle |
| `aegis-human_approval` | Human approval | Human-in-the-loop decision making |
| `aegis-research_analysis` | Research analysis | Parallel information gathering |
| `aegis-code_review_improvement` | Code review | Comprehensive code analysis |

### Response Format

```json
{
  "task_id": "uuid",
  "status": "completed",
  "agent_type": "llm", 
  "model": "aegis-multi_agent_collaboration",
  "continuation_id": "uuid",
  "result": {
    "content": "Workflow execution results...",
    "usage": {
      "workflow_steps": 5,
      "tools_executed": 3,
      "agents_involved": 3,
      "execution_time_seconds": 45.2
    },
    "execution_metadata": {
      "workflow_name": "multi_agent_collaboration",
      "execution_id": "uuid",
      "steps_executed": 5,
      "tools_used": ["analyze", "codereview"],
      "agents_involved": ["planner", "executor", "reviewer"]
    }
  }
}
```

### Error Handling

Aegis workflows provide detailed error information:

```json
{
  "status": "failed",
  "error": "Workflow execution failed: Agent 'executor' encountered error",
  "execution_metadata": {
    "workflow_name": "multi_agent_collaboration",
    "failed_step": "executor",
    "error_details": "Tool 'analyze' failed: File not found"
  }
}
```

## Performance and Cost

### Performance Characteristics

- **Latency**: Higher than single LLM calls due to multi-step processing
- **Throughput**: Optimized through parallel agent execution where possible
- **Scalability**: Limited by underlying LLM provider rate limits

### Cost Structure

Currently **free** for workflow orchestration (you pay only for underlying LLM calls):

- Workflow coordination: $0.00 per execution
- LLM agent calls: Standard provider rates
- Tool executions: Free (local MCP tools)
- State persistence: Free (in-memory), Redis costs if configured

## Monitoring and Debugging

### Execution Tracking

Monitor active workflows:

```bash
curl http://localhost:8080/metrics
```

Response includes workflow statistics:
```json
{
  "workflow_executions": {
    "active": 3,
    "completed_today": 47,
    "failed_today": 2
  }
}
```

### Workflow Status

Check individual workflow status:

```bash
curl http://localhost:8080/tasks/{workflow_execution_id}
```

### LangSmith Integration

Enable tracing for detailed workflow debugging:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-key"
```

View execution traces at [smith.langchain.com](https://smith.langchain.com)

## Troubleshooting

### Common Issues

#### 1. LangGraph Not Available

```
Error: LangGraph not available for Aegis provider
```

**Solution**: Install LangGraph dependencies:
```bash
pip install -r requirements-langgraph.txt
```

#### 2. MCP Client Connection Failed

```
Error: Failed to initialize Aegis provider: Connection refused
```

**Solution**: Ensure MCP server is running:
```bash
python server_http.py --host 0.0.0.0 --port 8080
```

#### 3. Workflow Execution Timeout

```
Error: Workflow execution failed: Timeout after 300 seconds
```

**Solution**: Increase timeout or simplify workflow:
```json
{
  "max_steps": 10,
  "timeout_seconds": 600
}
```

#### 4. Tool Not Found

```
Error: Tool 'analyze' failed: Tool not found
```

**Solution**: Ensure all required tools are available:
```bash
curl http://localhost:8080/mcp/tools/list
```

### Debug Mode

Enable detailed logging:

```bash
export LOG_LEVEL=DEBUG
python server_http.py
```

## Examples

### Complete Project Planning Workflow

```python
import asyncio
from providers.aegis import AegisProvider

async def plan_project():
    provider = AegisProvider()
    
    result = await provider.generate_content(
        prompt="""
        Plan and design a complete e-commerce web application with:
        - User authentication and authorization
        - Product catalog with search and filtering
        - Shopping cart and checkout process
        - Payment integration
        - Admin dashboard
        - Mobile responsiveness
        - Security best practices
        """,
        model_name="aegis-multi_agent_collaboration",
        temperature=0.3,
        files=["requirements.md", "architecture.md"],
        thinking_mode="high"
    )
    
    print(f"Project Plan:\n{result.content}")
    print(f"Execution took {result.usage.execution_time_seconds:.1f} seconds")
    print(f"Used {result.usage.agents_involved} agents and {result.usage.tools_executed} tools")

if __name__ == "__main__":
    asyncio.run(plan_project())
```

### Code Review Automation

```python
async def automated_code_review():
    provider = AegisProvider()
    
    result = await provider.generate_content(
        prompt="Perform comprehensive code review focusing on security, performance, and maintainability",
        model_name="aegis-code_review_improvement",
        files=[
            "src/auth/login.py",
            "src/auth/middleware.py", 
            "src/api/users.py",
            "tests/test_auth.py"
        ],
        temperature=0.2
    )
    
    print(f"Code Review Results:\n{result.content}")

asyncio.run(automated_code_review())
```

### Research with Human Approval

```python
async def research_with_approval():
    provider = AegisProvider()
    
    # First, conduct research
    research_result = await provider.generate_content(
        prompt="Research the latest trends in AI-powered development tools and their impact on productivity",
        model_name="aegis-research_analysis",
        use_websearch=True,
        temperature=0.4
    )
    
    # Then, get human approval for recommendations
    approval_result = await provider.generate_content(
        prompt=f"Based on this research, should we invest in these AI tools? Research: {research_result.content}",
        model_name="aegis-human_approval",
        workflow_params={
            "approval_timeout_minutes": 60,
            "required_approvers": 1
        }
    )
    
    print(f"Research: {research_result.content}")
    print(f"Approval Decision: {approval_result.content}")

asyncio.run(research_with_approval())
```

## Future Enhancements

The Aegis provider is designed to support the Phase 3 and 4 enhancements outlined in the LLM Architecture Future Plans:

- **Advanced streaming** with tool interaction during response generation
- **Multi-modal workflows** supporting image, audio, and video processing
- **Distributed orchestration** across multiple server instances
- **Custom framework adapters** for integration with other agent frameworks
- **Advanced state management** with branching and merging capabilities

## Contributing

To add new workflows or enhance existing ones:

1. Extend `WorkflowDefinition` in `integrations/langgraph_wrapper.py`
2. Add workflow to `_register_default_workflows()`
3. Update model validation in `providers/aegis.py`
4. Add tests in `tests/test_aegis_provider.py`
5. Update this documentation

## Support

For issues related to Aegis provider:

1. Check the troubleshooting section above
2. Review logs with `LOG_LEVEL=DEBUG`
3. Test with simpler workflows first
4. Report issues with full error traces and workflow definitions