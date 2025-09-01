# Zen MCP Server - Codebase Guide

## Overview

Zen MCP Server is a comprehensive Model Context Protocol (MCP) server that provides AI-powered tools for development workflows. This guide helps both LLMs and humans understand the codebase structure, patterns, and how to work with it effectively.

## Architecture Overview

```
zen-mcp-server/
├── server.py              # (Archived) STDIO MCP server entry point
├── server_mcp_http.py     # Main Streamable HTTP MCP server
├── tools/                  # MCP tool implementations
│   ├── shared/            # Shared base classes and models
│   ├── workflow/          # Workflow-based tools (analyze, debug, etc.)
│   ├── agent_*.py         # Agent orchestration tools
│   └── *.py              # Individual tools (chat, consensus, etc.)
├── providers/             # AI model provider implementations
├── utils/                 # Utility modules
├── systemprompts/         # System prompt definitions
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Core Concepts

### 1. MCP Tools
All tools inherit from `BaseTool` and implement the MCP protocol:
- **Simple Tools**: Direct AI model interaction (chat, thinkdeep)
- **Workflow Tools**: Multi-step processes (analyze, debug, codereview)
- **Agent Tools**: Orchestrate external CLI agents (agent_sync, agent_batch)

### 2. Model Providers
Abstracted AI model access through provider pattern:
- OpenAI, Gemini, XAI, OpenRouter, Custom providers
- Automatic model selection in AUTO mode
- Provider-specific capabilities and restrictions

### 3. Conversation Memory
Stateless-to-stateful bridging for multi-turn conversations:
- Thread-based conversation tracking
- File deduplication across turns
- Context-aware file embedding

## Key Patterns

### Tool Implementation Pattern
```python
class MyTool(BaseTool):
    def get_name(self) -> str:
        return "my_tool"
    
    def get_description(self) -> str:
        return "Tool description"
    
    def get_request_model(self):
        # Return Pydantic model for validation
        
    def get_system_prompt(self) -> str:
        # Return system prompt for AI
        
    async def prepare_prompt(self, request) -> str:
        # Prepare complete prompt
        
    async def execute(self, arguments: dict) -> list[TextContent]:
        # Main tool logic
```

### Workflow Tool Pattern
```python
class MyWorkflowTool(WorkflowTool):
    def get_required_actions(self, step_number, confidence, findings, total_steps, request=None):
        # Define actions for each workflow step
        
    def should_call_expert_analysis(self, consolidated_findings):
        # Determine when to call AI for analysis
```

### Provider Pattern
```python
class MyProvider(ModelProvider):
    def validate_model_name(self, model_name: str) -> bool:
        # Validate model availability
        
    async def generate_content(self, messages, model_name, **kwargs):
        # Generate AI response
```

## File Organization

### Tools Directory Structure
- `shared/`: Base classes, models, utilities shared across tools
- `workflow/`: Multi-step workflow tools with expert analysis
- Individual tool files: Single-purpose tools

### Key Files
- `server_mcp_http.py`: MCP HTTP server setup, tool registration, request handling
- `server.py` (archived): legacy STDIO MCP server
- `tools/shared/base_tool.py`: Base tool class with conversation memory
- `utils/conversation_memory.py`: Thread-based conversation tracking
- `providers/registry.py`: Model provider registry and selection

## Development Guidelines

### Adding New Tools
1. Create tool class inheriting from `BaseTool` or `WorkflowTool`
2. Implement required abstract methods
3. Add to `TOOLS` dictionary in `server.py`
4. Create comprehensive tests
5. Update documentation

### Adding New Providers
1. Create provider class inheriting from `ModelProvider`
2. Implement required methods
3. Register in provider registry
4. Add configuration and validation
5. Test with various models

### Testing Strategy
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Simulator tests for real AI model interactions
- Mock external dependencies appropriately

## Common Patterns and Utilities

### File Handling
```python
from utils.file_utils import read_file_content, read_files
# Handles absolute paths, security, token limits
```

### Conversation Context
```python
from utils.conversation_memory import get_conversation_memory
# Thread-based conversation tracking
```

### Model Selection
```python
# AUTO mode automatically selects best model
# Provider registry handles model resolution
```

### Error Handling
```python
# Consistent error responses
# Graceful degradation
# Comprehensive logging
```

## Configuration

### Environment Variables
- API keys for providers (OPENAI_API_KEY, etc.)
- Model restrictions (OPENAI_ALLOWED_MODELS, etc.)
- Custom endpoints (CUSTOM_API_URL)
- Feature flags (DISABLED_TOOLS)

### Model Configuration
- AUTO mode for automatic selection
- Provider-specific model lists
- Capability-based selection
- Fallback strategies

## Testing

### Test Categories
- Unit tests: Individual component testing
- Integration tests: End-to-end workflows
- Simulator tests: Real AI model interactions
- Performance tests: Load and stress testing

### Running Tests
```bash
# Unit tests only
python -m pytest tests/ -m "not integration"

# All tests including integration
python -m pytest tests/

# Simulator tests (requires API keys)
python communication_simulator_test.py --quick
```

## Debugging and Monitoring

### Logging
- Structured logging with levels
- Activity logging for tool executions
- Error tracking and reporting

### Log Files
- `logs/mcp_server.log`: Main server activity
- `logs/mcp_activity.log`: Tool execution tracking

### Monitoring Tools
```bash
# View logs
tail -f logs/mcp_server.log

# Check server status
./run-server.sh

# Run quality checks
./code_quality_checks.sh
```

## Best Practices

### Code Quality
- Follow Python type hints
- Use Pydantic for validation
- Implement comprehensive error handling
- Write descriptive docstrings

### Performance
- Implement token budgeting
- Use async/await appropriately
- Cache expensive operations
- Optimize file reading

### Security
- Validate all inputs
- Use absolute paths only
- Sanitize file access
- Implement proper authentication

## Extension Points

### Custom Tools
- Inherit from BaseTool or WorkflowTool
- Follow established patterns
- Implement proper validation

### Custom Providers
- Implement ModelProvider interface
- Handle authentication properly
- Support standard capabilities

### Custom Workflows
- Use WorkflowTool base class
- Implement step-by-step logic
- Support continuation and memory

This guide provides the foundation for understanding and working with the zen-mcp-server codebase effectively.
