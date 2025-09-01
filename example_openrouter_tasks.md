# Unified Tasks API Examples

The POST /tasks endpoint now supports three types of tasks:
1. **LLM Model Tasks** - Direct model execution via providers (sorted by throughput)
2. **Workflow Tasks** - Execute specific AI workflows (analyze, codereview, etc.)  
3. **AgentAPI Tasks** - Traditional external agent executables

## 1. LLM Model Tasks

Direct model execution with automatic provider selection (sorted by throughput):

### Request Format
```bash
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "llm",
    "model": "claude-3.5-sonnet",
    "message": "Write a Python function to calculate fibonacci numbers",
    "system_prompt": "You are a helpful coding assistant. Write clean, well-documented code.",
    "temperature": 0.7,
    "max_tokens": 2000,
    "task_description": "Generate fibonacci function"
  }'
```

### Response Format
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "agent_type": "llm",
  "model": "claude-3.5-sonnet",
  "provider": "google",
  "result": {
    "content": "def fibonacci(n):\n    \"\"\"Calculate the nth fibonacci number.\"\"\"\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "usage": {
      "input_tokens": 45,
      "output_tokens": 128,
      "total_tokens": 173
    },
    "model_name": "gemini-2.0-flash-exp",
    "finish_reason": "stop"
  },
  "request": {
    "message": "Write a Python function to calculate fibonacci numbers",
    "system_prompt": "You are a helpful coding assistant. Write clean, well-documented code.",
    "temperature": 0.7,
    "max_tokens": 2000,
    "task_description": "Generate fibonacci function"
  },
  "created_at": "2025-08-30T10:30:00Z",
  "completed_at": "2025-08-30T10:30:02Z"
}
```

## 2. Workflow Tasks

Execute specific AI workflows using built-in tools:

### Request Format
```bash
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": "analyze",
    "message": "def slow_function():\n    time.sleep(10)\n    return \"done\"",
    "workflow_params": {
      "focus": "performance"
    },
    "model": "claude-3.5-sonnet"
  }'
```

### Supported Workflows
- `analyze` - Code analysis and recommendations
- `codereview` - Code review and suggestions
- `refactor` - Code refactoring recommendations  
- `testgen` - Generate unit tests
- `debug` - Debug issue analysis
- `chat` - General AI conversation
- `thinkdeep` - Extended reasoning
- `consensus` - Multi-model consensus
- `docgen` - Documentation generation
- `secaudit` - Security audit
- `planner` - Task planning
- `challenge` - Challenge generation

### Response Format
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440001",
  "status": "completed",
  "agent_type": "workflow",
  "workflow": "analyze",
  "tool": "AnalyzeTool",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "**Performance Issues Found:**\n\n1. **Blocking Sleep Operation**: The function uses `time.sleep(10)` which blocks the entire thread..."
      }
    ],
    "metadata": {
      "model_used": "claude-3.5-sonnet",
      "issues_found": 2
    }
  },
  "created_at": "2025-08-30T10:30:00Z",
  "completed_at": "2025-08-30T10:30:03Z"
}
```

## 3. Traditional AgentAPI Tasks

Execute external agent executables:

### Request Format
```bash
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "aider",
    "task_description": "Refactor Python code",
    "message": "Please refactor the main.py file to use better error handling",
    "working_directory": "/path/to/project",
    "files": ["main.py"]
  }'
```

### Response Format
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440001", 
  "status": "pending"
}
```

## Key Differences

| Feature | LLM Tasks | Workflow Tasks | AgentAPI Tasks |
|---------|-----------|----------------|----------------|
| **Execution** | Immediate (sync) | Immediate (sync) | Asynchronous via AgentAPI |
| **Provider Selection** | Auto (throughput-sorted) | Auto (throughput-sorted) | External executables |
| **Response Time** | ~1-3 seconds | ~2-5 seconds | Minutes (depends on agent) |
| **Streaming** | No | No | Yes (via SSE) |
| **State Management** | Stateless | Stateless | Persistent via task manager |
| **Use Cases** | Direct model queries | Structured AI workflows | Complex agent workflows |

## Provider Selection (Throughput-Sorted)

Providers are automatically selected based on throughput for optimal performance:

1. **Custom/Local** - Fastest (no network latency)
2. **Google (Gemini)** - Fast native API
3. **OpenAI** - Fast native API  
4. **X.AI (Grok)** - Fast native API
5. **DIAL** - Unified API (may add latency)
6. **OpenRouter** - Aggregated (highest latency, most models)

## Supported Models

All models from configured providers are supported:
- `gemini-2.0-flash-exp` - Google Gemini (fastest)
- `gpt-4o` - OpenAI GPT-4 Omni
- `claude-3.5-sonnet` - Anthropic Claude via OpenRouter
- `grok-2` - X.AI Grok
- Custom model aliases from your configuration

## Configuration

Configure providers for optimal throughput:

```bash
# Primary providers (fastest)
export GEMINI_API_KEY="your-gemini-key"
export OPENAI_API_KEY="your-openai-key"
export XAI_API_KEY="your-grok-key"

# Secondary providers
export OPENROUTER_API_KEY="your-openrouter-key"
export DIAL_API_KEY="your-dial-key"

# Custom/Local provider (fastest)
export CUSTOM_API_URL="http://localhost:11434"  # e.g., Ollama
export CUSTOM_API_KEY=""  # Optional

# Cost limits (per 1M tokens)
export OPENROUTER_MAX_COST_PER_1M_TOTAL="10.00"  # $10 per 1M tokens

# Morph FastApply configuration (uses OpenRouter)
export ZEN_EDIT_PROVIDER="morph"  # or "builtin"
export MORPH_MODEL="anthropic/claude-3.5-sonnet"  # OpenRouter model for Morph
```

## Error Responses

### Missing Model
```json
{
  "error": "model_not_available",
  "details": "Model 'invalid-model' is not available from any configured provider"
}
```

### Invalid Workflow
```json
{
  "error": "invalid_workflow",
  "details": "Workflow 'badworkflow' not supported. Available: ['analyze', 'codereview', 'refactor', ...]"
}
```

### Task Failed
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "agent_type": "llm",
  "error": "Rate limit exceeded",
  "model": "gpt-4o",
  "created_at": "2025-08-30T10:30:00Z"
}
```