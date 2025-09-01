# Agent Orchestration Guide
Status: Implemented — tools and workflows are available; see reports in `docs/reports/`.

The zen-mcp-server now supports powerful agent orchestration capabilities that allow the primary MCP client (Claude, Gemini CLI, etc.) to delegate work to sub-agents and coordinate complex multi-agent workflows.

## Overview

Agent orchestration enables you to:
- **Delegate Tasks**: Assign specific tasks to specialized CLI agents (claude, aider, goose, etc.)
- **Parallel Execution**: Run multiple agents simultaneously for faster completion
- **Specialized Selection**: Choose the best agent for each specific task type
- **Background Processing**: Continue working while agents execute tasks in the background
- **Result Coordination**: Collect and aggregate results from multiple agents

## Available Tools

### 1. `agent_registry` - Discover Available Agents

Discover and describe available AgentAPI-supported CLI agents with their capabilities.

```json
{
  "agent_type": "claude",           // Optional: specific agent to query
  "check_availability": true,       // Check if agents are installed
  "include_capabilities": true      // Include detailed capability descriptions
}
```

**Example Usage:**
```json
{
  "check_availability": true,
  "include_capabilities": true
}
```

### 2. `agent_sync` - Synchronous Task Execution

Execute a single agent task synchronously and wait for completion.

```json
{
  "agent_type": "claude",
  "task_description": "Code review for authentication module",
  "message": "Please review the auth.py file for security issues and best practices",
  "timeout_seconds": 300,
  "working_directory": "/path/to/project",
  "agent_args": ["--allowedTools", "Edit Replace"],
  "env_vars": {"ANTHROPIC_API_KEY": "your-key"},
  "files": ["auth.py", "tests/test_auth.py"]
}
```

**Best for:** Quick tasks, code analysis, single-file edits, debugging assistance

### 3. `agent_async` - Asynchronous Task Execution

Launch a background agent task and return immediately with a task ID.

```json
{
  "agent_type": "aider",
  "task_description": "Refactor user management system",
  "message": "Please refactor the user management code to use dependency injection",
  "timeout_seconds": 1800,
  "priority": "high",
  "working_directory": "/path/to/project"
}
```

**Best for:** Long-running tasks, parallel execution, complex refactoring, comprehensive analysis

### 4. `agent_inbox` - Monitor and Retrieve Results

Check status and retrieve results from asynchronous agent tasks.

```json
{
  "task_id": "abc123-def456-789",   // Optional: specific task ID
  "action": "results",              // status, results, list, cancel
  "include_messages": true,
  "max_message_length": 1000
}
```

**Actions:**
- `status`: Check task status and progress
- `results`: Get complete results and conversation history
- `list`: List all active/recent tasks
- `cancel`: Cancel a running task

### 5. `agent_batch` - Parallel Task Coordination

Launch multiple agent tasks in parallel with coordination and result aggregation.

```json
{
  "tasks": [
    {
      "agent_type": "claude",
      "task_description": "Frontend: Home Page",
      "message": "Create a responsive home page component with modern design"
    },
    {
      "agent_type": "aider", 
      "task_description": "Backend: User API",
      "message": "Implement user CRUD API endpoints with validation"
    }
  ],
  "coordination_strategy": "parallel",
  "max_concurrent": 5,
  "fail_fast": false,
  "timeout_seconds": 3600,
  "batch_description": "CRUD todo app development"
}
```

## Agent Types and Capabilities

### Claude Code
- **Strengths**: Code generation, analysis, debugging, best practices
- **Use Cases**: Code reviews, feature implementation, documentation
- **Requirements**: `ANTHROPIC_API_KEY`

### Aider
- **Strengths**: Direct file editing, git integration, repository-wide changes
- **Use Cases**: Refactoring, implementing features, cross-file changes
- **Requirements**: `ANTHROPIC_API_KEY`, git repository

### Goose
- **Strengths**: Task automation, command execution, environment setup
- **Use Cases**: Build automation, testing workflows, deployment tasks
- **Requirements**: System permissions

### Other Agents
- **Codex**: OpenAI-powered code generation (`OPENAI_API_KEY`)
- **Gemini**: Google's AI for code tasks (`GOOGLE_API_KEY`)
- **Amp**: Sourcegraph's code intelligence
- **Cursor**: AI-powered code editor integration

## Complete Workflow Example: CRUD Todo App

Here's a complete example of building a CRUD todo app using agent orchestration:

### Step 1: Discover Available Agents

```json
{
  "tool": "agent_registry",
  "arguments": {
    "check_availability": true,
    "include_capabilities": true
  }
}
```

### Step 2: Launch Parallel Development Tasks

```json
{
  "tool": "agent_batch",
  "arguments": {
    "batch_description": "CRUD Todo App Development",
    "coordination_strategy": "parallel",
    "max_concurrent": 4,
    "tasks": [
      {
        "agent_type": "claude",
        "task_description": "Frontend: Home Page Component",
        "message": "Create a React home page component with:\n- Welcome message\n- Navigation to todo list\n- Modern, responsive design\n- TypeScript types",
        "working_directory": "./frontend",
        "files": ["src/components/", "src/types/"]
      },
      {
        "agent_type": "claude", 
        "task_description": "Frontend: Todo List Component",
        "message": "Create a React todo list component with:\n- Display todos with status\n- Add new todo functionality\n- Mark complete/incomplete\n- Delete todos\n- Filter by status",
        "working_directory": "./frontend"
      },
      {
        "agent_type": "aider",
        "task_description": "Backend: Database Schema",
        "message": "Create database schema and models for:\n- User authentication\n- Todo items with CRUD operations\n- Proper relationships and constraints\n- Migration files",
        "working_directory": "./backend"
      },
      {
        "agent_type": "aider",
        "task_description": "Backend: API Endpoints", 
        "message": "Implement REST API endpoints:\n- POST /auth/login, /auth/register\n- GET/POST/PUT/DELETE /api/todos\n- Proper validation and error handling\n- JWT authentication middleware",
        "working_directory": "./backend"
      }
    ]
  }
}
```

### Step 3: Monitor Progress

```json
{
  "tool": "agent_inbox",
  "arguments": {
    "action": "list"
  }
}
```

### Step 4: Get Individual Results

```json
{
  "tool": "agent_inbox", 
  "arguments": {
    "task_id": "frontend-home-task-id",
    "action": "results",
    "include_messages": true
  }
}
```

### Step 5: Launch Additional Tasks Based on Results

```json
{
  "tool": "agent_async",
  "arguments": {
    "agent_type": "claude",
    "task_description": "Integration: Connect Frontend to Backend",
    "message": "Create API client and integrate frontend components with backend:\n- API service layer\n- Error handling\n- Loading states\n- Type safety",
    "timeout_seconds": 1200
  }
}
```

## Best Practices

### Task Decomposition
- Break large projects into focused, independent tasks
- Assign tasks to agents based on their strengths
- Consider dependencies when planning parallel execution

### Agent Selection
- **Claude**: Best for analysis, planning, and high-quality code generation
- **Aider**: Best for direct file editing and repository-wide changes  
- **Goose**: Best for automation and system-level tasks

### Resource Management
- Use `max_concurrent` to control resource usage
- Set appropriate timeouts for different task types
- Monitor tasks regularly with `agent_inbox`

### Error Handling
- Use `fail_fast: false` for independent tasks
- Set realistic timeouts based on task complexity
- Have fallback strategies for failed tasks

## Troubleshooting

### Common Issues

1. **Agent Not Available**
   - Ensure AgentAPI is installed: `npm install -g agentapi`
   - Check agent installation: `which claude`, `which aider`, etc.
   - Verify required environment variables

2. **Task Timeout**
   - Increase timeout for complex tasks
   - Break large tasks into smaller chunks
   - Check agent logs for specific issues

3. **Port Conflicts**
   - Agent orchestration uses ports 3284-3384
   - Ensure ports are available
   - Check for other AgentAPI instances

4. **Authentication Errors**
   - Verify API keys are set correctly
   - Check environment variable names
   - Ensure keys have proper permissions

### Getting Help

- Use `agent_registry` to check agent availability
- Monitor tasks with `agent_inbox` for detailed status
- Check server logs for detailed error information
- Use `action: cancel` to stop problematic tasks

## Advanced Features

### Custom Environment Variables
```json
{
  "env_vars": {
    "CUSTOM_MODEL": "gpt-4",
    "DEBUG_MODE": "true",
    "PROJECT_ROOT": "/path/to/project"
  }
}
```

### File Context Sharing
```json
{
  "files": [
    "src/types/index.ts",
    "docs/api-spec.md", 
    "config/database.json"
  ]
}
```

### Priority-Based Execution
```json
{
  "priority": "high"  // low, normal, high
}
```

This agent orchestration system transforms zen-mcp-server into a powerful multi-agent coordination platform, enabling complex development workflows with specialized agent expertise.
