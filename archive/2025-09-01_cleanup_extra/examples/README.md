# MCP Streamable HTTP Examples

This directory contains comprehensive examples demonstrating the MCP Streamable HTTP functionality implemented in the Zen MCP Server.

## Quick Start

### 1. Start the MCP HTTP Server

```bash
# Start the streamable HTTP server (no authentication required)
python server_mcp_http.py

# Server starts on http://localhost:8080/mcp by default
# Check server status: curl http://localhost:8080/mcp
```

### 2. Test with Python Client

```bash
# Run demo mode to test all functionality
python clients/mcp_http_client.py --demo

# Interactive shell for manual testing  
python clients/mcp_http_client.py --interactive

# Connect to custom server
python clients/mcp_http_client.py --demo --server http://remote-server:8080/mcp
```

### 3. Test with TypeScript Client

```bash
cd clients
npm install        # Install TypeScript dependencies
npm run build      # Build TypeScript client

# Run demo
npm run demo

# Interactive mode
npm run interactive
```

## Core Features Demonstrated

### Remote Agent Orchestration
- Register remote MCP servers as agents
- Execute tools on remote servers
- Read resources from remote servers
- Cross-server coordination and workflows

### No Authentication Access
- Open HTTP access without authentication
- Simplified deployment for internal networks
- Quick prototyping and development

### Cross-Language Compatibility
- Python server with TypeScript clients
- JSON-RPC 2.0 protocol compliance
- Session management across languages

### Streaming Support
- Streamable HTTP transport (MCP 2025-03-26)
- Bidirectional communication
- Real-time tool execution

## Example Workflows

### Basic Tool Execution
```python
async with MCPStreamableHTTPClient("http://localhost:8080/mcp") as client:
    # Simple tool call
    result = await client.call_tool("echo", {"text": "Hello World"})
    print(result)
    
    # Math operation
    result = await client.call_tool("multiply", {"a": 6, "b": 7})
    print(f"6 * 7 = {result}")
```

### Resource Access
```python
# List available resources
resources = await client.list_resources()
for resource in resources:
    content = await client.read_resource(resource.uri)
    print(f"{resource.name}: {content[:100]}...")
```

### Remote Agent Registration
```python
# Using the mcp_agent tool to register remote servers
await client.call_tool("mcp_agent", {
    "action": "register",
    "agent_name": "remote-server",
    "agent_url": "http://remote-host:8080/mcp"
})

# Execute tool on remote agent
result = await client.call_tool("mcp_agent", {
    "action": "execute", 
    "agent_name": "remote-server",
    "tool_name": "analyze",
    "tool_args": {"code": "def hello(): pass"}
})
```

## Architecture Overview

```
┌─────────────────┐    HTTP/JSON-RPC    ┌─────────────────┐
│   MCP Client    │ ◄─────────────────► │   MCP Server    │
│  (Python/TS)    │                     │  (FastAPI)      │
└─────────────────┘                     └─────────────────┘
                                                │
                                                ▼
                                        ┌─────────────────┐
                                        │  Zen MCP Tools  │
                                        │  • chat         │
                                        │  • analyze      │
                                        │  • codereview   │
                                        │  • mcp_agent    │
                                        │  • ...          │
                                        └─────────────────┘
```

## Running Examples

All examples are located in this directory and can be run independently:

```bash
# Basic connection test
python basic_connection.py

# Multi-server orchestration
python remote_orchestration.py

# Cross-language workflow
python cross_language_demo.py

# Performance testing
python performance_test.py
```

## Troubleshooting

### Connection Issues
```bash
# Check server status
curl http://localhost:8080/mcp

# Test server health
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"ping","id":1}'
```

### Session Problems
- Session IDs are automatically managed via `Mcp-Session-Id` headers
- Each client connection gets a unique session
- Sessions persist across multiple requests

### CORS Issues
- Server includes `Access-Control-Allow-Origin: *` for development
- Modify CORS settings in `server_mcp_http.py` for production

## Next Steps

1. **Production Deployment**: Add authentication and HTTPS
2. **Load Balancing**: Scale with multiple server instances  
3. **Monitoring**: Add logging and metrics collection
4. **Security**: Implement proper auth and input validation