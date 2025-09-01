# KInfra Integration for Zen MCP HTTP Server
Location: This guide has been moved under `docs/howto/` for clarity.

This document explains the KInfra integration that adds smart port allocation and Cloudflare tunnel management to the Zen MCP HTTP Server.

## 🌟 Features

- **Smart Port Allocation**: Automatically find available ports with fallback strategies
- **Cloudflare Tunneling**: Expose your MCP server to `zen.kooshapari.com` automatically
- **Health Monitoring**: Built-in health checks and status reporting
- **Configuration Management**: YAML-based configuration with environment overrides
- **Graceful Shutdown**: Proper cleanup of tunnels and resources

## 🚀 Quick Start

### Basic Usage (Local Only)
```bash
# Start server with smart port allocation
python deploy_mcp_http.py

# Or use the original server directly
python server_mcp_http.py --port-strategy dynamic
```

### With Tunnel (Public Access)
```bash
# Enable Cloudflare tunnel to zen.kooshapari.com
python deploy_mcp_http.py --tunnel

# Use custom domain
python deploy_mcp_http.py --tunnel --domain custom.zen.kooshapari.com

# Specify preferred port + tunnel
python deploy_mcp_http.py --port 8080 --tunnel
```

### Advanced Configuration
```bash
# Use environment-based port selection
python deploy_mcp_http.py --port-strategy env --tunnel

# Dynamic port allocation with tunnel
python deploy_mcp_http.py --port-strategy dynamic --tunnel --domain dev.zen.kooshapari.com

# Debug mode
python deploy_mcp_http.py --tunnel --log-level DEBUG
```

## 📋 Prerequisites

### Required
- Python 3.8+
- FastAPI and Uvicorn (from requirements.txt)
- KInfra libraries (bundled in `./KInfra/`)

### For Tunneling
- [Cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) installed
- Cloudflare account (free tier works)
- Authentication: `cloudflared tunnel login`

## 🔧 Configuration

### Command Line Options

#### deploy_mcp_http.py
```
--host HOST               Host to bind to (default: 0.0.0.0)
--port PORT              Preferred port (smart allocation if not specified)
--tunnel                 Enable Cloudflare tunnel
--domain DOMAIN          Tunnel domain (default: zen.kooshapari.com)
--port-strategy STRATEGY Port allocation strategy (preferred/dynamic/env)
--log-level LEVEL        Logging level (DEBUG/INFO/WARN/ERROR)
--status                 Show server status
--config FILE            Configuration file path
```

#### server_mcp_http.py (Original)
```
--host HOST               Host to bind to
--port PORT              Preferred port
--tunnel                 Enable tunnel
--domain DOMAIN          Tunnel domain
--port-strategy STRATEGY Port allocation strategy
--log-level LEVEL        Logging level
```

### Environment Variables
```bash
# Port configuration
PORT=8080                           # Preferred port
PREFERRED_PORT=3000                 # Alternative preferred port

# Tunnel configuration  
KINFRA_TUNNEL_ENABLED=true         # Enable tunneling
TUNNEL_DOMAIN=custom.kooshapari.com # Custom tunnel domain
CLOUDFLARE_TUNNEL_TOKEN=xyz...      # For persistent tunnels

# KInfra configuration
KINFRA_LOG_LEVEL=DEBUG             # KInfra logging level
KINFRA_PORT_STRATEGY=dynamic       # Port allocation strategy
```

### Configuration File (config/kinfra.yml)

The KInfra configuration file provides detailed control over networking and tunnel behavior:

```yaml
networking:
  port_strategy: "preferred"        # Port allocation strategy
  port_range:
    min: 3000
    max: 9999
  blocked_ports: [22, 80, 443]     # Never allocate these

tunnel:
  type: "quick"                    # quick or persistent
  domain: "zen.kooshapari.com"
  timeout:
    startup: 60000                 # milliseconds
    health_check: 20000
```

## 📊 Monitoring and Status

### Health Endpoints
- `GET /healthz` - Simple health check
- `GET /status` - Detailed server status including tunnel information
- `GET /` - Server information and capabilities

### Status Command
```bash
# Check current server status
python deploy_mcp_http.py --status
```

### Example Status Output
```json
{
  "status": "healthy",
  "host": "0.0.0.0",
  "port": 8080,
  "local_url": "http://0.0.0.0:8080",
  "tunnel_enabled": true,
  "tunnel_status": "running",
  "tunnel_url": "https://zen.kooshapari.com",
  "tunnel_domain": "zen.kooshapari.com"
}
```

## 🔗 Connection URLs

When the server starts, you'll see output like:

```
🚀 Zen MCP Streamable HTTP server starting
📍 Local: http://localhost:8080
🔗 MCP endpoint: http://localhost:8080/mcp
📚 API docs: http://localhost:8080/docs
❤️  Health check: http://localhost:8080/healthz

🌍 Public URL: https://zen.kooshapari.com
🔗 Public MCP: https://zen.kooshapari.com/mcp
❤️  Public Health: https://zen.kooshapari.com/healthz
```

## 🧪 Testing the Integration

### Run Integration Tests
```bash
python test_kinfra_integration.py
```

### Manual Testing
```bash
# Test health endpoint
curl http://localhost:8080/healthz

# Test MCP endpoint
curl -X POST http://localhost:8080/mcp \\
  -H "Content-Type: application/json" \\
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'

# If tunnel is enabled, test public access
curl https://zen.kooshapari.com/healthz
```

## 🔧 Port Allocation Strategies

### Preferred (Default)
- Uses specified port if available
- Falls back to OS allocation if not available
- Best for development with consistent ports

```bash
python deploy_mcp_http.py --port 8080 --port-strategy preferred
```

### Dynamic
- Always uses OS-allocated port
- Best for avoiding conflicts in CI/CD
- Most reliable for automated deployments

```bash
python deploy_mcp_http.py --port-strategy dynamic
```

### Environment
- Uses PORT environment variable
- Falls back to OS allocation if PORT not set or unavailable
- Best for containerized deployments

```bash
PORT=3000 python deploy_mcp_http.py --port-strategy env
```

## 🌐 Tunnel Types

### Quick Tunnels (Default)
- Temporary tunnels with auto-generated subdomains
- Perfect for development and testing
- No configuration required beyond cloudflared authentication

```bash
python deploy_mcp_http.py --tunnel
```

### Persistent Tunnels
- Use named tunnels with custom domains
- Requires additional Cloudflare setup
- Best for production deployments

```yaml
# config/kinfra.yml
tunnel:
  type: "persistent"
  domain: "zen.kooshapari.com"
```

## 🐛 Troubleshooting

### KInfra Not Available
```
❌ KInfra not available: No module named 'aiohttp'
```
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Cloudflared Not Found
```
❌ cloudflared binary not found in PATH
```
**Solution**: Install cloudflared from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

### Authentication Required
```
❌ Tunnel setup failed: Authentication required
```
**Solution**: Authenticate with Cloudflare: `cloudflared tunnel login`

### Port Already in Use
```
⚠️ Port 8080 unavailable, falling back
```
**Solution**: This is normal - KInfra will automatically find an available port

### Tunnel Connection Failed
```
⚠️ Tunnel started but URL not yet available
```
**Solution**: Wait a few seconds for tunnel initialization, or check cloudflared logs

## 🔧 Development

### File Structure
```
zen-mcp-server/
├── server_mcp_http.py              # Enhanced with KInfra integration
├── deploy_mcp_http.py              # Deployment orchestrator
├── test_kinfra_integration.py      # Integration tests
├── config/
│   └── kinfra.yml                  # KInfra configuration
├── KInfra/                         # KInfra libraries (bundled)
│   └── libraries/python/
│       ├── tunnel_manager.py       # Tunnel management
│       └── kinfra_networking.py    # Smart networking
└── requirements.txt                # Updated with KInfra deps
```

### Integration Points

The integration adds the following capabilities to the Zen MCP server:

1. **ZenMCPStreamableServer** - Enhanced constructor with KInfra options
2. **Smart Port Allocation** - `allocate_smart_port()` method
3. **Tunnel Management** - `setup_tunnel()` and `cleanup_tunnel()` methods
4. **Health Monitoring** - Enhanced `/status` and new `/healthz` endpoints
5. **Configuration** - YAML configuration support with environment overrides

### API Changes

The server API is fully backward compatible. New optional parameters:

```python
server = ZenMCPStreamableServer(
    host="0.0.0.0",           # Existing
    port=None,                # Now optional - smart allocation
    enable_tunnel=False,      # NEW: Enable tunnel
    tunnel_domain="zen...",   # NEW: Custom domain
    port_strategy="preferred" # NEW: Port allocation strategy
)
```

## 📚 Examples

### Simple Development Server
```python
from server_mcp_http import ZenMCPStreamableServer

# Basic server with smart port allocation
server = ZenMCPStreamableServer()
server.run()
```

### Production Server with Tunnel
```python
from server_mcp_http import ZenMCPStreamableServer

# Production server with tunnel
server = ZenMCPStreamableServer(
    enable_tunnel=True,
    tunnel_domain="api.zen.kooshapari.com",
    port_strategy="dynamic"
)
server.run()
```

### Using the Deployment Script
```bash
#!/bin/bash
# Production deployment script

export KINFRA_TUNNEL_ENABLED=true
export TUNNEL_DOMAIN=api.zen.kooshapari.com
export LOG_LEVEL=INFO

python deploy_mcp_http.py \\
  --port-strategy dynamic \\
  --tunnel \\
  --log-level INFO
```

## 🤝 Contributing

The KInfra integration is designed to be:
- **Optional**: Server works without KInfra
- **Backward Compatible**: Existing usage unchanged
- **Configurable**: Multiple configuration methods
- **Testable**: Comprehensive test suite

To modify the integration:
1. Update `server_mcp_http.py` for core functionality
2. Update `deploy_mcp_http.py` for deployment logic
3. Update `config/kinfra.yml` for configuration options
4. Run `python test_kinfra_integration.py` to verify changes

## 📝 License

This integration maintains the same license as the Zen MCP Server project.
