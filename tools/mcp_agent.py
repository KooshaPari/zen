"""
MCP Agent Tool - HTTP Remote Agent Execution

This tool enables remote execution of agents via MCP (Model Context Protocol)
Streamable HTTP transport. It allows the Zen MCP Server to orchestrate agents
running on remote servers using the standard MCP protocol.
"""

import json
import logging
from datetime import datetime
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

from .base import BaseTool, ToolError

logger = logging.getLogger(__name__)


class MCPRemoteAgent(BaseModel):
    """Configuration for a remote MCP agent."""
    name: str = Field(description="Agent name/identifier")
    url: str = Field(description="MCP server URL (e.g., http://host:port/mcp)")
    description: str = Field(default="Remote MCP agent", description="Agent description")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    capabilities: dict[str, Any] = Field(default_factory=dict, description="Agent capabilities")
    session_id: Optional[str] = Field(default=None, description="MCP session ID")


class MCPAgentTool(BaseTool):
    """Tool for executing tasks on remote MCP agents via HTTP."""

    name = "mcp_agent"
    description = "Execute tasks on remote MCP agents using Streamable HTTP protocol"

    def __init__(self):
        super().__init__()
        self.remote_agents: dict[str, MCPRemoteAgent] = {}
        self.http_client: Optional[httpx.AsyncClient] = None

    async def get_input_schema(self) -> dict[str, Any]:
        """Get the input schema for the MCP agent tool."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["register", "list", "execute", "info", "remove"],
                    "description": "Action to perform"
                },
                "agent_name": {
                    "type": "string",
                    "description": "Name of the remote agent (for register/execute/info/remove)"
                },
                "agent_url": {
                    "type": "string",
                    "description": "MCP server URL (for register action)"
                },
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool to execute (for execute action)"
                },
                "tool_args": {
                    "type": "object",
                    "description": "Arguments for the tool (for execute action)"
                },
                "resource_uri": {
                    "type": "string",
                    "description": "Resource URI to read (for execute action with read_resource)"
                },
                "prompt_name": {
                    "type": "string",
                    "description": "Prompt name to get (for execute action with get_prompt)"
                },
                "prompt_args": {
                    "type": "object",
                    "description": "Prompt arguments (for execute action with get_prompt)"
                }
            },
            "required": ["action"],
            "additionalProperties": False
        }

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        return self.http_client

    async def _send_mcp_request(
        self,
        agent: MCPRemoteAgent,
        method: str,
        params: Optional[dict[str, Any]] = None,
        notification: bool = False
    ) -> Any:
        """Send JSON-RPC request to MCP server."""
        client = await self._get_http_client()

        # Build JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "method": method
        }

        if params:
            request["params"] = params

        if not notification:
            request["id"] = 1

        # Prepare headers
        headers = {"Content-Type": "application/json"}
        if agent.session_id:
            headers["Mcp-Session-Id"] = agent.session_id

        try:
            response = await client.post(
                agent.url,
                json=request,
                headers=headers,
                timeout=agent.timeout
            )

            # Update session ID if provided
            if "Mcp-Session-Id" in response.headers:
                agent.session_id = response.headers["Mcp-Session-Id"]

            # Parse response
            if response.headers.get("content-type", "").startswith("application/json"):
                data = response.json()
            else:
                data = json.loads(response.text)

            # Check for JSON-RPC error
            if "error" in data:
                error = data["error"]
                raise ToolError(f"MCP error {error.get('code', -1)}: {error.get('message', 'Unknown error')}")

            return data.get("result") if not notification else None

        except httpx.TimeoutException:
            raise ToolError(f"Timeout connecting to MCP agent: {agent.name}")
        except httpx.ConnectError:
            raise ToolError(f"Cannot connect to MCP agent: {agent.name} at {agent.url}")
        except Exception as e:
            raise ToolError(f"MCP request failed for {agent.name}: {str(e)}")

    async def _initialize_agent(self, agent: MCPRemoteAgent) -> bool:
        """Initialize MCP connection with agent."""
        try:
            # Send initialize request
            init_params = {
                "protocolVersion": "2025-03-26",
                "capabilities": {
                    "experimental": {
                        "streaming": True
                    }
                },
                "clientInfo": {
                    "name": "ZenMCP-Orchestrator",
                    "version": "1.0.0"
                }
            }

            result = await self._send_mcp_request(agent, "initialize", init_params)

            # Store agent capabilities
            agent.capabilities = result.get("capabilities", {})

            # Send initialized notification
            await self._send_mcp_request(agent, "notifications/initialized", notification=True)

            logger.info(f"Initialized MCP agent: {agent.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize agent {agent.name}: {e}")
            return False

    async def register_agent(self, name: str, url: str, description: str = None) -> dict[str, Any]:
        """Register a new remote MCP agent."""
        if name in self.remote_agents:
            return {
                "success": False,
                "message": f"Agent {name} already registered",
                "agent": name
            }

        # Create agent configuration
        agent = MCPRemoteAgent(
            name=name,
            url=url,
            description=description or f"Remote MCP agent: {name}"
        )

        # Initialize connection
        if not await self._initialize_agent(agent):
            return {
                "success": False,
                "message": f"Failed to initialize agent {name}",
                "agent": name
            }

        # Store agent
        self.remote_agents[name] = agent

        return {
            "success": True,
            "message": f"Successfully registered agent {name}",
            "agent": name,
            "url": url,
            "capabilities": agent.capabilities
        }

    async def list_agents(self) -> dict[str, Any]:
        """List all registered remote agents."""
        agents = []

        for _name, agent in self.remote_agents.items():
            agents.append({
                "name": agent.name,
                "url": agent.url,
                "description": agent.description,
                "capabilities": list(agent.capabilities.keys()),
                "session_active": agent.session_id is not None
            })

        return {
            "success": True,
            "message": f"Found {len(agents)} registered agents",
            "agents": agents
        }

    async def get_agent_info(self, name: str) -> dict[str, Any]:
        """Get detailed information about an agent."""
        if name not in self.remote_agents:
            return {
                "success": False,
                "message": f"Agent {name} not found",
                "agent": name
            }

        agent = self.remote_agents[name]

        try:
            # Get available tools
            tools_result = await self._send_mcp_request(agent, "tools/list")
            tools = [tool["name"] for tool in tools_result.get("tools", [])]

            # Get available resources
            resources_result = await self._send_mcp_request(agent, "resources/list")
            resources = [res["uri"] for res in resources_result.get("resources", [])]

            # Get available prompts
            prompts_result = await self._send_mcp_request(agent, "prompts/list")
            prompts = [prompt["name"] for prompt in prompts_result.get("prompts", [])]

            return {
                "success": True,
                "agent": name,
                "url": agent.url,
                "description": agent.description,
                "capabilities": agent.capabilities,
                "session_id": agent.session_id,
                "tools": tools,
                "resources": resources,
                "prompts": prompts
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to get info for agent {name}: {str(e)}",
                "agent": name
            }

    async def execute_tool(self, agent_name: str, tool_name: str, tool_args: dict[str, Any] = None) -> dict[str, Any]:
        """Execute a tool on a remote agent."""
        if agent_name not in self.remote_agents:
            return {
                "success": False,
                "message": f"Agent {agent_name} not found",
                "agent": agent_name
            }

        agent = self.remote_agents[agent_name]

        try:
            # Call tool
            params = {"name": tool_name}
            if tool_args:
                params["arguments"] = tool_args

            result = await self._send_mcp_request(agent, "tools/call", params)

            # Extract content
            content = result.get("content", [])
            if isinstance(content, list):
                text_content = []
                for item in content:
                    if item.get("type") == "text":
                        text_content.append(item.get("text", ""))

                if len(text_content) == 1:
                    output = text_content[0]
                elif text_content:
                    output = "\n".join(text_content)
                else:
                    output = str(result)
            else:
                output = str(result)

            return {
                "success": True,
                "agent": agent_name,
                "tool": tool_name,
                "arguments": tool_args or {},
                "result": output,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to execute tool {tool_name} on agent {agent_name}: {str(e)}",
                "agent": agent_name,
                "tool": tool_name
            }

    async def read_resource(self, agent_name: str, resource_uri: str) -> dict[str, Any]:
        """Read a resource from a remote agent."""
        if agent_name not in self.remote_agents:
            return {
                "success": False,
                "message": f"Agent {agent_name} not found",
                "agent": agent_name
            }

        agent = self.remote_agents[agent_name]

        try:
            result = await self._send_mcp_request(agent, "resources/read", {"uri": resource_uri})

            # Extract content
            contents = result.get("contents", [])
            if contents and isinstance(contents, list):
                for content in contents:
                    if content.get("mimeType") in ["text/plain", "application/json", None]:
                        return {
                            "success": True,
                            "agent": agent_name,
                            "resource_uri": resource_uri,
                            "content": content.get("text", ""),
                            "mime_type": content.get("mimeType", "text/plain")
                        }

            return {
                "success": True,
                "agent": agent_name,
                "resource_uri": resource_uri,
                "content": str(result),
                "mime_type": "application/json"
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to read resource {resource_uri} from agent {agent_name}: {str(e)}",
                "agent": agent_name,
                "resource_uri": resource_uri
            }

    async def get_prompt(self, agent_name: str, prompt_name: str, prompt_args: dict[str, Any] = None) -> dict[str, Any]:
        """Get a prompt from a remote agent."""
        if agent_name not in self.remote_agents:
            return {
                "success": False,
                "message": f"Agent {agent_name} not found",
                "agent": agent_name
            }

        agent = self.remote_agents[agent_name]

        try:
            params = {"name": prompt_name}
            if prompt_args:
                params["arguments"] = prompt_args

            result = await self._send_mcp_request(agent, "prompts/get", params)

            # Extract text from messages
            messages = result.get("messages", [])
            if messages:
                text_parts = []
                for message in messages:
                    content = message.get("content")
                    if isinstance(content, dict) and content.get("type") == "text":
                        text_parts.append(content.get("text", ""))
                    elif isinstance(content, str):
                        text_parts.append(content)

                prompt_text = "\n".join(text_parts)
            else:
                prompt_text = result.get("description", str(result))

            return {
                "success": True,
                "agent": agent_name,
                "prompt": prompt_name,
                "arguments": prompt_args or {},
                "content": prompt_text
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to get prompt {prompt_name} from agent {agent_name}: {str(e)}",
                "agent": agent_name,
                "prompt": prompt_name
            }

    async def remove_agent(self, name: str) -> dict[str, Any]:
        """Remove a registered remote agent."""
        if name not in self.remote_agents:
            return {
                "success": False,
                "message": f"Agent {name} not found",
                "agent": name
            }

        # Remove agent
        del self.remote_agents[name]

        return {
            "success": True,
            "message": f"Successfully removed agent {name}",
            "agent": name
        }

    async def execute(
        self,
        action: str,
        agent_name: str = None,
        agent_url: str = None,
        tool_name: str = None,
        tool_args: dict[str, Any] = None,
        resource_uri: str = None,
        prompt_name: str = None,
        prompt_args: dict[str, Any] = None,
        **kwargs
    ) -> dict[str, Any]:
        """Execute MCP agent operations."""

        try:
            if action == "register":
                if not agent_name or not agent_url:
                    raise ToolError("agent_name and agent_url required for register action")
                description = kwargs.get("description", f"Remote MCP agent: {agent_name}")
                return await self.register_agent(agent_name, agent_url, description)

            elif action == "list":
                return await self.list_agents()

            elif action == "info":
                if not agent_name:
                    raise ToolError("agent_name required for info action")
                return await self.get_agent_info(agent_name)

            elif action == "execute":
                if not agent_name:
                    raise ToolError("agent_name required for execute action")

                # Determine execution type
                if tool_name:
                    return await self.execute_tool(agent_name, tool_name, tool_args)
                elif resource_uri:
                    return await self.read_resource(agent_name, resource_uri)
                elif prompt_name:
                    return await self.get_prompt(agent_name, prompt_name, prompt_args)
                else:
                    raise ToolError("One of tool_name, resource_uri, or prompt_name required for execute action")

            elif action == "remove":
                if not agent_name:
                    raise ToolError("agent_name required for remove action")
                return await self.remove_agent(agent_name)

            else:
                raise ToolError(f"Unknown action: {action}")

        except ToolError:
            raise
        except Exception as e:
            logger.error(f"MCP agent tool error: {e}")
            raise ToolError(f"MCP agent operation failed: {str(e)}")

    async def cleanup(self):
        """Clean up HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
