#!/usr/bin/env python3
"""
MCP Streamable HTTP Client

This module provides a Python client for connecting to MCP servers using the
Streamable HTTP transport protocol (MCP spec 2025-03-26). It supports remote
connections with no authentication requirements.

Features:
- Streamable HTTP transport client
- Automatic session management
- JSON-RPC 2.0 protocol support
- Tool execution and resource access
- Streaming support for real-time updates
- Error handling and retry logic
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class MCPServerInfo:
    """Information about an MCP server."""
    name: str
    version: str
    protocol_version: str
    capabilities: dict[str, Any]
    url: str
    session_id: Optional[str] = None


@dataclass
class MCPTool:
    """MCP tool definition."""
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class MCPResource:
    """MCP resource definition."""
    uri: str
    name: str
    description: str
    mime_type: Optional[str] = None


@dataclass
class MCPPrompt:
    """MCP prompt definition."""
    name: str
    description: str
    arguments: list[dict[str, Any]]


class MCPStreamableHTTPClient:
    """Client for MCP Streamable HTTP protocol."""

    def __init__(self, server_url: str, timeout: int = 30):
        """
        Initialize MCP client.

        Args:
            server_url: URL of the MCP server (e.g., "http://localhost:8080/mcp")
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self.server_info: Optional[MCPServerInfo] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._request_id = 0

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self):
        """Connect to the MCP server and initialize session."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)

        try:
            # Initialize MCP session
            await self._initialize()
            logger.info(f"Connected to MCP server: {self.server_url}")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            if self.session:
                await self.session.close()
                self.session = None
            raise

    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.session:
            await self.session.close()
            self.session = None

        self.session_id = None
        self.server_info = None
        logger.info("Disconnected from MCP server")

    def _get_request_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id

    async def _send_request(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
        notification: bool = False
    ) -> Optional[dict[str, Any]]:
        """
        Send JSON-RPC request to MCP server.

        Args:
            method: MCP method name
            params: Method parameters
            notification: Whether this is a notification (no response expected)

        Returns:
            Response data if not a notification
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        # Build JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "method": method
        }

        if params:
            request["params"] = params

        if not notification:
            request["id"] = self._get_request_id()

        # Prepare headers
        headers = {"Content-Type": "application/json"}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        try:
            # Send request
            async with self.session.post(
                self.server_url,
                json=request,
                headers=headers
            ) as response:

                # Handle session ID from response headers
                if "Mcp-Session-Id" in response.headers:
                    self.session_id = response.headers["Mcp-Session-Id"]

                # Parse response
                if response.content_type == "application/json":
                    data = await response.json()
                else:
                    text = await response.text()
                    data = json.loads(text)

                # Check for JSON-RPC error
                if "error" in data:
                    error = data["error"]
                    raise Exception(f"MCP error {error.get('code', -1)}: {error.get('message', 'Unknown error')}")

                return data.get("result") if not notification else None

        except Exception as e:
            logger.error(f"MCP request failed for {method}: {e}")
            raise

    async def _initialize(self):
        """Initialize MCP connection."""
        init_params = {
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "experimental": {
                    "streaming": True
                }
            },
            "clientInfo": {
                "name": "ZenMCP-HTTPClient",
                "version": "1.0.0"
            }
        }

        result = await self._send_request("initialize", init_params)

        # Store server information
        self.server_info = MCPServerInfo(
            name=result.get("serverInfo", {}).get("name", "Unknown"),
            version=result.get("serverInfo", {}).get("version", "Unknown"),
            protocol_version=result.get("protocolVersion", "Unknown"),
            capabilities=result.get("capabilities", {}),
            url=self.server_url,
            session_id=self.session_id
        )

        # Send initialized notification
        await self._send_request("notifications/initialized", notification=True)

    async def get_server_info(self) -> MCPServerInfo:
        """Get server information."""
        if not self.server_info:
            raise RuntimeError("Not connected to server")
        return self.server_info

    async def list_tools(self) -> list[MCPTool]:
        """List available tools."""
        result = await self._send_request("tools/list")

        tools = []
        for tool_data in result.get("tools", []):
            tools.append(MCPTool(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {})
            ))

        return tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] = None) -> Any:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result
        """
        params = {"name": tool_name}
        if arguments:
            params["arguments"] = arguments

        result = await self._send_request("tools/call", params)

        # Extract text content from result
        content = result.get("content", [])
        if content and isinstance(content, list):
            text_content = []
            for item in content:
                if item.get("type") == "text":
                    text_content.append(item.get("text", ""))

            if len(text_content) == 1:
                return text_content[0]
            elif text_content:
                return "\n".join(text_content)

        return result

    async def list_resources(self) -> list[MCPResource]:
        """List available resources."""
        result = await self._send_request("resources/list")

        resources = []
        for resource_data in result.get("resources", []):
            resources.append(MCPResource(
                uri=resource_data["uri"],
                name=resource_data.get("name", ""),
                description=resource_data.get("description", ""),
                mime_type=resource_data.get("mimeType")
            ))

        return resources

    async def read_resource(self, uri: str) -> str:
        """
        Read a resource from the MCP server.

        Args:
            uri: Resource URI to read

        Returns:
            Resource content as string
        """
        result = await self._send_request("resources/read", {"uri": uri})

        contents = result.get("contents", [])
        if contents and isinstance(contents, list):
            # Return first text content
            for content in contents:
                if content.get("mimeType") in ["text/plain", "application/json", None]:
                    return content.get("text", "")

        return str(result)

    async def list_prompts(self) -> list[MCPPrompt]:
        """List available prompts."""
        result = await self._send_request("prompts/list")

        prompts = []
        for prompt_data in result.get("prompts", []):
            prompts.append(MCPPrompt(
                name=prompt_data["name"],
                description=prompt_data.get("description", ""),
                arguments=prompt_data.get("arguments", [])
            ))

        return prompts

    async def get_prompt(self, prompt_name: str, arguments: dict[str, Any] = None) -> str:
        """
        Get a prompt from the MCP server.

        Args:
            prompt_name: Name of the prompt
            arguments: Prompt arguments

        Returns:
            Prompt content as string
        """
        params = {"name": prompt_name}
        if arguments:
            params["arguments"] = arguments

        result = await self._send_request("prompts/get", params)

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

            return "\n".join(text_parts)

        return result.get("description", str(result))

    async def set_logging_level(self, level: str):
        """Set server logging level."""
        await self._send_request("logging/setLevel", {"level": level})

    async def test_connection(self) -> bool:
        """Test if connection to server is working."""
        try:
            if not self.session:
                return False

            async with self.session.get(self.server_url) as response:
                return response.status == 200
        except Exception:
            return False


class MCPClientManager:
    """Manages multiple MCP client connections."""

    def __init__(self):
        self.clients: dict[str, MCPStreamableHTTPClient] = {}

    async def add_server(self, name: str, server_url: str, timeout: int = 30) -> MCPStreamableHTTPClient:
        """
        Add and connect to an MCP server.

        Args:
            name: Friendly name for the server
            server_url: Server URL
            timeout: Connection timeout

        Returns:
            Connected MCP client
        """
        if name in self.clients:
            await self.remove_server(name)

        client = MCPStreamableHTTPClient(server_url, timeout)
        await client.connect()

        self.clients[name] = client
        logger.info(f"Added MCP server: {name} -> {server_url}")

        return client

    async def remove_server(self, name: str):
        """Remove and disconnect from an MCP server."""
        if name in self.clients:
            await self.clients[name].disconnect()
            del self.clients[name]
            logger.info(f"Removed MCP server: {name}")

    async def get_client(self, name: str) -> Optional[MCPStreamableHTTPClient]:
        """Get MCP client by name."""
        return self.clients.get(name)

    async def list_servers(self) -> list[tuple[str, MCPServerInfo]]:
        """List all connected servers."""
        servers = []
        for name, client in self.clients.items():
            if client.server_info:
                servers.append((name, client.server_info))
        return servers

    async def disconnect_all(self):
        """Disconnect from all servers."""
        for client in self.clients.values():
            await client.disconnect()
        self.clients.clear()
        logger.info("Disconnected from all MCP servers")


async def demo_mcp_client():
    """Demonstrate MCP client functionality."""

    # Test connection to local Zen MCP server
    server_url = "http://localhost:8080/mcp"

    print(f"🔌 Connecting to MCP server: {server_url}")

    try:
        async with MCPStreamableHTTPClient(server_url) as client:
            # Get server info
            server_info = await client.get_server_info()
            print(f"✅ Connected to: {server_info.name} v{server_info.version}")
            print(f"📡 Protocol version: {server_info.protocol_version}")
            print(f"🔧 Capabilities: {list(server_info.capabilities.keys())}")

            # List and call tools
            print("\n🛠️ Available Tools:")
            tools = await client.list_tools()
            for tool in tools[:5]:  # Show first 5 tools
                print(f"  • {tool.name}: {tool.description}")

            # Test some tools
            print("\n🔄 Testing Tools:")

            # Echo tool
            try:
                result = await client.call_tool("echo", {"text": "Hello from MCP client!"})
                print(f"  📢 echo: {result}")
            except Exception as e:
                print(f"  ❌ echo failed: {e}")

            # Get time tool
            try:
                result = await client.call_tool("get_time")
                print(f"  🕐 get_time: {result}")
            except Exception as e:
                print(f"  ❌ get_time failed: {e}")

            # Multiply tool
            try:
                result = await client.call_tool("multiply", {"a": 7, "b": 8})
                print(f"  🔢 multiply(7, 8): {result}")
            except Exception as e:
                print(f"  ❌ multiply failed: {e}")

            # List and read resources
            print("\n📚 Available Resources:")
            resources = await client.list_resources()
            for resource in resources:
                print(f"  • {resource.name}: {resource.uri}")

            if resources:
                print("\n📖 Reading Resource:")
                try:
                    content = await client.read_resource(resources[0].uri)
                    print(f"  📄 {resources[0].name}:\n{content[:200]}...")
                except Exception as e:
                    print(f"  ❌ Failed to read resource: {e}")

            # List and get prompts
            print("\n💬 Available Prompts:")
            prompts = await client.list_prompts()
            for prompt in prompts:
                print(f"  • {prompt.name}: {prompt.description}")

            if prompts:
                print("\n🎯 Getting Prompt:")
                try:
                    content = await client.get_prompt(prompts[0].name, {"topic": "tools"})
                    print(f"  📝 {prompts[0].name}:\n{content[:200]}...")
                except Exception as e:
                    print(f"  ❌ Failed to get prompt: {e}")

            print("\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("💡 Make sure the MCP server is running: python server_mcp_http.py")


async def interactive_mcp_client():
    """Interactive MCP client shell."""
    print("🧘 Zen MCP Interactive Client")
    print("="*40)

    server_url = input("MCP Server URL (default: http://localhost:8080/mcp): ").strip()
    if not server_url:
        server_url = "http://localhost:8080/mcp"

    try:
        async with MCPStreamableHTTPClient(server_url) as client:
            server_info = await client.get_server_info()
            print(f"✅ Connected to: {server_info.name} v{server_info.version}")

            # Load available tools and resources
            tools = await client.list_tools()
            resources = await client.list_resources()
            prompts = await client.list_prompts()

            print("\n📊 Server Capabilities:")
            print(f"  • Tools: {len(tools)}")
            print(f"  • Resources: {len(resources)}")
            print(f"  • Prompts: {len(prompts)}")

            print("\n💡 Commands:")
            print("  • 'tools' - List available tools")
            print("  • 'call <tool_name> [args...]' - Call a tool")
            print("  • 'resources' - List available resources")
            print("  • 'read <uri>' - Read a resource")
            print("  • 'prompts' - List available prompts")
            print("  • 'prompt <name> [args...]' - Get a prompt")
            print("  • 'exit' - Exit client")

            while True:
                try:
                    command = input("\nmcp> ").strip()

                    if command == "exit":
                        break
                    elif command == "tools":
                        for tool in tools:
                            print(f"  • {tool.name}: {tool.description}")
                    elif command.startswith("call "):
                        parts = command[5:].split()
                        if parts:
                            tool_name = parts[0]
                            # Simple argument parsing (key=value)
                            args = {}
                            for arg in parts[1:]:
                                if "=" in arg:
                                    key, value = arg.split("=", 1)
                                    # Try to parse as number
                                    try:
                                        if "." in value:
                                            args[key] = float(value)
                                        else:
                                            args[key] = int(value)
                                    except ValueError:
                                        args[key] = value

                            try:
                                result = await client.call_tool(tool_name, args if args else None)
                                print(f"Result: {result}")
                            except Exception as e:
                                print(f"Error: {e}")
                        else:
                            print("Usage: call <tool_name> [key=value ...]")
                    elif command == "resources":
                        for resource in resources:
                            print(f"  • {resource.name}: {resource.uri}")
                    elif command.startswith("read "):
                        uri = command[5:].strip()
                        try:
                            content = await client.read_resource(uri)
                            print(f"Content:\n{content}")
                        except Exception as e:
                            print(f"Error: {e}")
                    elif command == "prompts":
                        for prompt in prompts:
                            print(f"  • {prompt.name}: {prompt.description}")
                    elif command.startswith("prompt "):
                        parts = command[7:].split()
                        if parts:
                            prompt_name = parts[0]
                            # Simple argument parsing
                            args = {}
                            for arg in parts[1:]:
                                if "=" in arg:
                                    key, value = arg.split("=", 1)
                                    args[key] = value

                            try:
                                content = await client.get_prompt(prompt_name, args if args else None)
                                print(f"Prompt:\n{content}")
                            except Exception as e:
                                print(f"Error: {e}")
                        else:
                            print("Usage: prompt <name> [key=value ...]")
                    elif command:
                        print(f"Unknown command: {command}")

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")

            print("\n👋 Goodbye!")

    except Exception as e:
        print(f"❌ Connection failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCP Streamable HTTP Client")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--server", default="http://localhost:8080/mcp", help="Server URL")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    if args.demo:
        asyncio.run(demo_mcp_client())
    elif args.interactive:
        asyncio.run(interactive_mcp_client())
    else:
        print("Use --demo or --interactive to run the client")
        parser.print_help()
