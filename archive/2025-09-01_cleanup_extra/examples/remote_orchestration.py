#!/usr/bin/env python3
"""
Remote Agent Orchestration Example

This example demonstrates how to use the mcp_agent tool to register and
orchestrate multiple remote MCP servers, enabling distributed workflows.
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clients.mcp_http_client import MCPStreamableHTTPClient


class RemoteOrchestrationDemo:
    """Demonstrates remote MCP agent orchestration capabilities."""

    def __init__(self, main_server_url: str = "http://localhost:8080/mcp"):
        self.main_server_url = main_server_url
        self.client = None

    async def setup(self):
        """Initialize connection to main MCP server."""
        print("🔌 Connecting to main MCP server...")
        self.client = MCPStreamableHTTPClient(self.main_server_url)
        await self.client.connect()

        server_info = await self.client.get_server_info()
        print(f"✅ Connected to: {server_info.name} v{server_info.version}")

        # Check if mcp_agent tool is available
        tools = await self.client.list_tools()
        agent_tool = next((t for t in tools if t.name == "mcp_agent"), None)

        if not agent_tool:
            raise RuntimeError("mcp_agent tool not available on server")

        print("🛠️ MCP Agent tool is available")

    async def cleanup(self):
        """Clean up connections."""
        if self.client:
            await self.client.disconnect()

    async def demonstrate_agent_registration(self):
        """Show how to register remote MCP agents."""
        print("\n📡 Agent Registration Demo")
        print("=" * 40)

        # List current agents
        print("🔍 Listing current agents...")
        result = await self.client.call_tool("mcp_agent", {"action": "list"})
        print(f"Current agents: {result.get('agents', [])}")

        # Register a hypothetical remote agent
        print("\n📋 Registering remote agents...")

        # Example 1: Register a development server
        try:
            result = await self.client.call_tool("mcp_agent", {
                "action": "register",
                "agent_name": "dev-server",
                "agent_url": "http://dev-host:8080/mcp",
                "description": "Development MCP server for testing"
            })
            print(f"✅ dev-server: {result.get('message', 'Registration status unknown')}")
        except Exception as e:
            print(f"⚠️ dev-server: {e} (expected - demo server not running)")

        # Example 2: Register a production server
        try:
            result = await self.client.call_tool("mcp_agent", {
                "action": "register",
                "agent_name": "prod-server",
                "agent_url": "http://prod-host:8080/mcp",
                "description": "Production MCP server"
            })
            print(f"✅ prod-server: {result.get('message', 'Registration status unknown')}")
        except Exception as e:
            print(f"⚠️ prod-server: {e} (expected - demo server not running)")

        # Example 3: Register a local secondary server
        try:
            result = await self.client.call_tool("mcp_agent", {
                "action": "register",
                "agent_name": "local-secondary",
                "agent_url": "http://localhost:8081/mcp",
                "description": "Local secondary MCP server"
            })
            print(f"✅ local-secondary: {result.get('message', 'Registration status unknown')}")
        except Exception as e:
            print(f"⚠️ local-secondary: {e} (expected - secondary server not running)")

        # Show updated agent list
        print("\n📊 Updated agent list:")
        result = await self.client.call_tool("mcp_agent", {"action": "list"})
        agents = result.get("agents", [])

        if agents:
            for agent in agents:
                print(f"  • {agent.get('name', 'unknown')}: {agent.get('url', 'no-url')}")
                print(f"    Status: {'🟢 Active' if agent.get('session_active') else '🔴 Inactive'}")
        else:
            print("  No agents registered")

    async def demonstrate_agent_info(self):
        """Show how to get detailed agent information."""
        print("\n📋 Agent Information Demo")
        print("=" * 35)

        # List agents first
        result = await self.client.call_tool("mcp_agent", {"action": "list"})
        agents = result.get("agents", [])

        if not agents:
            print("ℹ️ No agents available for info demonstration")
            return

        # Get info for first agent
        agent_name = agents[0].get("name")
        print(f"🔍 Getting detailed info for: {agent_name}")

        try:
            result = await self.client.call_tool("mcp_agent", {
                "action": "info",
                "agent_name": agent_name
            })

            if result.get("success"):
                print(f"  📡 URL: {result.get('url', 'unknown')}")
                print(f"  📝 Description: {result.get('description', 'none')}")
                print(f"  🔧 Tools: {len(result.get('tools', []))}")
                print(f"  📚 Resources: {len(result.get('resources', []))}")
                print(f"  💬 Prompts: {len(result.get('prompts', []))}")

                # List some tools if available
                tools = result.get("tools", [])
                if tools:
                    print("  🛠️ Available tools:")
                    for tool in tools[:5]:  # Show first 5
                        print(f"    • {tool}")
                    if len(tools) > 5:
                        print(f"    ... and {len(tools) - 5} more")
            else:
                print(f"  ❌ Failed to get info: {result.get('message', 'unknown error')}")

        except Exception as e:
            print(f"  ❌ Info request failed: {e}")

    async def demonstrate_tool_execution(self):
        """Show how to execute tools on remote agents."""
        print("\n🚀 Remote Tool Execution Demo")
        print("=" * 40)

        # List agents
        result = await self.client.call_tool("mcp_agent", {"action": "list"})
        agents = result.get("agents", [])

        if not agents:
            print("ℹ️ No agents available for tool execution demo")
            return

        agent_name = agents[0].get("name")
        print(f"🎯 Executing tools on agent: {agent_name}")

        # Example 1: Execute echo tool
        print("\n1️⃣ Testing echo tool...")
        try:
            result = await self.client.call_tool("mcp_agent", {
                "action": "execute",
                "agent_name": agent_name,
                "tool_name": "echo",
                "tool_args": {"text": f"Hello from remote agent {agent_name}!"}
            })

            if result.get("success"):
                print(f"  ✅ Result: {result.get('result', 'no result')}")
            else:
                print(f"  ❌ Failed: {result.get('message', 'unknown error')}")
        except Exception as e:
            print(f"  ❌ Echo execution failed: {e}")

        # Example 2: Execute analysis tool
        print("\n2️⃣ Testing analyze tool...")
        try:
            sample_code = '''
def fibonacci(n):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''

            result = await self.client.call_tool("mcp_agent", {
                "action": "execute",
                "agent_name": agent_name,
                "tool_name": "analyze",
                "tool_args": {"code": sample_code}
            })

            if result.get("success"):
                analysis = result.get('result', 'no result')
                preview = analysis[:200] + "..." if len(analysis) > 200 else analysis
                print(f"  ✅ Analysis preview: {preview}")
            else:
                print(f"  ❌ Failed: {result.get('message', 'unknown error')}")
        except Exception as e:
            print(f"  ❌ Analysis execution failed: {e}")

        # Example 3: Execute time tool
        print("\n3️⃣ Testing get_time tool...")
        try:
            result = await self.client.call_tool("mcp_agent", {
                "action": "execute",
                "agent_name": agent_name,
                "tool_name": "get_time"
            })

            if result.get("success"):
                print(f"  ✅ Remote time: {result.get('result', 'no result')}")
            else:
                print(f"  ❌ Failed: {result.get('message', 'unknown error')}")
        except Exception as e:
            print(f"  ❌ Time execution failed: {e}")

    async def demonstrate_resource_access(self):
        """Show how to access resources on remote agents."""
        print("\n📚 Remote Resource Access Demo")
        print("=" * 40)

        # List agents
        result = await self.client.call_tool("mcp_agent", {"action": "list"})
        agents = result.get("agents", [])

        if not agents:
            print("ℹ️ No agents available for resource access demo")
            return

        agent_name = agents[0].get("name")
        print(f"📖 Accessing resources on agent: {agent_name}")

        try:
            # Try to read a common resource
            result = await self.client.call_tool("mcp_agent", {
                "action": "execute",
                "agent_name": agent_name,
                "resource_uri": "file://README.md"
            })

            if result.get("success"):
                content = result.get("content", "no content")
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"  ✅ README.md preview: {preview}")
            else:
                print(f"  ❌ Failed: {result.get('message', 'unknown error')}")

        except Exception as e:
            print(f"  ❌ Resource access failed: {e}")

    async def demonstrate_workflow_coordination(self):
        """Show coordinated workflow across multiple agents."""
        print("\n🔄 Multi-Agent Workflow Demo")
        print("=" * 40)

        # List available agents
        result = await self.client.call_tool("mcp_agent", {"action": "list"})
        agents = result.get("agents", [])

        if len(agents) < 1:
            print("ℹ️ Need at least 1 agent for workflow demo")
            return

        print(f"🎭 Coordinating workflow across {len(agents)} agent(s)")

        # Workflow: Analyze code on one agent, then review on another
        sample_code = '''
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
'''

        # Step 1: Analyze code
        agent1 = agents[0].get("name")
        print(f"\n1️⃣ Step 1: Analyzing code on {agent1}...")

        try:
            analysis_result = await self.client.call_tool("mcp_agent", {
                "action": "execute",
                "agent_name": agent1,
                "tool_name": "analyze",
                "tool_args": {"code": sample_code}
            })

            if analysis_result.get("success"):
                analysis = analysis_result.get("result", "")
                print(f"  ✅ Analysis completed ({len(analysis)} chars)")
            else:
                print(f"  ❌ Analysis failed: {analysis_result.get('message')}")
                analysis = "Analysis unavailable"

        except Exception as e:
            print(f"  ❌ Analysis step failed: {e}")
            analysis = "Analysis failed"

        # Step 2: Code review (use same agent or different if available)
        agent2 = agents[1].get("name") if len(agents) > 1 else agent1
        print(f"\n2️⃣ Step 2: Code review on {agent2}...")

        try:
            review_result = await self.client.call_tool("mcp_agent", {
                "action": "execute",
                "agent_name": agent2,
                "tool_name": "codereview",
                "tool_args": {
                    "code": sample_code,
                    "focus": "performance"
                }
            })

            if review_result.get("success"):
                review = review_result.get("result", "")
                print(f"  ✅ Code review completed ({len(review)} chars)")
            else:
                print(f"  ❌ Review failed: {review_result.get('message')}")

        except Exception as e:
            print(f"  ❌ Review step failed: {e}")

        print("\n🎯 Workflow coordination completed!")

    async def demonstrate_agent_management(self):
        """Show agent management operations."""
        print("\n🔧 Agent Management Demo")
        print("=" * 35)

        # Create a test agent registration
        test_agent_name = "test-agent-demo"

        print(f"➕ Registering temporary agent: {test_agent_name}")
        try:
            result = await self.client.call_tool("mcp_agent", {
                "action": "register",
                "agent_name": test_agent_name,
                "agent_url": "http://test-host:8080/mcp",
                "description": "Temporary test agent for demo"
            })
            print(f"  Status: {result.get('message', 'unknown')}")
        except Exception as e:
            print(f"  ❌ Registration failed: {e}")

        # List to confirm
        result = await self.client.call_tool("mcp_agent", {"action": "list"})
        agents = result.get("agents", [])
        test_agent_exists = any(a.get("name") == test_agent_name for a in agents)

        if test_agent_exists:
            print(f"  ✅ {test_agent_name} found in agent list")

            # Remove the test agent
            print(f"\n➖ Removing test agent: {test_agent_name}")
            try:
                result = await self.client.call_tool("mcp_agent", {
                    "action": "remove",
                    "agent_name": test_agent_name
                })
                print(f"  Status: {result.get('message', 'unknown')}")
            except Exception as e:
                print(f"  ❌ Removal failed: {e}")
        else:
            print(f"  ⚠️ {test_agent_name} not found in agent list")


async def main():
    """Run the complete remote orchestration demonstration."""

    print("🧘 Zen MCP Remote Agent Orchestration Demo")
    print("=" * 60)
    print("This demo shows how to orchestrate multiple MCP servers")
    print("using the mcp_agent tool for distributed workflows.")
    print()

    demo = RemoteOrchestrationDemo()

    try:
        # Setup connection
        await demo.setup()

        # Run all demonstrations
        await demo.demonstrate_agent_registration()
        await demo.demonstrate_agent_info()
        await demo.demonstrate_tool_execution()
        await demo.demonstrate_resource_access()
        await demo.demonstrate_workflow_coordination()
        await demo.demonstrate_agent_management()

        print("\n🎉 Remote orchestration demo completed!")
        print("\n💡 Next steps:")
        print("  • Start additional MCP servers on different ports")
        print("  • Register them as agents using the examples above")
        print("  • Build complex multi-server workflows")
        print("  • Monitor and manage your distributed MCP ecosystem")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Ensure main MCP server is running: python server_mcp_http.py")
        print("  2. Check server has mcp_agent tool available")
        print("  3. For full demo, start additional servers on different ports")

    finally:
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
