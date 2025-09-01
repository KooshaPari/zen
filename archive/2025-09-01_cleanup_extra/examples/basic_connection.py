#!/usr/bin/env python3
"""
Basic MCP Streamable HTTP Connection Example

This example demonstrates basic connectivity to a Zen MCP server using
the Streamable HTTP transport protocol with no authentication.
"""

import asyncio
import os
import sys

# Add parent directory to path to import the client
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clients.mcp_http_client import MCPStreamableHTTPClient


async def basic_connection_example():
    """Demonstrate basic MCP connection and tool execution."""

    print("🧘 Zen MCP Basic Connection Example")
    print("=" * 50)

    server_url = "http://localhost:8080/mcp"
    print(f"🔌 Connecting to: {server_url}")

    try:
        # Connect using async context manager
        async with MCPStreamableHTTPClient(server_url, timeout=10) as client:

            # 1. Get server information
            print("\n📋 Server Information:")
            server_info = await client.get_server_info()
            print(f"  Name: {server_info.name}")
            print(f"  Version: {server_info.version}")
            print(f"  Protocol: {server_info.protocol_version}")
            print(f"  Session: {server_info.session_id}")
            print(f"  Capabilities: {list(server_info.capabilities.keys())}")

            # 2. List available tools
            print("\n🛠️ Available Tools:")
            tools = await client.list_tools()
            print(f"  Total tools: {len(tools)}")

            for i, tool in enumerate(tools[:5], 1):  # Show first 5
                print(f"  {i}. {tool.name}: {tool.description}")

            if len(tools) > 5:
                print(f"  ... and {len(tools) - 5} more tools")

            # 3. Test basic tools
            print("\n🔄 Testing Core Tools:")

            # Test echo tool
            try:
                result = await client.call_tool("echo", {"text": "Hello MCP World!"})
                print(f"  ✅ echo: '{result}'")
            except Exception as e:
                print(f"  ❌ echo failed: {e}")

            # Test get_time tool
            try:
                result = await client.call_tool("get_time")
                print(f"  ✅ get_time: {result}")
            except Exception as e:
                print(f"  ❌ get_time failed: {e}")

            # Test multiply tool
            try:
                a, b = 42, 13
                result = await client.call_tool("multiply", {"a": a, "b": b})
                print(f"  ✅ multiply({a}, {b}): {result}")
            except Exception as e:
                print(f"  ❌ multiply failed: {e}")

            # 4. Test resources if available
            print("\n📚 Testing Resources:")
            resources = await client.list_resources()

            if resources:
                print(f"  Found {len(resources)} resources:")
                for resource in resources[:3]:  # Test first 3
                    print(f"    • {resource.name} ({resource.uri})")
                    try:
                        content = await client.read_resource(resource.uri)
                        preview = content[:100] + "..." if len(content) > 100 else content
                        print(f"      Content preview: {preview}")
                    except Exception as e:
                        print(f"      ❌ Failed to read: {e}")
            else:
                print("  No resources available")

            # 5. Test prompts if available
            print("\n💬 Testing Prompts:")
            prompts = await client.list_prompts()

            if prompts:
                print(f"  Found {len(prompts)} prompts:")
                for prompt in prompts[:2]:  # Test first 2
                    print(f"    • {prompt.name}: {prompt.description}")
                    try:
                        content = await client.get_prompt(prompt.name, {"topic": "example"})
                        preview = content[:100] + "..." if len(content) > 100 else content
                        print(f"      Content preview: {preview}")
                    except Exception as e:
                        print(f"      ❌ Failed to get: {e}")
            else:
                print("  No prompts available")

            # 6. Test connection health
            print("\n❤️ Connection Health Check:")
            is_healthy = await client.test_connection()
            print(f"  Connection status: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")

            print("\n🎉 Basic connection test completed successfully!")

    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print("\n💡 Troubleshooting steps:")
        print("  1. Ensure the MCP server is running:")
        print("     python server_mcp_http.py")
        print("  2. Check the server URL is correct")
        print("  3. Verify no firewall blocking the connection")
        return False

    return True


async def test_error_handling():
    """Test error handling scenarios."""

    print("\n🧪 Testing Error Handling:")
    print("-" * 30)

    server_url = "http://localhost:8080/mcp"

    try:
        async with MCPStreamableHTTPClient(server_url) as client:
            # Test invalid tool
            try:
                await client.call_tool("nonexistent_tool")
                print("  ❌ Expected error for invalid tool")
            except Exception as e:
                print(f"  ✅ Correctly handled invalid tool: {type(e).__name__}")

            # Test invalid tool arguments
            try:
                await client.call_tool("multiply", {"invalid": "args"})
                print("  ❌ Expected error for invalid arguments")
            except Exception as e:
                print(f"  ✅ Correctly handled invalid args: {type(e).__name__}")

            # Test invalid resource
            try:
                await client.read_resource("invalid://resource")
                print("  ❌ Expected error for invalid resource")
            except Exception as e:
                print(f"  ✅ Correctly handled invalid resource: {type(e).__name__}")

    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        return False

    return True


async def test_concurrent_requests():
    """Test concurrent request handling."""

    print("\n🚀 Testing Concurrent Requests:")
    print("-" * 35)

    server_url = "http://localhost:8080/mcp"

    try:
        async with MCPStreamableHTTPClient(server_url) as client:
            # Create multiple concurrent requests
            tasks = []
            for i in range(5):
                task = client.call_tool("echo", {"text": f"Concurrent request {i+1}"})
                tasks.append(task)

            # Wait for all requests to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_count = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"  ❌ Request {i+1} failed: {result}")
                else:
                    print(f"  ✅ Request {i+1}: {result}")
                    success_count += 1

            print(f"  📊 Success rate: {success_count}/{len(tasks)}")

    except Exception as e:
        print(f"  ❌ Concurrent test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    async def main():
        print("Starting MCP Streamable HTTP connection tests...")

        # Run basic connection test
        basic_success = await basic_connection_example()

        if basic_success:
            # Run additional tests
            await test_error_handling()
            await test_concurrent_requests()

            print("\n✨ All tests completed!")
        else:
            print("\n❌ Basic connection failed - skipping additional tests")
            sys.exit(1)

    asyncio.run(main())
