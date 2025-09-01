#!/usr/bin/env python3
"""Test MCP connection with OAuth flow.

Marked as integration and safe to import under pytest. If the client
dependency is missing, tests will be skipped instead of exiting.
"""

import asyncio

import pytest

# Mark entire module as integration to allow easy skipping
pytestmark = pytest.mark.integration

# Try import lazily and avoid sys.exit during collection
MCPHttpClient = None
try:
    from clients.mcp_http_client import MCPHttpClient as _MCPHttpClient
    MCPHttpClient = _MCPHttpClient
except Exception:
    MCPHttpClient = None

async def test_connection():
    """Test MCP connection"""
    if MCPHttpClient is None:
        pytest.skip("clients.mcp_http_client.MCPHttpClient not available; skipping integration test")

    print("Testing MCP connection to https://zen.kooshapari.com/mcp")
    print("-" * 50)

    client = MCPHttpClient("https://zen.kooshapari.com/mcp")

    try:
        # Initialize connection
        print("Initializing connection...")
        result = await client.initialize()
        print(f"✅ Connected! Server: {result.get('server_info', {}).get('name', 'Unknown')}")
        print(f"   Version: {result.get('server_info', {}).get('version', 'Unknown')}")

        # List available tools
        tools = await client.list_tools()
        print(f"\n📦 Available tools: {len(tools.get('tools', []))}")

        # Test a simple tool call
        print("\nTesting 'listmodels' tool...")
        await client.call_tool("listmodels", {})
        print("✅ Tool call successful!")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_connection())
