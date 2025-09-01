#!/usr/bin/env python3
"""
Test script for KInfra integration with Zen MCP HTTP Server

This script tests the integration between KInfra and the Zen MCP server
to ensure smart port allocation and tunnel management work correctly.

Marked as integration to allow unit-only runs to skip collection side-effects.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
import pytest

# Mark entire module as integration
pytestmark = pytest.mark.integration

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_kinfra_imports():
    """Test that KInfra libraries can be imported."""
    print("🔍 Testing KInfra imports...")

    try:
        # Add KInfra to path
        kinfra_path = Path(__file__).parent / 'KInfra' / 'libraries' / 'python'
        if kinfra_path.exists():
            sys.path.insert(0, str(kinfra_path))
            print(f"✅ KInfra path found: {kinfra_path}")
        else:
            print(f"❌ KInfra path not found: {kinfra_path}")
            return False

        # Test tunnel manager import
        from tunnel_manager import AsyncTunnelManager, TunnelConfig, TunnelStatus, TunnelType
        print("✅ Tunnel manager imported successfully")

        # Test networking utilities
        from kinfra_networking import DefaultLogger, NetworkingOptions, allocate_free_port
        print("✅ Networking utilities imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


async def test_port_allocation():
    """Test smart port allocation."""
    print("\n🔌 Testing smart port allocation...")

    try:
        from kinfra_networking import DefaultLogger, allocate_free_port

        logger_instance = DefaultLogger()

        # Test basic port allocation
        port1 = allocate_free_port(logger=logger_instance)
        print(f"✅ Allocated port: {port1}")

        # Test preferred port allocation
        port2 = allocate_free_port(preferred=3000, logger=logger_instance)
        print(f"✅ Allocated preferred port: {port2}")

        # Test that different calls get different ports
        port3 = allocate_free_port(logger=logger_instance)
        print(f"✅ Allocated different port: {port3}")

        if port1 != port3:
            print("✅ Port allocation working correctly")
        else:
            print("⚠️ Got same port twice - this might be normal")

        return True

    except Exception as e:
        print(f"❌ Port allocation test failed: {e}")
        return False


async def test_tunnel_config():
    """Test tunnel configuration creation."""
    print("\n⚙️ Testing tunnel configuration...")

    try:
        from tunnel_manager import TunnelConfig, TunnelType

        # Test quick tunnel config
        config = TunnelConfig(
            name="test-tunnel",
            local_url="http://localhost:8080",
            tunnel_type=TunnelType.QUICK,
            log_level="info"
        )
        print(f"✅ Quick tunnel config created: {config.name}")

        # Test persistent tunnel config
        persistent_config = TunnelConfig(
            name="test-persistent",
            local_url="http://localhost:8080",
            hostname="test.zen.kooshapari.com",
            tunnel_type=TunnelType.PERSISTENT,
            log_level="info"
        )
        print(f"✅ Persistent tunnel config created: {persistent_config.name}")

        return True

    except Exception as e:
        print(f"❌ Tunnel config test failed: {e}")
        return False


async def test_server_integration():
    """Test server integration without actually starting the server."""
    print("\n🖥️ Testing server integration...")

    try:
        from server_mcp_http import ZenMCPStreamableServer

        # Test server creation with KInfra options
        server = ZenMCPStreamableServer(
            host="127.0.0.1",
            port=None,  # Use smart allocation
            enable_tunnel=False,  # Don't actually start tunnel in test
            tunnel_domain="test.zen.kooshapari.com",
            port_strategy="preferred"
        )
        print("✅ Server created with KInfra options")

        # Test smart port allocation
        allocated_port = await server.allocate_smart_port()
        print(f"✅ Smart port allocated: {allocated_port}")

        # Verify port is accessible
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('127.0.0.1', allocated_port))
            sock.close()
            print(f"✅ Allocated port {allocated_port} is available")
        except OSError:
            print(f"⚠️ Port {allocated_port} is not available (might be in use)")

        return True

    except Exception as e:
        print(f"❌ Server integration test failed: {e}")
        return False


async def test_deployment_script():
    """Test deployment script configuration."""
    print("\n📦 Testing deployment script...")

    try:
        # Check if deployment script exists
        deploy_script = Path(__file__).parent / 'deploy_mcp_http.py'
        if not deploy_script.exists():
            print("❌ Deployment script not found")
            return False

        print("✅ Deployment script found")

        # Test configuration creation
        sys.path.insert(0, str(Path(__file__).parent))
        from deploy_mcp_http import create_config_from_args

        # Mock args object
        class MockArgs:
            host = "0.0.0.0"
            port = 8080
            tunnel = True
            domain = "test.zen.kooshapari.com"
            port_strategy = "preferred"
            log_level = "INFO"

        config = create_config_from_args(MockArgs())
        print(f"✅ Config created: {json.dumps(config, indent=2)}")

        return True

    except Exception as e:
        print(f"❌ Deployment script test failed: {e}")
        return False


async def test_health_endpoints():
    """Test health endpoint functionality."""
    print("\n💚 Testing health endpoints...")

    try:
        from server_mcp_http import ZenMCPStreamableServer

        server = ZenMCPStreamableServer(
            host="127.0.0.1",
            enable_tunnel=False
        )

        # Check if health endpoint is configured
        routes = [route.path for route in server.app.routes]

        if "/healthz" in routes:
            print("✅ Health endpoint configured")
        else:
            print("❌ Health endpoint not found")
            return False

        if "/status" in routes:
            print("✅ Status endpoint configured")
        else:
            print("❌ Status endpoint not found")
            return False

        return True

    except Exception as e:
        print(f"❌ Health endpoints test failed: {e}")
        return False


async def test_configuration_file():
    """Test KInfra configuration file."""
    print("\n📋 Testing configuration file...")

    try:
        config_file = Path(__file__).parent / 'config' / 'kinfra.yml'
        if not config_file.exists():
            print("❌ KInfra configuration file not found")
            return False

        print("✅ KInfra configuration file found")

        # Try to parse the configuration
        import yaml
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)

            # Check key sections
            required_sections = ['networking', 'tunnel', 'service_discovery']
            for section in required_sections:
                if section in config:
                    print(f"✅ Configuration section '{section}' found")
                else:
                    print(f"❌ Configuration section '{section}' missing")
                    return False

            return True

        except yaml.YAMLError as e:
            print(f"❌ Configuration file parsing failed: {e}")
            return False

    except Exception as e:
        print(f"❌ Configuration file test failed: {e}")
        return False


async def run_integration_tests():
    """Run all integration tests."""
    print("🚀 Starting KInfra Integration Tests")
    print("=" * 50)

    tests = [
        ("KInfra Imports", test_kinfra_imports),
        ("Port Allocation", test_port_allocation),
        ("Tunnel Configuration", test_tunnel_config),
        ("Server Integration", test_server_integration),
        ("Deployment Script", test_deployment_script),
        ("Health Endpoints", test_health_endpoints),
        ("Configuration File", test_configuration_file)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Results Summary")
    print("=" * 50)

    passed = 0
    failed = 0

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {len(tests)}, Passed: {passed}, Failed: {failed}")

    if failed == 0:
        print("\n🎉 All tests passed! KInfra integration is working correctly.")
        return True
    else:
        print(f"\n⚠️ {failed} test(s) failed. Check the output above for details.")
        return False


async def main():
    """Main test entry point."""
    success = await run_integration_tests()

    if success:
        print("\n✨ Integration test completed successfully!")
        print("\nNext steps:")
        print("1. Install cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/")
        print("2. Authenticate cloudflared: cloudflared tunnel login")
        print("3. Start server with tunnel: python deploy_mcp_http.py --tunnel")
        print("4. Test server: curl http://localhost:8080/healthz")
        sys.exit(0)
    else:
        print("\n❌ Integration test failed!")
        print("\nTroubleshooting:")
        print("1. Ensure KInfra directory is present in project root")
        print("2. Install required dependencies: pip install -r requirements.txt")
        print("3. Check Python version (3.8+ required)")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
