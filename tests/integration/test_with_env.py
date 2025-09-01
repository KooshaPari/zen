#!/usr/bin/env python3
"""
Test agent adapters with proper environment loaded
"""

import os
import sys
from pathlib import Path

# Load environment from .env file
from dotenv import load_dotenv

load_dotenv()

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from tools.shared.agent_models import AgentTaskRequest, AgentType
from utils.agent_adapters import run_adapter


def test_claude_with_env():
    """Test Claude adapter with proper environment and longer timeout"""
    print("=== Testing Claude Adapter with Environment ===")

    # Check if API key is loaded
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    print(f"OPENROUTER_API_KEY: {'[SET]' if openrouter_key else '[NOT SET]'}")

    request = AgentTaskRequest(
        agent_type=AgentType.CLAUDE,
        task_description="Simple Flask API test",
        message="Create a simple Flask app with one endpoint: GET /hello that returns {'message': 'Hello World'}. Save it as app.py",
        timeout_seconds=60,  # Longer timeout
        working_directory="/tmp/test_agent"
    )

    print("\nTesting request:")
    print(f"  Agent: {request.agent_type.value}")
    print(f"  Message: {request.message}")
    print(f"  Timeout: {request.timeout_seconds}s")
    print(f"  Working dir: {request.working_directory}")

    try:
        print("\nRunning adapter...")
        stdout, exit_code, stderr = run_adapter(request)

        print("\n=== Results ===")
        print(f"Exit code: {exit_code}")
        print(f"Stdout ({len(stdout)} chars):")
        print(stdout)
        print(f"\nStderr ({len(stderr)} chars):")
        print(stderr)

        if exit_code == 0:
            print("\n✅ SUCCESS - Claude task completed!")
        else:
            print(f"\n❌ FAILED - Exit code {exit_code}")

    except Exception as e:
        print(f"\n❌ Exception: {e}")


def test_aider_with_env():
    """Test Aider adapter with proper environment"""
    print("\n\n=== Testing Aider Adapter with Environment ===")

    request = AgentTaskRequest(
        agent_type=AgentType.AIDER,
        task_description="JavaScript API client",
        message="Create a JavaScript file that exports functions to interact with a REST API. Include fetchUsers(), createUser(userData), getUserById(id), updateUser(id, userData), deleteUser(id). Use modern fetch with async/await. Save as api_client.js",
        timeout_seconds=60,
        working_directory="/tmp/test_agent"
    )

    print("Testing request:")
    print(f"  Agent: {request.agent_type.value}")
    print(f"  Timeout: {request.timeout_seconds}s")

    try:
        print("Running adapter...")
        stdout, exit_code, stderr = run_adapter(request)

        print("\n=== Results ===")
        print(f"Exit code: {exit_code}")
        print(f"Stdout ({len(stdout)} chars):")
        print(stdout[:1000] + "..." if len(stdout) > 1000 else stdout)
        print(f"\nStderr ({len(stderr)} chars):")
        print(stderr[:500] + "..." if len(stderr) > 500 else stderr)

        if exit_code == 0:
            print("\n✅ SUCCESS - Aider task completed!")
        else:
            print(f"\n❌ FAILED - Exit code {exit_code}")

    except Exception as e:
        print(f"\n❌ Exception: {e}")


if __name__ == "__main__":
    test_claude_with_env()
    test_aider_with_env()
