#!/usr/bin/env python3
"""
Debug script to test agent adapters directly
"""

import os
import subprocess
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from tools.shared.agent_models import AgentTaskRequest, AgentType
from utils.agent_adapters import _get_agent_command, run_adapter


def test_command_availability():
    """Test if agent commands are available"""
    print("=== Testing Command Availability ===")

    for agent_type in [AgentType.CLAUDE, AgentType.AIDER, AgentType.GOOSE]:
        cmd = _get_agent_command(agent_type)
        print(f"\n{agent_type.value}: {cmd}")

        # Test if command exists
        try:
            result = subprocess.run(['which', cmd], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"  ✓ Found at: {result.stdout.strip()}")

                # Test version/help
                try:
                    version_result = subprocess.run([cmd, '--version'], capture_output=True, text=True, timeout=5)
                    if version_result.returncode == 0:
                        print(f"  ✓ Version: {version_result.stdout.strip()}")
                    else:
                        print(f"  ⚠ Version failed: {version_result.stderr.strip()}")
                except subprocess.TimeoutExpired:
                    print("  ⚠ Version command timed out")
                except Exception as e:
                    print(f"  ⚠ Version error: {e}")

            else:
                print("  ❌ Not found")
        except Exception as e:
            print(f"  ❌ Error: {e}")


def test_simple_adapter():
    """Test adapter with a simple request"""
    print("\n\n=== Testing Simple Adapter ===")

    # Test with a very simple request
    request = AgentTaskRequest(
        agent_type=AgentType.CLAUDE,
        task_description="Simple test",
        message="Say hello",
        timeout_seconds=10,  # Very short timeout
        working_directory="/tmp"
    )

    print(f"Testing request: {request.agent_type.value}")
    print(f"Message: {request.message}")
    print(f"Timeout: {request.timeout_seconds}s")

    try:
        stdout, exit_code, stderr = run_adapter(request)
        print("\nResults:")
        print(f"  Exit code: {exit_code}")
        print(f"  Stdout ({len(stdout)} chars): {stdout[:200]}{'...' if len(stdout) > 200 else ''}")
        print(f"  Stderr ({len(stderr)} chars): {stderr[:200]}{'...' if len(stderr) > 200 else ''}")

    except Exception as e:
        print(f"\nAdapter error: {e}")


def test_manual_command():
    """Test running the command manually to see what happens"""
    print("\n\n=== Testing Manual Command Execution ===")

    cmd = ['claude', '--allowedTools', 'Bash(git*) Edit Replace']
    message = "Create a simple hello.py file with print('Hello World')"

    print(f"Command: {' '.join(cmd)}")
    print(f"Message: {message}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/tmp"
        )

        print("Process started, communicating with 10s timeout...")
        stdout, stderr = proc.communicate(input=message, timeout=10)

        print("\nManual Results:")
        print(f"  Return code: {proc.returncode}")
        print(f"  Stdout: {stdout[:300]}{'...' if len(stdout) > 300 else ''}")
        print(f"  Stderr: {stderr[:300]}{'...' if len(stderr) > 300 else ''}")

    except subprocess.TimeoutExpired:
        print("⚠ Process timed out - killing...")
        proc.kill()
        stdout, stderr = proc.communicate()
        print(f"  After kill - Stdout: {stdout[:200]}{'...' if len(stdout) > 200 else ''}")
        print(f"  After kill - Stderr: {stderr[:200]}{'...' if len(stderr) > 200 else ''}")
    except Exception as e:
        print(f"Manual command error: {e}")


def test_environment():
    """Test environment variables that might be needed"""
    print("\n\n=== Testing Environment ===")

    # Check for common API keys and settings
    env_vars = [
        'ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY',
        'PATH', 'HOME', 'USER'
    ]

    for var in env_vars:
        value = os.getenv(var)
        if value:
            if 'KEY' in var:
                print(f"  {var}: {'*' * min(10, len(value))} ({len(value)} chars)")
            else:
                print(f"  {var}: {value}")
        else:
            print(f"  {var}: [NOT SET]")


if __name__ == "__main__":
    print("=== Agent Adapter Debugging ===")

    test_command_availability()
    test_environment()
    test_manual_command()
    test_simple_adapter()
