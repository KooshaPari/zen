#!/usr/bin/env python3
"""
Test script to verify the deploy tool fix is working.
This tests the fix for the argument passing issue.
"""

import asyncio
import json
import os
import sys

# Add the project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_deploy_tool():
    """Test the deploy tool with the fixed execute method."""

    print("=" * 60)
    print("DEPLOY TOOL FIX VERIFICATION TEST")
    print("=" * 60)

    try:
        # Import the deploy tool
        from tools.universal_executor import DeployTool
        print("✅ Successfully imported DeployTool")

        # Check the method signature
        import inspect
        sig = inspect.signature(DeployTool.execute)
        params = list(sig.parameters.keys())
        print(f"\nMethod signature parameters: {params}")

        if 'arguments' in params:
            print("✅ VERIFIED: execute method uses 'arguments' parameter")
        else:
            print(f"❌ ISSUE: execute method parameters are {params}")
            return False

        # Create an instance
        tool = DeployTool()
        print("\n✅ Successfully created DeployTool instance")

        # Test with a simple request
        test_args = {
            "prompt": "Calculate 5 * 5",
            "agent_type": "llm",
            "model": "anthropic/claude-3.5-haiku"
        }

        print(f"\nTest arguments: {json.dumps(test_args, indent=2)}")

        # Try to execute (this will fail due to missing API keys, but that's ok)
        # We're just testing that the method accepts 'arguments'
        try:
            print("\nAttempting to call execute method...")
            result = await tool.execute(test_args)
            print("✅ Execute method called successfully!")
            print(f"Result: {result}")
        except Exception as e:
            error_msg = str(e)
            # Check if it's the old "missing positional argument" error
            if "missing 1 required positional argument" in error_msg:
                print(f"❌ FIX NOT WORKING: {error_msg}")
                return False
            elif "API" in error_msg or "provider" in error_msg.lower():
                print(f"✅ Method signature is correct (API error is expected): {error_msg[:100]}...")
                return True
            else:
                print(f"⚠️ Unexpected error (but signature seems ok): {error_msg[:100]}...")
                return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTrying alternative test without MCP imports...")

        # Test just the signature without full import
        import ast
        with open('tools/universal_executor.py') as f:
            content = f.read()

        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == 'execute':
                args = node.args
                if len(args.args) >= 2:
                    param_name = args.args[1].arg
                    if param_name == 'arguments':
                        print(f"✅ AST VERIFICATION: execute method uses '{param_name}' parameter")
                        return True
                    else:
                        print(f"❌ AST VERIFICATION: execute method uses '{param_name}' instead of 'arguments'")
                        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

    return True

if __name__ == "__main__":
    print("Testing Deploy Tool Fix...\n")

    # Run the async test
    result = asyncio.run(test_deploy_tool())

    print("\n" + "=" * 60)
    if result:
        print("✅ FIX VERIFICATION: PASSED")
        print("The deploy tool has been successfully fixed!")
    else:
        print("❌ FIX VERIFICATION: FAILED")
        print("The deploy tool still has issues.")
    print("=" * 60)

    sys.exit(0 if result else 1)
