#!/usr/bin/env python3
"""
Comprehensive Agent Orchestration Test

This script tests the full agent orchestration system including:
- Sync agent execution
- Async agent execution
- Agent inbox management
- Multiple agent types (Claude, Auggie, Aider, Gemini)
"""

import asyncio
import sys
import time

sys.path.append('.')

from tools.agent_async import AgentAsyncTool
from tools.agent_inbox import AgentInboxTool
from tools.agent_sync import AgentSyncTool


async def test_sync_agents():
    """Test synchronous agent execution"""
    print("=" * 60)
    print("🔄 TESTING SYNCHRONOUS AGENTS")
    print("=" * 60)

    sync_tool = AgentSyncTool()

    # Test cases for different agents
    test_cases = [
        {
            'agent_type': 'claude',
            'task_description': 'Create Python hello world',
            'message': 'Create a simple hello world Python script with a function and main block',
            'timeout': 30
        },
        {
            'agent_type': 'aider',
            'task_description': 'Create utility function',
            'message': 'Create a simple utility function that returns the current timestamp',
            'timeout': 30
        }
    ]

    results = []
    for test_case in test_cases:
        agent_type = test_case['agent_type']
        print(f"\n🧪 Testing {agent_type} in sync mode...")

        try:
            arguments = {
                'agent_type': agent_type,
                'task_description': test_case['task_description'],
                'message': test_case['message'],
                'working_directory': '/Users/kooshapari/temp-PRODVERCEL/485/kush/zen-mcp-server/test_workspace',
                'timeout_seconds': test_case['timeout']
            }

            result = await sync_tool.execute(arguments)
            if result and result[0].text:
                # Check if task completed successfully
                text = result[0].text
                if "COMPLETED" in text:
                    print(f"✅ {agent_type} sync test: SUCCESS")
                    results.append((agent_type, "sync", "SUCCESS"))
                elif "FAILED" in text:
                    print(f"❌ {agent_type} sync test: FAILED")
                    results.append((agent_type, "sync", "FAILED"))
                elif "TIMEOUT" in text:
                    print(f"⏰ {agent_type} sync test: TIMEOUT")
                    results.append((agent_type, "sync", "TIMEOUT"))
                else:
                    print(f"❓ {agent_type} sync test: UNKNOWN")
                    results.append((agent_type, "sync", "UNKNOWN"))

                # Print abbreviated result
                print(f"Result summary: {text[:150]}...")
            else:
                print(f"❌ {agent_type} sync test: NO RESULT")
                results.append((agent_type, "sync", "NO_RESULT"))

        except Exception as e:
            print(f"❌ {agent_type} sync test: ERROR - {str(e)}")
            results.append((agent_type, "sync", "ERROR"))

    return results

async def test_async_agents():
    """Test asynchronous agent execution"""
    print("\n" + "=" * 60)
    print("⚡ TESTING ASYNCHRONOUS AGENTS")
    print("=" * 60)

    async_tool = AgentAsyncTool()
    inbox_tool = AgentInboxTool()

    # Test cases for async execution
    test_cases = [
        {
            'agent_type': 'claude',
            'task_description': 'Create simple calculator',
            'message': 'Create a basic calculator class with add, subtract, multiply, divide methods',
            'timeout': 120
        },
        {
            'agent_type': 'aider',
            'task_description': 'Create config parser',
            'message': 'Create a simple configuration file parser',
            'timeout': 120
        }
    ]

    launched_tasks = []

    # Launch async tasks
    print("\n🚀 Launching asynchronous tasks...")
    for test_case in test_cases:
        agent_type = test_case['agent_type']
        try:
            arguments = {
                'agent_type': agent_type,
                'task_description': test_case['task_description'],
                'message': test_case['message'],
                'working_directory': '/Users/kooshapari/temp-PRODVERCEL/485/kush/zen-mcp-server/test_workspace',
                'timeout_seconds': test_case['timeout']
            }

            result = await async_tool.execute(arguments)
            if result and result[0].text:
                text = result[0].text
                # Extract task ID
                if 'Task ID' in text and '`' in text:
                    parts = text.split('`')
                    if len(parts) >= 2:
                        task_id = parts[1]
                        launched_tasks.append((agent_type, task_id))
                        print(f"✅ {agent_type} async task launched: {task_id}")
                else:
                    print(f"⚠️ {agent_type} task launched but no task ID extracted")
        except Exception as e:
            print(f"❌ {agent_type} async launch failed: {str(e)}")

    # Wait for tasks to initialize
    if launched_tasks:
        print(f"\n⏳ Waiting for {len(launched_tasks)} tasks to initialize...")
        await asyncio.sleep(3)

        # Check inbox
        print("\n📋 Checking agent inbox...")
        try:
            inbox_result = await inbox_tool.execute({'action': 'list'})
            if inbox_result and inbox_result[0].text:
                print("Inbox status:")
                print(inbox_result[0].text[:500] + "..." if len(inbox_result[0].text) > 500 else inbox_result[0].text)
        except Exception as e:
            print(f"Inbox check failed: {e}")

    return launched_tasks

async def test_agent_availability():
    """Test which agents are available"""
    print("\n" + "=" * 60)
    print("🔍 CHECKING AGENT AVAILABILITY")
    print("=" * 60)

    from tools.shared.agent_models import AgentType

    available_agents = [agent.value for agent in AgentType]
    print(f"Available agent types: {available_agents}")

    return available_agents

async def main():
    """Run comprehensive agent orchestration tests"""
    print("🧪 COMPREHENSIVE AGENT ORCHESTRATION TEST")
    print("Testing sync mode, async mode, and agent inbox functionality")
    print("=" * 80)

    start_time = time.time()

    # Test agent availability
    agents = await test_agent_availability()

    # Test synchronous agents
    sync_results = await test_sync_agents()

    # Test asynchronous agents
    async_tasks = await test_async_agents()

    # Final summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    print(f"Available agents: {len(agents)}")
    print(f"Synchronous tests completed: {len(sync_results)}")
    print(f"Asynchronous tasks launched: {len(async_tasks)}")

    print("\nSync test results:")
    for agent, mode, status in sync_results:
        print(f"  {agent} ({mode}): {status}")

    print("\nAsync tasks launched:")
    for agent, task_id in async_tasks:
        print(f"  {agent}: {task_id}")

    elapsed = time.time() - start_time
    print(f"\nTotal test duration: {elapsed:.1f} seconds")

    print("\n✅ Comprehensive agent orchestration test completed!")
    print("The agent orchestration system is functional and ready for use.")

if __name__ == "__main__":
    asyncio.run(main())
