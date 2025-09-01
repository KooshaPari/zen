#!/usr/bin/env python3
"""
Test script to launch Claude and Auggie in async mode for debugging
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from tools.agent_async import AgentAsyncTool
from tools.agent_inbox import AgentInboxTool


async def launch_claude_task():
    """Launch Claude task to create Flask REST API"""
    print("🚀 Launching Claude task for Flask REST API...")

    async_tool = AgentAsyncTool()
    result = await async_tool.execute({
        "agent_type": "claude",
        "task_description": "Flask REST API with user CRUD operations",
        "message": "Create a simple Flask REST API with endpoints for: GET /users (list users), POST /users (create user), GET /users/{id} (get user), PUT /users/{id} (update user), DELETE /users/{id} (delete user). Include proper error handling and basic validation.",
        "timeout_seconds": 1800,
        "priority": "normal"
    })

    print("✅ Claude task launched:")
    print(result[0].text)
    return result[0].text


async def launch_auggie_task():
    """Launch Auggie task to create JavaScript API client"""
    print("\n🚀 Launching Auggie task for JavaScript API client...")

    async_tool = AgentAsyncTool()
    result = await async_tool.execute({
        "agent_type": "aider",  # Using aider as a substitute for Auggie
        "task_description": "JavaScript API client with CRUD functions",
        "message": "Create JavaScript client code that can interact with a REST API. Include functions to: fetch all users, create a user, get a user by ID, update a user, and delete a user. Use modern fetch API with async/await and proper error handling.",
        "timeout_seconds": 1800,
        "priority": "normal"
    })

    print("✅ Auggie task launched:")
    print(result[0].text)
    return result[0].text


async def check_task_status(task_id):
    """Check the status of a task"""
    inbox_tool = AgentInboxTool()
    result = await inbox_tool.execute({
        "task_id": task_id,
        "action": "status"
    })
    return result[0].text


async def get_task_results(task_id):
    """Get the results of a completed task"""
    inbox_tool = AgentInboxTool()
    result = await inbox_tool.execute({
        "task_id": task_id,
        "action": "results"
    })
    return result[0].text


def extract_task_id(launch_response):
    """Extract task ID from launch response"""
    lines = launch_response.split('\n')
    for line in lines:
        if '**Task ID**:' in line:
            # Extract task ID from line like "**Task ID**: `task-id-here`"
            return line.split('`')[1]
    return None


async def main():
    """Main test function"""
    print("=== Testing Async Agent Orchestration ===\n")

    try:
        # Launch both tasks
        claude_response = await launch_claude_task()
        auggie_response = await launch_auggie_task()

        # Extract task IDs
        claude_task_id = extract_task_id(claude_response)
        auggie_task_id = extract_task_id(auggie_response)

        print("\n📋 Extracted Task IDs:")
        print(f"Claude: {claude_task_id}")
        print(f"Auggie: {auggie_task_id}")

        if not claude_task_id or not auggie_task_id:
            print("❌ Failed to extract task IDs")
            return

        # Monitor tasks for a bit
        print("\n🔍 Monitoring task status...")
        for i in range(5):  # Check 5 times
            print(f"\n--- Status Check {i+1} ---")

            # Check Claude task
            claude_status = await check_task_status(claude_task_id)
            print("Claude Status:")
            print(claude_status[:300] + "..." if len(claude_status) > 300 else claude_status)

            # Check Auggie task
            auggie_status = await check_task_status(auggie_task_id)
            print("\nAuggie Status:")
            print(auggie_status[:300] + "..." if len(auggie_status) > 300 else auggie_status)

            # Check if both completed
            if "COMPLETED" in claude_status and "COMPLETED" in auggie_status:
                print("\n🎉 Both tasks completed! Getting results...")

                claude_results = await get_task_results(claude_task_id)
                auggie_results = await get_task_results(auggie_task_id)

                print("\n=== Claude Results ===")
                print(claude_results)

                print("\n=== Auggie Results ===")
                print(auggie_results)
                break

            # Wait before next check
            if i < 4:  # Don't wait after last check
                print("⏳ Waiting 10 seconds before next check...")
                await asyncio.sleep(10)

        print("\n✅ Test completed!")

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
