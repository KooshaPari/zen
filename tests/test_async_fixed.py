#!/usr/bin/env python3
"""
Test async agents with proper environment loaded
"""

import asyncio
import os
import sys
from pathlib import Path

# Load environment from .env file FIRST
from dotenv import load_dotenv

load_dotenv()

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from tools.agent_async import AgentAsyncTool
from tools.agent_inbox import AgentInboxTool


def extract_task_id(launch_response):
    """Extract task ID from launch response"""
    lines = launch_response.split('\n')
    for line in lines:
        if '**Task ID**:' in line:
            return line.split('`')[1]
    return None


async def test_async_workflow():
    """Test the complete async workflow with proper environment"""
    print("=== Testing Async Agent Workflow with Proper Environment ===\n")

    # Check environment
    print("Environment check:")
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    print(f"  OPENROUTER_API_KEY: {'[SET]' if openrouter_key else '[NOT SET]'}")

    if not openrouter_key:
        print("❌ No API key found! Make sure .env file has OPENROUTER_API_KEY")
        return

    try:
        # Launch Claude task
        print("\n🚀 Launching Claude task...")
        async_tool = AgentAsyncTool()
        claude_result = await async_tool.execute({
            "agent_type": "claude",
            "task_description": "Flask REST API with user CRUD operations",
            "message": "Create a simple Flask REST API with endpoints for: GET /users (list users), POST /users (create user), GET /users/{id} (get user), PUT /users/{id} (update user), DELETE /users/{id} (delete user). Include proper error handling and basic validation. Keep it simple and save as users_api.py",
            "timeout_seconds": 120,  # 2 minute timeout
            "working_directory": "/tmp/claude_task"
        })

        claude_task_id = extract_task_id(claude_result[0].text)
        print(f"✅ Claude task launched: {claude_task_id}")

        # Launch Aider task
        print("\n🚀 Launching Aider task...")
        aider_result = await async_tool.execute({
            "agent_type": "aider",
            "task_description": "JavaScript API client with CRUD functions",
            "message": "Create JavaScript client code that can interact with a REST API. Include functions to: fetch all users, create a user, get a user by ID, update a user, and delete a user. Use modern fetch API with async/await and proper error handling. Save as api_client.js",
            "timeout_seconds": 120,
            "working_directory": "/tmp/aider_task"
        })

        aider_task_id = extract_task_id(aider_result[0].text)
        print(f"✅ Aider task launched: {aider_task_id}")

        if not claude_task_id or not aider_task_id:
            print("❌ Failed to extract task IDs")
            return

        # Monitor both tasks
        print("\n🔍 Monitoring tasks...")
        inbox_tool = AgentInboxTool()

        claude_completed = False
        aider_completed = False
        max_checks = 20  # Check up to 20 times (10 minutes max)

        for i in range(max_checks):
            print(f"\n--- Status Check {i+1} ---")

            # Check Claude status
            if not claude_completed:
                try:
                    claude_status = await inbox_tool.execute({
                        "task_id": claude_task_id,
                        "action": "status"
                    })
                    status_text = claude_status[0].text
                    if "COMPLETED" in status_text:
                        print("✅ Claude task COMPLETED!")
                        claude_completed = True

                        # Get results
                        claude_results = await inbox_tool.execute({
                            "task_id": claude_task_id,
                            "action": "results"
                        })
                        print("\n=== Claude Results ===")
                        print(claude_results[0].text)

                    elif "FAILED" in status_text:
                        print("❌ Claude task FAILED!")
                        claude_completed = True
                        print(status_text[:500])
                    else:
                        print(f"⏳ Claude: {status_text.split('**Status**:')[1].split('**')[0].strip() if '**Status**:' in status_text else 'RUNNING'}")

                except Exception as e:
                    print(f"❌ Error checking Claude: {e}")
                    claude_completed = True

            # Check Aider status
            if not aider_completed:
                try:
                    aider_status = await inbox_tool.execute({
                        "task_id": aider_task_id,
                        "action": "status"
                    })
                    status_text = aider_status[0].text
                    if "COMPLETED" in status_text:
                        print("✅ Aider task COMPLETED!")
                        aider_completed = True

                        # Get results
                        aider_results = await inbox_tool.execute({
                            "task_id": aider_task_id,
                            "action": "results"
                        })
                        print("\n=== Aider Results ===")
                        print(aider_results[0].text)

                    elif "FAILED" in status_text:
                        print("❌ Aider task FAILED!")
                        aider_completed = True
                        print(status_text[:500])
                    else:
                        print(f"⏳ Aider: {status_text.split('**Status**:')[1].split('**')[0].strip() if '**Status**:' in status_text else 'RUNNING'}")

                except Exception as e:
                    print(f"❌ Error checking Aider: {e}")
                    aider_completed = True

            # If both completed, we're done
            if claude_completed and aider_completed:
                print("\n🎉 Both tasks completed!")
                break

            # Wait before next check
            if i < max_checks - 1:
                print("⏳ Waiting 30 seconds before next check...")
                await asyncio.sleep(30)

        if not (claude_completed and aider_completed):
            print(f"\n⏰ Monitoring timed out after {max_checks} checks")
            print(f"Claude completed: {claude_completed}, Aider completed: {aider_completed}")

        print("\n✅ Test completed!")

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_async_workflow())
