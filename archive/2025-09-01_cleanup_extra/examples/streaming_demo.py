#!/usr/bin/env python3
"""
Zen MCP Server Streaming Communication Demo

This example demonstrates the comprehensive streaming communication system
with real-time agent updates, natural language progress feeds, and the
interactive progress dashboard.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add the parent directory to Python path to import zen modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.shared.agent_models import AgentType
from utils.agent_prompts import AgentResponse, enhance_agent_message
from utils.languagization import NarrativeStyle, create_natural_progress_feed, get_progress_generator
from utils.streaming_protocol import InputTransformationPipeline, StreamMessageType, get_streaming_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamingDemo:
    """Demonstrates the streaming communication system."""

    def __init__(self):
        self.streaming_manager = get_streaming_manager()
        self.progress_generator = get_progress_generator(NarrativeStyle.CONVERSATIONAL)
        self.input_transformer = InputTransformationPipeline()

    async def demo_input_transformation(self):
        """Demonstrate input transformation with intent recognition."""
        print("\n🎯 Input Transformation Demo")
        print("=" * 50)

        test_inputs = [
            "Add authentication to the API with JWT tokens",
            "Fix the bug in the user registration process - it's urgent!",
            "Refactor the payment processing code to be more secure",
            "Write comprehensive tests for the shopping cart functionality",
            "Analyze the performance issues in the database queries"
        ]

        for user_input in test_inputs:
            print(f"\n📝 Input: {user_input}")

            enhanced = self.input_transformer.transform_input(user_input)

            print(f"🎯 Intent: {enhanced['intent']}")
            print(f"🔧 Complexity: {enhanced['complexity']}")
            print(f"⚡ Priority: {enhanced['priority']}")
            print(f"✅ Success Criteria: {', '.join(enhanced['success_criteria'])}")

            if enhanced['constraints']:
                print(f"🚫 Constraints: {', '.join(enhanced['constraints'])}")

            if enhanced['preferences']:
                print(f"⚙️ Preferences: {', '.join(enhanced['preferences'])}")

            print(f"📋 Implied Tasks: {len(enhanced['implied_tasks'])} tasks identified")

    async def demo_enhanced_communication_protocol(self):
        """Demonstrate the enhanced XML communication protocol."""
        print("\n🗣️ Enhanced Communication Protocol Demo")
        print("=" * 50)

        # Show enhanced prompt for Claude agent
        user_message = "Implement user authentication system with password hashing"
        enhanced_message = enhance_agent_message(user_message, AgentType.CLAUDE)

        print(f"📝 Original message: {user_message}")
        print("\n🔥 Enhanced message for Claude:")
        print("-" * 40)
        print(enhanced_message)

        # Simulate agent response with comprehensive XML tags
        simulated_response = """
<STATUS>working</STATUS>
<PROGRESS>step: 3/7</PROGRESS>
<CURRENT_ACTIVITY>Implementing password hashing with bcrypt</CURRENT_ACTIVITY>
<CONFIDENCE>high</CONFIDENCE>

<SUMMARY>Building authentication system with secure password hashing and JWT tokens.</SUMMARY>

<ACTIONS_COMPLETED>
- Created User model with email and password fields
- Added bcrypt dependency for password hashing
- Implemented user registration endpoint
</ACTIONS_COMPLETED>

<ACTIONS_IN_PROGRESS>
- Adding JWT token generation
- Creating login endpoint
</ACTIONS_IN_PROGRESS>

<ACTIONS_PLANNED>
- Add password validation middleware
- Write authentication tests
- Update API documentation
</ACTIONS_PLANNED>

<FILES_CREATED>
/src/models/user.py
/src/auth/password_utils.py
/src/routes/auth.py
</FILES_CREATED>

<FILES_MODIFIED>
/requirements.txt - Added bcrypt==4.0.1
/src/app.py - Added auth routes
</FILES_MODIFIED>

<DEPENDENCIES_ADDED>
bcrypt==4.0.1
pyjwt==2.8.0
</DEPENDENCIES_ADDED>

<TOOLS_USED>
- pip install bcrypt pyjwt
- python -m pytest tests/test_auth.py -v
</TOOLS_USED>

<QUESTIONS>
  <TECHNICAL>Should I use RS256 or HS256 for JWT signing?</TECHNICAL>
  <PREFERENCE>Do you want password complexity requirements?</PREFERENCE>
</QUESTIONS>

<RECOMMENDATIONS>
- Add rate limiting to login endpoint
- Implement 2FA for admin accounts
- Use environment variables for JWT secret
</RECOMMENDATIONS>

<TEST_RESULTS>
passed: 12
failed: 1
coverage: 87%
duration: 3.2s
</TEST_RESULTS>

<RESOURCES_USED>
memory: 145MB
cpu: 8%
tokens_consumed: 1,250
api_calls: 3
</RESOURCES_USED>
"""

        from utils.agent_prompts import AgentResponseParser

        parsed = AgentResponseParser.parse_response(simulated_response)
        formatted = AgentResponseParser.format_parsed_response(parsed)

        print("\n📊 Parsed Response:")
        print("-" * 40)
        print(formatted)

        # Generate natural language version
        natural_feed = create_natural_progress_feed(parsed, NarrativeStyle.CONVERSATIONAL)

        print("\n🗣️ Natural Language Feed:")
        print("-" * 40)
        print(natural_feed)

    async def demo_streaming_updates(self):
        """Demonstrate real-time streaming updates."""
        print("\n📡 Streaming Updates Demo")
        print("=" * 50)

        task_id = "demo-auth-task"

        # Simulate a series of streaming updates
        updates = [
            (StreamMessageType.STATUS_UPDATE, {
                "status": "starting",
                "message": "Beginning authentication system implementation"
            }),
            (StreamMessageType.ACTIVITY_UPDATE, {
                "content": "Analyzing current user model structure"
            }),
            (StreamMessageType.PROGRESS_UPDATE, {
                "content": "step: 1/7"
            }),
            (StreamMessageType.STATUS_UPDATE, {
                "status": "working",
                "message": "Implementing core authentication logic"
            }),
            (StreamMessageType.ACTION_UPDATE, {
                "tag": "ACTIONS_COMPLETED",
                "content": "- Created secure password hashing utility\n- Added user model validation"
            }),
            (StreamMessageType.PROGRESS_UPDATE, {
                "content": "42%"
            }),
            (StreamMessageType.ACTIVITY_UPDATE, {
                "content": "Writing authentication middleware"
            }),
            (StreamMessageType.QUESTION_UPDATE, {
                "content": "Should I implement session timeout?"
            }),
            (StreamMessageType.WARNING, {
                "content": "Current password policy might be too weak"
            }),
            (StreamMessageType.PROGRESS_UPDATE, {
                "content": "75%"
            }),
            (StreamMessageType.COMPLETION, {
                "status": "completed",
                "summary": "Authentication system implemented successfully",
                "files_created": ["/src/auth/middleware.py", "/tests/test_auth.py"],
                "files_modified": ["/src/app.py", "/requirements.txt"]
            })
        ]

        print(f"🎬 Simulating streaming updates for task: {task_id}")
        print("-" * 40)

        for i, (message_type, content) in enumerate(updates, 1):
            # Broadcast the streaming message
            await self.streaming_manager.broadcast_message(
                task_id,
                message_type,
                content,
                "claude"
            )

            # Generate natural language interpretation
            from utils.streaming_protocol import StreamMessage

            message = StreamMessage(
                id=f"msg-{i}",
                type=message_type,
                timestamp=datetime.utcnow().isoformat(),
                task_id=task_id,
                agent_type="claude",
                content=content,
                sequence=i
            )

            natural_update = self.progress_generator.process_stream_message(message)

            if natural_update:
                print(f"📢 {natural_update}")

            # Small delay between updates for realism
            await asyncio.sleep(0.5)

        print("\n✅ Streaming demo complete!")

        # Get overall task summary
        task_summary = self.progress_generator.get_task_summary(task_id)
        print("\n📋 Task Summary:")
        print(task_summary)

    async def demo_connection_management(self):
        """Demonstrate connection management features."""
        print("\n🔗 Connection Management Demo")
        print("=" * 50)

        # Show initial stats
        stats = self.streaming_manager.get_connection_stats()
        print(f"📊 Initial Stats: {json.dumps(stats, indent=2)}")

        # Simulate registering some connections
        task_ids = ["task-1", "task-2", "task-3"]
        connection_ids = []

        for i, task_id in enumerate(task_ids):
            connection_id = f"conn-{i+1}"
            connection_ids.append(connection_id)

            # Register SSE connection
            await self.streaming_manager.register_sse_connection(
                connection_id,
                [task_id]
            )

            print(f"✅ Registered connection {connection_id} for task {task_id}")

        # Show updated stats
        stats = self.streaming_manager.get_connection_stats()
        print(f"📊 After Registrations: {json.dumps(stats, indent=2)}")

        # Broadcast a message to all tasks
        for task_id in task_ids:
            await self.streaming_manager.broadcast_message(
                task_id,
                StreamMessageType.STATUS_UPDATE,
                {"status": "working", "message": f"Processing {task_id}"},
                "system"
            )

        print("📡 Broadcast messages sent to all tasks")

        # Clean up connections
        for connection_id in connection_ids:
            await self.streaming_manager.unregister_connection(connection_id)
            print(f"🗑️ Unregistered connection {connection_id}")

        # Show final stats
        stats = self.streaming_manager.get_connection_stats()
        print(f"📊 Final Stats: {json.dumps(stats, indent=2)}")

    async def demo_narrative_styles(self):
        """Demonstrate different narrative styles."""
        print("\n📝 Narrative Styles Demo")
        print("=" * 50)

        # Create a sample agent response
        sample_response = AgentResponse(
            status="working",
            progress="65%",
            current_activity="Implementing JWT token validation",
            confidence="high",
            summary="Building secure authentication system with comprehensive validation",
            actions_completed=[
                "Created user registration endpoint",
                "Added password hashing with bcrypt",
                "Implemented JWT token generation"
            ],
            files_created=[
                "/src/auth/jwt_utils.py",
                "/src/models/user.py"
            ],
            files_modified=[
                "/src/app.py",
                "/requirements.txt"
            ],
            questions_technical=[
                "Should we use RS256 or HS256 for JWT signing?"
            ],
            warnings=[
                "Current session timeout might be too short"
            ],
            recommendations=[
                "Add rate limiting to prevent brute force attacks",
                "Implement refresh token mechanism"
            ]
        )

        styles = [
            (NarrativeStyle.CONVERSATIONAL, "Friendly & Natural"),
            (NarrativeStyle.TECHNICAL, "Precise & Developer-focused"),
            (NarrativeStyle.EXECUTIVE, "High-level & Business-focused"),
            (NarrativeStyle.DETAILED, "Comprehensive & Step-by-step")
        ]

        for style, description in styles:
            print(f"\n🎭 {description} ({style.value}):")
            print("-" * 40)

            natural_feed = create_natural_progress_feed(sample_response, style)
            print(natural_feed)

    async def run_all_demos(self):
        """Run all demonstration scenarios."""
        print("\n🧘 Zen MCP Server - Comprehensive Streaming Demo")
        print("=" * 60)

        demos = [
            ("Input Transformation", self.demo_input_transformation),
            ("Enhanced Communication Protocol", self.demo_enhanced_communication_protocol),
            ("Streaming Updates", self.demo_streaming_updates),
            ("Connection Management", self.demo_connection_management),
            ("Narrative Styles", self.demo_narrative_styles)
        ]

        for name, demo_func in demos:
            try:
                await demo_func()
                await asyncio.sleep(1)  # Brief pause between demos
            except Exception as e:
                logger.error(f"Error in {name} demo: {e}")

        print("\n🎉 All demos completed!")
        print("\n🌐 To see the interactive dashboard:")
        print("   1. Install FastAPI: pip install fastapi uvicorn")
        print("   2. Run: python server_streaming.py")
        print("   3. Open: http://localhost:8000/dashboard")


async def main():
    """Main entry point for the streaming demo."""
    demo = StreamingDemo()
    await demo.run_all_demos()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)
