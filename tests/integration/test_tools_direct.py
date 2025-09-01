#!/usr/bin/env python3
"""Direct test of Zen MCP tools without server"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import tools directly
from tools.analyze import AnalyzeTool
from tools.chat import ChatTool
from tools.codereview import CodeReviewTool
from tools.consensus import ConsensusTool
from tools.planner import PlannerTool
from tools.thinkdeep import ThinkDeepTool


async def test_tools():
    """Test various MCP tools directly"""

    print("🧪 Testing Zen MCP Tools Directly\n")

    # Test 1: Chat Tool
    print("1️⃣ Testing Chat Tool...")
    try:
        chat_tool = ChatTool()
        chat_result = await chat_tool.execute({"prompt": "Hello! Can you explain what the Zen MCP server does?"})
        response_text = chat_result[0].text if chat_result else ""
        print(f"✅ Chat responded: {response_text[:200]}...")
    except Exception as e:
        print(f"❌ Chat tool failed: {e}")

    # Test 2: Analyze Tool
    print("\n2️⃣ Testing Analyze Tool...")
    try:
        analyze_tool = AnalyzeTool()
        test_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"""
        analyze_result = await analyze_tool.execute({"code": test_code})
        response_text = analyze_result[0].text if analyze_result else ""
        print(f"✅ Analyze completed: {response_text[:200]}...")
    except Exception as e:
        print(f"❌ Analyze tool failed: {e}")

    # Test 3: CodeReview Tool
    print("\n3️⃣ Testing CodeReview Tool...")
    try:
        review_tool = CodeReviewTool()
        review_result = await review_tool.execute({
            "code": test_code,
            "context": "This is a recursive Fibonacci implementation"
        })
        response_text = review_result[0].text if review_result else ""
        print(f"✅ CodeReview completed: {response_text[:200]}...")
    except Exception as e:
        print(f"❌ CodeReview tool failed: {e}")

    # Test 4: Planner Tool
    print("\n4️⃣ Testing Planner Tool...")
    try:
        planner_tool = PlannerTool()
        planner_result = await planner_tool.execute({
            "task": "Create a simple REST API for a todo list application"
        })
        response_text = planner_result[0].text if planner_result else ""
        print(f"✅ Planner completed: {response_text[:200]}...")
    except Exception as e:
        print(f"❌ Planner tool failed: {e}")

    # Test 5: ThinkDeep Tool
    print("\n5️⃣ Testing ThinkDeep Tool...")
    try:
        think_tool = ThinkDeepTool()
        think_result = await think_tool.execute({
            "question": "What are the trade-offs between microservices and monolithic architectures?"
        })
        response_text = think_result[0].text if think_result else ""
        print(f"✅ ThinkDeep completed: {response_text[:200]}...")
    except Exception as e:
        print(f"❌ ThinkDeep tool failed: {e}")

    # Test 6: Consensus Tool
    print("\n6️⃣ Testing Consensus Tool...")
    try:
        consensus_tool = ConsensusTool()
        consensus_result = await consensus_tool.execute({
            "question": "Should we use TypeScript or JavaScript for a new project?",
            "stance": "neutral"
        })
        response_text = consensus_result[0].text if consensus_result else ""
        print(f"✅ Consensus completed: {response_text[:200]}...")
    except Exception as e:
        print(f"❌ Consensus tool failed: {e}")

    print("\n✨ Tool testing complete!")

if __name__ == "__main__":
    asyncio.run(test_tools())
