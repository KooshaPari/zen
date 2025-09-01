import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.enhanced_model_router import smart_model_selection


async def test_routing():
    # Test adaptive model selection
    model, metadata = await smart_model_selection(
        task_type="chat",
        prompt="Hello, how are you?",
        files=[],
        optimization="speed"
    )

    print(f"Selected Model: {model}")
    print(f"Strategy: {metadata.get('strategy', 'unknown')}")
    print(f"Complexity Tier: {metadata.get('complexity_tier', 'N/A')}")
    print(f"Complexity Score: {metadata.get('complexity_score', 'N/A')}")
    print(f"Selection Time: {metadata.get('selection_time_ms', 'N/A')}ms")

    return model, metadata

# Run the async function
result = asyncio.run(test_routing())
print("\nAdaptive routing is working!")
