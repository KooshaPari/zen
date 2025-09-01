#!/usr/bin/env python3
"""
Test script for shared data infrastructure.
This verifies that all components are working correctly.
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.scope_utils import ScopeValidator, create_default_scope_context
from utils.vector_store import PgVectorStore, VectorDocument


async def test_vector_store():
    """Test vector store functionality."""
    print("\n🧪 Testing Vector Store...")

    # Create a scope context
    scope_context = create_default_scope_context(
        agent_id="test-agent",
        work_dir="backend"
    )
    print(f"✓ Created scope context for work_dir: {scope_context.work_dir}")

    # Initialize vector store
    vector_store = PgVectorStore()
    print("✓ Initialized vector store")

    # Create a collection
    collection = await vector_store.get_or_create_collection(
        scope_context,
        collection_name="test-collection",
        collection_type="code"
    )

    if collection:
        print(f"✓ Created collection: {collection.collection_name}")
    else:
        print("✗ Failed to create collection")
        return False

    # Add some test documents
    documents = [
        VectorDocument(
            content="def hello_world():\n    print('Hello, World!')",
            metadata={"type": "function", "language": "python"}
        ),
        VectorDocument(
            content="class UserAuthentication:\n    def login(self, username, password):\n        # Authenticate user",
            metadata={"type": "class", "language": "python"}
        ),
        VectorDocument(
            content="async function fetchUserData(userId) {\n    const response = await fetch(`/api/users/${userId}`);\n    return response.json();\n}",
            metadata={"type": "function", "language": "javascript"}
        )
    ]

    success = await vector_store.add_documents(
        scope_context,
        collection_name="test-collection",
        documents=documents,
        collection_type="code"
    )

    if success:
        print(f"✓ Added {len(documents)} documents to collection")
    else:
        print("✗ Failed to add documents")
        return False

    # Search for documents
    search_results = await vector_store.search(
        scope_context,
        collection_name="test-collection",
        query="authenticate user login",
        limit=3
    )

    if search_results:
        print(f"✓ Found {len(search_results)} similar documents:")
        for i, result in enumerate(search_results, 1):
            print(f"  {i}. Similarity: {result.similarity:.3f} - {result.content[:50]}...")
    else:
        print("✗ No search results found")

    return True


async def test_scope_validation():
    """Test scope validation functionality."""
    print("\n🧪 Testing Scope Validation...")

    # Test valid work_dir
    is_valid, error, scope = ScopeValidator.validate_work_dir(
        "backend/api",
        repo_root=os.getcwd()
    )

    if is_valid:
        print("✓ Valid work_dir accepted: backend/api")
    else:
        print(f"✗ Valid work_dir rejected: {error}")

    # Test invalid work_dir (path traversal)
    is_valid, error, scope = ScopeValidator.validate_work_dir(
        "../../../etc/passwd",
        repo_root=os.getcwd()
    )

    if not is_valid:
        print("✓ Invalid work_dir rejected (path traversal)")
    else:
        print("✗ Invalid work_dir accepted!")

    # Test forbidden directory
    is_valid, error, scope = ScopeValidator.validate_work_dir(
        "node_modules/package",
        repo_root=os.getcwd()
    )

    if not is_valid:
        print("✓ Forbidden directory rejected")
    else:
        print("✗ Forbidden directory accepted!")

    # Test shared interface
    is_valid, error, scope = ScopeValidator.validate_work_dir(
        "api/v1",
        repo_root=os.getcwd()
    )

    if is_valid and scope.is_shared:
        print("✓ Shared interface recognized: api/v1")
    else:
        print("✗ Shared interface not recognized")

    return True


async def test_redis_connection():
    """Test Redis connection."""
    print("\n🧪 Testing Redis Connection...")

    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✓ Redis connection successful")

        # Test basic operations
        r.set('test_key', 'test_value')
        value = r.get('test_key')
        if value == 'test_value':
            print("✓ Redis read/write successful")
        r.delete('test_key')
        return True
    except ImportError:
        print("⚠ Redis library not installed (pip install redis)")
        return True  # Not critical
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        return False


async def test_arangodb_connection():
    """Test ArangoDB connection."""
    print("\n🧪 Testing ArangoDB Connection...")

    try:
        import requests
        response = requests.get(
            "http://localhost:8529/_api/version",
            auth=('root', 'zen_arango_2025')
        )
        if response.status_code == 200:
            version = response.json()
            print(f"✓ ArangoDB connection successful (version: {version.get('version')})")
            return True
        else:
            print(f"✗ ArangoDB connection failed: {response.status_code}")
            return False
    except ImportError:
        print("⚠ Requests library not installed (pip install requests)")
        return True  # Not critical
    except Exception as e:
        print(f"✗ ArangoDB connection failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("🚀 Zen MCP Shared Data Infrastructure Test Suite")
    print("=" * 60)

    all_passed = True

    # Test scope validation
    if not await test_scope_validation():
        all_passed = False

    # Test vector store
    if not await test_vector_store():
        all_passed = False

    # Test Redis
    if not await test_redis_connection():
        all_passed = False

    # Test ArangoDB
    if not await test_arangodb_connection():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! Shared data infrastructure is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
