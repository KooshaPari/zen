#!/usr/bin/env python3
"""
Unit tests for shared data infrastructure including work_dir validation and scope enforcement.
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.scope_utils import (
    ScopeContext,
    ScopeValidator,
    WorkDirError,
    create_default_scope_context,
    extract_work_dir_from_args,
    inject_scope_context,
    validate_and_normalize_work_dir,
)


class TestScopeUtils:
    """Test suite for scope utilities."""

    def test_create_default_scope_context(self):
        """Test creating a default scope context."""
        context = create_default_scope_context(
            agent_id="test-agent",
            work_dir="backend",
            session_id="test-session"
        )

        assert context.agent_id == "test-agent"
        assert context.work_dir == "backend"
        assert context.session_id == "test-session"
        assert context.roles == ["user"]
        assert "read" in context.permissions
        assert "write" in context.permissions
        assert context.repo_root == os.getcwd()

    def test_scope_context_namespace_key(self):
        """Test namespace key generation."""
        context = ScopeContext(
            agent_id="test-agent",
            org_id="my-org",
            project_id="my-project",
            work_dir="backend/api"
        )

        # Should include org, project, and work_dir
        namespace = context.get_namespace_key()
        assert "my-org" in namespace
        assert "my-project" in namespace
        assert "backend_api" in namespace or "backend/api" in namespace

    def test_scope_context_cache_key(self):
        """Test cache key generation."""
        context = ScopeContext(
            agent_id="test-agent",
            org_id="my-org",
            project_id="my-project",
            work_dir="backend"
        )

        cache_key = context.get_cache_key()
        assert cache_key  # Should be non-empty
        assert len(cache_key) == 16  # Should be 16 characters (truncated SHA256)

        # Same context should generate same key
        context2 = ScopeContext(
            agent_id="test-agent",
            org_id="my-org",
            project_id="my-project",
            work_dir="backend"
        )
        assert context2.get_cache_key() == cache_key

    def test_validate_work_dir_valid(self):
        """Test validation of valid work directories."""
        repo_root = os.getcwd()

        # Empty work_dir (repo root)
        is_valid, error, scope = ScopeValidator.validate_work_dir("", repo_root)
        assert is_valid
        assert error == ""
        assert scope.work_dir == ""

        # Normal subdirectory
        is_valid, error, scope = ScopeValidator.validate_work_dir("backend", repo_root)
        assert is_valid
        assert error == ""
        assert scope.work_dir == "backend"

        # Nested directory
        is_valid, error, scope = ScopeValidator.validate_work_dir("backend/api/v1", repo_root)
        assert is_valid
        assert error == ""
        assert scope.work_dir == "backend/api/v1"

    def test_validate_work_dir_invalid(self):
        """Test validation of invalid work directories."""
        repo_root = os.getcwd()

        # Path traversal attempt
        is_valid, error, scope = ScopeValidator.validate_work_dir("../etc/passwd", repo_root)
        assert not is_valid
        assert "path traversal" in error.lower() or "invalid" in error.lower()
        assert scope is None

        # Home directory expansion
        is_valid, error, scope = ScopeValidator.validate_work_dir("~/secrets", repo_root)
        assert not is_valid
        assert scope is None

        # Forbidden directory
        is_valid, error, scope = ScopeValidator.validate_work_dir("node_modules/package", repo_root)
        assert not is_valid
        assert "forbidden" in error.lower()
        assert scope is None

        # .git directory
        is_valid, error, scope = ScopeValidator.validate_work_dir(".git/config", repo_root)
        assert not is_valid
        assert "forbidden" in error.lower()
        assert scope is None

    def test_validate_work_dir_shared_interfaces(self):
        """Test validation of shared interface directories."""
        repo_root = os.getcwd()

        # Shared interface directories should be marked
        shared_dirs = ["api", "contracts", "interfaces", "proto", "schema", "shared", "types"]

        for dir_name in shared_dirs:
            is_valid, error, scope = ScopeValidator.validate_work_dir(dir_name, repo_root)
            assert is_valid
            assert scope.is_shared
            assert scope.work_dir == dir_name

    def test_validate_work_dir_with_allowed_list(self):
        """Test validation with allowed work_dirs list."""
        repo_root = os.getcwd()
        allowed = ["backend", "frontend", "shared"]

        # Allowed directory
        is_valid, error, scope = ScopeValidator.validate_work_dir(
            "backend", repo_root, allowed
        )
        assert is_valid

        # Subdirectory of allowed
        is_valid, error, scope = ScopeValidator.validate_work_dir(
            "backend/api", repo_root, allowed
        )
        assert is_valid

        # Not in allowed list
        is_valid, error, scope = ScopeValidator.validate_work_dir(
            "scripts", repo_root, allowed
        )
        assert not is_valid
        assert "not in allowed list" in error

        # Shared interface should still be allowed
        is_valid, error, scope = ScopeValidator.validate_work_dir(
            "api", repo_root, allowed
        )
        assert is_valid  # Shared interfaces bypass allowed list

    def test_check_path_access(self):
        """Test path access checking."""
        context = ScopeContext(
            agent_id="test",
            work_dir="backend",
            repo_root=os.getcwd()
        )

        # Path within work_dir
        allowed, reason = ScopeValidator.check_path_access("backend/server.py", context)
        assert allowed
        assert "within work_dir" in reason.lower()

        # Path outside work_dir
        allowed, reason = ScopeValidator.check_path_access("frontend/app.js", context)
        assert not allowed
        assert "outside work_dir" in reason.lower()

        # Shared interface (readable)
        context.can_read_shared = True
        allowed, reason = ScopeValidator.check_path_access("api/schema.json", context)
        assert allowed
        assert "shared interface" in reason.lower()

    def test_filter_file_list(self):
        """Test filtering file lists based on scope."""
        context = ScopeContext(
            agent_id="test",
            work_dir="backend",
            repo_root=os.getcwd()
        )

        files = [
            "backend/server.py",
            "backend/models.py",
            "frontend/app.js",
            "tests/test_server.py",
            "api/schema.json"
        ]

        # Without shared access
        context.can_read_shared = False
        filtered = ScopeValidator.filter_file_list(files, context)
        assert "backend/server.py" in filtered
        assert "backend/models.py" in filtered
        assert "frontend/app.js" not in filtered
        assert "api/schema.json" not in filtered

        # With shared access
        context.can_read_shared = True
        filtered = ScopeValidator.filter_file_list(files, context)
        assert "api/schema.json" in filtered

    def test_inject_scope_context(self):
        """Test injecting scope context into tool arguments."""
        context = ScopeContext(
            agent_id="test",
            work_dir="backend",
            repo_root=os.getcwd()
        )

        args = {
            "prompt": "test prompt",
            "files": ["backend/a.py", "frontend/b.js", "api/c.json"]
        }

        # Inject context
        injected = inject_scope_context(args, context)

        # Should add work_dir
        assert injected["work_dir"] == "backend"

        # Should add scope context
        assert "_scope_context" in injected
        assert injected["_scope_context"]["agent_id"] == "test"

        # Should filter files
        context.can_read_shared = False
        injected = inject_scope_context(args, context)
        assert "backend/a.py" in injected["files"]
        assert "frontend/b.js" not in injected["files"]

    def test_extract_work_dir_from_args(self):
        """Test extracting work_dir from various argument formats."""
        # Direct work_dir
        assert extract_work_dir_from_args({"work_dir": "backend"}) == "backend"

        # Aliases
        assert extract_work_dir_from_args({"working_directory": "frontend"}) == "frontend"
        assert extract_work_dir_from_args({"workdir": "api"}) == "api"
        assert extract_work_dir_from_args({"directory": "tests"}) == "tests"

        # From file path
        assert extract_work_dir_from_args({"file_path": "backend/server.py"}) == "backend"
        assert extract_work_dir_from_args({"file": "api/v1/users.py"}) == "api/v1"

        # No work_dir found
        assert extract_work_dir_from_args({"prompt": "test"}) is None

    def test_validate_and_normalize_work_dir(self):
        """Test the main validation and normalization function."""
        # Valid directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a subdirectory
            subdir = os.path.join(tmpdir, "backend")
            os.makedirs(subdir)

            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # Test normalization
                rel, abs_path = validate_and_normalize_work_dir("backend")
                assert rel == "backend"
                assert abs_path == subdir

                # Test with ./ prefix
                rel, abs_path = validate_and_normalize_work_dir("./backend")
                assert rel == "backend"

                # Test empty (repo root)
                rel, abs_path = validate_and_normalize_work_dir("")
                assert rel == ""
                assert abs_path == tmpdir

            finally:
                os.chdir(original_cwd)

    def test_validate_and_normalize_work_dir_errors(self):
        """Test error cases for validation and normalization."""
        # Path traversal
        with pytest.raises(WorkDirError) as exc:
            validate_and_normalize_work_dir("../etc")
        assert "repo-relative path" in str(exc.value)

        # Absolute path
        with pytest.raises(WorkDirError) as exc:
            validate_and_normalize_work_dir("/etc/passwd")
        assert "repo-relative path" in str(exc.value)

        # Non-existent directory
        with pytest.raises(WorkDirError) as exc:
            validate_and_normalize_work_dir("nonexistent_dir_12345")
        assert "does not exist" in str(exc.value)


@pytest.mark.asyncio
class TestVectorStore:
    """Test suite for vector store integration."""

    @patch('utils.vector_store.POSTGRES_AVAILABLE', False)
    async def test_vector_store_unavailable(self):
        """Test behavior when PostgreSQL is not available."""
        from utils.vector_store import PgVectorStore

        store = PgVectorStore()
        context = create_default_scope_context("test", "backend")

        # Should return None/False when unavailable
        collection = await store.get_or_create_collection(context, "test-collection")
        assert collection is None

    @patch('utils.vector_store.REQUESTS_AVAILABLE', True)
    @patch('utils.vector_store.requests')
    async def test_embedding_provider_openrouter(self, mock_requests):
        """Test OpenRouter embedding provider."""
        from utils.vector_store import EmbeddingProvider

        # Mock OpenRouter response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 384}]
        }
        mock_requests.post.return_value = mock_response

        # Test embedding generation
        provider = EmbeddingProvider(provider="openrouter", api_key="test-key")
        embedding = await provider.embed_text("test text")

        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

        # Verify API call
        mock_requests.post.assert_called_once()
        call_args = mock_requests.post.call_args
        assert "openrouter.ai" in call_args[0][0]
        assert call_args[1]["json"]["input"] == "test text"

    @patch('utils.vector_store.REQUESTS_AVAILABLE', True)
    @patch('utils.vector_store.requests')
    async def test_embedding_provider_fallback(self, mock_requests):
        """Test fallback to random embeddings."""
        from utils.vector_store import EmbeddingProvider

        # Mock failed response
        mock_requests.post.side_effect = Exception("Connection error")

        provider = EmbeddingProvider(provider="openrouter", api_key="test-key")
        embedding = await provider.embed_text("test text")

        # Should fall back to random embeddings
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    async def test_embedding_cache(self):
        """Test embedding caching."""
        from utils.vector_store import EmbeddingProvider

        provider = EmbeddingProvider(provider="local")

        # Generate embedding
        text = "test text for caching"
        embedding1 = await provider.embed_text(text)

        # Should return cached version
        embedding2 = await provider.embed_text(text)
        assert embedding1 == embedding2

        # Cache should have the entry
        import hashlib
        cache_key = hashlib.md5(text.encode()).hexdigest()
        assert cache_key in provider._cache


@pytest.mark.asyncio
class TestKnowledgeGraph:
    """Test suite for knowledge graph integration."""

    @patch('utils.knowledge_graph.REQUESTS_AVAILABLE', False)
    async def test_knowledge_graph_unavailable(self):
        """Test behavior when requests library is not available."""
        from utils.knowledge_graph import ArangoKnowledgeGraph, GraphNode

        graph = ArangoKnowledgeGraph()
        context = create_default_scope_context("test", "backend")

        node = GraphNode(
            node_type="function",
            name="test_function",
            work_dir_id="test"
        )

        # Should return False when unavailable
        result = await graph.add_node(context, node)
        assert result is False

    @patch('utils.knowledge_graph.REQUESTS_AVAILABLE', True)
    @patch('utils.knowledge_graph.requests')
    async def test_add_node(self, mock_requests):
        """Test adding a node to the knowledge graph."""
        from utils.knowledge_graph import ArangoKnowledgeGraph, GraphNode, NodeType

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_requests.post.return_value = mock_response
        mock_requests.get.return_value = mock_response

        graph = ArangoKnowledgeGraph()
        context = create_default_scope_context("test", "backend")

        node = GraphNode(
            node_type=NodeType.FUNCTION,
            name="calculate_total",
            work_dir_id="test",
            file_path="backend/utils.py",
            line_start=10,
            line_end=20
        )

        result = await graph.add_node(context, node)
        assert result is True

    @patch('utils.knowledge_graph.REQUESTS_AVAILABLE', True)
    @patch('utils.knowledge_graph.requests')
    async def test_query_nodes(self, mock_requests):
        """Test querying nodes from the knowledge graph."""
        from utils.knowledge_graph import ArangoKnowledgeGraph, NodeType

        # Mock query response
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "result": [
                {
                    "node_id": "123",
                    "node_type": NodeType.FUNCTION,
                    "name": "test_func",
                    "work_dir_id": "test",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z"
                }
            ]
        }
        mock_requests.post.return_value = mock_response
        mock_requests.get.return_value = MagicMock(status_code=200, json=lambda: {"result": []})

        graph = ArangoKnowledgeGraph()
        context = create_default_scope_context("test", "backend")

        nodes = await graph.query_nodes(context, node_type=NodeType.FUNCTION)
        assert len(nodes) == 1
        assert nodes[0].name == "test_func"

    @patch('utils.knowledge_graph.REQUESTS_AVAILABLE', True)
    @patch('utils.knowledge_graph.requests')
    async def test_find_dependencies(self, mock_requests):
        """Test finding dependencies of a node."""
        from utils.knowledge_graph import ArangoKnowledgeGraph

        # Mock traversal response
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "result": [
                {
                    "node": {"node_id": "1", "name": "func1", "node_type": "function"},
                    "edge": {"edge_type": "calls", "from_node_id": "1", "to_node_id": "2"},
                    "path": {}
                },
                {
                    "node": {"node_id": "2", "name": "func2", "node_type": "function"},
                    "edge": None,
                    "path": {}
                }
            ]
        }
        mock_requests.post.return_value = mock_response
        mock_requests.get.return_value = MagicMock(status_code=200, json=lambda: {"result": []})

        graph = ArangoKnowledgeGraph()
        context = create_default_scope_context("test", "backend")

        deps = await graph.find_dependencies(context, "node1", depth=2)
        assert "nodes" in deps
        assert "edges" in deps
        assert len(deps["nodes"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
