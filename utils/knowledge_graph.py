"""
Knowledge Graph Integration with ArangoDB

This module provides the knowledge graph infrastructure for the Zen MCP Server,
enabling relationship tracking, dependency analysis, and cross-cutting concerns
using ArangoDB as a multi-model database (document + graph).

Key Features:
- Code structure modeling (classes, functions, modules)
- API endpoint tracking and relationships
- Test coverage mapping
- Dependency graph analysis
- Cross-repository relationships
- Work_dir scoped queries
- Multi-tenant isolation
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

from pydantic import BaseModel, Field

from utils.scope_utils import ScopeContext

logger = logging.getLogger(__name__)


class NodeType:
    """Standard node types in the knowledge graph."""
    # Code entities
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"

    # API entities
    ENDPOINT = "endpoint"
    SCHEMA = "schema"
    REQUEST = "request"
    RESPONSE = "response"

    # Test entities
    TEST_SUITE = "test_suite"
    TEST_CASE = "test_case"
    ASSERTION = "assertion"

    # Documentation
    DOCUMENT = "document"
    SECTION = "section"

    # Infrastructure
    SERVICE = "service"
    DATABASE = "database"
    QUEUE = "queue"

    # Meta
    WORK_DIR = "work_dir"
    PROJECT = "project"
    ORGANIZATION = "organization"


class EdgeType:
    """Standard edge types (relationships) in the knowledge graph."""
    # Code relationships
    IMPORTS = "imports"
    CALLS = "calls"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    USES = "uses"
    DEFINES = "defines"
    CONTAINS = "contains"

    # API relationships
    EXPOSES = "exposes"
    CONSUMES = "consumes"
    VALIDATES = "validates"

    # Test relationships
    TESTS = "tests"
    COVERS = "covers"
    MOCKS = "mocks"

    # Documentation relationships
    DOCUMENTS = "documents"
    REFERENCES = "references"

    # Dependency relationships
    DEPENDS_ON = "depends_on"
    REQUIRED_BY = "required_by"

    # Version relationships
    REPLACES = "replaces"
    REPLACED_BY = "replaced_by"

    # Ownership
    OWNS = "owns"
    BELONGS_TO = "belongs_to"


class GraphNode(BaseModel):
    """Represents a node in the knowledge graph."""

    node_id: str = Field(default_factory=lambda: str(uuid4()))
    node_type: str = Field(..., description="Type of node (from NodeType)")
    name: str = Field(..., description="Name of the entity")
    work_dir_id: str = Field(..., description="Work directory scope")

    # Metadata
    file_path: Optional[str] = Field(None, description="Source file path")
    line_start: Optional[int] = Field(None, description="Starting line number")
    line_end: Optional[int] = Field(None, description="Ending line number")
    language: Optional[str] = Field(None, description="Programming language")

    # Content
    content: Optional[str] = Field(None, description="Node content (code, text, etc.)")
    signature: Optional[str] = Field(None, description="Function/method signature")
    docstring: Optional[str] = Field(None, description="Documentation string")

    # Metrics
    complexity: Optional[int] = Field(None, description="Cyclomatic complexity")
    lines_of_code: Optional[int] = Field(None, description="Lines of code")
    test_coverage: Optional[float] = Field(None, description="Test coverage percentage")

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Additional properties
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """Represents an edge (relationship) in the knowledge graph."""

    edge_id: str = Field(default_factory=lambda: str(uuid4()))
    edge_type: str = Field(..., description="Type of relationship (from EdgeType)")
    from_node_id: str = Field(..., description="Source node ID")
    to_node_id: str = Field(..., description="Target node ID")
    work_dir_id: str = Field(..., description="Work directory scope")

    # Edge properties
    weight: float = Field(default=1.0, description="Relationship strength/weight")
    properties: dict[str, Any] = Field(default_factory=dict)

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ArangoKnowledgeGraph:
    """
    ArangoDB-based knowledge graph for code relationships and dependencies.

    This class manages the graph database, providing methods for:
    - Creating and querying nodes/edges
    - Traversing relationships
    - Analyzing dependencies
    - Finding patterns and anti-patterns
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        username: str = None,
        password: str = None,
        database: str = None
    ):
        """
        Initialize the knowledge graph connection.

        Args:
            host: ArangoDB host
            port: ArangoDB port
            username: Database username
            password: Database password
            database: Database name
        """
        self.host = host or os.getenv("ARANGO_HOST", "localhost")
        self.port = port or int(os.getenv("ARANGO_PORT", "8529"))
        self.username = username or os.getenv("ARANGO_USER", "root")
        self.password = password or os.getenv("ARANGO_PASSWORD", "zen_arango_2025")
        self.database = database or os.getenv("ARANGO_DATABASE", "zen_knowledge")

        self.base_url = f"http://{self.host}:{self.port}"
        self.auth = (self.username, self.password)

        # Collection names
        self.nodes_collection = "code_nodes"
        self.edges_collection = "code_relationships"

        # Initialize database and collections
        self._init_database()

    def _init_database(self):
        """Initialize database and collections if they don't exist."""
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests library not available, knowledge graph disabled")
            return

        try:
            # Create database if it doesn't exist
            response = requests.get(
                f"{self.base_url}/_api/database",
                auth=self.auth
            )

            if response.status_code == 200:
                databases = response.json().get("result", [])
                if self.database not in databases:
                    requests.post(
                        f"{self.base_url}/_api/database",
                        json={"name": self.database},
                        auth=self.auth
                    )
                    logger.info(f"Created database: {self.database}")

            # Switch to the database
            db_url = f"{self.base_url}/_db/{self.database}"

            # Create collections if they don't exist
            for collection in [self.nodes_collection, self.edges_collection]:
                response = requests.get(
                    f"{db_url}/_api/collection/{collection}",
                    auth=self.auth
                )

                if response.status_code == 404:
                    # Determine collection type
                    collection_type = 3 if collection == self.edges_collection else 2  # 3=edge, 2=document

                    requests.post(
                        f"{db_url}/_api/collection",
                        json={"name": collection, "type": collection_type},
                        auth=self.auth
                    )
                    logger.info(f"Created collection: {collection}")

            # Create indexes for efficient queries
            self._create_indexes()

        except Exception as e:
            logger.error(f"Failed to initialize ArangoDB: {e}")

    def _create_indexes(self):
        """Create indexes for efficient querying."""
        if not REQUESTS_AVAILABLE:
            return

        try:
            db_url = f"{self.base_url}/_db/{self.database}"

            # Index on work_dir_id for nodes
            requests.post(
                f"{db_url}/_api/index",
                json={
                    "type": "persistent",
                    "fields": ["work_dir_id"],
                    "unique": False,
                    "sparse": False
                },
                params={"collection": self.nodes_collection},
                auth=self.auth
            )

            # Compound index on work_dir_id and node_type
            requests.post(
                f"{db_url}/_api/index",
                json={
                    "type": "persistent",
                    "fields": ["work_dir_id", "node_type"],
                    "unique": False,
                    "sparse": False
                },
                params={"collection": self.nodes_collection},
                auth=self.auth
            )

            # Index on work_dir_id for edges
            requests.post(
                f"{db_url}/_api/index",
                json={
                    "type": "persistent",
                    "fields": ["work_dir_id"],
                    "unique": False,
                    "sparse": False
                },
                params={"collection": self.edges_collection},
                auth=self.auth
            )

        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")

    async def add_node(
        self,
        scope_context: ScopeContext,
        node: GraphNode
    ) -> bool:
        """
        Add a node to the knowledge graph.

        Args:
            scope_context: Scope context with work_dir
            node: Node to add

        Returns:
            True if successful, False otherwise
        """
        if not REQUESTS_AVAILABLE:
            return False

        try:
            # Get work_dir_id from scope
            work_dir_id = scope_context.get_namespace_key()
            node.work_dir_id = work_dir_id

            db_url = f"{self.base_url}/_db/{self.database}"

            # Convert node to dict
            node_data = node.model_dump()
            node_data["_key"] = node.node_id

            # Insert node
            response = requests.post(
                f"{db_url}/_api/document/{self.nodes_collection}",
                json=node_data,
                auth=self.auth
            )

            if response.status_code in [201, 202]:
                logger.debug(f"Added node: {node.name} ({node.node_type})")
                return True
            else:
                logger.error(f"Failed to add node: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error adding node: {e}")
            return False

    async def add_edge(
        self,
        scope_context: ScopeContext,
        edge: GraphEdge
    ) -> bool:
        """
        Add an edge to the knowledge graph.

        Args:
            scope_context: Scope context with work_dir
            edge: Edge to add

        Returns:
            True if successful, False otherwise
        """
        if not REQUESTS_AVAILABLE:
            return False

        try:
            # Get work_dir_id from scope
            work_dir_id = scope_context.get_namespace_key()
            edge.work_dir_id = work_dir_id

            db_url = f"{self.base_url}/_db/{self.database}"

            # Convert edge to dict with proper ArangoDB format
            edge_data = edge.model_dump()
            edge_data["_key"] = edge.edge_id
            edge_data["_from"] = f"{self.nodes_collection}/{edge.from_node_id}"
            edge_data["_to"] = f"{self.nodes_collection}/{edge.to_node_id}"

            # Insert edge
            response = requests.post(
                f"{db_url}/_api/document/{self.edges_collection}",
                json=edge_data,
                auth=self.auth
            )

            if response.status_code in [201, 202]:
                logger.debug(f"Added edge: {edge.edge_type} ({edge.from_node_id} -> {edge.to_node_id})")
                return True
            else:
                logger.error(f"Failed to add edge: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error adding edge: {e}")
            return False

    async def query_nodes(
        self,
        scope_context: ScopeContext,
        node_type: Optional[str] = None,
        name_pattern: Optional[str] = None,
        limit: int = 100
    ) -> list[GraphNode]:
        """
        Query nodes from the knowledge graph.

        Args:
            scope_context: Scope context with work_dir
            node_type: Filter by node type
            name_pattern: Filter by name pattern
            limit: Maximum number of results

        Returns:
            List of matching nodes
        """
        if not REQUESTS_AVAILABLE:
            return []

        try:
            work_dir_id = scope_context.get_namespace_key()
            db_url = f"{self.base_url}/_db/{self.database}"

            # Build AQL query
            aql_query = f"""
                FOR node IN {self.nodes_collection}
                FILTER node.work_dir_id == @work_dir_id
            """

            bind_vars = {"work_dir_id": work_dir_id}

            if node_type:
                aql_query += " FILTER node.node_type == @node_type"
                bind_vars["node_type"] = node_type

            if name_pattern:
                aql_query += " FILTER CONTAINS(LOWER(node.name), LOWER(@pattern))"
                bind_vars["pattern"] = name_pattern

            aql_query += f" LIMIT {limit} RETURN node"

            # Execute query
            response = requests.post(
                f"{db_url}/_api/cursor",
                json={"query": aql_query, "bindVars": bind_vars},
                auth=self.auth
            )

            if response.status_code == 201:
                result = response.json()
                nodes = []
                for node_data in result.get("result", []):
                    # Remove ArangoDB internal fields
                    node_data.pop("_id", None)
                    node_data.pop("_key", None)
                    node_data.pop("_rev", None)
                    nodes.append(GraphNode(**node_data))
                return nodes
            else:
                logger.error(f"Query failed: {response.text}")
                return []

        except Exception as e:
            logger.error(f"Error querying nodes: {e}")
            return []

    async def find_dependencies(
        self,
        scope_context: ScopeContext,
        node_id: str,
        depth: int = 2,
        direction: str = "outbound"
    ) -> dict[str, Any]:
        """
        Find dependencies of a node.

        Args:
            scope_context: Scope context with work_dir
            node_id: Node to analyze
            depth: Traversal depth
            direction: "outbound", "inbound", or "any"

        Returns:
            Dictionary with nodes and edges in the dependency graph
        """
        if not REQUESTS_AVAILABLE:
            return {"nodes": [], "edges": []}

        try:
            work_dir_id = scope_context.get_namespace_key()
            db_url = f"{self.base_url}/_db/{self.database}"

            # Build traversal query
            aql_query = f"""
                FOR v, e, p IN 1..@depth {direction.upper()}
                CONCAT(@nodes_collection, '/', @start_node)
                {self.edges_collection}
                FILTER v.work_dir_id == @work_dir_id OR v._key == @start_node
                RETURN {{
                    node: v,
                    edge: e,
                    path: p
                }}
            """

            bind_vars = {
                "depth": depth,
                "start_node": node_id,
                "work_dir_id": work_dir_id,
                "nodes_collection": self.nodes_collection
            }

            # Execute query
            response = requests.post(
                f"{db_url}/_api/cursor",
                json={"query": aql_query, "bindVars": bind_vars},
                auth=self.auth
            )

            if response.status_code == 201:
                result = response.json()

                nodes = {}
                edges = []

                for item in result.get("result", []):
                    # Process node
                    node_data = item["node"]
                    node_id = node_data.get("_key") or node_data.get("node_id")
                    if node_id and node_id not in nodes:
                        # Clean ArangoDB fields
                        node_data.pop("_id", None)
                        node_data.pop("_key", None)
                        node_data.pop("_rev", None)
                        nodes[node_id] = node_data

                    # Process edge if exists
                    if item.get("edge"):
                        edge_data = item["edge"]
                        # Clean ArangoDB fields
                        edge_data.pop("_id", None)
                        edge_data.pop("_key", None)
                        edge_data.pop("_rev", None)
                        edge_data.pop("_from", None)
                        edge_data.pop("_to", None)
                        edges.append(edge_data)

                return {
                    "nodes": list(nodes.values()),
                    "edges": edges
                }
            else:
                logger.error(f"Traversal failed: {response.text}")
                return {"nodes": [], "edges": []}

        except Exception as e:
            logger.error(f"Error finding dependencies: {e}")
            return {"nodes": [], "edges": []}

    async def find_cycles(
        self,
        scope_context: ScopeContext
    ) -> list[list[str]]:
        """
        Find circular dependencies in the graph.

        Args:
            scope_context: Scope context with work_dir

        Returns:
            List of cycles (each cycle is a list of node IDs)
        """
        if not REQUESTS_AVAILABLE:
            return []

        try:
            work_dir_id = scope_context.get_namespace_key()
            db_url = f"{self.base_url}/_db/{self.database}"

            # Query to find cycles
            aql_query = f"""
                FOR node IN {self.nodes_collection}
                FILTER node.work_dir_id == @work_dir_id
                LET cycles = (
                    FOR v, e, p IN 2..10 OUTBOUND
                    CONCAT(@nodes_collection, '/', node._key)
                    {self.edges_collection}
                    FILTER v._key == node._key
                    RETURN p.vertices[*]._key
                )
                FILTER LENGTH(cycles) > 0
                RETURN DISTINCT cycles
            """

            bind_vars = {
                "work_dir_id": work_dir_id,
                "nodes_collection": self.nodes_collection
            }

            # Execute query
            response = requests.post(
                f"{db_url}/_api/cursor",
                json={"query": aql_query, "bindVars": bind_vars},
                auth=self.auth
            )

            if response.status_code == 201:
                result = response.json()
                cycles = []
                for cycle_group in result.get("result", []):
                    for cycle in cycle_group:
                        if cycle not in cycles:
                            cycles.append(cycle)
                return cycles
            else:
                logger.error(f"Cycle detection failed: {response.text}")
                return []

        except Exception as e:
            logger.error(f"Error finding cycles: {e}")
            return []

    async def analyze_impact(
        self,
        scope_context: ScopeContext,
        node_id: str
    ) -> dict[str, Any]:
        """
        Analyze the impact of changes to a node.

        Args:
            scope_context: Scope context with work_dir
            node_id: Node to analyze

        Returns:
            Dictionary with impact analysis results
        """
        # Find all nodes that depend on this node
        dependents = await self.find_dependencies(
            scope_context, node_id, depth=3, direction="inbound"
        )

        # Find all nodes this node depends on
        dependencies = await self.find_dependencies(
            scope_context, node_id, depth=3, direction="outbound"
        )

        # Calculate impact metrics
        direct_impact = len([n for n in dependents["nodes"] if n])
        indirect_impact = len(dependents["edges"])

        return {
            "node_id": node_id,
            "direct_impact": direct_impact,
            "indirect_impact": indirect_impact,
            "affected_nodes": dependents["nodes"],
            "dependencies": dependencies["nodes"],
            "risk_level": "high" if direct_impact > 10 else "medium" if direct_impact > 5 else "low"
        }

    def close(self):
        """Close the connection (no-op for REST API)."""
        pass


# Global instance
_knowledge_graph = None


def get_knowledge_graph() -> ArangoKnowledgeGraph:
    """Get or create the global knowledge graph instance."""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = ArangoKnowledgeGraph()
    return _knowledge_graph
