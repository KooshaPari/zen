"""
Vector Store Integration with PostgreSQL pgvector

This module provides the vector storage and retrieval infrastructure for the
Zen MCP Server, enabling semantic search, RAG/CAG workflows, and memory persistence
using PostgreSQL with the pgvector extension.

Key Features:
- Multi-tenant vector storage with work_dir scoping
- Hybrid search (vector + text/BM25)
- Collection management (code, knowledge, memory)
- Batch embedding and ingestion
- Similarity search with filtering
- Integration with conversation and agent memory
- Performance optimization with caching
"""

import asyncio
import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

import numpy as np

try:
    import asyncpg
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor
    from psycopg2.pool import SimpleConnectionPool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    asyncpg = None
    psycopg2 = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

from pydantic import BaseModel, Field

from utils.scope_utils import ScopeContext

logger = logging.getLogger(__name__)


class VectorCollection(BaseModel):
    """Represents a vector collection in the database."""

    collection_id: str = Field(default_factory=lambda: str(uuid4()))
    work_dir_id: str = Field(..., description="Work directory ID")
    collection_name: str = Field(..., description="Collection name")
    collection_type: str = Field(..., description="Collection type (code, knowledge, memory, other)")
    dimension: int = Field(default=384, description="Vector dimension")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorDocument(BaseModel):
    """Represents a document to be vectorized and stored."""

    content: str = Field(..., description="Document content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    doc_id: Optional[str] = Field(default=None, description="Optional document ID")
    embedding: Optional[list[float]] = Field(default=None, description="Pre-computed embedding")


class VectorSearchResult(BaseModel):
    """Result from vector similarity search."""

    doc_id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    similarity: float = Field(..., description="Similarity score (0-1)")
    metadata: dict[str, Any] = Field(default_factory=dict)
    distance: Optional[float] = Field(default=None, description="Vector distance")


class EmbeddingProvider:
    """Handles embedding generation from various sources."""

    def __init__(self, provider: str = None, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize embedding provider.

        Args:
            provider: Provider type (local, openai, openrouter, auto)
            api_url: API endpoint URL
            api_key: API key for authentication
        """
        # Auto-detect provider based on environment variables
        if provider is None or provider == "auto":
            if os.getenv("OPENROUTER_API_KEY"):
                provider = "openrouter"
                api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            elif os.getenv("OPENAI_API_KEY"):
                provider = "openai"
                api_key = api_key or os.getenv("OPENAI_API_KEY")
            else:
                provider = "local"

        self.provider = provider
        self.api_url = api_url or os.getenv("TEI_API_URL", "http://localhost:8090")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv(f"{provider.upper()}_API_KEY")

        # Dimension based on provider/model
        if provider == "openrouter":
            # OpenRouter models can have different dimensions
            model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
            if "3-small" in model:
                self.dimension = 384  # Can be configured up to 1536
            elif "3-large" in model:
                self.dimension = 1024  # Can be configured up to 3072
            else:
                self.dimension = 384  # Default
        else:
            self.dimension = 384  # BGE-small dimension

        # Cache for embeddings
        self._cache = {}
        self._cache_size = 1000

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        embedding = await self._generate_embedding(text)

        # Update cache
        if len(self._cache) < self._cache_size:
            self._cache[cache_key] = embedding

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        # Check cache and separate cached vs uncached
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self._cache:
                embeddings.append(self._cache[cache_key])
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = await self._generate_batch_embeddings(uncached_texts)

            # Fill in the results and update cache
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                if len(self._cache) < self._cache_size:
                    cache_key = hashlib.md5(texts[idx].encode()).hexdigest()
                    self._cache[cache_key] = embedding

        return embeddings

    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using the configured provider."""
        if self.provider == "local":
            return await self._local_embedding(text)
        elif self.provider == "openai":
            return await self._openai_embedding(text)
        elif self.provider == "openrouter":
            return await self._openrouter_embedding(text)
        else:
            # Fallback to random embedding for testing
            return self._random_embedding()

    async def _generate_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate batch embeddings using the configured provider."""
        if self.provider == "local":
            return await self._local_batch_embeddings(texts)
        else:
            # Fall back to individual embeddings
            tasks = [self._generate_embedding(text) for text in texts]
            return await asyncio.gather(*tasks)

    async def _local_embedding(self, text: str) -> list[float]:
        """Generate embedding using local TEI server."""
        if not REQUESTS_AVAILABLE:
            return self._random_embedding()

        try:
            response = requests.post(
                f"{self.api_url}/embed",
                json={"inputs": text},
                headers={"Content-Type": "application/json"},
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0] if isinstance(result[0], list) else result

            logger.warning(f"TEI embedding failed: {response.status_code}")
            return self._random_embedding()

        except Exception as e:
            logger.warning(f"TEI embedding error: {e}")
            return self._random_embedding()

    async def _local_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate batch embeddings using local TEI server."""
        if not REQUESTS_AVAILABLE:
            return [self._random_embedding() for _ in texts]

        try:
            response = requests.post(
                f"{self.api_url}/embed",
                json={"inputs": texts},
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) == len(texts):
                    return result

            logger.warning(f"TEI batch embedding failed: {response.status_code}")
            return [self._random_embedding() for _ in texts]

        except Exception as e:
            logger.warning(f"TEI batch embedding error: {e}")
            return [self._random_embedding() for _ in texts]

    async def _openai_embedding(self, text: str) -> list[float]:
        """Generate embedding using OpenAI API."""
        # TODO: Implement OpenAI embedding when API key is available
        return self._random_embedding()

    async def _openrouter_embedding(self, text: str) -> list[float]:
        """Generate embedding using OpenRouter API."""
        if not REQUESTS_AVAILABLE:
            return self._random_embedding()

        if not self.api_key:
            logger.warning("OpenRouter API key not set, using random embeddings")
            return self._random_embedding()

        try:
            # OpenRouter embedding models (as of 2025)
            # Options: openai/text-embedding-3-small, openai/text-embedding-3-large,
            #          voyage/voyage-3, cohere/embed-english-v3.0
            model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")

            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                json={
                    "model": model,
                    "input": text,
                    "dimensions": self.dimension  # Request specific dimensions
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/zen-mcp-server"),
                    "X-Title": os.getenv("OPENROUTER_TITLE", "Zen MCP Server")
                },
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                if "data" in result and len(result["data"]) > 0:
                    embedding = result["data"][0].get("embedding", [])

                    # Ensure correct dimension
                    if len(embedding) != self.dimension:
                        # Truncate or pad as needed
                        if len(embedding) > self.dimension:
                            embedding = embedding[:self.dimension]
                        else:
                            embedding.extend([0.0] * (self.dimension - len(embedding)))

                    return embedding

            logger.warning(f"OpenRouter embedding failed: {response.status_code} - {response.text[:200]}")
            return self._random_embedding()

        except Exception as e:
            logger.warning(f"OpenRouter embedding error: {e}")
            return self._random_embedding()

    def _random_embedding(self) -> list[float]:
        """Generate random embedding for testing."""
        vec = np.random.randn(self.dimension)
        vec = vec / np.linalg.norm(vec)  # Normalize
        return vec.tolist()


class PgVectorStore:
    """
    PostgreSQL pgvector store for vector storage and retrieval.

    This class manages vector collections, documents, and similarity search
    with multi-tenant support via work_dir scoping.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        pool_size: int = 10,
        embedding_provider: Optional[EmbeddingProvider] = None
    ):
        """
        Initialize the vector store.

        Args:
            connection_string: PostgreSQL connection string
            pool_size: Connection pool size
            embedding_provider: Provider for generating embeddings
        """
        self.connection_string = connection_string or os.getenv(
            "POSTGRES_VECTOR_URL",
            "postgresql://zen_user:zen_secure_pass_2025@localhost:5433/zen_vector"
        )
        self.pool_size = pool_size
        self.embedding_provider = embedding_provider or EmbeddingProvider()

        # Connection pool
        self.pool = None
        self.async_pool = None

        # Cache for collections
        self._collection_cache = {}

        # Initialize connection
        self._init_connection()

    def _init_connection(self):
        """Initialize database connection pool."""
        if not POSTGRES_AVAILABLE:
            logger.warning("PostgreSQL libraries not available, vector store disabled")
            return

        try:
            self.pool = SimpleConnectionPool(
                1,
                self.pool_size,
                self.connection_string
            )
            logger.info("PgVector connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PgVector connection: {e}")
            self.pool = None

    def _get_connection(self):
        """Get a connection from the pool."""
        if self.pool:
            return self.pool.getconn()
        return None

    def _put_connection(self, conn):
        """Return a connection to the pool."""
        if self.pool and conn:
            self.pool.putconn(conn)

    async def get_or_create_collection(
        self,
        scope_context: ScopeContext,
        collection_name: str,
        collection_type: str = "other"
    ) -> Optional[VectorCollection]:
        """
        Get or create a vector collection for the given scope.

        Args:
            scope_context: Scope context with work_dir and identity
            collection_name: Name of the collection
            collection_type: Type of collection (code, knowledge, memory, other)

        Returns:
            VectorCollection object or None if failed
        """
        # Generate cache key
        cache_key = f"{scope_context.get_namespace_key()}:{collection_name}"

        # Check cache
        if cache_key in self._collection_cache:
            return self._collection_cache[cache_key]

        conn = self._get_connection()
        if not conn:
            return None

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get or create organization
                cur.execute("""
                    INSERT INTO zen_shared.organizations (org_name)
                    VALUES (%s)
                    ON CONFLICT (org_name) DO UPDATE SET org_name = EXCLUDED.org_name
                    RETURNING org_id
                """, (scope_context.org_id or "default",))
                org_id = cur.fetchone()["org_id"]

                # Get or create project
                cur.execute("""
                    INSERT INTO zen_shared.projects (org_id, project_name)
                    VALUES (%s, %s)
                    ON CONFLICT (org_id, project_name) DO UPDATE SET project_name = EXCLUDED.project_name
                    RETURNING project_id
                """, (org_id, scope_context.project_id or "default"))
                project_id = cur.fetchone()["project_id"]

                # Get or create work_dir
                cur.execute("""
                    INSERT INTO zen_shared.work_dirs (project_id, work_dir_path)
                    VALUES (%s, %s)
                    ON CONFLICT (project_id, work_dir_path) DO UPDATE SET work_dir_path = EXCLUDED.work_dir_path
                    RETURNING work_dir_id
                """, (project_id, scope_context.work_dir or "/"))
                work_dir_id = cur.fetchone()["work_dir_id"]

                # Get or create collection
                cur.execute("""
                    INSERT INTO zen_shared.vector_collections
                    (work_dir_id, collection_name, collection_type, dimension)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (work_dir_id, collection_name)
                    DO UPDATE SET collection_type = EXCLUDED.collection_type
                    RETURNING *
                """, (work_dir_id, collection_name, collection_type, self.embedding_provider.dimension))

                collection_data = cur.fetchone()
                conn.commit()

                collection = VectorCollection(
                    collection_id=str(collection_data["collection_id"]),
                    work_dir_id=str(work_dir_id),
                    collection_name=collection_name,
                    collection_type=collection_type,
                    dimension=collection_data["dimension"],
                    created_at=collection_data["created_at"],
                    metadata=collection_data["metadata"] or {}
                )

                # Update cache
                self._collection_cache[cache_key] = collection

                return collection

        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            conn.rollback()
            return None
        finally:
            self._put_connection(conn)

    async def add_documents(
        self,
        scope_context: ScopeContext,
        collection_name: str,
        documents: list[VectorDocument],
        collection_type: str = "other"
    ) -> bool:
        """
        Add documents to a collection.

        Args:
            scope_context: Scope context
            collection_name: Collection name
            documents: List of documents to add
            collection_type: Type of collection

        Returns:
            True if successful, False otherwise
        """
        # Get or create collection
        collection = await self.get_or_create_collection(
            scope_context, collection_name, collection_type
        )
        if not collection:
            return False

        # Generate embeddings for documents without them
        texts_to_embed = []
        embed_indices = []

        for i, doc in enumerate(documents):
            if doc.embedding is None:
                texts_to_embed.append(doc.content)
                embed_indices.append(i)

        if texts_to_embed:
            embeddings = await self.embedding_provider.embed_batch(texts_to_embed)
            for i, idx in enumerate(embed_indices):
                documents[idx].embedding = embeddings[i]

        # Insert documents
        conn = self._get_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                for doc in documents:
                    doc_id = doc.doc_id or str(uuid4())

                    # Convert embedding to PostgreSQL array format
                    embedding_str = '[' + ','.join(map(str, doc.embedding)) + ']'

                    cur.execute("""
                        INSERT INTO zen_shared.vectors
                        (vector_id, collection_id, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s::vector, %s)
                        ON CONFLICT (vector_id) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            updated_at = NOW()
                    """, (
                        doc_id,
                        collection.collection_id,
                        doc.content,
                        embedding_str,
                        Json(doc.metadata)
                    ))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            conn.rollback()
            return False
        finally:
            self._put_connection(conn)

    async def search(
        self,
        scope_context: ScopeContext,
        collection_name: str,
        query: str,
        limit: int = 10,
        threshold: float = 0.7,
        filter_metadata: Optional[dict[str, Any]] = None
    ) -> list[VectorSearchResult]:
        """
        Search for similar documents in a collection.

        Args:
            scope_context: Scope context
            collection_name: Collection name
            query: Query text
            limit: Maximum number of results
            threshold: Similarity threshold (0-1)
            filter_metadata: Optional metadata filters

        Returns:
            List of search results
        """
        # Get collection
        cache_key = f"{scope_context.get_namespace_key()}:{collection_name}"
        collection = self._collection_cache.get(cache_key)

        if not collection:
            collection = await self.get_or_create_collection(
                scope_context, collection_name
            )
            if not collection:
                return []

        # Generate query embedding
        query_embedding = await self.embedding_provider.embed_text(query)
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        conn = self._get_connection()
        if not conn:
            return []

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build query with optional metadata filter
                sql = """
                    SELECT
                        vector_id,
                        content,
                        1 - (embedding <=> %s::vector) AS similarity,
                        metadata
                    FROM zen_shared.vectors
                    WHERE collection_id = %s
                        AND 1 - (embedding <=> %s::vector) >= %s
                """

                params = [embedding_str, collection.collection_id, embedding_str, threshold]

                if filter_metadata:
                    sql += " AND metadata @> %s::jsonb"
                    params.append(Json(filter_metadata))

                sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
                params.extend([embedding_str, limit])

                cur.execute(sql, params)
                rows = cur.fetchall()

                results = []
                for row in rows:
                    results.append(VectorSearchResult(
                        doc_id=str(row["vector_id"]),
                        content=row["content"],
                        similarity=float(row["similarity"]),
                        metadata=row["metadata"] or {}
                    ))

                return results

        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
        finally:
            self._put_connection(conn)

    async def hybrid_search(
        self,
        scope_context: ScopeContext,
        collection_name: str,
        query: str,
        text_query: Optional[str] = None,
        limit: int = 10,
        vector_weight: float = 0.7
    ) -> list[VectorSearchResult]:
        """
        Perform hybrid search combining vector and text search.

        Args:
            scope_context: Scope context
            collection_name: Collection name
            query: Query for vector search
            text_query: Optional separate text query
            limit: Maximum number of results
            vector_weight: Weight for vector similarity (0-1)

        Returns:
            List of search results
        """
        # Get collection
        cache_key = f"{scope_context.get_namespace_key()}:{collection_name}"
        collection = self._collection_cache.get(cache_key)

        if not collection:
            collection = await self.get_or_create_collection(
                scope_context, collection_name
            )
            if not collection:
                return []

        # Use text_query if provided, otherwise use query
        text_query = text_query or query

        # Generate query embedding
        query_embedding = await self.embedding_provider.embed_text(query)
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        conn = self._get_connection()
        if not conn:
            return []

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM zen_shared.hybrid_search(%s, %s::vector, %s, %s, %s)
                """, (
                    collection.collection_id,
                    embedding_str,
                    text_query,
                    limit,
                    vector_weight
                ))

                rows = cur.fetchall()

                results = []
                for row in rows:
                    results.append(VectorSearchResult(
                        doc_id=str(row["vector_id"]),
                        content=row["content"],
                        similarity=float(row["combined_score"]),
                        metadata=row["metadata"] or {}
                    ))

                return results

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fall back to vector-only search
            return await self.search(
                scope_context, collection_name, query, limit
            )
        finally:
            self._put_connection(conn)

    async def delete_collection(
        self,
        scope_context: ScopeContext,
        collection_name: str
    ) -> bool:
        """Delete a collection and all its documents."""
        cache_key = f"{scope_context.get_namespace_key()}:{collection_name}"
        collection = self._collection_cache.get(cache_key)

        if not collection:
            return True  # Already doesn't exist

        conn = self._get_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM zen_shared.vector_collections
                    WHERE collection_id = %s
                """, (collection.collection_id,))
                conn.commit()

                # Remove from cache
                del self._collection_cache[cache_key]

                return True

        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            conn.rollback()
            return False
        finally:
            self._put_connection(conn)

    def close(self):
        """Close all connections."""
        if self.pool:
            self.pool.closeall()


# Global instance
_vector_store = None


def get_vector_store() -> PgVectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = PgVectorStore()
    return _vector_store
