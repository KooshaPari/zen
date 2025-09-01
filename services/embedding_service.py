#!/usr/bin/env python3
"""
Local Embedding Service for Zen MCP Server

This service provides embeddings using sentence-transformers locally,
compatible with the TEI API format for seamless integration.

Features:
- ARM64/x86_64 compatible
- Multiple model support
- Batch processing
- Caching
- TEI-compatible API
"""

import hashlib
import logging
import os
from typing import Any, Optional, Union

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, will use mock embeddings")


class EmbedRequest(BaseModel):
    """Request model for embeddings."""
    inputs: Union[str, list[str]] = Field(..., description="Text or list of texts to embed")
    model: Optional[str] = Field(None, description="Model to use (ignored, for compatibility)")
    dimensions: Optional[int] = Field(None, description="Output dimensions (for truncation)")
    truncate: Optional[bool] = Field(True, description="Whether to truncate long inputs")


class EmbedResponse(BaseModel):
    """Response model for embeddings (TEI-compatible)."""
    embeddings: Union[list[float], list[list[float]]] = Field(..., description="Embedding vectors")
    model: str = Field(..., description="Model used")
    usage: Optional[dict[str, int]] = Field(None, description="Token usage statistics")


class OpenAIEmbedResponse(BaseModel):
    """OpenAI-compatible embedding response."""
    object: str = "list"
    data: list[dict[str, Any]]
    model: str
    usage: dict[str, int]


class EmbeddingService:
    """Embedding service with caching and multiple model support."""

    def __init__(self, model_name: str = None, cache_size: int = 10000):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformers model to use
            cache_size: Maximum number of embeddings to cache
        """
        # Default models by priority
        DEFAULT_MODELS = [
            "BAAI/bge-small-en-v1.5",  # 384 dimensions, good quality
            "sentence-transformers/all-MiniLM-L6-v2",  # 384 dimensions, fast
            "sentence-transformers/all-mpnet-base-v2",  # 768 dimensions, high quality
        ]

        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODELS[0])
        self.model = None
        self.dimension = None
        self._cache = {}
        self.cache_size = cache_size
        self.total_requests = 0
        self.cache_hits = 0

        if TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            logger.warning("Using mock embeddings (sentence-transformers not installed)")
            self.dimension = 384

    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

            # Get model dimension
            test_embedding = self.model.encode("test", normalize_embeddings=True)
            self.dimension = len(test_embedding)

            logger.info(f"Model loaded successfully. Dimension: {self.dimension}")

        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Try fallback models
            for fallback in ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]:
                if fallback != self.model_name:
                    try:
                        logger.info(f"Trying fallback model: {fallback}")
                        self.model = SentenceTransformer(fallback)
                        self.model_name = fallback
                        test_embedding = self.model.encode("test", normalize_embeddings=True)
                        self.dimension = len(test_embedding)
                        logger.info(f"Fallback model loaded. Dimension: {self.dimension}")
                        break
                    except:
                        continue

            if self.model is None:
                logger.error("No models could be loaded, using mock embeddings")
                self.dimension = 384

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _mock_embedding(self, text: str) -> list[float]:
        """Generate mock embedding for testing."""
        # Use hash to get consistent embeddings for same text
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        vec = np.random.randn(self.dimension)
        vec = vec / np.linalg.norm(vec)  # Normalize
        return vec.tolist()

    def embed_text(self, text: str, dimensions: Optional[int] = None) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            dimensions: Optional output dimensions (truncation)

        Returns:
            Embedding vector
        """
        self.total_requests += 1

        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            self.cache_hits += 1
            embedding = self._cache[cache_key]
        else:
            # Generate embedding
            if self.model is not None:
                embedding = self.model.encode(text, normalize_embeddings=True).tolist()
            else:
                embedding = self._mock_embedding(text)

            # Update cache
            if len(self._cache) < self.cache_size:
                self._cache[cache_key] = embedding

        # Handle dimension adjustment if requested
        if dimensions and dimensions != len(embedding):
            if dimensions < len(embedding):
                # Truncate
                embedding = embedding[:dimensions]
            else:
                # Pad with zeros
                embedding.extend([0.0] * (dimensions - len(embedding)))

        return embedding

    def embed_batch(self, texts: list[str], dimensions: Optional[int] = None) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            dimensions: Optional output dimensions

        Returns:
            List of embedding vectors
        """
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                self.cache_hits += 1
                embeddings.append(self._cache[cache_key])
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        self.total_requests += len(texts)

        # Generate embeddings for uncached texts
        if uncached_texts:
            if self.model is not None:
                new_embeddings = self.model.encode(uncached_texts, normalize_embeddings=True).tolist()
            else:
                new_embeddings = [self._mock_embedding(text) for text in uncached_texts]

            # Fill results and update cache
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                if len(self._cache) < self.cache_size:
                    cache_key = self._get_cache_key(texts[idx])
                    self._cache[cache_key] = embedding

        # Handle dimension adjustment
        if dimensions:
            adjusted = []
            for embedding in embeddings:
                if dimensions < len(embedding):
                    adjusted.append(embedding[:dimensions])
                elif dimensions > len(embedding):
                    padded = embedding.copy()
                    padded.extend([0.0] * (dimensions - len(embedding)))
                    adjusted.append(padded)
                else:
                    adjusted.append(embedding)
            embeddings = adjusted

        return embeddings

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        return {
            "model": self.model_name,
            "dimension": self.dimension,
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_size": len(self._cache),
            "cache_hit_rate": self.cache_hits / max(1, self.total_requests),
            "model_loaded": self.model is not None
        }


# Create FastAPI app
app = FastAPI(title="Zen Embedding Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding service
embedding_service = EmbeddingService()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Zen Embedding Service",
        "model": embedding_service.model_name,
        "dimension": embedding_service.dimension,
        "api_endpoints": ["/embed", "/embeddings", "/health", "/stats"]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": embedding_service.model is not None,
        "stats": embedding_service.get_stats()
    }


@app.get("/stats")
async def stats():
    """Get service statistics."""
    return embedding_service.get_stats()


@app.post("/embed")
async def embed_tei(request: EmbedRequest):
    """
    TEI-compatible embedding endpoint.

    Returns embeddings in TEI format.
    """
    try:
        if isinstance(request.inputs, str):
            # Single text
            embedding = embedding_service.embed_text(request.inputs, request.dimensions)
            return {
                "embeddings": embedding,
                "model": embedding_service.model_name
            }
        else:
            # Batch of texts
            embeddings = embedding_service.embed_batch(request.inputs, request.dimensions)
            return {
                "embeddings": embeddings,
                "model": embedding_service.model_name
            }

    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings")
async def embed_openai(request: EmbedRequest):
    """
    OpenAI-compatible embedding endpoint.

    Returns embeddings in OpenAI format.
    """
    try:
        if isinstance(request.inputs, str):
            texts = [request.inputs]
        else:
            texts = request.inputs

        embeddings = embedding_service.embed_batch(texts, request.dimensions)

        # Format as OpenAI response
        data = []
        for i, embedding in enumerate(embeddings):
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding
            })

        return {
            "object": "list",
            "data": data,
            "model": embedding_service.model_name,
            "usage": {
                "prompt_tokens": sum(len(text.split()) for text in texts),
                "total_tokens": sum(len(text.split()) for text in texts)
            }
        }

    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings")
async def embed_openai_v1(request: EmbedRequest):
    """OpenAI v1 API compatible endpoint (alias)."""
    return await embed_openai(request)


if __name__ == "__main__":
    # Run the service
    port = int(os.getenv("EMBEDDING_PORT", "8090"))
    host = os.getenv("EMBEDDING_HOST", "0.0.0.0")

    logger.info(f"Starting embedding service on {host}:{port}")
    logger.info(f"Model: {embedding_service.model_name}")
    logger.info(f"Dimension: {embedding_service.dimension}")

    uvicorn.run(app, host=host, port=port)
