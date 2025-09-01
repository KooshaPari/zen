"""
Model Performance Tracker with Cost, Time, and Token Optimization

This module provides comprehensive tracking and optimization for AI model selection,
considering multiple factors:
- Cost per token (input/output pricing)
- Response time and latency
- Token usage (input/output counts)
- Success/failure rates
- Quality scores
- Task-specific performance patterns

The system learns from historical data to make optimal model selections based on
configurable optimization strategies (cost, speed, quality, balanced).
"""

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2.pool import SimpleConnectionPool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None

from pydantic import BaseModel, Field

from utils.scope_utils import ScopeContext
from utils.vector_store import EmbeddingProvider

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Model selection optimization strategies."""
    COST = "cost"  # Minimize cost
    SPEED = "speed"  # Minimize latency
    QUALITY = "quality"  # Maximize quality/success
    BALANCED = "balanced"  # Balance all factors
    THROUGHPUT = "throughput"  # Maximize tokens/second
    COST_QUALITY = "cost_quality"  # Best quality within budget
    ADAPTIVE = "adaptive"  # Learn from patterns


class ModelPricing(BaseModel):
    """Pricing information for a model."""

    model_name: str = Field(..., description="Model identifier")
    provider: str = Field(..., description="Provider name")

    # Pricing per 1M tokens (industry standard)
    input_price_per_1m: float = Field(..., description="Input token price per 1M tokens")
    output_price_per_1m: float = Field(..., description="Output token price per 1M tokens")

    # Optional tiered pricing
    pricing_tiers: Optional[list[dict[str, Any]]] = Field(None, description="Volume-based pricing tiers")

    # Cache pricing (if applicable)
    cache_write_price_per_1m: Optional[float] = Field(None, description="Cache write price")
    cache_read_price_per_1m: Optional[float] = Field(None, description="Cache read price")

    # Additional costs
    request_price: Optional[float] = Field(0.0, description="Fixed price per request")

    # Currency
    currency: str = Field("USD", description="Pricing currency")

    # Metadata
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def calculate_cost(self, input_tokens: int, output_tokens: int,
                      cached_tokens: int = 0) -> float:
        """Calculate total cost for token usage."""
        input_cost = (input_tokens / 1_000_000) * self.input_price_per_1m
        output_cost = (output_tokens / 1_000_000) * self.output_price_per_1m

        # Apply cache discount if applicable
        if cached_tokens > 0 and self.cache_read_price_per_1m:
            cache_discount = (cached_tokens / 1_000_000) * (
                self.input_price_per_1m - self.cache_read_price_per_1m
            )
            input_cost -= cache_discount

        return input_cost + output_cost + (self.request_price or 0)


@dataclass
class ModelPerformanceMetrics:
    """Comprehensive metrics for model performance."""

    # Identification
    model_name: str
    provider: str
    task_type: str
    task_hash: str  # Hash of task characteristics

    # Token metrics
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0

    # Time metrics (in seconds)
    first_token_time: float = 0.0  # Time to first token
    total_time: float = 0.0  # Total generation time
    queue_time: float = 0.0  # Time spent in queue

    # Cost metrics
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    cost_per_token: float = 0.0

    # Performance metrics
    tokens_per_second: float = 0.0
    success: bool = True
    error_type: Optional[str] = None
    quality_score: float = 1.0  # 0-1, subjective quality

    # Context
    context_length: int = 0
    max_context: int = 0
    temperature: float = 0.7

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = None
    request_id: Optional[str] = None

    def calculate_efficiency_score(self, weights: dict[str, float] = None) -> float:
        """Calculate overall efficiency score based on weighted factors."""
        if weights is None:
            weights = {
                "cost": 0.3,
                "speed": 0.3,
                "quality": 0.3,
                "throughput": 0.1
            }

        # Normalize metrics to 0-1 scale
        cost_score = 1.0 / (1.0 + self.cost_per_token * 1000)  # Lower cost = higher score
        speed_score = 1.0 / (1.0 + self.total_time)  # Faster = higher score
        quality_score = self.quality_score
        throughput_score = min(self.tokens_per_second / 100, 1.0)  # Cap at 100 tps

        return (
            weights.get("cost", 0.3) * cost_score +
            weights.get("speed", 0.3) * speed_score +
            weights.get("quality", 0.3) * quality_score +
            weights.get("throughput", 0.1) * throughput_score
        )


class ModelPerformanceTracker:
    """
    Tracks and optimizes model performance across multiple dimensions.

    Features:
    - Real-time performance tracking
    - Historical analysis with pgvector
    - Cost optimization
    - Adaptive model selection
    - A/B testing support
    - Anomaly detection
    """

    # Default model pricing (as of 2025)
    DEFAULT_PRICING = {
        # Google models
        "gemini-2.0-flash-exp": ModelPricing(
            model_name="gemini-2.0-flash-exp",
            provider="google",
            input_price_per_1m=0.0,  # Free during experimental
            output_price_per_1m=0.0
        ),
        "gemini-1.5-flash": ModelPricing(
            model_name="gemini-1.5-flash",
            provider="google",
            input_price_per_1m=0.075,
            output_price_per_1m=0.30,
            cache_read_price_per_1m=0.01875
        ),
        "gemini-1.5-flash-8b": ModelPricing(
            model_name="gemini-1.5-flash-8b",
            provider="google",
            input_price_per_1m=0.0375,
            output_price_per_1m=0.15,
            cache_read_price_per_1m=0.01
        ),
        "gemini-1.5-pro": ModelPricing(
            model_name="gemini-1.5-pro",
            provider="google",
            input_price_per_1m=1.25,
            output_price_per_1m=5.00,
            cache_read_price_per_1m=0.3125
        ),

        # OpenAI models
        "gpt-4o": ModelPricing(
            model_name="gpt-4o",
            provider="openai",
            input_price_per_1m=2.50,
            output_price_per_1m=10.00,
            cache_read_price_per_1m=1.25
        ),
        "gpt-4o-mini": ModelPricing(
            model_name="gpt-4o-mini",
            provider="openai",
            input_price_per_1m=0.15,
            output_price_per_1m=0.60,
            cache_read_price_per_1m=0.075
        ),
        "o1-preview": ModelPricing(
            model_name="o1-preview",
            provider="openai",
            input_price_per_1m=15.00,
            output_price_per_1m=60.00
        ),
        "o1-mini": ModelPricing(
            model_name="o1-mini",
            provider="openai",
            input_price_per_1m=3.00,
            output_price_per_1m=12.00
        ),

        # Anthropic models
        "claude-3-5-sonnet-20241022": ModelPricing(
            model_name="claude-3-5-sonnet-20241022",
            provider="anthropic",
            input_price_per_1m=3.00,
            output_price_per_1m=15.00,
            cache_read_price_per_1m=0.30
        ),
        "claude-3-5-haiku-20241022": ModelPricing(
            model_name="claude-3-5-haiku-20241022",
            provider="anthropic",
            input_price_per_1m=1.00,
            output_price_per_1m=5.00
        ),

        # OpenRouter models (includes routing fee)
        "deepseek/deepseek-chat": ModelPricing(
            model_name="deepseek/deepseek-chat",
            provider="openrouter",
            input_price_per_1m=0.14,
            output_price_per_1m=0.28
        ),
        "google/gemini-2.0-flash-thinking-exp": ModelPricing(
            model_name="google/gemini-2.0-flash-thinking-exp",
            provider="openrouter",
            input_price_per_1m=0.0,  # Free experimental
            output_price_per_1m=0.0
        ),
        "qwen/qwen-2.5-coder-32b-instruct": ModelPricing(
            model_name="qwen/qwen-2.5-coder-32b-instruct",
            provider="openrouter",
            input_price_per_1m=0.18,
            output_price_per_1m=0.18
        ),
    }

    def __init__(
        self,
        connection_string: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    ):
        """
        Initialize the performance tracker.

        Args:
            connection_string: PostgreSQL connection string
            embedding_provider: Provider for task embeddings
            optimization_strategy: Default optimization strategy
        """
        self.connection_string = connection_string or os.getenv(
            "POSTGRES_VECTOR_URL",
            "postgresql://zen_user:zen_secure_pass_2025@localhost:5433/zen_vector"
        )
        self.embedding_provider = embedding_provider or EmbeddingProvider()
        self.optimization_strategy = optimization_strategy

        # Connection pool
        self.pool = None

        # In-memory caches
        self.pricing_cache = dict(self.DEFAULT_PRICING)
        self.performance_cache = {}  # Recent performance data
        self.model_scores = {}  # Calculated efficiency scores

        # Performance window
        self.performance_window = timedelta(days=30)  # Consider last 30 days

        # Initialize connection
        self._init_connection()

    def _init_connection(self):
        """Initialize database connection pool."""
        if not POSTGRES_AVAILABLE:
            logger.warning("PostgreSQL not available, performance tracking disabled")
            return

        try:
            self.pool = SimpleConnectionPool(
                1, 10, self.connection_string
            )
            logger.info("Performance tracker connection initialized")

            # Ensure tables exist
            self._create_tables()

        except Exception as e:
            logger.error(f"Failed to initialize performance tracker: {e}")
            self.pool = None

    def _create_tables(self):
        """Create performance tracking tables if they don't exist."""
        if not self.pool:
            return

        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                # Model performance table
                # Use env-configured embedding dimension to align with RAG vectors
                try:
                    _dim = int(os.getenv("RAG_VECTOR_DIM", "768"))
                except Exception:
                    _dim = 768

                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS model_performance (
                        performance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        model_name VARCHAR(255) NOT NULL,
                        provider VARCHAR(100) NOT NULL,
                        task_type VARCHAR(100),
                        task_hash VARCHAR(64),
                        task_embedding vector({_dim}),

                        -- Token metrics
                        input_tokens INTEGER,
                        output_tokens INTEGER,
                        cached_tokens INTEGER DEFAULT 0,
                        total_tokens INTEGER,

                        -- Time metrics (milliseconds)
                        first_token_time_ms INTEGER,
                        total_time_ms INTEGER,
                        queue_time_ms INTEGER DEFAULT 0,

                        -- Cost metrics (in cents for precision)
                        input_cost_cents NUMERIC(10,4),
                        output_cost_cents NUMERIC(10,4),
                        total_cost_cents NUMERIC(10,4),

                        -- Performance metrics
                        tokens_per_second NUMERIC(10,2),
                        success BOOLEAN DEFAULT true,
                        error_type VARCHAR(100),
                        quality_score NUMERIC(3,2) DEFAULT 1.0,

                        -- Context
                        context_length INTEGER,
                        max_context INTEGER,
                        temperature NUMERIC(3,2),

                        -- Metadata
                        work_dir_id VARCHAR(255),
                        session_id VARCHAR(255),
                        request_id VARCHAR(255),
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

                        -- Indexes
                        INDEX idx_model_performance_model (model_name, provider),
                        INDEX idx_model_performance_task (task_type, task_hash),
                        INDEX idx_model_performance_time (timestamp),
                        INDEX idx_model_performance_work_dir (work_dir_id)
                    )
                """)

                # Model pricing table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_pricing (
                        pricing_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        model_name VARCHAR(255) NOT NULL,
                        provider VARCHAR(100) NOT NULL,

                        -- Pricing per 1M tokens (in cents)
                        input_price_cents NUMERIC(10,4),
                        output_price_cents NUMERIC(10,4),
                        cache_write_price_cents NUMERIC(10,4),
                        cache_read_price_cents NUMERIC(10,4),
                        request_price_cents NUMERIC(10,4) DEFAULT 0,

                        -- Metadata
                        currency VARCHAR(3) DEFAULT 'USD',
                        effective_date DATE,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

                        -- Unique constraint
                        UNIQUE(model_name, provider, effective_date)
                    )
                """)

                # Model optimization scores (pre-calculated)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_optimization_scores (
                        score_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        model_name VARCHAR(255) NOT NULL,
                        provider VARCHAR(100) NOT NULL,
                        task_type VARCHAR(100),
                        optimization_strategy VARCHAR(50),

                        -- Scores
                        efficiency_score NUMERIC(3,2),
                        cost_score NUMERIC(3,2),
                        speed_score NUMERIC(3,2),
                        quality_score NUMERIC(3,2),
                        throughput_score NUMERIC(3,2),

                        -- Statistics
                        sample_count INTEGER,
                        success_rate NUMERIC(3,2),
                        avg_latency_ms INTEGER,
                        avg_cost_cents NUMERIC(10,4),

                        -- Metadata
                        calculated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        valid_until TIMESTAMP WITH TIME ZONE,

                        -- Indexes
                        INDEX idx_optimization_model (model_name, provider),
                        INDEX idx_optimization_strategy (optimization_strategy, task_type)
                    )
                """)

                conn.commit()
                logger.info("Performance tracking tables created/verified")

        except Exception as e:
            logger.error(f"Error creating performance tables: {e}")
            conn.rollback()
        finally:
            self.pool.putconn(conn)

    async def track_performance(
        self,
        metrics: ModelPerformanceMetrics,
        scope_context: Optional[ScopeContext] = None
    ) -> bool:
        """
        Track model performance metrics.

        Args:
            metrics: Performance metrics to track
            scope_context: Optional scope context for multi-tenancy

        Returns:
            True if successfully tracked
        """
        if not self.pool:
            # Store in memory cache if DB unavailable
            cache_key = f"{metrics.model_name}:{metrics.task_hash}"
            if cache_key not in self.performance_cache:
                self.performance_cache[cache_key] = []
            self.performance_cache[cache_key].append(metrics)
            return True

        conn = self.pool.getconn()
        try:
            # Generate task embedding if not cached
            task_embedding = await self._get_task_embedding(metrics.task_type, metrics.task_hash)

            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO model_performance (
                        model_name, provider, task_type, task_hash, task_embedding,
                        input_tokens, output_tokens, cached_tokens, total_tokens,
                        first_token_time_ms, total_time_ms, queue_time_ms,
                        input_cost_cents, output_cost_cents, total_cost_cents,
                        tokens_per_second, success, error_type, quality_score,
                        context_length, max_context, temperature,
                        work_dir_id, session_id, request_id
                    ) VALUES (
                        %s, %s, %s, %s, %s::vector,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s
                    )
                """, (
                    metrics.model_name, metrics.provider, metrics.task_type,
                    metrics.task_hash, task_embedding,
                    metrics.input_tokens, metrics.output_tokens, metrics.cached_tokens,
                    metrics.total_tokens,
                    int(metrics.first_token_time * 1000),
                    int(metrics.total_time * 1000),
                    int(metrics.queue_time * 1000),
                    metrics.input_cost * 100,  # Convert to cents
                    metrics.output_cost * 100,
                    metrics.total_cost * 100,
                    metrics.tokens_per_second, metrics.success, metrics.error_type,
                    metrics.quality_score,
                    metrics.context_length, metrics.max_context, metrics.temperature,
                    scope_context.get_namespace_key() if scope_context else None,
                    metrics.session_id, metrics.request_id
                ))

                conn.commit()

                # Update optimization scores asynchronously
                asyncio.create_task(self._update_optimization_scores(
                    metrics.model_name, metrics.provider, metrics.task_type
                ))

                return True

        except Exception as e:
            logger.error(f"Error tracking performance: {e}")
            conn.rollback()
            return False
        finally:
            self.pool.putconn(conn)

    async def select_optimal_model(
        self,
        task_type: str,
        task_description: str,
        context_length: int,
        optimization_strategy: Optional[OptimizationStrategy] = None,
        budget_cents: Optional[float] = None,
        max_latency_ms: Optional[int] = None,
        min_quality: Optional[float] = None,
        scope_context: Optional[ScopeContext] = None
    ) -> tuple[str, dict[str, Any]]:
        """
        Select optimal model based on requirements and historical performance.

        Args:
            task_type: Type of task (e.g., "code_review", "chat", "analysis")
            task_description: Description for embedding similarity
            context_length: Required context window size
            optimization_strategy: Override default strategy
            budget_cents: Maximum budget in cents
            max_latency_ms: Maximum acceptable latency
            min_quality: Minimum quality score (0-1)
            scope_context: Scope context for filtering

        Returns:
            Tuple of (model_name, selection_metadata)
        """
        strategy = optimization_strategy or self.optimization_strategy

        # Get candidate models
        candidates = await self._get_candidate_models(
            task_type, context_length, scope_context
        )

        if not candidates:
            # Fallback to default based on context size
            if context_length > 128000:
                return "gemini-1.5-pro", {"reason": "large_context_fallback"}
            elif context_length > 32000:
                return "gemini-1.5-flash", {"reason": "medium_context_fallback"}
            else:
                return "gemini-2.0-flash-exp", {"reason": "default_fallback"}

        # Score candidates based on strategy
        scored_models = []
        task_embedding = await self._get_task_embedding(task_type, task_description)

        for model in candidates:
            score = await self._score_model(
                model, task_embedding, strategy,
                budget_cents, max_latency_ms, min_quality
            )
            if score > 0:
                scored_models.append((model["model_name"], score, model))

        if not scored_models:
            return candidates[0]["model_name"], {"reason": "no_viable_candidates"}

        # Sort by score and return best
        scored_models.sort(key=lambda x: x[1], reverse=True)
        best_model, best_score, metadata = scored_models[0]

        return best_model, {
            "score": best_score,
            "strategy": strategy.value,
            "candidates_evaluated": len(candidates),
            "avg_cost": metadata.get("avg_cost_cents", 0) / 100,
            "avg_latency_ms": metadata.get("avg_latency_ms", 0),
            "success_rate": metadata.get("success_rate", 1.0),
            "reason": "optimization_selection"
        }

    async def _get_candidate_models(
        self,
        task_type: str,
        context_length: int,
        scope_context: Optional[ScopeContext] = None
    ) -> list[dict[str, Any]]:
        """Get candidate models based on requirements."""
        if not self.pool:
            # Return static list if DB unavailable
            return [
                {"model_name": "gemini-2.0-flash-exp", "provider": "google"},
                {"model_name": "gemini-1.5-flash", "provider": "google"},
                {"model_name": "gpt-4o-mini", "provider": "openai"},
            ]

        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get models with recent performance data
                query = """
                    SELECT
                        model_name,
                        provider,
                        AVG(total_cost_cents) as avg_cost_cents,
                        AVG(total_time_ms) as avg_latency_ms,
                        AVG(tokens_per_second) as avg_throughput,
                        AVG(quality_score) as avg_quality,
                        COUNT(*) as sample_count,
                        SUM(CASE WHEN success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate
                    FROM model_performance
                    WHERE timestamp > NOW() - INTERVAL '30 days'
                        AND context_length <= max_context
                """

                params = []
                if task_type:
                    query += " AND task_type = %s"
                    params.append(task_type)

                if scope_context:
                    query += " AND work_dir_id = %s"
                    params.append(scope_context.get_namespace_key())

                query += """
                    GROUP BY model_name, provider
                    HAVING COUNT(*) >= 5  -- Minimum samples
                    ORDER BY success_rate DESC, avg_quality DESC
                """

                cur.execute(query, params)
                return cur.fetchall()

        except Exception as e:
            logger.error(f"Error getting candidate models: {e}")
            return []
        finally:
            self.pool.putconn(conn)

    async def _score_model(
        self,
        model_info: dict[str, Any],
        task_embedding: list[float],
        strategy: OptimizationStrategy,
        budget_cents: Optional[float],
        max_latency_ms: Optional[int],
        min_quality: Optional[float]
    ) -> float:
        """Score a model based on strategy and constraints."""
        # Apply hard constraints
        if budget_cents and model_info.get("avg_cost_cents", 0) > budget_cents:
            return 0.0
        if max_latency_ms and model_info.get("avg_latency_ms", 0) > max_latency_ms:
            return 0.0
        if min_quality and model_info.get("avg_quality", 1.0) < min_quality:
            return 0.0

        # Calculate weighted score based on strategy
        weights = self._get_strategy_weights(strategy)

        # Normalize metrics
        cost_score = 1.0 / (1.0 + model_info.get("avg_cost_cents", 0) / 100)
        speed_score = 1.0 / (1.0 + model_info.get("avg_latency_ms", 0) / 1000)
        quality_score = model_info.get("avg_quality", 1.0)
        throughput_score = min(model_info.get("avg_throughput", 50) / 100, 1.0)
        reliability_score = model_info.get("success_rate", 1.0)

        # Apply weights
        score = (
            weights["cost"] * cost_score +
            weights["speed"] * speed_score +
            weights["quality"] * quality_score +
            weights["throughput"] * throughput_score +
            weights["reliability"] * reliability_score
        )

        # Boost score for models with more samples (confidence factor)
        sample_boost = min(model_info.get("sample_count", 0) / 100, 1.2)
        score *= sample_boost

        return score

    def _get_strategy_weights(self, strategy: OptimizationStrategy) -> dict[str, float]:
        """Get optimization weights for a strategy."""
        weights = {
            OptimizationStrategy.COST: {
                "cost": 0.5, "speed": 0.1, "quality": 0.2,
                "throughput": 0.1, "reliability": 0.1
            },
            OptimizationStrategy.SPEED: {
                "cost": 0.1, "speed": 0.5, "quality": 0.2,
                "throughput": 0.1, "reliability": 0.1
            },
            OptimizationStrategy.QUALITY: {
                "cost": 0.1, "speed": 0.1, "quality": 0.5,
                "throughput": 0.1, "reliability": 0.2
            },
            OptimizationStrategy.BALANCED: {
                "cost": 0.2, "speed": 0.2, "quality": 0.3,
                "throughput": 0.1, "reliability": 0.2
            },
            OptimizationStrategy.THROUGHPUT: {
                "cost": 0.1, "speed": 0.2, "quality": 0.2,
                "throughput": 0.4, "reliability": 0.1
            },
            OptimizationStrategy.COST_QUALITY: {
                "cost": 0.35, "speed": 0.05, "quality": 0.4,
                "throughput": 0.05, "reliability": 0.15
            },
            OptimizationStrategy.ADAPTIVE: {
                # Adaptive learns from patterns
                "cost": 0.2, "speed": 0.2, "quality": 0.2,
                "throughput": 0.2, "reliability": 0.2
            }
        }
        return weights.get(strategy, weights[OptimizationStrategy.BALANCED])

    async def _get_task_embedding(self, task_type: str, task_description: str) -> list[float]:
        """Generate embedding for task similarity matching."""
        # Create task signature
        task_signature = f"{task_type}:{task_description}"

        # Check cache
        cache_key = hashlib.md5(task_signature.encode()).hexdigest()
        if cache_key in self.performance_cache:
            return self.performance_cache[cache_key]

        # Generate embedding
        embedding = await self.embedding_provider.embed_text(task_signature)

        # Cache it
        self.performance_cache[cache_key] = embedding

        # Convert to PostgreSQL array format
        return '[' + ','.join(map(str, embedding)) + ']'

    async def _update_optimization_scores(
        self,
        model_name: str,
        provider: str,
        task_type: str
    ):
        """Update pre-calculated optimization scores."""
        if not self.pool:
            return

        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                # Calculate scores for each strategy
                for strategy in OptimizationStrategy:
                    weights = self._get_strategy_weights(strategy)

                    # Get recent performance stats
                    cur.execute("""
                        SELECT
                            AVG(total_cost_cents) as avg_cost,
                            AVG(total_time_ms) as avg_latency,
                            AVG(tokens_per_second) as avg_throughput,
                            AVG(quality_score) as avg_quality,
                            COUNT(*) as sample_count,
                            SUM(CASE WHEN success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate
                        FROM model_performance
                        WHERE model_name = %s
                            AND provider = %s
                            AND task_type = %s
                            AND timestamp > NOW() - INTERVAL '30 days'
                    """, (model_name, provider, task_type))

                    stats = cur.fetchone()
                    if not stats or stats[4] < 5:  # Not enough samples
                        continue

                    # Calculate individual scores
                    cost_score = 1.0 / (1.0 + float(stats[0] or 0) / 100)
                    speed_score = 1.0 / (1.0 + float(stats[1] or 0) / 1000)
                    quality_score = float(stats[3] or 1.0)
                    throughput_score = min(float(stats[2] or 50) / 100, 1.0)

                    # Calculate weighted efficiency
                    efficiency = (
                        weights["cost"] * cost_score +
                        weights["speed"] * speed_score +
                        weights["quality"] * quality_score +
                        weights["throughput"] * throughput_score +
                        weights["reliability"] * float(stats[5] or 1.0)
                    )

                    # Upsert optimization scores
                    cur.execute("""
                        INSERT INTO model_optimization_scores (
                            model_name, provider, task_type, optimization_strategy,
                            efficiency_score, cost_score, speed_score, quality_score,
                            throughput_score, sample_count, success_rate,
                            avg_latency_ms, avg_cost_cents
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (model_name, provider, task_type, optimization_strategy)
                        DO UPDATE SET
                            efficiency_score = EXCLUDED.efficiency_score,
                            cost_score = EXCLUDED.cost_score,
                            speed_score = EXCLUDED.speed_score,
                            quality_score = EXCLUDED.quality_score,
                            throughput_score = EXCLUDED.throughput_score,
                            sample_count = EXCLUDED.sample_count,
                            success_rate = EXCLUDED.success_rate,
                            avg_latency_ms = EXCLUDED.avg_latency_ms,
                            avg_cost_cents = EXCLUDED.avg_cost_cents,
                            calculated_at = CURRENT_TIMESTAMP
                    """, (
                        model_name, provider, task_type, strategy.value,
                        efficiency, cost_score, speed_score, quality_score,
                        throughput_score, stats[4], stats[5],
                        stats[1], stats[0]
                    ))

                conn.commit()

        except Exception as e:
            logger.error(f"Error updating optimization scores: {e}")
            conn.rollback()
        finally:
            self.pool.putconn(conn)

    async def get_performance_report(
        self,
        model_name: Optional[str] = None,
        task_type: Optional[str] = None,
        days: int = 30,
        scope_context: Optional[ScopeContext] = None
    ) -> dict[str, Any]:
        """
        Generate performance report for models.

        Returns comprehensive statistics and recommendations.
        """
        if not self.pool:
            return {"error": "Database not available"}

        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build query
                query = """
                    SELECT
                        model_name,
                        provider,
                        COUNT(*) as total_requests,
                        SUM(input_tokens) as total_input_tokens,
                        SUM(output_tokens) as total_output_tokens,
                        SUM(total_cost_cents) / 100.0 as total_cost_usd,
                        AVG(total_cost_cents) / 100.0 as avg_cost_usd,
                        AVG(total_time_ms) as avg_latency_ms,
                        MIN(total_time_ms) as min_latency_ms,
                        MAX(total_time_ms) as max_latency_ms,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_time_ms) as p50_latency_ms,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_time_ms) as p95_latency_ms,
                        AVG(tokens_per_second) as avg_tokens_per_second,
                        AVG(quality_score) as avg_quality,
                        SUM(CASE WHEN success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate,
                        SUM(CASE WHEN error_type IS NOT NULL THEN 1 ELSE 0 END) as error_count,
                        ARRAY_AGG(DISTINCT error_type) FILTER (WHERE error_type IS NOT NULL) as error_types
                    FROM model_performance
                    WHERE timestamp > NOW() - INTERVAL '%s days'
                """

                params = [days]

                if model_name:
                    query += " AND model_name = %s"
                    params.append(model_name)

                if task_type:
                    query += " AND task_type = %s"
                    params.append(task_type)

                if scope_context:
                    query += " AND work_dir_id = %s"
                    params.append(scope_context.get_namespace_key())

                query += " GROUP BY model_name, provider ORDER BY total_requests DESC"

                cur.execute(query, params)
                results = cur.fetchall()

                # Calculate cost savings opportunities
                if results:
                    best_cost = min(r["avg_cost_usd"] for r in results if r["success_rate"] > 0.95)
                    for r in results:
                        r["potential_savings_usd"] = max(0, (r["avg_cost_usd"] - best_cost) * r["total_requests"])

                # Get top errors
                cur.execute("""
                    SELECT error_type, COUNT(*) as count, ARRAY_AGG(DISTINCT model_name) as models
                    FROM model_performance
                    WHERE error_type IS NOT NULL
                        AND timestamp > NOW() - INTERVAL '%s days'
                    GROUP BY error_type
                    ORDER BY count DESC
                    LIMIT 10
                """, (days,))
                top_errors = cur.fetchall()

                return {
                    "period_days": days,
                    "models": results,
                    "top_errors": top_errors,
                    "total_cost_usd": sum(r["total_cost_usd"] for r in results),
                    "total_requests": sum(r["total_requests"] for r in results),
                    "recommendations": self._generate_recommendations(results)
                }

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": str(e)}
        finally:
            self.pool.putconn(conn)

    def _generate_recommendations(self, results: list[dict]) -> list[str]:
        """Generate recommendations based on performance data."""
        recommendations = []

        if not results:
            return ["Insufficient data for recommendations"]

        # Find underperforming models
        avg_success = sum(r["success_rate"] for r in results) / len(results)
        for r in results:
            if r["success_rate"] < avg_success * 0.9:
                recommendations.append(
                    f"Consider replacing {r['model_name']} (success rate: {r['success_rate']:.1%})"
                )

        # Find cost optimization opportunities
        cost_sorted = sorted(results, key=lambda x: x["avg_cost_usd"])
        if len(cost_sorted) > 1:
            cheapest = cost_sorted[0]
            for r in cost_sorted[1:]:
                if r["total_requests"] > 100 and r["avg_cost_usd"] > cheapest["avg_cost_usd"] * 2:
                    savings = r["potential_savings_usd"]
                    recommendations.append(
                        f"Switch {r['model_name']} to {cheapest['model_name']} to save ${savings:.2f}"
                    )

        # Find latency issues
        for r in results:
            if r["p95_latency_ms"] and r["p95_latency_ms"] > 10000:  # >10s P95
                recommendations.append(
                    f"High latency detected for {r['model_name']} (P95: {r['p95_latency_ms']/1000:.1f}s)"
                )

        return recommendations if recommendations else ["All models performing within acceptable parameters"]

    def update_model_pricing(self, pricing: ModelPricing) -> bool:
        """Update model pricing information."""
        self.pricing_cache[pricing.model_name] = pricing

        if not self.pool:
            return True

        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO model_pricing (
                        model_name, provider,
                        input_price_cents, output_price_cents,
                        cache_write_price_cents, cache_read_price_cents,
                        request_price_cents, currency
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_name, provider, effective_date)
                    DO UPDATE SET
                        input_price_cents = EXCLUDED.input_price_cents,
                        output_price_cents = EXCLUDED.output_price_cents,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    pricing.model_name, pricing.provider,
                    pricing.input_price_per_1m * 100 / 1_000_000,  # Convert to cents
                    pricing.output_price_per_1m * 100 / 1_000_000,
                    pricing.cache_write_price_per_1m * 100 / 1_000_000 if pricing.cache_write_price_per_1m else None,
                    pricing.cache_read_price_per_1m * 100 / 1_000_000 if pricing.cache_read_price_per_1m else None,
                    pricing.request_price * 100 if pricing.request_price else 0,
                    pricing.currency
                ))
                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error updating pricing: {e}")
            conn.rollback()
            return False
        finally:
            self.pool.putconn(conn)


# Global instance
_performance_tracker = None


def get_performance_tracker() -> ModelPerformanceTracker:
    """Get or create the global performance tracker instance."""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = ModelPerformanceTracker()
    return _performance_tracker
