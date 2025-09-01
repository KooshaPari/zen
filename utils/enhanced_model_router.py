"""
Enhanced Model Router with Performance-Based Selection

This module extends the basic model router with intelligent selection based on:
- Historical performance data from pgvector
- Real-time cost/token/latency tracking
- Task similarity matching using embeddings
- Adaptive learning from success/failure patterns
- Multi-objective optimization (cost, speed, quality)

The router automatically stores decisions and outcomes, learning from patterns
to improve future selections.
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from utils.model_performance_tracker import (
    ModelPerformanceMetrics,
    ModelPerformanceTracker,
    ModelPricing,
    OptimizationStrategy,
    get_performance_tracker,
)
from utils.model_router import ComplexitySignals, ModelRouter, get_model_router
from utils.scope_utils import ScopeContext

# Import adaptive learning components
try:
    from utils.adaptive_learning_engine import (
        AdaptiveLearningEngine,
        ContextConstraints,
        CostConstraints,
        OptimizationObjective,
        PerformanceTargets,
        get_adaptive_engine,
    )
    from utils.adaptive_learning_engine import ModelSelectionContext as ALModelContext
    from utils.context_aware_predictor import ContextAwarePredictor, get_context_predictor
    from utils.cost_performance_optimizer import (
        CostPerformanceOptimizer,
        OptimizationMode,
        get_cost_performance_optimizer,
    )
    from utils.cost_performance_optimizer import OptimizationConstraints as CPOConstraints
    ADAPTIVE_LEARNING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Adaptive learning components not available: {e}")
    ADAPTIVE_LEARNING_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ModelSelectionContext:
    """Context for model selection decision."""

    # Task information
    task_type: str
    task_description: str = ""
    tool_name: Optional[str] = None

    # Complexity signals
    complexity_signals: Optional[ComplexitySignals] = None
    estimated_tokens: Optional[int] = None

    # Constraints
    max_cost_usd: Optional[float] = None
    max_latency_seconds: Optional[float] = None
    min_quality_score: Optional[float] = None
    required_context_window: Optional[int] = None

    # Optimization preferences
    optimization_strategy: Optional[OptimizationStrategy] = None

    # Context
    scope_context: Optional[ScopeContext] = None
    session_id: Optional[str] = None

    # Previous attempts (for retry logic)
    failed_models: list[str] = None
    retry_count: int = 0

    def __post_init__(self):
        if self.failed_models is None:
            self.failed_models = []
        if self.complexity_signals is None:
            self.complexity_signals = ComplexitySignals()

    def get_task_hash(self) -> str:
        """Generate hash for task characteristics."""
        task_sig = f"{self.task_type}:{self.task_description}:{self.tool_name}"
        return hashlib.md5(task_sig.encode()).hexdigest()[:16]


class EnhancedModelRouter:
    """
    Enhanced model router that learns from performance history.

    This router combines static configuration with dynamic performance
    tracking to make optimal model selections.
    """

    def __init__(
        self,
        base_router: Optional[ModelRouter] = None,
        performance_tracker: Optional[ModelPerformanceTracker] = None,
        enable_learning: bool = True
    ):
        """
        Initialize enhanced router.

        Args:
            base_router: Base model router with static config
            performance_tracker: Performance tracking system
            enable_learning: Whether to use historical data for selection
        """
        self.base_router = base_router or get_model_router()
        self.performance_tracker = performance_tracker or get_performance_tracker()
        self.enable_learning = enable_learning and os.getenv("ZEN_ADAPTIVE_ROUTING", "1") == "1"

        # Initialize adaptive learning components if available
        self.adaptive_engine = None
        self.context_predictor = None
        self.cost_optimizer = None

        if ADAPTIVE_LEARNING_AVAILABLE and self.enable_learning:
            try:
                self.adaptive_engine = get_adaptive_engine()
                self.context_predictor = get_context_predictor()
                self.cost_optimizer = get_cost_performance_optimizer()
                logger.info("Adaptive learning components initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize adaptive learning: {e}")

        # Track active selections for performance measurement
        self.active_selections = {}  # request_id -> (model, start_time, context)

        # Cache for recent decisions
        self.decision_cache = {}  # task_hash -> (model, timestamp)

        logger.info(f"Enhanced model router initialized (learning={'enabled' if self.enable_learning else 'disabled'})")

    async def select_model(
        self,
        context: ModelSelectionContext
    ) -> tuple[str, dict[str, Any]]:
        """
        Select optimal model based on context and history.

        Args:
            context: Selection context with requirements

        Returns:
            Tuple of (model_name, selection_metadata)
        """
        start_time = time.time()

        # Step 1: Get base router recommendation
        base_decision = self.base_router.decide(
            task_type=context.task_type,
            signals=context.complexity_signals,
            est_tokens=context.estimated_tokens,
            allow_long_context=context.required_context_window and context.required_context_window > 32000
        )

        base_model = base_decision.get("chosen_model", "gemini-2.0-flash-exp")
        candidates = base_decision.get("candidates", [base_model])

        # Step 2: If learning disabled, return base decision
        if not self.enable_learning:
            return base_model, {
                "strategy": "static_config",
                "base_decision": base_decision,
                "selection_time_ms": int((time.time() - start_time) * 1000)
            }

        # Step 3: Try adaptive learning engine first if available
        if self.adaptive_engine and context.estimated_tokens:
            try:
                # Build adaptive learning context
                al_context = ALModelContext(
                    task_type=context.task_type,
                    input_tokens=context.estimated_tokens or 1000,
                    expected_output_tokens=min(4000, context.estimated_tokens // 2),
                    context_constraints=ContextConstraints(
                        max_context_tokens=context.required_context_window or 32768,
                        available_context_tokens=context.required_context_window or 32768,
                        reserved_output_tokens=4000,
                        conversation_history_tokens=0,
                        system_prompt_tokens=500
                    ),
                    cost_constraints=CostConstraints(
                        max_cost_per_request=context.max_cost_usd,
                        daily_budget=100.0,  # Default daily budget
                        current_daily_spend=0.0
                    ),
                    performance_targets=PerformanceTargets(
                        target_quality_score=context.min_quality_score or 0.8,
                        max_acceptable_latency_ms=int(context.max_latency_seconds * 1000) if context.max_latency_seconds else 5000,
                        min_tokens_per_second=20.0
                    ),
                    optimization_objective=self._map_optimization_objective(context.optimization_strategy),
                    recent_failures=context.failed_models,
                    is_retry=context.retry_count > 0,
                    complexity_score=base_decision.get("score", 50) / 100.0
                )

                # Get adaptive selection
                optimal_model, prediction = await self.adaptive_engine.select_optimal_model(al_context)

                # Track the selection
                request_id = context.session_id or hashlib.md5(str(time.time()).encode()).hexdigest()
                self.active_selections[request_id] = (
                    optimal_model,
                    time.time(),
                    context,
                    al_context.request_id  # Store AL request ID for reconciliation
                )

                return optimal_model, {
                    "strategy": "adaptive_learning",
                    "base_recommendation": base_model,
                    "predicted_cost": prediction.predicted_cost,
                    "predicted_latency_ms": prediction.predicted_latency_ms,
                    "predicted_quality": prediction.predicted_quality,
                    "predicted_tps": prediction.predicted_tps,
                    "confidence_score": prediction.confidence_score,
                    "context_fit": prediction.context_fit,
                    "complexity_tier": base_decision.get("tier"),
                    "complexity_score": base_decision.get("score"),
                    "selection_time_ms": int((time.time() - start_time) * 1000),
                    "request_id": request_id
                }

            except Exception as e:
                logger.warning(f"Adaptive learning selection failed: {e}")
                # Fall through to performance-based selection

        # Step 4: Apply performance-based selection
        try:
            # Convert constraints to tracker format
            budget_cents = context.max_cost_usd * 100 if context.max_cost_usd else None
            max_latency_ms = context.max_latency_seconds * 1000 if context.max_latency_seconds else None

            # Get optimal model from performance history
            optimal_model, perf_metadata = await self.performance_tracker.select_optimal_model(
                task_type=context.task_type,
                task_description=context.task_description,
                context_length=context.estimated_tokens or 0,
                optimization_strategy=context.optimization_strategy,
                budget_cents=budget_cents,
                max_latency_ms=max_latency_ms,
                min_quality=context.min_quality_score,
                scope_context=context.scope_context
            )

            # Filter out failed models
            if optimal_model in context.failed_models:
                # Find next best candidate
                for candidate in candidates:
                    if candidate not in context.failed_models:
                        optimal_model = candidate
                        perf_metadata["fallback_reason"] = "previous_failure"
                        break

            # Track the selection
            request_id = context.session_id or hashlib.md5(str(time.time()).encode()).hexdigest()
            self.active_selections[request_id] = (
                optimal_model,
                time.time(),
                context
            )

            return optimal_model, {
                "strategy": "performance_optimized",
                "base_recommendation": base_model,
                "performance_metadata": perf_metadata,
                "complexity_tier": base_decision.get("tier"),
                "complexity_score": base_decision.get("score"),
                "candidates_considered": candidates,
                "selection_time_ms": int((time.time() - start_time) * 1000),
                "request_id": request_id
            }

        except Exception as e:
            logger.warning(f"Performance-based selection failed, using base: {e}")
            return base_model, {
                "strategy": "fallback_to_base",
                "base_decision": base_decision,
                "error": str(e),
                "selection_time_ms": int((time.time() - start_time) * 1000)
            }

    async def track_model_performance(
        self,
        request_id: str,
        input_tokens: int,
        output_tokens: int,
        success: bool,
        error_type: Optional[str] = None,
        quality_score: Optional[float] = None,
        first_token_time: Optional[float] = None,
        cached_tokens: int = 0
    ) -> bool:
        """
        Track performance metrics for a model execution.

        Args:
            request_id: Request ID from selection
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            success: Whether execution succeeded
            error_type: Type of error if failed
            quality_score: Quality rating (0-1)
            first_token_time: Time to first token
            cached_tokens: Number of cached tokens used

        Returns:
            True if successfully tracked
        """
        if request_id not in self.active_selections:
            logger.warning(f"No active selection found for request {request_id}")
            return False

        selection_data = self.active_selections.pop(request_id)

        # Handle both old format (3-tuple) and new format (4-tuple with AL request ID)
        if len(selection_data) == 4:
            model_name, start_time, context, al_request_id = selection_data
        else:
            model_name, start_time, context = selection_data
            al_request_id = None

        total_time = time.time() - start_time

        # Get pricing information
        pricing = self.performance_tracker.pricing_cache.get(
            model_name,
            ModelPricing(
                model_name=model_name,
                provider="unknown",
                input_price_per_1m=0.0,
                output_price_per_1m=0.0
            )
        )

        # Calculate costs
        total_cost = pricing.calculate_cost(input_tokens, output_tokens, cached_tokens)

        # Create metrics
        metrics = ModelPerformanceMetrics(
            model_name=model_name,
            provider=pricing.provider,
            task_type=context.task_type,
            task_hash=context.get_task_hash(),

            # Token metrics
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            total_tokens=input_tokens + output_tokens,

            # Time metrics
            first_token_time=first_token_time or 0.0,
            total_time=total_time,
            queue_time=0.0,  # Could be tracked separately

            # Cost metrics
            input_cost=(input_tokens / 1_000_000) * pricing.input_price_per_1m,
            output_cost=(output_tokens / 1_000_000) * pricing.output_price_per_1m,
            total_cost=total_cost,
            cost_per_token=total_cost / (input_tokens + output_tokens) if (input_tokens + output_tokens) > 0 else 0,

            # Performance metrics
            tokens_per_second=output_tokens / total_time if total_time > 0 else 0,
            success=success,
            error_type=error_type,
            quality_score=quality_score or 1.0,

            # Context
            context_length=context.estimated_tokens or 0,
            max_context=context.required_context_window or 0,
            temperature=0.7,  # Could be tracked from actual params

            # Metadata
            session_id=context.session_id,
            request_id=request_id
        )

        # Track in database
        tracked = await self.performance_tracker.track_performance(
            metrics, context.scope_context
        )

        # Reconcile with adaptive learning if applicable
        if tracked and al_request_id and self.adaptive_engine:
            try:
                await self.adaptive_engine.record_actual_performance(
                    al_request_id,
                    model_name,
                    metrics
                )
                logger.debug(f"Reconciled adaptive learning prediction for {al_request_id}")
            except Exception as e:
                logger.warning(f"Failed to reconcile adaptive learning: {e}")

        if tracked:
            logger.info(
                f"Tracked performance: {model_name} - "
                f"tokens: {input_tokens}/{output_tokens}, "
                f"cost: ${total_cost:.4f}, "
                f"time: {total_time:.2f}s, "
                f"success: {success}"
            )

        return tracked

    async def get_model_recommendations(
        self,
        task_type: str,
        scope_context: Optional[ScopeContext] = None
    ) -> list[dict[str, Any]]:
        """
        Get model recommendations for a task type.

        Returns ranked list of models with performance statistics.
        """
        if not self.enable_learning:
            # Return static recommendations
            base_decision = self.base_router.decide(
                task_type=task_type,
                signals=ComplexitySignals()
            )
            return [
                {"model": m, "source": "config"}
                for m in base_decision.get("candidates", [])
            ]

        # Get performance-based recommendations
        candidates = await self.performance_tracker._get_candidate_models(
            task_type, 0, scope_context
        )

        recommendations = []
        for candidate in candidates[:5]:  # Top 5
            recommendations.append({
                "model": candidate["model_name"],
                "provider": candidate["provider"],
                "avg_cost_usd": candidate.get("avg_cost_cents", 0) / 100,
                "avg_latency_ms": candidate.get("avg_latency_ms", 0),
                "success_rate": candidate.get("success_rate", 1.0),
                "avg_quality": candidate.get("avg_quality", 1.0),
                "sample_count": candidate.get("sample_count", 0),
                "source": "performance_history"
            })

        return recommendations

    async def update_model_pricing(
        self,
        model_name: str,
        provider: str,
        input_price_per_1m: float,
        output_price_per_1m: float,
        **kwargs
    ) -> bool:
        """
        Update pricing information for a model.

        Args:
            model_name: Model identifier
            provider: Provider name
            input_price_per_1m: Input token price per million
            output_price_per_1m: Output token price per million
            **kwargs: Additional pricing parameters

        Returns:
            True if successfully updated
        """
        pricing = ModelPricing(
            model_name=model_name,
            provider=provider,
            input_price_per_1m=input_price_per_1m,
            output_price_per_1m=output_price_per_1m,
            **kwargs
        )

        return self.performance_tracker.update_model_pricing(pricing)

    async def get_performance_report(
        self,
        model_name: Optional[str] = None,
        task_type: Optional[str] = None,
        days: int = 30,
        scope_context: Optional[ScopeContext] = None
    ) -> dict[str, Any]:
        """
        Get comprehensive performance report.

        Returns statistics, cost analysis, and recommendations.
        """
        return await self.performance_tracker.get_performance_report(
            model_name, task_type, days, scope_context
        )

    def set_optimization_strategy(self, strategy: OptimizationStrategy):
        """Update default optimization strategy."""
        self.performance_tracker.optimization_strategy = strategy
        logger.info(f"Optimization strategy updated to: {strategy.value}")

    def _map_optimization_objective(self, strategy: Optional[OptimizationStrategy]) -> OptimizationObjective:
        """Map performance tracker strategy to adaptive learning objective."""
        if not strategy:
            return OptimizationObjective.BALANCED

        mapping = {
            OptimizationStrategy.COST: OptimizationObjective.MINIMIZE_COST,
            OptimizationStrategy.SPEED: OptimizationObjective.MINIMIZE_LATENCY,
            OptimizationStrategy.QUALITY: OptimizationObjective.MAXIMIZE_QUALITY,
            OptimizationStrategy.BALANCED: OptimizationObjective.BALANCED,
            OptimizationStrategy.THROUGHPUT: OptimizationObjective.MAXIMIZE_THROUGHPUT,
        }

        return mapping.get(strategy, OptimizationObjective.BALANCED)


# Global instance
_enhanced_router = None


def get_enhanced_router() -> EnhancedModelRouter:
    """Get or create the global enhanced router instance."""
    global _enhanced_router
    if _enhanced_router is None:
        _enhanced_router = EnhancedModelRouter()
    return _enhanced_router


async def smart_model_selection(
    task_type: str,
    prompt: str = "",
    files: list[str] = None,
    max_cost_usd: Optional[float] = None,
    optimization: str = "balanced",
    scope_context: Optional[ScopeContext] = None
) -> tuple[str, dict[str, Any]]:
    """
    Convenience function for smart model selection.

    Args:
        task_type: Type of task (chat, analysis, etc.)
        prompt: User prompt for context
        files: List of files being processed
        max_cost_usd: Budget constraint
        optimization: Strategy (cost, speed, quality, balanced)
        scope_context: Scope context

    Returns:
        Tuple of (model_name, metadata)
    """
    router = get_enhanced_router()

    # Build complexity signals
    signals = ComplexitySignals(
        prompt_chars=len(prompt),
        files_count=len(files) if files else 0
    )

    # Estimate tokens (rough heuristic)
    estimated_tokens = len(prompt) // 4  # Rough char-to-token ratio
    if files:
        estimated_tokens += len(files) * 500  # Assume ~500 tokens per file

    # Map optimization string to enum
    strategy_map = {
        "cost": OptimizationStrategy.COST,
        "speed": OptimizationStrategy.SPEED,
        "quality": OptimizationStrategy.QUALITY,
        "balanced": OptimizationStrategy.BALANCED,
        "throughput": OptimizationStrategy.THROUGHPUT,
    }
    strategy = strategy_map.get(optimization, OptimizationStrategy.BALANCED)

    # Create selection context
    context = ModelSelectionContext(
        task_type=task_type,
        task_description=prompt[:100],  # First 100 chars for embedding
        complexity_signals=signals,
        estimated_tokens=estimated_tokens,
        max_cost_usd=max_cost_usd,
        optimization_strategy=strategy,
        scope_context=scope_context
    )

    return await router.select_model(context)
