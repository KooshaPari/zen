"""
Adaptive Learning Engine for Multi-Objective Model Optimization

This module implements a comprehensive ML-based system for optimizing model selection
based on cost, performance, time, and context constraints. It learns from historical
data to predict optimal model choices while respecting token limits and budget constraints.

Key Features:
- Multi-objective optimization (cost, performance, time, quality)
- Context window awareness and token budget management
- Real-time prediction with confidence scoring
- Continuous learning from prediction errors
- Adaptive strategy selection based on workload patterns
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    StandardScaler = None

from utils.model_performance_tracker import ModelPerformanceMetrics, ModelPerformanceTracker

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Primary optimization objectives."""
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_QUALITY = "maximize_quality"
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    BALANCED = "balanced"
    COST_PERFORMANCE_RATIO = "cost_performance_ratio"
    CONTEXT_EFFICIENT = "context_efficient"


@dataclass
class ContextConstraints:
    """Context window and token budget constraints."""
    max_context_tokens: int
    available_context_tokens: int
    reserved_output_tokens: int
    conversation_history_tokens: int
    system_prompt_tokens: int

    @property
    def usable_input_tokens(self) -> int:
        """Calculate tokens available for new input."""
        return (self.available_context_tokens -
                self.reserved_output_tokens -
                self.conversation_history_tokens -
                self.system_prompt_tokens)

    @property
    def context_utilization(self) -> float:
        """Calculate context window utilization percentage."""
        used = (self.conversation_history_tokens +
                self.system_prompt_tokens +
                self.reserved_output_tokens)
        return min(1.0, used / self.max_context_tokens)


@dataclass
class CostConstraints:
    """Budget and cost constraints."""
    max_cost_per_request: Optional[float] = None
    daily_budget: Optional[float] = None
    monthly_budget: Optional[float] = None
    current_daily_spend: float = 0.0
    current_monthly_spend: float = 0.0

    @property
    def remaining_daily_budget(self) -> Optional[float]:
        """Calculate remaining daily budget."""
        if self.daily_budget:
            return max(0, self.daily_budget - self.current_daily_spend)
        return None

    @property
    def can_afford(self) -> bool:
        """Check if we have budget remaining."""
        if self.daily_budget and self.current_daily_spend >= self.daily_budget:
            return False
        if self.monthly_budget and self.current_monthly_spend >= self.monthly_budget:
            return False
        return True


@dataclass
class PerformanceTargets:
    """Performance and quality targets."""
    target_quality_score: float = 0.8  # 0-1 scale
    max_acceptable_latency_ms: int = 5000
    min_tokens_per_second: float = 10.0
    max_retries: int = 2
    timeout_seconds: int = 60


@dataclass
class PredictedPerformance:
    """Predicted performance metrics for a model."""
    model_name: str
    predicted_cost: float
    predicted_latency_ms: float
    predicted_tps: float
    predicted_quality: float
    predicted_success_rate: float
    confidence_score: float
    context_fit: bool
    budget_fit: bool

    # Optimization scores
    cost_efficiency_score: float = 0.0
    performance_score: float = 0.0
    overall_score: float = 0.0

    # Risk factors
    context_overflow_risk: float = 0.0
    timeout_risk: float = 0.0
    quality_risk: float = 0.0


@dataclass
class ModelSelectionContext:
    """Complete context for model selection."""
    task_type: str
    input_tokens: int
    expected_output_tokens: int
    context_constraints: ContextConstraints
    cost_constraints: CostConstraints
    performance_targets: PerformanceTargets
    optimization_objective: OptimizationObjective

    # Historical context
    recent_failures: list[str] = field(default_factory=list)
    conversation_depth: int = 0
    is_retry: bool = False

    # Task characteristics
    complexity_score: float = 0.5
    requires_reasoning: bool = False
    requires_creativity: bool = False
    requires_factual_accuracy: bool = True

    # Metadata
    request_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


if TORCH_AVAILABLE:
    class PerformancePredictor(nn.Module):
        """Neural network for predicting model performance."""

        def __init__(self, input_dim: int = 32, hidden_dim: int = 128):
            super().__init__()

            # Shared feature extraction
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            )

            # Task-specific heads
            self.cost_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softplus()  # Ensure positive cost
            )

            self.latency_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softplus()  # Ensure positive latency
            )

            self.quality_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # Quality score 0-1
            )

            self.tps_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softplus()  # Ensure positive TPS
            )

            self.success_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # Success probability 0-1
            )

        def forward(self, x):
            features = self.feature_extractor(x)

            cost = self.cost_head(features)
            latency = self.latency_head(features)
            quality = self.quality_head(features)
            tps = self.tps_head(features)
            success = self.success_head(features)

            return {
                'cost': cost,
                'latency': latency,
                'quality': quality,
                'tps': tps,
                'success': success
            }
else:
    # Define a no-op placeholder to avoid import-time failures when torch isn't available
    class PerformancePredictor:  # type: ignore
        pass


class AdaptiveLearningEngine:
    """
    Main engine for adaptive model selection with multi-objective optimization.
    """

    def __init__(self,
                 performance_tracker: Optional[ModelPerformanceTracker] = None,
                 model_path: Optional[str] = None):
        """
        Initialize the adaptive learning engine.

        Args:
            performance_tracker: Tracker for historical performance data
            model_path: Path to pre-trained model weights
        """
        self.performance_tracker = performance_tracker or ModelPerformanceTracker()
        self.model_path = model_path or "models/adaptive_predictor.pth"

        # Initialize predictor if PyTorch available
        self.predictor = None
        self.scaler = None
        if TORCH_AVAILABLE:
            self.predictor = PerformancePredictor()
            self._load_model()

        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()

        # Model catalog with characteristics
        self.model_catalog = self._initialize_model_catalog()

        # Learning parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.update_frequency = 100  # Update model every N predictions

        # Prediction cache
        self.prediction_cache = {}
        self.cache_ttl_seconds = 300  # 5 minutes

        # Metrics tracking
        self.prediction_errors = []
        self.selection_history = []
        self.actual_performance_data = []  # Store reconciled data for training
        self.learning_metrics = {
            'updates': 0,
            'last_update': None,
            'training_samples': 0,
            'validation_loss': None
        }

    def _initialize_model_catalog(self) -> dict[str, dict[str, Any]]:
        """Initialize catalog of available models with their characteristics."""
        return {
            # Small, fast models
            "gemini-2.5-flash": {
                "context_window": 32768,
                "max_output": 8192,
                "strengths": ["speed", "cost"],
                "weaknesses": ["complex_reasoning"],
                "typical_tps": 150,
                "typical_latency_ms": 300
            },
            "gpt-4o-mini": {
                "context_window": 128000,
                "max_output": 16384,
                "strengths": ["speed", "cost", "general_tasks"],
                "weaknesses": ["complex_reasoning"],
                "typical_tps": 120,
                "typical_latency_ms": 400
            },

            # Balanced models
            "claude-3-5-haiku-20241022": {
                "context_window": 200000,
                "max_output": 8192,
                "strengths": ["balance", "quality", "speed"],
                "weaknesses": ["cost_at_scale"],
                "typical_tps": 80,
                "typical_latency_ms": 600
            },
            "gpt-4o": {
                "context_window": 128000,
                "max_output": 16384,
                "strengths": ["quality", "reasoning", "vision"],
                "weaknesses": ["cost"],
                "typical_tps": 60,
                "typical_latency_ms": 800
            },

            # High-quality models
            "claude-3-5-sonnet-20241022": {
                "context_window": 200000,
                "max_output": 8192,
                "strengths": ["quality", "reasoning", "coding"],
                "weaknesses": ["cost", "speed"],
                "typical_tps": 40,
                "typical_latency_ms": 1200
            },
            "gemini-2.5-pro": {
                "context_window": 2097152,  # 2M context
                "max_output": 8192,
                "strengths": ["massive_context", "quality"],
                "weaknesses": ["cost", "speed"],
                "typical_tps": 30,
                "typical_latency_ms": 1500
            },

            # Specialized models
            "o1-preview": {
                "context_window": 128000,
                "max_output": 32768,
                "strengths": ["reasoning", "problem_solving"],
                "weaknesses": ["cost", "speed"],
                "typical_tps": 20,
                "typical_latency_ms": 3000
            },
            "deepseek-chat": {
                "context_window": 32768,
                "max_output": 4096,
                "strengths": ["cost", "coding"],
                "weaknesses": ["availability"],
                "typical_tps": 100,
                "typical_latency_ms": 500
            }
        }

    def _extract_features(self, context: ModelSelectionContext, model_name: str) -> np.ndarray:
        """Extract feature vector for prediction."""
        model_info = self.model_catalog.get(model_name, {})

        features = [
            # Token metrics
            context.input_tokens / 1000,
            context.expected_output_tokens / 1000,
            context.context_constraints.context_utilization,
            context.context_constraints.usable_input_tokens / 1000,

            # Model characteristics
            model_info.get("context_window", 32768) / 100000,
            model_info.get("typical_tps", 50) / 100,
            model_info.get("typical_latency_ms", 1000) / 1000,

            # Task characteristics
            context.complexity_score,
            float(context.requires_reasoning),
            float(context.requires_creativity),
            float(context.requires_factual_accuracy),

            # Historical context
            context.conversation_depth / 10,
            float(context.is_retry),
            float(model_name in context.recent_failures),

            # Cost constraints
            float(context.cost_constraints.can_afford),
            (context.cost_constraints.remaining_daily_budget or 100) / 100,

            # Performance targets
            context.performance_targets.target_quality_score,
            context.performance_targets.max_acceptable_latency_ms / 5000,
            context.performance_targets.min_tokens_per_second / 100,

            # Optimization objective encoding (one-hot)
            float(context.optimization_objective == OptimizationObjective.MINIMIZE_COST),
            float(context.optimization_objective == OptimizationObjective.MAXIMIZE_QUALITY),
            float(context.optimization_objective == OptimizationObjective.MINIMIZE_LATENCY),
            float(context.optimization_objective == OptimizationObjective.MAXIMIZE_THROUGHPUT),
            float(context.optimization_objective == OptimizationObjective.BALANCED),
            float(context.optimization_objective == OptimizationObjective.COST_PERFORMANCE_RATIO),
            float(context.optimization_objective == OptimizationObjective.CONTEXT_EFFICIENT),

            # Time features
            datetime.now().hour / 24,  # Hour of day
            datetime.now().weekday() / 7,  # Day of week

            # Padding to reach input_dim=32
            0.0, 0.0, 0.0, 0.0
        ]

        return np.array(features[:32], dtype=np.float32)

    async def predict_performance(self,
                                 context: ModelSelectionContext,
                                 model_name: str) -> PredictedPerformance:
        """
        Predict performance metrics for a specific model given the context.

        Args:
            context: Selection context with constraints and targets
            model_name: Model to evaluate

        Returns:
            Predicted performance metrics
        """
        # Check cache
        cache_key = f"{context.request_id}:{model_name}"
        if cache_key in self.prediction_cache:
            cached = self.prediction_cache[cache_key]
            if (datetime.now(timezone.utc) - cached['timestamp']).seconds < self.cache_ttl_seconds:
                return cached['prediction']

        # Extract features
        features = self._extract_features(context, model_name)

        # Get model info
        model_info = self.model_catalog.get(model_name, {})

        # Make prediction
        if TORCH_AVAILABLE and self.predictor:
            with torch.no_grad():
                input_tensor = torch.FloatTensor(features).unsqueeze(0)
                predictions = self.predictor(input_tensor)

                predicted_cost = float(predictions['cost'][0])
                predicted_latency = float(predictions['latency'][0]) * 1000  # Convert to ms
                predicted_quality = float(predictions['quality'][0])
                predicted_tps = float(predictions['tps'][0]) * 100  # Scale up
                predicted_success = float(predictions['success'][0])

                # Calculate confidence based on model uncertainty
                confidence = self._calculate_confidence(context, model_name)
        else:
            # Fallback heuristic predictions
            predicted_cost = self._heuristic_cost_prediction(context, model_name)
            predicted_latency = model_info.get("typical_latency_ms", 1000)
            predicted_quality = self._heuristic_quality_prediction(context, model_name)
            predicted_tps = model_info.get("typical_tps", 50)
            predicted_success = 0.95 if model_name not in context.recent_failures else 0.7
            confidence = 0.6  # Lower confidence for heuristic predictions

        # Check constraints
        context_fit = (context.input_tokens + context.expected_output_tokens <=
                      model_info.get("context_window", 32768))

        budget_fit = True
        if context.cost_constraints.max_cost_per_request:
            budget_fit = predicted_cost <= context.cost_constraints.max_cost_per_request

        # Calculate optimization scores
        cost_efficiency_score = self._calculate_cost_efficiency(
            predicted_cost, predicted_quality, predicted_tps
        )

        performance_score = self._calculate_performance_score(
            predicted_latency, predicted_tps, predicted_quality
        )

        overall_score = self._calculate_overall_score(
            context, predicted_cost, predicted_latency,
            predicted_quality, predicted_tps
        )

        # Calculate risk factors
        context_overflow_risk = max(0, min(1,
            (context.input_tokens + context.expected_output_tokens) /
            model_info.get("context_window", 32768)
        ))

        timeout_risk = max(0, min(1,
            predicted_latency / (context.performance_targets.timeout_seconds * 1000)
        ))

        quality_risk = max(0,
            context.performance_targets.target_quality_score - predicted_quality
        )

        prediction = PredictedPerformance(
            model_name=model_name,
            predicted_cost=predicted_cost,
            predicted_latency_ms=predicted_latency,
            predicted_tps=predicted_tps,
            predicted_quality=predicted_quality,
            predicted_success_rate=predicted_success,
            confidence_score=confidence,
            context_fit=context_fit,
            budget_fit=budget_fit,
            cost_efficiency_score=cost_efficiency_score,
            performance_score=performance_score,
            overall_score=overall_score,
            context_overflow_risk=context_overflow_risk,
            timeout_risk=timeout_risk,
            quality_risk=quality_risk
        )

        # Cache prediction
        self.prediction_cache[cache_key] = {
            'prediction': prediction,
            'timestamp': datetime.now(timezone.utc)
        }

        return prediction

    def _heuristic_cost_prediction(self,
                                  context: ModelSelectionContext,
                                  model_name: str) -> float:
        """Heuristic cost prediction based on model and token counts."""
        # Get base pricing from performance tracker
        pricing = self.performance_tracker.model_pricing.get(model_name)
        if not pricing:
            # Default pricing based on model tier
            if "mini" in model_name or "flash" in model_name:
                input_price = 0.00015
                output_price = 0.0006
            elif "haiku" in model_name or "4o" in model_name:
                input_price = 0.001
                output_price = 0.004
            else:
                input_price = 0.003
                output_price = 0.015
        else:
            input_price = pricing.input_price_per_1k / 1000
            output_price = pricing.output_price_per_1k / 1000

        estimated_cost = (
            (context.input_tokens * input_price) +
            (context.expected_output_tokens * output_price)
        )

        # Adjust for complexity
        estimated_cost *= (1 + context.complexity_score * 0.2)

        return estimated_cost

    def _heuristic_quality_prediction(self,
                                     context: ModelSelectionContext,
                                     model_name: str) -> float:
        """Heuristic quality prediction based on model tier and task."""
        base_quality = 0.5

        # Model tier bonuses
        if "o1" in model_name or "claude-3-5-sonnet" in model_name:
            base_quality = 0.95
        elif "gpt-4o" in model_name or "claude-3-5-haiku" in model_name:
            base_quality = 0.85
        elif "gemini-2.5-pro" in model_name:
            base_quality = 0.9
        elif "mini" in model_name or "flash" in model_name:
            base_quality = 0.7

        # Task adjustments
        if context.requires_reasoning and "o1" not in model_name:
            base_quality *= 0.9
        if context.requires_creativity and "claude" in model_name:
            base_quality *= 1.05
        if context.complexity_score > 0.7:
            base_quality *= 0.95

        return min(1.0, base_quality)

    def _calculate_confidence(self,
                            context: ModelSelectionContext,
                            model_name: str) -> float:
        """Calculate confidence score for prediction."""
        # Base confidence from historical data availability
        historical_samples = len(self.performance_tracker.get_model_history(
            model_name, limit=100
        ))

        base_confidence = min(0.9, 0.3 + (historical_samples / 100) * 0.6)

        # Adjust for context similarity to historical data
        if context.is_retry:
            base_confidence *= 0.8

        if model_name in context.recent_failures:
            base_confidence *= 0.7

        # Adjust for prediction recency
        if self.prediction_errors:
            recent_error_rate = np.mean(list(self.prediction_errors[-10:]))
            base_confidence *= (1 - min(0.3, recent_error_rate))

        return base_confidence

    def _calculate_cost_efficiency(self, cost: float, quality: float, tps: float) -> float:
        """Calculate cost efficiency score."""
        # Quality per dollar
        quality_per_dollar = quality / max(0.0001, cost)

        # Tokens per dollar
        tokens_per_dollar = tps / max(0.0001, cost)

        # Combined efficiency (normalized)
        efficiency = (
            (quality_per_dollar / 1000) * 0.7 +  # Quality weighted more
            (tokens_per_dollar / 100000) * 0.3
        )

        return min(1.0, efficiency)

    def _calculate_performance_score(self, latency: float, tps: float, quality: float) -> float:
        """Calculate overall performance score."""
        # Normalize metrics
        latency_score = max(0, 1 - (latency / 5000))  # 5s as worst case
        tps_score = min(1, tps / 200)  # 200 TPS as excellent

        # Weighted combination
        return (quality * 0.5) + (tps_score * 0.3) + (latency_score * 0.2)

    def _calculate_overall_score(self,
                                context: ModelSelectionContext,
                                cost: float,
                                latency: float,
                                quality: float,
                                tps: float) -> float:
        """Calculate overall score based on optimization objective."""

        if context.optimization_objective == OptimizationObjective.MINIMIZE_COST:
            # Inverse cost normalized
            return max(0, 1 - (cost / 0.1))  # $0.10 as expensive threshold

        elif context.optimization_objective == OptimizationObjective.MAXIMIZE_QUALITY:
            return quality

        elif context.optimization_objective == OptimizationObjective.MINIMIZE_LATENCY:
            return max(0, 1 - (latency / 5000))

        elif context.optimization_objective == OptimizationObjective.MAXIMIZE_THROUGHPUT:
            return min(1, tps / 200)

        elif context.optimization_objective == OptimizationObjective.COST_PERFORMANCE_RATIO:
            return self._calculate_cost_efficiency(cost, quality, tps)

        elif context.optimization_objective == OptimizationObjective.CONTEXT_EFFICIENT:
            # Favor models that fit well within context
            context_score = 1 - context.context_constraints.context_utilization
            return (context_score * 0.5) + (quality * 0.3) + (min(1, tps/100) * 0.2)

        else:  # BALANCED
            cost_score = max(0, 1 - (cost / 0.05))
            latency_score = max(0, 1 - (latency / 2000))
            tps_score = min(1, tps / 100)

            return (quality * 0.35) + (cost_score * 0.25) + (latency_score * 0.2) + (tps_score * 0.2)

    async def select_optimal_model(self, context: ModelSelectionContext) -> tuple[str, PredictedPerformance]:
        """
        Select the optimal model for the given context.

        Args:
            context: Complete selection context

        Returns:
            Tuple of (selected_model_name, predicted_performance)
        """
        candidates = []

        # Evaluate all available models
        for model_name in self.model_catalog.keys():
            try:
                prediction = await self.predict_performance(context, model_name)

                # Skip models that don't fit constraints
                if not prediction.context_fit:
                    logger.debug(f"Skipping {model_name}: doesn't fit context window")
                    continue

                if not prediction.budget_fit:
                    logger.debug(f"Skipping {model_name}: exceeds budget")
                    continue

                # Skip models with high risk
                if prediction.timeout_risk > 0.7:
                    logger.debug(f"Skipping {model_name}: high timeout risk")
                    continue

                if prediction.quality_risk > 0.3:
                    logger.debug(f"Skipping {model_name}: quality below target")
                    continue

                candidates.append(prediction)

            except Exception as e:
                logger.warning(f"Error evaluating model {model_name}: {e}")
                continue

        if not candidates:
            # Fallback to cheapest model that fits
            logger.warning("No models meet all constraints, using fallback")
            return await self._select_fallback_model(context)

        # Sort by overall score
        candidates.sort(key=lambda x: x.overall_score, reverse=True)

        # Select best candidate
        selected = candidates[0]

        # Log selection
        logger.info(f"Selected {selected.model_name} with score {selected.overall_score:.3f}")
        logger.debug(f"  Cost: ${selected.predicted_cost:.4f}, "
                    f"Latency: {selected.predicted_latency_ms:.0f}ms, "
                    f"Quality: {selected.predicted_quality:.2f}, "
                    f"TPS: {selected.predicted_tps:.1f}")

        # Track selection
        self.selection_history.append({
            'timestamp': datetime.now(timezone.utc),
            'context': context,
            'selected_model': selected.model_name,
            'prediction': selected,
            'alternatives': len(candidates) - 1
        })

        return selected.model_name, selected

    async def _select_fallback_model(self, context: ModelSelectionContext) -> tuple[str, PredictedPerformance]:
        """Select a fallback model when no candidates meet constraints."""
        # Try progressively larger models until one fits
        fallback_order = [
            "gemini-2.5-flash",
            "gpt-4o-mini",
            "claude-3-5-haiku-20241022",
            "gpt-4o",
            "gemini-2.5-pro"  # Massive context as last resort
        ]

        for model_name in fallback_order:
            model_info = self.model_catalog.get(model_name, {})
            if (context.input_tokens + context.expected_output_tokens <=
                model_info.get("context_window", 32768)):

                prediction = await self.predict_performance(context, model_name)
                return model_name, prediction

        # Ultimate fallback
        model_name = "gemini-2.5-flash"
        prediction = await self.predict_performance(context, model_name)
        return model_name, prediction

    async def record_actual_performance(self,
                                       request_id: str,
                                       model_name: str,
                                       actual_metrics: ModelPerformanceMetrics):
        """
        Record actual performance and update learning.

        Args:
            request_id: Request ID from context
            model_name: Model that was used
            actual_metrics: Actual performance metrics
        """
        # Find corresponding prediction
        selection = None
        for item in self.selection_history:
            if item['context'].request_id == request_id:
                selection = item
                break

        if not selection:
            logger.warning(f"No selection history found for request {request_id}")
            return

        prediction = selection['prediction']

        # Calculate prediction errors
        cost_error = abs(prediction.predicted_cost - actual_metrics.total_cost) / max(0.0001, actual_metrics.total_cost)
        latency_error = abs(prediction.predicted_latency_ms - actual_metrics.total_time * 1000) / max(1, actual_metrics.total_time * 1000)
        quality_error = abs(prediction.predicted_quality - actual_metrics.quality_score)
        tps_error = abs(prediction.predicted_tps - actual_metrics.tokens_per_second) / max(1, actual_metrics.tokens_per_second)

        # Track errors
        mean_error = np.mean([cost_error, latency_error, quality_error, tps_error])
        self.prediction_errors.append(mean_error)

        # Store actual performance data for training
        self.actual_performance_data.append({
            'context': selection['context'],
            'model_name': model_name,
            'features': self._extract_features(selection['context'], model_name),
            'actual_cost': actual_metrics.total_cost,
            'actual_latency': actual_metrics.total_time * 1000,
            'actual_quality': actual_metrics.quality_score,
            'actual_tps': actual_metrics.tokens_per_second,
            'actual_success': float(actual_metrics.success),
            'timestamp': datetime.now(timezone.utc)
        })

        # Limit stored data size
        if len(self.actual_performance_data) > 10000:
            self.actual_performance_data = self.actual_performance_data[-10000:]

        # Log significant errors for debugging
        if mean_error > 0.3:
            logger.warning(f"High prediction error ({mean_error:.2f}) for {model_name}")
            logger.debug(f"  Cost error: {cost_error:.2f}, Latency error: {latency_error:.2f}")
            logger.debug(f"  Quality error: {quality_error:.2f}, TPS error: {tps_error:.2f}")

        # Update model if enough data collected
        if len(self.prediction_errors) >= self.update_frequency:
            await self._update_model()

    def _prepare_training_data(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare training data from actual performance history.

        Returns:
            Tuple of (features, targets) arrays or (None, None) if insufficient data
        """
        if not self.actual_performance_data:
            return None, None

        try:
            # Extract features and targets from stored data
            X = []
            y = []

            for data in self.actual_performance_data:
                # Features are already extracted and stored
                X.append(data['features'])

                # Normalize targets for better training
                # Cost: scale to 0-1 range (assuming max $10 per request)
                cost_normalized = min(data['actual_cost'] / 10.0, 1.0)

                # Latency: scale to 0-1 range (assuming max 10000ms)
                latency_normalized = min(data['actual_latency'] / 10000.0, 1.0)

                # Quality: already 0-1
                quality = data['actual_quality']

                # TPS: scale to 0-1 range (assuming max 200 TPS)
                tps_normalized = min(data['actual_tps'] / 200.0, 1.0)

                # Success: already 0-1
                success = data['actual_success']

                # Target vector matches PerformancePredictor output structure
                targets = [
                    cost_normalized,
                    latency_normalized,
                    quality,
                    tps_normalized,
                    success
                ]
                y.append(targets)

            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            # Apply feature scaling if scaler available
            if self.scaler and SKLEARN_AVAILABLE:
                # Fit scaler on current data
                X = self.scaler.fit_transform(X)

            return X, y

        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return None, None

    async def _update_model(self):
        """Update the neural network with recent data."""
        if not TORCH_AVAILABLE or not self.predictor:
            return

        # Check if we have enough reconciled data
        if len(self.actual_performance_data) < 20:
            logger.debug(f"Not enough training data yet: {len(self.actual_performance_data)}/20")
            return

        logger.info("Starting model update with recent performance data...")

        try:
            # Prepare training data
            X, y = self._prepare_training_data()

            if X is None or y is None:
                logger.warning("Failed to prepare training data")
                return

            # Split into train/validation
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)

            # Training parameters
            batch_size = min(32, len(X_train))
            epochs = 50
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0

            # Setup optimizer
            optimizer = torch.optim.Adam(self.predictor.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # Training loop
            for epoch in range(epochs):
                # Training phase
                self.predictor.train()
                train_loss = 0.0

                # Process in batches
                for i in range(0, len(X_train_tensor), batch_size):
                    batch_X = X_train_tensor[i:i+batch_size]
                    batch_y = y_train_tensor[i:i+batch_size]

                    optimizer.zero_grad()
                    outputs = self.predictor(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                avg_train_loss = train_loss / (len(X_train_tensor) / batch_size)

                # Validation phase
                self.predictor.eval()
                with torch.no_grad():
                    val_outputs = self.predictor(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self._save_model()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

                if epoch % 10 == 0:
                    logger.debug(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}")

            logger.info(f"Model updated successfully. Final validation loss: {best_val_loss:.4f}")

            # Update learning metrics
            self.learning_metrics['updates'] += 1
            self.learning_metrics['last_update'] = datetime.now(timezone.utc)
            self.learning_metrics['training_samples'] = len(X)
            self.learning_metrics['validation_loss'] = best_val_loss

            # Clear prediction errors after successful update
            self.prediction_errors = []

        except Exception as e:
            logger.error(f"Failed to update model: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        # Clear old predictions from cache
        current_time = datetime.now(timezone.utc)
        expired_keys = []
        for key, value in self.prediction_cache.items():
            if (current_time - value['timestamp']).seconds > self.cache_ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self.prediction_cache[key]

    def _save_model(self):
        """Save model weights to disk."""
        if not TORCH_AVAILABLE or not self.predictor:
            return

        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save({
                'model_state_dict': self.predictor.state_dict(),
                'scaler_state': self.scaler.__dict__ if self.scaler else None,
                'metadata': {
                    'version': '1.0.0',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'training_samples': len(self.selection_history)
                }
            }, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def _load_model(self):
        """Load model weights from disk."""
        if not TORCH_AVAILABLE or not self.predictor:
            return

        if not os.path.exists(self.model_path):
            logger.info("No pre-trained model found, starting fresh")
            return

        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.predictor.load_state_dict(checkpoint['model_state_dict'])

            if self.scaler and checkpoint.get('scaler_state'):
                self.scaler.__dict__.update(checkpoint['scaler_state'])

            logger.info(f"Model loaded from {self.model_path}")
            logger.debug(f"Model metadata: {checkpoint.get('metadata', {})}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def get_performance_summary(self) -> dict[str, Any]:
        """Get summary of engine performance."""
        if not self.selection_history:
            return {"status": "no_data"}

        recent_selections = self.selection_history[-100:]

        # Model usage distribution
        model_counts = {}
        for item in recent_selections:
            model = item['selected_model']
            model_counts[model] = model_counts.get(model, 0) + 1

        # Optimization objective distribution
        objective_counts = {}
        for item in recent_selections:
            obj = item['context'].optimization_objective.value
            objective_counts[obj] = objective_counts.get(obj, 0) + 1

        # Average scores
        avg_scores = {
            'overall': np.mean([item['prediction'].overall_score for item in recent_selections]),
            'cost_efficiency': np.mean([item['prediction'].cost_efficiency_score for item in recent_selections]),
            'performance': np.mean([item['prediction'].performance_score for item in recent_selections])
        }

        # Prediction accuracy
        recent_errors = self.prediction_errors[-100:] if self.prediction_errors else []

        return {
            'total_selections': len(self.selection_history),
            'recent_selections': len(recent_selections),
            'model_distribution': model_counts,
            'objective_distribution': objective_counts,
            'average_scores': avg_scores,
            'prediction_accuracy': 1 - np.mean(recent_errors) if recent_errors else None,
            'cache_size': len(self.prediction_cache),
            'last_update': self.selection_history[-1]['timestamp'].isoformat() if self.selection_history else None
        }


# Global instance
_adaptive_engine = None


def get_adaptive_engine() -> AdaptiveLearningEngine:
    """Get or create the global adaptive learning engine."""
    global _adaptive_engine
    if _adaptive_engine is None:
        _adaptive_engine = AdaptiveLearningEngine()
    return _adaptive_engine
