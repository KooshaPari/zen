"""
Cost-Performance Optimizer for Multi-Dimensional Model Selection

This module implements sophisticated optimization algorithms to balance cost,
performance, and time constraints while respecting context limits. It uses
historical data and real-time metrics to make optimal model selection decisions.

Key Features:
- Pareto-optimal frontier analysis
- Dynamic weight adjustment based on patterns
- Cost-per-quality-unit optimization
- Time-sensitive routing decisions
- Budget-aware model selection
- ROI (Return on Investment) calculations
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class OptimizationMode(Enum):
    """Optimization modes for different scenarios."""
    BUDGET_CONSTRAINED = "budget_constrained"  # Strict budget limits
    QUALITY_FIRST = "quality_first"  # Maximize quality within reason
    SPEED_CRITICAL = "speed_critical"  # Minimize latency
    BALANCED_EFFICIENCY = "balanced_efficiency"  # Balance all factors
    COST_MINIMIZE = "cost_minimize"  # Absolute minimum cost
    THROUGHPUT_MAXIMIZE = "throughput_maximize"  # Maximum tokens/second
    ADAPTIVE = "adaptive"  # Learn from patterns


@dataclass
class PerformanceMetric:
    """Single performance metric with metadata."""
    value: float
    unit: str
    timestamp: datetime
    confidence: float = 1.0

    def age_hours(self) -> float:
        """Get age of metric in hours."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds() / 3600


@dataclass
class ModelPerformanceProfile:
    """Complete performance profile for a model."""
    model_name: str

    # Cost metrics (per 1K tokens)
    input_cost: PerformanceMetric
    output_cost: PerformanceMetric

    # Performance metrics
    latency_p50: PerformanceMetric  # Median latency
    latency_p95: PerformanceMetric  # 95th percentile latency
    latency_p99: PerformanceMetric  # 99th percentile latency

    tokens_per_second: PerformanceMetric
    time_to_first_token: PerformanceMetric

    # Quality metrics
    quality_score: PerformanceMetric  # 0-1 scale
    success_rate: PerformanceMetric  # 0-1 scale
    retry_rate: PerformanceMetric  # 0-1 scale

    # Context metrics
    max_context_tokens: int
    typical_context_usage: PerformanceMetric  # Average usage ratio

    # Derived metrics
    cost_per_quality: Optional[float] = None
    quality_per_second: Optional[float] = None
    efficiency_score: Optional[float] = None

    def calculate_derived_metrics(self):
        """Calculate derived performance metrics."""
        # Cost per quality unit
        avg_cost = (self.input_cost.value + self.output_cost.value) / 2
        self.cost_per_quality = avg_cost / max(0.01, self.quality_score.value)

        # Quality delivered per second
        self.quality_per_second = (
            self.quality_score.value * self.tokens_per_second.value
        )

        # Overall efficiency score
        self.efficiency_score = (
            (self.quality_score.value * 0.4) +
            (1 / max(1, self.cost_per_quality) * 0.3) +
            (min(1, self.tokens_per_second.value / 100) * 0.2) +
            (self.success_rate.value * 0.1)
        )


@dataclass
class OptimizationConstraints:
    """Constraints for optimization."""
    max_cost_per_request: Optional[float] = None
    max_latency_ms: Optional[int] = None
    min_quality_score: Optional[float] = None
    min_success_rate: Optional[float] = None
    max_context_tokens: Optional[int] = None

    # Budget constraints
    remaining_daily_budget: Optional[float] = None
    remaining_monthly_budget: Optional[float] = None

    # Time constraints
    deadline_ms: Optional[int] = None  # Must complete by

    # Throughput constraints
    min_tokens_per_second: Optional[float] = None

    def is_satisfied_by(self, profile: ModelPerformanceProfile,
                       estimated_tokens: int) -> tuple[bool, list[str]]:
        """
        Check if a model profile satisfies constraints.

        Returns:
            Tuple of (satisfied, list_of_violations)
        """
        violations = []

        # Cost check
        if self.max_cost_per_request:
            estimated_cost = (
                (estimated_tokens / 1000) *
                (profile.input_cost.value + profile.output_cost.value) / 2
            )
            if estimated_cost > self.max_cost_per_request:
                violations.append(f"Cost ${estimated_cost:.4f} > ${self.max_cost_per_request:.4f}")

        # Latency check
        if self.max_latency_ms and profile.latency_p50.value > self.max_latency_ms:
            violations.append(f"Latency {profile.latency_p50.value}ms > {self.max_latency_ms}ms")

        # Quality check
        if self.min_quality_score and profile.quality_score.value < self.min_quality_score:
            violations.append(f"Quality {profile.quality_score.value:.2f} < {self.min_quality_score:.2f}")

        # Success rate check
        if self.min_success_rate and profile.success_rate.value < self.min_success_rate:
            violations.append(f"Success rate {profile.success_rate.value:.2f} < {self.min_success_rate:.2f}")

        # Context check
        if self.max_context_tokens and estimated_tokens > self.max_context_tokens:
            violations.append(f"Tokens {estimated_tokens} > {self.max_context_tokens}")

        # Throughput check
        if self.min_tokens_per_second and profile.tokens_per_second.value < self.min_tokens_per_second:
            violations.append(f"TPS {profile.tokens_per_second.value:.1f} < {self.min_tokens_per_second:.1f}")

        return len(violations) == 0, violations


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    selected_model: str
    optimization_score: float

    # Component scores
    cost_score: float
    performance_score: float
    quality_score: float

    # Predicted metrics
    predicted_cost: float
    predicted_latency_ms: float
    predicted_quality: float
    predicted_tps: float

    # Decision factors
    primary_factor: str  # What drove the decision
    tradeoffs: dict[str, float]  # What was traded off

    # Alternative options
    alternatives: list[dict[str, Any]]
    pareto_optimal: bool = False

    # Confidence
    confidence: float = 0.0
    reasoning: str = ""


class ParetoFrontier:
    """Manages Pareto-optimal frontier for multi-objective optimization."""

    def __init__(self, objectives: list[str]):
        """
        Initialize Pareto frontier.

        Args:
            objectives: List of objective names to optimize
        """
        self.objectives = objectives
        self.points = []  # List of (model_name, objective_values)
        self.frontier = []  # Pareto-optimal points

    def add_point(self, model_name: str, values: dict[str, float]):
        """Add a point to consideration."""
        objective_values = [values[obj] for obj in self.objectives]
        self.points.append((model_name, objective_values))

    def compute_frontier(self, minimize: Optional[set[str]] = None):
        """
        Compute Pareto-optimal frontier.

        Args:
            minimize: Set of objectives to minimize (others maximized)
        """
        minimize = minimize or set()
        self.frontier = []

        for i, (model_i, values_i) in enumerate(self.points):
            is_dominated = False

            for j, (_model_j, values_j) in enumerate(self.points):
                if i == j:
                    continue

                # Check if j dominates i
                dominates = True
                strictly_better = False

                for k, obj in enumerate(self.objectives):
                    if obj in minimize:
                        # Lower is better
                        if values_i[k] < values_j[k]:
                            dominates = False
                            break
                        elif values_i[k] > values_j[k]:
                            strictly_better = True
                    else:
                        # Higher is better
                        if values_i[k] > values_j[k]:
                            dominates = False
                            break
                        elif values_i[k] < values_j[k]:
                            strictly_better = True

                if dominates and strictly_better:
                    is_dominated = True
                    break

            if not is_dominated:
                self.frontier.append((model_i, values_i))

    def get_optimal_points(self) -> list[tuple[str, dict[str, float]]]:
        """Get Pareto-optimal points with objective values."""
        result = []
        for model, values in self.frontier:
            obj_dict = {self.objectives[i]: values[i] for i in range(len(self.objectives))}
            result.append((model, obj_dict))
        return result


class CostPerformanceOptimizer:
    """
    Main optimizer for balancing cost, performance, and time.
    """

    def __init__(self):
        """Initialize the optimizer."""
        self.model_profiles = {}
        self.optimization_history = []
        self.weight_adaptation = WeightAdapter()

        # Default weights for different modes
        self.mode_weights = {
            OptimizationMode.BUDGET_CONSTRAINED: {
                'cost': 0.6, 'performance': 0.2, 'quality': 0.2
            },
            OptimizationMode.QUALITY_FIRST: {
                'cost': 0.2, 'performance': 0.2, 'quality': 0.6
            },
            OptimizationMode.SPEED_CRITICAL: {
                'cost': 0.2, 'performance': 0.6, 'quality': 0.2
            },
            OptimizationMode.BALANCED_EFFICIENCY: {
                'cost': 0.33, 'performance': 0.33, 'quality': 0.34
            },
            OptimizationMode.COST_MINIMIZE: {
                'cost': 0.8, 'performance': 0.1, 'quality': 0.1
            },
            OptimizationMode.THROUGHPUT_MAXIMIZE: {
                'cost': 0.2, 'performance': 0.7, 'quality': 0.1
            }
        }

    def update_model_profile(self, model_name: str, metrics: dict[str, Any]):
        """
        Update performance profile for a model.

        Args:
            model_name: Name of the model
            metrics: Performance metrics
        """
        current_time = datetime.now(timezone.utc)

        # Create or update profile
        if model_name not in self.model_profiles:
            profile = ModelPerformanceProfile(
                model_name=model_name,
                input_cost=PerformanceMetric(
                    metrics.get('input_cost', 0.001), 'USD/1K', current_time
                ),
                output_cost=PerformanceMetric(
                    metrics.get('output_cost', 0.002), 'USD/1K', current_time
                ),
                latency_p50=PerformanceMetric(
                    metrics.get('latency_p50', 1000), 'ms', current_time
                ),
                latency_p95=PerformanceMetric(
                    metrics.get('latency_p95', 2000), 'ms', current_time
                ),
                latency_p99=PerformanceMetric(
                    metrics.get('latency_p99', 3000), 'ms', current_time
                ),
                tokens_per_second=PerformanceMetric(
                    metrics.get('tps', 50), 'tokens/s', current_time
                ),
                time_to_first_token=PerformanceMetric(
                    metrics.get('ttft', 500), 'ms', current_time
                ),
                quality_score=PerformanceMetric(
                    metrics.get('quality', 0.8), 'score', current_time
                ),
                success_rate=PerformanceMetric(
                    metrics.get('success_rate', 0.95), 'ratio', current_time
                ),
                retry_rate=PerformanceMetric(
                    metrics.get('retry_rate', 0.05), 'ratio', current_time
                ),
                max_context_tokens=metrics.get('max_context', 32768),
                typical_context_usage=PerformanceMetric(
                    metrics.get('context_usage', 0.3), 'ratio', current_time
                )
            )
        else:
            profile = self.model_profiles[model_name]
            # Update existing metrics
            if 'input_cost' in metrics:
                profile.input_cost = PerformanceMetric(metrics['input_cost'], 'USD/1K', current_time)
            if 'latency_p50' in metrics:
                profile.latency_p50 = PerformanceMetric(metrics['latency_p50'], 'ms', current_time)
            if 'tps' in metrics:
                profile.tokens_per_second = PerformanceMetric(metrics['tps'], 'tokens/s', current_time)
            if 'quality' in metrics:
                profile.quality_score = PerformanceMetric(metrics['quality'], 'score', current_time)

        # Calculate derived metrics
        profile.calculate_derived_metrics()

        self.model_profiles[model_name] = profile

    def optimize(self,
                estimated_tokens: int,
                mode: OptimizationMode = OptimizationMode.BALANCED_EFFICIENCY,
                constraints: Optional[OptimizationConstraints] = None,
                available_models: Optional[list[str]] = None) -> OptimizationResult:
        """
        Perform optimization to select best model.

        Args:
            estimated_tokens: Estimated total tokens for request
            mode: Optimization mode
            constraints: Optimization constraints
            available_models: List of available models to consider

        Returns:
            Optimization result with selected model
        """
        constraints = constraints or OptimizationConstraints()

        # Get candidate models
        if available_models:
            candidates = [m for m in available_models if m in self.model_profiles]
        else:
            candidates = list(self.model_profiles.keys())

        if not candidates:
            logger.warning("No models available for optimization")
            return self._create_fallback_result()

        # Filter by hard constraints
        valid_candidates = []
        for model_name in candidates:
            profile = self.model_profiles[model_name]
            satisfied, violations = constraints.is_satisfied_by(profile, estimated_tokens)

            if satisfied:
                valid_candidates.append(model_name)
            else:
                logger.debug(f"Model {model_name} violates constraints: {violations}")

        if not valid_candidates:
            logger.warning("No models satisfy all constraints, relaxing...")
            valid_candidates = candidates  # Use all candidates

        # Perform optimization based on mode
        if mode == OptimizationMode.ADAPTIVE:
            weights = self.weight_adaptation.get_adapted_weights(self.optimization_history)
        else:
            weights = self.mode_weights.get(mode, self.mode_weights[OptimizationMode.BALANCED_EFFICIENCY])

        # Score each candidate
        scored_candidates = []
        for model_name in valid_candidates:
            profile = self.model_profiles[model_name]
            score, components = self._score_model(profile, estimated_tokens, weights, constraints)
            scored_candidates.append((model_name, score, components, profile))

        # Sort by score (higher is better)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Check for Pareto optimality
        pareto = ParetoFrontier(['cost', 'performance', 'quality'])
        for model_name, _, components, _ in scored_candidates:
            pareto.add_point(model_name, {
                'cost': -components['cost'],  # Negative because we want to minimize
                'performance': components['performance'],
                'quality': components['quality']
            })

        pareto.compute_frontier(minimize={'cost'})
        pareto_models = {model for model, _ in pareto.get_optimal_points()}

        # Select best model
        best_model, best_score, best_components, best_profile = scored_candidates[0]

        # Calculate predicted metrics
        predicted_cost = (estimated_tokens / 1000) * (
            (best_profile.input_cost.value + best_profile.output_cost.value) / 2
        )

        # Create result
        result = OptimizationResult(
            selected_model=best_model,
            optimization_score=best_score,
            cost_score=best_components['cost'],
            performance_score=best_components['performance'],
            quality_score=best_components['quality'],
            predicted_cost=predicted_cost,
            predicted_latency_ms=best_profile.latency_p50.value,
            predicted_quality=best_profile.quality_score.value,
            predicted_tps=best_profile.tokens_per_second.value,
            primary_factor=self._identify_primary_factor(best_components, weights),
            tradeoffs=self._calculate_tradeoffs(scored_candidates),
            alternatives=[
                {
                    'model': name,
                    'score': score,
                    'pareto_optimal': name in pareto_models
                }
                for name, score, _, _ in scored_candidates[1:6]  # Top 5 alternatives
            ],
            pareto_optimal=best_model in pareto_models,
            confidence=self._calculate_confidence(best_profile),
            reasoning=self._generate_reasoning(best_model, best_components, mode)
        )

        # Record optimization
        self.optimization_history.append({
            'timestamp': datetime.now(timezone.utc),
            'result': result,
            'mode': mode,
            'constraints': constraints,
            'tokens': estimated_tokens
        })

        return result

    def _score_model(self,
                    profile: ModelPerformanceProfile,
                    estimated_tokens: int,
                    weights: dict[str, float],
                    constraints: OptimizationConstraints) -> tuple[float, dict[str, float]]:
        """
        Score a model based on weighted criteria.

        Returns:
            Tuple of (overall_score, component_scores)
        """
        # Cost score (inverse - lower cost = higher score)
        estimated_cost = (estimated_tokens / 1000) * (
            (profile.input_cost.value + profile.output_cost.value) / 2
        )

        # Normalize cost (assume $0.10 is expensive, $0.001 is cheap)
        cost_score = max(0, 1 - (estimated_cost / 0.10))

        # Performance score (based on latency and throughput)
        latency_score = max(0, 1 - (profile.latency_p50.value / 5000))  # 5s as worst
        tps_score = min(1, profile.tokens_per_second.value / 200)  # 200 TPS as best
        ttft_score = max(0, 1 - (profile.time_to_first_token.value / 2000))  # 2s as worst

        performance_score = (
            latency_score * 0.3 +
            tps_score * 0.5 +
            ttft_score * 0.2
        )

        # Quality score (already 0-1)
        quality_score = profile.quality_score.value

        # Apply constraint penalties
        if constraints.max_cost_per_request and estimated_cost > constraints.max_cost_per_request:
            cost_score *= 0.5  # Penalty for exceeding budget

        if constraints.max_latency_ms and profile.latency_p50.value > constraints.max_latency_ms:
            performance_score *= 0.5  # Penalty for being too slow

        if constraints.min_quality_score and quality_score < constraints.min_quality_score:
            quality_score *= 0.5  # Penalty for low quality

        # Calculate weighted overall score
        overall_score = (
            cost_score * weights.get('cost', 0.33) +
            performance_score * weights.get('performance', 0.33) +
            quality_score * weights.get('quality', 0.34)
        )

        components = {
            'cost': cost_score,
            'performance': performance_score,
            'quality': quality_score
        }

        return overall_score, components

    def _identify_primary_factor(self,
                                components: dict[str, float],
                                weights: dict[str, float]) -> str:
        """Identify which factor contributed most to selection."""
        weighted_contributions = {
            factor: score * weights.get(factor, 0)
            for factor, score in components.items()
        }

        return max(weighted_contributions, key=weighted_contributions.get)

    def _calculate_tradeoffs(self, scored_candidates: list) -> dict[str, float]:
        """Calculate what was traded off in the selection."""
        if len(scored_candidates) < 2:
            return {}

        selected = scored_candidates[0]
        alternatives = scored_candidates[1:]

        tradeoffs = {}

        # Compare selected with alternatives
        _, _, selected_components, selected_profile = selected

        for _, _, alt_components, _alt_profile in alternatives:
            # Cost tradeoff
            if alt_components['cost'] > selected_components['cost']:
                tradeoffs['cost_savings'] = max(
                    tradeoffs.get('cost_savings', 0),
                    alt_components['cost'] - selected_components['cost']
                )

            # Performance tradeoff
            if alt_components['performance'] > selected_components['performance']:
                tradeoffs['performance_loss'] = max(
                    tradeoffs.get('performance_loss', 0),
                    alt_components['performance'] - selected_components['performance']
                )

            # Quality tradeoff
            if alt_components['quality'] > selected_components['quality']:
                tradeoffs['quality_loss'] = max(
                    tradeoffs.get('quality_loss', 0),
                    alt_components['quality'] - selected_components['quality']
                )

        return tradeoffs

    def _calculate_confidence(self, profile: ModelPerformanceProfile) -> float:
        """Calculate confidence in the optimization decision."""
        confidence = 1.0

        # Reduce confidence for old data
        for metric in [profile.input_cost, profile.latency_p50, profile.quality_score]:
            age = metric.age_hours()
            if age > 24:
                confidence *= 0.9
            if age > 72:
                confidence *= 0.8
            if age > 168:  # 1 week
                confidence *= 0.7

        # Reduce confidence for low success rate
        if profile.success_rate.value < 0.9:
            confidence *= profile.success_rate.value

        # Reduce confidence for high retry rate
        if profile.retry_rate.value > 0.1:
            confidence *= (1 - profile.retry_rate.value)

        return max(0.1, confidence)

    def _generate_reasoning(self,
                          model: str,
                          components: dict[str, float],
                          mode: OptimizationMode) -> str:
        """Generate human-readable reasoning for the selection."""
        reasons = []

        if mode == OptimizationMode.BUDGET_CONSTRAINED:
            reasons.append("Operating under budget constraints")
        elif mode == OptimizationMode.QUALITY_FIRST:
            reasons.append("Prioritizing output quality")
        elif mode == OptimizationMode.SPEED_CRITICAL:
            reasons.append("Optimizing for minimum latency")

        # Component-based reasoning
        if components['cost'] > 0.8:
            reasons.append("Excellent cost efficiency")
        elif components['cost'] < 0.3:
            reasons.append("Higher cost but justified by requirements")

        if components['performance'] > 0.8:
            reasons.append("Superior performance characteristics")
        elif components['performance'] < 0.3:
            reasons.append("Acceptable performance for use case")

        if components['quality'] > 0.8:
            reasons.append("High quality output expected")
        elif components['quality'] < 0.5:
            reasons.append("Quality trade-off for other benefits")

        return f"Selected {model}: " + "; ".join(reasons)

    def _create_fallback_result(self) -> OptimizationResult:
        """Create a fallback result when optimization fails."""
        return OptimizationResult(
            selected_model="gemini-2.5-flash",  # Default fallback
            optimization_score=0.5,
            cost_score=0.8,
            performance_score=0.7,
            quality_score=0.6,
            predicted_cost=0.001,
            predicted_latency_ms=500,
            predicted_quality=0.7,
            predicted_tps=100,
            primary_factor="fallback",
            tradeoffs={},
            alternatives=[],
            pareto_optimal=False,
            confidence=0.3,
            reasoning="Fallback selection due to optimization failure"
        )

    def get_roi_analysis(self,
                        model_name: str,
                        estimated_tokens: int,
                        value_per_quality_unit: float = 1.0) -> dict[str, float]:
        """
        Calculate ROI (Return on Investment) for a model.

        Args:
            model_name: Model to analyze
            estimated_tokens: Estimated tokens
            value_per_quality_unit: Business value per quality unit

        Returns:
            ROI analysis metrics
        """
        if model_name not in self.model_profiles:
            return {'error': 'Model profile not found'}

        profile = self.model_profiles[model_name]

        # Calculate costs
        estimated_cost = (estimated_tokens / 1000) * (
            (profile.input_cost.value + profile.output_cost.value) / 2
        )

        # Calculate value
        quality_delivered = profile.quality_score.value
        tokens_delivered = estimated_tokens * profile.success_rate.value

        value = quality_delivered * value_per_quality_unit

        # ROI calculation
        roi = ((value - estimated_cost) / estimated_cost) * 100 if estimated_cost > 0 else 0

        # Time to value
        time_to_value_seconds = profile.latency_p50.value / 1000

        # Value per second
        value_per_second = value / time_to_value_seconds if time_to_value_seconds > 0 else 0

        return {
            'estimated_cost': estimated_cost,
            'estimated_value': value,
            'roi_percentage': roi,
            'time_to_value_seconds': time_to_value_seconds,
            'value_per_second': value_per_second,
            'quality_delivered': quality_delivered,
            'tokens_delivered': tokens_delivered,
            'cost_per_quality': profile.cost_per_quality or 0,
            'break_even_quality': estimated_cost / value_per_quality_unit
        }


class WeightAdapter:
    """Adapts optimization weights based on historical patterns."""

    def __init__(self):
        """Initialize weight adapter."""
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.weight_history = []
        self.performance_feedback = []

    def get_adapted_weights(self, history: list[dict]) -> dict[str, float]:
        """
        Get adapted weights based on historical performance.

        Args:
            history: Optimization history

        Returns:
            Adapted weights
        """
        if len(history) < 10:
            # Not enough history, use balanced weights
            return {'cost': 0.33, 'performance': 0.33, 'quality': 0.34}

        # Analyze recent performance
        recent = history[-20:]

        # Calculate average scores by primary factor
        factor_scores = {'cost': [], 'performance': [], 'quality': []}

        for entry in recent:
            result = entry['result']
            primary = result.primary_factor
            score = result.optimization_score

            if primary in factor_scores:
                factor_scores[primary].append(score)

        # Calculate mean scores
        mean_scores = {
            factor: np.mean(scores) if scores else 0.5
            for factor, scores in factor_scores.items()
        }

        # Adjust weights based on performance
        current_weights = {'cost': 0.33, 'performance': 0.33, 'quality': 0.34}

        for factor in current_weights:
            if mean_scores[factor] < 0.6:
                # This factor is underperforming, increase its weight
                current_weights[factor] *= (1 + self.learning_rate)
            elif mean_scores[factor] > 0.8:
                # This factor is doing well, can reduce weight slightly
                current_weights[factor] *= (1 - self.learning_rate * 0.5)

        # Normalize weights to sum to 1
        total = sum(current_weights.values())
        normalized = {k: v/total for k, v in current_weights.items()}

        # Apply momentum from history
        if self.weight_history:
            prev_weights = self.weight_history[-1]
            for factor in normalized:
                normalized[factor] = (
                    self.momentum * prev_weights.get(factor, 0.33) +
                    (1 - self.momentum) * normalized[factor]
                )

        self.weight_history.append(normalized)

        return normalized


# Global instance
_optimizer = None


def get_cost_performance_optimizer() -> CostPerformanceOptimizer:
    """Get or create the global cost-performance optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = CostPerformanceOptimizer()
    return _optimizer
