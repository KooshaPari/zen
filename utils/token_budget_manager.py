"""
Token Budget Manager for Allocation and Optimization

This module manages token budgets across requests, models, and time periods,
ensuring efficient allocation while respecting constraints and budgets.

Key Features:
- Daily/monthly token budget tracking
- Per-request token allocation
- Model-specific token limits
- Budget overflow prevention
- Historical usage analysis
- Predictive budget planning
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BudgetPeriod(Enum):
    """Budget tracking periods."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class AllocationStrategy(Enum):
    """Token allocation strategies."""
    FIXED = "fixed"  # Fixed allocation per request
    PROPORTIONAL = "proportional"  # Proportional to request complexity
    DYNAMIC = "dynamic"  # Dynamic based on usage patterns
    PRIORITY = "priority"  # Priority-based allocation
    ADAPTIVE = "adaptive"  # Learn from historical patterns


@dataclass
class TokenBudget:
    """Token budget configuration."""
    period: BudgetPeriod
    total_tokens: int
    allocated_tokens: int = 0
    used_tokens: int = 0
    reserved_tokens: int = 0

    # Cost budgets
    total_budget_usd: Optional[float] = None
    allocated_budget_usd: float = 0.0
    used_budget_usd: float = 0.0

    # Time window
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_end: Optional[datetime] = None

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def available_tokens(self) -> int:
        """Calculate available tokens."""
        return max(0, self.total_tokens - self.allocated_tokens - self.reserved_tokens)

    @property
    def utilization_rate(self) -> float:
        """Calculate utilization rate."""
        if self.total_tokens == 0:
            return 0.0
        return min(1.0, (self.used_tokens / self.total_tokens))

    @property
    def allocation_rate(self) -> float:
        """Calculate allocation rate."""
        if self.total_tokens == 0:
            return 0.0
        return min(1.0, (self.allocated_tokens / self.total_tokens))

    @property
    def is_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return self.available_tokens <= 0

    @property
    def is_near_limit(self) -> bool:
        """Check if near budget limit."""
        return self.utilization_rate > 0.8


@dataclass
class TokenAllocation:
    """Individual token allocation."""
    allocation_id: str
    request_id: str
    model_name: str

    # Token counts
    requested_tokens: int
    allocated_tokens: int
    used_tokens: int = 0

    # Cost tracking
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0

    # Priority and metadata
    priority: int = 5  # 1-10, higher is more important
    strategy: AllocationStrategy = AllocationStrategy.DYNAMIC

    # Status
    is_active: bool = True
    is_completed: bool = False
    exceeded_allocation: bool = False

    # Timestamps
    allocated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    @property
    def remaining_tokens(self) -> int:
        """Calculate remaining tokens."""
        return max(0, self.allocated_tokens - self.used_tokens)

    @property
    def usage_rate(self) -> float:
        """Calculate usage rate."""
        if self.allocated_tokens == 0:
            return 0.0
        return min(1.0, self.used_tokens / self.allocated_tokens)


class TokenBudgetManager:
    """
    Manages token budgets and allocations across the system.
    """

    def __init__(self):
        """Initialize the token budget manager."""
        # Budget configurations by period
        self.budgets: dict[BudgetPeriod, TokenBudget] = {}

        # Active allocations
        self.allocations: dict[str, TokenAllocation] = {}

        # Historical data for learning
        self.allocation_history: list[TokenAllocation] = []

        # Model-specific limits
        self.model_limits = {
            "gemini-2.5-flash": {"max_tokens": 32768, "daily_limit": 1000000},
            "gpt-4o-mini": {"max_tokens": 128000, "daily_limit": 500000},
            "claude-3-5-haiku-20241022": {"max_tokens": 200000, "daily_limit": 300000},
            "gpt-4o": {"max_tokens": 128000, "daily_limit": 200000},
            "claude-3-5-sonnet-20241022": {"max_tokens": 200000, "daily_limit": 100000},
            "gemini-2.5-pro": {"max_tokens": 2097152, "daily_limit": 50000},
            "o1-preview": {"max_tokens": 128000, "daily_limit": 50000},
        }

        # Default allocation strategies by task type
        self.task_strategies = {
            "chat": AllocationStrategy.PROPORTIONAL,
            "analysis": AllocationStrategy.DYNAMIC,
            "generation": AllocationStrategy.FIXED,
            "review": AllocationStrategy.PRIORITY,
            "debug": AllocationStrategy.ADAPTIVE
        }

        logger.info("Token budget manager initialized")

    def set_budget(self,
                   period: BudgetPeriod,
                   total_tokens: int,
                   total_budget_usd: Optional[float] = None) -> TokenBudget:
        """
        Set or update a budget for a period.

        Args:
            period: Budget period
            total_tokens: Total token budget
            total_budget_usd: Optional USD budget

        Returns:
            TokenBudget object
        """
        # Calculate period end
        now = datetime.now(timezone.utc)
        if period == BudgetPeriod.HOURLY:
            period_end = now + timedelta(hours=1)
        elif period == BudgetPeriod.DAILY:
            period_end = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
        elif period == BudgetPeriod.WEEKLY:
            period_end = now + timedelta(days=7 - now.weekday())
        else:  # MONTHLY
            next_month = now.month + 1 if now.month < 12 else 1
            next_year = now.year if now.month < 12 else now.year + 1
            period_end = now.replace(year=next_year, month=next_month, day=1) - timedelta(seconds=1)

        budget = TokenBudget(
            period=period,
            total_tokens=total_tokens,
            total_budget_usd=total_budget_usd,
            period_start=now,
            period_end=period_end
        )

        self.budgets[period] = budget

        logger.info(f"Set {period.value} budget: {total_tokens} tokens, ${total_budget_usd or 0:.2f}")

        return budget

    def allocate_tokens(self,
                       request_id: str,
                       model_name: str,
                       requested_tokens: int,
                       task_type: str = "general",
                       priority: int = 5,
                       estimated_cost_usd: float = 0.0) -> Optional[TokenAllocation]:
        """
        Allocate tokens for a request.

        Args:
            request_id: Request identifier
            model_name: Model to use
            requested_tokens: Number of tokens requested
            task_type: Type of task
            priority: Priority level (1-10)
            estimated_cost_usd: Estimated cost

        Returns:
            TokenAllocation or None if cannot allocate
        """
        # Check model limits
        model_limit = self.model_limits.get(model_name, {})
        max_tokens = model_limit.get("max_tokens", 100000)

        if requested_tokens > max_tokens:
            logger.warning(f"Request exceeds model limit: {requested_tokens} > {max_tokens}")
            requested_tokens = max_tokens

        # Get allocation strategy
        strategy = self.task_strategies.get(task_type, AllocationStrategy.DYNAMIC)

        # Calculate actual allocation based on strategy
        allocated_tokens = self._calculate_allocation(
            requested_tokens,
            model_name,
            strategy,
            priority
        )

        if allocated_tokens <= 0:
            logger.warning(f"Cannot allocate tokens for request {request_id}")
            return None

        # Check budgets
        can_allocate = True
        for period, budget in self.budgets.items():
            if budget.available_tokens < allocated_tokens:
                logger.warning(f"Insufficient {period.value} budget: {budget.available_tokens} < {allocated_tokens}")
                can_allocate = False
                break

            if budget.total_budget_usd and estimated_cost_usd > 0:
                if budget.allocated_budget_usd + estimated_cost_usd > budget.total_budget_usd:
                    logger.warning(f"Would exceed {period.value} cost budget")
                    can_allocate = False
                    break

        if not can_allocate:
            return None

        # Create allocation
        allocation = TokenAllocation(
            allocation_id=f"{request_id}_{datetime.now(timezone.utc).timestamp()}",
            request_id=request_id,
            model_name=model_name,
            requested_tokens=requested_tokens,
            allocated_tokens=allocated_tokens,
            estimated_cost_usd=estimated_cost_usd,
            priority=priority,
            strategy=strategy
        )

        # Update budgets
        for budget in self.budgets.values():
            budget.allocated_tokens += allocated_tokens
            budget.allocated_budget_usd += estimated_cost_usd
            budget.last_updated = datetime.now(timezone.utc)

        # Store allocation
        self.allocations[allocation.allocation_id] = allocation

        logger.debug(f"Allocated {allocated_tokens} tokens for {request_id} using {model_name}")

        return allocation

    def update_usage(self,
                    allocation_id: str,
                    used_tokens: int,
                    actual_cost_usd: float = 0.0) -> bool:
        """
        Update token usage for an allocation.

        Args:
            allocation_id: Allocation identifier
            used_tokens: Actual tokens used
            actual_cost_usd: Actual cost incurred

        Returns:
            True if updated successfully
        """
        if allocation_id not in self.allocations:
            logger.warning(f"Allocation {allocation_id} not found")
            return False

        allocation = self.allocations[allocation_id]
        tokens_delta = used_tokens - allocation.used_tokens
        cost_delta = actual_cost_usd - allocation.actual_cost_usd

        # Update allocation
        allocation.used_tokens = used_tokens
        allocation.actual_cost_usd = actual_cost_usd

        if used_tokens > allocation.allocated_tokens:
            allocation.exceeded_allocation = True
            logger.warning(f"Allocation {allocation_id} exceeded: {used_tokens} > {allocation.allocated_tokens}")

        # Update budgets
        for budget in self.budgets.values():
            budget.used_tokens += tokens_delta
            budget.used_budget_usd += cost_delta
            budget.last_updated = datetime.now(timezone.utc)

        return True

    def complete_allocation(self,
                          allocation_id: str,
                          final_tokens: Optional[int] = None,
                          final_cost: Optional[float] = None) -> bool:
        """
        Mark an allocation as completed.

        Args:
            allocation_id: Allocation identifier
            final_tokens: Final token count
            final_cost: Final cost

        Returns:
            True if completed successfully
        """
        if allocation_id not in self.allocations:
            logger.warning(f"Allocation {allocation_id} not found")
            return False

        allocation = self.allocations[allocation_id]

        # Update final usage if provided
        if final_tokens is not None:
            self.update_usage(allocation_id, final_tokens, final_cost or allocation.actual_cost_usd)

        # Mark as completed
        allocation.is_active = False
        allocation.is_completed = True
        allocation.completed_at = datetime.now(timezone.utc)

        # Move to history
        self.allocation_history.append(allocation)
        if len(self.allocation_history) > 10000:
            # Keep only recent history
            self.allocation_history = self.allocation_history[-10000:]

        # Free up allocated tokens
        freed_tokens = allocation.allocated_tokens - allocation.used_tokens
        if freed_tokens > 0:
            for budget in self.budgets.values():
                budget.allocated_tokens = max(0, budget.allocated_tokens - freed_tokens)

        logger.debug(f"Completed allocation {allocation_id}: {allocation.used_tokens} tokens used")

        return True

    def _calculate_allocation(self,
                            requested_tokens: int,
                            model_name: str,
                            strategy: AllocationStrategy,
                            priority: int) -> int:
        """
        Calculate actual token allocation based on strategy.

        Args:
            requested_tokens: Requested tokens
            model_name: Model name
            strategy: Allocation strategy
            priority: Priority level

        Returns:
            Allocated token count
        """
        if strategy == AllocationStrategy.FIXED:
            # Fixed allocation (min of requested and limit)
            model_limit = self.model_limits.get(model_name, {})
            return min(requested_tokens, model_limit.get("max_tokens", requested_tokens))

        elif strategy == AllocationStrategy.PROPORTIONAL:
            # Proportional to available budget
            min_budget = min(b.available_tokens for b in self.budgets.values()) if self.budgets else requested_tokens
            return min(requested_tokens, int(min_budget * 0.1))  # Max 10% of available

        elif strategy == AllocationStrategy.PRIORITY:
            # Priority-based (higher priority gets more)
            base_allocation = requested_tokens * (priority / 10)
            return int(min(requested_tokens, base_allocation))

        elif strategy == AllocationStrategy.ADAPTIVE:
            # Learn from history
            if self.allocation_history:
                # Find similar allocations
                similar = [
                    a for a in self.allocation_history[-100:]
                    if a.model_name == model_name and a.is_completed
                ]

                if similar:
                    # Use average of similar allocations
                    avg_used = sum(a.used_tokens for a in similar) / len(similar)
                    # Add 20% buffer
                    return int(min(requested_tokens, avg_used * 1.2))

            # Fall back to proportional
            return self._calculate_allocation(requested_tokens, model_name, AllocationStrategy.PROPORTIONAL, priority)

        else:  # DYNAMIC
            # Dynamic based on current utilization
            if self.budgets:
                avg_utilization = sum(b.utilization_rate for b in self.budgets.values()) / len(self.budgets)

                if avg_utilization < 0.5:
                    # Low utilization, be generous
                    return requested_tokens
                elif avg_utilization < 0.8:
                    # Moderate utilization
                    return int(requested_tokens * 0.8)
                else:
                    # High utilization, be conservative
                    return int(requested_tokens * 0.5)

            return requested_tokens

    def get_budget_status(self, period: Optional[BudgetPeriod] = None) -> dict[str, Any]:
        """
        Get budget status.

        Args:
            period: Specific period or None for all

        Returns:
            Budget status information
        """
        if period:
            budget = self.budgets.get(period)
            if not budget:
                return {"error": f"No budget set for {period.value}"}

            return {
                "period": period.value,
                "total_tokens": budget.total_tokens,
                "used_tokens": budget.used_tokens,
                "allocated_tokens": budget.allocated_tokens,
                "available_tokens": budget.available_tokens,
                "utilization_rate": budget.utilization_rate,
                "allocation_rate": budget.allocation_rate,
                "total_budget_usd": budget.total_budget_usd,
                "used_budget_usd": budget.used_budget_usd,
                "is_exhausted": budget.is_exhausted,
                "is_near_limit": budget.is_near_limit,
                "period_end": budget.period_end.isoformat() if budget.period_end else None
            }

        # Return all budgets
        return {
            period.value: self.get_budget_status(period)
            for period in self.budgets.keys()
        }

    def get_allocation_statistics(self) -> dict[str, Any]:
        """Get allocation statistics."""
        active_allocations = [a for a in self.allocations.values() if a.is_active]
        completed_allocations = [a for a in self.allocation_history if a.is_completed]

        if not self.allocation_history and not active_allocations:
            return {"status": "no_data"}

        exceeded_allocations = [
            a for a in self.allocation_history
            if a.exceeded_allocation
        ]

        return {
            "active_allocations": len(active_allocations),
            "completed_allocations": len(completed_allocations),
            "total_allocations": len(self.allocation_history) + len(active_allocations),

            "exceeded_rate": len(exceeded_allocations) / len(completed_allocations) if completed_allocations else 0,

            "token_stats": {
                "total_allocated": sum(a.allocated_tokens for a in self.allocation_history),
                "total_used": sum(a.used_tokens for a in self.allocation_history),
                "average_efficiency": sum(a.usage_rate for a in completed_allocations) / len(completed_allocations) if completed_allocations else 0
            },

            "cost_stats": {
                "total_estimated": sum(a.estimated_cost_usd for a in self.allocation_history),
                "total_actual": sum(a.actual_cost_usd for a in self.allocation_history),
                "cost_accuracy": 1 - abs(
                    sum(a.actual_cost_usd - a.estimated_cost_usd for a in completed_allocations) /
                    sum(a.estimated_cost_usd for a in completed_allocations)
                ) if completed_allocations and sum(a.estimated_cost_usd for a in completed_allocations) > 0 else 1.0
            },

            "strategy_distribution": {
                strategy.value: len([a for a in self.allocation_history if a.strategy == strategy])
                for strategy in AllocationStrategy
            }
        }

    def predict_budget_exhaustion(self, period: BudgetPeriod) -> Optional[datetime]:
        """
        Predict when a budget will be exhausted.

        Args:
            period: Budget period to check

        Returns:
            Predicted exhaustion time or None
        """
        budget = self.budgets.get(period)
        if not budget or budget.is_exhausted:
            return None

        # Calculate current usage rate
        time_elapsed = (datetime.now(timezone.utc) - budget.period_start).total_seconds()
        if time_elapsed <= 0:
            return None

        tokens_per_second = budget.used_tokens / time_elapsed

        if tokens_per_second <= 0:
            return None

        # Predict exhaustion
        remaining_tokens = budget.available_tokens
        seconds_to_exhaustion = remaining_tokens / tokens_per_second

        predicted_exhaustion = datetime.now(timezone.utc) + timedelta(seconds=seconds_to_exhaustion)

        # Don't predict beyond period end
        if budget.period_end and predicted_exhaustion > budget.period_end:
            return None

        return predicted_exhaustion


# Global instance
_budget_manager = None


def get_budget_manager() -> TokenBudgetManager:
    """Get or create the global token budget manager."""
    global _budget_manager
    if _budget_manager is None:
        _budget_manager = TokenBudgetManager()
    return _budget_manager
