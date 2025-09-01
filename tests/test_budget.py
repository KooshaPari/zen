import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.token_budget_manager import BudgetPeriod, get_budget_manager

# Create budget manager
manager = get_budget_manager()

# Set daily budget
budget = manager.set_budget(
    period=BudgetPeriod.DAILY,
    total_tokens=1000000,
    total_budget_usd=10.0
)

print(f"Created daily budget: {budget.total_tokens} tokens, ${budget.total_budget_usd}")

# Allocate some tokens
allocation = manager.allocate_tokens(
    request_id="test-001",
    model_name="gpt-4o-mini",
    requested_tokens=1000,
    task_type="chat",
    priority=5,
    estimated_cost_usd=0.01
)

if allocation:
    print(f"Allocated {allocation.allocated_tokens} tokens for request {allocation.request_id}")

    # Update usage
    manager.update_usage(
        allocation.allocation_id,
        used_tokens=500,
        actual_cost_usd=0.005
    )
    print("Updated usage: 500 tokens used")

# Get status
status = manager.get_budget_status(BudgetPeriod.DAILY)
print("\nBudget Status:")
print(f"- Total: {status['total_tokens']} tokens")
print(f"- Used: {status['used_tokens']} tokens")
print(f"- Available: {status['available_tokens']} tokens")
print(f"- Utilization: {status['utilization_rate']*100:.1f}%")
