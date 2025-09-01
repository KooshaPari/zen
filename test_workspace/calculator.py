"""Simple calculator module with basic arithmetic functions."""

from __future__ import annotations


def add(a: float, b: float) -> float:
    """Return the sum of a and b."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Return the difference a - b."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Return the product of a and b."""
    return a * b


def divide(a: float, b: float) -> float:
    """Return the quotient a / b.

    Raises:
        ZeroDivisionError: If b is 0.
    """
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b

