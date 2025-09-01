"""
Common Utilities and Patterns for Zen MCP Server

This module extracts common patterns and utilities used across multiple
tools and modules to reduce code duplication and improve maintainability.

Features:
- Common error handling patterns
- Shared validation utilities
- Response formatting helpers
- Logging utilities
- Common decorators and mixins
"""

import functools
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

from mcp.types import TextContent

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorHandler:
    """Common error handling patterns for tools."""

    @staticmethod
    def format_error_response(error: Exception, context: str = "") -> list[TextContent]:
        """Format an error into a standardized response."""
        error_msg = f"Error in {context}: {str(error)}" if context else f"Error: {str(error)}"
        logger.error(error_msg, exc_info=True)
        return [TextContent(type="text", text=f"❌ {error_msg}")]

    @staticmethod
    def format_validation_error(field: str, value: Any, expected: str) -> str:
        """Format a validation error message."""
        return f"Invalid {field}: '{value}'. Expected {expected}."

    @staticmethod
    def format_file_error(file_path: str, error: Exception) -> str:
        """Format a file access error message."""
        return f"Cannot access file '{file_path}': {str(error)}"


class ResponseFormatter:
    """Common response formatting utilities."""

    @staticmethod
    def format_success(message: str, data: Optional[dict] = None) -> list[TextContent]:
        """Format a success response."""
        content = f"✅ {message}"
        if data:
            content += f"\n\nDetails:\n{ResponseFormatter._format_data(data)}"
        return [TextContent(type="text", text=content)]

    @staticmethod
    def format_info(message: str, details: Optional[str] = None) -> list[TextContent]:
        """Format an informational response."""
        content = f"ℹ️ {message}"
        if details:
            content += f"\n\n{details}"
        return [TextContent(type="text", text=content)]

    @staticmethod
    def format_warning(message: str, details: Optional[str] = None) -> list[TextContent]:
        """Format a warning response."""
        content = f"⚠️ {message}"
        if details:
            content += f"\n\n{details}"
        return [TextContent(type="text", text=content)]

    @staticmethod
    def format_list(items: list[str], title: str = "Items") -> str:
        """Format a list of items with bullets."""
        if not items:
            return f"No {title.lower()} found."

        formatted_items = [f"• {item}" for item in items]
        return f"{title}:\n" + "\n".join(formatted_items)

    @staticmethod
    def format_table(data: list[dict[str, Any]], headers: Optional[list[str]] = None) -> str:
        """Format data as a simple table."""
        if not data:
            return "No data available."

        if headers is None:
            headers = list(data[0].keys()) if data else []

        # Calculate column widths
        widths = {}
        for header in headers:
            widths[header] = max(len(header), max(len(str(row.get(header, ""))) for row in data))

        # Format header
        header_row = " | ".join(header.ljust(widths[header]) for header in headers)
        separator = "-" * len(header_row)

        # Format data rows
        data_rows = []
        for row in data:
            formatted_row = " | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers)
            data_rows.append(formatted_row)

        return f"{header_row}\n{separator}\n" + "\n".join(data_rows)

    @staticmethod
    def _format_data(data: dict) -> str:
        """Format dictionary data for display."""
        lines = []
        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                lines.append(f"{key}: {', '.join(str(v) for v in value)}")
            elif isinstance(value, dict):
                lines.append(f"{key}: {len(value)} items")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)


class ValidationUtils:
    """Common validation utilities."""

    @staticmethod
    def validate_file_path(path: str) -> bool:
        """Validate that a file path is safe and accessible."""
        try:
            resolved_path = Path(path).resolve()
            return resolved_path.exists() and resolved_path.is_file()
        except (OSError, ValueError):
            return False

    @staticmethod
    def validate_directory_path(path: str) -> bool:
        """Validate that a directory path is safe and accessible."""
        try:
            resolved_path = Path(path).resolve()
            return resolved_path.exists() and resolved_path.is_dir()
        except (OSError, ValueError):
            return False

    @staticmethod
    def validate_model_name(model_name: str, allowed_models: list[str]) -> bool:
        """Validate that a model name is in the allowed list."""
        if not allowed_models:  # No restrictions
            return True
        return model_name in allowed_models

    @staticmethod
    def validate_temperature(temperature: float) -> bool:
        """Validate temperature value is in acceptable range."""
        return 0.0 <= temperature <= 2.0

    @staticmethod
    def validate_timeout(timeout: Union[int, float]) -> bool:
        """Validate timeout value is positive."""
        return timeout > 0


class LoggingUtils:
    """Common logging utilities."""

    @staticmethod
    def log_tool_execution(tool_name: str, arguments: dict[str, Any]):
        """Log tool execution start."""
        arg_summary = {k: f"<{type(v).__name__}>" if len(str(v)) > 50 else v for k, v in arguments.items()}
        logger.info(f"🔧 {tool_name} tool called with arguments: {arg_summary}")

    @staticmethod
    def log_tool_completion(tool_name: str, success: bool, duration: float):
        """Log tool execution completion."""
        status = "✅ completed" if success else "❌ failed"
        logger.info(f"🔧 {tool_name} tool {status} in {duration:.2f}s")

    @staticmethod
    def log_file_access(file_path: str, success: bool, size: Optional[int] = None):
        """Log file access attempts."""
        if success:
            size_info = f" ({size} bytes)" if size else ""
            logger.debug(f"📁 File read successfully: {file_path}{size_info}")
        else:
            logger.warning(f"📁 File access failed: {file_path}")

    @staticmethod
    def log_model_request(provider: str, model: str, tokens: Optional[int] = None):
        """Log AI model requests."""
        token_info = f" ({tokens} tokens)" if tokens else ""
        logger.debug(f"🤖 Model request: {provider}/{model}{token_info}")


def timing_decorator(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to time function execution."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            LoggingUtils.log_tool_completion(func.__name__, True, duration)
            return result
        except Exception:
            duration = time.time() - start_time
            LoggingUtils.log_tool_completion(func.__name__, False, duration)
            raise

    return wrapper


def async_timing_decorator(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to time async function execution."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            LoggingUtils.log_tool_completion(func.__name__, True, duration)
            return result
        except Exception:
            duration = time.time() - start_time
            LoggingUtils.log_tool_completion(func.__name__, False, duration)
            raise

    return wrapper


def retry_decorator(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function execution on failure."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")

            if last_exception:
                raise last_exception

        return wrapper

    return decorator


class PerformanceMonitor:
    """Simple performance monitoring utilities."""

    def __init__(self):
        self._metrics: dict[str, list[float]] = {}

    def record_execution_time(self, operation: str, duration: float):
        """Record execution time for an operation."""
        if operation not in self._metrics:
            self._metrics[operation] = []
        self._metrics[operation].append(duration)

    def get_average_time(self, operation: str) -> Optional[float]:
        """Get average execution time for an operation."""
        if operation not in self._metrics or not self._metrics[operation]:
            return None
        return sum(self._metrics[operation]) / len(self._metrics[operation])

    def get_stats(self, operation: str) -> Optional[dict[str, float]]:
        """Get detailed statistics for an operation."""
        if operation not in self._metrics or not self._metrics[operation]:
            return None

        times = self._metrics[operation]
        return {
            "count": len(times),
            "average": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "total": sum(times),
        }

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all operations."""
        return {op: self.get_stats(op) for op in self._metrics.keys()}


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


class SafeDict(dict):
    """Dictionary that returns None for missing keys instead of raising KeyError."""

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return None

    def get_required(self, key: str, error_msg: Optional[str] = None):
        """Get a required key, raising ValueError if missing."""
        value = self.get(key)
        if value is None:
            msg = error_msg or f"Required key '{key}' is missing"
            raise ValueError(msg)
        return value
