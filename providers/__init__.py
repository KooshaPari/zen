"""Model provider abstractions for supporting multiple AI providers.

Imports are guarded so optional providers (e.g., Gemini) don't break environments
where their dependencies aren't installed. Tests and runtime can still use other
providers like OpenRouter without requiring google-genai.
"""

from .base import ModelCapabilities, ModelProvider, ModelResponse
from .openai_compatible import OpenAICompatibleProvider
from .openai_provider import OpenAIModelProvider
from .openrouter import OpenRouterProvider
from .registry import ModelProviderRegistry

# Optional Gemini provider import
try:  # pragma: no cover - environment may not have google-genai installed
    from .gemini import GeminiModelProvider  # type: ignore
    _HAS_GEMINI = True
except Exception:  # broad to avoid import-time issues in constrained envs
    GeminiModelProvider = None  # type: ignore
    _HAS_GEMINI = False

__all__ = [
    "ModelProvider",
    "ModelResponse",
    "ModelCapabilities",
    "ModelProviderRegistry",
    "OpenAIModelProvider",
    "OpenAICompatibleProvider",
    "OpenRouterProvider",
]

# Only expose GeminiModelProvider if available
if _HAS_GEMINI:
    __all__.append("GeminiModelProvider")
