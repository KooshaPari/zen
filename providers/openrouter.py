"""OpenRouter provider implementation."""

import logging
import os
from typing import Optional

from .base import (
    ModelCapabilities,
    ModelResponse,
    ProviderType,
    RangeTemperatureConstraint,
)
from .openai_compatible import OpenAICompatibleProvider
from .openrouter_registry import OpenRouterModelRegistry


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter unified API provider.

    OpenRouter provides access to multiple AI models through a single API endpoint.
    See https://openrouter.ai for available models and pricing.
    """

    FRIENDLY_NAME = "OpenRouter"

    # Custom headers required by OpenRouter
    DEFAULT_HEADERS = {
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/BeehiveInnovations/zen-mcp-server"),
        "X-Title": os.getenv("OPENROUTER_TITLE", "Zen MCP Server"),
    }

    # Cost threshold (per 1M input+output tokens) — 0 = only free models
    MAX_COST_TOTAL_PER_1M = float(os.getenv("OPENROUTER_MAX_COST_PER_1M_TOTAL", "-1"))
    # Live pricing cache (shared across instances)
    _pricing_cache: dict[str, tuple[float, float]] = {}
    _pricing_cache_time: float = 0.0
    PRICING_CACHE_TTL_SEC = int(os.getenv("OPENROUTER_PRICING_CACHE_TTL", "600"))  # 10 minutes default
    # Model registry for managing configurations and aliases
    _registry: Optional[OpenRouterModelRegistry] = None

    def __init__(self, api_key: str, **kwargs):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key
            **kwargs: Additional configuration
        """
        base_url = "https://openrouter.ai/api/v1"
        super().__init__(api_key, base_url=base_url, **kwargs)

        # Initialize model registry
        if OpenRouterProvider._registry is None:
            OpenRouterProvider._registry = OpenRouterModelRegistry()
            # Log loaded models and aliases only on first load
            models = self._registry.list_models()
            aliases = self._registry.list_aliases()
            logging.info(f"OpenRouter loaded {len(models)} models with {len(aliases)} aliases")

    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve model aliases to OpenRouter model names.

        Args:
            model_name: Input model name or alias

        Returns:
            Resolved OpenRouter model name
        """
        # Try to resolve through registry
        config = self._registry.resolve(model_name)

        if config:
            if config.model_name != model_name:
                logging.info(f"Resolved model alias '{model_name}' to '{config.model_name}'")
            return config.model_name
        else:
            # If not found in registry, return as-is
            # This allows using models not in our config file
            logging.debug(f"Model '{model_name}' not found in registry, using as-is")
            return model_name

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for a model.

        Args:
            model_name: Name of the model (or alias)

        Returns:
            ModelCapabilities from registry or generic defaults
        """
        # Try to get from registry first
        capabilities = self._registry.get_capabilities(model_name)

        if capabilities:
            return capabilities
        else:
            # Resolve any potential aliases and create generic capabilities
            resolved_name = self._resolve_model_name(model_name)

            logging.debug(
                f"Using generic capabilities for '{resolved_name}' via OpenRouter. "
                "Consider adding to custom_models.json for specific capabilities."
            )

            # Create generic capabilities with conservative defaults
            capabilities = ModelCapabilities(
                provider=ProviderType.OPENROUTER,
                model_name=resolved_name,
                friendly_name=self.FRIENDLY_NAME,
                context_window=32_768,  # Conservative default context window
                max_output_tokens=32_768,
                supports_extended_thinking=False,
                supports_system_prompts=True,
                supports_streaming=True,
                supports_function_calling=False,
                temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 1.0),
            )

            # Mark as generic for validation purposes
            capabilities._is_generic = True

            return capabilities

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.OPENROUTER

    def _within_cost_threshold(self, model_name: str) -> bool:
        """Check if model's cost is within configured threshold.

        Returns True if no threshold configured or pricing unknown (fail-open unless threshold is 0),
        otherwise compares (input_cost + output_cost) per 1M to the threshold.
        """
        try:
            max_total = self.MAX_COST_TOTAL_PER_1M
            if max_total < 0:
                return True  # no threshold configured
            config = self._registry.resolve(model_name) if self._registry else None
            if not config:
                # Conservative: block unknown pricing when guard enabled (max_total >= 0)
                return False
            ic = getattr(config, "input_cost_per_1k", None)
            oc = getattr(config, "output_cost_per_1k", None)
            if ic is None and oc is None:
                # Try live pricing fallback
                self._refresh_live_pricing()
                prices = self._pricing_cache.get(model_name) or self._pricing_cache.get(config.model_name)
                if prices:
                    ic_live, oc_live = prices
                    total_live = (ic_live or 0.0) + (oc_live or 0.0)
                    return total_live * 1000000.0 <= max_total + 1e-12  # convert live per-token to per-1M
                # Conservative: block unknown pricing when guard enabled
                return False
            total = (ic or 0.0) + (oc or 0.0)
            return total * 1000.0 <= max_total + 1e-12  # convert per-1K config to per-1M comparison
        except Exception:
            return True

    def _refresh_live_pricing(self) -> None:
        """Fetch live pricing from OpenRouter models endpoint and cache per-1K costs.

        Stores prompt/completion prices as floats per token (as returned). Use ×1000 for per-1K.
        """
        import time

        import httpx

        now = time.time()
        if now - self._pricing_cache_time < self.PRICING_CACHE_TTL_SEC and self._pricing_cache:
            return

        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as c:
                r = c.get("https://openrouter.ai/api/v1/models")
                r.raise_for_status()
                jd = r.json()
                cache: dict[str, tuple[float, float]] = {}
                for m in jd.get("data", []):
                    mid = m.get("id") or m.get("canonical_slug")
                    pricing = m.get("pricing") or {}
                    p = float(pricing.get("prompt", 0) or 0)
                    q = float(pricing.get("completion", 0) or 0)
                    if mid:
                        cache[mid] = (p, q)
                        # Also alias by canonical_slug if different
                        slug = m.get("canonical_slug")
                        if slug:
                            cache[slug] = (p, q)
                # Update cache if non-empty
                if cache:
                    self.__class__._pricing_cache = cache
                    self.__class__._pricing_cache_time = now
        except Exception:
            # Silently ignore network errors (fallback to static config)
            pass

    def validate_model_name(self, model_name: str) -> bool:
        """Validate if the model name is allowed.

        As the catch-all provider, OpenRouter accepts any model name that wasn't
        handled by higher-priority providers. OpenRouter will validate based on
        the API key's permissions and local restrictions.

        Args:
            model_name: Model name to validate

        Returns:
            True if model is allowed, False if restricted
        """
        # Check model restrictions if configured
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service()
        if restriction_service:
            # Check if model name itself is allowed
            if restriction_service.is_allowed(self.get_provider_type(), model_name):
                # Also enforce cost gate
                return self._within_cost_threshold(model_name)

            # Also check aliases - model_name might be an alias
            model_config = self._registry.resolve(model_name)
            if model_config and model_config.aliases:
                for alias in model_config.aliases:
                    if restriction_service.is_allowed(self.get_provider_type(), alias):
                        return self._within_cost_threshold(alias)

            # If restrictions are configured and model/alias not in allowed list, reject
            return False
        # No restrictions configured - enforce cost gate if set
        return self._within_cost_threshold(model_name)
        # Note: generate_content performs a cost check too

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using the OpenRouter API.

        Args:
            prompt: User prompt to send to the model
            model_name: Name of the model (or alias) to use
            system_prompt: Optional system prompt for model behavior
            temperature: Sampling temperature
            max_output_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            ModelResponse with generated content and metadata
        """
        # Resolve model alias to actual OpenRouter model name
        resolved_model = self._resolve_model_name(model_name)

        # Always disable streaming for OpenRouter
        # MCP doesn't use streaming, and this avoids issues with O3 model access
        if "stream" not in kwargs:
            kwargs["stream"] = False
        # Enforce cost gate at call time
        if not self._within_cost_threshold(resolved_model):
            raise ValueError(
                f"OpenRouter model '{model_name}' exceeds cost threshold OPENROUTER_MAX_COST_PER_1M_TOTAL={self.MAX_COST_TOTAL_PER_1M}."
            )

        # Call parent method with resolved model name
        return super().generate_content(
            prompt=prompt,
            model_name=resolved_model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if the model supports extended thinking mode.

        Currently, no models via OpenRouter support extended thinking.
        This may change as new models become available.

        Args:
            model_name: Model to check

        Returns:
            False (no OpenRouter models currently support thinking mode)
        """
        return False

    def list_models(self, respect_restrictions: bool = True) -> list[str]:
        """Return a list of model names supported by this provider.

        Args:
            respect_restrictions: Whether to apply provider-specific restriction logic.

        Returns:
            List of model names available from this provider
        """
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service() if respect_restrictions else None
        models = []

        if self._registry:
            for model_name in self._registry.list_models():
                # =====================================================================================
                # CRITICAL ALIAS-AWARE RESTRICTION CHECKING (Fixed Issue #98)
                # =====================================================================================
                # Previously, restrictions only checked full model names (e.g., "google/gemini-2.5-pro")
                # but users specify aliases in OPENROUTER_ALLOWED_MODELS (e.g., "pro").
                # This caused "no models available" error even with valid restrictions.
                #
                # Fix: Check both model name AND all aliases against restrictions
                # TEST COVERAGE: tests/test_provider_routing_bugs.py::TestOpenRouterAliasRestrictions
                # =====================================================================================
                if restriction_service:
                    # Get model config to check aliases as well
                    model_config = self._registry.resolve(model_name)
                    allowed = False

                    # Check if model name itself is allowed
                    if restriction_service.is_allowed(self.get_provider_type(), model_name):
                        allowed = True

                    # Also check aliases
                    if not allowed and model_config and model_config.aliases:
                        for alias in model_config.aliases:
                            if restriction_service.is_allowed(self.get_provider_type(), alias):
                                allowed = True
                                break

                    if not allowed:
                        continue

                # Enforce cost gate regardless of restriction config
                if not self._within_cost_threshold(model_name):
                    continue

                models.append(model_name)

        return models

    def list_all_known_models(self) -> list[str]:
        """Return all model names known by this provider, including alias targets.

        Returns:
            List of all model names and alias targets known by this provider
        """
        all_models = set()

        if self._registry:
            # Get all models and aliases from the registry
            all_models.update(model.lower() for model in self._registry.list_models())
            all_models.update(alias.lower() for alias in self._registry.list_aliases())

            # For each alias, also add its target
            for alias in self._registry.list_aliases():
                config = self._registry.resolve(alias)
                if config:
                    all_models.add(config.model_name.lower())

        return list(all_models)

    def get_model_configurations(self) -> dict[str, ModelCapabilities]:
        """Get model configurations from the registry.

        For OpenRouter, we convert registry configurations to ModelCapabilities objects.

        Returns:
            Dictionary mapping model names to their ModelCapabilities objects
        """
        configs = {}

        if self._registry:
            # Get all models from registry
            for model_name in self._registry.list_models():
                # Only include models that this provider validates
                if self.validate_model_name(model_name):
                    config = self._registry.resolve(model_name)
                    if config and not config.is_custom:  # Only OpenRouter models, not custom ones
                        # Use ModelCapabilities directly from registry
                        configs[model_name] = config

        return configs

    def get_all_model_aliases(self) -> dict[str, list[str]]:
        """Get all model aliases from the registry.

        Returns:
            Dictionary mapping model names to their list of aliases
        """
        # Since aliases are now included in the configurations,
        # we can use the base class implementation
        return super().get_all_model_aliases()
