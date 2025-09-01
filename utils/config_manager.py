"""
Centralized Configuration Manager for Zen MCP Server

This module provides a unified interface for managing all configuration
settings across the application. It consolidates environment variable
handling, validation, and provides type-safe access to configuration values.

Features:
- Centralized configuration management
- Environment variable validation
- Type-safe configuration access
- Default value handling
- Configuration validation and error reporting
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for AI model settings."""

    default_model: str
    is_auto_mode: bool
    temperature_analytical: float
    temperature_balanced: float
    temperature_creative: float
    thinking_mode_default: str
    consensus_timeout: float
    consensus_max_instances: int


@dataclass
class ServerConfig:
    """Configuration for MCP server settings."""

    version: str
    author: str
    updated: str
    prompt_size_limit: int
    locale: str
    threading_enabled: bool


@dataclass
class SecurityConfig:
    """Configuration for security settings."""

    excluded_dirs: list[str]
    max_file_size: int
    allowed_extensions: list[str]
    enable_path_validation: bool


@dataclass
class ProviderConfig:
    """Configuration for AI provider settings."""

    openai_api_key: Optional[str]
    openai_allowed_models: list[str]
    google_api_key: Optional[str]
    google_allowed_models: list[str]
    xai_api_key: Optional[str]
    xai_allowed_models: list[str]
    openrouter_api_key: Optional[str]
    openrouter_allowed_models: list[str]
    custom_api_url: Optional[str]
    disabled_tools: list[str]


class ConfigManager:
    """
    Centralized configuration manager for the Zen MCP Server.

    This class provides a single point of access for all configuration
    settings, with proper validation and type safety.
    """

    def __init__(self):
        self._model_config: Optional[ModelConfig] = None
        self._server_config: Optional[ServerConfig] = None
        self._security_config: Optional[SecurityConfig] = None
        self._provider_config: Optional[ProviderConfig] = None
        self._load_configuration()

    def _load_configuration(self):
        """Load all configuration from environment variables and defaults."""
        try:
            self._model_config = self._load_model_config()
            self._server_config = self._load_server_config()
            self._security_config = self._load_security_config()
            self._provider_config = self._load_provider_config()
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _load_model_config(self) -> ModelConfig:
        """Load model-related configuration."""
        default_model = os.getenv("DEFAULT_MODEL", "auto")
        return ModelConfig(
            default_model=default_model,
            is_auto_mode=default_model.lower() == "auto",
            temperature_analytical=float(os.getenv("TEMPERATURE_ANALYTICAL", "0.2")),
            temperature_balanced=float(os.getenv("TEMPERATURE_BALANCED", "0.5")),
            temperature_creative=float(os.getenv("TEMPERATURE_CREATIVE", "0.7")),
            thinking_mode_default=os.getenv("DEFAULT_THINKING_MODE_THINKDEEP", "high"),
            consensus_timeout=float(os.getenv("DEFAULT_CONSENSUS_TIMEOUT", "120.0")),
            consensus_max_instances=int(os.getenv("DEFAULT_CONSENSUS_MAX_INSTANCES_PER_COMBINATION", "2")),
        )

    def _load_server_config(self) -> ServerConfig:
        """Load server-related configuration."""
        return ServerConfig(
            version=os.getenv("ZEN_VERSION", "5.11.0"),
            author=os.getenv("ZEN_AUTHOR", "Fahad Gilani"),
            updated=os.getenv("ZEN_UPDATED", "2025-08-26"),
            prompt_size_limit=int(os.getenv("MCP_PROMPT_SIZE_LIMIT", "200000")),
            locale=os.getenv("LOCALE", ""),
            threading_enabled=os.getenv("THREADING_ENABLED", "true").lower() == "true",
        )

    def _load_security_config(self) -> SecurityConfig:
        """Load security-related configuration."""
        excluded_dirs_str = os.getenv("EXCLUDED_DIRS", ".git,.svn,.hg,node_modules,__pycache__,.pytest_cache")
        excluded_dirs = [d.strip() for d in excluded_dirs_str.split(",") if d.strip()]

        allowed_extensions_str = os.getenv("ALLOWED_EXTENSIONS", ".py,.js,.ts,.md,.txt,.json,.yaml,.yml")
        allowed_extensions = [e.strip() for e in allowed_extensions_str.split(",") if e.strip()]

        return SecurityConfig(
            excluded_dirs=excluded_dirs,
            max_file_size=int(os.getenv("MAX_FILE_SIZE", "10485760")),  # 10MB
            allowed_extensions=allowed_extensions,
            enable_path_validation=os.getenv("ENABLE_PATH_VALIDATION", "true").lower() == "true",
        )

    def _load_provider_config(self) -> ProviderConfig:
        """Load AI provider configuration."""

        def parse_model_list(env_var: str, default: str = "") -> list[str]:
            """Parse comma-separated model list from environment variable."""
            models_str = os.getenv(env_var, default)
            return [m.strip() for m in models_str.split(",") if m.strip()] if models_str else []

        disabled_tools_str = os.getenv("DISABLED_TOOLS", "")
        disabled_tools = [t.strip() for t in disabled_tools_str.split(",") if t.strip()]

        return ProviderConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_allowed_models=parse_model_list("OPENAI_ALLOWED_MODELS"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_allowed_models=parse_model_list("GOOGLE_ALLOWED_MODELS"),
            xai_api_key=os.getenv("XAI_API_KEY"),
            xai_allowed_models=parse_model_list("XAI_ALLOWED_MODELS"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_allowed_models=parse_model_list("OPENROUTER_ALLOWED_MODELS"),
            custom_api_url=os.getenv("CUSTOM_API_URL"),
            disabled_tools=disabled_tools,
        )

    @property
    def model(self) -> ModelConfig:
        """Get model configuration."""
        if self._model_config is None:
            raise RuntimeError("Configuration not loaded")
        return self._model_config

    @property
    def server(self) -> ServerConfig:
        """Get server configuration."""
        if self._server_config is None:
            raise RuntimeError("Configuration not loaded")
        return self._server_config

    @property
    def security(self) -> SecurityConfig:
        """Get security configuration."""
        if self._security_config is None:
            raise RuntimeError("Configuration not loaded")
        return self._security_config

    @property
    def providers(self) -> ProviderConfig:
        """Get provider configuration."""
        if self._provider_config is None:
            raise RuntimeError("Configuration not loaded")
        return self._provider_config

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider."""
        provider_lower = provider.lower()
        if provider_lower == "openai":
            return self.providers.openai_api_key
        elif provider_lower in ["google", "gemini"]:
            return self.providers.google_api_key
        elif provider_lower == "xai":
            return self.providers.xai_api_key
        elif provider_lower == "openrouter":
            return self.providers.openrouter_api_key
        else:
            return None

    def get_allowed_models(self, provider: str) -> list[str]:
        """Get allowed models for a specific provider."""
        provider_lower = provider.lower()
        if provider_lower == "openai":
            return self.providers.openai_allowed_models
        elif provider_lower in ["google", "gemini"]:
            return self.providers.google_allowed_models
        elif provider_lower == "xai":
            return self.providers.xai_allowed_models
        elif provider_lower == "openrouter":
            return self.providers.openrouter_allowed_models
        else:
            return []

    def is_tool_disabled(self, tool_name: str) -> bool:
        """Check if a tool is disabled."""
        return tool_name in self.providers.disabled_tools

    def validate_configuration(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check for required API keys if not in auto mode
        if not self.model.is_auto_mode:
            if not any(
                [
                    self.providers.openai_api_key,
                    self.providers.google_api_key,
                    self.providers.xai_api_key,
                    self.providers.openrouter_api_key,
                    self.providers.custom_api_url,
                ]
            ):
                issues.append("No API keys configured for any provider")

        # Validate temperature ranges
        for temp_name, temp_value in [
            ("analytical", self.model.temperature_analytical),
            ("balanced", self.model.temperature_balanced),
            ("creative", self.model.temperature_creative),
        ]:
            if not 0.0 <= temp_value <= 2.0:
                issues.append(f"Temperature {temp_name} ({temp_value}) must be between 0.0 and 2.0")

        # Validate file size limits
        if self.security.max_file_size <= 0:
            issues.append("Max file size must be positive")

        return issues

    def reload(self):
        """Reload configuration from environment variables."""
        self._load_configuration()


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def reload_config():
    """Reload the global configuration."""
    global _config_manager
    if _config_manager is not None:
        _config_manager.reload()
    else:
        _config_manager = ConfigManager()
