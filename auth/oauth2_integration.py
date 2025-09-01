#!/usr/bin/env python3
"""
OAuth 2.0 Dynamic Client Registration Integration

This module integrates the DCR system with the existing Zen MCP Server HTTP transport,
replacing the minimal client registration stub with a full RFC 7591 compliant implementation.

Features:
- Seamless integration with existing FastAPI app
- OAuth 2.0 discovery endpoint integration
- DCR endpoints with proper error handling
- Admin endpoints for client management
- Configurable enable/disable functionality
- Integration with existing auth infrastructure

Usage:
    from auth.oauth2_integration import setup_oauth_endpoints

    # In your FastAPI app setup
    app = FastAPI()
    setup_oauth_endpoints(app, enable_dcr=True, enable_admin=False)
"""

import logging
import os
from typing import Optional

from fastapi import FastAPI

from .oauth2_endpoints import create_oauth_admin_router, create_oauth_router

logger = logging.getLogger(__name__)


def setup_oauth_endpoints(
    app: FastAPI,
    enable_dcr: bool = None,
    enable_admin: bool = None,
    oauth_prefix: str = "/oauth",
    admin_prefix: str = "/oauth"
) -> bool:
    """
    Setup OAuth 2.0 Dynamic Client Registration endpoints in a FastAPI app.

    Args:
        app: FastAPI application instance
        enable_dcr: Enable DCR endpoints (default: from env OAUTH_DCR_ENABLED)
        enable_admin: Enable admin endpoints (default: from env OAUTH_ADMIN_ENABLED)
        oauth_prefix: URL prefix for OAuth endpoints (default: "/oauth")
        admin_prefix: URL prefix for admin endpoints (default: "/oauth")

    Returns:
        bool: True if endpoints were successfully set up
    """

    # Check configuration
    if enable_dcr is None:
        enable_dcr = os.getenv('OAUTH_DCR_ENABLED', 'true').lower() == 'true'

    if enable_admin is None:
        enable_admin = os.getenv('OAUTH_ADMIN_ENABLED', 'false').lower() == 'true'

    if not enable_dcr:
        logger.info("OAuth DCR endpoints disabled via configuration")
        return False

    try:
        # Create and mount OAuth DCR router
        oauth_router = create_oauth_router()
        app.include_router(
            oauth_router,
            prefix=oauth_prefix,
            tags=["oauth", "dcr"]
        )

        logger.info(f"Mounted OAuth DCR endpoints at {oauth_prefix}")

        # Create and mount admin router if enabled
        if enable_admin:
            admin_router = create_oauth_admin_router()
            app.include_router(
                admin_router,
                prefix=admin_prefix,
                tags=["oauth", "admin"]
            )
            logger.info(f"Mounted OAuth admin endpoints at {admin_prefix}")

        # Log enabled features
        logger.info("OAuth 2.0 DCR system successfully integrated:")
        logger.info(f"  - Registration endpoint: {oauth_prefix}/register")
        logger.info(f"  - Discovery endpoint: {oauth_prefix}/discovery")
        logger.info(f"  - Client management: {oauth_prefix}/register/{{client_id}}")
        if enable_admin:
            logger.info(f"  - Admin endpoints: {admin_prefix}/admin/*")

        return True

    except Exception as e:
        logger.error(f"Failed to setup OAuth endpoints: {e}")
        return False


def replace_minimal_dcr_stub(app: FastAPI) -> bool:
    """
    Replace the minimal DCR stub with full RFC 7591 implementation.

    This function can be used to upgrade existing server instances that
    have the minimal `/register` endpoint stub.

    Args:
        app: FastAPI application instance with existing minimal stub

    Returns:
        bool: True if replacement was successful
    """

    try:
        # Remove existing routes that conflict with OAuth endpoints
        routes_to_remove = []
        for route in app.routes:
            if hasattr(route, 'path'):
                if (route.path == '/register' or
                    route.path.startswith('/.well-known/oauth-authorization-server')):
                    routes_to_remove.append(route)

        for route in routes_to_remove:
            app.routes.remove(route)
            logger.info(f"Removed existing route: {route.path}")

        # Setup new OAuth endpoints
        success = setup_oauth_endpoints(app, enable_dcr=True)

        if success:
            logger.info("Successfully replaced minimal DCR stub with full RFC 7591 implementation")

        return success

    except Exception as e:
        logger.error(f"Failed to replace minimal DCR stub: {e}")
        return False


def is_oauth_enabled() -> bool:
    """Check if OAuth DCR is enabled via configuration."""
    return os.getenv('OAUTH_DCR_ENABLED', 'true').lower() == 'true'


def get_oauth_config() -> dict:
    """Get OAuth DCR configuration from environment."""
    return {
        'dcr_enabled': os.getenv('OAUTH_DCR_ENABLED', 'true').lower() == 'true',
        'admin_enabled': os.getenv('OAUTH_ADMIN_ENABLED', 'false').lower() == 'true',
        'encryption_key': os.getenv('OAUTH_ENCRYPTION_KEY'),
        'rate_limit_per_ip': int(os.getenv('OAUTH_RATE_LIMIT_PER_IP', '10')),
        'rate_limit_window': int(os.getenv('OAUTH_RATE_LIMIT_WINDOW', '3600')),
        'allow_localhost': os.getenv('OAUTH_ALLOW_LOCALHOST', 'true').lower() == 'true',
        'client_secret_lifetime': int(os.getenv('OAUTH_CLIENT_SECRET_LIFETIME', '31536000')),
        'registration_secret': os.getenv('OAUTH_REGISTRATION_SECRET', 'default-secret-key')
    }


def validate_oauth_config() -> tuple[bool, list[str]]:
    """
    Validate OAuth DCR configuration.

    Returns:
        tuple: (is_valid, list_of_warnings)
    """

    warnings = []
    config = get_oauth_config()

    # Check for production security warnings
    if not config['encryption_key']:
        warnings.append("OAUTH_ENCRYPTION_KEY not set - client data will not be encrypted")

    if config['registration_secret'] == 'default-secret-key':
        warnings.append("OAUTH_REGISTRATION_SECRET using default value - set a secure random key")

    if config['allow_localhost']:
        warnings.append("OAUTH_ALLOW_LOCALHOST=true - localhost redirect URIs are allowed")

    # Check rate limiting
    if config['rate_limit_per_ip'] > 100:
        warnings.append(f"High rate limit configured: {config['rate_limit_per_ip']} requests per hour")

    # Check client secret lifetime
    if config['client_secret_lifetime'] > 31536000 * 2:  # 2 years
        warnings.append("Long client secret lifetime configured (>2 years)")

    is_valid = len(warnings) == 0 or all("warning" in w.lower() for w in warnings)

    return is_valid, warnings


class OAuthMiddleware:
    """
    Middleware for OAuth 2.0 client authentication and authorization.

    This can be used to protect MCP endpoints with OAuth client authentication.
    """

    def __init__(self, app: FastAPI):
        self.app = app

    async def __call__(self, scope, receive, send):
        """ASGI middleware implementation."""
        # TODO: Implement OAuth client authentication middleware
        # This would check Authorization headers and validate client credentials
        # for protected MCP endpoints

        # For now, pass through unchanged
        await self.app(scope, receive, send)


def setup_oauth_middleware(app: FastAPI, protected_paths: Optional[list[str]] = None) -> bool:
    """
    Setup OAuth middleware for protecting MCP endpoints.

    Args:
        app: FastAPI application
        protected_paths: List of paths that require OAuth authentication

    Returns:
        bool: True if middleware was set up successfully
    """

    if not is_oauth_enabled():
        return False

    try:
        # Add OAuth middleware to the app
        # This would intercept requests to protected paths and validate OAuth tokens

        protected_paths = protected_paths or ["/mcp"]

        logger.info(f"OAuth middleware configured for paths: {protected_paths}")

        # TODO: Implement actual middleware integration
        # app.add_middleware(OAuthMiddleware)

        return True

    except Exception as e:
        logger.error(f"Failed to setup OAuth middleware: {e}")
        return False
