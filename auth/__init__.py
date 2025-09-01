"""
Authentication module for Zen MCP Server

This module provides comprehensive authentication capabilities including:
- WebAuthn device authentication (Touch ID, Face ID, Windows Hello, security keys)
- OAuth 2.0 authorization server with PKCE support
- JWT token generation and validation
- Device-based authentication flows
- Session management and security

Components:
- webauthn_flow: WebAuthn device authentication
- oauth2_server: OAuth 2.0 authorization server
- device_auth: Cross-platform device authentication utilities
- macos_keychain: macOS Keychain integration
- windows_hello: Windows Hello integration
"""

# WebAuthn authentication
from .webauthn_flow import (
    DEVICE_AUTH_HTML,
    AuthChallenge,
    DeviceCredential,
    MCPDeviceAuthMiddleware,
    WebAuthnDeviceAuth,
)

# OAuth 2.0 authorization server
try:
    from .oauth2_server import (
        AccessToken,
        AuthorizationCode,
        ErrorCode,
        GrantType,
        OAuth2Error,
        OAuth2Middleware,
        OAuth2Server,
        OAuthClient,
        RefreshToken,
        TokenType,
        create_oauth2_server,
    )
    OAUTH2_AVAILABLE = True
except ImportError:
    OAUTH2_AVAILABLE = False

# OAuth 2.0 JWT Token Management
try:
    from .oauth2_tokens import (
        JWKS,
        AlgorithmType,
        JWKSKey,
        OAuth2TokenManager,
        RateLimiter,
        TokenClaims,
        TokenMetadata,
        TokenSecurity,
        TokenStatus,
        create_oauth2_token_manager,
    )
    from .oauth2_tokens import TokenType as OAuth2TokenType
    OAUTH2_TOKENS_AVAILABLE = True
except ImportError:
    OAUTH2_TOKENS_AVAILABLE = False

# Device authentication utilities (skip broken imports)
DeviceAuthManager = None
MacOSKeychainAuth = None
WindowsHelloAuth = None

# Export main classes and functions
__all__ = [
    # WebAuthn
    "WebAuthnDeviceAuth",
    "MCPDeviceAuthMiddleware",
    "AuthChallenge",
    "DeviceCredential",
    "DEVICE_AUTH_HTML",

    # OAuth 2.0 (if available)
    "OAuth2Server" if OAUTH2_AVAILABLE else None,
    "OAuth2Middleware" if OAUTH2_AVAILABLE else None,
    "OAuth2Error" if OAUTH2_AVAILABLE else None,
    "OAuthClient" if OAUTH2_AVAILABLE else None,
    "create_oauth2_server" if OAUTH2_AVAILABLE else None,

    # OAuth 2.0 JWT Token Management (if available)
    "OAuth2TokenManager" if OAUTH2_TOKENS_AVAILABLE else None,
    "OAuth2TokenType" if OAUTH2_TOKENS_AVAILABLE else None,
    "AlgorithmType" if OAUTH2_TOKENS_AVAILABLE else None,
    "TokenStatus" if OAUTH2_TOKENS_AVAILABLE else None,
    "TokenClaims" if OAUTH2_TOKENS_AVAILABLE else None,
    "TokenMetadata" if OAUTH2_TOKENS_AVAILABLE else None,
    "JWKSKey" if OAUTH2_TOKENS_AVAILABLE else None,
    "JWKS" if OAUTH2_TOKENS_AVAILABLE else None,
    "RateLimiter" if OAUTH2_TOKENS_AVAILABLE else None,
    "TokenSecurity" if OAUTH2_TOKENS_AVAILABLE else None,
    "create_oauth2_token_manager" if OAUTH2_TOKENS_AVAILABLE else None,

    # Utilities
    "DeviceAuthManager",
    "MacOSKeychainAuth",
    "WindowsHelloAuth",

    # Availability flags
    "OAUTH2_AVAILABLE",
    "OAUTH2_TOKENS_AVAILABLE"
]

# Remove None values from __all__
__all__ = [item for item in __all__ if item is not None]
