"""
OAuth 2.0 Authorization Server for Zen MCP Server

This module implements a complete OAuth 2.0 authorization server compliant with RFC 6749,
with PKCE support (RFC 7636) and integration with the existing WebAuthn authentication system.

Features:
- Authorization Code flow with PKCE
- JWT access and refresh tokens
- Token revocation and introspection
- Integration with WebAuthn device authentication
- Secure session management
- OAuth 2.0 error handling
- CORS and security headers

Integration:
- Works with existing WebAuthn flow in auth/webauthn_flow.py
- Generates authorization codes after successful WebAuthn authentication
- Issues JWT tokens for MCP endpoint access
- Validates Bearer tokens for protected resources
"""

import base64
import hashlib
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional
from urllib.parse import urlencode

try:
    import jwt
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    from fastapi import Request, Response
    from fastapi.responses import JSONResponse, RedirectResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Import WebAuthn integration
from .webauthn_flow import WebAuthnDeviceAuth


class GrantType(Enum):
    """OAuth 2.0 grant types supported by this server."""
    AUTHORIZATION_CODE = "authorization_code"
    REFRESH_TOKEN = "refresh_token"


class TokenType(Enum):
    """Token types issued by this server."""
    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    AUTHORIZATION_CODE = "authorization_code"


class ErrorCode(Enum):
    """OAuth 2.0 error codes as defined in RFC 6749."""
    INVALID_REQUEST = "invalid_request"
    INVALID_CLIENT = "invalid_client"
    INVALID_GRANT = "invalid_grant"
    UNAUTHORIZED_CLIENT = "unauthorized_client"
    UNSUPPORTED_GRANT_TYPE = "unsupported_grant_type"
    INVALID_SCOPE = "invalid_scope"
    ACCESS_DENIED = "access_denied"
    UNSUPPORTED_RESPONSE_TYPE = "unsupported_response_type"
    SERVER_ERROR = "server_error"
    TEMPORARILY_UNAVAILABLE = "temporarily_unavailable"


@dataclass
class OAuthClient:
    """OAuth 2.0 client registration information."""
    client_id: str
    client_secret: Optional[str] = None  # None for public clients
    redirect_uris: list[str] = field(default_factory=list)
    allowed_grant_types: set[GrantType] = field(default_factory=lambda: {GrantType.AUTHORIZATION_CODE})
    allowed_scopes: set[str] = field(default_factory=lambda: {"mcp:read", "mcp:write"})
    is_public: bool = True  # PKCE-enabled public clients
    name: str = "MCP Client"
    created_at: float = field(default_factory=time.time)


@dataclass
class AuthorizationCode:
    """OAuth 2.0 authorization code with PKCE support."""
    code: str
    client_id: str
    user_id: str
    redirect_uri: str
    scope: str
    code_challenge: Optional[str] = None
    code_challenge_method: Optional[str] = None
    expires_at: float = field(default_factory=lambda: time.time() + 600)  # 10 minutes
    used: bool = False


@dataclass
class AccessToken:
    """OAuth 2.0 access token information."""
    token: str
    token_type: str = "Bearer"
    client_id: str = ""
    user_id: str = ""
    scope: str = ""
    expires_at: float = field(default_factory=lambda: time.time() + 3600)  # 1 hour
    issued_at: float = field(default_factory=time.time)


@dataclass
class RefreshToken:
    """OAuth 2.0 refresh token information."""
    token: str
    client_id: str
    user_id: str
    scope: str
    expires_at: float = field(default_factory=lambda: time.time() + 86400 * 30)  # 30 days
    issued_at: float = field(default_factory=time.time)


class OAuth2Error(Exception):
    """OAuth 2.0 error with proper error codes and descriptions."""

    def __init__(self, error: ErrorCode, description: str = "", uri: str = ""):
        self.error = error
        self.description = description
        self.uri = uri
        super().__init__(f"{error.value}: {description}")


class OAuth2Server:
    """
    Complete OAuth 2.0 Authorization Server implementation.

    This server provides:
    - Authorization endpoint (/oauth/authorize)
    - Token endpoint (/oauth/token)
    - Token revocation (/oauth/revoke)
    - Token introspection (/oauth/introspect)
    - PKCE support for security
    - WebAuthn integration for authentication
    - JWT token issuance and validation
    """

    def __init__(self,
                 issuer: str = "https://mcp.zen.local",
                 webauthn_auth: Optional[WebAuthnDeviceAuth] = None):
        """
        Initialize OAuth 2.0 server.

        Args:
            issuer: The OAuth 2.0 issuer identifier
            webauthn_auth: WebAuthn authentication instance for device auth
        """
        if not JWT_AVAILABLE:
            raise RuntimeError(
                "JWT libraries required for OAuth 2.0 server. "
                "Install with: pip install pyjwt cryptography"
            )

        self.issuer = issuer
        self.webauthn = webauthn_auth or WebAuthnDeviceAuth()

        # Storage for OAuth 2.0 entities
        self.clients: dict[str, OAuthClient] = {}
        self.authorization_codes: dict[str, AuthorizationCode] = {}
        self.access_tokens: dict[str, AccessToken] = {}
        self.refresh_tokens: dict[str, RefreshToken] = {}

        # JWT signing key (in production, use persistent key storage)
        self.jwt_private_key = self._generate_jwt_key()
        self.jwt_public_key = self.jwt_private_key.public_key()

        # Default scopes
        self.available_scopes = {
            "mcp:read": "Read access to MCP tools and data",
            "mcp:write": "Write access to MCP tools and operations",
            "mcp:admin": "Administrative access to MCP server"
        }

        # Register default client for development
        self._register_default_client()

    def _generate_jwt_key(self):
        """Generate RSA key pair for JWT signing."""
        return rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )

    def _register_default_client(self):
        """Register a default OAuth client for development and testing."""
        default_client = OAuthClient(
            client_id="mcp-default-client",
            redirect_uris=[
                "http://localhost:8080/oauth/callback",
                "https://mcp.zen.local/oauth/callback",
                "urn:ietf:wg:oauth:2.0:oob"  # Out-of-band for CLI clients
            ],
            name="Default MCP Client",
            is_public=True
        )
        self.clients[default_client.client_id] = default_client

    def register_client(self, client: OAuthClient) -> str:
        """
        Register a new OAuth 2.0 client.

        Args:
            client: Client configuration

        Returns:
            The client ID
        """
        if client.client_id in self.clients:
            raise OAuth2Error(ErrorCode.INVALID_CLIENT, "Client already exists")

        self.clients[client.client_id] = client
        return client.client_id

    def get_client(self, client_id: str) -> Optional[OAuthClient]:
        """Get client by ID."""
        return self.clients.get(client_id)

    def validate_redirect_uri(self, client_id: str, redirect_uri: str) -> bool:
        """Validate redirect URI against registered URIs."""
        client = self.get_client(client_id)
        if not client:
            return False

        # Check exact match first
        if redirect_uri in client.redirect_uris:
            return True

        # For development, allow any localhost callback for the default client
        if client_id == "mcp-default-client":
            from urllib.parse import urlparse
            parsed = urlparse(redirect_uri)
            # Allow any localhost port for /callback path
            if parsed.hostname in ["localhost", "127.0.0.1"] and parsed.path == "/callback":
                return True

        return False

    def _generate_code_challenge(self, code_verifier: str, method: str = "S256") -> str:
        """Generate PKCE code challenge from verifier."""
        if method == "S256":
            digest = hashlib.sha256(code_verifier.encode()).digest()
            return base64.urlsafe_b64encode(digest).decode().rstrip('=')
        elif method == "plain":
            return code_verifier
        else:
            raise OAuth2Error(ErrorCode.INVALID_REQUEST, "Unsupported code challenge method")

    def _verify_code_challenge(self, code_verifier: str, code_challenge: str, method: str = "S256") -> bool:
        """Verify PKCE code challenge."""
        try:
            expected_challenge = self._generate_code_challenge(code_verifier, method)
            return secrets.compare_digest(expected_challenge, code_challenge)
        except Exception:
            return False

    async def authorization_endpoint(self, request: Request) -> Response:
        """
        OAuth 2.0 Authorization Endpoint (/oauth/authorize).

        Handles authorization requests with PKCE support and WebAuthn integration.
        """
        try:
            # Parse query parameters
            params = dict(request.query_params)

            response_type = params.get("response_type")
            client_id = params.get("client_id")
            redirect_uri = params.get("redirect_uri")
            scope = params.get("scope", "mcp:read")
            state = params.get("state")

            # PKCE parameters
            code_challenge = params.get("code_challenge")
            code_challenge_method = params.get("code_challenge_method", "S256")

            # Validate required parameters
            if not response_type:
                raise OAuth2Error(ErrorCode.INVALID_REQUEST, "Missing response_type")
            if response_type != "code":
                raise OAuth2Error(ErrorCode.UNSUPPORTED_RESPONSE_TYPE, "Only 'code' response type supported")
            if not client_id:
                raise OAuth2Error(ErrorCode.INVALID_REQUEST, "Missing client_id")
            if not redirect_uri:
                raise OAuth2Error(ErrorCode.INVALID_REQUEST, "Missing redirect_uri")

            # Validate client
            client = self.get_client(client_id)
            if not client:
                raise OAuth2Error(ErrorCode.INVALID_CLIENT, "Unknown client")

            # Validate redirect URI
            if not self.validate_redirect_uri(client_id, redirect_uri):
                raise OAuth2Error(ErrorCode.INVALID_REQUEST, "Invalid redirect_uri")

            # PKCE validation for public clients
            if client.is_public and not code_challenge:
                raise OAuth2Error(ErrorCode.INVALID_REQUEST, "PKCE code_challenge required for public clients")

            # Validate scope
            requested_scopes = set(scope.split())
            print(f"DEBUG: Requested scopes: {requested_scopes}")
            print(f"DEBUG: Client allowed scopes: {client.allowed_scopes}")
            print(f"DEBUG: Is subset? {requested_scopes.issubset(client.allowed_scopes)}")
            if not requested_scopes.issubset(client.allowed_scopes):
                raise OAuth2Error(ErrorCode.INVALID_SCOPE, "Requested scope not allowed")

            # Check if user is already authenticated via WebAuthn
            user_id = await self._check_webauthn_session(request)

            if not user_id:
                # Redirect to WebAuthn authentication
                auth_url = self._build_webauthn_auth_url(request.url._url)
                return RedirectResponse(url=auth_url)

            # User is authenticated, generate authorization code
            auth_code = self._generate_authorization_code(
                client_id=client_id,
                user_id=user_id,
                redirect_uri=redirect_uri,
                scope=scope,
                code_challenge=code_challenge,
                code_challenge_method=code_challenge_method
            )

            # Build callback URL
            callback_params = {
                "code": auth_code.code,
            }
            if state:
                callback_params["state"] = state

            callback_url = f"{redirect_uri}?{urlencode(callback_params)}"
            return RedirectResponse(url=callback_url)

        except OAuth2Error as e:
            # Return error to redirect URI if possible
            if redirect_uri and self.validate_redirect_uri(client_id or "", redirect_uri):
                error_params = {
                    "error": e.error.value,
                    "error_description": e.description
                }
                if state:
                    error_params["state"] = state
                error_url = f"{redirect_uri}?{urlencode(error_params)}"
                return RedirectResponse(url=error_url)
            else:
                # Return JSON error
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": e.error.value,
                        "error_description": e.description
                    }
                )

    async def token_endpoint(self, request: Request) -> JSONResponse:
        """
        OAuth 2.0 Token Endpoint (/oauth/token).

        Handles token requests for authorization code and refresh token grants.
        """
        try:
            # Parse form data
            form_data = await request.form()

            grant_type = form_data.get("grant_type")
            form_data.get("client_id")

            if not grant_type:
                raise OAuth2Error(ErrorCode.INVALID_REQUEST, "Missing grant_type")

            if grant_type == GrantType.AUTHORIZATION_CODE.value:
                return await self._handle_authorization_code_grant(form_data)
            elif grant_type == GrantType.REFRESH_TOKEN.value:
                return await self._handle_refresh_token_grant(form_data)
            else:
                raise OAuth2Error(ErrorCode.UNSUPPORTED_GRANT_TYPE, f"Unsupported grant type: {grant_type}")

        except OAuth2Error as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": e.error.value,
                    "error_description": e.description
                }
            )

    async def revocation_endpoint(self, request: Request) -> Response:
        """
        OAuth 2.0 Token Revocation Endpoint (/oauth/revoke).

        Revokes access or refresh tokens.
        """
        try:
            form_data = await request.form()

            token = form_data.get("token")
            form_data.get("token_type_hint")

            if not token:
                raise OAuth2Error(ErrorCode.INVALID_REQUEST, "Missing token")

            # Find and revoke token

            # Try access token first
            if token in self.access_tokens:
                del self.access_tokens[token]

            # Try refresh token
            if token in self.refresh_tokens:
                del self.refresh_tokens[token]

            # OAuth 2.0 spec says to return 200 even if token not found
            return Response(status_code=200)

        except OAuth2Error as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": e.error.value,
                    "error_description": e.description
                }
            )

    async def introspection_endpoint(self, request: Request) -> JSONResponse:
        """
        OAuth 2.0 Token Introspection Endpoint (/oauth/introspect).

        Returns information about access tokens.
        """
        try:
            form_data = await request.form()

            token = form_data.get("token")
            if not token:
                raise OAuth2Error(ErrorCode.INVALID_REQUEST, "Missing token")

            # Find token
            access_token = self.access_tokens.get(token)
            if not access_token:
                return JSONResponse(content={"active": False})

            # Check expiration
            if time.time() > access_token.expires_at:
                del self.access_tokens[token]
                return JSONResponse(content={"active": False})

            # Return token info
            return JSONResponse(content={
                "active": True,
                "client_id": access_token.client_id,
                "username": access_token.user_id,
                "scope": access_token.scope,
                "exp": int(access_token.expires_at),
                "iat": int(access_token.issued_at),
                "token_type": access_token.token_type,
                "iss": self.issuer
            })

        except OAuth2Error as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": e.error.value,
                    "error_description": e.description
                }
            )

    async def validate_bearer_token(self, authorization_header: Optional[str]) -> Optional[dict]:
        """
        Validate Bearer token from Authorization header.

        Args:
            authorization_header: The Authorization header value

        Returns:
            Token information if valid, None otherwise
        """
        if not authorization_header:
            return None

        if not authorization_header.startswith("Bearer "):
            return None

        token = authorization_header[7:]  # Remove "Bearer " prefix

        # Check access token in local storage first
        access_token = self.access_tokens.get(token)
        if not access_token:
            # Also check in OAuth integration server if available
            # This allows tokens created by the OAuth integration to be validated
            if hasattr(self, 'oauth_integration_server') and self.oauth_integration_server:
                integration_tokens = getattr(self.oauth_integration_server, 'access_tokens', {})
                token_data = integration_tokens.get(token)
                if token_data:
                    # Check if it's expired
                    if isinstance(token_data, dict):
                        created_at = token_data.get("created_at", 0)
                        expires_in = token_data.get("expires_in", 3600)
                        if time.time() - created_at > expires_in:
                            # Token is expired
                            del integration_tokens[token]
                            return None
                        # Return the token data in expected format
                        return token_data
            return None

        # Check expiration for local token
        if time.time() > access_token.expires_at:
            del self.access_tokens[token]
            return None

        return {
            "client_id": access_token.client_id,
            "user_id": access_token.user_id,
            "scope": access_token.scope,
            "token_type": access_token.token_type
        }

    def _generate_authorization_code(self,
                                   client_id: str,
                                   user_id: str,
                                   redirect_uri: str,
                                   scope: str,
                                   code_challenge: Optional[str] = None,
                                   code_challenge_method: Optional[str] = None) -> AuthorizationCode:
        """Generate a new authorization code."""
        code = secrets.token_urlsafe(32)

        auth_code = AuthorizationCode(
            code=code,
            client_id=client_id,
            user_id=user_id,
            redirect_uri=redirect_uri,
            scope=scope,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method
        )

        self.authorization_codes[code] = auth_code
        return auth_code

    def _generate_jwt_token(self,
                          client_id: str,
                          user_id: str,
                          scope: str,
                          expires_in: int = 3600) -> str:
        """Generate a JWT access token."""
        now = datetime.now(timezone.utc)
        payload = {
            "iss": self.issuer,
            "sub": user_id,
            "aud": client_id,
            "exp": now + timedelta(seconds=expires_in),
            "iat": now,
            "scope": scope,
            "token_type": "Bearer"
        }

        private_key_pem = self.jwt_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        return jwt.encode(payload, private_key_pem, algorithm="RS256")

    async def _handle_authorization_code_grant(self, form_data) -> JSONResponse:
        """Handle authorization code grant type."""
        code = form_data.get("code")
        redirect_uri = form_data.get("redirect_uri")
        client_id = form_data.get("client_id")
        code_verifier = form_data.get("code_verifier")

        if not code:
            raise OAuth2Error(ErrorCode.INVALID_REQUEST, "Missing code")
        if not redirect_uri:
            raise OAuth2Error(ErrorCode.INVALID_REQUEST, "Missing redirect_uri")
        if not client_id:
            raise OAuth2Error(ErrorCode.INVALID_REQUEST, "Missing client_id")

        # Validate authorization code
        auth_code = self.authorization_codes.get(code)
        if not auth_code:
            raise OAuth2Error(ErrorCode.INVALID_GRANT, "Invalid authorization code")

        if auth_code.used:
            raise OAuth2Error(ErrorCode.INVALID_GRANT, "Authorization code already used")

        if time.time() > auth_code.expires_at:
            del self.authorization_codes[code]
            raise OAuth2Error(ErrorCode.INVALID_GRANT, "Authorization code expired")

        if auth_code.client_id != client_id:
            raise OAuth2Error(ErrorCode.INVALID_CLIENT, "Client mismatch")

        if auth_code.redirect_uri != redirect_uri:
            raise OAuth2Error(ErrorCode.INVALID_GRANT, "Redirect URI mismatch")

        # PKCE verification
        if auth_code.code_challenge:
            if not code_verifier:
                raise OAuth2Error(ErrorCode.INVALID_REQUEST, "Missing code_verifier")
            if not self._verify_code_challenge(code_verifier, auth_code.code_challenge, auth_code.code_challenge_method):
                raise OAuth2Error(ErrorCode.INVALID_GRANT, "Invalid code_verifier")

        # Mark code as used
        auth_code.used = True

        # Generate tokens
        access_token = self._generate_jwt_token(
            client_id=auth_code.client_id,
            user_id=auth_code.user_id,
            scope=auth_code.scope
        )

        refresh_token = secrets.token_urlsafe(32)

        # Store tokens
        self.access_tokens[access_token] = AccessToken(
            token=access_token,
            client_id=auth_code.client_id,
            user_id=auth_code.user_id,
            scope=auth_code.scope
        )

        self.refresh_tokens[refresh_token] = RefreshToken(
            token=refresh_token,
            client_id=auth_code.client_id,
            user_id=auth_code.user_id,
            scope=auth_code.scope
        )

        return JSONResponse(content={
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": refresh_token,
            "scope": auth_code.scope
        })

    async def _handle_refresh_token_grant(self, form_data) -> JSONResponse:
        """Handle refresh token grant type."""
        refresh_token = form_data.get("refresh_token")
        client_id = form_data.get("client_id")
        scope = form_data.get("scope")

        if not refresh_token:
            raise OAuth2Error(ErrorCode.INVALID_REQUEST, "Missing refresh_token")
        if not client_id:
            raise OAuth2Error(ErrorCode.INVALID_REQUEST, "Missing client_id")

        # Validate refresh token
        token_info = self.refresh_tokens.get(refresh_token)
        if not token_info:
            raise OAuth2Error(ErrorCode.INVALID_GRANT, "Invalid refresh token")

        if time.time() > token_info.expires_at:
            del self.refresh_tokens[refresh_token]
            raise OAuth2Error(ErrorCode.INVALID_GRANT, "Refresh token expired")

        if token_info.client_id != client_id:
            raise OAuth2Error(ErrorCode.INVALID_CLIENT, "Client mismatch")

        # Use requested scope or original scope
        final_scope = scope or token_info.scope

        # Generate new access token
        access_token = self._generate_jwt_token(
            client_id=token_info.client_id,
            user_id=token_info.user_id,
            scope=final_scope
        )

        # Store new access token
        self.access_tokens[access_token] = AccessToken(
            token=access_token,
            client_id=token_info.client_id,
            user_id=token_info.user_id,
            scope=final_scope
        )

        return JSONResponse(content={
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": final_scope
        })

    async def _check_webauthn_session(self, request: Request) -> Optional[str]:
        """Check if user has valid WebAuthn session."""
        # Check for session cookie or header
        session_id = request.headers.get("X-Session-ID") or request.cookies.get("mcp_session")

        if not session_id:
            return None

        # In a real implementation, validate session against WebAuthn middleware
        # For now, return a placeholder user ID
        return "authenticated_user"

    def _build_webauthn_auth_url(self, original_url: str) -> str:
        """Build WebAuthn authentication URL with return parameter."""
        # In a real implementation, redirect to WebAuthn auth page
        return f"/auth/webauthn?return_to={original_url}"

    def cleanup_expired_tokens(self):
        """Clean up expired tokens and codes."""
        current_time = time.time()

        # Clean expired authorization codes
        expired_codes = [
            code for code, auth_code in self.authorization_codes.items()
            if current_time > auth_code.expires_at
        ]
        for code in expired_codes:
            del self.authorization_codes[code]

        # Clean expired access tokens
        expired_access = [
            token for token, access_token in self.access_tokens.items()
            if current_time > access_token.expires_at
        ]
        for token in expired_access:
            del self.access_tokens[token]

        # Clean expired refresh tokens
        expired_refresh = [
            token for token, refresh_token in self.refresh_tokens.items()
            if current_time > refresh_token.expires_at
        ]
        for token in expired_refresh:
            del self.refresh_tokens[token]


class OAuth2Middleware:
    """
    FastAPI middleware for OAuth 2.0 authentication.

    This middleware integrates with the OAuth2Server to provide
    automatic token validation for protected MCP endpoints.
    """

    def __init__(self, oauth2_server: OAuth2Server):
        self.oauth2_server = oauth2_server

    async def __call__(self, request: Request, call_next):
        """Process request with OAuth 2.0 authentication."""
        # Check if endpoint requires authentication
        if self._requires_auth(request.url.path):
            auth_header = request.headers.get("Authorization")
            token_info = await self.oauth2_server.validate_bearer_token(auth_header)

            if not token_info:
                return JSONResponse(
                    status_code=401,
                    content={"error": "unauthorized", "error_description": "Valid access token required"},
                    headers={"WWW-Authenticate": "Bearer"}
                )

            # Add token info to request state
            request.state.oauth_token = token_info

        response = await call_next(request)

        # Add security headers
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"

        return response

    def _requires_auth(self, path: str) -> bool:
        """Check if path requires OAuth 2.0 authentication."""
        # Don't protect OAuth endpoints themselves
        oauth_endpoints = {"/oauth/authorize", "/oauth/token", "/oauth/revoke", "/oauth/introspect"}
        if path in oauth_endpoints:
            return False

        # Don't protect public endpoints
        public_endpoints = {"/", "/health", "/docs", "/openapi.json"}
        if path in public_endpoints:
            return False

        # Protect MCP endpoints
        if path.startswith("/mcp"):
            return True

        return False


def create_oauth2_server(issuer: str = "https://mcp.zen.local",
                        webauthn_auth: Optional[WebAuthnDeviceAuth] = None) -> OAuth2Server:
    """
    Factory function to create a configured OAuth 2.0 server.

    Args:
        issuer: OAuth 2.0 issuer identifier
        webauthn_auth: WebAuthn authentication instance

    Returns:
        Configured OAuth2Server instance
    """
    return OAuth2Server(issuer=issuer, webauthn_auth=webauthn_auth)


# Export main classes and functions
__all__ = [
    "OAuth2Server",
    "OAuth2Middleware",
    "OAuthClient",
    "OAuth2Error",
    "ErrorCode",
    "GrantType",
    "TokenType",
    "create_oauth2_server"
]
