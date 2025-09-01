#!/usr/bin/env python3
"""
Tests for OAuth 2.0 Authorization Server

This module tests the OAuth 2.0 authorization server implementation,
including authorization code flow, PKCE validation, token generation,
and WebAuthn integration.
"""

import base64
import json
import secrets
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from fastapi import Request
    from fastapi.datastructures import FormData, QueryParams
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE or not JWT_AVAILABLE,
    reason="FastAPI and PyJWT required for OAuth 2.0 tests"
)

from auth.oauth2_server import ErrorCode, OAuth2Error, OAuth2Server, OAuthClient, create_oauth2_server
from auth.webauthn_flow import WebAuthnDeviceAuth


class TestOAuth2Server:
    """Test OAuth 2.0 authorization server functionality."""

    @pytest.fixture
    def webauthn_mock(self):
        """Mock WebAuthn authentication."""
        return MagicMock(spec=WebAuthnDeviceAuth)

    @pytest.fixture
    def oauth2_server(self, webauthn_mock):
        """Create OAuth 2.0 server for testing."""
        return OAuth2Server(
            issuer="https://test.example.com",
            webauthn_auth=webauthn_mock
        )

    @pytest.fixture
    def test_client(self, oauth2_server):
        """Create test OAuth client."""
        client = OAuthClient(
            client_id="test-client",
            redirect_uris=["https://app.example.com/callback"],
            name="Test Client"
        )
        oauth2_server.register_client(client)
        return client

    def test_oauth2_server_initialization(self, oauth2_server):
        """Test OAuth 2.0 server initialization."""
        assert oauth2_server.issuer == "https://test.example.com"
        assert len(oauth2_server.available_scopes) > 0
        assert "mcp:read" in oauth2_server.available_scopes
        assert "mcp:write" in oauth2_server.available_scopes

        # Check default client registration
        assert "mcp-default-client" in oauth2_server.clients

    def test_client_registration(self, oauth2_server):
        """Test OAuth client registration."""
        client = OAuthClient(
            client_id="new-test-client",
            redirect_uris=["https://new.example.com/callback"],
            name="New Test Client"
        )

        client_id = oauth2_server.register_client(client)
        assert client_id == "new-test-client"
        assert oauth2_server.get_client("new-test-client") == client

    def test_duplicate_client_registration(self, oauth2_server):
        """Test duplicate client registration raises error."""
        client = OAuthClient(
            client_id="duplicate-client",
            redirect_uris=["https://dup.example.com/callback"]
        )

        oauth2_server.register_client(client)

        with pytest.raises(OAuth2Error) as exc_info:
            oauth2_server.register_client(client)
        assert exc_info.value.error == ErrorCode.INVALID_CLIENT

    def test_redirect_uri_validation(self, oauth2_server, test_client):
        """Test redirect URI validation."""
        # Valid URI
        assert oauth2_server.validate_redirect_uri(
            test_client.client_id,
            "https://app.example.com/callback"
        )

        # Invalid URI
        assert not oauth2_server.validate_redirect_uri(
            test_client.client_id,
            "https://malicious.com/callback"
        )

        # Unknown client
        assert not oauth2_server.validate_redirect_uri(
            "unknown-client",
            "https://app.example.com/callback"
        )

    def test_pkce_code_challenge_generation(self, oauth2_server):
        """Test PKCE code challenge generation and verification."""
        code_verifier = secrets.token_urlsafe(32)

        # S256 method
        challenge_s256 = oauth2_server._generate_code_challenge(code_verifier, "S256")
        assert oauth2_server._verify_code_challenge(code_verifier, challenge_s256, "S256")

        # Plain method
        challenge_plain = oauth2_server._generate_code_challenge(code_verifier, "plain")
        assert challenge_plain == code_verifier
        assert oauth2_server._verify_code_challenge(code_verifier, challenge_plain, "plain")

    def test_authorization_code_generation(self, oauth2_server, test_client):
        """Test authorization code generation."""
        auth_code = oauth2_server._generate_authorization_code(
            client_id=test_client.client_id,
            user_id="test@example.com",
            redirect_uri="https://app.example.com/callback",
            scope="mcp:read mcp:write",
            code_challenge="test-challenge",
            code_challenge_method="S256"
        )

        assert auth_code.client_id == test_client.client_id
        assert auth_code.user_id == "test@example.com"
        assert auth_code.scope == "mcp:read mcp:write"
        assert auth_code.code_challenge == "test-challenge"
        assert not auth_code.used
        assert time.time() < auth_code.expires_at

    def test_jwt_token_generation(self, oauth2_server, test_client):
        """Test JWT token generation."""
        token = oauth2_server._generate_jwt_token(
            client_id=test_client.client_id,
            user_id="test@example.com",
            scope="mcp:read mcp:write"
        )

        # Verify JWT structure
        assert len(token.split('.')) == 3

        # Decode and verify payload
        oauth2_server.jwt_public_key.public_bytes(
            encoding=jwt.get_unverified_header(token)['alg'] == 'RS256' and
                    jwt.algorithms.get_default_algorithms()['RS256'].prepare_key(
                        oauth2_server.jwt_public_key
                    ).public_bytes_raw() or
                    oauth2_server.jwt_public_key.public_bytes(
                        encoding=jwt.get_default_algorithms()['RS256'].hash_alg.name,
                        format=jwt.get_default_algorithms()['RS256'].hash_alg.digest_size
                    )
        )

        # For testing, just check the token is valid JWT format
        parts = token.split('.')
        assert len(parts) == 3

        # Decode header and payload (without verification for testing)
        header = json.loads(base64.urlsafe_b64decode(parts[0] + '=='))
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + '=='))

        assert header['alg'] == 'RS256'
        assert payload['iss'] == oauth2_server.issuer
        assert payload['sub'] == "test@example.com"
        assert payload['aud'] == test_client.client_id

    @pytest.mark.asyncio
    async def test_authorization_endpoint_missing_params(self, oauth2_server):
        """Test authorization endpoint with missing parameters."""
        # Mock request with missing parameters
        request = MagicMock()
        request.query_params = QueryParams("")

        response = await oauth2_server.authorization_endpoint(request)

        # Should return error response
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_authorization_endpoint_invalid_client(self, oauth2_server):
        """Test authorization endpoint with invalid client."""
        request = MagicMock()
        request.query_params = QueryParams(
            "response_type=code&client_id=invalid-client&redirect_uri=https://example.com/callback"
        )

        response = await oauth2_server.authorization_endpoint(request)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_authorization_endpoint_valid_request(self, oauth2_server, test_client):
        """Test authorization endpoint with valid request."""
        request = MagicMock()
        request.query_params = QueryParams(
            f"response_type=code&client_id={test_client.client_id}"
            f"&redirect_uri={test_client.redirect_uris[0]}"
            f"&scope=mcp:read&code_challenge=test-challenge&code_challenge_method=S256"
        )
        request.url._url = "https://test.example.com/oauth/authorize"

        # Mock WebAuthn session check to return authenticated user
        with patch.object(oauth2_server, '_check_webauthn_session', return_value="test@example.com"):
            response = await oauth2_server.authorization_endpoint(request)

        # Should redirect with authorization code
        assert response.status_code == 307
        assert "code=" in response.headers["location"]

    @pytest.mark.asyncio
    async def test_token_endpoint_authorization_code_grant(self, oauth2_server, test_client):
        """Test token endpoint with authorization code grant."""
        # First create an authorization code
        code_verifier = secrets.token_urlsafe(32)
        code_challenge = oauth2_server._generate_code_challenge(code_verifier, "S256")

        auth_code = oauth2_server._generate_authorization_code(
            client_id=test_client.client_id,
            user_id="test@example.com",
            redirect_uri=test_client.redirect_uris[0],
            scope="mcp:read",
            code_challenge=code_challenge,
            code_challenge_method="S256"
        )

        # Mock form data
        form_data = {
            "grant_type": "authorization_code",
            "code": auth_code.code,
            "redirect_uri": test_client.redirect_uris[0],
            "client_id": test_client.client_id,
            "code_verifier": code_verifier
        }

        request = MagicMock()
        request.form = AsyncMock(return_value=form_data)

        response = await oauth2_server.token_endpoint(request)

        assert response.status_code == 200
        content = json.loads(response.body)
        assert "access_token" in content
        assert "refresh_token" in content
        assert content["token_type"] == "Bearer"

    @pytest.mark.asyncio
    async def test_token_endpoint_invalid_grant(self, oauth2_server, test_client):
        """Test token endpoint with invalid authorization code."""
        form_data = {
            "grant_type": "authorization_code",
            "code": "invalid-code",
            "redirect_uri": test_client.redirect_uris[0],
            "client_id": test_client.client_id,
            "code_verifier": "invalid-verifier"
        }

        request = MagicMock()
        request.form = AsyncMock(return_value=form_data)

        response = await oauth2_server.token_endpoint(request)

        assert response.status_code == 400
        content = json.loads(response.body)
        assert content["error"] == "invalid_grant"

    @pytest.mark.asyncio
    async def test_revocation_endpoint(self, oauth2_server):
        """Test token revocation endpoint."""
        # Create a token to revoke
        token = "test-access-token"
        oauth2_server.access_tokens[token] = MagicMock()

        form_data = {"token": token}
        request = MagicMock()
        request.form = AsyncMock(return_value=form_data)

        response = await oauth2_server.revocation_endpoint(request)

        assert response.status_code == 200
        assert token not in oauth2_server.access_tokens

    @pytest.mark.asyncio
    async def test_introspection_endpoint_active_token(self, oauth2_server, test_client):
        """Test token introspection for active token."""
        # Create an active token
        from auth.oauth2_server import AccessToken
        token = "test-active-token"
        access_token = AccessToken(
            token=token,
            client_id=test_client.client_id,
            user_id="test@example.com",
            scope="mcp:read",
            expires_at=time.time() + 3600
        )
        oauth2_server.access_tokens[token] = access_token

        form_data = {"token": token}
        request = MagicMock()
        request.form = AsyncMock(return_value=form_data)

        response = await oauth2_server.introspection_endpoint(request)

        assert response.status_code == 200
        content = json.loads(response.body)
        assert content["active"] is True
        assert content["client_id"] == test_client.client_id

    @pytest.mark.asyncio
    async def test_introspection_endpoint_inactive_token(self, oauth2_server):
        """Test token introspection for inactive token."""
        form_data = {"token": "invalid-token"}
        request = MagicMock()
        request.form = AsyncMock(return_value=form_data)

        response = await oauth2_server.introspection_endpoint(request)

        assert response.status_code == 200
        content = json.loads(response.body)
        assert content["active"] is False

    @pytest.mark.asyncio
    async def test_bearer_token_validation(self, oauth2_server, test_client):
        """Test Bearer token validation."""
        # Create a valid token
        from auth.oauth2_server import AccessToken
        token = "test-bearer-token"
        access_token = AccessToken(
            token=token,
            client_id=test_client.client_id,
            user_id="test@example.com",
            scope="mcp:read mcp:write"
        )
        oauth2_server.access_tokens[token] = access_token

        # Test valid token
        token_info = await oauth2_server.validate_bearer_token(f"Bearer {token}")
        assert token_info is not None
        assert token_info["user_id"] == "test@example.com"
        assert token_info["client_id"] == test_client.client_id

        # Test invalid token
        token_info = await oauth2_server.validate_bearer_token("Bearer invalid-token")
        assert token_info is None

        # Test malformed header
        token_info = await oauth2_server.validate_bearer_token("Invalid header")
        assert token_info is None

    def test_cleanup_expired_tokens(self, oauth2_server):
        """Test cleanup of expired tokens and codes."""
        from auth.oauth2_server import AccessToken, AuthorizationCode

        # Add expired tokens
        expired_token = AccessToken(
            token="expired-token",
            expires_at=time.time() - 100
        )
        oauth2_server.access_tokens["expired-token"] = expired_token

        # Add valid token
        valid_token = AccessToken(
            token="valid-token",
            expires_at=time.time() + 3600
        )
        oauth2_server.access_tokens["valid-token"] = valid_token

        # Add expired authorization code
        expired_code = AuthorizationCode(
            code="expired-code",
            client_id="test-client",
            user_id="test@example.com",
            redirect_uri="https://example.com/callback",
            scope="mcp:read",
            expires_at=time.time() - 100
        )
        oauth2_server.authorization_codes["expired-code"] = expired_code

        # Run cleanup
        oauth2_server.cleanup_expired_tokens()

        # Check expired items removed
        assert "expired-token" not in oauth2_server.access_tokens
        assert "expired-code" not in oauth2_server.authorization_codes

        # Check valid items preserved
        assert "valid-token" in oauth2_server.access_tokens

    def test_create_oauth2_server_factory(self):
        """Test OAuth 2.0 server factory function."""
        webauthn = MagicMock()
        server = create_oauth2_server(
            issuer="https://factory-test.com",
            webauthn_auth=webauthn
        )

        assert isinstance(server, OAuth2Server)
        assert server.issuer == "https://factory-test.com"
        assert server.webauthn == webauthn


class TestOAuth2Middleware:
    """Test OAuth 2.0 middleware functionality."""

    @pytest.fixture
    def oauth2_server(self):
        """Create OAuth 2.0 server for middleware testing."""
        return MagicMock(spec=OAuth2Server)

    @pytest.fixture
    def middleware(self, oauth2_server):
        """Create OAuth 2.0 middleware."""
        from auth.oauth2_server import OAuth2Middleware
        return OAuth2Middleware(oauth2_server)

    @pytest.mark.asyncio
    async def test_middleware_public_endpoint(self, middleware):
        """Test middleware allows public endpoints."""
        request = MagicMock()
        request.url.path = "/health"
        request.state = MagicMock()

        call_next = AsyncMock(return_value=MagicMock())

        await middleware(request, call_next)

        call_next.assert_called_once_with(request)
        assert not hasattr(request.state, 'oauth_token')

    @pytest.mark.asyncio
    async def test_middleware_protected_endpoint_no_token(self, middleware, oauth2_server):
        """Test middleware blocks protected endpoints without token."""
        request = MagicMock()
        request.url.path = "/mcp/tools"
        request.headers.get.return_value = None

        oauth2_server.validate_bearer_token.return_value = None

        call_next = AsyncMock()

        response = await middleware(request, call_next)

        assert response.status_code == 401
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_middleware_protected_endpoint_valid_token(self, middleware, oauth2_server):
        """Test middleware allows protected endpoints with valid token."""
        request = MagicMock()
        request.url.path = "/mcp/tools"
        request.headers.get.return_value = "Bearer valid-token"
        request.state = MagicMock()

        token_info = {"user_id": "test@example.com", "client_id": "test-client"}
        oauth2_server.validate_bearer_token.return_value = token_info

        call_next = AsyncMock(return_value=MagicMock())

        await middleware(request, call_next)

        call_next.assert_called_once_with(request)
        assert request.state.oauth_token == token_info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
