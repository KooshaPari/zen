"""
Test suite for OAuth2 Token Management System

This test suite demonstrates the comprehensive functionality of the OAuth2TokenManager
including token generation, validation, refresh, revocation, and security features.
"""

import time

import pytest

from auth.oauth2_tokens import (
    AlgorithmType,
    TokenSecurity,
    TokenStatus,
    TokenType,
    create_oauth2_token_manager,
)


class TestTokenSecurity:
    """Test security utilities."""

    def test_generate_secure_key(self):
        """Test secure key generation."""
        key1 = TokenSecurity.generate_secure_key()
        key2 = TokenSecurity.generate_secure_key()

        assert len(key1) == 32
        assert len(key2) == 32
        assert key1 != key2  # Should be different

    def test_generate_fingerprint(self):
        """Test token fingerprint generation."""
        token = "test_token"
        secret = b"secret_key"

        fp1 = TokenSecurity.generate_fingerprint(token, secret)
        fp2 = TokenSecurity.generate_fingerprint(token, secret)
        fp3 = TokenSecurity.generate_fingerprint("different_token", secret)

        assert fp1 == fp2  # Same inputs should produce same output
        assert fp1 != fp3  # Different inputs should produce different output
        assert len(fp1) == 64  # SHA256 hex digest length

    def test_constant_time_compare(self):
        """Test constant time string comparison."""
        assert TokenSecurity.constant_time_compare("hello", "hello")
        assert not TokenSecurity.constant_time_compare("hello", "world")
        assert not TokenSecurity.constant_time_compare("hello", "hello2")

    def test_derive_key(self):
        """Test key derivation."""
        password = b"password123"
        salt1 = b"salt1234567890ab"
        salt2 = b"salt2345678901bc"

        key1 = TokenSecurity.derive_key(password, salt1)
        key2 = TokenSecurity.derive_key(password, salt1)  # Same inputs
        key3 = TokenSecurity.derive_key(password, salt2)  # Different salt

        assert len(key1) == 32
        assert key1 == key2  # Same inputs should produce same key
        assert key1 != key3  # Different salt should produce different key


class TestOAuth2TokenManagerHS256:
    """Test OAuth2TokenManager with HMAC-SHA256 algorithm."""

    @pytest.fixture
    def token_manager(self):
        """Create token manager for testing."""
        return create_oauth2_token_manager(
            issuer="https://test-issuer.example.com",
            algorithm=AlgorithmType.HS256,
            access_token_expiry=3600,  # 1 hour
            refresh_token_expiry=86400,  # 24 hours for testing
        )

    def test_initialization(self, token_manager):
        """Test token manager initialization."""
        assert token_manager.issuer == "https://test-issuer.example.com"
        assert token_manager.algorithm == AlgorithmType.HS256
        assert token_manager.access_token_expiry == 3600
        assert token_manager.refresh_token_expiry == 86400
        assert token_manager.secret_key is not None
        assert len(token_manager.secret_key) == 32

    def test_generate_token_pair(self, token_manager):
        """Test token pair generation."""
        access_token, refresh_token, metadata = token_manager.generate_token_pair(
            user_id="test_user",
            client_id="test_client",
            audience="https://api.example.com",
            scope="read write",
            mcp_context={"tool": "chat", "version": "1.0"}
        )

        assert isinstance(access_token, str)
        assert isinstance(refresh_token, str)
        assert len(access_token) > 100  # JWT should be reasonably long
        assert len(refresh_token) > 100

        assert metadata["expires_in"] == 3600
        assert metadata["token_type"] == "Bearer"
        assert metadata["scope"] == "read write"
        assert "access_token_id" in metadata
        assert "refresh_token_id" in metadata

    def test_validate_access_token(self, token_manager):
        """Test access token validation."""
        access_token, _, _ = token_manager.generate_token_pair(
            user_id="test_user",
            client_id="test_client",
            audience="https://api.example.com",
            scope="read write"
        )

        # Valid token
        status, claims = token_manager.validate_token(
            access_token,
            expected_audience="https://api.example.com"
        )

        assert status == TokenStatus.VALID
        assert claims is not None
        assert claims["sub"] == "test_user"
        assert claims["client_id"] == "test_client"
        assert claims["aud"] == "https://api.example.com"
        assert claims["scope"] == "read write"
        assert claims["token_type"] == TokenType.ACCESS

    def test_validate_token_with_wrong_audience(self, token_manager):
        """Test token validation with wrong audience."""
        access_token, _, _ = token_manager.generate_token_pair(
            user_id="test_user",
            client_id="test_client",
            audience="https://api.example.com",
            scope="read write"
        )

        # Wrong audience
        status, claims = token_manager.validate_token(
            access_token,
            expected_audience="https://wrong-api.example.com"
        )

        assert status == TokenStatus.INVALID
        assert claims is None

    def test_validate_token_with_required_scope(self, token_manager):
        """Test token validation with required scope."""
        access_token, _, _ = token_manager.generate_token_pair(
            user_id="test_user",
            client_id="test_client",
            audience="https://api.example.com",
            scope="read write admin"
        )

        # Valid scope
        status, claims = token_manager.validate_token(
            access_token,
            required_scope="read write"
        )
        assert status == TokenStatus.VALID

        # Missing scope
        status, claims = token_manager.validate_token(
            access_token,
            required_scope="read write delete"
        )
        assert status == TokenStatus.INVALID

    def test_refresh_token_flow(self, token_manager):
        """Test token refresh flow."""
        # Generate initial token pair
        access_token1, refresh_token1, metadata1 = token_manager.generate_token_pair(
            user_id="test_user",
            client_id="test_client",
            audience="https://api.example.com",
            scope="read write"
        )

        # Wait a moment to ensure different timestamps
        time.sleep(0.1)

        # Refresh tokens
        access_token2, refresh_token2, metadata2 = token_manager.refresh_token(
            refresh_token1
        )

        # New tokens should be different
        assert access_token1 != access_token2
        assert refresh_token1 != refresh_token2

        # Old refresh token should be revoked
        status, _ = token_manager.validate_token(refresh_token1)
        assert status == TokenStatus.REVOKED

        # New tokens should be valid
        status, _ = token_manager.validate_token(access_token2)
        assert status == TokenStatus.VALID

        status, _ = token_manager.validate_token(refresh_token2)
        assert status == TokenStatus.VALID

    def test_refresh_token_with_reduced_scope(self, token_manager):
        """Test token refresh with reduced scope."""
        # Generate initial token pair
        _, refresh_token, _ = token_manager.generate_token_pair(
            user_id="test_user",
            client_id="test_client",
            audience="https://api.example.com",
            scope="read write admin"
        )

        # Refresh with reduced scope
        access_token, _, _ = token_manager.refresh_token(
            refresh_token,
            new_scope="read write"
        )

        # Validate new token has reduced scope
        status, claims = token_manager.validate_token(access_token)
        assert status == TokenStatus.VALID
        assert claims["scope"] == "read write"

    def test_refresh_token_with_invalid_scope(self, token_manager):
        """Test token refresh with invalid expanded scope."""
        # Generate initial token pair
        _, refresh_token, _ = token_manager.generate_token_pair(
            user_id="test_user",
            client_id="test_client",
            audience="https://api.example.com",
            scope="read write"
        )

        # Try to refresh with expanded scope (should fail)
        with pytest.raises(ValueError, match="New scope must be subset"):
            token_manager.refresh_token(
                refresh_token,
                new_scope="read write admin"
            )

    def test_revoke_token(self, token_manager):
        """Test token revocation."""
        access_token, refresh_token, _ = token_manager.generate_token_pair(
            user_id="test_user",
            client_id="test_client",
            audience="https://api.example.com"
        )

        # Tokens should be valid initially
        status, _ = token_manager.validate_token(access_token)
        assert status == TokenStatus.VALID

        # Revoke access token
        result = token_manager.revoke_token(access_token)
        assert result is True

        # Token should now be revoked
        status, _ = token_manager.validate_token(access_token)
        assert status == TokenStatus.REVOKED

        # Refresh token should still be valid
        status, _ = token_manager.validate_token(refresh_token)
        assert status == TokenStatus.VALID

    def test_revoke_user_tokens(self, token_manager):
        """Test revoking all tokens for a user."""
        # Generate multiple token pairs for the same user
        tokens = []
        for i in range(3):
            access_token, refresh_token, _ = token_manager.generate_token_pair(
                user_id="test_user",
                client_id=f"client_{i}",
                audience="https://api.example.com"
            )
            tokens.extend([access_token, refresh_token])

        # All tokens should be valid
        for token in tokens:
            status, _ = token_manager.validate_token(token)
            assert status == TokenStatus.VALID

        # Revoke all tokens for user
        count = token_manager.revoke_user_tokens("test_user")
        assert count == 6  # 3 pairs * 2 tokens each

        # All tokens should now be revoked
        for token in tokens:
            status, _ = token_manager.validate_token(token)
            assert status == TokenStatus.REVOKED

    def test_introspect_token(self, token_manager):
        """Test token introspection."""
        access_token, _, _ = token_manager.generate_token_pair(
            user_id="test_user",
            client_id="test_client",
            audience="https://api.example.com",
            scope="read write"
        )

        # Introspect valid token
        response = token_manager.introspect_token(access_token)

        assert response["active"] is True
        assert response["client_id"] == "test_client"
        assert response["username"] == "test_user"
        assert response["scope"] == "read write"
        assert "exp" in response
        assert "iat" in response
        assert "jti" in response

        # Revoke token and introspect again
        token_manager.revoke_token(access_token)
        response = token_manager.introspect_token(access_token)
        assert response["active"] is False

    def test_invalid_token_introspection(self, token_manager):
        """Test introspection of invalid token."""
        response = token_manager.introspect_token("invalid.token.here")
        assert response["active"] is False
        assert response["token_type"] == "Bearer"
        assert len(response) == 2  # Only active and token_type

    def test_cleanup_expired_tokens(self, token_manager):
        """Test cleanup of expired tokens."""
        # Create token manager with very short expiry
        short_expiry_manager = create_oauth2_token_manager(
            issuer="https://test-issuer.example.com",
            algorithm=AlgorithmType.HS256,
            access_token_expiry=1,  # 1 second
            refresh_token_expiry=1,
        )

        # Generate tokens
        access_token, refresh_token, _ = short_expiry_manager.generate_token_pair(
            user_id="test_user",
            client_id="test_client",
            audience="https://api.example.com"
        )

        # Wait for tokens to expire
        time.sleep(2)

        # Tokens should be expired
        status, _ = short_expiry_manager.validate_token(access_token)
        assert status == TokenStatus.EXPIRED

        # Cleanup expired tokens
        count = short_expiry_manager.cleanup_expired_tokens()
        assert count == 2  # access + refresh token

        # Storage should be cleaned
        assert len(short_expiry_manager.tokens) == 0
        assert len(short_expiry_manager.blacklisted_tokens) == 0

    def test_get_token_statistics(self, token_manager):
        """Test token statistics."""
        initial_stats = token_manager.get_token_statistics()
        assert initial_stats["total_tokens"] == 0
        assert initial_stats["active_tokens"] == 0

        # Generate some tokens
        access_token1, refresh_token1, _ = token_manager.generate_token_pair(
            user_id="user1", client_id="client1", audience="api1"
        )
        access_token2, refresh_token2, _ = token_manager.generate_token_pair(
            user_id="user2", client_id="client2", audience="api2"
        )

        # Check stats
        stats = token_manager.get_token_statistics()
        assert stats["total_tokens"] == 4
        assert stats["active_tokens"] == 4
        assert stats["access_tokens"] == 2
        assert stats["refresh_tokens"] == 2
        assert stats["revoked_tokens"] == 0

        # Revoke one token
        token_manager.revoke_token(access_token1)

        stats = token_manager.get_token_statistics()
        assert stats["revoked_tokens"] == 1
        assert stats["active_tokens"] == 3


class TestOAuth2TokenManagerRS256:
    """Test OAuth2TokenManager with RSA-SHA256 algorithm."""

    @pytest.fixture
    def token_manager(self):
        """Create RSA token manager for testing."""
        return create_oauth2_token_manager(
            issuer="https://test-issuer.example.com",
            algorithm=AlgorithmType.RS256,
        )

    def test_rsa_initialization(self, token_manager):
        """Test RSA token manager initialization."""
        assert token_manager.algorithm == AlgorithmType.RS256
        assert token_manager.rsa_private_key is not None
        assert token_manager.rsa_public_key is not None
        assert token_manager.secret_key is None

    def test_rsa_token_generation_and_validation(self, token_manager):
        """Test RSA token generation and validation."""
        access_token, _, _ = token_manager.generate_token_pair(
            user_id="test_user",
            client_id="test_client",
            audience="https://api.example.com"
        )

        status, claims = token_manager.validate_token(access_token)
        assert status == TokenStatus.VALID
        assert claims["sub"] == "test_user"

    def test_get_jwks(self, token_manager):
        """Test JWKS generation for RSA keys."""
        jwks = token_manager.get_jwks()

        assert "keys" in jwks
        assert len(jwks["keys"]) == 1

        key = jwks["keys"][0]
        assert key["kty"] == "RSA"
        assert key["use"] == "sig"
        assert key["alg"] == "RS256"
        assert "n" in key  # RSA modulus
        assert "e" in key  # RSA exponent
        assert "kid" in key  # Key ID


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limiting(self):
        """Test rate limiting on token operations."""
        # Create token manager with very low rate limit
        token_manager = create_oauth2_token_manager(
            issuer="https://test-issuer.example.com",
            rate_limit_requests=2,
            rate_limit_window=3600,
        )

        # First two requests should succeed
        for _i in range(2):
            access_token, _, _ = token_manager.generate_token_pair(
                user_id="test_user",
                client_id="test_client",
                audience="https://api.example.com"
            )
            assert access_token is not None

        # Third request should fail
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            token_manager.generate_token_pair(
                user_id="test_user",
                client_id="test_client",
                audience="https://api.example.com"
            )


class TestSecurityFeatures:
    """Test security features."""

    @pytest.fixture
    def token_manager(self):
        return create_oauth2_token_manager(
            issuer="https://test-issuer.example.com"
        )

    def test_token_fingerprinting(self, token_manager):
        """Test token fingerprinting security feature."""
        access_token, _, _ = token_manager.generate_token_pair(
            user_id="test_user",
            client_id="test_client",
            audience="https://api.example.com"
        )

        # Validate token (should work with correct fingerprint)
        status, claims = token_manager.validate_token(access_token)
        assert status == TokenStatus.VALID
        assert "fingerprint" in claims

    def test_mcp_context(self, token_manager):
        """Test MCP-specific context in tokens."""
        mcp_context = {
            "tool": "chat",
            "session": "abc123",
            "version": "1.0",
            "features": ["streaming", "memory"]
        }

        access_token, _, _ = token_manager.generate_token_pair(
            user_id="test_user",
            client_id="test_client",
            audience="https://api.example.com",
            mcp_context=mcp_context
        )

        status, claims = token_manager.validate_token(access_token)
        assert status == TokenStatus.VALID
        assert claims["mcp_context"] == mcp_context

    def test_session_tracking(self, token_manager):
        """Test session tracking in tokens."""
        session_id = "session_123"

        access_token, _, _ = token_manager.generate_token_pair(
            user_id="test_user",
            client_id="test_client",
            audience="https://api.example.com",
            session_id=session_id
        )

        status, claims = token_manager.validate_token(access_token)
        assert status == TokenStatus.VALID
        assert claims["session_id"] == session_id


if __name__ == "__main__":
    # Run basic functionality test
    print("Running OAuth2 Token Management System tests...")

    # Test HMAC-SHA256
    print("\n1. Testing HMAC-SHA256 algorithm...")
    manager_hs256 = create_oauth2_token_manager(
        issuer="https://zen-mcp-server.example.com",
        algorithm=AlgorithmType.HS256
    )

    access_token, refresh_token, metadata = manager_hs256.generate_token_pair(
        user_id="demo_user",
        client_id="mcp_client",
        audience="https://api.zen-mcp.com",
        scope="chat analyze codereview",
        mcp_context={
            "tool": "chat",
            "session": "demo_session",
            "capabilities": ["streaming", "memory", "context"]
        }
    )

    print(f"✓ Generated access token: {access_token[:50]}...")
    print(f"✓ Generated refresh token: {refresh_token[:50]}...")

    status, claims = manager_hs256.validate_token(access_token)
    print(f"✓ Token validation: {status}")
    print(f"✓ Claims: {claims['sub']}, {claims['scope']}, {claims['mcp_context']}")

    # Test RSA-SHA256
    print("\n2. Testing RSA-SHA256 algorithm...")
    manager_rs256 = create_oauth2_token_manager(
        issuer="https://zen-mcp-server.example.com",
        algorithm=AlgorithmType.RS256
    )

    access_token_rsa, _, _ = manager_rs256.generate_token_pair(
        user_id="demo_user",
        client_id="mcp_client",
        audience="https://api.zen-mcp.com",
        scope="read write admin"
    )

    print(f"✓ Generated RSA access token: {access_token_rsa[:50]}...")

    jwks = manager_rs256.get_jwks()
    print(f"✓ JWKS generated with {len(jwks['keys'])} keys")

    # Test token refresh
    print("\n3. Testing token refresh...")
    new_access, new_refresh, _ = manager_hs256.refresh_token(refresh_token)
    print(f"✓ Refreshed access token: {new_access[:50]}...")

    # Test introspection
    print("\n4. Testing token introspection...")
    introspection = manager_hs256.introspect_token(new_access)
    print(f"✓ Token active: {introspection['active']}")
    print(f"✓ Token scope: {introspection['scope']}")

    # Test statistics
    print("\n5. Token statistics...")
    stats = manager_hs256.get_token_statistics()
    print(f"✓ Total tokens: {stats['total_tokens']}")
    print(f"✓ Active tokens: {stats['active_tokens']}")
    print(f"✓ Algorithm: {stats['algorithm']}")

    print("\n✅ All tests completed successfully!")
