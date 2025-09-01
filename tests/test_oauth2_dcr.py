#!/usr/bin/env python3
"""
Tests for OAuth 2.0 Dynamic Client Registration (DCR) Implementation

Tests RFC 7591 compliance, security features, and integration with Zen MCP Server.
"""

import os
import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from auth.oauth2_dcr import (
    ClientAuthMethod,
    ClientMetadata,
    ClientStore,
    ClientType,
    DCRError,
    DCRManager,
    GrantType,
    RegisteredClient,
    ResponseType,
    get_dcr_manager,
)


class TestClientMetadata:
    """Test client metadata validation and serialization."""

    def test_minimal_valid_metadata(self):
        """Test minimal valid client metadata."""
        metadata = ClientMetadata(
            client_name="Test Client",
            redirect_uris=["https://example.com/callback"]
        )

        assert metadata.client_name == "Test Client"
        assert metadata.client_type == ClientType.CONFIDENTIAL
        assert metadata.token_endpoint_auth_method == ClientAuthMethod.CLIENT_SECRET_BASIC
        assert GrantType.AUTHORIZATION_CODE in metadata.grant_types
        assert ResponseType.CODE in metadata.response_types

    def test_public_client_metadata(self):
        """Test public client metadata validation."""
        metadata = ClientMetadata(
            client_name="Public Client",
            client_type=ClientType.PUBLIC,
            token_endpoint_auth_method=ClientAuthMethod.NONE,
            grant_types=[GrantType.AUTHORIZATION_CODE]
        )

        assert metadata.client_type == ClientType.PUBLIC
        assert metadata.token_endpoint_auth_method == ClientAuthMethod.NONE

    def test_redirect_uri_validation(self):
        """Test redirect URI validation."""
        # Valid HTTPS URI
        metadata = ClientMetadata(
            client_name="Test Client",
            redirect_uris=["https://example.com/callback"]
        )
        assert metadata.redirect_uris is not None

        # Custom scheme (mobile app)
        metadata = ClientMetadata(
            client_name="Mobile Client",
            redirect_uris=["com.example.app://oauth/callback"]
        )
        assert metadata.redirect_uris is not None

    def test_scope_validation(self):
        """Test scope format validation."""
        # Valid scopes
        metadata = ClientMetadata(
            client_name="Test Client",
            scope="read write mcp:tools mcp:resources"
        )
        assert metadata.scope == "read write mcp:tools mcp:resources"

        # Invalid scope should raise validation error
        with pytest.raises(ValueError):
            ClientMetadata(
                client_name="Test Client",
                scope="invalid@scope with spaces"
            )

    def test_mcp_extensions(self):
        """Test MCP-specific extensions."""
        metadata = ClientMetadata(
            client_name="MCP Client",
            mcp_capabilities=["tools", "resources", "prompts"],
            mcp_transport_protocols=["http", "websocket"],
            mcp_session_timeout=3600
        )

        assert "tools" in metadata.mcp_capabilities
        assert "http" in metadata.mcp_transport_protocols
        assert metadata.mcp_session_timeout == 3600


class TestClientStore:
    """Test client storage functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.client_store = ClientStore()

    @pytest.mark.asyncio
    async def test_store_and_retrieve_client(self):
        """Test storing and retrieving a client."""
        client = RegisteredClient(
            client_id="test_client_123",
            client_secret="secret_456",
            client_id_issued_at=int(time.time()),
            client_name="Test Client",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        # Store client
        await self.client_store.store_client(client, ttl_seconds=3600)

        # Retrieve client
        retrieved = await self.client_store.get_client("test_client_123")

        assert retrieved is not None
        assert retrieved.client_id == "test_client_123"
        assert retrieved.client_secret == "secret_456"
        assert retrieved.client_name == "Test Client"

    @pytest.mark.asyncio
    async def test_update_client(self):
        """Test updating an existing client."""
        # Create and store initial client
        client = RegisteredClient(
            client_id="update_test",
            client_name="Original Name",
            client_id_issued_at=int(time.time()),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        await self.client_store.store_client(client)

        # Update client
        client.client_name = "Updated Name"
        client.updated_at = datetime.now(timezone.utc)

        success = await self.client_store.update_client(client)
        assert success

        # Verify update
        retrieved = await self.client_store.get_client("update_test")
        assert retrieved.client_name == "Updated Name"

    @pytest.mark.asyncio
    async def test_delete_client(self):
        """Test deleting a client."""
        # Store client
        client = RegisteredClient(
            client_id="delete_test",
            client_id_issued_at=int(time.time()),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        await self.client_store.store_client(client)

        # Delete client
        success = await self.client_store.delete_client("delete_test")
        assert success

        # Verify deletion
        retrieved = await self.client_store.get_client("delete_test")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_list_clients(self):
        """Test listing all clients."""
        # Store multiple clients
        for i in range(3):
            client = RegisteredClient(
                client_id=f"list_test_{i}",
                client_id_issued_at=int(time.time()),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            await self.client_store.store_client(client)

        # List clients
        client_ids = await self.client_store.list_clients()

        assert len([cid for cid in client_ids if cid.startswith("list_test_")]) >= 3

    def test_encryption(self):
        """Test client data encryption."""
        encrypted_store = ClientStore(encryption_key="test_encryption_key")

        assert encrypted_store.encryption_enabled
        assert encrypted_store.cipher is not None

        # Test encryption/decryption
        test_data = "sensitive client data"
        encrypted = encrypted_store._encrypt_data(test_data)
        decrypted = encrypted_store._decrypt_data(encrypted)

        assert encrypted != test_data
        assert decrypted == test_data


class TestDCRManager:
    """Test Dynamic Client Registration Manager."""

    def setup_method(self):
        """Setup test environment."""
        self.dcr_manager = DCRManager(rate_limit_per_ip=100)  # High limit for testing

    @pytest.mark.asyncio
    async def test_register_confidential_client(self):
        """Test registering a confidential client."""
        metadata = ClientMetadata(
            client_name="Confidential Test Client",
            redirect_uris=["https://example.com/callback"],
            client_type=ClientType.CONFIDENTIAL,
            grant_types=[GrantType.AUTHORIZATION_CODE, GrantType.REFRESH_TOKEN],
            scope="read write"
        )

        registered_client = await self.dcr_manager.register_client(
            metadata=metadata,
            client_ip="192.168.1.100",
            user_agent="Test User Agent"
        )

        assert registered_client.client_id.startswith("zen_mcp_")
        assert registered_client.client_secret is not None
        assert registered_client.client_secret_expires_at is not None
        assert registered_client.registration_access_token is not None
        assert registered_client.client_name == "Confidential Test Client"
        assert registered_client.client_type == ClientType.CONFIDENTIAL
        assert GrantType.AUTHORIZATION_CODE in registered_client.grant_types

    @pytest.mark.asyncio
    async def test_register_public_client(self):
        """Test registering a public client."""
        metadata = ClientMetadata(
            client_name="Public Test Client",
            client_type=ClientType.PUBLIC,
            token_endpoint_auth_method=ClientAuthMethod.NONE,
            grant_types=[GrantType.AUTHORIZATION_CODE]
        )

        registered_client = await self.dcr_manager.register_client(
            metadata=metadata,
            client_ip="192.168.1.101"
        )

        assert registered_client.client_id.startswith("zen_mcp_")
        assert registered_client.client_secret is None
        assert registered_client.client_secret_expires_at is None
        assert registered_client.client_type == ClientType.PUBLIC
        assert registered_client.token_endpoint_auth_method == ClientAuthMethod.NONE

    @pytest.mark.asyncio
    async def test_register_mcp_client(self):
        """Test registering a client with MCP extensions."""
        metadata = ClientMetadata(
            client_name="MCP Client",
            redirect_uris=["https://mcp-client.example.com/callback"],
            mcp_capabilities=["tools", "resources"],
            mcp_transport_protocols=["http"],
            mcp_session_timeout=7200,
            scope="mcp:tools mcp:resources"
        )

        registered_client = await self.dcr_manager.register_client(
            metadata=metadata,
            client_ip="192.168.1.102"
        )

        assert "tools" in registered_client.mcp_capabilities
        assert "http" in registered_client.mcp_transport_protocols
        assert registered_client.mcp_session_timeout == 7200

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test client registration rate limiting."""
        # Set very low rate limit
        rate_limited_manager = DCRManager(rate_limit_per_ip=1, rate_limit_window=3600)

        metadata = ClientMetadata(client_name="Rate Test")

        # First registration should succeed
        await rate_limited_manager.register_client(metadata, "192.168.1.200")

        # Second registration from same IP should fail
        with pytest.raises(DCRError) as exc_info:
            await rate_limited_manager.register_client(metadata, "192.168.1.200")

        assert exc_info.value.error_code == "too_many_requests"
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_get_client(self):
        """Test retrieving client information."""
        # Register a client first
        metadata = ClientMetadata(client_name="Get Test Client")
        registered = await self.dcr_manager.register_client(metadata, "192.168.1.103")

        # Retrieve the client
        retrieved = await self.dcr_manager.get_client(registered.client_id)

        assert retrieved is not None
        assert retrieved.client_id == registered.client_id
        assert retrieved.client_name == "Get Test Client"

    @pytest.mark.asyncio
    async def test_update_client(self):
        """Test updating client metadata."""
        # Register a client
        metadata = ClientMetadata(
            client_name="Original Name",
            redirect_uris=["https://original.example.com/callback"]
        )
        registered = await self.dcr_manager.register_client(metadata, "192.168.1.104")

        # Update client metadata
        updated_metadata = ClientMetadata(
            client_name="Updated Name",
            redirect_uris=["https://updated.example.com/callback"],
            scope="read write admin"
        )

        updated_client = await self.dcr_manager.update_client(
            client_id=registered.client_id,
            metadata=updated_metadata,
            registration_token=registered.registration_access_token,
            client_ip="192.168.1.104"
        )

        assert updated_client.client_name == "Updated Name"
        assert "https://updated.example.com/callback" in updated_client.redirect_uris
        assert updated_client.scope == "read write admin"

    @pytest.mark.asyncio
    async def test_delete_client(self):
        """Test deleting a registered client."""
        # Register a client
        metadata = ClientMetadata(client_name="Delete Test Client")
        registered = await self.dcr_manager.register_client(metadata, "192.168.1.105")

        # Delete the client
        success = await self.dcr_manager.delete_client(
            client_id=registered.client_id,
            registration_token=registered.registration_access_token,
            client_ip="192.168.1.105"
        )

        assert success

        # Verify client is deleted
        retrieved = await self.dcr_manager.get_client(registered.client_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_authenticate_client_secret_basic(self):
        """Test client authentication with client_secret_basic."""
        metadata = ClientMetadata(
            client_name="Auth Test Client",
            token_endpoint_auth_method=ClientAuthMethod.CLIENT_SECRET_BASIC
        )
        registered = await self.dcr_manager.register_client(metadata, "192.168.1.106")

        # Test valid authentication
        is_valid = await self.dcr_manager.authenticate_client(
            client_id=registered.client_id,
            client_secret=registered.client_secret
        )
        assert is_valid

        # Test invalid secret
        is_valid = await self.dcr_manager.authenticate_client(
            client_id=registered.client_id,
            client_secret="wrong_secret"
        )
        assert not is_valid

    @pytest.mark.asyncio
    async def test_authenticate_public_client(self):
        """Test public client authentication (no secret)."""
        metadata = ClientMetadata(
            client_name="Public Auth Test",
            client_type=ClientType.PUBLIC,
            token_endpoint_auth_method=ClientAuthMethod.NONE
        )
        registered = await self.dcr_manager.register_client(metadata, "192.168.1.107")

        # Public client should authenticate without secret
        is_valid = await self.dcr_manager.authenticate_client(
            client_id=registered.client_id
        )
        assert is_valid

    @pytest.mark.asyncio
    async def test_list_clients(self):
        """Test listing all registered clients."""
        # Register multiple clients
        for i in range(3):
            metadata = ClientMetadata(client_name=f"List Test Client {i}")
            await self.dcr_manager.register_client(metadata, f"192.168.1.{150 + i}")

        # List clients
        clients = await self.dcr_manager.list_clients(limit=10)

        assert len(clients) >= 3
        assert all(isinstance(client, RegisteredClient) for client in clients)

    def test_registration_token_generation_and_verification(self):
        """Test registration access token generation and verification."""
        client_id = "test_client_12345"

        # Generate token
        token = self.dcr_manager._generate_registration_token(client_id)

        assert token is not None
        assert len(token) > 20

        # Verify valid token
        is_valid = self.dcr_manager._verify_registration_token(token, client_id)
        assert is_valid

        # Verify invalid token
        is_valid = self.dcr_manager._verify_registration_token("invalid_token", client_id)
        assert not is_valid

        # Verify token with wrong client ID
        is_valid = self.dcr_manager._verify_registration_token(token, "wrong_client_id")
        assert not is_valid

    @pytest.mark.asyncio
    async def test_invalid_metadata_validation(self):
        """Test validation of invalid client metadata."""
        # Test unsupported grant type (will be caught by Pydantic validation)
        with pytest.raises(ValueError):
            ClientMetadata(
                client_name="Invalid Client",
                grant_types=["unsupported_grant_type"]
            )

        # Test public client with secret auth method
        with pytest.raises(DCRError) as exc_info:
            invalid_metadata = ClientMetadata(
                client_name="Invalid Public Client",
                client_type=ClientType.PUBLIC,
                token_endpoint_auth_method=ClientAuthMethod.CLIENT_SECRET_BASIC
            )
            await self.dcr_manager.register_client(invalid_metadata, "192.168.1.200")

        assert exc_info.value.error_code == "invalid_client_metadata"

    def test_client_id_generation_uniqueness(self):
        """Test client ID generation produces unique values."""
        client_ids = set()

        for _ in range(100):
            client_id = self.dcr_manager._generate_client_id()
            assert client_id not in client_ids, "Generated duplicate client ID"
            client_ids.add(client_id)
            assert client_id.startswith("zen_mcp_")

    def test_client_secret_generation_security(self):
        """Test client secret generation security properties."""
        secrets = set()

        for _ in range(100):
            secret = self.dcr_manager._generate_client_secret()
            assert secret not in secrets, "Generated duplicate client secret"
            secrets.add(secret)
            assert len(secret) >= 40, "Client secret too short"

    def test_get_registration_metadata(self):
        """Test DCR endpoint metadata for discovery."""
        metadata = self.dcr_manager.get_registration_metadata()

        assert metadata["registration_endpoint"] == "/oauth/register"
        assert "authorization_code" in metadata["grant_types_supported"]
        assert "code" in metadata["response_types_supported"]
        assert "client_secret_basic" in metadata["token_endpoint_auth_methods_supported"]
        assert metadata["client_id_issued_at_supported"] is True
        assert metadata["client_secret_expires_at_supported"] is True
        assert "mcp_capabilities" in metadata["mcp_extensions_supported"]


class TestDCRIntegration:
    """Test DCR integration with Zen MCP Server infrastructure."""

    def test_get_dcr_manager_singleton(self):
        """Test DCR manager singleton pattern."""
        manager1 = get_dcr_manager()
        manager2 = get_dcr_manager()

        assert manager1 is manager2
        assert isinstance(manager1, DCRManager)

    @patch.dict(os.environ, {
        'OAUTH_ENCRYPTION_KEY': 'test_encryption_key',
        'OAUTH_RATE_LIMIT_PER_IP': '5',
        'OAUTH_RATE_LIMIT_WINDOW': '1800'
    })
    def test_dcr_manager_configuration(self):
        """Test DCR manager configuration from environment."""
        # Clear singleton to test configuration
        import auth.oauth2_dcr
        auth.oauth2_dcr._dcr_manager = None

        manager = get_dcr_manager()

        assert manager.rate_limit_per_ip == 5
        assert manager.rate_limit_window == 1800
        assert manager.client_store.encryption_enabled


@pytest.mark.asyncio
class TestDCRErrorHandling:
    """Test error handling in DCR operations."""

    def setup_method(self):
        self.dcr_manager = DCRManager()

    async def test_invalid_client_id_error(self):
        """Test error when accessing non-existent client."""
        with pytest.raises(DCRError) as exc_info:
            await self.dcr_manager.update_client(
                client_id="non_existent_client",
                metadata=ClientMetadata(client_name="Test"),
                registration_token="invalid_token",
                client_ip="192.168.1.1"
            )

        assert exc_info.value.error_code == "invalid_token"
        assert exc_info.value.status_code == 401

    async def test_invalid_token_error(self):
        """Test error with invalid registration token."""
        # Register a client first
        metadata = ClientMetadata(client_name="Token Test")
        registered = await self.dcr_manager.register_client(metadata, "192.168.1.1")

        # Try to update with invalid token
        with pytest.raises(DCRError) as exc_info:
            await self.dcr_manager.update_client(
                client_id=registered.client_id,
                metadata=metadata,
                registration_token="invalid_token",
                client_ip="192.168.1.1"
            )

        assert exc_info.value.error_code == "invalid_token"

    def test_dcr_error_properties(self):
        """Test DCRError exception properties."""
        error = DCRError("invalid_client_metadata", "Invalid metadata provided", 400)

        assert error.error_code == "invalid_client_metadata"
        assert error.error_description == "Invalid metadata provided"
        assert error.status_code == 400
        assert "invalid_client_metadata" in str(error)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
