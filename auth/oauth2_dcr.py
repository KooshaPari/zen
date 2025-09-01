#!/usr/bin/env python3
"""
Dynamic Client Registration (DCR) for OAuth 2.0 Clients - RFC 7591

This module implements RFC 7591 compliant Dynamic Client Registration for OAuth 2.0 clients,
providing secure client registration and management capabilities for the Zen MCP Server.

Features:
- RFC 7591 compliant Dynamic Client Registration
- Client registration endpoint (/oauth/register)
- Client management endpoints (GET/PUT/DELETE /oauth/register/{client_id})
- Automatic client_id and client_secret generation
- Client metadata validation and storage
- Support for various client types (public, confidential)
- Rate limiting on registration attempts
- Client authentication for management operations
- Secure credential generation and storage
- Audit logging for all client operations
- Integration with MCP client registration patterns

Security Features:
- Secure cryptographic client credential generation
- Client metadata validation and sanitization
- Registration rate limiting with IP-based tracking
- Comprehensive audit logging for compliance
- Support for client authentication methods
- Token-based client management protection
- Secure storage with optional encryption

Integration:
- Works with OAuth 2.0 server for client validation
- Integrates with existing storage backend system
- Supports MCP client registration patterns
- Compatible with existing audit trail system
- Rate limiting using existing infrastructure
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pydantic import BaseModel, Field, HttpUrl, validator

# Import existing Zen MCP Server infrastructure
from utils.audit_trail import (
    AuditEventCategory,
    AuditSeverity,
    create_audit_log,
)
from utils.kafka_events import EventType
from utils.ratelimit import allow
from utils.storage_backend import get_storage_backend

logger = logging.getLogger(__name__)


class ClientType(str, Enum):
    """OAuth 2.0 client types as defined in RFC 6749."""

    CONFIDENTIAL = "confidential"
    PUBLIC = "public"


class ClientAuthMethod(str, Enum):
    """Client authentication methods as defined in RFC 7591."""

    CLIENT_SECRET_POST = "client_secret_post"
    CLIENT_SECRET_BASIC = "client_secret_basic"
    CLIENT_SECRET_JWT = "client_secret_jwt"
    PRIVATE_KEY_JWT = "private_key_jwt"
    NONE = "none"  # For public clients


class GrantType(str, Enum):
    """OAuth 2.0 grant types."""

    AUTHORIZATION_CODE = "authorization_code"
    IMPLICIT = "implicit"
    PASSWORD = "password"
    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"
    JWT_BEARER = "urn:ietf:params:oauth:grant-type:jwt-bearer"
    SAML2_BEARER = "urn:ietf:params:oauth:grant-type:saml2-bearer"


class ResponseType(str, Enum):
    """OAuth 2.0 response types."""

    CODE = "code"
    TOKEN = "token"
    ID_TOKEN = "id_token"


class DCRError(Exception):
    """Base exception for DCR operations."""

    def __init__(self, error_code: str, error_description: str, status_code: int = 400):
        self.error_code = error_code
        self.error_description = error_description
        self.status_code = status_code
        super().__init__(f"{error_code}: {error_description}")


class ClientMetadata(BaseModel):
    """Client metadata as defined in RFC 7591 and RFC 7592."""

    # Required for registration
    client_name: Optional[str] = Field(None, max_length=255)
    client_uri: Optional[HttpUrl] = None
    # Allow custom schemes (e.g., mobile apps) by accepting strings and
    # validating schemes manually in a validator.
    redirect_uris: Optional[list[str]] = Field(None, min_items=1)

    # OAuth 2.0 parameters
    client_type: ClientType = ClientType.CONFIDENTIAL
    token_endpoint_auth_method: ClientAuthMethod = ClientAuthMethod.CLIENT_SECRET_BASIC
    grant_types: list[GrantType] = Field(default_factory=lambda: [GrantType.AUTHORIZATION_CODE])
    response_types: list[ResponseType] = Field(default_factory=lambda: [ResponseType.CODE])
    scope: Optional[str] = Field(None, max_length=1000)

    # Additional metadata
    logo_uri: Optional[HttpUrl] = None
    tos_uri: Optional[HttpUrl] = None
    policy_uri: Optional[HttpUrl] = None
    software_id: Optional[str] = Field(None, max_length=255)
    software_version: Optional[str] = Field(None, max_length=100)
    software_statement: Optional[str] = None  # JWT for software statement

    # Contact information
    contacts: Optional[list[str]] = Field(None, max_items=10)

    # Security attributes
    jwks_uri: Optional[HttpUrl] = None
    jwks: Optional[dict[str, Any]] = None
    sector_identifier_uri: Optional[HttpUrl] = None
    subject_type: Optional[str] = Field(None, pattern="^(public|pairwise)$")
    id_token_signed_response_alg: Optional[str] = "RS256"
    id_token_encrypted_response_alg: Optional[str] = None
    id_token_encrypted_response_enc: Optional[str] = None
    userinfo_signed_response_alg: Optional[str] = None
    userinfo_encrypted_response_alg: Optional[str] = None
    userinfo_encrypted_response_enc: Optional[str] = None
    request_object_signing_alg: Optional[str] = None
    request_object_encryption_alg: Optional[str] = None
    request_object_encryption_enc: Optional[str] = None

    # Client lifetime
    client_id_issued_at: Optional[int] = None
    client_secret_expires_at: Optional[int] = None

    # MCP-specific extensions
    mcp_capabilities: Optional[list[str]] = Field(None, max_items=50)
    mcp_transport_protocols: Optional[list[str]] = Field(None, max_items=10)
    mcp_session_timeout: Optional[int] = Field(None, ge=300, le=86400)  # 5 min to 24 hours

    @validator('redirect_uris')
    def validate_redirect_uris(cls, v):
        if v:
            for uri in v:
                parsed = urlparse(str(uri))
                # Validate URI security: allow http/https, and common app schemes
                if parsed.scheme not in ['http', 'https']:
                    # Permit custom app schemes like com.example.app://callback or app.example://
                    if not (parsed.scheme.startswith('com.') or parsed.scheme.startswith('app.')):
                        raise ValueError(f"Invalid redirect URI scheme: {parsed.scheme}")
                # Prevent localhost for production clients (configurable)
                if os.getenv('OAUTH_ALLOW_LOCALHOST', 'true').lower() != 'true':
                    if parsed.hostname in ['localhost', '127.0.0.1', '::1']:
                        raise ValueError("Localhost redirect URIs not allowed in production")
        return v

    @validator('grant_types')
    def validate_grant_types(cls, v):
        if v:
            # Validate grant type combinations
            if GrantType.IMPLICIT in v and GrantType.AUTHORIZATION_CODE in v:
                # Mixed flows - ensure proper response types
                pass
            if GrantType.CLIENT_CREDENTIALS in v and len(v) > 1:
                # Client credentials should typically be used alone
                logger.warning("Client credentials grant combined with other grant types")
        return v

    @validator('scope')
    def validate_scope(cls, v):
        if v:
            # Validate scope format and content
            scopes = v.split()
            for scope in scopes:
                if not scope.replace('-', '').replace('_', '').replace(':', '').replace('.', '').isalnum():
                    raise ValueError(f"Invalid scope format: {scope}")
            # Limit total scopes
            if len(scopes) > 100:
                raise ValueError("Too many scopes requested (max 100)")
        return v


class RegisteredClient(BaseModel):
    """Registered OAuth 2.0 client information."""

    # Client identification
    client_id: str
    client_secret: Optional[str] = None
    client_secret_expires_at: Optional[int] = None

    # Registration metadata
    client_id_issued_at: int
    registration_access_token: Optional[str] = None
    registration_client_uri: Optional[str] = None

    # Client metadata (all fields from ClientMetadata)
    client_name: Optional[str] = None
    client_uri: Optional[str] = None
    redirect_uris: Optional[list[str]] = None
    client_type: ClientType = ClientType.CONFIDENTIAL
    token_endpoint_auth_method: ClientAuthMethod = ClientAuthMethod.CLIENT_SECRET_BASIC
    grant_types: list[GrantType] = Field(default_factory=lambda: [GrantType.AUTHORIZATION_CODE])
    response_types: list[ResponseType] = Field(default_factory=lambda: [ResponseType.CODE])
    scope: Optional[str] = None
    logo_uri: Optional[str] = None
    tos_uri: Optional[str] = None
    policy_uri: Optional[str] = None
    software_id: Optional[str] = None
    software_version: Optional[str] = None
    software_statement: Optional[str] = None
    contacts: Optional[list[str]] = None
    jwks_uri: Optional[str] = None
    jwks: Optional[dict[str, Any]] = None
    sector_identifier_uri: Optional[str] = None
    subject_type: Optional[str] = None
    id_token_signed_response_alg: Optional[str] = "RS256"
    mcp_capabilities: Optional[list[str]] = None
    mcp_transport_protocols: Optional[list[str]] = None
    mcp_session_timeout: Optional[int] = None

    # Internal metadata
    created_at: datetime
    updated_at: datetime
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    rate_limit_count: int = 0

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class ClientStore:
    """Secure storage for registered OAuth 2.0 clients."""

    def __init__(self, encryption_key: Optional[str] = None):
        self.storage = get_storage_backend()
        self.encryption_enabled = bool(encryption_key)

        if self.encryption_enabled:
            # Derive encryption key
            salt = b"oauth2_dcr_client_store_salt"
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(encryption_key.encode()))
            self.cipher = Fernet(key)
        else:
            self.cipher = None

    def _encrypt_data(self, data: str) -> str:
        """Encrypt client data if encryption is enabled."""
        if self.cipher:
            return self.cipher.encrypt(data.encode()).decode()
        return data

    def _decrypt_data(self, data: str) -> str:
        """Decrypt client data if encryption is enabled."""
        if self.cipher:
            return self.cipher.decrypt(data.encode()).decode()
        return data

    async def store_client(self, client: RegisteredClient, ttl_seconds: int = 31536000) -> None:
        """Store a registered client with optional encryption."""
        try:
            client_data = client.json()
            encrypted_data = self._encrypt_data(client_data)

            key = f"oauth2:client:{client.client_id}"
            self.storage.setex(key, ttl_seconds, encrypted_data)

            # Store client ID in index for enumeration
            index_key = "oauth2:clients:index"
            existing_index = self.storage.get(index_key) or "[]"
            try:
                client_ids = json.loads(self._decrypt_data(existing_index))
            except (json.JSONDecodeError, ValueError):
                client_ids = []

            if client.client_id not in client_ids:
                client_ids.append(client.client_id)
                updated_index = json.dumps(client_ids)
                encrypted_index = self._encrypt_data(updated_index)
                self.storage.setex(index_key, ttl_seconds, encrypted_index)

            logger.info(f"Stored OAuth2 client: {client.client_id}")

        except Exception as e:
            logger.error(f"Failed to store client {client.client_id}: {e}")
            raise DCRError("server_error", "Failed to store client registration")

    async def get_client(self, client_id: str) -> Optional[RegisteredClient]:
        """Retrieve a registered client by client_id."""
        try:
            key = f"oauth2:client:{client_id}"
            encrypted_data = self.storage.get(key)

            if not encrypted_data:
                return None

            client_data = self._decrypt_data(encrypted_data)
            return RegisteredClient.parse_raw(client_data)

        except Exception as e:
            logger.error(f"Failed to retrieve client {client_id}: {e}")
            return None

    async def update_client(self, client: RegisteredClient) -> bool:
        """Update an existing registered client."""
        try:
            existing = await self.get_client(client.client_id)
            if not existing:
                return False

            client.updated_at = datetime.now(timezone.utc)
            await self.store_client(client)
            return True

        except Exception as e:
            logger.error(f"Failed to update client {client.client_id}: {e}")
            return False

    async def delete_client(self, client_id: str) -> bool:
        """Delete a registered client."""
        try:
            # Remove from main storage
            key = f"oauth2:client:{client_id}"
            # Note: Redis storage doesn't have a delete method in our interface
            # We'll set an expired entry to effectively delete it
            self.storage.setex(key, 1, "")

            # Remove from index
            index_key = "oauth2:clients:index"
            existing_index = self.storage.get(index_key) or "[]"
            try:
                client_ids = json.loads(self._decrypt_data(existing_index))
                if client_id in client_ids:
                    client_ids.remove(client_id)
                    updated_index = json.dumps(client_ids)
                    encrypted_index = self._encrypt_data(updated_index)
                    self.storage.setex(index_key, 31536000, encrypted_index)  # 1 year TTL
            except (json.JSONDecodeError, ValueError):
                pass

            logger.info(f"Deleted OAuth2 client: {client_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete client {client_id}: {e}")
            return False

    async def list_clients(self, limit: int = 100) -> list[str]:
        """List all registered client IDs."""
        try:
            index_key = "oauth2:clients:index"
            encrypted_index = self.storage.get(index_key) or "[]"
            client_ids = json.loads(self._decrypt_data(encrypted_index))
            return client_ids[:limit]
        except Exception as e:
            logger.error(f"Failed to list clients: {e}")
            return []


class DCRManager:
    """Dynamic Client Registration Manager implementing RFC 7591."""

    def __init__(self,
                 encryption_key: Optional[str] = None,
                 rate_limit_per_ip: int = 10,
                 rate_limit_window: int = 3600):

        # Initialize client store
        self.client_store = ClientStore(encryption_key)

        # Rate limiting configuration
        self.rate_limit_per_ip = rate_limit_per_ip
        self.rate_limit_window = rate_limit_window

        # Supported features
        self.supported_grant_types = [
            GrantType.AUTHORIZATION_CODE,
            GrantType.CLIENT_CREDENTIALS,
            GrantType.REFRESH_TOKEN,
        ]

        self.supported_response_types = [
            ResponseType.CODE,
        ]

        self.supported_auth_methods = [
            ClientAuthMethod.CLIENT_SECRET_POST,
            ClientAuthMethod.CLIENT_SECRET_BASIC,
            ClientAuthMethod.NONE,  # For public clients
        ]

        logger.info("DCR Manager initialized with secure client storage")

    def _generate_client_id(self) -> str:
        """Generate a secure, unique client ID."""
        # Use timestamp + random for uniqueness and ordering
        timestamp = str(int(time.time()))
        random_part = secrets.token_hex(16)
        return f"zen_mcp_{timestamp}_{random_part}"

    def _generate_client_secret(self) -> str:
        """Generate a cryptographically secure client secret."""
        # Generate 256-bit secret (44 characters in base64url)
        secret_bytes = secrets.token_bytes(32)
        return base64.urlsafe_b64encode(secret_bytes).decode().rstrip('=')

    def _generate_registration_token(self, client_id: str) -> str:
        """Generate a registration access token for client management."""
        # Create HMAC-signed token
        timestamp = str(int(time.time()))
        message = f"{client_id}:{timestamp}"
        secret_key = os.getenv('OAUTH_REGISTRATION_SECRET', 'default-secret-key')
        signature = hmac.new(
            secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        token_data = f"{message}:{signature}"
        return base64.urlsafe_b64encode(token_data.encode()).decode().rstrip('=')

    def _verify_registration_token(self, token: str, client_id: str) -> bool:
        """Verify a registration access token."""
        try:
            # Decode and verify token
            token_data = base64.urlsafe_b64decode(token.encode() + b'==').decode()
            parts = token_data.split(':')

            if len(parts) != 3:
                return False

            token_client_id, timestamp, signature = parts

            if token_client_id != client_id:
                return False

            # Verify signature
            message = f"{token_client_id}:{timestamp}"
            secret_key = os.getenv('OAUTH_REGISTRATION_SECRET', 'default-secret-key')
            expected_signature = hmac.new(
                secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                return False

            # Check token age (valid for 24 hours)
            token_age = int(time.time()) - int(timestamp)
            return token_age < 86400

        except Exception:
            return False

    def _validate_client_metadata(self, metadata: ClientMetadata) -> None:
        """Validate client metadata according to RFC 7591."""

        # Validate grant types
        for grant_type in metadata.grant_types:
            if grant_type not in self.supported_grant_types:
                raise DCRError(
                    "invalid_client_metadata",
                    f"Unsupported grant type: {grant_type}"
                )

        # Validate response types
        for response_type in metadata.response_types:
            if response_type not in self.supported_response_types:
                raise DCRError(
                    "invalid_client_metadata",
                    f"Unsupported response type: {response_type}"
                )

        # Validate authentication method
        if metadata.token_endpoint_auth_method not in self.supported_auth_methods:
            raise DCRError(
                "invalid_client_metadata",
                f"Unsupported authentication method: {metadata.token_endpoint_auth_method}"
            )

        # Validate client type vs auth method consistency
        if metadata.client_type == ClientType.PUBLIC:
            if metadata.token_endpoint_auth_method != ClientAuthMethod.NONE:
                raise DCRError(
                    "invalid_client_metadata",
                    "Public clients must use 'none' authentication method"
                )

        # Redirect URIs are optional at registration to simplify onboarding,
        # including for authorization code flows. They will be enforced during
        # authorization if required by the flow.

        # Validate scope format
        if metadata.scope:
            # Basic scope format checks are handled in the Pydantic validator above.
            # No forbidden scopes are enforced at registration time in this implementation.
            pass

        # Validate MCP-specific parameters
        if metadata.mcp_capabilities:
            allowed_capabilities = {
                'tools', 'resources', 'prompts', 'logging', 'sampling'
            }  # Configurable
            for capability in metadata.mcp_capabilities:
                if capability not in allowed_capabilities:
                    raise DCRError(
                        "invalid_client_metadata",
                        f"Unsupported MCP capability: {capability}"
                    )

        if metadata.mcp_transport_protocols:
            allowed_protocols = {'stdio', 'http', 'websocket'}  # Configurable
            for protocol in metadata.mcp_transport_protocols:
                if protocol not in allowed_protocols:
                    raise DCRError(
                        "invalid_client_metadata",
                        f"Unsupported MCP transport protocol: {protocol}"
                    )

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client IP is within rate limits."""
        return allow(
            scope=f"dcr_registration:{client_ip}",
            max_per_window=self.rate_limit_per_ip,
            window_seconds=self.rate_limit_window
        )

    async def register_client(self,
                            metadata: ClientMetadata,
                            client_ip: str,
                            user_agent: Optional[str] = None) -> RegisteredClient:
        """Register a new OAuth 2.0 client."""

        # Rate limiting
        if not self._check_rate_limit(client_ip):
            await create_audit_log(
                event_type=EventType.AGENT_REGISTERED,
                category=AuditEventCategory.AUTHENTICATION,
                action="client_registration_rate_limited",
                description=f"Rate limit exceeded for IP: {client_ip}",
                severity=AuditSeverity.MEDIUM,
                source_ip=client_ip,
                user_agent=user_agent,
                outcome="failure"
            )
            raise DCRError(
                "too_many_requests",
                "Rate limit exceeded for client registration",
                429
            )

        # Validate metadata
        try:
            self._validate_client_metadata(metadata)
        except DCRError:
            await create_audit_log(
                event_type=EventType.AGENT_REGISTERED,
                category=AuditEventCategory.AUTHENTICATION,
                action="client_registration_validation_failed",
                description="Client metadata validation failed",
                severity=AuditSeverity.LOW,
                source_ip=client_ip,
                user_agent=user_agent,
                outcome="failure",
                payload=metadata.dict()
            )
            raise

        # Generate client credentials
        client_id = self._generate_client_id()
        client_secret = None
        client_secret_expires_at = None

        if metadata.client_type == ClientType.CONFIDENTIAL:
            client_secret = self._generate_client_secret()
            # Set secret expiration (configurable, default 1 year)
            secret_lifetime = int(os.getenv('OAUTH_CLIENT_SECRET_LIFETIME', '31536000'))
            client_secret_expires_at = int(time.time()) + secret_lifetime

        # Generate registration access token
        registration_token = self._generate_registration_token(client_id)
        registration_uri = f"/oauth/register/{client_id}"

        # Create registered client
        now = datetime.now(timezone.utc)
        registered_client = RegisteredClient(
            client_id=client_id,
            client_secret=client_secret,
            client_secret_expires_at=client_secret_expires_at,
            client_id_issued_at=int(now.timestamp()),
            registration_access_token=registration_token,
            registration_client_uri=registration_uri,

            # Copy metadata fields
            client_name=metadata.client_name,
            client_uri=str(metadata.client_uri) if metadata.client_uri else None,
            redirect_uris=[str(uri) for uri in metadata.redirect_uris] if metadata.redirect_uris else None,
            client_type=metadata.client_type,
            token_endpoint_auth_method=metadata.token_endpoint_auth_method,
            grant_types=metadata.grant_types,
            response_types=metadata.response_types,
            scope=metadata.scope,
            logo_uri=str(metadata.logo_uri) if metadata.logo_uri else None,
            tos_uri=str(metadata.tos_uri) if metadata.tos_uri else None,
            policy_uri=str(metadata.policy_uri) if metadata.policy_uri else None,
            software_id=metadata.software_id,
            software_version=metadata.software_version,
            software_statement=metadata.software_statement,
            contacts=metadata.contacts,
            jwks_uri=str(metadata.jwks_uri) if metadata.jwks_uri else None,
            jwks=metadata.jwks,
            sector_identifier_uri=str(metadata.sector_identifier_uri) if metadata.sector_identifier_uri else None,
            subject_type=metadata.subject_type,
            id_token_signed_response_alg=metadata.id_token_signed_response_alg,
            mcp_capabilities=metadata.mcp_capabilities,
            mcp_transport_protocols=metadata.mcp_transport_protocols,
            mcp_session_timeout=metadata.mcp_session_timeout,

            # Internal fields
            created_at=now,
            updated_at=now,
            is_active=True,
            rate_limit_count=0
        )

        # Store client
        await self.client_store.store_client(registered_client)

        # Audit log
        await create_audit_log(
            event_type=EventType.AGENT_REGISTERED,
            category=AuditEventCategory.AUTHENTICATION,
            action="oauth2_client_registered",
            description=f"OAuth2 client registered: {client_id}",
            severity=AuditSeverity.INFO,
            source_ip=client_ip,
            user_agent=user_agent,
            outcome="success",
            payload={
                "client_id": client_id,
                "client_name": metadata.client_name,
                "client_type": metadata.client_type.value,
                "grant_types": [gt.value for gt in metadata.grant_types],
                "mcp_capabilities": metadata.mcp_capabilities
            }
        )

        logger.info(f"Registered new OAuth2 client: {client_id} ({metadata.client_name})")
        return registered_client

    async def get_client(self, client_id: str) -> Optional[RegisteredClient]:
        """Retrieve client information."""
        return await self.client_store.get_client(client_id)

    async def update_client(self,
                          client_id: str,
                          metadata: ClientMetadata,
                          registration_token: str,
                          client_ip: str,
                          user_agent: Optional[str] = None) -> RegisteredClient:
        """Update an existing client's metadata."""

        # Verify registration token
        if not self._verify_registration_token(registration_token, client_id):
            await create_audit_log(
                event_type=EventType.AUTHENTICATION_FAILURE,
                category=AuditEventCategory.AUTHORIZATION,
                action="oauth2_client_update_unauthorized",
                description=f"Invalid registration token for client: {client_id}",
                severity=AuditSeverity.MEDIUM,
                source_ip=client_ip,
                user_agent=user_agent,
                outcome="failure"
            )
            raise DCRError("invalid_token", "Invalid registration access token", 401)

        # Get existing client
        existing_client = await self.client_store.get_client(client_id)
        if not existing_client:
            raise DCRError("invalid_client_id", "Client not found", 404)

        # Validate new metadata
        self._validate_client_metadata(metadata)

        # Update client with new metadata
        existing_client.client_name = metadata.client_name
        existing_client.client_uri = str(metadata.client_uri) if metadata.client_uri else None
        existing_client.redirect_uris = [str(uri) for uri in metadata.redirect_uris] if metadata.redirect_uris else None
        existing_client.client_type = metadata.client_type
        existing_client.token_endpoint_auth_method = metadata.token_endpoint_auth_method
        existing_client.grant_types = metadata.grant_types
        existing_client.response_types = metadata.response_types
        existing_client.scope = metadata.scope
        existing_client.logo_uri = str(metadata.logo_uri) if metadata.logo_uri else None
        existing_client.tos_uri = str(metadata.tos_uri) if metadata.tos_uri else None
        existing_client.policy_uri = str(metadata.policy_uri) if metadata.policy_uri else None
        existing_client.software_id = metadata.software_id
        existing_client.software_version = metadata.software_version
        existing_client.contacts = metadata.contacts
        existing_client.jwks_uri = str(metadata.jwks_uri) if metadata.jwks_uri else None
        existing_client.jwks = metadata.jwks
        existing_client.subject_type = metadata.subject_type
        existing_client.mcp_capabilities = metadata.mcp_capabilities
        existing_client.mcp_transport_protocols = metadata.mcp_transport_protocols
        existing_client.mcp_session_timeout = metadata.mcp_session_timeout
        existing_client.updated_at = datetime.now(timezone.utc)

        # Store updated client
        success = await self.client_store.update_client(existing_client)
        if not success:
            raise DCRError("server_error", "Failed to update client")

        # Audit log
        await create_audit_log(
            event_type=EventType.CONFIGURATION_CHANGED,
            category=AuditEventCategory.CONFIGURATION_CHANGE,
            action="oauth2_client_updated",
            description=f"OAuth2 client updated: {client_id}",
            severity=AuditSeverity.INFO,
            source_ip=client_ip,
            user_agent=user_agent,
            outcome="success",
            payload={
                "client_id": client_id,
                "client_name": metadata.client_name,
                "updated_fields": ["metadata"]
            }
        )

        logger.info(f"Updated OAuth2 client: {client_id}")
        return existing_client

    async def delete_client(self,
                          client_id: str,
                          registration_token: str,
                          client_ip: str,
                          user_agent: Optional[str] = None) -> bool:
        """Delete a registered client."""

        # Verify registration token
        if not self._verify_registration_token(registration_token, client_id):
            await create_audit_log(
                event_type=EventType.AUTHENTICATION_FAILURE,
                category=AuditEventCategory.AUTHORIZATION,
                action="oauth2_client_delete_unauthorized",
                description=f"Invalid registration token for client deletion: {client_id}",
                severity=AuditSeverity.MEDIUM,
                source_ip=client_ip,
                user_agent=user_agent,
                outcome="failure"
            )
            raise DCRError("invalid_token", "Invalid registration access token", 401)

        # Check if client exists
        existing_client = await self.client_store.get_client(client_id)
        if not existing_client:
            raise DCRError("invalid_client_id", "Client not found", 404)

        # Delete client
        success = await self.client_store.delete_client(client_id)
        if not success:
            raise DCRError("server_error", "Failed to delete client")

        # Audit log
        await create_audit_log(
            event_type=EventType.AGENT_DEREGISTERED,
            category=AuditEventCategory.AUTHENTICATION,
            action="oauth2_client_deleted",
            description=f"OAuth2 client deleted: {client_id}",
            severity=AuditSeverity.INFO,
            source_ip=client_ip,
            user_agent=user_agent,
            outcome="success",
            payload={
                "client_id": client_id,
                "client_name": existing_client.client_name
            }
        )

        logger.info(f"Deleted OAuth2 client: {client_id}")
        return True

    async def authenticate_client(self,
                                client_id: str,
                                client_secret: Optional[str] = None,
                                client_assertion: Optional[str] = None,
                                client_assertion_type: Optional[str] = None) -> bool:
        """Authenticate a client using various authentication methods."""

        client = await self.get_client(client_id)
        if not client or not client.is_active:
            return False

        # Check secret expiration
        if (client.client_secret_expires_at and
            time.time() > client.client_secret_expires_at):
            return False

        # Authenticate based on method
        auth_method = client.token_endpoint_auth_method

        if auth_method == ClientAuthMethod.NONE:
            # Public client - no authentication required
            return client.client_type == ClientType.PUBLIC

        elif auth_method in [ClientAuthMethod.CLIENT_SECRET_POST, ClientAuthMethod.CLIENT_SECRET_BASIC]:
            # Secret-based authentication
            if not client_secret or not client.client_secret:
                return False
            return hmac.compare_digest(client_secret, client.client_secret)

        elif auth_method == ClientAuthMethod.CLIENT_SECRET_JWT:
            # JWT with shared secret - not implemented in this basic version
            logger.warning(f"Client secret JWT authentication not implemented for {client_id}")
            return False

        elif auth_method == ClientAuthMethod.PRIVATE_KEY_JWT:
            # JWT with private key - not implemented in this basic version
            logger.warning(f"Private key JWT authentication not implemented for {client_id}")
            return False

        return False

    async def list_clients(self, limit: int = 100) -> list[RegisteredClient]:
        """List all registered clients (admin function)."""
        client_ids = await self.client_store.list_clients(limit)
        clients = []

        for client_id in client_ids:
            client = await self.client_store.get_client(client_id)
            if client:
                clients.append(client)

        return clients

    def get_registration_metadata(self) -> dict[str, Any]:
        """Get DCR endpoint metadata for discovery."""
        return {
            "registration_endpoint": "/oauth/register",
            "registration_endpoint_auth_methods_supported": [
                "none"  # Open registration
            ],
            "grant_types_supported": [gt.value for gt in self.supported_grant_types],
            "response_types_supported": [rt.value for rt in self.supported_response_types],
            "token_endpoint_auth_methods_supported": [
                method.value for method in self.supported_auth_methods
            ],
            "client_id_issued_at_supported": True,
            "client_secret_expires_at_supported": True,
            "registration_access_token_supported": True,
            "mcp_extensions_supported": [
                "mcp_capabilities",
                "mcp_transport_protocols",
                "mcp_session_timeout"
            ]
        }


# Global DCR manager instance
_dcr_manager: Optional[DCRManager] = None


def get_dcr_manager() -> DCRManager:
    """Get global DCR manager instance."""
    global _dcr_manager

    if _dcr_manager is None:
        encryption_key = os.getenv('OAUTH_ENCRYPTION_KEY')
        rate_limit = int(os.getenv('OAUTH_RATE_LIMIT_PER_IP', '10'))
        rate_window = int(os.getenv('OAUTH_RATE_LIMIT_WINDOW', '3600'))

        _dcr_manager = DCRManager(
            encryption_key=encryption_key,
            rate_limit_per_ip=rate_limit,
            rate_limit_window=rate_window
        )

        logger.info("Initialized global DCR manager")

    return _dcr_manager
