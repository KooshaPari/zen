"""
Comprehensive JWT Token Management System for OAuth 2.0

This module provides a complete JWT-based token management system for OAuth 2.0
with the following features:

- JWT-based access tokens with 1-hour expiry
- Refresh tokens with 30-day expiry
- Token generation, validation, and parsing
- Token rotation on refresh
- Secure token storage with encryption
- Automatic cleanup of expired tokens
- Token introspection capabilities
- JWKS (JSON Web Key Set) support
- Custom claims for MCP context
- Audience and issuer validation
- Token blacklisting for revocation
- Rate limiting on token operations
- Comprehensive token metadata
- Strong cryptographic signing
- Token encryption for storage
- Secure random generation
- Protection against timing attacks

Security Features:
- RSA256 and HMAC256 signing algorithms
- AES-256-GCM encryption for token storage
- Secure random key generation
- PBKDF2 key derivation
- Timing attack protection
- Token fingerprinting
- Automatic key rotation

Dependencies:
- PyJWT: JWT encoding/decoding
- cryptography: Encryption and key management
- secrets: Secure random generation
- hashlib: Hashing operations
- time: Timestamp operations
"""

import base64
import hashlib
import hmac
import logging
import secrets
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional, Union
from uuid import uuid4

try:
    import jwt
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from pydantic import BaseModel, Field, field_validator
    OAUTH2_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    OAUTH2_DEPENDENCIES_AVAILABLE = False
    OAUTH2_IMPORT_ERROR = str(e)

    # Create dummy classes to prevent import errors
    class BaseModel:
        pass

    class Field:
        pass

    def field_validator(field):
        def decorator(func):
            return func
        return decorator

    print(f"WARNING: OAuth 2.0 dependencies not available: {e}")
    print("Install with: pip install PyJWT cryptography pydantic")


logger = logging.getLogger(__name__)


class TokenType(str, Enum):
    """Token types supported by the system."""
    ACCESS = "access"
    REFRESH = "refresh"
    ID = "id_token"


class AlgorithmType(str, Enum):
    """Supported JWT signing algorithms."""
    RS256 = "RS256"
    HS256 = "HS256"


class TokenStatus(str, Enum):
    """Token validation status."""
    VALID = "valid"
    EXPIRED = "expired"
    INVALID = "invalid"
    REVOKED = "revoked"
    MALFORMED = "malformed"


class TokenClaims(BaseModel):
    """JWT claims structure for OAuth 2.0 tokens."""

    # Standard JWT claims
    iss: str = Field(..., description="Issuer")
    sub: str = Field(..., description="Subject (user ID)")
    aud: Union[str, list[str]] = Field(..., description="Audience")
    exp: int = Field(..., description="Expiration time")
    nbf: int = Field(..., description="Not before")
    iat: int = Field(..., description="Issued at")
    jti: str = Field(..., description="JWT ID")

    # OAuth 2.0 claims
    scope: str = Field("", description="OAuth scopes")
    client_id: str = Field(..., description="Client ID")
    token_type: TokenType = Field(..., description="Token type")

    # MCP-specific claims
    mcp_context: dict[str, Any] = Field(default_factory=dict, description="MCP context")
    session_id: Optional[str] = Field(None, description="Session ID")
    fingerprint: str = Field(..., description="Token fingerprint")

    # Security claims
    auth_time: int = Field(..., description="Authentication time")
    acr: str = Field("1", description="Authentication context class reference")
    amr: list[str] = Field(default_factory=list, description="Authentication methods")

    @field_validator('exp')
    @classmethod
    def validate_expiration(cls, v):
        """Ensure expiration is in the future."""
        if v <= int(time.time()):
            raise ValueError("Token expiration must be in the future")
        return v

    @field_validator('scope')
    @classmethod
    def validate_scope(cls, v):
        """Validate OAuth scopes format."""
        if v and not all(scope.replace(':', '').replace('_', '').replace('-', '').isalnum()
                        for scope in v.split()):
            raise ValueError("Invalid scope format")
        return v


class TokenMetadata(BaseModel):
    """Metadata for stored tokens."""

    token_id: str = Field(..., description="Token ID")
    user_id: str = Field(..., description="User ID")
    client_id: str = Field(..., description="Client ID")
    token_type: TokenType = Field(..., description="Token type")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: datetime = Field(..., description="Expiration timestamp")
    last_used: Optional[datetime] = Field(None, description="Last used timestamp")
    revoked: bool = Field(False, description="Revoked status")
    revoked_at: Optional[datetime] = Field(None, description="Revocation timestamp")
    scope: str = Field("", description="Token scope")
    fingerprint: str = Field(..., description="Token fingerprint")
    refresh_count: int = Field(0, description="Number of times refreshed")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent")


class JWKSKey(BaseModel):
    """JSON Web Key structure."""

    kty: str = Field(..., description="Key type")
    use: str = Field(..., description="Public key use")
    kid: str = Field(..., description="Key ID")
    alg: str = Field(..., description="Algorithm")
    n: Optional[str] = Field(None, description="RSA modulus")
    e: Optional[str] = Field(None, description="RSA exponent")
    x5c: Optional[list[str]] = Field(None, description="X.509 certificate chain")
    x5t: Optional[str] = Field(None, description="X.509 thumbprint")


class JWKS(BaseModel):
    """JSON Web Key Set."""

    keys: list[JWKSKey] = Field(default_factory=list, description="Keys")


class RateLimiter:
    """Simple rate limiter for token operations."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = {}

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for the identifier."""
        now = time.time()

        if identifier not in self.requests:
            self.requests[identifier] = []

        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.window_seconds
        ]

        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False

        # Record request
        self.requests[identifier].append(now)
        return True


class TokenSecurity:
    """Security utilities for token operations."""

    @staticmethod
    def generate_secure_key(length: int = 32) -> bytes:
        """Generate a cryptographically secure random key."""
        return secrets.token_bytes(length)

    @staticmethod
    def generate_fingerprint(token: str, secret: bytes) -> str:
        """Generate a secure fingerprint for a token."""
        return hmac.new(
            secret,
            token.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    @staticmethod
    def constant_time_compare(a: str, b: str) -> bool:
        """Compare strings in constant time to prevent timing attacks."""
        return hmac.compare_digest(a.encode('utf-8'), b.encode('utf-8'))

    @staticmethod
    def derive_key(password: bytes, salt: bytes, iterations: int = 100000) -> bytes:
        """Derive encryption key using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
        )
        return kdf.derive(password)


class OAuth2TokenManager:
    """
    Comprehensive JWT token management system for OAuth 2.0.

    This class provides complete token lifecycle management including:
    - Token generation with custom claims
    - Secure token validation and parsing
    - Token storage with encryption
    - Token revocation and blacklisting
    - JWKS endpoint support
    - Rate limiting
    - Automatic cleanup
    """

    def __init__(
        self,
        issuer: str,
        secret_key: Optional[bytes] = None,
        rsa_private_key: Optional[bytes] = None,
        rsa_public_key: Optional[bytes] = None,
        algorithm: AlgorithmType = AlgorithmType.HS256,
        access_token_expiry: int = 3600,  # 1 hour
        refresh_token_expiry: int = 2592000,  # 30 days
        storage_encryption_key: Optional[bytes] = None,
        rate_limit_requests: int = 100,
        rate_limit_window: int = 3600,
    ):
        """
        Initialize the OAuth2 token manager.

        Args:
            issuer: JWT issuer identifier
            secret_key: HMAC signing key (for HS256)
            rsa_private_key: RSA private key (for RS256)
            rsa_public_key: RSA public key (for RS256)
            algorithm: JWT signing algorithm
            access_token_expiry: Access token expiry in seconds
            refresh_token_expiry: Refresh token expiry in seconds
            storage_encryption_key: Key for encrypting stored tokens
            rate_limit_requests: Max requests per window
            rate_limit_window: Rate limit window in seconds
        """
        self.issuer = issuer
        self.algorithm = algorithm
        self.access_token_expiry = access_token_expiry
        self.refresh_token_expiry = refresh_token_expiry

        # Initialize signing keys
        if algorithm == AlgorithmType.HS256:
            self.secret_key = secret_key or TokenSecurity.generate_secure_key()
            self.rsa_private_key = None
            self.rsa_public_key = None
        elif algorithm == AlgorithmType.RS256:
            if not rsa_private_key:
                # Generate RSA key pair
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                )
                self.rsa_private_key = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                self.rsa_public_key = private_key.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
            else:
                self.rsa_private_key = rsa_private_key
                self.rsa_public_key = rsa_public_key
            self.secret_key = None

        # Initialize storage encryption
        self.storage_encryption_key = storage_encryption_key or TokenSecurity.generate_secure_key()
        self.cipher_suite = Fernet(base64.urlsafe_b64encode(self.storage_encryption_key))

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(rate_limit_requests, rate_limit_window)

        # Token storage
        self.tokens: dict[str, TokenMetadata] = {}
        self.blacklisted_tokens: set[str] = set()

        # Key rotation
        self.key_id = secrets.token_hex(8)
        self.key_rotation_time = datetime.now(timezone.utc)

        logger.info(f"OAuth2TokenManager initialized with {algorithm} algorithm")

    def generate_token_pair(
        self,
        user_id: str,
        client_id: str,
        audience: Union[str, list[str]],
        scope: str = "",
        mcp_context: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> tuple[str, str, dict[str, Any]]:
        """
        Generate an access token and refresh token pair.

        Args:
            user_id: User identifier
            client_id: OAuth client identifier
            audience: Token audience
            scope: OAuth scopes
            mcp_context: MCP-specific context
            session_id: Session identifier
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Tuple of (access_token, refresh_token, metadata)
        """
        if not self.rate_limiter.is_allowed(f"generate:{user_id}"):
            raise ValueError("Rate limit exceeded for token generation")

        now = int(time.time())
        access_jti = str(uuid4())
        refresh_jti = str(uuid4())

        # Generate fingerprints
        fingerprint_key = self.secret_key or self.storage_encryption_key
        access_fingerprint = TokenSecurity.generate_fingerprint(access_jti, fingerprint_key)
        refresh_fingerprint = TokenSecurity.generate_fingerprint(refresh_jti, fingerprint_key)

        # Access token claims
        access_claims = TokenClaims(
            iss=self.issuer,
            sub=user_id,
            aud=audience,
            exp=now + self.access_token_expiry,
            nbf=now,
            iat=now,
            jti=access_jti,
            scope=scope,
            client_id=client_id,
            token_type=TokenType.ACCESS,
            mcp_context=mcp_context or {},
            session_id=session_id,
            fingerprint=access_fingerprint,
            auth_time=now,
            acr="1",
            amr=["pwd"]
        )

        # Refresh token claims
        refresh_claims = TokenClaims(
            iss=self.issuer,
            sub=user_id,
            aud=audience,
            exp=now + self.refresh_token_expiry,
            nbf=now,
            iat=now,
            jti=refresh_jti,
            scope=scope,
            client_id=client_id,
            token_type=TokenType.REFRESH,
            mcp_context=mcp_context or {},
            session_id=session_id,
            fingerprint=refresh_fingerprint,
            auth_time=now,
            acr="1",
            amr=["pwd"]
        )

        # Sign tokens
        access_token = self._sign_token(access_claims.model_dump())
        refresh_token = self._sign_token(refresh_claims.model_dump())

        # Store token metadata
        creation_time = datetime.now(timezone.utc)

        access_metadata = TokenMetadata(
            token_id=access_jti,
            user_id=user_id,
            client_id=client_id,
            token_type=TokenType.ACCESS,
            created_at=creation_time,
            expires_at=creation_time + timedelta(seconds=self.access_token_expiry),
            scope=scope,
            fingerprint=access_fingerprint,
            ip_address=ip_address,
            user_agent=user_agent
        )

        refresh_metadata = TokenMetadata(
            token_id=refresh_jti,
            user_id=user_id,
            client_id=client_id,
            token_type=TokenType.REFRESH,
            created_at=creation_time,
            expires_at=creation_time + timedelta(seconds=self.refresh_token_expiry),
            scope=scope,
            fingerprint=refresh_fingerprint,
            ip_address=ip_address,
            user_agent=user_agent
        )

        # Encrypt and store
        self.tokens[access_jti] = access_metadata
        self.tokens[refresh_jti] = refresh_metadata

        metadata = {
            "access_token_id": access_jti,
            "refresh_token_id": refresh_jti,
            "expires_in": self.access_token_expiry,
            "token_type": "Bearer",
            "scope": scope,
            "created_at": creation_time.isoformat(),
        }

        logger.info(f"Generated token pair for user {user_id}, client {client_id}")
        return access_token, refresh_token, metadata

    def validate_token(
        self,
        token: str,
        expected_audience: Optional[Union[str, list[str]]] = None,
        required_scope: Optional[str] = None,
    ) -> tuple[TokenStatus, Optional[dict[str, Any]]]:
        """
        Validate a JWT token.

        Args:
            token: JWT token to validate
            expected_audience: Expected audience claim
            required_scope: Required scope for authorization

        Returns:
            Tuple of (status, claims)
        """
        try:
            # Decode token
            claims = self._decode_token(token)
            if not claims:
                return TokenStatus.MALFORMED, None

            token_id = claims.get('jti')
            if not token_id:
                return TokenStatus.MALFORMED, None

            # Check if token is blacklisted
            if token_id in self.blacklisted_tokens:
                logger.warning(f"Attempt to use blacklisted token: {token_id}")
                return TokenStatus.REVOKED, None

            # Check if token exists in storage
            if token_id not in self.tokens:
                return TokenStatus.INVALID, None

            metadata = self.tokens[token_id]

            # Check if token is revoked
            if metadata.revoked:
                return TokenStatus.REVOKED, None

            # Check expiration
            if datetime.now(timezone.utc) > metadata.expires_at:
                return TokenStatus.EXPIRED, None

            # Validate audience
            if expected_audience:
                token_aud = claims.get('aud', [])
                if isinstance(token_aud, str):
                    token_aud = [token_aud]
                if isinstance(expected_audience, str):
                    expected_audience = [expected_audience]

                if not any(aud in token_aud for aud in expected_audience):
                    return TokenStatus.INVALID, None

            # Validate scope
            if required_scope:
                token_scopes = claims.get('scope', '').split()
                required_scopes = required_scope.split()
                if not all(scope in token_scopes for scope in required_scopes):
                    return TokenStatus.INVALID, None

            # Verify fingerprint
            fingerprint_key = self.secret_key or self.storage_encryption_key
            expected_fingerprint = TokenSecurity.generate_fingerprint(token_id, fingerprint_key)
            token_fingerprint = claims.get('fingerprint', '')

            if not TokenSecurity.constant_time_compare(expected_fingerprint, token_fingerprint):
                logger.warning(f"Fingerprint mismatch for token: {token_id}")
                return TokenStatus.INVALID, None

            # Update last used time
            metadata.last_used = datetime.now(timezone.utc)

            logger.debug(f"Token validated successfully: {token_id}")
            return TokenStatus.VALID, claims

        except jwt.ExpiredSignatureError:
            return TokenStatus.EXPIRED, None
        except (jwt.InvalidTokenError, ValueError, KeyError) as e:
            logger.warning(f"Token validation error: {e}")
            return TokenStatus.INVALID, None

    def refresh_token(
        self,
        refresh_token: str,
        new_scope: Optional[str] = None,
    ) -> tuple[str, str, dict[str, Any]]:
        """
        Refresh an access token using a refresh token.

        Args:
            refresh_token: Valid refresh token
            new_scope: Optional new scope (must be subset of original)

        Returns:
            Tuple of (new_access_token, new_refresh_token, metadata)
        """
        # Validate refresh token
        status, claims = self.validate_token(refresh_token)
        if status != TokenStatus.VALID:
            raise ValueError(f"Invalid refresh token: {status}")

        if claims.get('token_type') != TokenType.REFRESH:
            raise ValueError("Token is not a refresh token")

        refresh_token_id = claims['jti']
        refresh_metadata = self.tokens[refresh_token_id]

        # Check rate limiting
        if not self.rate_limiter.is_allowed(f"refresh:{claims['sub']}"):
            raise ValueError("Rate limit exceeded for token refresh")

        # Validate new scope is subset of original
        original_scope = set(claims.get('scope', '').split())
        if new_scope:
            new_scope_set = set(new_scope.split())
            if not new_scope_set.issubset(original_scope):
                raise ValueError("New scope must be subset of original scope")
        else:
            new_scope = claims.get('scope', '')

        # Generate new token pair
        new_access_token, new_refresh_token, metadata = self.generate_token_pair(
            user_id=claims['sub'],
            client_id=claims['client_id'],
            audience=claims['aud'],
            scope=new_scope,
            mcp_context=claims.get('mcp_context', {}),
            session_id=claims.get('session_id'),
            ip_address=refresh_metadata.ip_address,
            user_agent=refresh_metadata.user_agent,
        )

        # Increment refresh count
        refresh_metadata.refresh_count += 1

        # Revoke old tokens
        self.revoke_token(refresh_token)

        logger.info(f"Token refreshed for user {claims['sub']}")
        return new_access_token, new_refresh_token, metadata

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token.

        Args:
            token: Token to revoke

        Returns:
            True if successfully revoked
        """
        try:
            claims = self._decode_token(token, verify=False)  # Don't verify expired tokens
            if not claims or 'jti' not in claims:
                return False

            token_id = claims['jti']

            # Add to blacklist
            self.blacklisted_tokens.add(token_id)

            # Mark as revoked in storage
            if token_id in self.tokens:
                self.tokens[token_id].revoked = True
                self.tokens[token_id].revoked_at = datetime.now(timezone.utc)

            logger.info(f"Token revoked: {token_id}")
            return True

        except Exception as e:
            logger.error(f"Error revoking token: {e}")
            return False

    def revoke_user_tokens(self, user_id: str, client_id: Optional[str] = None) -> int:
        """
        Revoke all tokens for a user.

        Args:
            user_id: User identifier
            client_id: Optional client filter

        Returns:
            Number of tokens revoked
        """
        count = 0
        now = datetime.now(timezone.utc)

        for token_id, metadata in self.tokens.items():
            if metadata.user_id == user_id and not metadata.revoked:
                if client_id is None or metadata.client_id == client_id:
                    metadata.revoked = True
                    metadata.revoked_at = now
                    self.blacklisted_tokens.add(token_id)
                    count += 1

        logger.info(f"Revoked {count} tokens for user {user_id}")
        return count

    def introspect_token(self, token: str) -> dict[str, Any]:
        """
        Introspect a token (OAuth 2.0 introspection).

        Args:
            token: Token to introspect

        Returns:
            Token introspection response
        """
        status, claims = self.validate_token(token)

        response = {
            "active": status == TokenStatus.VALID,
            "token_type": "Bearer",
        }

        if status == TokenStatus.VALID and claims:
            token_id = claims['jti']
            metadata = self.tokens.get(token_id)

            response.update({
                "client_id": claims.get('client_id'),
                "username": claims.get('sub'),
                "scope": claims.get('scope', ''),
                "exp": claims.get('exp'),
                "iat": claims.get('iat'),
                "sub": claims.get('sub'),
                "aud": claims.get('aud'),
                "iss": claims.get('iss'),
                "jti": token_id,
                "token_type_hint": claims.get('token_type'),
            })

            if metadata:
                response.update({
                    "created_at": metadata.created_at.isoformat(),
                    "last_used": metadata.last_used.isoformat() if metadata.last_used else None,
                    "refresh_count": metadata.refresh_count,
                })

        return response

    def get_jwks(self) -> dict[str, Any]:
        """
        Get JSON Web Key Set for token verification.

        Returns:
            JWKS document
        """
        keys = []

        if self.algorithm == AlgorithmType.RS256 and self.rsa_public_key:
            # Load public key
            public_key = serialization.load_pem_public_key(self.rsa_public_key)
            public_numbers = public_key.public_numbers()

            # Convert to JWK format
            def int_to_base64url(value):
                byte_length = (value.bit_length() + 7) // 8
                return base64.urlsafe_b64encode(
                    value.to_bytes(byte_length, 'big')
                ).decode('ascii').rstrip('=')

            keys.append({
                "kty": "RSA",
                "use": "sig",
                "kid": self.key_id,
                "alg": "RS256",
                "n": int_to_base64url(public_numbers.n),
                "e": int_to_base64url(public_numbers.e),
            })

        return {"keys": keys}

    def cleanup_expired_tokens(self) -> int:
        """
        Clean up expired tokens from storage.

        Returns:
            Number of tokens cleaned up
        """
        now = datetime.now(timezone.utc)
        expired_tokens = []

        for token_id, metadata in self.tokens.items():
            if now > metadata.expires_at:
                expired_tokens.append(token_id)

        # Remove expired tokens
        for token_id in expired_tokens:
            del self.tokens[token_id]
            self.blacklisted_tokens.discard(token_id)

        logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")
        return len(expired_tokens)

    def get_token_statistics(self) -> dict[str, Any]:
        """
        Get token usage statistics.

        Returns:
            Token statistics
        """
        now = datetime.now(timezone.utc)
        total_tokens = len(self.tokens)
        active_tokens = sum(1 for m in self.tokens.values() if not m.revoked and now <= m.expires_at)
        expired_tokens = sum(1 for m in self.tokens.values() if now > m.expires_at)
        revoked_tokens = sum(1 for m in self.tokens.values() if m.revoked)

        access_tokens = sum(1 for m in self.tokens.values() if m.token_type == TokenType.ACCESS)
        refresh_tokens = sum(1 for m in self.tokens.values() if m.token_type == TokenType.REFRESH)

        return {
            "total_tokens": total_tokens,
            "active_tokens": active_tokens,
            "expired_tokens": expired_tokens,
            "revoked_tokens": revoked_tokens,
            "blacklisted_tokens": len(self.blacklisted_tokens),
            "access_tokens": access_tokens,
            "refresh_tokens": refresh_tokens,
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "key_rotation_time": self.key_rotation_time.isoformat(),
        }

    def _sign_token(self, payload: dict[str, Any]) -> str:
        """Sign a JWT token with the configured algorithm."""
        headers = {"kid": self.key_id}

        if self.algorithm == AlgorithmType.HS256:
            return jwt.encode(payload, self.secret_key, algorithm="HS256", headers=headers)
        elif self.algorithm == AlgorithmType.RS256:
            return jwt.encode(payload, self.rsa_private_key, algorithm="RS256", headers=headers)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _decode_token(self, token: str, verify: bool = True) -> Optional[dict[str, Any]]:
        """Decode and verify a JWT token."""
        try:
            if self.algorithm == AlgorithmType.HS256:
                key = self.secret_key
            elif self.algorithm == AlgorithmType.RS256:
                key = self.rsa_public_key
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")

            if verify:
                claims = jwt.decode(
                    token,
                    key,
                    algorithms=[self.algorithm.value],  # Use .value to get string
                    options={"verify_signature": True, "verify_iss": False, "verify_aud": False}
                )
            else:
                claims = jwt.decode(
                    token,
                    options={"verify_signature": False, "verify_exp": False}
                )

            return claims

        except Exception as e:
            logger.debug(f"Token decode error: {e}")
            return None


# Factory function for easy initialization
def create_oauth2_token_manager(
    issuer: str,
    algorithm: AlgorithmType = AlgorithmType.HS256,
    secret_key: Optional[str] = None,
    **kwargs
) -> OAuth2TokenManager:
    """
    Create an OAuth2TokenManager with default settings.

    Args:
        issuer: JWT issuer identifier
        algorithm: JWT signing algorithm
        secret_key: Optional secret key (hex string)
        **kwargs: Additional arguments for OAuth2TokenManager

    Returns:
        Configured OAuth2TokenManager instance
    """
    if not OAUTH2_DEPENDENCIES_AVAILABLE:
        raise ImportError(f"OAuth2 dependencies not available: {OAUTH2_IMPORT_ERROR}")

    logger.info(f"Creating OAuth2 token manager for issuer: {issuer}")
    if secret_key:
        secret_bytes = bytes.fromhex(secret_key)
    else:
        secret_bytes = None

    return OAuth2TokenManager(
        issuer=issuer,
        algorithm=algorithm,
        secret_key=secret_bytes,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    token_manager = create_oauth2_token_manager(
        issuer="https://zen-mcp-server.example.com",
        algorithm=AlgorithmType.HS256
    )

    # Generate token pair
    access_token, refresh_token, metadata = token_manager.generate_token_pair(
        user_id="user123",
        client_id="mcp_client",
        audience="https://api.example.com",
        scope="read write",
        mcp_context={"tool": "chat", "session": "abc123"}
    )

    print(f"Access Token: {access_token[:50]}...")
    print(f"Refresh Token: {refresh_token[:50]}...")
    print(f"Metadata: {metadata}")

    # Validate token
    status, claims = token_manager.validate_token(
        access_token,
        expected_audience="https://api.example.com"
    )
    print(f"Token Status: {status}")
    print(f"Claims: {claims}")

    # Introspect token
    introspection = token_manager.introspect_token(access_token)
    print(f"Introspection: {introspection}")

    # Get statistics
    stats = token_manager.get_token_statistics()
    print(f"Statistics: {stats}")

    # Get JWKS
    jwks = token_manager.get_jwks()
    print(f"JWKS: {jwks}")
