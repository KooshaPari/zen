"""
OAuth 2.0 data models and authorization flow components
Integrated with WebAuthn biometric authentication
"""

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse


class ResponseType(Enum):
    CODE = "code"
    TOKEN = "token"  # Implicit flow (less secure)
    ID_TOKEN = "id_token"  # OpenID Connect


class GrantType(Enum):
    AUTHORIZATION_CODE = "authorization_code"
    IMPLICIT = "implicit"
    RESOURCE_OWNER = "password"
    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"


class CodeChallengeMethod(Enum):
    PLAIN = "plain"
    S256 = "S256"


@dataclass
class OAuthScope:
    """OAuth 2.0 scope definition"""
    name: str
    description: str
    required: bool = False


@dataclass
class OAuthClient:
    """OAuth 2.0 client application"""
    client_id: str
    client_secret: str  # Optional for public clients
    name: str
    description: str
    redirect_uris: list[str]
    allowed_scopes: set[str]
    grant_types: set[GrantType] = field(default_factory=lambda: {GrantType.AUTHORIZATION_CODE})
    is_public: bool = False  # Public clients don't need client_secret
    require_pkce: bool = True  # Require PKCE for security
    trusted: bool = False  # Trusted clients skip consent
    created_at: float = field(default_factory=time.time)

    def supports_grant_type(self, grant_type: GrantType) -> bool:
        """Check if client supports a specific grant type"""
        return grant_type in self.grant_types

    def supports_redirect_uri(self, redirect_uri: str) -> bool:
        """Check if redirect URI is allowed for this client"""
        return redirect_uri in self.redirect_uris

    def supports_scope(self, scope: str) -> bool:
        """Check if scope is allowed for this client"""
        return scope in self.allowed_scopes


@dataclass
class AuthorizationRequest:
    """OAuth 2.0 authorization request state"""
    request_id: str
    client_id: str
    redirect_uri: str
    response_type: ResponseType
    scopes: set[str]
    state: Optional[str] = None
    code_challenge: Optional[str] = None
    code_challenge_method: Optional[CodeChallengeMethod] = None
    user_id: Optional[str] = None
    webauthn_challenge: Optional[str] = None  # Link to WebAuthn challenge
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 600)  # 10 minutes
    consent_given: bool = False
    device_info: dict = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if authorization request has expired"""
        return time.time() > self.expires_at

    def validate_pkce(self, code_verifier: str) -> bool:
        """Validate PKCE code verifier against stored challenge"""
        if not self.code_challenge or not self.code_challenge_method:
            return True  # PKCE not used

        if self.code_challenge_method == CodeChallengeMethod.PLAIN:
            return self.code_challenge == code_verifier
        elif self.code_challenge_method == CodeChallengeMethod.S256:
            # SHA256(code_verifier) base64url-encoded
            import base64
            digest = hashlib.sha256(code_verifier.encode()).digest()
            expected = base64.urlsafe_b64encode(digest).decode().rstrip('=')
            return self.code_challenge == expected

        return False


@dataclass
class AuthorizationCode:
    """OAuth 2.0 authorization code"""
    code: str
    client_id: str
    user_id: str
    redirect_uri: str
    scopes: set[str]
    code_challenge: Optional[str] = None
    code_challenge_method: Optional[CodeChallengeMethod] = None
    webauthn_credential_id: Optional[str] = None  # Link to WebAuthn credential
    device_info: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 600)  # 10 minutes
    used: bool = False

    def is_expired(self) -> bool:
        """Check if authorization code has expired"""
        return time.time() > self.expires_at

    def is_valid(self) -> bool:
        """Check if authorization code is valid for use"""
        return not self.used and not self.is_expired()

    def use_code(self) -> bool:
        """Mark code as used (codes can only be used once)"""
        if self.is_valid():
            self.used = True
            return True
        return False


@dataclass
class AccessToken:
    """OAuth 2.0 access token"""
    token: str
    token_type: str = "Bearer"
    client_id: str = ""
    user_id: str = ""
    scopes: set[str] = field(default_factory=set)
    expires_at: float = field(default_factory=lambda: time.time() + 3600)  # 1 hour
    refresh_token: Optional[str] = None
    webauthn_session_id: Optional[str] = None  # Link to WebAuthn session
    device_info: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if access token has expired"""
        return time.time() > self.expires_at

    def is_valid(self) -> bool:
        """Check if access token is valid"""
        return not self.is_expired()

    def has_scope(self, required_scope: str) -> bool:
        """Check if token has required scope"""
        return required_scope in self.scopes


@dataclass
class RefreshToken:
    """OAuth 2.0 refresh token"""
    token: str
    client_id: str
    user_id: str
    scopes: set[str]
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 86400 * 30)  # 30 days
    used_count: int = 0
    max_uses: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if refresh token has expired"""
        return time.time() > self.expires_at

    def is_valid(self) -> bool:
        """Check if refresh token is valid"""
        if self.is_expired():
            return False
        if self.max_uses and self.used_count >= self.max_uses:
            return False
        return True

    def use_token(self) -> bool:
        """Use refresh token (increment use count)"""
        if self.is_valid():
            self.used_count += 1
            return True
        return False


class OAuth2Error(Exception):
    """OAuth 2.0 error with standard error codes"""

    def __init__(self, error: str, error_description: str = "", error_uri: str = "", state: str = ""):
        self.error = error
        self.error_description = error_description
        self.error_uri = error_uri
        self.state = state
        super().__init__(f"{error}: {error_description}")

    def to_dict(self) -> dict[str, str]:
        """Convert error to dictionary for JSON response"""
        result = {"error": self.error}
        if self.error_description:
            result["error_description"] = self.error_description
        if self.error_uri:
            result["error_uri"] = self.error_uri
        if self.state:
            result["state"] = self.state
        return result

    def to_redirect_params(self) -> str:
        """Convert error to URL parameters for redirect"""
        return urlencode(self.to_dict())


# Standard OAuth 2.0 error codes
class OAuth2Errors:
    INVALID_REQUEST = "invalid_request"
    UNAUTHORIZED_CLIENT = "unauthorized_client"
    ACCESS_DENIED = "access_denied"
    UNSUPPORTED_RESPONSE_TYPE = "unsupported_response_type"
    INVALID_SCOPE = "invalid_scope"
    SERVER_ERROR = "server_error"
    TEMPORARILY_UNAVAILABLE = "temporarily_unavailable"
    INVALID_CLIENT = "invalid_client"
    INVALID_GRANT = "invalid_grant"
    UNSUPPORTED_GRANT_TYPE = "unsupported_grant_type"


# Standard OAuth 2.0 scopes
class StandardScopes:
    """Standard OAuth 2.0 scopes for MCP server"""

    READ = OAuthScope(
        name="read",
        description="Read access to MCP tools and resources",
        required=True
    )

    WRITE = OAuthScope(
        name="write",
        description="Write access to MCP tools and file operations"
    )

    ADMIN = OAuthScope(
        name="admin",
        description="Administrative access to server configuration"
    )

    TOOLS = OAuthScope(
        name="tools",
        description="Access to execute MCP tools"
    )

    FILES = OAuthScope(
        name="files",
        description="Access to file system operations"
    )

    PROFILE = OAuthScope(
        name="profile",
        description="Access to user profile information"
    )

    @classmethod
    def get_all_scopes(cls) -> dict[str, OAuthScope]:
        """Get all available scopes"""
        return {
            "read": cls.READ,
            "write": cls.WRITE,
            "admin": cls.ADMIN,
            "tools": cls.TOOLS,
            "files": cls.FILES,
            "profile": cls.PROFILE,
        }

    @classmethod
    def get_default_scopes(cls) -> set[str]:
        """Get default scopes for new clients"""
        return {"read", "tools"}

    @classmethod
    def validate_scopes(cls, requested_scopes: set[str]) -> set[str]:
        """Validate and filter requested scopes.

        Accepts either standard scope names (e.g., "read", "write") or
        MCP-prefixed forms (e.g., "mcp:read", "mcp:write"). The bare
        "mcp" meta-scope expands to a reasonable default set ("read",
        "tools"). Returns the normalized standard scope names.
        """
        available_scopes = cls.get_all_scopes()

        normalized: set[str] = set()
        for s in requested_scopes:
            if not s:
                continue
            # Expand meta-scope "mcp" to defaults
            if s == "mcp":
                normalized.update({"read", "tools"})
                continue
            # Strip known prefix "mcp:" if present
            if s.startswith("mcp:"):
                s = s.split(":", 1)[1]
            # Keep only recognized scopes
            if s in available_scopes:
                normalized.add(s)

        return normalized


def generate_oauth_token(length: int = 32) -> str:
    """Generate a secure random token for OAuth use"""
    return secrets.token_urlsafe(length)


def generate_client_credentials() -> tuple[str, str]:
    """Generate client ID and secret for new OAuth client"""
    client_id = f"mcp_client_{secrets.token_hex(8)}"
    client_secret = secrets.token_urlsafe(64)
    return client_id, client_secret


def parse_authorization_header(auth_header: str) -> Optional[tuple[str, str]]:
    """Parse OAuth authorization header for client credentials"""
    if not auth_header.startswith("Basic "):
        return None

    try:
        import base64
        encoded = auth_header[6:]  # Remove "Basic "
        decoded = base64.b64decode(encoded).decode()
        if ":" in decoded:
            client_id, client_secret = decoded.split(":", 1)
            return client_id, client_secret
    except Exception:
        pass

    return None


def build_authorization_url(base_url: str, params: dict[str, str]) -> str:
    """Build authorization URL with parameters"""
    parsed = urlparse(base_url)
    query_params = parse_qs(parsed.query)

    # Add new parameters
    for key, value in params.items():
        query_params[key] = [value]

    # Rebuild URL
    new_query = urlencode({k: v[0] for k, v in query_params.items()})
    return urlunparse(parsed._replace(query=new_query))
