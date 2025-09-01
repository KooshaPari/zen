# OAuth2 JWT Token Management System

This document provides comprehensive documentation for the OAuth2 JWT token management system implemented in `auth/oauth2_tokens.py`.

## Overview

The OAuth2TokenManager provides a complete JWT-based authentication and authorization system for the Zen MCP Server with the following key features:

### Core Features

- **JWT-based Access Tokens**: Short-lived tokens (1 hour default) for API access
- **Refresh Tokens**: Long-lived tokens (30 days default) for token renewal
- **Token Generation**: Secure token creation with custom claims
- **Token Validation**: Comprehensive validation with audience and scope checking
- **Token Rotation**: Automatic token refresh with security rotation
- **Token Revocation**: Individual and bulk token revocation capabilities

### Security Features

- **Strong Cryptographic Signing**: Support for HMAC-SHA256 and RSA-SHA256 algorithms
- **Token Encryption**: AES-256-GCM encryption for stored tokens
- **Secure Random Generation**: Cryptographically secure token generation
- **Fingerprinting**: Token integrity verification with HMAC fingerprints
- **Timing Attack Protection**: Constant-time string comparisons
- **Key Rotation**: Automatic cryptographic key rotation capabilities
- **Rate Limiting**: Configurable rate limiting on token operations

### OAuth2 Compliance

- **Standard Claims**: Full support for OAuth2 and JWT standard claims
- **Scope Management**: OAuth2 scope validation and inheritance
- **Audience Validation**: Multi-audience token support
- **Token Introspection**: OAuth2 token introspection endpoint compatibility
- **JWKS Support**: JSON Web Key Set for public key distribution

### MCP Integration

- **MCP Context**: Custom claims for MCP-specific context and metadata
- **Session Tracking**: Session-aware token management
- **Tool Integration**: Designed for MCP tool authentication
- **Client Metadata**: IP address and user agent tracking

## Quick Start

### Installation

Install the required dependencies:

```bash
pip install PyJWT cryptography pydantic
```

### Basic Usage

```python
from auth.oauth2_tokens import create_oauth2_token_manager, AlgorithmType

# Create token manager
token_manager = create_oauth2_token_manager(
    issuer="https://your-mcp-server.com",
    algorithm=AlgorithmType.HS256,
    access_token_expiry=3600,    # 1 hour
    refresh_token_expiry=2592000  # 30 days
)

# Generate token pair
access_token, refresh_token, metadata = token_manager.generate_token_pair(
    user_id="user123",
    client_id="mcp_client",
    audience="https://api.your-server.com",
    scope="read write",
    mcp_context={
        "tool": "chat",
        "session": "session_abc",
        "version": "1.0"
    }
)

# Validate access token
status, claims = token_manager.validate_token(
    access_token,
    expected_audience="https://api.your-server.com",
    required_scope="read"
)

if status == TokenStatus.VALID:
    user_id = claims["sub"]
    scopes = claims["scope"].split()
    mcp_context = claims["mcp_context"]
    # Process authenticated request
```

## Architecture

### Core Components

#### OAuth2TokenManager

The main class that orchestrates all token operations:

```python
class OAuth2TokenManager:
    def __init__(
        self,
        issuer: str,
        secret_key: Optional[bytes] = None,
        rsa_private_key: Optional[bytes] = None,
        rsa_public_key: Optional[bytes] = None,
        algorithm: AlgorithmType = AlgorithmType.HS256,
        access_token_expiry: int = 3600,
        refresh_token_expiry: int = 2592000,
        storage_encryption_key: Optional[bytes] = None,
        rate_limit_requests: int = 100,
        rate_limit_window: int = 3600,
    )
```

#### Token Claims Structure

```python
class TokenClaims(BaseModel):
    # Standard JWT claims
    iss: str  # Issuer
    sub: str  # Subject (user ID)
    aud: Union[str, List[str]]  # Audience
    exp: int  # Expiration time
    nbf: int  # Not before
    iat: int  # Issued at
    jti: str  # JWT ID
    
    # OAuth 2.0 claims
    scope: str  # OAuth scopes
    client_id: str  # Client ID
    token_type: TokenType  # Token type
    
    # MCP-specific claims
    mcp_context: Dict[str, Any]  # MCP context
    session_id: Optional[str]  # Session ID
    fingerprint: str  # Token fingerprint
    
    # Security claims
    auth_time: int  # Authentication time
    acr: str  # Authentication context class
    amr: List[str]  # Authentication methods
```

### Security Architecture

#### Signing Algorithms

**HMAC-SHA256 (HS256)**
- Symmetric key algorithm
- Fast and efficient
- Suitable for single-service deployments
- Keys must be shared between services

```python
# HMAC-SHA256 setup
token_manager = create_oauth2_token_manager(
    issuer="https://your-server.com",
    algorithm=AlgorithmType.HS256,
    secret_key="your_hex_secret_key"  # Optional, auto-generated if not provided
)
```

**RSA-SHA256 (RS256)**
- Asymmetric key algorithm
- Public key can be shared openly
- Suitable for distributed systems
- Supports JWKS for public key distribution

```python
# RSA-SHA256 setup
token_manager = create_oauth2_token_manager(
    issuer="https://your-server.com",
    algorithm=AlgorithmType.RS256,
    # Keys auto-generated if not provided
    rsa_private_key=private_key_bytes,
    rsa_public_key=public_key_bytes
)

# Get JWKS for public key distribution
jwks = token_manager.get_jwks()
```

#### Token Storage Security

- **Encryption**: All stored tokens are encrypted using AES-256-GCM
- **Key Derivation**: PBKDF2 for deriving encryption keys
- **Secure Storage**: Token metadata stored with encryption
- **Memory Safety**: Sensitive data cleared from memory when possible

#### Security Measures

1. **Fingerprinting**: Each token contains a cryptographic fingerprint
2. **Rate Limiting**: Configurable rate limits on token operations
3. **Timing Attack Protection**: Constant-time comparisons
4. **Key Rotation**: Support for cryptographic key rotation
5. **Blacklisting**: Revoked tokens are blacklisted
6. **Audit Trail**: Complete audit trail of token operations

## API Reference

### Token Generation

```python
def generate_token_pair(
    self,
    user_id: str,
    client_id: str,
    audience: Union[str, List[str]],
    scope: str = "",
    mcp_context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> Tuple[str, str, Dict[str, Any]]
```

Generates an access token and refresh token pair.

**Parameters:**
- `user_id`: Unique user identifier
- `client_id`: OAuth2 client identifier
- `audience`: Token audience (API endpoint)
- `scope`: Space-separated OAuth2 scopes
- `mcp_context`: MCP-specific context data
- `session_id`: Session identifier
- `ip_address`: Client IP address
- `user_agent`: Client user agent string

**Returns:**
Tuple of (access_token, refresh_token, metadata)

### Token Validation

```python
def validate_token(
    self,
    token: str,
    expected_audience: Optional[Union[str, List[str]]] = None,
    required_scope: Optional[str] = None,
) -> Tuple[TokenStatus, Optional[Dict[str, Any]]]
```

Validates a JWT token with comprehensive checks.

**Parameters:**
- `token`: JWT token to validate
- `expected_audience`: Expected audience claim
- `required_scope`: Required OAuth2 scope

**Returns:**
Tuple of (TokenStatus, claims)

**Token Status Values:**
- `TokenStatus.VALID`: Token is valid and active
- `TokenStatus.EXPIRED`: Token has expired
- `TokenStatus.INVALID`: Token is malformed or invalid
- `TokenStatus.REVOKED`: Token has been revoked
- `TokenStatus.MALFORMED`: Token format is invalid

### Token Refresh

```python
def refresh_token(
    self,
    refresh_token: str,
    new_scope: Optional[str] = None,
) -> Tuple[str, str, Dict[str, Any]]
```

Refreshes an access token using a valid refresh token.

**Parameters:**
- `refresh_token`: Valid refresh token
- `new_scope`: Optional new scope (must be subset of original)

**Returns:**
Tuple of (new_access_token, new_refresh_token, metadata)

### Token Revocation

```python
def revoke_token(self, token: str) -> bool
```

Revokes a single token.

```python
def revoke_user_tokens(self, user_id: str, client_id: Optional[str] = None) -> int
```

Revokes all tokens for a user, optionally filtered by client.

### Token Introspection

```python
def introspect_token(self, token: str) -> Dict[str, Any]
```

Provides OAuth2 token introspection response.

**Example Response:**
```json
{
    "active": true,
    "client_id": "mcp_client",
    "username": "user123",
    "scope": "read write",
    "exp": 1640995200,
    "iat": 1640991600,
    "sub": "user123",
    "aud": "https://api.example.com",
    "iss": "https://mcp-server.com",
    "jti": "token_id_123",
    "token_type_hint": "access",
    "created_at": "2021-12-31T12:00:00Z",
    "last_used": "2021-12-31T12:30:00Z",
    "refresh_count": 0
}
```

### JWKS Support

```python
def get_jwks(self) -> Dict[str, Any]
```

Returns JSON Web Key Set for RSA public keys.

**Example JWKS:**
```json
{
    "keys": [
        {
            "kty": "RSA",
            "use": "sig",
            "kid": "key_id_123",
            "alg": "RS256",
            "n": "base64_encoded_modulus",
            "e": "base64_encoded_exponent"
        }
    ]
}
```

### Maintenance Operations

```python
def cleanup_expired_tokens(self) -> int
```

Removes expired tokens from storage and returns count of cleaned tokens.

```python
def get_token_statistics(self) -> Dict[str, Any]
```

Returns comprehensive token usage statistics.

## Configuration Examples

### Production Configuration

```python
import secrets
from auth.oauth2_tokens import create_oauth2_token_manager, AlgorithmType

# Production-ready configuration
token_manager = create_oauth2_token_manager(
    issuer="https://mcp-server.yourcompany.com",
    algorithm=AlgorithmType.RS256,  # RSA for distributed systems
    access_token_expiry=3600,       # 1 hour
    refresh_token_expiry=2592000,   # 30 days
    rate_limit_requests=1000,       # Higher limit for production
    rate_limit_window=3600,         # 1 hour window
)
```

### Development Configuration

```python
# Development configuration
token_manager = create_oauth2_token_manager(
    issuer="https://localhost:8080",
    algorithm=AlgorithmType.HS256,  # HMAC for simplicity
    access_token_expiry=86400,      # 24 hours for development
    refresh_token_expiry=604800,    # 7 days
    rate_limit_requests=100,        # Lower limit
    rate_limit_window=3600,
)
```

### MCP-Specific Configuration

```python
# MCP server integration
token_manager = create_oauth2_token_manager(
    issuer="https://zen-mcp-server.com",
    algorithm=AlgorithmType.RS256,
    access_token_expiry=3600,
    refresh_token_expiry=2592000,
)

# Generate tokens for MCP tools
access_token, refresh_token, metadata = token_manager.generate_token_pair(
    user_id="mcp_user_123",
    client_id="zen_mcp_client",
    audience=["https://api.zen-mcp.com", "https://tools.zen-mcp.com"],
    scope="chat:read chat:write analyze:read codereview:read codereview:write",
    mcp_context={
        "server_version": "5.11.0",
        "client_capabilities": ["streaming", "memory", "context"],
        "session_metadata": {
            "started_at": "2023-12-31T12:00:00Z",
            "last_activity": "2023-12-31T12:30:00Z"
        },
        "tool_access": {
            "chat": {"max_history": 100},
            "analyze": {"max_files": 50},
            "codereview": {"max_lines": 10000}
        }
    },
    session_id="mcp_session_abc123",
)
```

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from auth.oauth2_tokens import OAuth2TokenManager, TokenStatus

app = FastAPI()
security = HTTPBearer()
token_manager = create_oauth2_token_manager(
    issuer="https://your-api.com",
    algorithm=AlgorithmType.RS256
)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    status, claims = token_manager.validate_token(
        token,
        expected_audience="https://your-api.com"
    )
    
    if status != TokenStatus.VALID:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return claims

@app.post("/api/chat")
async def chat_endpoint(
    request: ChatRequest,
    claims: dict = Depends(verify_token)
):
    user_id = claims["sub"]
    scopes = claims["scope"].split()
    mcp_context = claims["mcp_context"]
    
    if "chat:write" not in scopes:
        raise HTTPException(status_code=403, detail="Insufficient scope")
    
    # Process chat request with user context
    return {"response": "Chat processed", "user": user_id}
```

### MCP Tool Authentication

```python
from tools.shared.base_tool import BaseTool
from auth.oauth2_tokens import TokenStatus

class AuthenticatedChatTool(BaseTool):
    def __init__(self, token_manager: OAuth2TokenManager):
        super().__init__()
        self.token_manager = token_manager
    
    async def call_tool(self, arguments: dict, context: dict = None):
        # Extract token from context or headers
        token = context.get("authorization") if context else None
        if not token:
            raise ValueError("Authentication required")
        
        # Validate token
        status, claims = self.token_manager.validate_token(
            token,
            expected_audience="https://zen-mcp-server.com",
            required_scope="chat:write"
        )
        
        if status != TokenStatus.VALID:
            raise ValueError(f"Invalid token: {status}")
        
        # Extract user and MCP context
        user_id = claims["sub"]
        mcp_context = claims["mcp_context"]
        session_id = claims.get("session_id")
        
        # Process tool call with authenticated context
        result = await self._process_chat(
            arguments,
            user_id=user_id,
            mcp_context=mcp_context,
            session_id=session_id
        )
        
        return result
```

## Testing

### Unit Tests

The system includes comprehensive unit tests in `auth/test_oauth2_tokens.py`:

```bash
# Run tests
python -m pytest auth/test_oauth2_tokens.py -v

# Run with coverage
python -m pytest auth/test_oauth2_tokens.py --cov=auth.oauth2_tokens --cov-report=html
```

### Manual Testing

```bash
# Run the test module directly
cd /path/to/zen-mcp-server
python auth/test_oauth2_tokens.py
```

### Load Testing

```python
import time
import concurrent.futures
from auth.oauth2_tokens import create_oauth2_token_manager

def load_test_token_generation():
    token_manager = create_oauth2_token_manager(
        issuer="https://test.com",
        rate_limit_requests=10000,  # High limit for testing
    )
    
    def generate_token(i):
        return token_manager.generate_token_pair(
            user_id=f"user_{i}",
            client_id="test_client",
            audience="https://api.test.com"
        )
    
    # Generate 1000 tokens concurrently
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(generate_token, i) for i in range(1000)]
        results = [future.result() for future in futures]
    
    end_time = time.time()
    print(f"Generated 1000 tokens in {end_time - start_time:.2f} seconds")
    print(f"Rate: {1000 / (end_time - start_time):.2f} tokens/second")
```

## Security Considerations

### Production Deployment

1. **Key Management**: Use a secure key management system for production keys
2. **Secret Rotation**: Implement regular secret key rotation
3. **Rate Limiting**: Configure appropriate rate limits based on usage patterns
4. **Monitoring**: Monitor token usage and security events
5. **Audit Logging**: Enable comprehensive audit logging
6. **HTTPS Only**: Always use HTTPS for token transmission
7. **Secure Storage**: Use encrypted storage for token metadata

### Best Practices

1. **Short Token Lifetimes**: Use short access token lifetimes (1 hour or less)
2. **Scope Limitation**: Grant minimal required scopes
3. **Audience Validation**: Always validate token audience
4. **Refresh Rotation**: Rotate refresh tokens on each use
5. **Revocation**: Implement token revocation for security incidents
6. **Client Authentication**: Implement strong client authentication
7. **PKCE**: Consider PKCE for public clients

### Threat Mitigation

1. **Token Theft**: Short lifetimes and refresh rotation limit exposure
2. **Replay Attacks**: JTI claims and fingerprinting prevent replay
3. **Timing Attacks**: Constant-time comparisons prevent timing analysis
4. **Brute Force**: Rate limiting prevents brute force attacks
5. **Key Compromise**: Key rotation limits compromise impact

## Troubleshooting

### Common Issues

**"Invalid token" errors:**
- Check token expiration
- Verify audience matches expected value
- Ensure required scopes are present
- Check token has not been revoked

**"Rate limit exceeded" errors:**
- Increase rate limit settings
- Implement client-side rate limiting
- Use token caching to reduce requests

**"Fingerprint mismatch" errors:**
- Check token integrity
- Verify token wasn't modified in transit
- Ensure correct secret key is used

**"Malformed token" errors:**
- Verify token format (three parts separated by dots)
- Check token encoding
- Ensure token wasn't truncated

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("auth.oauth2_tokens")

# Create token manager with debug logging
token_manager = create_oauth2_token_manager(
    issuer="https://debug.com",
    algorithm=AlgorithmType.HS256
)

# Debug token validation
status, claims = token_manager.validate_token(token)
print(f"Status: {status}")
if claims:
    print(f"Claims: {claims}")
```

## Migration Guide

### From Simple Authentication

If migrating from a simple authentication system:

1. **Install Dependencies**: Add PyJWT and cryptography
2. **Initialize Token Manager**: Create OAuth2TokenManager instance
3. **Update Token Generation**: Replace simple tokens with JWT generation
4. **Update Validation**: Replace simple validation with comprehensive JWT validation
5. **Add Scope Checking**: Implement OAuth2 scope validation
6. **Update Storage**: Migrate to encrypted token storage

### From Other JWT Libraries

If migrating from another JWT library:

1. **Token Claims**: Map existing claims to TokenClaims structure
2. **Signing Keys**: Convert existing keys to OAuth2TokenManager format
3. **Validation Logic**: Replace with OAuth2TokenManager validation
4. **Refresh Logic**: Implement refresh token flow
5. **Storage**: Implement encrypted token storage
6. **Rate Limiting**: Add rate limiting to prevent abuse

## Performance Optimization

### Token Generation Optimization

- Use HMAC-SHA256 for single-service deployments (faster than RSA)
- Pre-generate key material to avoid runtime generation
- Implement token pooling for high-throughput scenarios

### Validation Optimization

- Cache public keys for RSA validation
- Implement token validation middleware
- Use connection pooling for token storage backends

### Storage Optimization

- Implement periodic cleanup of expired tokens
- Use efficient storage backends (Redis, etc.)
- Implement token metadata indexing

## Changelog

### Version 1.0.0
- Initial release with comprehensive OAuth2 JWT support
- HMAC-SHA256 and RSA-SHA256 signing algorithms
- Token generation, validation, refresh, and revocation
- Rate limiting and security features
- MCP-specific claims and context support
- Comprehensive test suite and documentation

## License

This OAuth2 JWT Token Management System is part of the Zen MCP Server project and is subject to the same license terms.

## Support

For issues, questions, or contributions related to the OAuth2 token management system:

1. Check the troubleshooting section above
2. Review the test cases for usage examples
3. Create an issue in the project repository
4. Consult the MCP protocol documentation for integration details