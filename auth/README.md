# Authentication Module - Zen MCP Server

This directory contains the complete authentication system for the Zen MCP Server, providing secure access control for MCP endpoints and tools.

## Components

### 1. OAuth 2.0 Authorization Server (`oauth2_server.py`)

A complete OAuth 2.0 authorization server implementation compliant with RFC 6749, featuring:

#### Features
- **Authorization Code Flow with PKCE** (RFC 7636) - Secure for public clients
- **JWT Access Tokens** - Standards-based token format with claims
- **Refresh Token Support** - Long-lived token renewal
- **Token Revocation** - Secure token invalidation
- **Token Introspection** - Token validation endpoint
- **WebAuthn Integration** - Device-based authentication
- **Security Headers** - CORS and cache control
- **Error Handling** - Proper OAuth 2.0 error responses

#### Supported Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/oauth/authorize` | GET | Authorization endpoint for code flow |
| `/oauth/token` | POST | Token endpoint for code/refresh exchange |
| `/oauth/revoke` | POST | Token revocation endpoint |
| `/oauth/introspect` | POST | Token introspection endpoint |

#### Supported Grant Types
- `authorization_code` - Standard authorization code flow
- `refresh_token` - Token refresh flow

#### Supported Scopes
- `mcp:read` - Read access to MCP tools and data
- `mcp:write` - Write access to MCP tools and operations
- `mcp:admin` - Administrative access to MCP server

#### Example Usage

```python
from auth.oauth2_server import OAuth2Server, create_oauth2_server
from auth.webauthn_flow import WebAuthnDeviceAuth

# Create WebAuthn authentication
webauthn = WebAuthnDeviceAuth(rp_id="your-domain.com")

# Create OAuth 2.0 server
oauth2_server = create_oauth2_server(
    issuer="https://your-domain.com",
    webauthn_auth=webauthn
)

# Register a client
client = OAuthClient(
    client_id="my-mcp-client",
    redirect_uris=["https://app.example.com/callback"],
    name="My MCP Application"
)
oauth2_server.register_client(client)
```

### 2. WebAuthn Device Authentication (`webauthn_flow.py`)

WebAuthn-based device authentication supporting modern biometric authentication:

#### Supported Authenticators
- **Touch ID** (macOS, iOS)
- **Face ID** (macOS, iOS)
- **Windows Hello** (Windows 10/11)
- **Hardware Security Keys** (YubiKey, etc.)
- **Built-in Platform Authenticators**

#### Features
- Device registration and management
- Biometric authentication requirements
- Challenge-response authentication
- Session management
- HTML interface for browser integration

#### Example Usage

```python
from auth.webauthn_flow import WebAuthnDeviceAuth

webauthn = WebAuthnDeviceAuth(rp_id="your-domain.com")

# Register a device
registration_options = await webauthn.initiate_registration(
    user_id="user@example.com",
    device_name="MacBook Pro Touch ID"
)

# Complete registration after WebAuthn ceremony
success = await webauthn.complete_registration(credential_response)

# Authenticate with registered device
auth_options = await webauthn.initiate_authentication("user@example.com")
user_id = await webauthn.verify_authentication(auth_response)
```

### 3. Platform-Specific Authentication

#### macOS Keychain (`macos_keychain.py`)
- Integration with macOS Keychain Services
- Secure credential storage
- Touch ID integration

#### Windows Hello (`windows_hello.py`) 
- Windows Hello biometric authentication
- Windows Credential Manager integration
- PIN and biometric fallbacks

### 4. Device Authentication Manager (`device_auth.py`)
- Cross-platform device authentication
- Automatic platform detection
- Unified authentication interface

## Integration with MCP Server

### FastAPI Integration

```python
from fastapi import FastAPI
from auth.oauth2_server import OAuth2Server, OAuth2Middleware

app = FastAPI()

# Create OAuth 2.0 server
oauth2_server = OAuth2Server()

# Add OAuth 2.0 middleware
oauth2_middleware = OAuth2Middleware(oauth2_server)
app.middleware("http")(oauth2_middleware)

# Setup OAuth endpoints
@app.get("/oauth/authorize")
async def authorize(request: Request):
    return await oauth2_server.authorization_endpoint(request)

@app.post("/oauth/token")
async def token(request: Request):
    return await oauth2_server.token_endpoint(request)

# Protected endpoint
@app.get("/mcp/tools")
async def list_tools(request: Request):
    # Token validation is automatic via middleware
    token_info = request.state.oauth_token
    return {"tools": [...], "user": token_info["user_id"]}
```

### Token Validation

The OAuth 2.0 server provides automatic Bearer token validation:

```python
# Validate token manually
token_info = await oauth2_server.validate_bearer_token(
    authorization_header="Bearer your-jwt-token"
)

if token_info:
    user_id = token_info["user_id"]
    client_id = token_info["client_id"] 
    scopes = token_info["scope"].split()
```

## Security Considerations

### OAuth 2.0 Security
- **PKCE Required** - All public clients must use PKCE
- **Short-lived Access Tokens** - 1 hour expiration by default
- **Secure Token Storage** - In-memory storage (use Redis/database in production)
- **Proper Error Handling** - No information leakage in error responses
- **CORS Configuration** - Proper origin validation

### WebAuthn Security
- **User Verification Required** - Biometric/PIN authentication mandatory
- **Platform Authenticators** - Built-in device authenticators preferred
- **Challenge Expiration** - 5-minute challenge timeout
- **Credential Verification** - Proper attestation validation

### General Security
- **HTTPS Required** - All authentication flows require TLS
- **Secure Headers** - Cache-Control, Pragma headers set
- **Session Management** - Secure session token generation
- **Input Validation** - Proper parameter validation and sanitization

## Development and Testing

### Requirements
```bash
pip install PyJWT cryptography fastapi uvicorn
```

### Running the Demo
```bash
# Start the OAuth 2.0 integration demo
python examples/oauth2_integration.py

# Visit http://localhost:8080/demo for interactive demo
```

### Testing OAuth Flow

1. **Authorization Request**:
   ```
   GET /oauth/authorize?response_type=code&client_id=mcp-default-client&redirect_uri=http://localhost:8080/callback&scope=mcp:read+mcp:write&code_challenge=abc123&code_challenge_method=S256
   ```

2. **Token Exchange**:
   ```
   POST /oauth/token
   Content-Type: application/x-www-form-urlencoded
   
   grant_type=authorization_code&code=auth_code&redirect_uri=http://localhost:8080/callback&client_id=mcp-default-client&code_verifier=xyz789
   ```

3. **Protected Resource Access**:
   ```
   GET /mcp/tools
   Authorization: Bearer your-jwt-token
   ```

## Production Deployment

### Database Storage
Replace in-memory storage with persistent storage:

```python
# Use Redis for token storage
import redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Use PostgreSQL for client registration
from sqlalchemy import create_engine
engine = create_engine("postgresql://user:pass@localhost/oauth2")
```

### Key Management
Use proper key management for JWT signing:

```python
# Load RSA keys from secure storage
with open('/etc/ssl/private/oauth2-key.pem', 'rb') as f:
    private_key = serialization.load_pem_private_key(f.read(), password=None)
```

### Environment Configuration
```bash
# Environment variables
export OAUTH2_ISSUER="https://your-domain.com"
export OAUTH2_JWT_PRIVATE_KEY_PATH="/etc/ssl/private/oauth2-key.pem"
export OAUTH2_REDIS_URL="redis://localhost:6379/0"
export WEBAUTHN_RP_ID="your-domain.com"
export WEBAUTHN_RP_NAME="Your App Name"
```

## Standards Compliance

This implementation follows these standards:
- **RFC 6749** - OAuth 2.0 Authorization Framework
- **RFC 7636** - PKCE Extension for OAuth 2.0
- **RFC 7519** - JSON Web Token (JWT)
- **RFC 7662** - OAuth 2.0 Token Introspection
- **RFC 7009** - OAuth 2.0 Token Revocation
- **WebAuthn Level 2** - Web Authentication Standard

## Troubleshooting

### Common Issues

1. **"PyJWT not found"** - Install JWT dependencies:
   ```bash
   pip install PyJWT cryptography
   ```

2. **"Invalid code_verifier"** - Ensure PKCE implementation matches S256:
   ```javascript
   // Correct PKCE implementation
   const codeVerifier = base64URLEncode(crypto.getRandomValues(new Uint8Array(32)));
   const codeChallenge = base64URLEncode(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(codeVerifier)));
   ```

3. **"WebAuthn ceremony failed"** - Check browser compatibility:
   - Use HTTPS (required for WebAuthn)
   - Ensure authenticator is available
   - Check console for JavaScript errors

### Debug Mode
Enable debug logging for OAuth 2.0 flows:

```python
import logging
logging.getLogger("auth.oauth2_server").setLevel(logging.DEBUG)
```

## Contributing

When contributing to the authentication module:

1. Follow RFC specifications exactly
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure security best practices are followed
5. Test with real authenticators and browsers

## License

This authentication module is part of the Zen MCP Server project and follows the same license terms.