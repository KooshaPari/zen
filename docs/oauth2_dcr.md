# OAuth 2.0 Dynamic Client Registration (DCR) - RFC 7591

This document describes the implementation of RFC 7591 compliant Dynamic Client Registration (DCR) for the Zen MCP Server. The DCR system provides secure, automated client registration and management capabilities for OAuth 2.0 clients.

## Overview

The DCR implementation provides:

- **RFC 7591 Compliance**: Full compliance with Dynamic Client Registration specification
- **Client Registration**: Automated client registration with secure credential generation
- **Client Management**: CRUD operations for registered clients
- **MCP Extensions**: Support for MCP-specific client capabilities and metadata
- **Security Features**: Rate limiting, audit logging, encrypted storage, and secure credential management
- **Integration**: Seamless integration with existing Zen MCP Server infrastructure

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   OAuth Client  │───▶│   DCR Endpoints  │───▶│   DCR Manager   │
│                 │    │                  │    │                 │
│ - Registration  │    │ POST /oauth/     │    │ - Validation    │
│ - Management    │    │      register    │    │ - Storage       │
│ - Authentication│    │ GET/PUT/DELETE   │    │ - Rate Limiting │
│                 │    │ /oauth/register/ │    │ - Audit Logging │
│                 │    │ {client_id}      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Client Store   │
                       │                  │
                       │ - Encrypted      │
                       │   Storage        │
                       │ - Redis/Memory   │
                       │ - TTL Support    │
                       └──────────────────┘
```

## Features

### RFC 7591 Compliant Features

- **Dynamic Registration**: Clients can register themselves without manual intervention
- **Client Metadata**: Support for all standard OAuth 2.0 client metadata fields
- **Registration Access Token**: Secure tokens for client management operations
- **Client Authentication**: Multiple authentication methods (client_secret_basic, client_secret_post, none)
- **Grant Type Support**: Authorization code, client credentials, refresh token flows
- **Error Handling**: Proper OAuth 2.0 error responses with standard error codes

### Security Features

- **Secure Credential Generation**: Cryptographically secure client IDs and secrets
- **Registration Rate Limiting**: IP-based rate limiting to prevent abuse
- **Encrypted Storage**: Optional encryption of client data at rest
- **Audit Logging**: Comprehensive audit trail for all client operations
- **Token-based Access Control**: Registration access tokens for client management
- **Metadata Validation**: Strict validation of client metadata and security parameters

### MCP-Specific Extensions

- **MCP Capabilities**: Clients can specify MCP capabilities (tools, resources, prompts)
- **Transport Protocols**: Support for MCP transport protocol preferences (HTTP, WebSocket)
- **Session Configuration**: MCP-specific session timeout and behavior settings
- **Integration Ready**: Pre-configured for Zen MCP Server integration

## API Endpoints

### Client Registration

```http
POST /oauth/register
Content-Type: application/json

{
  "client_name": "My Application",
  "client_uri": "https://myapp.example.com",
  "redirect_uris": ["https://myapp.example.com/callback"],
  "client_type": "confidential",
  "token_endpoint_auth_method": "client_secret_basic",
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "scope": "read write mcp:tools",
  "mcp_capabilities": ["tools", "resources"],
  "mcp_transport_protocols": ["http"]
}
```

**Response (201 Created):**
```json
{
  "client_id": "zen_mcp_1640995200_a1b2c3d4e5f6",
  "client_secret": "GzfUQ8kLm9N2pRs4TvWx7YzAbCdEfGhI",
  "client_secret_expires_at": 1672531200,
  "client_id_issued_at": 1640995200,
  "registration_access_token": "eyJhbGciOiJIUzI1NiIs...",
  "registration_client_uri": "/oauth/register/zen_mcp_1640995200_a1b2c3d4e5f6",
  "client_name": "My Application",
  "client_uri": "https://myapp.example.com",
  "redirect_uris": ["https://myapp.example.com/callback"],
  "client_type": "confidential",
  "token_endpoint_auth_method": "client_secret_basic",
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "scope": "read write mcp:tools",
  "mcp_capabilities": ["tools", "resources"],
  "mcp_transport_protocols": ["http"]
}
```

### Client Configuration Retrieval

```http
GET /oauth/register/{client_id}
Authorization: Bearer {registration_access_token}
```

**Response (200 OK):**
```json
{
  "client_id": "zen_mcp_1640995200_a1b2c3d4e5f6",
  "client_id_issued_at": 1640995200,
  "client_secret_expires_at": 1672531200,
  "client_name": "My Application",
  "client_type": "confidential",
  "grant_types": ["authorization_code", "refresh_token"],
  "scope": "read write mcp:tools",
  "created_at": "2021-12-31T12:00:00Z",
  "updated_at": "2021-12-31T12:00:00Z",
  "is_active": true
}
```

### Client Configuration Update

```http
PUT /oauth/register/{client_id}
Authorization: Bearer {registration_access_token}
Content-Type: application/json

{
  "client_name": "Updated Application Name",
  "scope": "read write admin mcp:tools mcp:resources"
}
```

### Client Deletion

```http
DELETE /oauth/register/{client_id}
Authorization: Bearer {registration_access_token}
```

**Response:** 204 No Content

### Discovery Metadata

```http
GET /oauth/discovery
```

**Response:**
```json
{
  "issuer": "https://zen-mcp.example.com",
  "registration_endpoint": "https://zen-mcp.example.com/oauth/register",
  "grant_types_supported": ["authorization_code", "client_credentials", "refresh_token"],
  "response_types_supported": ["code"],
  "token_endpoint_auth_methods_supported": ["client_secret_post", "client_secret_basic", "none"],
  "client_id_issued_at_supported": true,
  "client_secret_expires_at_supported": true,
  "registration_access_token_supported": true,
  "mcp_extensions_supported": ["mcp_capabilities", "mcp_transport_protocols", "mcp_session_timeout"]
}
```

## Client Types and Authentication Methods

### Confidential Clients

Confidential clients can securely store client credentials:

```json
{
  "client_type": "confidential",
  "token_endpoint_auth_method": "client_secret_basic",
  "grant_types": ["authorization_code", "refresh_token"]
}
```

**Supported Authentication Methods:**
- `client_secret_basic`: HTTP Basic authentication
- `client_secret_post`: Client credentials in POST body

### Public Clients

Public clients cannot securely store credentials:

```json
{
  "client_type": "public", 
  "token_endpoint_auth_method": "none",
  "grant_types": ["authorization_code"]
}
```

Public clients must use PKCE (Proof Key for Code Exchange) for security.

## MCP-Specific Extensions

### MCP Capabilities

Specify which MCP capabilities the client supports:

```json
{
  "mcp_capabilities": ["tools", "resources", "prompts", "logging", "sampling"]
}
```

**Available Capabilities:**
- `tools`: MCP tool execution
- `resources`: MCP resource access
- `prompts`: MCP prompt templates
- `logging`: MCP operation logging
- `sampling`: MCP request sampling

### Transport Protocols

Specify preferred MCP transport protocols:

```json
{
  "mcp_transport_protocols": ["http", "websocket", "stdio"]
}
```

### Session Configuration

Configure MCP session behavior:

```json
{
  "mcp_session_timeout": 3600
}
```

## Error Handling

The DCR system returns standard OAuth 2.0 error responses:

### Rate Limiting (429 Too Many Requests)
```json
{
  "error": "too_many_requests",
  "error_description": "Rate limit exceeded for client registration"
}
```

### Invalid Metadata (400 Bad Request)
```json
{
  "error": "invalid_client_metadata",
  "error_description": "Unsupported grant type: implicit"
}
```

### Invalid Token (401 Unauthorized)
```json
{
  "error": "invalid_token",
  "error_description": "Invalid registration access token"
}
```

### Client Not Found (404 Not Found)
```json
{
  "error": "invalid_client_id", 
  "error_description": "Client not found"
}
```

## Configuration

### Environment Variables

```bash
# Enable/disable DCR
OAUTH_DCR_ENABLED=true

# Enable admin endpoints
OAUTH_ADMIN_ENABLED=false

# Security settings
OAUTH_ENCRYPTION_KEY=your-encryption-key-here
OAUTH_REGISTRATION_SECRET=your-registration-secret-here

# Rate limiting
OAUTH_RATE_LIMIT_PER_IP=10
OAUTH_RATE_LIMIT_WINDOW=3600

# Client settings
OAUTH_CLIENT_SECRET_LIFETIME=31536000
OAUTH_ALLOW_LOCALHOST=true
```

### Configuration Validation

```python
from auth.oauth2_integration import validate_oauth_config

is_valid, warnings = validate_oauth_config()
for warning in warnings:
    print(f"Warning: {warning}")
```

## Integration

### FastAPI Integration

```python
from fastapi import FastAPI
from auth.oauth2_integration import setup_oauth_endpoints

app = FastAPI()

# Setup OAuth DCR endpoints
setup_oauth_endpoints(app, enable_dcr=True, enable_admin=False)
```

### Replace Minimal Stub

If you have an existing minimal DCR stub, replace it:

```python
from auth.oauth2_integration import replace_minimal_dcr_stub

success = replace_minimal_dcr_stub(app)
if success:
    print("Successfully upgraded to full RFC 7591 DCR")
```

### Direct Usage

```python
from auth.oauth2_dcr import get_dcr_manager, ClientMetadata, ClientType

# Get DCR manager
dcr_manager = get_dcr_manager()

# Register a client
metadata = ClientMetadata(
    client_name="My Client",
    client_type=ClientType.CONFIDENTIAL
)

registered_client = await dcr_manager.register_client(
    metadata=metadata,
    client_ip="192.168.1.100"
)
```

## Security Considerations

### Production Deployment

1. **Use HTTPS**: Always use HTTPS in production
2. **Set Secure Secrets**: Generate secure random keys for encryption and signing
3. **Configure Rate Limits**: Set appropriate rate limits for your environment
4. **Enable Audit Logging**: Monitor client registration and management activities
5. **Restrict Admin Access**: Enable admin endpoints only when needed with proper authentication
6. **Validate Redirect URIs**: Carefully validate redirect URIs to prevent open redirects

### Client Secret Management

- Client secrets are generated using cryptographically secure random generation
- Secrets are stored encrypted when encryption is enabled
- Secret expiration is enforced (default: 1 year)
- Secrets should be transmitted only over secure channels

### Rate Limiting

- IP-based rate limiting prevents registration abuse
- Default: 10 registrations per IP per hour
- Failed attempts count toward rate limit
- Rate limits are enforced using existing Zen MCP Server infrastructure

## Testing

### Unit Tests

```bash
# Run DCR-specific tests
python -m pytest tests/test_oauth2_dcr.py -v

# Run with coverage
python -m pytest tests/test_oauth2_dcr.py --cov=auth.oauth2_dcr --cov-report=html
```

### Demo Script

```bash
# Run comprehensive demo
python examples/oauth2_dcr_demo.py
```

### HTTP Client Testing

```bash
# Register a client
curl -X POST https://zen-mcp.example.com/oauth/register \
  -H "Content-Type: application/json" \
  -d '{
    "client_name": "Test Client",
    "redirect_uris": ["https://example.com/callback"]
  }'

# Get client info (use registration_access_token from registration response)
curl -X GET https://zen-mcp.example.com/oauth/register/{client_id} \
  -H "Authorization: Bearer {registration_access_token}"
```

## Monitoring and Observability

### Audit Logs

All DCR operations are logged to the audit trail:

```python
from utils.audit_trail import get_audit_manager

audit_manager = await get_audit_manager()
entries = await audit_manager.query_audit_trail(
    category=AuditEventCategory.AUTHENTICATION,
    limit=100
)
```

### Metrics

Monitor these key metrics:
- Client registration rate
- Registration failures
- Rate limit violations
- Client authentication attempts
- Token verification failures

### Health Checks

```http
GET /healthz
```

Response includes DCR status:
```json
{
  "status": "ok",
  "oauth_dcr_enabled": true,
  "registered_clients_count": 42
}
```

## Compliance

### RFC 7591 Compliance

The implementation is fully compliant with RFC 7591:
- ✅ Client registration endpoint
- ✅ Client information endpoint  
- ✅ Client configuration endpoint
- ✅ Registration access tokens
- ✅ Proper error responses
- ✅ Metadata validation

### Security Standards

- **NIST Cybersecurity Framework**: Audit logging and access control
- **OWASP**: Rate limiting and secure credential management
- **RFC 6749**: OAuth 2.0 security considerations
- **RFC 6819**: OAuth 2.0 threat model and security considerations

## Troubleshooting

### Common Issues

**Client Registration Fails**
- Check client metadata validation
- Verify redirect URI format
- Check rate limiting status

**Client Authentication Fails**
- Verify client secret is correct
- Check client type vs auth method consistency
- Ensure client secret hasn't expired

**Token Verification Fails**
- Check registration access token format
- Verify token signature
- Check token expiration (24 hours)

### Debug Mode

Enable detailed logging:

```bash
LOG_LEVEL=DEBUG OAUTH_DCR_ENABLED=true python server_mcp_http.py
```

### Configuration Issues

```python
from auth.oauth2_integration import validate_oauth_config

is_valid, warnings = validate_oauth_config()
print(f"Valid: {is_valid}")
for warning in warnings:
    print(f"Warning: {warning}")
```

## Future Enhancements

- **JWT Client Assertion**: Support for RFC 7523 JWT client authentication
- **Software Statement**: Support for RFC 7591 software statements
- **Client Metadata Policies**: Configurable client metadata policies
- **Federation Support**: Integration with OAuth 2.0 federation protocols
- **Advanced Rate Limiting**: Per-client and global rate limiting
- **Client Lifecycle Management**: Automated client expiration and renewal

## Support

For issues and questions:
- Check the logs: `tail -f logs/mcp_server.log`
- Run diagnostics: `python examples/oauth2_dcr_demo.py`
- Review audit trail for security events
- Validate configuration settings

## References

- [RFC 7591 - OAuth 2.0 Dynamic Client Registration Protocol](https://tools.ietf.org/html/rfc7591)
- [RFC 7592 - OAuth 2.0 Dynamic Client Registration Management Protocol](https://tools.ietf.org/html/rfc7592) 
- [RFC 6749 - The OAuth 2.0 Authorization Framework](https://tools.ietf.org/html/rfc6749)
- [RFC 8414 - OAuth 2.0 Authorization Server Metadata](https://tools.ietf.org/html/rfc8414)
- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)