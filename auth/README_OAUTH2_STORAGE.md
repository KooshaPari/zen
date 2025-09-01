# OAuth 2.0 Encrypted Storage Layer

A comprehensive, secure storage layer for OAuth 2.0 data persistence with SQLite backend and AES-256 encryption.

## Overview

The OAuth 2.0 Encrypted Storage Layer provides enterprise-grade security for OAuth 2.0 authorization servers with:

- **AES-256-GCM Encryption**: All sensitive data encrypted at rest
- **PBKDF2 Key Derivation**: 100,000 iterations with SHA-256
- **HMAC Integrity Verification**: Tamper-evident storage with SHA-256
- **SQLite Backend**: ACID compliance with WAL mode
- **Comprehensive Audit Trail**: Complete operation logging
- **Automatic Schema Migration**: Seamless database evolution
- **Transaction Support**: Multi-operation consistency
- **Secure Token Management**: Hash-based storage with replay protection
- **System Keychain Integration**: macOS Keychain support (optional)

## Quick Start

### Installation

```bash
pip install cryptography pydantic
```

### Basic Usage

```python
from auth.oauth2_storage import get_oauth2_storage, ClientType, GrantType

# Initialize storage (singleton)
storage = get_oauth2_storage(
    db_path="oauth2.db",
    enable_audit=True
)

# Register OAuth2 client
client = storage.register_client(
    client_name="My Web App",
    client_type=ClientType.CONFIDENTIAL,
    redirect_uris=["https://myapp.com/callback"],
    grant_types=[GrantType.AUTHORIZATION_CODE, GrantType.REFRESH_TOKEN],
    scope="read write profile"
)

# Create authorization code
auth_code = storage.create_authorization_code(
    client_id=client.client_id,
    user_id="user123",
    redirect_uri="https://myapp.com/callback",
    scope="read write"
)

# Exchange for access token
access_token = storage.create_access_token(
    client_id=client.client_id,
    user_id="user123",
    scope="read write",
    expires_in=3600
)

# Create refresh token
refresh_token = storage.create_refresh_token(
    access_token_id=access_token.token_id,
    client_id=client.client_id,
    user_id="user123",
    scope="read write"
)
```

## Core Components

### Data Models

#### OAuth2Client
Represents an OAuth 2.0 client application.
```python
@dataclass
class OAuth2Client:
    client_id: str                    # Unique client identifier
    client_secret: Optional[str]      # Secret for confidential clients
    client_name: str                  # Human-readable name
    client_type: ClientType           # public | confidential
    redirect_uris: List[str]          # Allowed redirect URIs
    grant_types: List[GrantType]      # Supported grant types
    scope: str                        # Default scope
    created_at: datetime              # Registration timestamp
    updated_at: datetime              # Last update timestamp
    metadata: Dict[str, Any]          # Additional client data
```

#### AuthorizationCode
Represents an OAuth 2.0 authorization code.
```python
@dataclass
class AuthorizationCode:
    code: str                         # Authorization code value
    client_id: str                    # Associated client
    user_id: str                      # Authorized user
    redirect_uri: str                 # Callback URI
    scope: str                        # Requested scope
    code_challenge: Optional[str]     # PKCE challenge
    code_challenge_method: Optional[str]  # PKCE method
    expires_at: datetime              # Expiration time
    created_at: datetime              # Issue time
    used: bool                        # Usage status
    metadata: Dict[str, Any]          # Additional data
```

#### AccessToken
Represents an OAuth 2.0 access token.
```python
@dataclass
class AccessToken:
    token_id: str                     # Internal token identifier
    access_token: str                 # Token value (ephemeral)
    client_id: str                    # Associated client
    user_id: str                      # Token owner
    scope: str                        # Granted scope
    token_type: str                   # Token type (Bearer)
    expires_at: Optional[datetime]    # Expiration time
    created_at: Optional[datetime]    # Issue time
    revoked: bool                     # Revocation status
    metadata: Dict[str, Any]          # Additional data
```

#### RefreshToken
Represents an OAuth 2.0 refresh token.
```python
@dataclass
class RefreshToken:
    token_id: str                     # Internal token identifier
    refresh_token: str                # Token value (ephemeral)
    access_token_id: str              # Associated access token
    client_id: str                    # Associated client
    user_id: str                      # Token owner
    scope: str                        # Granted scope
    expires_at: Optional[datetime]    # Expiration time
    created_at: Optional[datetime]    # Issue time
    revoked: bool                     # Revocation status
    metadata: Dict[str, Any]          # Additional data
```

#### OAuth2Session
Represents an OAuth 2.0 authorization session.
```python
@dataclass
class OAuth2Session:
    session_id: str                   # Session identifier
    state: str                        # OAuth state parameter
    client_id: str                    # Associated client
    user_id: Optional[str]            # Authenticated user
    redirect_uri: str                 # Callback URI
    scope: str                        # Requested scope
    code_challenge: Optional[str]     # PKCE challenge
    code_challenge_method: Optional[str]  # PKCE method
    expires_at: datetime              # Session expiration
    created_at: datetime              # Creation time
    completed: bool                   # Completion status
    metadata: Dict[str, Any]          # Session data
```

### Storage Operations

#### Client Management
```python
# Register new client
client = storage.register_client(
    client_name="Example App",
    client_type=ClientType.CONFIDENTIAL,
    redirect_uris=["https://app.example.com/callback"],
    grant_types=[GrantType.AUTHORIZATION_CODE],
    scope="read write"
)

# Retrieve client
client = storage.get_client(client_id)

# Update client
storage.update_client(
    client_id,
    client_name="Updated App Name",
    redirect_uris=["https://newapp.example.com/callback"]
)

# List clients
clients = storage.list_clients(limit=50, offset=0)

# Delete client (cascades to tokens)
storage.delete_client(client_id)
```

#### Authorization Code Flow
```python
# Create session
session = storage.create_session(
    state="random_state",
    client_id=client_id,
    redirect_uri="https://app.example.com/callback",
    scope="read write"
)

# Update session with user
storage.update_session(
    session.session_id,
    user_id="user123"
)

# Issue authorization code
auth_code = storage.create_authorization_code(
    client_id=client_id,
    user_id="user123",
    redirect_uri="https://app.example.com/callback",
    scope="read write",
    expires_in=600  # 10 minutes
)

# Verify and use code (one-time use)
code_valid = storage.use_authorization_code(auth_code.code)
```

#### Token Management
```python
# Create access token
access_token = storage.create_access_token(
    client_id=client_id,
    user_id="user123",
    scope="read write",
    expires_in=3600  # 1 hour
)

# Validate access token
token = storage.get_access_token(access_token.access_token)
if token and not token.revoked and token.expires_at > datetime.now(timezone.utc):
    # Token is valid
    print(f"Valid token for user: {token.user_id}")

# Create refresh token
refresh_token = storage.create_refresh_token(
    access_token_id=access_token.token_id,
    client_id=client_id,
    user_id="user123",
    scope="read write",
    expires_in=30*24*3600  # 30 days
)

# Use refresh token
refresh_token_obj = storage.use_refresh_token(refresh_token.refresh_token)
if refresh_token_obj:
    # Create new access token
    new_access_token = storage.create_access_token(
        client_id=refresh_token_obj.client_id,
        user_id=refresh_token_obj.user_id,
        scope=refresh_token_obj.scope
    )

# Revoke tokens
storage.revoke_access_token(access_token.access_token)
storage.revoke_refresh_token(refresh_token.refresh_token)
```

#### Transaction Support
```python
from auth.oauth2_storage import oauth2_transaction

# Atomic operations
with oauth2_transaction(storage):
    client = storage.register_client(...)
    access_token = storage.create_access_token(
        client_id=client.client_id,
        user_id="user123",
        scope="read"
    )
    # Both operations succeed or fail together
```

### Security Features

#### Encryption
All sensitive data is encrypted using AES-256-GCM:
- Client secrets
- User identifiers
- Redirect URIs
- Scopes
- PKCE challenges
- Session state
- Audit details

#### Key Management
```python
# Custom encryption key
storage = OAuth2EncryptedStorage(
    encryption_key="your-secret-key-here"
)

# Environment variable
export OAUTH2_ENCRYPTION_KEY="your-secret-key"

# macOS Keychain (automatic)
# Key stored securely in system keychain
```

#### Token Security
- Access tokens stored as SHA-256 hashes (non-reversible)
- Refresh tokens stored as SHA-256 hashes (non-reversible)
- Authorization codes are single-use with replay protection
- Secure random generation using `secrets` module

#### Data Integrity
- HMAC-SHA256 verification for all stored records
- Prevents tampering and detects corruption
- Uses timing-safe comparison to prevent timing attacks

### Audit and Monitoring

#### Audit Events
The system logs comprehensive audit events:
- `CLIENT_REGISTERED` - New client registration
- `CLIENT_UPDATED` - Client configuration changes
- `CLIENT_DELETED` - Client removal
- `AUTH_CODE_ISSUED` - Authorization code generation
- `AUTH_CODE_USED` - Authorization code consumption
- `ACCESS_TOKEN_ISSUED` - Access token creation
- `ACCESS_TOKEN_REVOKED` - Access token revocation
- `REFRESH_TOKEN_ISSUED` - Refresh token creation
- `REFRESH_TOKEN_USED` - Refresh token consumption
- `REFRESH_TOKEN_REVOKED` - Refresh token revocation
- `SESSION_CREATED` - OAuth session initiation
- `SESSION_COMPLETED` - OAuth session completion
- `CLEANUP_PERFORMED` - Automatic data cleanup
- `SECURITY_VIOLATION` - Security policy violations

#### Audit Queries
```python
# Get recent audit logs
logs = storage.get_audit_logs(
    limit=100,
    event_type="ACCESS_TOKEN_ISSUED",
    client_id="client_123",
    start_time=datetime.now(timezone.utc) - timedelta(days=1)
)

# Analyze audit data
for log in logs:
    print(f"{log['timestamp']}: {log['event_type']} - {log['entity_id']}")
    print(f"  User: {log['user_id']}")
    print(f"  Details: {log['details']}")
```

#### Storage Statistics
```python
stats = storage.get_storage_stats()
print(f"Clients: {stats['clients']}")
print(f"Active tokens: {stats['active_access_tokens']}")
print(f"Audit logs: {stats['audit_logs']}")
print(f"Database size: {stats['database_size_bytes']} bytes")
```

### Data Lifecycle

#### Automatic Cleanup
The storage system automatically cleans up expired data:
- Expired authorization codes
- Expired access tokens
- Expired refresh tokens
- Expired sessions
- Old audit logs (configurable retention)

```python
# Manual cleanup
cleanup_result = storage.cleanup_expired_data()
print(f"Cleaned up: {cleanup_result}")

# Configure cleanup interval
storage = OAuth2EncryptedStorage(
    cleanup_interval=1800  # 30 minutes
)
```

#### Data Retention
Configure retention policies:
```python
# Cleanup old audit logs (default: 6 months)
# Modify cleanup_expired_data() to adjust retention
```

### Database Schema

The storage uses a comprehensive SQLite schema with:

#### Tables
- `oauth2_clients` - OAuth2 client registrations
- `authorization_codes` - Authorization codes with expiration
- `access_tokens` - Access tokens with hash storage
- `refresh_tokens` - Refresh tokens with relationships
- `oauth2_sessions` - Authorization sessions
- `oauth2_audit_log` - Complete audit trail
- `schema_version` - Database migration tracking

#### Constraints
- Foreign key constraints for data consistency
- Check constraints for valid enum values
- Unique constraints for token hashes
- NOT NULL constraints for required fields

#### Indices
- Performance indices on frequently queried fields
- Expiration time indices for efficient cleanup
- Client relationship indices for fast lookups

## Configuration

### Environment Variables
```bash
# Encryption key (optional)
export OAUTH2_ENCRYPTION_KEY="your-secret-key-here"

# Database location (optional)
export OAUTH2_DB_PATH="/path/to/oauth2.db"

# Enable audit logging (optional)
export OAUTH2_ENABLE_AUDIT="true"
```

### Initialization Options
```python
storage = OAuth2EncryptedStorage(
    db_path="custom_oauth2.db",           # Custom database path
    encryption_key="custom-key",          # Custom encryption key
    enable_audit=True,                    # Enable audit logging
    cleanup_interval=3600                 # Cleanup every hour
)
```

## Integration Examples

### FastAPI Integration
```python
from fastapi import FastAPI, HTTPException, Depends
from auth.oauth2_storage import get_oauth2_storage, ClientType, GrantType

app = FastAPI()
storage = get_oauth2_storage()

@app.post("/oauth/register")
async def register_client(request: ClientRegistrationRequest):
    client = storage.register_client(
        client_name=request.client_name,
        client_type=ClientType.CONFIDENTIAL,
        redirect_uris=request.redirect_uris,
        grant_types=[GrantType.AUTHORIZATION_CODE, GrantType.REFRESH_TOKEN],
        scope=request.scope or "read"
    )
    
    return {
        "client_id": client.client_id,
        "client_secret": client.client_secret,
        "created_at": client.created_at.isoformat()
    }

@app.post("/oauth/token")
async def token_endpoint(request: TokenRequest):
    if request.grant_type == "authorization_code":
        # Verify authorization code
        auth_code = storage.get_authorization_code(request.code)
        if not auth_code or auth_code.used:
            raise HTTPException(400, "Invalid authorization code")
        
        # Mark code as used
        storage.use_authorization_code(request.code)
        
        # Issue tokens
        access_token = storage.create_access_token(
            client_id=auth_code.client_id,
            user_id=auth_code.user_id,
            scope=auth_code.scope
        )
        
        refresh_token = storage.create_refresh_token(
            access_token_id=access_token.token_id,
            client_id=auth_code.client_id,
            user_id=auth_code.user_id,
            scope=auth_code.scope
        )
        
        return {
            "access_token": access_token.access_token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": refresh_token.refresh_token,
            "scope": access_token.scope
        }
```

### Flask Integration
```python
from flask import Flask, request, jsonify
from auth.oauth2_storage import get_oauth2_storage

app = Flask(__name__)
storage = get_oauth2_storage()

@app.route("/oauth/authorize", methods=["GET", "POST"])
def authorize():
    if request.method == "GET":
        # Create authorization session
        session = storage.create_session(
            state=request.args.get("state"),
            client_id=request.args.get("client_id"),
            redirect_uri=request.args.get("redirect_uri"),
            scope=request.args.get("scope", "read")
        )
        
        # Redirect to login/consent page
        return redirect(f"/login?session={session.session_id}")
    
    elif request.method == "POST":
        # Handle consent
        session_id = request.form.get("session_id")
        user_id = request.form.get("user_id")  # From authentication
        
        session = storage.get_session(session_id)
        if not session:
            return jsonify({"error": "invalid_session"}), 400
        
        # Issue authorization code
        auth_code = storage.create_authorization_code(
            client_id=session.client_id,
            user_id=user_id,
            redirect_uri=session.redirect_uri,
            scope=session.scope
        )
        
        # Mark session as completed
        storage.update_session(session_id, user_id=user_id, completed=True)
        
        # Redirect back to client
        return redirect(f"{session.redirect_uri}?code={auth_code.code}&state={session.state}")
```

## Testing

### Validation Script
```bash
# Run comprehensive validation
python validate_oauth2_storage.py

# Run demo with sample data
python demo_oauth2_storage.py

# Run standalone tests (if available)
python test_oauth2_standalone.py
```

### Unit Testing
```python
import unittest
from auth.oauth2_storage import OAuth2EncryptedStorage, ClientType, GrantType

class TestOAuth2Storage(unittest.TestCase):
    def setUp(self):
        self.storage = OAuth2EncryptedStorage(
            db_path=":memory:",  # In-memory database
            encryption_key="test-key-12345"
        )
    
    def test_client_registration(self):
        client = self.storage.register_client(
            client_name="Test Client",
            client_type=ClientType.CONFIDENTIAL,
            redirect_uris=["https://test.com/callback"],
            grant_types=[GrantType.AUTHORIZATION_CODE],
            scope="read"
        )
        
        self.assertIsNotNone(client.client_id)
        self.assertIsNotNone(client.client_secret)
        
        # Verify retrieval
        retrieved = self.storage.get_client(client.client_id)
        self.assertEqual(retrieved.client_name, "Test Client")
```

## Production Deployment

### Security Checklist
- [ ] Use strong, randomly generated encryption keys
- [ ] Store encryption keys in secure key management system
- [ ] Enable audit logging for compliance
- [ ] Configure appropriate token expiration times
- [ ] Set up database backups
- [ ] Monitor database size and performance
- [ ] Implement log rotation for audit trails
- [ ] Use HTTPS for all OAuth endpoints
- [ ] Validate redirect URIs strictly
- [ ] Implement rate limiting
- [ ] Monitor for security violations

### Performance Optimization
```python
# Configure cleanup interval based on load
storage = OAuth2EncryptedStorage(
    cleanup_interval=3600,  # More frequent for high load
)

# Use connection pooling for high concurrency
# Consider read replicas for audit queries
# Index optimization for specific query patterns
```

### Backup and Recovery
```bash
# Backup database
cp oauth2.db oauth2.db.backup.$(date +%Y%m%d_%H%M%S)

# Restore database
cp oauth2.db.backup.20240101_120000 oauth2.db

# Export audit data
sqlite3 oauth2.db ".mode csv" ".once audit_export.csv" "SELECT * FROM oauth2_audit_log"
```

## Compliance and Standards

### OAuth 2.0 Compliance
- RFC 6749: The OAuth 2.0 Authorization Framework
- RFC 7636: Proof Key for Code Exchange (PKCE)
- RFC 6750: Bearer Token Usage
- RFC 7009: Token Revocation
- RFC 7662: Token Introspection (partial)

### Security Standards
- NIST SP 800-63B: Authentication and Lifecycle Management
- OWASP OAuth 2.0 Security Best Practices
- FIDO Alliance WebAuthn (for device authentication)

### Data Protection
- GDPR compliance through encryption and audit trails
- Right to erasure through secure deletion
- Data minimization through selective encryption
- Pseudonymization through token hashing

## Troubleshooting

### Common Issues

#### Import Errors
```
ImportError: No module named 'cryptography'
```
**Solution:** Install cryptography package
```bash
pip install cryptography
```

#### Database Permission Errors
```
sqlite3.OperationalError: unable to open database file
```
**Solution:** Check file permissions and directory access

#### Key Derivation Warnings
```
Using ephemeral encryption key - tokens will not survive restart
```
**Solution:** Configure persistent encryption key
```bash
export OAUTH2_ENCRYPTION_KEY="your-secure-key-here"
```

#### Integrity Verification Failures
```
OAuth2SecurityError: Client data integrity verification failed
```
**Solution:** Database corruption or key mismatch. Restore from backup.

### Debug Mode
```python
import logging
logging.getLogger('auth.oauth2_storage').setLevel(logging.DEBUG)

# Enable detailed logging
storage = OAuth2EncryptedStorage(enable_audit=True)
```

### Performance Monitoring
```python
# Monitor storage statistics
stats = storage.get_storage_stats()
print(f"Database size: {stats['database_size_bytes']} bytes")
print(f"Active tokens: {stats['active_access_tokens']}")

# Monitor cleanup performance
cleanup_result = storage.cleanup_expired_data()
print(f"Cleanup removed: {sum(cleanup_result.values())} items")
```

## API Reference

### OAuth2EncryptedStorage

Main storage class providing OAuth 2.0 data persistence.

#### Constructor
```python
OAuth2EncryptedStorage(
    db_path: Optional[str] = None,
    encryption_key: Optional[str] = None,
    enable_audit: bool = True,
    cleanup_interval: int = 3600
)
```

#### Client Methods
- `register_client()` - Register new OAuth2 client
- `get_client()` - Retrieve client by ID
- `update_client()` - Update client configuration
- `delete_client()` - Remove client and cascade data
- `list_clients()` - List registered clients

#### Authorization Code Methods
- `create_authorization_code()` - Issue authorization code
- `get_authorization_code()` - Retrieve authorization code
- `use_authorization_code()` - Mark code as used

#### Access Token Methods
- `create_access_token()` - Issue access token
- `get_access_token()` - Retrieve and validate token
- `revoke_access_token()` - Revoke access token

#### Refresh Token Methods
- `create_refresh_token()` - Issue refresh token
- `get_refresh_token()` - Retrieve refresh token
- `use_refresh_token()` - Consume refresh token
- `revoke_refresh_token()` - Revoke refresh token

#### Session Methods
- `create_session()` - Create authorization session
- `get_session()` - Retrieve session data
- `update_session()` - Update session state
- `delete_session()` - Remove session

#### Utility Methods
- `get_audit_logs()` - Query audit trail
- `get_storage_stats()` - Get storage statistics
- `cleanup_expired_data()` - Manual cleanup
- `shutdown()` - Graceful shutdown

### Global Functions
- `get_oauth2_storage()` - Get singleton storage instance
- `shutdown_oauth2_storage()` - Shutdown global storage
- `oauth2_transaction()` - Transaction context manager

## License

This implementation is part of the Zen MCP Server project and follows the same licensing terms.

## Support

For issues, questions, or contributions:
1. Check the validation script results
2. Review the troubleshooting section
3. Examine audit logs for security events
4. Verify encryption key configuration
5. Test with the demo script

The OAuth 2.0 Encrypted Storage Layer provides enterprise-grade security and compliance for OAuth 2.0 authorization servers while maintaining high performance and ease of use.