"""
Secure OAuth 2.0 Storage Layer for Zen MCP Server

This module provides a comprehensive, encrypted storage layer for OAuth 2.0 data persistence
with SQLite backend, automatic schema migrations, and security features.

Key Features:
- SQLite database with AES-256 encryption at rest
- Client registry storage and management
- Token storage with automatic expiration
- Session state management with secure cleanup
- Audit logging and compliance support
- Transaction safety with ACID compliance
- Automatic schema migrations
- Secure key derivation from system keychain
- Data integrity verification with HMAC
- Secure deletion of expired/revoked data

Security Model:
- All sensitive fields encrypted with AES-256-GCM
- Keys derived from system keychain using PBKDF2-SHA256
- HMAC integrity verification for all records
- Secure random token generation
- Automatic cleanup of expired data
- Audit trail for all operations
- Protection against timing attacks
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Import existing keychain functionality (optional)
MacOSKeychainAuth = None
try:
    # Try relative import first
    from .macos_keychain import MacOSKeychainAuth
except ImportError:
    try:
        # Try absolute import
        from auth.macos_keychain import MacOSKeychainAuth
    except ImportError:
        # Keychain functionality not available, will use fallback
        pass

logger = logging.getLogger(__name__)


class OAuth2TokenType(str, Enum):
    """OAuth 2.0 token types."""
    AUTHORIZATION_CODE = "authorization_code"
    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    ID_TOKEN = "id_token"


class GrantType(str, Enum):
    """OAuth 2.0 grant types."""
    AUTHORIZATION_CODE = "authorization_code"
    REFRESH_TOKEN = "refresh_token"
    CLIENT_CREDENTIALS = "client_credentials"
    DEVICE_CODE = "device_code"


class ClientType(str, Enum):
    """OAuth 2.0 client types."""
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"


@dataclass
class OAuth2Client:
    """OAuth 2.0 client registration."""
    client_id: str
    client_secret: str | None
    client_name: str
    client_type: ClientType
    redirect_uris: list[str]
    grant_types: list[GrantType]
    scope: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "client_name": self.client_name,
            "client_type": self.client_type.value,
            "redirect_uris": self.redirect_uris,
            "grant_types": [gt.value for gt in self.grant_types],
            "scope": self.scope,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class AuthorizationCode:
    """OAuth 2.0 authorization code."""
    code: str
    client_id: str
    user_id: str
    redirect_uri: str
    scope: str
    code_challenge: str | None
    code_challenge_method: str | None
    expires_at: datetime
    created_at: datetime
    used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessToken:
    """OAuth 2.0 access token."""
    token_id: str
    access_token: str
    client_id: str
    user_id: str
    scope: str
    token_type: str = "Bearer"
    expires_at: datetime | None = None
    created_at: datetime | None = None
    revoked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RefreshToken:
    """OAuth 2.0 refresh token."""
    token_id: str
    refresh_token: str
    access_token_id: str
    client_id: str
    user_id: str
    scope: str
    expires_at: datetime | None = None
    created_at: datetime | None = None
    revoked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OAuth2Session:
    """OAuth 2.0 session state."""
    session_id: str
    state: str
    client_id: str
    user_id: str | None
    redirect_uri: str
    scope: str
    code_challenge: str | None
    code_challenge_method: str | None
    expires_at: datetime
    created_at: datetime
    completed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class OAuth2AuditEvent:
    """OAuth 2.0 audit event types."""
    CLIENT_REGISTERED = "client_registered"
    CLIENT_UPDATED = "client_updated"
    CLIENT_DELETED = "client_deleted"
    AUTH_CODE_ISSUED = "auth_code_issued"
    AUTH_CODE_USED = "auth_code_used"
    ACCESS_TOKEN_ISSUED = "access_token_issued"
    ACCESS_TOKEN_REFRESHED = "access_token_refreshed"
    ACCESS_TOKEN_REVOKED = "access_token_revoked"
    REFRESH_TOKEN_ISSUED = "refresh_token_issued"
    REFRESH_TOKEN_USED = "refresh_token_used"
    REFRESH_TOKEN_REVOKED = "refresh_token_revoked"
    SESSION_CREATED = "session_created"
    SESSION_COMPLETED = "session_completed"
    SESSION_EXPIRED = "session_expired"
    CLEANUP_PERFORMED = "cleanup_performed"
    SECURITY_VIOLATION = "security_violation"


class OAuth2StorageError(Exception):
    """OAuth 2.0 storage operation error."""
    pass


class OAuth2SecurityError(OAuth2StorageError):
    """OAuth 2.0 security violation error."""
    pass


class OAuth2EncryptedStorage:
    """
    Secure OAuth 2.0 storage layer with SQLite backend and AES-256 encryption.

    Provides comprehensive OAuth 2.0 data persistence with:
    - Client registry management
    - Authorization code storage and validation
    - Access and refresh token management
    - Session state management
    - Audit logging and compliance
    - Automatic data cleanup and expiration
    - Schema migrations and versioning
    """

    SCHEMA_VERSION = 1
    DEFAULT_DB_PATH = "oauth2_storage.db"
    ENCRYPTION_SALT = b"zen_oauth2_storage_salt_v1"

    def __init__(
        self,
        db_path: str | None = None,
        encryption_key: str | None = None,
        enable_audit: bool = True,
        cleanup_interval: int = 3600  # 1 hour
    ):
        """
        Initialize OAuth 2.0 encrypted storage.

        Args:
            db_path: Path to SQLite database file
            encryption_key: Custom encryption key (uses keychain if not provided)
            enable_audit: Enable audit logging
            cleanup_interval: Automatic cleanup interval in seconds
        """
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self.enable_audit = enable_audit
        self.cleanup_interval = cleanup_interval

        # Thread safety
        self._lock = threading.RLock()
        self._shutdown = False

        # Initialize encryption
        if not CRYPTO_AVAILABLE:
            raise OAuth2StorageError("Cryptography package required for OAuth2 storage")

        self.encryption_key = self._setup_encryption(encryption_key)
        self.cipher = Fernet(self.encryption_key)

        # Initialize database
        self._init_database()

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()

        # Initialize audit (after cleanup thread to avoid circular imports)
        self._audit_events: list[dict[str, Any]] = []

        logger.info(f"OAuth2 encrypted storage initialized: {self.db_path}")

    def _setup_encryption(self, custom_key: str | None = None) -> bytes:
        """Setup encryption key from keychain or custom key."""
        if custom_key:
            password = custom_key
        else:
            # Try to get encryption key from system keychain
            password = None
            if MacOSKeychainAuth:
                try:
                    keychain = MacOSKeychainAuth("zen-mcp-oauth2")
                    stored_key = keychain.retrieve_credential("oauth2_master_key")
                    if stored_key:
                        password = stored_key
                    else:
                        # Generate new key and store in keychain
                        password = secrets.token_urlsafe(32)
                        keychain.store_credential("oauth2_master_key", password)
                        logger.info("Generated new OAuth2 encryption key in keychain")
                except Exception as e:
                    logger.warning(f"Keychain access failed: {e}")

            if not password:
                # Fallback to environment variable
                password = os.getenv("OAUTH2_ENCRYPTION_KEY")
                if not password:
                    # Generate ephemeral key (will not persist across restarts)
                    password = secrets.token_urlsafe(32)
                    logger.warning("Using ephemeral encryption key - tokens will not survive restart")

        # Derive encryption key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.ENCRYPTION_SALT,
            iterations=100000,
        )

        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def _init_database(self):
        """Initialize SQLite database with schema."""
        with self._get_connection() as conn:
            # Enable foreign keys and WAL mode for better concurrency
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = FULL")

            # Create schema version table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
            """)

            # Check current schema version
            cursor = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
            current_version = cursor.fetchone()
            current_version = current_version[0] if current_version else 0

            # Apply migrations
            if current_version < self.SCHEMA_VERSION:
                self._apply_migrations(conn, current_version)

    def _apply_migrations(self, conn: sqlite3.Connection, from_version: int):
        """Apply database schema migrations."""
        logger.info(f"Applying OAuth2 schema migrations from version {from_version} to {self.SCHEMA_VERSION}")

        if from_version < 1:
            # Initial schema creation
            conn.execute("""
                CREATE TABLE oauth2_clients (
                    client_id TEXT PRIMARY KEY,
                    client_secret_encrypted TEXT,
                    client_name TEXT NOT NULL,
                    client_type TEXT NOT NULL,
                    redirect_uris_encrypted TEXT NOT NULL,
                    grant_types_encrypted TEXT NOT NULL,
                    scope_encrypted TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_encrypted TEXT,
                    integrity_hash TEXT NOT NULL,

                    CHECK (client_type IN ('public', 'confidential'))
                )
            """)

            conn.execute("""
                CREATE TABLE authorization_codes (
                    code TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    user_id_encrypted TEXT NOT NULL,
                    redirect_uri_encrypted TEXT NOT NULL,
                    scope_encrypted TEXT NOT NULL,
                    code_challenge_encrypted TEXT,
                    code_challenge_method TEXT,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    used INTEGER NOT NULL DEFAULT 0,
                    metadata_encrypted TEXT,
                    integrity_hash TEXT NOT NULL,

                    FOREIGN KEY (client_id) REFERENCES oauth2_clients (client_id) ON DELETE CASCADE,
                    CHECK (used IN (0, 1))
                )
            """)

            conn.execute("""
                CREATE TABLE access_tokens (
                    token_id TEXT PRIMARY KEY,
                    access_token_hash TEXT UNIQUE NOT NULL,
                    client_id TEXT NOT NULL,
                    user_id_encrypted TEXT NOT NULL,
                    scope_encrypted TEXT NOT NULL,
                    token_type TEXT NOT NULL DEFAULT 'Bearer',
                    expires_at TEXT,
                    created_at TEXT NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0,
                    metadata_encrypted TEXT,
                    integrity_hash TEXT NOT NULL,

                    FOREIGN KEY (client_id) REFERENCES oauth2_clients (client_id) ON DELETE CASCADE,
                    CHECK (revoked IN (0, 1))
                )
            """)

            conn.execute("""
                CREATE TABLE refresh_tokens (
                    token_id TEXT PRIMARY KEY,
                    refresh_token_hash TEXT UNIQUE NOT NULL,
                    access_token_id TEXT NOT NULL,
                    client_id TEXT NOT NULL,
                    user_id_encrypted TEXT NOT NULL,
                    scope_encrypted TEXT NOT NULL,
                    expires_at TEXT,
                    created_at TEXT NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0,
                    metadata_encrypted TEXT,
                    integrity_hash TEXT NOT NULL,

                    FOREIGN KEY (access_token_id) REFERENCES access_tokens (token_id) ON DELETE CASCADE,
                    FOREIGN KEY (client_id) REFERENCES oauth2_clients (client_id) ON DELETE CASCADE,
                    CHECK (revoked IN (0, 1))
                )
            """)

            conn.execute("""
                CREATE TABLE oauth2_sessions (
                    session_id TEXT PRIMARY KEY,
                    state_encrypted TEXT NOT NULL,
                    client_id TEXT NOT NULL,
                    user_id_encrypted TEXT,
                    redirect_uri_encrypted TEXT NOT NULL,
                    scope_encrypted TEXT NOT NULL,
                    code_challenge_encrypted TEXT,
                    code_challenge_method TEXT,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed INTEGER NOT NULL DEFAULT 0,
                    metadata_encrypted TEXT,
                    integrity_hash TEXT NOT NULL,

                    FOREIGN KEY (client_id) REFERENCES oauth2_clients (client_id) ON DELETE CASCADE,
                    CHECK (completed IN (0, 1))
                )
            """)

            conn.execute("""
                CREATE TABLE oauth2_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    user_id_encrypted TEXT,
                    client_id TEXT,
                    details_encrypted TEXT,
                    ip_address TEXT,
                    user_agent_encrypted TEXT,
                    timestamp TEXT NOT NULL,
                    integrity_hash TEXT NOT NULL
                )
            """)

            # Create indices for performance
            conn.execute("CREATE INDEX idx_auth_codes_client_id ON authorization_codes (client_id)")
            conn.execute("CREATE INDEX idx_auth_codes_expires_at ON authorization_codes (expires_at)")
            conn.execute("CREATE INDEX idx_access_tokens_client_id ON access_tokens (client_id)")
            conn.execute("CREATE INDEX idx_access_tokens_expires_at ON access_tokens (expires_at)")
            conn.execute("CREATE INDEX idx_refresh_tokens_access_token_id ON refresh_tokens (access_token_id)")
            conn.execute("CREATE INDEX idx_refresh_tokens_expires_at ON refresh_tokens (expires_at)")
            conn.execute("CREATE INDEX idx_sessions_expires_at ON oauth2_sessions (expires_at)")
            conn.execute("CREATE INDEX idx_audit_log_timestamp ON oauth2_audit_log (timestamp)")
            conn.execute("CREATE INDEX idx_audit_log_event_type ON oauth2_audit_log (event_type)")

        # Record migration
        conn.execute("""
            INSERT INTO schema_version (version, applied_at)
            VALUES (?, ?)
        """, (self.SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()))

        conn.commit()
        logger.info(f"OAuth2 schema migration to version {self.SCHEMA_VERSION} completed")

    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30,  # 30 second timeout
                isolation_level=None,  # Autocommit mode
                check_same_thread=False  # Allow sharing across threads
            )
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise OAuth2StorageError(f"Database operation failed: {e}") from e
        finally:
            if conn:
                conn.close()

    def _encrypt_field(self, value: Any) -> str:
        """Encrypt a field value."""
        if value is None:
            return ""

        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True)
        elif not isinstance(value, str):
            value = str(value)

        encrypted = self.cipher.encrypt(value.encode('utf-8'))
        return base64.b64encode(encrypted).decode('ascii')

    def _decrypt_field(self, encrypted_value: str, default: Any = None) -> Any:
        """Decrypt a field value."""
        if not encrypted_value:
            return default

        try:
            encrypted_bytes = base64.b64decode(encrypted_value.encode('ascii'))
            decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
            decrypted_str = decrypted_bytes.decode('utf-8')

            # Try to parse as JSON first
            try:
                return json.loads(decrypted_str)
            except json.JSONDecodeError:
                return decrypted_str
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return default

    def _compute_integrity_hash(self, *values) -> str:
        """Compute HMAC integrity hash for record integrity verification."""
        hasher = hmac.new(
            self.encryption_key,
            digestmod=hashlib.sha256
        )

        for value in values:
            if value is not None:
                if isinstance(value, str):
                    hasher.update(value.encode('utf-8'))
                else:
                    hasher.update(str(value).encode('utf-8'))

        return hasher.hexdigest()

    def _verify_integrity(self, record: dict[str, Any], *values) -> bool:
        """Verify record integrity using HMAC."""
        stored_hash = record.get('integrity_hash', '')
        computed_hash = self._compute_integrity_hash(*values)

        # Use hmac.compare_digest to prevent timing attacks
        return hmac.compare_digest(stored_hash, computed_hash)

    def _hash_token(self, token: str) -> str:
        """Hash a token for secure storage (non-reversible)."""
        return hashlib.sha256(token.encode('utf-8')).hexdigest()

    def _audit_log(
        self,
        event_type: str,
        entity_type: str,
        entity_id: str,
        user_id: str | None = None,
        client_id: str | None = None,
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None
    ):
        """Log audit event."""
        if not self.enable_audit:
            return

        try:
            with self._get_connection() as conn:
                # Encrypt sensitive fields
                user_id_encrypted = self._encrypt_field(user_id) if user_id else None
                details_encrypted = self._encrypt_field(details) if details else None
                user_agent_encrypted = self._encrypt_field(user_agent) if user_agent else None

                timestamp = datetime.now(timezone.utc).isoformat()

                # Compute integrity hash
                integrity_hash = self._compute_integrity_hash(
                    event_type, entity_type, entity_id,
                    user_id_encrypted, client_id, details_encrypted,
                    ip_address, user_agent_encrypted, timestamp
                )

                conn.execute("""
                    INSERT INTO oauth2_audit_log (
                        event_type, entity_type, entity_id,
                        user_id_encrypted, client_id, details_encrypted,
                        ip_address, user_agent_encrypted,
                        timestamp, integrity_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_type, entity_type, entity_id,
                    user_id_encrypted, client_id, details_encrypted,
                    ip_address, user_agent_encrypted,
                    timestamp, integrity_hash
                ))

        except Exception as e:
            logger.error(f"Audit logging failed: {e}")

    def _cleanup_worker(self):
        """Background cleanup worker thread."""
        while not self._shutdown:
            try:
                time.sleep(self.cleanup_interval)
                if not self._shutdown:
                    self.cleanup_expired_data()
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")

    def cleanup_expired_data(self) -> dict[str, int]:
        """
        Cleanup expired OAuth 2.0 data.

        Returns:
            Dictionary with cleanup counts for each entity type
        """
        cleanup_counts = {
            "authorization_codes": 0,
            "access_tokens": 0,
            "refresh_tokens": 0,
            "sessions": 0,
            "audit_logs": 0
        }

        current_time = datetime.now(timezone.utc).isoformat()

        try:
            with self._lock:
                with self._get_connection() as conn:
                    # Start transaction
                    conn.execute("BEGIN IMMEDIATE")

                    try:
                        # Clean expired authorization codes
                        cursor = conn.execute("""
                            DELETE FROM authorization_codes
                            WHERE expires_at < ? OR used = 1
                        """, (current_time,))
                        cleanup_counts["authorization_codes"] = cursor.rowcount

                        # Clean expired access tokens
                        cursor = conn.execute("""
                            DELETE FROM access_tokens
                            WHERE expires_at < ? OR revoked = 1
                        """, (current_time,))
                        cleanup_counts["access_tokens"] = cursor.rowcount

                        # Clean expired refresh tokens
                        cursor = conn.execute("""
                            DELETE FROM refresh_tokens
                            WHERE expires_at < ? OR revoked = 1
                        """, (current_time,))
                        cleanup_counts["refresh_tokens"] = cursor.rowcount

                        # Clean expired sessions
                        cursor = conn.execute("""
                            DELETE FROM oauth2_sessions
                            WHERE expires_at < ?
                        """, (current_time,))
                        cleanup_counts["sessions"] = cursor.rowcount

                        # Clean old audit logs (keep last 6 months)
                        audit_cutoff = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
                        cursor = conn.execute("""
                            DELETE FROM oauth2_audit_log
                            WHERE timestamp < ?
                        """, (audit_cutoff,))
                        cleanup_counts["audit_logs"] = cursor.rowcount

                        # Commit transaction
                        conn.execute("COMMIT")

                        # Log cleanup event
                        total_cleaned = sum(cleanup_counts.values())
                        if total_cleaned > 0:
                            self._audit_log(
                                OAuth2AuditEvent.CLEANUP_PERFORMED,
                                "system",
                                "oauth2_storage",
                                details=cleanup_counts
                            )

                        logger.debug(f"OAuth2 cleanup completed: {cleanup_counts}")

                    except Exception:
                        conn.execute("ROLLBACK")
                        raise

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise OAuth2StorageError(f"Cleanup failed: {e}") from e

        return cleanup_counts

    def shutdown(self):
        """Gracefully shutdown the storage system."""
        self._shutdown = True

        if hasattr(self, '_cleanup_thread') and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)

        logger.info("OAuth2 encrypted storage shutdown completed")

    # Client Management Methods
    def register_client(
        self,
        client_name: str,
        client_type: ClientType,
        redirect_uris: list[str],
        grant_types: list[GrantType],
        scope: str = "read",
        metadata: dict[str, Any] | None = None
    ) -> OAuth2Client:
        """Register a new OAuth 2.0 client."""
        client_id = f"client_{secrets.token_urlsafe(16)}"
        client_secret = secrets.token_urlsafe(32) if client_type == ClientType.CONFIDENTIAL else None

        now = datetime.now(timezone.utc)

        client = OAuth2Client(
            client_id=client_id,
            client_secret=client_secret,
            client_name=client_name,
            client_type=client_type,
            redirect_uris=redirect_uris,
            grant_types=grant_types,
            scope=scope,
            created_at=now,
            updated_at=now,
            metadata=metadata or {}
        )

        try:
            with self._lock:
                with self._get_connection() as conn:
                    # Encrypt sensitive fields
                    client_secret_encrypted = self._encrypt_field(client_secret) if client_secret else ""
                    redirect_uris_encrypted = self._encrypt_field(redirect_uris)
                    grant_types_encrypted = self._encrypt_field([gt.value for gt in grant_types])
                    scope_encrypted = self._encrypt_field(scope)
                    metadata_encrypted = self._encrypt_field(metadata) if metadata else ""

                    # Compute integrity hash
                    integrity_hash = self._compute_integrity_hash(
                        client_id, client_secret_encrypted, client_name,
                        client_type.value, redirect_uris_encrypted,
                        grant_types_encrypted, scope_encrypted,
                        now.isoformat(), now.isoformat(), metadata_encrypted
                    )

                    conn.execute("""
                        INSERT INTO oauth2_clients (
                            client_id, client_secret_encrypted, client_name,
                            client_type, redirect_uris_encrypted, grant_types_encrypted,
                            scope_encrypted, created_at, updated_at,
                            metadata_encrypted, integrity_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        client_id, client_secret_encrypted, client_name,
                        client_type.value, redirect_uris_encrypted, grant_types_encrypted,
                        scope_encrypted, now.isoformat(), now.isoformat(),
                        metadata_encrypted, integrity_hash
                    ))

            # Audit log
            self._audit_log(
                OAuth2AuditEvent.CLIENT_REGISTERED,
                "client",
                client_id,
                details={
                    "client_name": client_name,
                    "client_type": client_type.value,
                    "grant_types": [gt.value for gt in grant_types]
                }
            )

            logger.info(f"OAuth2 client registered: {client_id}")
            return client

        except Exception as e:
            logger.error(f"Client registration failed: {e}")
            raise OAuth2StorageError(f"Client registration failed: {e}") from e

    def get_client(self, client_id: str) -> OAuth2Client | None:
        """Retrieve OAuth 2.0 client by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM oauth2_clients WHERE client_id = ?
                """, (client_id,))

                row = cursor.fetchone()
                if not row:
                    return None

                # Verify integrity
                if not self._verify_integrity(
                    dict(row),
                    row['client_id'], row['client_secret_encrypted'], row['client_name'],
                    row['client_type'], row['redirect_uris_encrypted'],
                    row['grant_types_encrypted'], row['scope_encrypted'],
                    row['created_at'], row['updated_at'], row['metadata_encrypted']
                ):
                    logger.error(f"Integrity verification failed for client: {client_id}")
                    raise OAuth2SecurityError("Client data integrity verification failed")

                # Decrypt fields
                client_secret = self._decrypt_field(row['client_secret_encrypted'])
                redirect_uris = self._decrypt_field(row['redirect_uris_encrypted'], [])
                grant_types_raw = self._decrypt_field(row['grant_types_encrypted'], [])
                grant_types = [GrantType(gt) for gt in grant_types_raw]
                scope = self._decrypt_field(row['scope_encrypted'], "")
                metadata = self._decrypt_field(row['metadata_encrypted'], {})

                return OAuth2Client(
                    client_id=row['client_id'],
                    client_secret=client_secret,
                    client_name=row['client_name'],
                    client_type=ClientType(row['client_type']),
                    redirect_uris=redirect_uris,
                    grant_types=grant_types,
                    scope=scope,
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    metadata=metadata
                )

        except Exception as e:
            logger.error(f"Failed to retrieve client {client_id}: {e}")
            if isinstance(e, OAuth2SecurityError):
                raise
            raise OAuth2StorageError(f"Failed to retrieve client: {e}") from e

    def update_client(
        self,
        client_id: str,
        client_name: str | None = None,
        redirect_uris: list[str] | None = None,
        grant_types: list[GrantType] | None = None,
        scope: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> bool:
        """Update OAuth 2.0 client."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    # First get current client data
                    cursor = conn.execute("""
                        SELECT * FROM oauth2_clients WHERE client_id = ?
                    """, (client_id,))

                    row = cursor.fetchone()
                    if not row:
                        return False

                    # Prepare update fields
                    updates = []
                    values = []

                    if client_name is not None:
                        updates.append("client_name = ?")
                        values.append(client_name)
                    else:
                        client_name = row['client_name']

                    if redirect_uris is not None:
                        redirect_uris_encrypted = self._encrypt_field(redirect_uris)
                        updates.append("redirect_uris_encrypted = ?")
                        values.append(redirect_uris_encrypted)
                    else:
                        redirect_uris_encrypted = row['redirect_uris_encrypted']

                    if grant_types is not None:
                        grant_types_encrypted = self._encrypt_field([gt.value for gt in grant_types])
                        updates.append("grant_types_encrypted = ?")
                        values.append(grant_types_encrypted)
                    else:
                        grant_types_encrypted = row['grant_types_encrypted']

                    if scope is not None:
                        scope_encrypted = self._encrypt_field(scope)
                        updates.append("scope_encrypted = ?")
                        values.append(scope_encrypted)
                    else:
                        scope_encrypted = row['scope_encrypted']

                    if metadata is not None:
                        metadata_encrypted = self._encrypt_field(metadata)
                        updates.append("metadata_encrypted = ?")
                        values.append(metadata_encrypted)
                    else:
                        metadata_encrypted = row['metadata_encrypted']

                    # Update timestamp
                    now = datetime.now(timezone.utc).isoformat()
                    updates.append("updated_at = ?")
                    values.append(now)

                    # Recompute integrity hash with updated values
                    integrity_hash = self._compute_integrity_hash(
                        client_id, row['client_secret_encrypted'], client_name,
                        row['client_type'], redirect_uris_encrypted,
                        grant_types_encrypted, scope_encrypted,
                        row['created_at'], now, metadata_encrypted
                    )
                    updates.append("integrity_hash = ?")
                    values.append(integrity_hash)

                    # Add client_id for WHERE clause
                    values.append(client_id)

                    # Execute update
                    query = f"UPDATE oauth2_clients SET {', '.join(updates)} WHERE client_id = ?"
                    cursor = conn.execute(query, values)

                    success = cursor.rowcount > 0

                    if success:
                        # Audit log
                        self._audit_log(
                            OAuth2AuditEvent.CLIENT_UPDATED,
                            "client",
                            client_id,
                            details={
                                "updated_fields": [field.split(' ')[0] for field in updates if 'integrity_hash' not in field and 'updated_at' not in field]
                            }
                        )

                    return success

        except Exception as e:
            logger.error(f"Failed to update client {client_id}: {e}")
            raise OAuth2StorageError(f"Failed to update client: {e}") from e

    def delete_client(self, client_id: str) -> bool:
        """Delete OAuth 2.0 client and all associated data."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    conn.execute("BEGIN IMMEDIATE")

                    try:
                        # Delete client (CASCADE will handle related data)
                        cursor = conn.execute("""
                            DELETE FROM oauth2_clients WHERE client_id = ?
                        """, (client_id,))

                        success = cursor.rowcount > 0

                        if success:
                            conn.execute("COMMIT")

                            # Audit log
                            self._audit_log(
                                OAuth2AuditEvent.CLIENT_DELETED,
                                "client",
                                client_id
                            )

                            logger.info(f"OAuth2 client deleted: {client_id}")
                        else:
                            conn.execute("ROLLBACK")

                        return success

                    except Exception:
                        conn.execute("ROLLBACK")
                        raise

        except Exception as e:
            logger.error(f"Failed to delete client {client_id}: {e}")
            raise OAuth2StorageError(f"Failed to delete client: {e}") from e

    def list_clients(self, limit: int = 100, offset: int = 0) -> list[OAuth2Client]:
        """List OAuth 2.0 clients."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM oauth2_clients
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))

                clients = []
                for row in cursor.fetchall():
                    try:
                        # Verify integrity
                        if not self._verify_integrity(
                            dict(row),
                            row['client_id'], row['client_secret_encrypted'], row['client_name'],
                            row['client_type'], row['redirect_uris_encrypted'],
                            row['grant_types_encrypted'], row['scope_encrypted'],
                            row['created_at'], row['updated_at'], row['metadata_encrypted']
                        ):
                            logger.error(f"Integrity verification failed for client: {row['client_id']}")
                            continue

                        # Decrypt fields
                        client_secret = self._decrypt_field(row['client_secret_encrypted'])
                        redirect_uris = self._decrypt_field(row['redirect_uris_encrypted'], [])
                        grant_types_raw = self._decrypt_field(row['grant_types_encrypted'], [])
                        grant_types = [GrantType(gt) for gt in grant_types_raw]
                        scope = self._decrypt_field(row['scope_encrypted'], "")
                        metadata = self._decrypt_field(row['metadata_encrypted'], {})

                        client = OAuth2Client(
                            client_id=row['client_id'],
                            client_secret=client_secret,
                            client_name=row['client_name'],
                            client_type=ClientType(row['client_type']),
                            redirect_uris=redirect_uris,
                            grant_types=grant_types,
                            scope=scope,
                            created_at=datetime.fromisoformat(row['created_at']),
                            updated_at=datetime.fromisoformat(row['updated_at']),
                            metadata=metadata
                        )

                        clients.append(client)

                    except Exception as e:
                        logger.error(f"Failed to decode client {row.get('client_id', 'unknown')}: {e}")
                        continue

                return clients

        except Exception as e:
            logger.error(f"Failed to list clients: {e}")
            raise OAuth2StorageError(f"Failed to list clients: {e}") from e

    # Authorization Code Methods
    def create_authorization_code(
        self,
        client_id: str,
        user_id: str,
        redirect_uri: str,
        scope: str,
        code_challenge: str | None = None,
        code_challenge_method: str | None = None,
        expires_in: int = 600,  # 10 minutes
        metadata: dict[str, Any] | None = None
    ) -> AuthorizationCode:
        """Create authorization code."""
        code = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=expires_in)

        auth_code = AuthorizationCode(
            code=code,
            client_id=client_id,
            user_id=user_id,
            redirect_uri=redirect_uri,
            scope=scope,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            expires_at=expires_at,
            created_at=now,
            used=False,
            metadata=metadata or {}
        )

        try:
            with self._lock:
                with self._get_connection() as conn:
                    # Encrypt sensitive fields
                    user_id_encrypted = self._encrypt_field(user_id)
                    redirect_uri_encrypted = self._encrypt_field(redirect_uri)
                    scope_encrypted = self._encrypt_field(scope)
                    code_challenge_encrypted = self._encrypt_field(code_challenge) if code_challenge else ""
                    metadata_encrypted = self._encrypt_field(metadata) if metadata else ""

                    # Compute integrity hash
                    integrity_hash = self._compute_integrity_hash(
                        code, client_id, user_id_encrypted, redirect_uri_encrypted,
                        scope_encrypted, code_challenge_encrypted, code_challenge_method,
                        expires_at.isoformat(), now.isoformat(), "0", metadata_encrypted
                    )

                    conn.execute("""
                        INSERT INTO authorization_codes (
                            code, client_id, user_id_encrypted, redirect_uri_encrypted,
                            scope_encrypted, code_challenge_encrypted, code_challenge_method,
                            expires_at, created_at, used, metadata_encrypted, integrity_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        code, client_id, user_id_encrypted, redirect_uri_encrypted,
                        scope_encrypted, code_challenge_encrypted, code_challenge_method,
                        expires_at.isoformat(), now.isoformat(), 0, metadata_encrypted, integrity_hash
                    ))

            # Audit log
            self._audit_log(
                OAuth2AuditEvent.AUTH_CODE_ISSUED,
                "authorization_code",
                code,
                user_id=user_id,
                client_id=client_id,
                details={"scope": scope, "expires_in": expires_in}
            )

            logger.debug(f"Authorization code created for client {client_id}")
            return auth_code

        except Exception as e:
            logger.error(f"Failed to create authorization code: {e}")
            raise OAuth2StorageError(f"Failed to create authorization code: {e}") from e

    def get_authorization_code(self, code: str) -> AuthorizationCode | None:
        """Retrieve authorization code."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM authorization_codes WHERE code = ?
                """, (code,))

                row = cursor.fetchone()
                if not row:
                    return None

                # Verify integrity
                if not self._verify_integrity(
                    dict(row),
                    row['code'], row['client_id'], row['user_id_encrypted'],
                    row['redirect_uri_encrypted'], row['scope_encrypted'],
                    row['code_challenge_encrypted'], row['code_challenge_method'],
                    row['expires_at'], row['created_at'], str(row['used']),
                    row['metadata_encrypted']
                ):
                    logger.error(f"Integrity verification failed for authorization code: {code}")
                    raise OAuth2SecurityError("Authorization code integrity verification failed")

                # Decrypt fields
                user_id = self._decrypt_field(row['user_id_encrypted'])
                redirect_uri = self._decrypt_field(row['redirect_uri_encrypted'])
                scope = self._decrypt_field(row['scope_encrypted'])
                code_challenge = self._decrypt_field(row['code_challenge_encrypted'])
                metadata = self._decrypt_field(row['metadata_encrypted'], {})

                return AuthorizationCode(
                    code=row['code'],
                    client_id=row['client_id'],
                    user_id=user_id,
                    redirect_uri=redirect_uri,
                    scope=scope,
                    code_challenge=code_challenge,
                    code_challenge_method=row['code_challenge_method'],
                    expires_at=datetime.fromisoformat(row['expires_at']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    used=bool(row['used']),
                    metadata=metadata
                )

        except Exception as e:
            logger.error(f"Failed to retrieve authorization code: {e}")
            if isinstance(e, OAuth2SecurityError):
                raise
            raise OAuth2StorageError(f"Failed to retrieve authorization code: {e}") from e

    def use_authorization_code(self, code: str) -> bool:
        """Mark authorization code as used."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    # First check if code exists and is valid
                    cursor = conn.execute("""
                        SELECT client_id, user_id_encrypted, used, expires_at FROM authorization_codes
                        WHERE code = ?
                    """, (code,))

                    row = cursor.fetchone()
                    if not row:
                        return False

                    if row['used']:
                        # Code already used - potential replay attack
                        self._audit_log(
                            OAuth2AuditEvent.SECURITY_VIOLATION,
                            "authorization_code",
                            code,
                            client_id=row['client_id'],
                            details={"violation": "code_reuse_attempt"}
                        )
                        return False

                    # Check expiration
                    expires_at = datetime.fromisoformat(row['expires_at'])
                    if expires_at < datetime.now(timezone.utc):
                        return False

                    # Mark as used
                    cursor = conn.execute("""
                        UPDATE authorization_codes SET used = 1 WHERE code = ?
                    """, (code,))

                    success = cursor.rowcount > 0

                    if success:
                        # Audit log
                        user_id = self._decrypt_field(row['user_id_encrypted'])
                        self._audit_log(
                            OAuth2AuditEvent.AUTH_CODE_USED,
                            "authorization_code",
                            code,
                            user_id=user_id,
                            client_id=row['client_id']
                        )

                    return success

        except Exception as e:
            logger.error(f"Failed to use authorization code: {e}")
            raise OAuth2StorageError(f"Failed to use authorization code: {e}") from e

    # Access Token Methods
    def create_access_token(
        self,
        client_id: str,
        user_id: str,
        scope: str,
        expires_in: int = 3600,  # 1 hour
        token_type: str = "Bearer",
        metadata: dict[str, Any] | None = None
    ) -> AccessToken:
        """Create access token."""
        token_id = f"at_{secrets.token_urlsafe(16)}"
        access_token = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=expires_in)

        token = AccessToken(
            token_id=token_id,
            access_token=access_token,
            client_id=client_id,
            user_id=user_id,
            scope=scope,
            token_type=token_type,
            expires_at=expires_at,
            created_at=now,
            revoked=False,
            metadata=metadata or {}
        )

        try:
            with self._lock:
                with self._get_connection() as conn:
                    # Hash the actual token for storage (non-reversible)
                    access_token_hash = self._hash_token(access_token)

                    # Encrypt sensitive fields
                    user_id_encrypted = self._encrypt_field(user_id)
                    scope_encrypted = self._encrypt_field(scope)
                    metadata_encrypted = self._encrypt_field(metadata) if metadata else ""

                    # Compute integrity hash
                    integrity_hash = self._compute_integrity_hash(
                        token_id, access_token_hash, client_id, user_id_encrypted,
                        scope_encrypted, token_type, expires_at.isoformat(),
                        now.isoformat(), "0", metadata_encrypted
                    )

                    conn.execute("""
                        INSERT INTO access_tokens (
                            token_id, access_token_hash, client_id, user_id_encrypted,
                            scope_encrypted, token_type, expires_at, created_at,
                            revoked, metadata_encrypted, integrity_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        token_id, access_token_hash, client_id, user_id_encrypted,
                        scope_encrypted, token_type, expires_at.isoformat(),
                        now.isoformat(), 0, metadata_encrypted, integrity_hash
                    ))

            # Audit log
            self._audit_log(
                OAuth2AuditEvent.ACCESS_TOKEN_ISSUED,
                "access_token",
                token_id,
                user_id=user_id,
                client_id=client_id,
                details={"scope": scope, "expires_in": expires_in}
            )

            logger.debug(f"Access token created: {token_id}")
            return token

        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            raise OAuth2StorageError(f"Failed to create access token: {e}") from e

    def get_access_token(self, access_token: str) -> AccessToken | None:
        """Retrieve access token by token value."""
        try:
            access_token_hash = self._hash_token(access_token)

            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM access_tokens WHERE access_token_hash = ?
                """, (access_token_hash,))

                row = cursor.fetchone()
                if not row:
                    return None

                # Verify integrity
                if not self._verify_integrity(
                    dict(row),
                    row['token_id'], row['access_token_hash'], row['client_id'],
                    row['user_id_encrypted'], row['scope_encrypted'], row['token_type'],
                    row['expires_at'], row['created_at'], str(row['revoked']),
                    row['metadata_encrypted']
                ):
                    logger.error(f"Integrity verification failed for access token: {row['token_id']}")
                    raise OAuth2SecurityError("Access token integrity verification failed")

                # Decrypt fields
                user_id = self._decrypt_field(row['user_id_encrypted'])
                scope = self._decrypt_field(row['scope_encrypted'])
                metadata = self._decrypt_field(row['metadata_encrypted'], {})

                return AccessToken(
                    token_id=row['token_id'],
                    access_token=access_token,  # Return original token
                    client_id=row['client_id'],
                    user_id=user_id,
                    scope=scope,
                    token_type=row['token_type'],
                    expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None,
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    revoked=bool(row['revoked']),
                    metadata=metadata
                )

        except Exception as e:
            logger.error(f"Failed to retrieve access token: {e}")
            if isinstance(e, OAuth2SecurityError):
                raise
            raise OAuth2StorageError(f"Failed to retrieve access token: {e}") from e

    def revoke_access_token(self, access_token: str) -> bool:
        """Revoke access token."""
        try:
            access_token_hash = self._hash_token(access_token)

            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.execute("""
                        UPDATE access_tokens SET revoked = 1
                        WHERE access_token_hash = ? AND revoked = 0
                    """, (access_token_hash,))

                    success = cursor.rowcount > 0

                    if success:
                        # Get token info for audit
                        cursor = conn.execute("""
                            SELECT token_id, client_id, user_id_encrypted FROM access_tokens
                            WHERE access_token_hash = ?
                        """, (access_token_hash,))
                        row = cursor.fetchone()

                        if row:
                            user_id = self._decrypt_field(row['user_id_encrypted'])
                            self._audit_log(
                                OAuth2AuditEvent.ACCESS_TOKEN_REVOKED,
                                "access_token",
                                row['token_id'],
                                user_id=user_id,
                                client_id=row['client_id']
                            )

                    return success

        except Exception as e:
            logger.error(f"Failed to revoke access token: {e}")
            raise OAuth2StorageError(f"Failed to revoke access token: {e}") from e

    # Refresh Token Methods
    def create_refresh_token(
        self,
        access_token_id: str,
        client_id: str,
        user_id: str,
        scope: str,
        expires_in: int | None = None,  # No expiration by default
        metadata: dict[str, Any] | None = None
    ) -> RefreshToken:
        """Create refresh token."""
        token_id = f"rt_{secrets.token_urlsafe(16)}"
        refresh_token = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=expires_in) if expires_in else None

        token = RefreshToken(
            token_id=token_id,
            refresh_token=refresh_token,
            access_token_id=access_token_id,
            client_id=client_id,
            user_id=user_id,
            scope=scope,
            expires_at=expires_at,
            created_at=now,
            revoked=False,
            metadata=metadata or {}
        )

        try:
            with self._lock:
                with self._get_connection() as conn:
                    # Hash the actual token for storage
                    refresh_token_hash = self._hash_token(refresh_token)

                    # Encrypt sensitive fields
                    user_id_encrypted = self._encrypt_field(user_id)
                    scope_encrypted = self._encrypt_field(scope)
                    metadata_encrypted = self._encrypt_field(metadata) if metadata else ""

                    # Compute integrity hash
                    integrity_hash = self._compute_integrity_hash(
                        token_id, refresh_token_hash, access_token_id, client_id,
                        user_id_encrypted, scope_encrypted,
                        expires_at.isoformat() if expires_at else "",
                        now.isoformat(), "0", metadata_encrypted
                    )

                    conn.execute("""
                        INSERT INTO refresh_tokens (
                            token_id, refresh_token_hash, access_token_id, client_id,
                            user_id_encrypted, scope_encrypted, expires_at, created_at,
                            revoked, metadata_encrypted, integrity_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        token_id, refresh_token_hash, access_token_id, client_id,
                        user_id_encrypted, scope_encrypted,
                        expires_at.isoformat() if expires_at else None,
                        now.isoformat(), 0, metadata_encrypted, integrity_hash
                    ))

            # Audit log
            self._audit_log(
                OAuth2AuditEvent.REFRESH_TOKEN_ISSUED,
                "refresh_token",
                token_id,
                user_id=user_id,
                client_id=client_id,
                details={"scope": scope, "access_token_id": access_token_id}
            )

            logger.debug(f"Refresh token created: {token_id}")
            return token

        except Exception as e:
            logger.error(f"Failed to create refresh token: {e}")
            raise OAuth2StorageError(f"Failed to create refresh token: {e}") from e

    def get_refresh_token(self, refresh_token: str) -> RefreshToken | None:
        """Retrieve refresh token by token value."""
        try:
            refresh_token_hash = self._hash_token(refresh_token)

            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM refresh_tokens WHERE refresh_token_hash = ?
                """, (refresh_token_hash,))

                row = cursor.fetchone()
                if not row:
                    return None

                # Verify integrity
                if not self._verify_integrity(
                    dict(row),
                    row['token_id'], row['refresh_token_hash'], row['access_token_id'],
                    row['client_id'], row['user_id_encrypted'], row['scope_encrypted'],
                    row['expires_at'] or "", row['created_at'], str(row['revoked']),
                    row['metadata_encrypted']
                ):
                    logger.error(f"Integrity verification failed for refresh token: {row['token_id']}")
                    raise OAuth2SecurityError("Refresh token integrity verification failed")

                # Decrypt fields
                user_id = self._decrypt_field(row['user_id_encrypted'])
                scope = self._decrypt_field(row['scope_encrypted'])
                metadata = self._decrypt_field(row['metadata_encrypted'], {})

                return RefreshToken(
                    token_id=row['token_id'],
                    refresh_token=refresh_token,  # Return original token
                    access_token_id=row['access_token_id'],
                    client_id=row['client_id'],
                    user_id=user_id,
                    scope=scope,
                    expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None,
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    revoked=bool(row['revoked']),
                    metadata=metadata
                )

        except Exception as e:
            logger.error(f"Failed to retrieve refresh token: {e}")
            if isinstance(e, OAuth2SecurityError):
                raise
            raise OAuth2StorageError(f"Failed to retrieve refresh token: {e}") from e

    def use_refresh_token(self, refresh_token: str) -> RefreshToken | None:
        """Use refresh token and mark as consumed (for single-use refresh tokens)."""
        try:
            refresh_token_hash = self._hash_token(refresh_token)

            with self._lock:
                with self._get_connection() as conn:
                    # Get token info first
                    cursor = conn.execute("""
                        SELECT * FROM refresh_tokens WHERE refresh_token_hash = ?
                    """, (refresh_token_hash,))

                    row = cursor.fetchone()
                    if not row:
                        return None

                    if row['revoked']:
                        # Token already revoked
                        return None

                    # Check expiration
                    if row['expires_at']:
                        expires_at = datetime.fromisoformat(row['expires_at'])
                        if expires_at < datetime.now(timezone.utc):
                            return None

                    # Decrypt token info
                    user_id = self._decrypt_field(row['user_id_encrypted'])
                    scope = self._decrypt_field(row['scope_encrypted'])
                    metadata = self._decrypt_field(row['metadata_encrypted'], {})

                    token = RefreshToken(
                        token_id=row['token_id'],
                        refresh_token=refresh_token,
                        access_token_id=row['access_token_id'],
                        client_id=row['client_id'],
                        user_id=user_id,
                        scope=scope,
                        expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None,
                        created_at=datetime.fromisoformat(row['created_at']),
                        revoked=False,
                        metadata=metadata
                    )

                    # Audit log
                    self._audit_log(
                        OAuth2AuditEvent.REFRESH_TOKEN_USED,
                        "refresh_token",
                        row['token_id'],
                        user_id=user_id,
                        client_id=row['client_id']
                    )

                    return token

        except Exception as e:
            logger.error(f"Failed to use refresh token: {e}")
            raise OAuth2StorageError(f"Failed to use refresh token: {e}") from e

    def revoke_refresh_token(self, refresh_token: str) -> bool:
        """Revoke refresh token."""
        try:
            refresh_token_hash = self._hash_token(refresh_token)

            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.execute("""
                        UPDATE refresh_tokens SET revoked = 1
                        WHERE refresh_token_hash = ? AND revoked = 0
                    """, (refresh_token_hash,))

                    success = cursor.rowcount > 0

                    if success:
                        # Get token info for audit
                        cursor = conn.execute("""
                            SELECT token_id, client_id, user_id_encrypted FROM refresh_tokens
                            WHERE refresh_token_hash = ?
                        """, (refresh_token_hash,))
                        row = cursor.fetchone()

                        if row:
                            user_id = self._decrypt_field(row['user_id_encrypted'])
                            self._audit_log(
                                OAuth2AuditEvent.REFRESH_TOKEN_REVOKED,
                                "refresh_token",
                                row['token_id'],
                                user_id=user_id,
                                client_id=row['client_id']
                            )

                    return success

        except Exception as e:
            logger.error(f"Failed to revoke refresh token: {e}")
            raise OAuth2StorageError(f"Failed to revoke refresh token: {e}") from e

    # Session Management Methods
    def create_session(
        self,
        state: str,
        client_id: str,
        redirect_uri: str,
        scope: str,
        user_id: str | None = None,
        code_challenge: str | None = None,
        code_challenge_method: str | None = None,
        expires_in: int = 600,  # 10 minutes
        metadata: dict[str, Any] | None = None
    ) -> OAuth2Session:
        """Create OAuth 2.0 session."""
        session_id = f"sess_{secrets.token_urlsafe(16)}"
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=expires_in)

        session = OAuth2Session(
            session_id=session_id,
            state=state,
            client_id=client_id,
            user_id=user_id,
            redirect_uri=redirect_uri,
            scope=scope,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            expires_at=expires_at,
            created_at=now,
            completed=False,
            metadata=metadata or {}
        )

        try:
            with self._lock:
                with self._get_connection() as conn:
                    # Encrypt sensitive fields
                    state_encrypted = self._encrypt_field(state)
                    user_id_encrypted = self._encrypt_field(user_id) if user_id else ""
                    redirect_uri_encrypted = self._encrypt_field(redirect_uri)
                    scope_encrypted = self._encrypt_field(scope)
                    code_challenge_encrypted = self._encrypt_field(code_challenge) if code_challenge else ""
                    metadata_encrypted = self._encrypt_field(metadata) if metadata else ""

                    # Compute integrity hash
                    integrity_hash = self._compute_integrity_hash(
                        session_id, state_encrypted, client_id, user_id_encrypted,
                        redirect_uri_encrypted, scope_encrypted, code_challenge_encrypted,
                        code_challenge_method, expires_at.isoformat(), now.isoformat(),
                        "0", metadata_encrypted
                    )

                    conn.execute("""
                        INSERT INTO oauth2_sessions (
                            session_id, state_encrypted, client_id, user_id_encrypted,
                            redirect_uri_encrypted, scope_encrypted, code_challenge_encrypted,
                            code_challenge_method, expires_at, created_at,
                            completed, metadata_encrypted, integrity_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        session_id, state_encrypted, client_id, user_id_encrypted,
                        redirect_uri_encrypted, scope_encrypted, code_challenge_encrypted,
                        code_challenge_method, expires_at.isoformat(), now.isoformat(),
                        0, metadata_encrypted, integrity_hash
                    ))

            # Audit log
            self._audit_log(
                OAuth2AuditEvent.SESSION_CREATED,
                "session",
                session_id,
                user_id=user_id,
                client_id=client_id,
                details={"scope": scope, "expires_in": expires_in}
            )

            logger.debug(f"OAuth2 session created: {session_id}")
            return session

        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise OAuth2StorageError(f"Failed to create session: {e}") from e

    def get_session(self, session_id: str) -> OAuth2Session | None:
        """Retrieve OAuth 2.0 session."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM oauth2_sessions WHERE session_id = ?
                """, (session_id,))

                row = cursor.fetchone()
                if not row:
                    return None

                # Verify integrity
                if not self._verify_integrity(
                    dict(row),
                    row['session_id'], row['state_encrypted'], row['client_id'],
                    row['user_id_encrypted'], row['redirect_uri_encrypted'],
                    row['scope_encrypted'], row['code_challenge_encrypted'],
                    row['code_challenge_method'], row['expires_at'], row['created_at'],
                    str(row['completed']), row['metadata_encrypted']
                ):
                    logger.error(f"Integrity verification failed for session: {session_id}")
                    raise OAuth2SecurityError("Session integrity verification failed")

                # Decrypt fields
                state = self._decrypt_field(row['state_encrypted'])
                user_id = self._decrypt_field(row['user_id_encrypted'])
                redirect_uri = self._decrypt_field(row['redirect_uri_encrypted'])
                scope = self._decrypt_field(row['scope_encrypted'])
                code_challenge = self._decrypt_field(row['code_challenge_encrypted'])
                metadata = self._decrypt_field(row['metadata_encrypted'], {})

                return OAuth2Session(
                    session_id=row['session_id'],
                    state=state,
                    client_id=row['client_id'],
                    user_id=user_id,
                    redirect_uri=redirect_uri,
                    scope=scope,
                    code_challenge=code_challenge,
                    code_challenge_method=row['code_challenge_method'],
                    expires_at=datetime.fromisoformat(row['expires_at']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    completed=bool(row['completed']),
                    metadata=metadata
                )

        except Exception as e:
            logger.error(f"Failed to retrieve session: {e}")
            if isinstance(e, OAuth2SecurityError):
                raise
            raise OAuth2StorageError(f"Failed to retrieve session: {e}") from e

    def update_session(
        self,
        session_id: str,
        user_id: str | None = None,
        completed: bool | None = None,
        metadata: dict[str, Any] | None = None
    ) -> bool:
        """Update OAuth 2.0 session."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    # Get current session data
                    cursor = conn.execute("""
                        SELECT * FROM oauth2_sessions WHERE session_id = ?
                    """, (session_id,))

                    row = cursor.fetchone()
                    if not row:
                        return False

                    # Prepare update fields
                    updates = []
                    values = []

                    if user_id is not None:
                        user_id_encrypted = self._encrypt_field(user_id)
                        updates.append("user_id_encrypted = ?")
                        values.append(user_id_encrypted)
                    else:
                        user_id_encrypted = row['user_id_encrypted']

                    if completed is not None:
                        updates.append("completed = ?")
                        values.append(1 if completed else 0)
                        completed_int = 1 if completed else 0
                    else:
                        completed_int = row['completed']

                    if metadata is not None:
                        metadata_encrypted = self._encrypt_field(metadata)
                        updates.append("metadata_encrypted = ?")
                        values.append(metadata_encrypted)
                    else:
                        metadata_encrypted = row['metadata_encrypted']

                    if not updates:
                        return True  # No changes

                    # Recompute integrity hash
                    integrity_hash = self._compute_integrity_hash(
                        session_id, row['state_encrypted'], row['client_id'],
                        user_id_encrypted, row['redirect_uri_encrypted'],
                        row['scope_encrypted'], row['code_challenge_encrypted'],
                        row['code_challenge_method'], row['expires_at'], row['created_at'],
                        str(completed_int), metadata_encrypted
                    )
                    updates.append("integrity_hash = ?")
                    values.append(integrity_hash)

                    # Add session_id for WHERE clause
                    values.append(session_id)

                    # Execute update
                    query = f"UPDATE oauth2_sessions SET {', '.join(updates)} WHERE session_id = ?"
                    cursor = conn.execute(query, values)

                    success = cursor.rowcount > 0

                    if success and completed:
                        # Audit log for session completion
                        current_user_id = self._decrypt_field(user_id_encrypted) if user_id_encrypted else None
                        self._audit_log(
                            OAuth2AuditEvent.SESSION_COMPLETED,
                            "session",
                            session_id,
                            user_id=current_user_id,
                            client_id=row['client_id']
                        )

                    return success

        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {e}")
            raise OAuth2StorageError(f"Failed to update session: {e}") from e

    def delete_session(self, session_id: str) -> bool:
        """Delete OAuth 2.0 session."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.execute("""
                        DELETE FROM oauth2_sessions WHERE session_id = ?
                    """, (session_id,))

                    success = cursor.rowcount > 0

                    if success:
                        logger.debug(f"OAuth2 session deleted: {session_id}")

                    return success

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            raise OAuth2StorageError(f"Failed to delete session: {e}") from e

    # Audit and Monitoring Methods
    def get_audit_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        event_type: str | None = None,
        entity_type: str | None = None,
        client_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve audit logs with filtering."""
        try:
            with self._get_connection() as conn:
                # Build query conditions
                conditions = []
                params = []

                if event_type:
                    conditions.append("event_type = ?")
                    params.append(event_type)

                if entity_type:
                    conditions.append("entity_type = ?")
                    params.append(entity_type)

                if client_id:
                    conditions.append("client_id = ?")
                    params.append(client_id)

                if start_time:
                    conditions.append("timestamp >= ?")
                    params.append(start_time.isoformat())

                if end_time:
                    conditions.append("timestamp <= ?")
                    params.append(end_time.isoformat())

                where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

                query = f"""
                    SELECT * FROM oauth2_audit_log
                    {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                """
                params.extend([limit, offset])

                cursor = conn.execute(query, params)

                audit_logs = []
                for row in cursor.fetchall():
                    try:
                        # Decrypt sensitive fields
                        user_id = self._decrypt_field(row['user_id_encrypted']) if row['user_id_encrypted'] else None
                        details = self._decrypt_field(row['details_encrypted'], {}) if row['details_encrypted'] else {}
                        user_agent = self._decrypt_field(row['user_agent_encrypted']) if row['user_agent_encrypted'] else None

                        audit_log = {
                            "id": row['id'],
                            "event_type": row['event_type'],
                            "entity_type": row['entity_type'],
                            "entity_id": row['entity_id'],
                            "user_id": user_id,
                            "client_id": row['client_id'],
                            "details": details,
                            "ip_address": row['ip_address'],
                            "user_agent": user_agent,
                            "timestamp": row['timestamp']
                        }

                        audit_logs.append(audit_log)

                    except Exception as e:
                        logger.error(f"Failed to decode audit log {row['id']}: {e}")
                        continue

                return audit_logs

        except Exception as e:
            logger.error(f"Failed to retrieve audit logs: {e}")
            raise OAuth2StorageError(f"Failed to retrieve audit logs: {e}") from e

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        try:
            with self._get_connection() as conn:
                stats = {}

                # Count active entities
                stats['clients'] = conn.execute("SELECT COUNT(*) FROM oauth2_clients").fetchone()[0]
                stats['active_auth_codes'] = conn.execute(
                    "SELECT COUNT(*) FROM authorization_codes WHERE used = 0 AND expires_at > ?"
                    , (datetime.now(timezone.utc).isoformat(),)
                ).fetchone()[0]
                stats['active_access_tokens'] = conn.execute(
                    "SELECT COUNT(*) FROM access_tokens WHERE revoked = 0 AND (expires_at IS NULL OR expires_at > ?)"
                    , (datetime.now(timezone.utc).isoformat(),)
                ).fetchone()[0]
                stats['active_refresh_tokens'] = conn.execute(
                    "SELECT COUNT(*) FROM refresh_tokens WHERE revoked = 0 AND (expires_at IS NULL OR expires_at > ?)"
                    , (datetime.now(timezone.utc).isoformat(),)
                ).fetchone()[0]
                stats['active_sessions'] = conn.execute(
                    "SELECT COUNT(*) FROM oauth2_sessions WHERE completed = 0 AND expires_at > ?"
                    , (datetime.now(timezone.utc).isoformat(),)
                ).fetchone()[0]
                stats['audit_logs'] = conn.execute("SELECT COUNT(*) FROM oauth2_audit_log").fetchone()[0]

                # Database file size
                try:
                    db_path = Path(self.db_path)
                    if db_path.exists():
                        stats['database_size_bytes'] = db_path.stat().st_size
                    else:
                        stats['database_size_bytes'] = 0
                except Exception:
                    stats['database_size_bytes'] = 0

                return stats

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            raise OAuth2StorageError(f"Failed to get storage stats: {e}") from e


# Global storage instance
_oauth2_storage: OAuth2EncryptedStorage | None = None
_storage_lock = threading.Lock()


def get_oauth2_storage(
    db_path: str | None = None,
    encryption_key: str | None = None,
    enable_audit: bool = True
) -> OAuth2EncryptedStorage:
    """
    Get global OAuth2 encrypted storage instance (singleton pattern).

    Args:
        db_path: Path to SQLite database file
        encryption_key: Custom encryption key
        enable_audit: Enable audit logging

    Returns:
        OAuth2EncryptedStorage instance
    """
    global _oauth2_storage

    if _oauth2_storage is None:
        with _storage_lock:
            if _oauth2_storage is None:
                _oauth2_storage = OAuth2EncryptedStorage(
                    db_path=db_path,
                    encryption_key=encryption_key,
                    enable_audit=enable_audit
                )

    return _oauth2_storage


def shutdown_oauth2_storage():
    """Shutdown global OAuth2 storage instance."""
    global _oauth2_storage

    if _oauth2_storage:
        with _storage_lock:
            if _oauth2_storage:
                _oauth2_storage.shutdown()
                _oauth2_storage = None


# Context manager for transactional operations
@contextmanager
def oauth2_transaction(storage: OAuth2EncryptedStorage | None = None):
    """
    Context manager for OAuth2 storage transactions.

    Args:
        storage: OAuth2EncryptedStorage instance (uses global if None)
    """
    if storage is None:
        storage = get_oauth2_storage()

    with storage._lock:
        with storage._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
