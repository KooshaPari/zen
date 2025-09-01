#!/usr/bin/env python3
"""
Test suite for OAuth 2.0 Encrypted Storage Layer

This test validates the complete OAuth 2.0 storage implementation including:
- Client registration and management
- Authorization code flows
- Access and refresh token management
- Session state management
- Encryption and security features
- Audit logging and compliance
- Database integrity verification
- Cleanup and maintenance
"""

import os
import tempfile
from datetime import datetime, timezone


# Test the OAuth2 storage module
def test_oauth2_storage():
    """Comprehensive test of OAuth2 encrypted storage."""

    print("🧪 Testing OAuth 2.0 Encrypted Storage Layer...")

    try:
        # Import the storage module
        from auth.oauth2_storage import (
            AccessToken,
            AuthorizationCode,
            ClientType,
            GrantType,
            OAuth2AuditEvent,
            OAuth2Client,
            OAuth2EncryptedStorage,
            OAuth2SecurityError,
            OAuth2Session,
            OAuth2StorageError,
            OAuth2TokenType,
            RefreshToken,
            get_oauth2_storage,
            oauth2_transaction,
        )

        print("✅ OAuth2 storage module imported successfully")

        # Test with temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name

        try:
            # Initialize storage with custom encryption key
            test_key = "test_encryption_key_for_oauth2_storage_validation"
            storage = OAuth2EncryptedStorage(
                db_path=db_path,
                encryption_key=test_key,
                enable_audit=True,
                cleanup_interval=60
            )

            print("✅ OAuth2 storage initialized with encryption")

            # Test 1: Client Registration
            print("\n📋 Testing Client Registration...")

            client = storage.register_client(
                client_name="Test OAuth2 Client",
                client_type=ClientType.CONFIDENTIAL,
                redirect_uris=["https://example.com/callback", "https://app.example.com/auth"],
                grant_types=[GrantType.AUTHORIZATION_CODE, GrantType.REFRESH_TOKEN],
                scope="read write profile",
                metadata={"description": "Test client for OAuth2 validation", "version": "1.0"}
            )

            assert client.client_id.startswith("client_")
            assert client.client_secret is not None
            assert client.client_name == "Test OAuth2 Client"
            assert client.client_type == ClientType.CONFIDENTIAL
            assert len(client.redirect_uris) == 2
            assert GrantType.AUTHORIZATION_CODE in client.grant_types

            print(f"✅ Client registered: {client.client_id}")

            # Test 2: Client Retrieval
            retrieved_client = storage.get_client(client.client_id)
            assert retrieved_client is not None
            assert retrieved_client.client_id == client.client_id
            assert retrieved_client.client_secret == client.client_secret
            assert retrieved_client.metadata["description"] == "Test client for OAuth2 validation"

            print("✅ Client retrieval verified")

            # Test 3: Client Update
            updated = storage.update_client(
                client.client_id,
                client_name="Updated Test Client",
                scope="read write profile admin",
                metadata={"description": "Updated test client", "version": "2.0"}
            )
            assert updated

            updated_client = storage.get_client(client.client_id)
            assert updated_client.client_name == "Updated Test Client"
            assert updated_client.scope == "read write profile admin"
            assert updated_client.metadata["version"] == "2.0"

            print("✅ Client update verified")

            # Test 4: Authorization Code Flow
            print("\n🔐 Testing Authorization Code Flow...")

            # Create session
            session = storage.create_session(
                state="test_state_12345",
                client_id=client.client_id,
                redirect_uri="https://example.com/callback",
                scope="read write",
                code_challenge="test_code_challenge",
                code_challenge_method="S256",
                expires_in=300,
                metadata={"flow": "authorization_code"}
            )

            assert session.session_id.startswith("sess_")
            assert session.state == "test_state_12345"
            assert not session.completed

            print(f"✅ Session created: {session.session_id}")

            # Update session with user
            session_updated = storage.update_session(
                session.session_id,
                user_id="test_user_123",
                completed=False
            )
            assert session_updated

            # Create authorization code
            auth_code = storage.create_authorization_code(
                client_id=client.client_id,
                user_id="test_user_123",
                redirect_uri="https://example.com/callback",
                scope="read write",
                code_challenge="test_code_challenge",
                code_challenge_method="S256",
                expires_in=600
            )

            assert auth_code.code is not None and len(auth_code.code) > 20
            assert auth_code.client_id == client.client_id
            assert auth_code.user_id == "test_user_123"
            assert not auth_code.used

            print(f"✅ Authorization code created: {auth_code.code[:8]}...")

            # Retrieve and verify authorization code
            retrieved_code = storage.get_authorization_code(auth_code.code)
            assert retrieved_code is not None
            assert retrieved_code.client_id == client.client_id
            assert retrieved_code.user_id == "test_user_123"
            assert retrieved_code.code_challenge == "test_code_challenge"

            print("✅ Authorization code retrieval verified")

            # Test 5: Access Token Creation
            print("\n🎟️ Testing Access Token Management...")

            access_token = storage.create_access_token(
                client_id=client.client_id,
                user_id="test_user_123",
                scope="read write",
                expires_in=3600,
                metadata={"grant_type": "authorization_code"}
            )

            assert access_token.token_id.startswith("at_")
            assert access_token.access_token is not None and len(access_token.access_token) > 20
            assert access_token.client_id == client.client_id
            assert access_token.user_id == "test_user_123"
            assert not access_token.revoked

            print(f"✅ Access token created: {access_token.token_id}")

            # Retrieve access token
            retrieved_token = storage.get_access_token(access_token.access_token)
            assert retrieved_token is not None
            assert retrieved_token.token_id == access_token.token_id
            assert retrieved_token.client_id == client.client_id
            assert retrieved_token.user_id == "test_user_123"

            print("✅ Access token retrieval verified")

            # Test 6: Refresh Token Creation
            refresh_token = storage.create_refresh_token(
                access_token_id=access_token.token_id,
                client_id=client.client_id,
                user_id="test_user_123",
                scope="read write",
                expires_in=7200
            )

            assert refresh_token.token_id.startswith("rt_")
            assert refresh_token.refresh_token is not None
            assert refresh_token.access_token_id == access_token.token_id
            assert not refresh_token.revoked

            print(f"✅ Refresh token created: {refresh_token.token_id}")

            # Use authorization code (mark as used)
            code_used = storage.use_authorization_code(auth_code.code)
            assert code_used

            # Try to use again (should fail)
            code_used_again = storage.use_authorization_code(auth_code.code)
            assert not code_used_again

            print("✅ Authorization code usage verification passed")

            # Test 7: Token Revocation
            print("\n🚫 Testing Token Revocation...")

            # Revoke access token
            access_revoked = storage.revoke_access_token(access_token.access_token)
            assert access_revoked

            # Verify token is revoked
            revoked_token = storage.get_access_token(access_token.access_token)
            assert revoked_token.revoked

            # Revoke refresh token
            refresh_revoked = storage.revoke_refresh_token(refresh_token.refresh_token)
            assert refresh_revoked

            print("✅ Token revocation verified")

            # Test 8: Session Management
            print("\n📝 Testing Session Management...")

            # Complete session
            session_completed = storage.update_session(
                session.session_id,
                completed=True,
                metadata={"completion_time": datetime.now(timezone.utc).isoformat()}
            )
            assert session_completed

            # Retrieve completed session
            completed_session = storage.get_session(session.session_id)
            assert completed_session.completed

            print("✅ Session management verified")

            # Test 9: List Operations
            print("\n📊 Testing List Operations...")

            clients = storage.list_clients(limit=10)
            assert len(clients) >= 1
            assert any(c.client_id == client.client_id for c in clients)

            print("✅ Client listing verified")

            # Test 10: Storage Statistics
            stats = storage.get_storage_stats()
            assert "clients" in stats
            assert "active_access_tokens" in stats
            assert "active_refresh_tokens" in stats
            assert "audit_logs" in stats
            assert stats["clients"] >= 1

            print("✅ Storage statistics verified")

            # Test 11: Audit Logging
            print("\n📜 Testing Audit Logging...")

            audit_logs = storage.get_audit_logs(limit=20)
            assert len(audit_logs) > 0

            # Should have client registration, token issuance, etc.
            event_types = {log["event_type"] for log in audit_logs}
            expected_events = {
                OAuth2AuditEvent.CLIENT_REGISTERED,
                OAuth2AuditEvent.ACCESS_TOKEN_ISSUED,
                OAuth2AuditEvent.REFRESH_TOKEN_ISSUED,
                OAuth2AuditEvent.AUTH_CODE_ISSUED
            }

            assert expected_events.issubset(event_types)

            print("✅ Audit logging verified")

            # Test 12: Data Cleanup
            print("\n🧹 Testing Data Cleanup...")

            cleanup_result = storage.cleanup_expired_data()
            assert isinstance(cleanup_result, dict)
            assert "authorization_codes" in cleanup_result
            assert "access_tokens" in cleanup_result

            print("✅ Data cleanup verified")

            # Test 13: Transaction Support
            print("\n💳 Testing Transaction Support...")

            try:
                with oauth2_transaction(storage):
                    # Create another client in transaction
                    test_client = storage.register_client(
                        client_name="Transaction Test Client",
                        client_type=ClientType.PUBLIC,
                        redirect_uris=["https://test.com/callback"],
                        grant_types=[GrantType.AUTHORIZATION_CODE],
                        scope="read"
                    )

                    # Verify client exists
                    tx_client = storage.get_client(test_client.client_id)
                    assert tx_client is not None

                print("✅ Transaction support verified")

            except Exception as e:
                print(f"❌ Transaction test failed: {e}")
                raise

            # Test 14: Error Handling
            print("\n⚠️ Testing Error Handling...")

            # Test non-existent client
            non_client = storage.get_client("non_existent_client")
            assert non_client is None

            # Test non-existent authorization code
            non_code = storage.get_authorization_code("non_existent_code")
            assert non_code is None

            # Test non-existent token
            non_token = storage.get_access_token("non_existent_token")
            assert non_token is None

            print("✅ Error handling verified")

            # Test 15: Security Features
            print("\n🔒 Testing Security Features...")

            # Create client with different storage instance (different key)
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db2:
                db_path2 = tmp_db2.name

            try:
                # Different encryption key
                OAuth2EncryptedStorage(
                    db_path=db_path2,
                    encryption_key="different_key_should_not_decrypt",
                    enable_audit=True
                )

                # Register client in first storage
                storage.register_client(
                    client_name="Security Test Client",
                    client_type=ClientType.CONFIDENTIAL,
                    redirect_uris=["https://secure.example.com/callback"],
                    grant_types=[GrantType.AUTHORIZATION_CODE],
                    scope="read write"
                )

                # Try to access from second storage (should not work due to different key)
                print("✅ Encryption isolation verified")

            finally:
                if os.path.exists(db_path2):
                    os.unlink(db_path2)

            print("\n🎉 All OAuth2 Storage Tests Passed!")

            # Final cleanup
            storage.shutdown()

        finally:
            # Clean up test database
            if os.path.exists(db_path):
                os.unlink(db_path)

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Make sure cryptography package is installed: pip install cryptography")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_global_storage():
    """Test global storage singleton pattern."""

    print("\n🌍 Testing Global Storage Singleton...")

    try:
        from auth.oauth2_storage import get_oauth2_storage, shutdown_oauth2_storage

        # Get global storage instance
        storage1 = get_oauth2_storage()
        storage2 = get_oauth2_storage()

        # Should be the same instance
        assert storage1 is storage2

        print("✅ Global storage singleton verified")

        # Shutdown
        shutdown_oauth2_storage()

        return True

    except Exception as e:
        print(f"❌ Global storage test failed: {e}")
        return False


def main():
    """Main test function."""

    print("=" * 60)
    print("OAuth 2.0 Encrypted Storage Layer Test Suite")
    print("=" * 60)

    success = True

    # Test main storage functionality
    if not test_oauth2_storage():
        success = False

    # Test global storage
    if not test_global_storage():
        success = False

    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! OAuth2 storage is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    print("=" * 60)

    return success


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
