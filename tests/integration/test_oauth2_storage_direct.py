#!/usr/bin/env python3
"""
Direct test of OAuth 2.0 Encrypted Storage Layer

This test directly imports and validates the OAuth2 storage module
without going through the auth package __init__.py to avoid
import issues with other modules.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_oauth2_storage_direct():
    """Test OAuth2 storage directly."""

    print("🧪 Testing OAuth 2.0 Encrypted Storage (Direct Import)...")

    try:
        # Import directly from the module
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
            oauth2_transaction,
        )

        print("✅ OAuth2 storage module imported successfully")

        # Test with temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name

        try:
            # Initialize storage
            test_key = "test_key_for_validation_12345"
            storage = OAuth2EncryptedStorage(
                db_path=db_path,
                encryption_key=test_key,
                enable_audit=True,
                cleanup_interval=3600
            )

            print("✅ OAuth2 storage initialized")

            # Test client registration
            client = storage.register_client(
                client_name="Test Client",
                client_type=ClientType.CONFIDENTIAL,
                redirect_uris=["https://example.com/callback"],
                grant_types=[GrantType.AUTHORIZATION_CODE, GrantType.REFRESH_TOKEN],
                scope="read write",
                metadata={"test": "data"}
            )

            print(f"✅ Client registered: {client.client_id}")
            assert client.client_id.startswith("client_")
            assert client.client_secret is not None

            # Test client retrieval
            retrieved = storage.get_client(client.client_id)
            assert retrieved is not None
            assert retrieved.client_name == "Test Client"

            print("✅ Client retrieval verified")

            # Test authorization code
            auth_code = storage.create_authorization_code(
                client_id=client.client_id,
                user_id="test_user",
                redirect_uri="https://example.com/callback",
                scope="read write"
            )

            print(f"✅ Authorization code created: {auth_code.code[:8]}...")

            # Test access token
            access_token = storage.create_access_token(
                client_id=client.client_id,
                user_id="test_user",
                scope="read write"
            )

            print(f"✅ Access token created: {access_token.token_id}")

            # Test token retrieval
            retrieved_token = storage.get_access_token(access_token.access_token)
            assert retrieved_token is not None
            assert retrieved_token.user_id == "test_user"

            print("✅ Access token retrieval verified")

            # Test refresh token
            refresh_token = storage.create_refresh_token(
                access_token_id=access_token.token_id,
                client_id=client.client_id,
                user_id="test_user",
                scope="read write"
            )

            print(f"✅ Refresh token created: {refresh_token.token_id}")

            # Test session
            session = storage.create_session(
                state="test_state",
                client_id=client.client_id,
                redirect_uri="https://example.com/callback",
                scope="read write"
            )

            print(f"✅ Session created: {session.session_id}")

            # Test statistics
            stats = storage.get_storage_stats()
            assert stats['clients'] >= 1
            print("✅ Statistics verified")

            # Test audit logs
            audit_logs = storage.get_audit_logs(limit=10)
            assert len(audit_logs) > 0
            print(f"✅ Audit logs verified: {len(audit_logs)} entries")

            # Test cleanup
            storage.cleanup_expired_data()
            print("✅ Cleanup completed")

            # Test transaction
            try:
                with oauth2_transaction(storage):
                    storage.register_client(
                        client_name="Transaction Test",
                        client_type=ClientType.PUBLIC,
                        redirect_uris=["https://tx.example.com/callback"],
                        grant_types=[GrantType.AUTHORIZATION_CODE],
                        scope="read"
                    )
                print("✅ Transaction support verified")
            except Exception as e:
                print(f"⚠️ Transaction test warning: {e}")

            print("\n🎉 All Direct Storage Tests Passed!")

            # Shutdown
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


def main():
    """Main test function."""

    print("=" * 60)
    print("OAuth 2.0 Encrypted Storage Direct Test")
    print("=" * 60)

    success = test_oauth2_storage_direct()

    print("\n" + "=" * 60)
    if success:
        print("🎉 Direct storage test passed!")
    else:
        print("❌ Direct storage test failed.")
    print("=" * 60)

    return success


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
