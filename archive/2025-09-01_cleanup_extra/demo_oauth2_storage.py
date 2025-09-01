#!/usr/bin/env python3
"""
OAuth 2.0 Encrypted Storage Demo

This demo showcases the capabilities of the OAuth 2.0 encrypted storage layer
including client management, token flows, session handling, and security features.
"""

import os
import sys
from datetime import datetime
from pathlib import Path


def demo_oauth2_storage():
    """Demonstrate OAuth2 storage capabilities."""

    print("🚀 OAuth 2.0 Encrypted Storage Demo")
    print("=" * 50)

    try:
        # Import OAuth2 storage
        from auth.oauth2_storage import ClientType, GrantType, OAuth2TokenType, get_oauth2_storage, oauth2_transaction

        print("✅ OAuth2 storage imported successfully")

        # Initialize storage (will use default database location)
        storage = get_oauth2_storage(
            db_path="demo_oauth2.db",
            enable_audit=True
        )

        print("✅ OAuth2 storage initialized")
        print(f"📍 Database location: {Path('demo_oauth2.db').absolute()}")

        # Demo 1: Client Registration
        print("\n📋 Demo 1: Client Registration")
        print("-" * 30)

        # Register a web application client
        web_client = storage.register_client(
            client_name="Demo Web Application",
            client_type=ClientType.CONFIDENTIAL,
            redirect_uris=[
                "https://myapp.example.com/auth/callback",
                "https://myapp.example.com/auth/silent"
            ],
            grant_types=[
                GrantType.AUTHORIZATION_CODE,
                GrantType.REFRESH_TOKEN
            ],
            scope="read write profile email",
            metadata={
                "description": "Demo web application with full OAuth2 flow",
                "homepage": "https://myapp.example.com",
                "contact_email": "admin@example.com"
            }
        )

        print(f"🆔 Client ID: {web_client.client_id}")
        print(f"🔑 Client Secret: {web_client.client_secret[:8]}... (truncated)")
        print(f"📱 Client Type: {web_client.client_type.value}")
        print(f"🔗 Redirect URIs: {', '.join(web_client.redirect_uris)}")
        print(f"🎯 Grant Types: {', '.join([gt.value for gt in web_client.grant_types])}")
        print(f"🌍 Scope: {web_client.scope}")

        # Register a mobile/SPA client
        mobile_client = storage.register_client(
            client_name="Demo Mobile App",
            client_type=ClientType.PUBLIC,
            redirect_uris=[
                "com.example.mobileapp://oauth/callback",
                "https://mobile.example.com/auth"
            ],
            grant_types=[GrantType.AUTHORIZATION_CODE],
            scope="read profile",
            metadata={
                "description": "Demo mobile application with PKCE",
                "platform": "iOS/Android",
                "app_store_url": "https://apps.example.com/mobile"
            }
        )

        print(f"\n📱 Mobile Client ID: {mobile_client.client_id}")
        print(f"🔒 Client Type: {mobile_client.client_type.value} (no secret)")

        # Demo 2: Authorization Flow
        print("\n🔐 Demo 2: Authorization Code Flow")
        print("-" * 35)

        # Create OAuth session
        session = storage.create_session(
            state="abc123def456",
            client_id=web_client.client_id,
            redirect_uri="https://myapp.example.com/auth/callback",
            scope="read write profile",
            code_challenge="dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk",
            code_challenge_method="S256",
            expires_in=600  # 10 minutes
        )

        print(f"🎫 Session ID: {session.session_id}")
        print(f"🎲 State: {session.state}")
        print(f"🔗 Redirect URI: {session.redirect_uri}")
        print(f"⏰ Expires: {session.expires_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        # Simulate user authentication and consent
        print("\n👤 Simulating user authentication...")
        user_id = "demo_user_12345"

        # Update session with authenticated user
        storage.update_session(
            session.session_id,
            user_id=user_id
        )

        # Issue authorization code
        auth_code = storage.create_authorization_code(
            client_id=web_client.client_id,
            user_id=user_id,
            redirect_uri="https://myapp.example.com/auth/callback",
            scope="read write profile",
            code_challenge="dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk",
            code_challenge_method="S256"
        )

        print(f"🎟️ Authorization Code: {auth_code.code[:12]}... (truncated)")
        print(f"👤 User ID: {auth_code.user_id}")
        print(f"🌍 Scope: {auth_code.scope}")

        # Demo 3: Token Exchange
        print("\n🎟️ Demo 3: Token Exchange")
        print("-" * 25)

        # Use authorization code to get access token
        code_valid = storage.use_authorization_code(auth_code.code)
        print(f"✅ Authorization code used: {code_valid}")

        # Create access token
        access_token = storage.create_access_token(
            client_id=web_client.client_id,
            user_id=user_id,
            scope="read write profile",
            expires_in=3600,  # 1 hour
            metadata={
                "grant_type": "authorization_code",
                "device_type": "web_browser"
            }
        )

        print(f"🎫 Access Token ID: {access_token.token_id}")
        print(f"🔑 Access Token: {access_token.access_token[:16]}... (truncated)")
        print(f"⏰ Expires: {access_token.expires_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        # Create refresh token
        refresh_token = storage.create_refresh_token(
            access_token_id=access_token.token_id,
            client_id=web_client.client_id,
            user_id=user_id,
            scope="read write profile",
            expires_in=30 * 24 * 3600  # 30 days
        )

        print(f"🔄 Refresh Token ID: {refresh_token.token_id}")
        print(f"🔑 Refresh Token: {refresh_token.refresh_token[:16]}... (truncated)")

        # Demo 4: Token Validation
        print("\n🔍 Demo 4: Token Validation")
        print("-" * 25)

        # Validate access token
        validated_token = storage.get_access_token(access_token.access_token)
        if validated_token and not validated_token.revoked:
            print("✅ Access token is valid")
            print(f"   User: {validated_token.user_id}")
            print(f"   Client: {validated_token.client_id}")
            print(f"   Scope: {validated_token.scope}")
        else:
            print("❌ Access token is invalid or revoked")

        # Demo 5: Storage Statistics
        print("\n📊 Demo 5: Storage Statistics")
        print("-" * 28)

        stats = storage.get_storage_stats()
        print(f"👥 Registered Clients: {stats['clients']}")
        print(f"🎟️ Active Auth Codes: {stats['active_auth_codes']}")
        print(f"🎫 Active Access Tokens: {stats['active_access_tokens']}")
        print(f"🔄 Active Refresh Tokens: {stats['active_refresh_tokens']}")
        print(f"🎭 Active Sessions: {stats['active_sessions']}")
        print(f"📜 Audit Log Entries: {stats['audit_logs']}")
        print(f"💾 Database Size: {stats['database_size_bytes']} bytes")

        # Demo 6: Audit Trail
        print("\n📜 Demo 6: Audit Trail")
        print("-" * 22)

        audit_logs = storage.get_audit_logs(limit=10)
        print(f"📋 Recent audit events ({len(audit_logs)} entries):")

        for log in audit_logs[:5]:  # Show last 5 events
            timestamp = datetime.fromisoformat(log['timestamp']).strftime('%H:%M:%S')
            event_type = log['event_type']
            entity_type = log['entity_type']
            print(f"   [{timestamp}] {event_type} - {entity_type}")

        if len(audit_logs) > 5:
            print(f"   ... and {len(audit_logs) - 5} more entries")

        # Demo 7: Client Management
        print("\n👥 Demo 7: Client Management")
        print("-" * 27)

        # List all clients
        clients = storage.list_clients(limit=10)
        print(f"📋 Registered clients ({len(clients)} total):")

        for client in clients:
            print(f"   🆔 {client.client_id}")
            print(f"      📱 Name: {client.client_name}")
            print(f"      🔒 Type: {client.client_type.value}")
            print(f"      📅 Created: {client.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print()

        # Demo 8: Transaction Example
        print("💳 Demo 8: Transactional Operations")
        print("-" * 34)

        try:
            with oauth2_transaction(storage):
                # Create multiple related entities in a transaction
                batch_client = storage.register_client(
                    client_name="Batch Processing Client",
                    client_type=ClientType.CONFIDENTIAL,
                    redirect_uris=["https://batch.example.com/callback"],
                    grant_types=[GrantType.CLIENT_CREDENTIALS],
                    scope="batch_process admin"
                )

                batch_token = storage.create_access_token(
                    client_id=batch_client.client_id,
                    user_id="system",
                    scope="batch_process",
                    expires_in=24 * 3600  # 24 hours
                )

                print("✅ Transactional client and token creation successful")
                print(f"   Client: {batch_client.client_id}")
                print(f"   Token: {batch_token.token_id}")

        except Exception as e:
            print(f"❌ Transaction failed: {e}")

        # Demo 9: Security Features
        print("\n🔒 Demo 9: Security Features")
        print("-" * 26)

        print("🔐 Encryption: All sensitive data encrypted at rest with AES-256")
        print("🔑 Key Derivation: PBKDF2-SHA256 with 100,000 iterations")
        print("🛡️ Integrity: HMAC verification for all stored records")
        print("🎲 Tokens: Cryptographically secure random generation")
        print("🧹 Cleanup: Automatic expiration and cleanup of old data")
        print("📜 Audit: Complete audit trail of all operations")

        # Demo 10: Cleanup
        print("\n🧹 Demo 10: Data Cleanup")
        print("-" * 23)

        cleanup_result = storage.cleanup_expired_data()
        print("🗑️ Cleanup completed:")
        for entity_type, count in cleanup_result.items():
            if count > 0:
                print(f"   {entity_type}: {count} items removed")

        total_cleaned = sum(cleanup_result.values())
        if total_cleaned == 0:
            print("   No expired data found")

        print("\n🎉 Demo completed successfully!")
        print(f"💾 Demo data saved to: {Path('demo_oauth2.db').absolute()}")
        print("🔍 You can inspect the database with any SQLite browser")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the cryptography package is installed:")
        print("pip install cryptography")
        return False

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main demo function."""

    # Check if database exists and offer to clean it
    db_path = Path("demo_oauth2.db")
    if db_path.exists():
        print(f"📁 Demo database already exists: {db_path.absolute()}")
        response = input("🗑️ Remove existing database? [y/N]: ").strip().lower()
        if response in ('y', 'yes'):
            os.unlink(db_path)
            print("✅ Existing database removed")
        else:
            print("📊 Will use existing database (may contain previous demo data)")

    success = demo_oauth2_storage()

    if success:
        print("\n" + "=" * 50)
        print("🎉 OAuth2 Storage Demo completed successfully!")
        print("\nNext steps:")
        print("1. Explore the demo database with an SQLite browser")
        print("2. Run the test suite: python test_oauth2_storage.py")
        print("3. Integrate with your OAuth2 authorization server")
        print("=" * 50)

    return success


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
