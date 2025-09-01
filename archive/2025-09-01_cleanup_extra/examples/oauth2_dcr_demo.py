#!/usr/bin/env python3
"""
OAuth 2.0 Dynamic Client Registration (DCR) Demo

This script demonstrates how to use the Zen MCP Server's RFC 7591 compliant
Dynamic Client Registration system. It shows client registration, management,
and integration with the MCP protocol.

Features demonstrated:
- Client registration with various configurations
- Client authentication and authorization
- MCP-specific extensions and capabilities
- Client management operations (update, delete)
- Error handling and security features

Usage:
    python examples/oauth2_dcr_demo.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from auth.oauth2_dcr import (
        ClientAuthMethod,
        ClientMetadata,
        ClientType,
        DCRError,
        DCRManager,
        GrantType,
        ResponseType,
        get_dcr_manager,
    )
    from auth.oauth2_integration import get_oauth_config, validate_oauth_config
    from utils.audit_trail import get_audit_manager
    DCR_AVAILABLE = True
except ImportError as e:
    logger.error(f"DCR modules not available: {e}")
    DCR_AVAILABLE = False


async def demo_basic_client_registration():
    """Demonstrate basic client registration."""
    print("\n=== Basic Client Registration ===")

    # Create DCR manager
    dcr_manager = DCRManager()

    # Register a basic confidential client
    metadata = ClientMetadata(
        client_name="Demo Web Application",
        client_uri="https://demo.example.com",
        redirect_uris=["https://demo.example.com/oauth/callback"],
        client_type=ClientType.CONFIDENTIAL,
        token_endpoint_auth_method=ClientAuthMethod.CLIENT_SECRET_BASIC,
        grant_types=[GrantType.AUTHORIZATION_CODE, GrantType.REFRESH_TOKEN],
        response_types=[ResponseType.CODE],
        scope="read write mcp:tools",
        contacts=["admin@demo.example.com"],
        logo_uri="https://demo.example.com/logo.png",
        tos_uri="https://demo.example.com/terms"
    )

    try:
        registered_client = await dcr_manager.register_client(
            metadata=metadata,
            client_ip="192.168.1.100",
            user_agent="DCR Demo Script/1.0"
        )

        print("✅ Successfully registered client:")
        print(f"   Client ID: {registered_client.client_id}")
        print(f"   Client Name: {registered_client.client_name}")
        print(f"   Client Type: {registered_client.client_type.value}")
        print(f"   Auth Method: {registered_client.token_endpoint_auth_method.value}")
        print(f"   Grant Types: {[gt.value for gt in registered_client.grant_types]}")
        print(f"   Scope: {registered_client.scope}")
        print(f"   Has Client Secret: {'Yes' if registered_client.client_secret else 'No'}")
        print(f"   Registration Token: {registered_client.registration_access_token[:20]}...")

        return registered_client

    except DCRError as e:
        print(f"❌ Registration failed: {e.error_code} - {e.error_description}")
        return None


async def demo_public_client_registration():
    """Demonstrate public client registration."""
    print("\n=== Public Client Registration ===")

    dcr_manager = DCRManager()

    # Register a public client (mobile/SPA)
    metadata = ClientMetadata(
        client_name="Demo Mobile App",
        client_uri="https://mobile-app.example.com",
        redirect_uris=["com.example.demoapp://oauth/callback"],
        client_type=ClientType.PUBLIC,
        token_endpoint_auth_method=ClientAuthMethod.NONE,
        grant_types=[GrantType.AUTHORIZATION_CODE],
        response_types=[ResponseType.CODE],
        scope="read mcp:resources"
    )

    try:
        registered_client = await dcr_manager.register_client(
            metadata=metadata,
            client_ip="192.168.1.101",
            user_agent="Mobile App Registration/1.0"
        )

        print("✅ Successfully registered public client:")
        print(f"   Client ID: {registered_client.client_id}")
        print(f"   Client Name: {registered_client.client_name}")
        print(f"   Client Type: {registered_client.client_type.value}")
        print(f"   Redirect URI: {registered_client.redirect_uris[0]}")
        print(f"   No Client Secret: {registered_client.client_secret is None}")

        return registered_client

    except DCRError as e:
        print(f"❌ Public client registration failed: {e.error_code} - {e.error_description}")
        return None


async def demo_mcp_client_registration():
    """Demonstrate MCP-specific client registration."""
    print("\n=== MCP Client Registration ===")

    dcr_manager = DCRManager()

    # Register an MCP client with capabilities
    metadata = ClientMetadata(
        client_name="MCP Development Client",
        client_uri="https://mcp-dev.example.com",
        redirect_uris=["https://mcp-dev.example.com/auth/callback"],
        client_type=ClientType.CONFIDENTIAL,
        grant_types=[GrantType.CLIENT_CREDENTIALS, GrantType.AUTHORIZATION_CODE],
        scope="mcp:tools mcp:resources mcp:prompts",

        # MCP-specific extensions
        mcp_capabilities=["tools", "resources", "prompts", "logging"],
        mcp_transport_protocols=["http", "websocket"],
        mcp_session_timeout=7200,  # 2 hours

        # Software information
        software_id="mcp-dev-client",
        software_version="1.2.0",
        contacts=["dev@example.com"]
    )

    try:
        registered_client = await dcr_manager.register_client(
            metadata=metadata,
            client_ip="192.168.1.102",
            user_agent="MCP Development Client/1.2.0"
        )

        print("✅ Successfully registered MCP client:")
        print(f"   Client ID: {registered_client.client_id}")
        print(f"   MCP Capabilities: {registered_client.mcp_capabilities}")
        print(f"   Transport Protocols: {registered_client.mcp_transport_protocols}")
        print(f"   Session Timeout: {registered_client.mcp_session_timeout}s")
        print(f"   Client Credentials Grant: {'Yes' if GrantType.CLIENT_CREDENTIALS in registered_client.grant_types else 'No'}")

        return registered_client

    except DCRError as e:
        print(f"❌ MCP client registration failed: {e.error_code} - {e.error_description}")
        return None


async def demo_client_management(client):
    """Demonstrate client management operations."""
    print("\n=== Client Management Operations ===")

    if not client:
        print("❌ No client available for management demo")
        return

    dcr_manager = DCRManager()

    # Retrieve client information
    print("1. Retrieving client information...")
    retrieved_client = await dcr_manager.get_client(client.client_id)
    if retrieved_client:
        print(f"   ✅ Retrieved client: {retrieved_client.client_name}")
    else:
        print("   ❌ Failed to retrieve client")
        return

    # Update client metadata
    print("2. Updating client metadata...")
    updated_metadata = ClientMetadata(
        client_name="Updated Demo Client",
        client_uri="https://updated.example.com",
        redirect_uris=["https://updated.example.com/oauth/callback"],
        client_type=client.client_type,
        token_endpoint_auth_method=client.token_endpoint_auth_method,
        grant_types=client.grant_types,
        response_types=client.response_types,
        scope="read write admin mcp:tools",  # Added admin scope
        contacts=["admin@updated.example.com"]
    )

    try:
        updated_client = await dcr_manager.update_client(
            client_id=client.client_id,
            metadata=updated_metadata,
            registration_token=client.registration_access_token,
            client_ip="192.168.1.100"
        )
        print("   ✅ Client updated successfully")
        print(f"   New name: {updated_client.client_name}")
        print(f"   New scope: {updated_client.scope}")

    except DCRError as e:
        print(f"   ❌ Update failed: {e.error_code} - {e.error_description}")
        return

    # Test client authentication
    print("3. Testing client authentication...")
    if client.client_type == ClientType.CONFIDENTIAL:
        is_authenticated = await dcr_manager.authenticate_client(
            client_id=client.client_id,
            client_secret=client.client_secret
        )
        print(f"   Authentication result: {'✅ Valid' if is_authenticated else '❌ Invalid'}")

        # Test with wrong secret
        is_authenticated = await dcr_manager.authenticate_client(
            client_id=client.client_id,
            client_secret="wrong_secret"
        )
        print(f"   Wrong secret test: {'❌ Should be invalid' if is_authenticated else '✅ Correctly rejected'}")
    else:
        is_authenticated = await dcr_manager.authenticate_client(client_id=client.client_id)
        print(f"   Public client auth: {'✅ Valid' if is_authenticated else '❌ Invalid'}")


async def demo_client_deletion(client):
    """Demonstrate client deletion."""
    print("\n=== Client Deletion ===")

    if not client:
        print("❌ No client available for deletion demo")
        return

    dcr_manager = DCRManager()

    # Confirm client exists before deletion
    existing_client = await dcr_manager.get_client(client.client_id)
    if not existing_client:
        print("❌ Client not found for deletion")
        return

    print(f"Deleting client: {existing_client.client_name} ({existing_client.client_id})")

    try:
        success = await dcr_manager.delete_client(
            client_id=client.client_id,
            registration_token=client.registration_access_token,
            client_ip="192.168.1.100"
        )

        if success:
            print("✅ Client deleted successfully")

            # Verify deletion
            deleted_client = await dcr_manager.get_client(client.client_id)
            if deleted_client is None:
                print("✅ Client deletion confirmed")
            else:
                print("❌ Client still exists after deletion")
        else:
            print("❌ Client deletion failed")

    except DCRError as e:
        print(f"❌ Deletion failed: {e.error_code} - {e.error_description}")


async def demo_rate_limiting():
    """Demonstrate rate limiting functionality."""
    print("\n=== Rate Limiting Demo ===")

    # Create manager with very restrictive rate limits
    rate_limited_manager = DCRManager(rate_limit_per_ip=2, rate_limit_window=60)

    metadata = ClientMetadata(
        client_name="Rate Limit Test Client",
        redirect_uris=["https://ratelimit.example.com/callback"]
    )

    client_ip = "192.168.1.200"
    successful_registrations = 0

    print("Attempting multiple registrations from same IP...")

    for i in range(4):
        try:
            await rate_limited_manager.register_client(
                metadata=metadata,
                client_ip=client_ip
            )
            successful_registrations += 1
            print(f"   Registration {i+1}: ✅ Success")

        except DCRError as e:
            if e.error_code == "too_many_requests":
                print(f"   Registration {i+1}: ⚠️  Rate limited (as expected)")
            else:
                print(f"   Registration {i+1}: ❌ Unexpected error: {e.error_code}")

    print(f"Rate limiting test: {successful_registrations}/4 registrations succeeded")


async def demo_discovery_metadata():
    """Demonstrate DCR discovery metadata."""
    print("\n=== DCR Discovery Metadata ===")

    dcr_manager = DCRManager()
    metadata = dcr_manager.get_registration_metadata()

    print("OAuth 2.0 DCR Server Metadata:")
    print(json.dumps(metadata, indent=2))


def demo_configuration():
    """Demonstrate configuration validation."""
    print("\n=== Configuration Demo ===")

    # Get current configuration
    config = get_oauth_config()
    print("Current OAuth DCR Configuration:")
    for key, value in config.items():
        if 'secret' in key.lower() or 'key' in key.lower():
            # Mask sensitive values
            value_str = str(value)
            display_value = f"***{value_str[-4:]}" if value and len(value_str) > 4 else "***"
        else:
            display_value = value
        print(f"  {key}: {display_value}")

    # Validate configuration
    is_valid, warnings = validate_oauth_config()
    print(f"\nConfiguration Valid: {'✅ Yes' if is_valid else '❌ No'}")

    if warnings:
        print("Configuration Warnings:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    else:
        print("✅ No configuration warnings")


async def demo_error_handling():
    """Demonstrate error handling."""
    print("\n=== Error Handling Demo ===")

    dcr_manager = DCRManager()

    # Test invalid metadata
    print("1. Testing invalid metadata...")
    try:
        invalid_metadata = ClientMetadata(
            client_name="Invalid Client",
            client_type=ClientType.PUBLIC,
            token_endpoint_auth_method=ClientAuthMethod.CLIENT_SECRET_BASIC  # Invalid for public client
        )
        await dcr_manager.register_client(invalid_metadata, "192.168.1.1")
    except DCRError as e:
        print(f"   ✅ Caught expected error: {e.error_code} - {e.error_description}")

    # Test non-existent client operations
    print("2. Testing operations on non-existent client...")
    try:
        await dcr_manager.get_client("non_existent_client_id")
        print("   ⚠️  Non-existent client returned something (unexpected)")
    except:
        pass  # This might not throw an exception, just return None

    # Test invalid token
    print("3. Testing invalid registration token...")
    try:
        await dcr_manager.update_client(
            client_id="some_client",
            metadata=ClientMetadata(client_name="Test"),
            registration_token="invalid_token",
            client_ip="192.168.1.1"
        )
    except DCRError as e:
        print(f"   ✅ Caught expected error: {e.error_code} - {e.error_description}")


async def main():
    """Main demo function."""
    print("🔐 OAuth 2.0 Dynamic Client Registration (DCR) Demo")
    print("=" * 60)

    if not DCR_AVAILABLE:
        print("❌ DCR modules not available. Please check your installation.")
        return

    # Configuration demo
    demo_configuration()

    # Discovery metadata
    await demo_discovery_metadata()

    # Basic client registration
    basic_client = await demo_basic_client_registration()

    # Public client registration
    public_client = await demo_public_client_registration()

    # MCP client registration
    mcp_client = await demo_mcp_client_registration()

    # Client management operations
    if basic_client:
        await demo_client_management(basic_client)

    # Rate limiting demo
    await demo_rate_limiting()

    # Error handling demo
    await demo_error_handling()

    # Clean up - delete test clients
    print("\n=== Cleanup ===")
    for client in [basic_client, public_client, mcp_client]:
        if client:
            try:
                await demo_client_deletion(client)
            except Exception as e:
                print(f"⚠️  Failed to delete client {client.client_id}: {e}")

    print("\n✅ Demo completed successfully!")
    print("\nNext steps:")
    print("1. Start the MCP HTTP server with OAuth DCR enabled")
    print("2. Use HTTP clients to test the /oauth/register endpoints")
    print("3. Integrate with your OAuth 2.0 authorization flows")
    print("4. Configure production security settings")


if __name__ == "__main__":
    asyncio.run(main())
