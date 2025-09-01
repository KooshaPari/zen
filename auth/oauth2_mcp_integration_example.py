"""
OAuth2 JWT Token Management Integration Example for Zen MCP Server

This example demonstrates how to integrate the OAuth2TokenManager with the Zen MCP Server
for secure authentication and authorization of MCP tool access.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from auth.oauth2_tokens import AlgorithmType, TokenStatus, create_oauth2_token_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPAuthenticatedServer:
    """
    MCP Server with OAuth2 JWT authentication.

    This class demonstrates how to integrate OAuth2 token management
    with MCP server operations.
    """

    def __init__(self, issuer: str = "https://zen-mcp-server.com"):
        """Initialize the authenticated MCP server."""
        self.token_manager = create_oauth2_token_manager(
            issuer=issuer,
            algorithm=AlgorithmType.HS256,
            access_token_expiry=3600,      # 1 hour
            refresh_token_expiry=2592000,  # 30 days
            rate_limit_requests=1000,      # 1000 requests per hour
            rate_limit_window=3600
        )

        # MCP tool permissions mapping
        self.tool_permissions = {
            "chat": ["chat:read", "chat:write"],
            "analyze": ["analyze:read", "analyze:execute"],
            "codereview": ["codereview:read", "codereview:write"],
            "planner": ["planner:read", "planner:write"],
            "consensus": ["consensus:read", "consensus:write"],
            "debug": ["debug:read", "debug:execute"],
            "testgen": ["testgen:read", "testgen:write"],
            "refactor": ["refactor:read", "refactor:write"],
            "secaudit": ["secaudit:read", "secaudit:execute"]
        }

        logger.info("MCP Server initialized with OAuth2 authentication")

    async def authenticate_client(
        self,
        client_id: str,
        user_id: str,
        requested_tools: list = None,
        session_metadata: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Authenticate a client and generate tokens for MCP access.

        Args:
            client_id: Client identifier (e.g., "claude_desktop", "zen_web_client")
            user_id: User identifier
            requested_tools: List of requested tools
            session_metadata: Additional session metadata

        Returns:
            Authentication response with tokens
        """
        try:
            # Determine required scopes based on requested tools
            required_scopes = []
            if requested_tools:
                for tool in requested_tools:
                    if tool in self.tool_permissions:
                        required_scopes.extend(self.tool_permissions[tool])
            else:
                # Default: grant read access to basic tools
                required_scopes = ["chat:read", "chat:write", "analyze:read"]

            scope = " ".join(set(required_scopes))  # Remove duplicates

            # Create MCP context
            mcp_context = {
                "server_version": "5.11.0",
                "client_capabilities": ["streaming", "memory", "context"],
                "session_metadata": session_metadata or {},
                "tool_access": {
                    tool: {"granted": True, "permissions": perms}
                    for tool, perms in self.tool_permissions.items()
                    if any(perm in required_scopes for perm in perms)
                },
                "authentication": {
                    "method": "oauth2_jwt",
                    "authenticated_at": datetime.utcnow().isoformat(),
                    "client_id": client_id
                }
            }

            # Generate token pair
            access_token, refresh_token, metadata = self.token_manager.generate_token_pair(
                user_id=user_id,
                client_id=client_id,
                audience=["https://api.zen-mcp.com", "https://tools.zen-mcp.com"],
                scope=scope,
                mcp_context=mcp_context,
                session_id=f"mcp_session_{user_id}_{client_id}",
                ip_address=session_metadata.get("ip_address") if session_metadata else None,
                user_agent=session_metadata.get("user_agent") if session_metadata else None
            )

            return {
                "success": True,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "Bearer",
                "expires_in": metadata["expires_in"],
                "scope": scope,
                "granted_tools": list(mcp_context["tool_access"].keys()),
                "session_id": mcp_context["session_metadata"].get("session_id"),
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Authentication failed for user {user_id}: {e}")
            return {
                "success": False,
                "error": "authentication_failed",
                "error_description": str(e)
            }

    async def authorize_tool_access(
        self,
        token: str,
        tool_name: str,
        operation: str = "execute"
    ) -> dict[str, Any]:
        """
        Authorize tool access using JWT token.

        Args:
            token: JWT access token
            tool_name: Name of the tool to access
            operation: Operation type (read, write, execute)

        Returns:
            Authorization response
        """
        try:
            # Get required permission for tool and operation
            if tool_name not in self.tool_permissions:
                return {
                    "authorized": False,
                    "error": "unknown_tool",
                    "error_description": f"Tool '{tool_name}' is not available"
                }

            # Determine required scope
            tool_perms = self.tool_permissions[tool_name]
            required_scope = None

            for perm in tool_perms:
                if operation in perm or (operation == "execute" and "write" in perm):
                    required_scope = perm
                    break

            if not required_scope:
                required_scope = tool_perms[0]  # Default to first permission

            # Validate token
            status, claims = self.token_manager.validate_token(
                token,
                expected_audience="https://tools.zen-mcp.com",
                required_scope=required_scope
            )

            if status != TokenStatus.VALID:
                return {
                    "authorized": False,
                    "error": "invalid_token",
                    "error_description": f"Token validation failed: {status}",
                    "status": status.value
                }

            # Extract authorization context
            user_id = claims["sub"]
            client_id = claims["client_id"]
            mcp_context = claims.get("mcp_context", {})

            # Check tool-specific permissions
            tool_access = mcp_context.get("tool_access", {})
            if tool_name not in tool_access or not tool_access[tool_name].get("granted", False):
                return {
                    "authorized": False,
                    "error": "access_denied",
                    "error_description": f"Access to tool '{tool_name}' not granted"
                }

            return {
                "authorized": True,
                "user_id": user_id,
                "client_id": client_id,
                "tool_name": tool_name,
                "operation": operation,
                "scope": claims["scope"],
                "session_id": claims.get("session_id"),
                "mcp_context": mcp_context,
                "expires_at": claims["exp"]
            }

        except Exception as e:
            logger.error(f"Authorization failed for tool {tool_name}: {e}")
            return {
                "authorized": False,
                "error": "authorization_failed",
                "error_description": str(e)
            }

    async def refresh_client_tokens(self, refresh_token: str) -> dict[str, Any]:
        """
        Refresh client tokens.

        Args:
            refresh_token: Valid refresh token

        Returns:
            New token pair or error
        """
        try:
            new_access_token, new_refresh_token, metadata = self.token_manager.refresh_token(
                refresh_token
            )

            return {
                "success": True,
                "access_token": new_access_token,
                "refresh_token": new_refresh_token,
                "token_type": "Bearer",
                "expires_in": metadata["expires_in"],
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return {
                "success": False,
                "error": "refresh_failed",
                "error_description": str(e)
            }

    async def revoke_client_access(self, user_id: str, client_id: Optional[str] = None) -> dict[str, Any]:
        """
        Revoke all tokens for a user/client.

        Args:
            user_id: User identifier
            client_id: Optional client identifier

        Returns:
            Revocation response
        """
        try:
            count = self.token_manager.revoke_user_tokens(user_id, client_id)

            return {
                "success": True,
                "revoked_tokens": count,
                "user_id": user_id,
                "client_id": client_id
            }

        except Exception as e:
            logger.error(f"Token revocation failed for user {user_id}: {e}")
            return {
                "success": False,
                "error": "revocation_failed",
                "error_description": str(e)
            }

    def get_server_statistics(self) -> dict[str, Any]:
        """Get server authentication statistics."""
        return self.token_manager.get_token_statistics()


# Example usage and testing
async def demonstrate_mcp_oauth_integration():
    """Demonstrate OAuth2 integration with MCP server."""
    print("🔐 OAuth2 JWT Integration with Zen MCP Server")
    print("=" * 50)

    # Initialize authenticated MCP server
    mcp_server = MCPAuthenticatedServer()

    # 1. Authenticate Claude Desktop client
    print("\n1. Authenticating Claude Desktop client...")
    auth_response = await mcp_server.authenticate_client(
        client_id="claude_desktop",
        user_id="developer_alice",
        requested_tools=["chat", "analyze", "codereview"],
        session_metadata={
            "ip_address": "192.168.1.100",
            "user_agent": "Claude Desktop 1.0",
            "client_version": "1.0.0"
        }
    )

    if auth_response["success"]:
        print("✓ Authentication successful!")
        print(f"  Access Token: {auth_response['access_token'][:50]}...")
        print(f"  Granted Tools: {auth_response['granted_tools']}")
        print(f"  Expires in: {auth_response['expires_in']} seconds")

        access_token = auth_response["access_token"]
        refresh_token = auth_response["refresh_token"]
    else:
        print(f"✗ Authentication failed: {auth_response['error']}")
        return

    # 2. Authorize tool access
    print("\n2. Authorizing tool access...")

    # Test chat tool access
    chat_auth = await mcp_server.authorize_tool_access(
        access_token,
        "chat",
        "write"
    )
    print(f"Chat tool authorization: {'✓ Authorized' if chat_auth['authorized'] else '✗ Denied'}")

    # Test analyze tool access
    analyze_auth = await mcp_server.authorize_tool_access(
        access_token,
        "analyze",
        "execute"
    )
    print(f"Analyze tool authorization: {'✓ Authorized' if analyze_auth['authorized'] else '✗ Denied'}")

    # Test unauthorized tool
    admin_auth = await mcp_server.authorize_tool_access(
        access_token,
        "admin_panel",
        "execute"
    )
    print(f"Admin panel authorization: {'✓ Authorized' if admin_auth['authorized'] else '✗ Denied (Expected)'}")

    # 3. Token refresh
    print("\n3. Testing token refresh...")
    refresh_response = await mcp_server.refresh_client_tokens(refresh_token)
    if refresh_response["success"]:
        print("✓ Token refresh successful!")
        new_access_token = refresh_response["access_token"]

        # Test with new token
        new_auth = await mcp_server.authorize_tool_access(
            new_access_token,
            "chat",
            "write"
        )
        print(f"New token authorization: {'✓ Works' if new_auth['authorized'] else '✗ Failed'}")
    else:
        print(f"✗ Token refresh failed: {refresh_response['error']}")

    # 4. Server statistics
    print("\n4. Server statistics...")
    stats = mcp_server.get_server_statistics()
    print(f"✓ Total tokens: {stats['total_tokens']}")
    print(f"✓ Active tokens: {stats['active_tokens']}")
    print(f"✓ Algorithm: {stats['algorithm']}")

    # 5. Revoke access
    print("\n5. Revoking client access...")
    revoke_response = await mcp_server.revoke_client_access(
        "developer_alice",
        "claude_desktop"
    )
    if revoke_response["success"]:
        print(f"✓ Revoked {revoke_response['revoked_tokens']} tokens")

        # Test revoked token
        revoked_auth = await mcp_server.authorize_tool_access(
            access_token,
            "chat",
            "write"
        )
        print(f"Revoked token test: {'✗ Still works (unexpected)' if revoked_auth['authorized'] else '✓ Properly denied'}")

    print("\n" + "=" * 50)
    print("🎉 MCP OAuth2 Integration Demo Complete!")


# Example middleware for FastAPI/HTTP server integration
class OAuth2MCPMiddleware:
    """Middleware for integrating OAuth2 tokens with HTTP endpoints."""

    def __init__(self, mcp_server: MCPAuthenticatedServer):
        self.mcp_server = mcp_server

    async def authenticate_request(self, authorization_header: Optional[str]) -> dict[str, Any]:
        """
        Authenticate HTTP request using Authorization header.

        Args:
            authorization_header: Authorization header value

        Returns:
            Authentication context or error
        """
        if not authorization_header:
            return {"error": "missing_authorization_header"}

        try:
            scheme, token = authorization_header.split(" ", 1)
            if scheme.lower() != "bearer":
                return {"error": "invalid_authorization_scheme"}
        except ValueError:
            return {"error": "malformed_authorization_header"}

        # Validate token for API access
        status, claims = self.mcp_server.token_manager.validate_token(
            token,
            expected_audience="https://api.zen-mcp.com"
        )

        if status != TokenStatus.VALID:
            return {"error": "invalid_token", "status": status.value}

        return {
            "authenticated": True,
            "user_id": claims["sub"],
            "client_id": claims["client_id"],
            "scope": claims["scope"],
            "mcp_context": claims.get("mcp_context", {}),
            "session_id": claims.get("session_id")
        }


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_mcp_oauth_integration())
