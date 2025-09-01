#!/usr/bin/env python3
"""
OAuth 2.0 Integration Example for Zen MCP Server

This example demonstrates how to integrate the OAuth 2.0 authorization server
with the existing MCP HTTP server to provide secure, token-based authentication.

Features demonstrated:
- OAuth 2.0 server setup with WebAuthn integration
- Authorization endpoint configuration
- Token endpoint for client credentials
- Bearer token validation middleware
- PKCE support for public clients
- Integration with existing MCP tools

Usage:
    python examples/oauth2_integration.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import uvicorn
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from auth.oauth2_server import OAuth2Middleware, create_oauth2_server
from auth.webauthn_flow import WebAuthnDeviceAuth


class OAuth2IntegrationDemo:
    """Demo application showing OAuth 2.0 integration with MCP server."""

    def __init__(self):
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI required for demo. Install with: pip install fastapi uvicorn")

        self.app = FastAPI(
            title="Zen MCP Server with OAuth 2.0",
            description="OAuth 2.0 secured MCP server with WebAuthn integration",
            version="1.0.0"
        )

        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, specify exact origins
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

        # Initialize OAuth 2.0 server
        self.webauthn = WebAuthnDeviceAuth()
        self.oauth2_server = create_oauth2_server(
            issuer="https://mcp.demo.local",
            webauthn_auth=self.webauthn
        )

        # Add OAuth 2.0 middleware
        oauth2_middleware = OAuth2Middleware(self.oauth2_server)
        self.app.middleware("http")(oauth2_middleware)

        # Setup endpoints
        self.setup_oauth_endpoints()
        self.setup_mcp_endpoints()
        self.setup_demo_endpoints()

    def setup_oauth_endpoints(self):
        """Setup OAuth 2.0 protocol endpoints."""

        @self.app.get("/oauth/authorize")
        async def authorize_endpoint(request: Request):
            """OAuth 2.0 Authorization Endpoint"""
            return await self.oauth2_server.authorization_endpoint(request)

        @self.app.post("/oauth/token")
        async def token_endpoint(request: Request):
            """OAuth 2.0 Token Endpoint"""
            return await self.oauth2_server.token_endpoint(request)

        @self.app.post("/oauth/revoke")
        async def revoke_endpoint(request: Request):
            """OAuth 2.0 Token Revocation Endpoint"""
            return await self.oauth2_server.revocation_endpoint(request)

        @self.app.post("/oauth/introspect")
        async def introspect_endpoint(request: Request):
            """OAuth 2.0 Token Introspection Endpoint"""
            return await self.oauth2_server.introspection_endpoint(request)

    def setup_mcp_endpoints(self):
        """Setup MCP protocol endpoints with OAuth 2.0 protection."""

        @self.app.get("/mcp/tools")
        async def list_tools(request: Request):
            """List available MCP tools (protected endpoint)"""
            # This endpoint requires OAuth 2.0 authentication
            # The middleware will validate the Bearer token
            token_info = getattr(request.state, 'oauth_token', None)

            return JSONResponse(content={
                "tools": [
                    {"name": "chat", "description": "Chat with AI assistant"},
                    {"name": "analyze", "description": "Analyze code and files"},
                    {"name": "debug", "description": "Debug code and troubleshoot"},
                    {"name": "planner", "description": "Create project plans"}
                ],
                "authenticated_user": token_info.get("user_id") if token_info else None,
                "client_id": token_info.get("client_id") if token_info else None
            })

        @self.app.post("/mcp/tools/{tool_name}")
        async def execute_tool(tool_name: str, request: Request):
            """Execute an MCP tool (protected endpoint)"""
            token_info = getattr(request.state, 'oauth_token', None)

            # Parse request body
            try:
                body = await request.json()
            except:
                body = {}

            # Simulate tool execution
            result = {
                "tool": tool_name,
                "input": body,
                "result": f"Executed {tool_name} successfully",
                "authenticated_user": token_info.get("user_id") if token_info else None,
                "timestamp": "2023-01-01T12:00:00Z"
            }

            return JSONResponse(content=result)

    def setup_demo_endpoints(self):
        """Setup demo and documentation endpoints."""

        @self.app.get("/")
        async def root():
            """Root endpoint with OAuth 2.0 information"""
            return JSONResponse(content={
                "name": "Zen MCP Server with OAuth 2.0",
                "version": "1.0.0",
                "oauth2": {
                    "issuer": self.oauth2_server.issuer,
                    "authorization_endpoint": "/oauth/authorize",
                    "token_endpoint": "/oauth/token",
                    "revocation_endpoint": "/oauth/revoke",
                    "introspection_endpoint": "/oauth/introspect",
                    "scopes_supported": list(self.oauth2_server.available_scopes.keys()),
                    "response_types_supported": ["code"],
                    "grant_types_supported": ["authorization_code", "refresh_token"],
                    "code_challenge_methods_supported": ["S256", "plain"]
                },
                "endpoints": {
                    "protected": ["/mcp/tools", "/mcp/tools/{tool_name}"],
                    "public": ["/", "/demo", "/oauth/*"]
                }
            })

        @self.app.get("/demo", response_class=HTMLResponse)
        async def demo_page():
            """OAuth 2.0 demo page with PKCE flow"""
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>OAuth 2.0 Demo - Zen MCP Server</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                    button { background: #007AFF; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 16px; cursor: pointer; margin: 5px; }
                    button:hover { background: #0056CC; }
                    button:disabled { background: #ccc; cursor: not-allowed; }
                    .error { color: #FF3B30; }
                    .success { color: #34C759; }
                    .info { color: #007AFF; }
                    .code-block { background: #f5f5f5; padding: 15px; border-radius: 8px; overflow-x: auto; }
                    input { width: 100%; padding: 12px; margin: 8px 0; border: 1px solid #ccc; border-radius: 8px; box-sizing: border-box; }
                    .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
                </style>
            </head>
            <body>
                <h1>🔐 OAuth 2.0 Demo - Zen MCP Server</h1>

                <div class="section">
                    <h2>OAuth 2.0 Configuration</h2>
                    <div class="code-block">
                        <strong>Authorization Endpoint:</strong> /oauth/authorize<br>
                        <strong>Token Endpoint:</strong> /oauth/token<br>
                        <strong>Client ID:</strong> mcp-default-client<br>
                        <strong>Redirect URI:</strong> http://localhost:8080/oauth/callback<br>
                        <strong>PKCE:</strong> Required (S256)
                    </div>
                </div>

                <div class="section">
                    <h2>Step 1: OAuth 2.0 Authorization Flow</h2>
                    <p>Start the OAuth 2.0 authorization flow with PKCE:</p>
                    <button onclick="startOAuthFlow()">Start Authorization Flow</button>
                    <div id="auth-status"></div>
                </div>

                <div class="section">
                    <h2>Step 2: Access Token</h2>
                    <p>After authorization, your access token will appear here:</p>
                    <input type="text" id="access-token" placeholder="Access token will appear here after authorization" readonly>
                    <button onclick="testToken()" id="test-token-btn" disabled>Test Access Token</button>
                </div>

                <div class="section">
                    <h2>Step 3: Protected MCP Endpoints</h2>
                    <p>Use your access token to call protected MCP endpoints:</p>
                    <button onclick="callProtectedEndpoint('/mcp/tools')" id="call-tools-btn" disabled>GET /mcp/tools</button>
                    <button onclick="callProtectedEndpoint('/mcp/tools/chat')" id="call-chat-btn" disabled>POST /mcp/tools/chat</button>
                    <div id="api-results"></div>
                </div>

                <script>
                    let codeVerifier = '';
                    let accessToken = '';

                    // Generate PKCE code verifier and challenge
                    function generateCodeVerifier() {
                        const array = new Uint8Array(32);
                        crypto.getRandomValues(array);
                        return base64URLEncode(array);
                    }

                    async function generateCodeChallenge(verifier) {
                        const encoder = new TextEncoder();
                        const data = encoder.encode(verifier);
                        const digest = await crypto.subtle.digest('SHA-256', data);
                        return base64URLEncode(new Uint8Array(digest));
                    }

                    function base64URLEncode(arr) {
                        return btoa(String.fromCharCode.apply(null, arr))
                            .replace(/\\\\+/g, '-')
                            .replace(/\\\\//g, '_')
                            .replace(/=/g, '');
                    }

                    async function startOAuthFlow() {
                        try {
                            // Generate PKCE parameters
                            codeVerifier = generateCodeVerifier();
                            const codeChallenge = await generateCodeChallenge(codeVerifier);

                            // Build authorization URL
                            const params = new URLSearchParams({
                                response_type: 'code',
                                client_id: 'mcp-default-client',
                                redirect_uri: window.location.origin + '/oauth/callback',
                                scope: 'mcp:read mcp:write',
                                state: 'demo-state-' + Math.random(),
                                code_challenge: codeChallenge,
                                code_challenge_method: 'S256'
                            });

                            const authUrl = `/oauth/authorize?${params.toString()}`;

                            document.getElementById('auth-status').innerHTML =
                                `<div class="info">Redirecting to authorization endpoint...<br><a href="${authUrl}" target="_blank">Click here if not redirected</a></div>`;

                            // In a real app, this would redirect the user
                            // For demo purposes, we'll simulate the callback
                            setTimeout(() => simulateCallback(), 2000);

                        } catch (error) {
                            document.getElementById('auth-status').innerHTML =
                                `<div class="error">Error: ${error.message}</div>`;
                        }
                    }

                    async function simulateCallback() {
                        try {
                            // Simulate authorization code callback
                            const authCode = 'simulated-auth-code-' + Math.random();

                            // Exchange code for tokens
                            const tokenResponse = await fetch('/oauth/token', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                                body: new URLSearchParams({
                                    grant_type: 'authorization_code',
                                    code: authCode,
                                    redirect_uri: window.location.origin + '/oauth/callback',
                                    client_id: 'mcp-default-client',
                                    code_verifier: codeVerifier
                                })
                            });

                            if (tokenResponse.ok) {
                                const tokens = await tokenResponse.json();
                                accessToken = tokens.access_token;

                                document.getElementById('access-token').value = accessToken;
                                document.getElementById('test-token-btn').disabled = false;
                                document.getElementById('call-tools-btn').disabled = false;
                                document.getElementById('call-chat-btn').disabled = false;

                                document.getElementById('auth-status').innerHTML =
                                    '<div class="success">✅ Authorization successful! Access token received.</div>';
                            } else {
                                const error = await tokenResponse.json();
                                throw new Error(`Token exchange failed: ${error.error_description || error.error}`);
                            }

                        } catch (error) {
                            document.getElementById('auth-status').innerHTML =
                                `<div class="error">Token exchange error: ${error.message}</div>`;
                        }
                    }

                    async function testToken() {
                        try {
                            const response = await fetch('/oauth/introspect', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                                body: new URLSearchParams({ token: accessToken })
                            });

                            const result = await response.json();

                            if (result.active) {
                                document.getElementById('auth-status').innerHTML +=
                                    '<div class="success">✅ Token is valid and active!</div>';
                            } else {
                                document.getElementById('auth-status').innerHTML +=
                                    '<div class="error">❌ Token is not active.</div>';
                            }

                        } catch (error) {
                            document.getElementById('auth-status').innerHTML +=
                                `<div class="error">Token test error: ${error.message}</div>`;
                        }
                    }

                    async function callProtectedEndpoint(endpoint) {
                        try {
                            const method = endpoint.includes('/tools/') ? 'POST' : 'GET';
                            const body = method === 'POST' ? JSON.stringify({
                                message: 'Hello from OAuth 2.0 demo!'
                            }) : undefined;

                            const response = await fetch(endpoint, {
                                method: method,
                                headers: {
                                    'Authorization': `Bearer ${accessToken}`,
                                    'Content-Type': 'application/json'
                                },
                                body: body
                            });

                            const result = await response.json();

                            document.getElementById('api-results').innerHTML =
                                `<div class="code-block"><strong>${method} ${endpoint}</strong><br><pre>${JSON.stringify(result, null, 2)}</pre></div>`;

                        } catch (error) {
                            document.getElementById('api-results').innerHTML =
                                `<div class="error">API call error: ${error.message}</div>`;
                        }
                    }
                </script>
            </body>
            </html>
            """
            return html_content

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint (public)"""
            return JSONResponse(content={
                "status": "healthy",
                "oauth2_server": "active",
                "webauthn": "available",
                "timestamp": "2023-01-01T12:00:00Z"
            })

    async def start_server(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the OAuth 2.0 integrated MCP server."""
        print(f"🚀 Starting Zen MCP Server with OAuth 2.0 on {host}:{port}")
        print(f"📋 OAuth 2.0 Demo: http://{host}:{port}/demo")
        print(f"🔐 Authorization Endpoint: http://{host}:{port}/oauth/authorize")
        print(f"🎫 Token Endpoint: http://{host}:{port}/oauth/token")
        print(f"🛡️  Protected MCP Endpoints: http://{host}:{port}/mcp/*")

        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )

        server = uvicorn.Server(config)
        await server.serve()


async def main():
    """Main demo function."""
    demo = OAuth2IntegrationDemo()
    await demo.start_server()


if __name__ == "__main__":
    asyncio.run(main())
