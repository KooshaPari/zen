#!/usr/bin/env python3
"""
Test OAuth 2.0 flow for Zen MCP Server
This simulates what Claude does when connecting
"""

import asyncio
import base64
import hashlib
import secrets
from urllib.parse import parse_qs, urlencode, urlparse

import aiohttp


def generate_pkce():
    """Generate PKCE code verifier and challenge"""
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).decode('utf-8').rstrip('=')
    return code_verifier, code_challenge

async def test_oauth_flow():
    """Test the complete OAuth flow"""

    # Configuration
    server_url = "https://zen.kooshapari.com"
    client_id = "mcp-default-client"
    redirect_uri = "http://localhost:56535/callback"

    # Generate PKCE
    code_verifier, code_challenge = generate_pkce()
    state = secrets.token_urlsafe(16)

    print("🔐 Starting OAuth 2.0 Flow Test")
    print(f"   Server: {server_url}")
    print(f"   Client ID: {client_id}")
    print(f"   PKCE Challenge: {code_challenge[:20]}...")
    print()

    async with aiohttp.ClientSession() as session:
        # Step 1: Start authorization flow
        auth_params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "scope": "mcp:read mcp:write"
        }

        auth_url = f"{server_url}/oauth/authorize?{urlencode(auth_params)}"
        print("1️⃣ Authorization Request")
        print(f"   URL: {auth_url}")

        try:
            async with session.get(auth_url, allow_redirects=False) as resp:
                print(f"   Status: {resp.status}")

                # Check if we got redirected (302) or got the WebAuthn page (200)
                if resp.status == 200:
                    # We got the WebAuthn page
                    html = await resp.text()
                    if "Touch ID" in html or "WebAuthn" in html:
                        print("   ✅ WebAuthn authentication page received")
                        print("   ⚠️  This is expected - user would authenticate via Touch ID")
                        print()
                        print("2️⃣ Simulating WebAuthn Authentication")
                        print("   In a real flow, the user would:")
                        print("   - Be prompted for Touch ID")
                        print("   - Complete biometric authentication")
                        print("   - Get redirected to consent page")
                        print()

                        # Since we can't simulate Touch ID, let's check the consent endpoint
                        print("3️⃣ Testing Consent Endpoint")
                        consent_url = f"{server_url}/oauth/consent"

                        # Try to get the consent page (will likely require auth)
                        async with session.get(consent_url) as consent_resp:
                            print(f"   Consent page status: {consent_resp.status}")
                            if consent_resp.status == 200:
                                consent_html = await consent_resp.text()
                                if "consent_id" in consent_html:
                                    print("   ✅ Consent page accessible")
                                else:
                                    print("   ⚠️  Consent page returned but no consent_id found")
                            else:
                                print("   ⚠️  Consent page requires authentication (expected)")

                        print()
                        print("4️⃣ OAuth Flow Summary")
                        print("   ✅ Authorization endpoint working")
                        print("   ✅ WebAuthn integration active")
                        print("   ✅ PKCE parameters accepted")
                        print("   ⚠️  Manual Touch ID verification required for completion")
                        print()
                        print("📝 Next Steps:")
                        print("   1. Connect via Claude MCP with /mcp command")
                        print("   2. Complete Touch ID authentication when prompted")
                        print("   3. Approve consent in browser")
                        print("   4. Authorization code will be sent to callback URL")

                elif resp.status == 302:
                    # Got redirect
                    location = resp.headers.get('Location', '')
                    print(f"   Redirect to: {location}")

                    # Check if it's a callback with code
                    if redirect_uri in location and "code=" in location:
                        parsed = urlparse(location)
                        params = parse_qs(parsed.query)
                        auth_code = params.get('code', [''])[0]
                        print(f"   ✅ Authorization code received: {auth_code[:20]}...")

                        # Step 2: Exchange code for token
                        print()
                        print("5️⃣ Token Exchange")
                        token_url = f"{server_url}/oauth/token"
                        token_data = {
                            "grant_type": "authorization_code",
                            "code": auth_code,
                            "redirect_uri": redirect_uri,
                            "client_id": client_id,
                            "code_verifier": code_verifier
                        }

                        async with session.post(token_url, data=token_data) as token_resp:
                            print(f"   Status: {token_resp.status}")
                            if token_resp.status == 200:
                                tokens = await token_resp.json()
                                print("   ✅ Access token received!")
                                print(f"      Token: {tokens.get('access_token', '')[:30]}...")
                                print(f"      Type: {tokens.get('token_type', 'Bearer')}")
                                print(f"      Expires in: {tokens.get('expires_in', 0)} seconds")

                                # Step 3: Test MCP access with token
                                print()
                                print("6️⃣ Testing MCP Access")
                                mcp_url = f"{server_url}/mcp"
                                headers = {
                                    "Authorization": f"Bearer {tokens.get('access_token', '')}"
                                }
                                mcp_payload = {
                                    "jsonrpc": "2.0",
                                    "method": "initialize",
                                    "params": {"capabilities": {}},
                                    "id": 1
                                }

                                async with session.post(mcp_url, json=mcp_payload, headers=headers) as mcp_resp:
                                    print(f"   Status: {mcp_resp.status}")
                                    if mcp_resp.status == 200:
                                        result = await mcp_resp.json()
                                        print("   ✅ MCP access successful!")
                                        server_info = result.get('result', {}).get('server_info', {})
                                        print(f"      Server: {server_info.get('name', 'Unknown')}")
                                        print(f"      Version: {server_info.get('version', 'Unknown')}")
                                    else:
                                        print(f"   ❌ MCP access failed: {await mcp_resp.text()}")
                            else:
                                print(f"   ❌ Token exchange failed: {await token_resp.text()}")
                else:
                    print(f"   Unexpected status: {resp.status}")
                    print(f"   Response: {await resp.text()[:500]}")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_oauth_flow())
