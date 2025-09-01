#!/usr/bin/env python3
"""
OAuth Flow Fix - Patch for server_mcp_http.py

The issue: After successful WebAuthn verification, the OAuth flow doesn't complete
because there's no session cookie set, causing an infinite redirect loop.

Solution: Modify the flow to properly handle the authorization code generation
after consent approval.
"""


def apply_oauth_fix():
    """Apply fix to the OAuth flow in server_mcp_http.py"""

    # Read the current server file
    server_file = "server_mcp_http.py"
    with open(server_file) as f:
        content = f.read()

    # Fix 1: Modify the oauth_consent endpoint to pass through authorization params
    # After consent approval, we need to redirect with all the OAuth params preserved
    fix1 = """
                    # Store consent approval
                    logger.info(f"✅ POST /oauth/consent approved for {client_id} cid={self._cid(request)}")
                    
                    # Get the original OAuth params from the form (they should be hidden fields)
                    response_type = form.get("response_type", "code")
                    redirect_uri = form.get("redirect_uri", "")
                    state = form.get("state", "")
                    code_challenge = form.get("code_challenge", "")
                    code_challenge_method = form.get("code_challenge_method", "S256")
                    
                    # Generate authorization code directly here
                    from auth.oauth2_server import OAuth2Server
                    if self.oauth2_server:
                        # Create a simple session indicator
                        auth_code = self.oauth2_server._generate_authorization_code(
                            client_id=client_id,
                            user_id="device_user",  # From device auth
                            redirect_uri=redirect_uri,
                            scope=scope or "mcp:read",
                            code_challenge=code_challenge,
                            code_challenge_method=code_challenge_method
                        )
                        
                        # Redirect to callback with authorization code
                        from urllib.parse import urlencode
                        callback_params = {"code": auth_code.code}
                        if state:
                            callback_params["state"] = state
                        
                        callback_url = f"{redirect_uri}?{urlencode(callback_params)}"
                        return RedirectResponse(url=callback_url, status_code=303)
    """

    # Fix 2: Update the consent form to include all OAuth params as hidden fields
    fix2 = """
                            <input type="hidden" name="client_id" value="{client_id}">
                            <input type="hidden" name="scope" value="{scope}">
                            <input type="hidden" name="consent_id" value="{params.get('consent_id', '')}">
                            <input type="hidden" name="response_type" value="{params.get('response_type', 'code')}">
                            <input type="hidden" name="redirect_uri" value="{params.get('redirect_uri', '')}">
                            <input type="hidden" name="state" value="{params.get('state', '')}">
                            <input type="hidden" name="code_challenge" value="{params.get('code_challenge', '')}">
                            <input type="hidden" name="code_challenge_method" value="{params.get('code_challenge_method', 'S256')}">
    """

    print("OAuth Flow Fix")
    print("-" * 50)
    print("Issues identified:")
    print("1. After WebAuthn verification, no session cookie is set")
    print("2. The authorization endpoint checks for session and redirects infinitely")
    print("3. The consent form doesn't preserve OAuth parameters")
    print()
    print("Fixes needed:")
    print("1. Pass OAuth params through consent form as hidden fields")
    print("2. Generate auth code directly after consent approval")
    print("3. Redirect to callback URL with authorization code")
    print()
    print("To apply manually:")
    print("1. Edit server_mcp_http.py")
    print("2. In the oauth_consent POST handler (around line 1110)")
    print("3. Add code to generate authorization code and redirect")
    print("4. Update consent form to include hidden OAuth params")

if __name__ == "__main__":
    apply_oauth_fix()
