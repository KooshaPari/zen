"""
OAuth 2.0 Integration Layer with WebAuthn Authentication
Provides complete OAuth authorization server functionality
"""

import json
import os
import secrets
import smtplib
import ssl
import time
from email.message import EmailMessage
from urllib.parse import urlencode, urlparse

try:
    from fastapi import Request, Response
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from .oauth2_models import OAuth2Error, OAuth2Errors, StandardScopes, generate_oauth_token, parse_authorization_header
from .webauthn_flow import WebAuthnDeviceAuth


class OAuthAuthorizationServer:
    """OAuth 2.0 Authorization Server with WebAuthn authentication"""

    def __init__(self, webauthn_auth: WebAuthnDeviceAuth,
                 server_domain: str = "kooshapari.com"):
        self.webauthn = webauthn_auth
        self.server_domain = server_domain
        # Central store for pending authorization requests (thread-safe enough for single-process)
        self.authorization_requests = {}

        # Backward compatibility: some code paths referenced
        # self.webauthn.authorization_requests directly. To avoid AttributeError
        # and ensure a single source of truth, alias that attribute to our store.
        try:
            # Only set if missing to avoid overwriting any external wiring
            if not hasattr(self.webauthn, "authorization_requests"):
                self.webauthn.authorization_requests = self.authorization_requests
        except Exception:
            # Safe no-op if the provider forbids dynamic attributes
            pass
        # Require operator approval gate (reuse pairing env)
        self.require_operator_approval = os.getenv("PAIRING_REQUIRE_OPERATOR_APPROVAL", "true").lower() in ("1","true","on","yes")
        # Invite tracking (one-time use)
        self.consumed_invites = set()

    async def handle_authorization_request(self, request: Request) -> Response:
        """Handle OAuth 2.0 authorization endpoint (/oauth/authorize)"""

        try:
            # Extract query parameters
            client_id = request.query_params.get("client_id")
            redirect_uri = request.query_params.get("redirect_uri")
            response_type = request.query_params.get("response_type", "code")
            scope = request.query_params.get("scope", "read")
            state = request.query_params.get("state")
            code_challenge = request.query_params.get("code_challenge")
            code_challenge_method = request.query_params.get("code_challenge_method")

            # Validate required parameters
            if not client_id or not redirect_uri:
                raise OAuth2Error(
                    OAuth2Errors.INVALID_REQUEST,
                    "Missing required parameters: client_id, redirect_uri"
                )

            if response_type != "code":
                raise OAuth2Error(
                    OAuth2Errors.UNSUPPORTED_RESPONSE_TYPE,
                    "Only 'code' response type is supported"
                )

            # Parse scopes
            requested_scopes = set(scope.split())
            validated_scopes = StandardScopes.validate_scopes(requested_scopes)

            if not validated_scopes:
                raise OAuth2Error(
                    OAuth2Errors.INVALID_SCOPE,
                    "No valid scopes requested"
                )

            # Initiate OAuth authorization with WebAuthn
            auth_initiation = await self.webauthn.initiate_authentication(user_id="oauth_user")

            # Generate a unique request ID
            request_id = secrets.token_urlsafe(32)

            # Create OAuth authorization context
            auth_context = {
                "request_id": request_id,
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "scopes": validated_scopes,
                "state": state,
                "code_challenge": code_challenge,
                "code_challenge_method": code_challenge_method,
                "webauthn_challenge": auth_initiation,
                "client_name": client_id,
                "scope_descriptions": {
                    scope: f"Access to {scope} functionality" for scope in validated_scopes
                },
                "created_at": time.time()
            }

            # Store the authorization request for later retrieval during consent
            self.authorization_requests[request_id] = auth_context

            # Return authorization page with WebAuthn challenge
            return self._render_authorization_page(auth_context)

        except OAuth2Error as e:
            # Redirect error to client if we have a valid redirect_uri
            if redirect_uri and self._is_valid_redirect_uri(redirect_uri):
                error_params = e.to_dict()
                if state:
                    error_params["state"] = state
                error_url = f"{redirect_uri}?{urlencode(error_params)}"
                return RedirectResponse(url=error_url, status_code=302)
            else:
                # Return error page if redirect is not safe
                return self._render_error_page(e)

    async def handle_authorization_consent(self, request: Request) -> Response:
        """Handle user consent and WebAuthn authentication"""

        try:
            # Parse form data or query params; do not require username
            try:
                form_data = await request.form()
            except Exception:
                form_data = {}
            request_id = form_data.get("request_id") or request.query_params.get("request_id")
            action = form_data.get("action") or request.query_params.get("action") or "allow"
            # Use a device-based user identity; remove username requirement
            user_id = "device"

            if not request_id:
                raise OAuth2Error(OAuth2Errors.INVALID_REQUEST, "Missing request_id")

            # Get authorization request
            auth_request = self.authorization_requests.get(request_id)
            if not auth_request or (hasattr(auth_request, 'is_expired') and auth_request.is_expired()):
                raise OAuth2Error(OAuth2Errors.INVALID_REQUEST, "Invalid or expired authorization request")

            if action == "deny":
                # User denied authorization
                error = OAuth2Error(OAuth2Errors.ACCESS_DENIED, "User denied authorization")
                error_params = error.to_dict()
                if auth_request.get("state"):
                    error_params["state"] = auth_request["state"]
                error_url = f"{auth_request['redirect_uri']}?{urlencode(error_params)}"
                return RedirectResponse(url=error_url, status_code=302)

            # Check if user has existing WebAuthn credentials
            has_credentials = False
            try:
                # Check if user has registered devices
                user_credentials = [
                    cred for cred in self.webauthn.credentials.values()
                    if cred.user_id == user_id
                ]
                has_credentials = len(user_credentials) > 0
            except Exception:
                pass

            # Operator approval gate - ONLY for new registrations (no existing credentials)
            if self.require_operator_approval and not has_credentials:
                if not auth_request.get("operator_required"):
                    auth_request["operator_required"] = True
                    auth_request["operator_approved"] = False
                    auth_request["operator_token"] = secrets.token_urlsafe(16)
                    self.authorization_requests[request_id] = auth_request
                    try:
                        # Prefer app-provided local base (set by HTTP server)
                        try:
                            local_base = getattr(request.app.state, "local_base", "") if request else ""
                        except Exception:
                            local_base = ""
                        if not local_base:
                            # Derive from base_url or PORT env as fallback
                            base_url = str(request.base_url) if request else ""
                            server_port = None
                            if base_url and ":" in base_url:
                                port_part = base_url.split(":")[-1].rstrip("/")
                                server_port = port_part if port_part.isdigit() else None
                            if not server_port:
                                server_port = os.getenv("PORT", "8080")
                            local_base = f"http://localhost:{server_port}"
                        approval_url = f"{local_base}/oauth/operator/approve/{request_id}"
                        auto_approval_url = f"{local_base}/oauth/operator/approve/{request_id}?token={auth_request['operator_token']}"

                        print(
                            f"OAUTH_APPROVAL_REQUIRED (NEW USER) request_id={request_id} operator_token={auth_request['operator_token']} approval_url={approval_url}",
                            flush=True,
                        )
                        if os.getenv("OAUTH_APPROVAL_LOG_BOX", "true").lower() in ("1","true","on","yes"):
                            lines = [
                                f"Request ID: {request_id}",
                                f"Approval URL: {approval_url}",
                                f"Auto-Approval URL: {auto_approval_url}",
                                f"Operator Token: {auth_request['operator_token']}",
                            ]
                            width = max(len(s) for s in lines) + 2
                            top = "+" + ("-" * (width + 2)) + "+"
                            print(top)
                            title = "OAUTH OPERATOR APPROVAL"
                            print(f"| {title.ljust(width)} |")
                            print("+" + ("-" * (width + 2)) + "+")
                            for s in lines:
                                print(f"| {s.ljust(width)} |")
                            print(top)
                    except Exception:
                        pass
                if not auth_request.get("operator_approved"):
                    return self._render_waiting_for_operator_page(request_id, user_id)

                # Approved: begin WebAuthn registration for first device
                device_name = f"Device-{user_id.split('@')[0]}" if '@' in user_id else f"Device-{user_id}"
                webauthn_options = await self.webauthn.initiate_registration(user_id, device_name)
                # Persist the challenge so /oauth/verify can match the request
                try:
                    pk = webauthn_options.get("publicKey", {})
                    challenge_b64 = pk.get("challenge")
                    if challenge_b64:
                        auth_request["webauthn_challenge"] = challenge_b64
                        auth_request["user_id"] = user_id
                        self.authorization_requests[request_id] = auth_request
                except Exception:
                    pass
                # Mark registration flow for client script
                webauthn_options["is_registration"] = True
            else:
                # Known user: prompt for authentication
                webauthn_options = await self.webauthn.initiate_authentication(user_id)
                # Persist challenge for fallback matching if client omits it
                try:
                    pk = webauthn_options.get("publicKey", {})
                    challenge_b64 = pk.get("challenge")
                    if challenge_b64:
                        auth_request["webauthn_challenge"] = challenge_b64
                        auth_request["user_id"] = user_id
                        self.authorization_requests[request_id] = auth_request
                except Exception:
                    pass

            # Store the association between WebAuthn challenge and OAuth request
            if 'publicKey' in webauthn_options and 'challenge' in webauthn_options['publicKey']:
                challenge_id = webauthn_options['publicKey']['challenge']
                auth_request['webauthn_challenge'] = challenge_id
                auth_request['user_id'] = user_id  # Store user_id for later use
                self.authorization_requests[request_id] = auth_request

            # Add OAuth context to options for UI and verification
            webauthn_options.setdefault('oauth_context', {})
            webauthn_options['oauth_context'].update({
                'request_id': request_id,
                'client_id': auth_request.get('client_id'),
                'client_name': auth_request.get('client_name', auth_request.get('client_id', 'Unknown Client')),
                'redirect_uri': auth_request.get('redirect_uri'),
                'state': auth_request.get('state'),
                'webauthn_challenge': auth_request.get('webauthn_challenge')
            })

            return self._render_webauthn_page(webauthn_options)

        except OAuth2Error as e:
            return self._render_error_page(e)

    async def handle_webauthn_verification(self, request: Request) -> Response:
        """Handle WebAuthn verification and code generation"""

        try:
            # Parse WebAuthn response
            body = await request.json()

            # Check if this is a registration or authentication flow
            is_registration = body.get("is_registration", False)
            user_id = None

            if is_registration:
                # Complete WebAuthn registration
                success = await self.webauthn.complete_registration(body)
                if success:
                    # Extract user_id from the OAuth context
                    oauth_context = body.get("oauth_context", {})
                    request_id = oauth_context.get("request_id")
                    if request_id and request_id in self.authorization_requests:
                        user_id = self.authorization_requests[request_id].get("user_id")
            else:
                # Verify WebAuthn authentication using the existing method
                user_id = await self.webauthn.verify_authentication(body)

            if not user_id:
                raise OAuth2Error(OAuth2Errors.ACCESS_DENIED, "WebAuthn authentication failed")

            # Generate authorization code for the authenticated user
            # Generate short-lived authorization code
            authorization_code = "authz_" + generate_oauth_token(24)

            # Find the OAuth request associated with this flow
            oauth_context = body.get("oauth_context", {}) if isinstance(body, dict) else {}
            request_id = oauth_context.get("request_id")
            auth_request = self.authorization_requests.get(request_id) if request_id else None
            if not auth_request:
                # Fallback to challenge matching if request_id missing
                challenge = body.get("challenge")
                for req_id, req_data in self.authorization_requests.items():
                    if req_data.get("webauthn_challenge") == challenge:
                        auth_request = req_data
                        request_id = req_id
                        break

            if not auth_request:
                raise OAuth2Error(OAuth2Errors.SERVER_ERROR, "Authorization request not found")

            # Store the authorization code for later exchange
            # (In a real implementation, this would be stored securely with expiration)
            if not hasattr(self, 'authorization_codes'):
                self.authorization_codes = {}

            self.authorization_codes[authorization_code] = {
                "user_id": user_id,
                "client_id": auth_request.get("client_id"),
                "redirect_uri": auth_request.get("redirect_uri"),
                "scopes": auth_request.get("scopes"),
                "code_challenge": auth_request.get("code_challenge"),
                "code_challenge_method": auth_request.get("code_challenge_method"),
                "created_at": time.time()
            }

            # Clean up the authorization request
            del self.authorization_requests[request_id]

            # Optional webhook on OAuth success
            try:
                import asyncio
                import hashlib
                import hmac

                import httpx
                webhook_url = os.getenv("OAUTH_WEBHOOK_URL", "").strip()
                if webhook_url:
                    async def _notify():
                        body = json.dumps({
                            "event": "oauth_authorized",
                            "client_id": auth_request.get("client_id"),
                            "user_id": user_id,
                            "redirect_uri": auth_request.get("redirect_uri"),
                            "scopes": list(auth_request.get("scopes") or []),
                            "code": authorization_code,
                        }).encode()
                        headers = {"Content-Type": "application/json", "X-OAuth-Event": "authorized"}
                        secret = os.getenv("OAUTH_WEBHOOK_SECRET", "").encode()
                        if secret:
                            sig = hmac.new(secret, body, hashlib.sha256).hexdigest()
                            headers["X-OAuth-Signature"] = f"sha256={sig}"
                        method = os.getenv("OAUTH_WEBHOOK_METHOD", "POST").upper()
                        timeout = float(os.getenv("OAUTH_WEBHOOK_TIMEOUT", "5.0"))
                        async with httpx.AsyncClient(timeout=timeout) as client:
                            if method == "POST":
                                await client.post(webhook_url, content=body, headers=headers)
                            else:
                                await client.request(method, webhook_url, content=body, headers=headers)
                    asyncio.create_task(_notify())
            except Exception:
                pass

            # Instead of redirecting, return the authorization code in JSON
            # This allows the client-side JavaScript to handle the redirect
            # and avoid CORS issues when redirecting from https to http
            response_data = {
                "code": authorization_code,
                "redirect_uri": auth_request.get('redirect_uri')
            }

            # Include state if present
            if auth_request.get("state"):
                response_data["state"] = auth_request.get("state")

            # Build the full redirect URL for the client to use
            redirect_params = {"code": authorization_code}
            if auth_request.get("state"):
                redirect_params["state"] = auth_request.get("state")
            response_data["redirect_url"] = f"{auth_request.get('redirect_uri')}?{urlencode(redirect_params)}"

            return JSONResponse(content=response_data, status_code=200)

        except OAuth2Error as e:
            return self._render_error_page(e)

    async def handle_token_request(self, request: Request) -> Response:
        """Handle OAuth 2.0 token endpoint (/oauth/token)"""

        try:
            # Parse form data
            form_data = await request.form()
            grant_type = form_data.get("grant_type")

            if grant_type != "authorization_code":
                raise OAuth2Error(
                    OAuth2Errors.UNSUPPORTED_GRANT_TYPE,
                    "Only authorization_code grant type is supported"
                )

            # Extract token request parameters
            code = form_data.get("code")
            redirect_uri = form_data.get("redirect_uri")
            client_id = form_data.get("client_id")
            code_verifier = form_data.get("code_verifier")

            # Try client authentication via Authorization header if no client_id in form
            if not client_id:
                auth_header = request.headers.get("Authorization")
                if auth_header:
                    client_creds = parse_authorization_header(auth_header)
                    if client_creds:
                        client_id, client_secret = client_creds
                        # For now, skip client secret verification for public clients
                        # TODO: Implement proper client credential verification

            if not code or not redirect_uri or not client_id:
                raise OAuth2Error(
                    OAuth2Errors.INVALID_REQUEST,
                    "Missing required parameters: code, redirect_uri, client_id"
                )

            # Exchange code for tokens
            # Validate the authorization code
            if code not in self.authorization_codes:
                raise OAuth2Error(OAuth2Errors.INVALID_GRANT, "Invalid authorization code")

            code_data = self.authorization_codes[code]

            # Validate code hasn't expired (codes expire after 10 minutes)
            if time.time() - code_data.get("created_at", 0) > 600:
                del self.authorization_codes[code]
                raise OAuth2Error(OAuth2Errors.INVALID_GRANT, "Authorization code expired")

            # Validate client_id matches
            if code_data.get("client_id") != client_id:
                raise OAuth2Error(OAuth2Errors.INVALID_GRANT, "Code was issued to a different client")

            # Validate redirect_uri matches
            if code_data.get("redirect_uri") != redirect_uri:
                raise OAuth2Error(OAuth2Errors.INVALID_GRANT, "Redirect URI mismatch")

            # Validate PKCE if code_challenge was present
            if code_data.get("code_challenge"):
                if not code_verifier:
                    raise OAuth2Error(OAuth2Errors.INVALID_REQUEST, "Code verifier required")

                # Verify code_verifier against code_challenge
                import base64
                import hashlib
                verifier_hash = base64.urlsafe_b64encode(
                    hashlib.sha256(code_verifier.encode()).digest()
                ).decode().rstrip("=")

                if verifier_hash != code_data.get("code_challenge"):
                    raise OAuth2Error(OAuth2Errors.INVALID_GRANT, "Code verifier mismatch")

            # Generate access token
            access_token = "access_" + generate_oauth_token(32)
            refresh_token = "refresh_" + generate_oauth_token(32)

            # Store token info (in production, use secure storage)
            if not hasattr(self, 'access_tokens'):
                self.access_tokens = {}

            token_data = {
                "client_id": client_id,
                "user_id": code_data.get("user_id"),
                "scopes": code_data.get("scopes", []),
                "created_at": time.time(),
                "expires_in": 3600,  # 1 hour
                "active": True,  # Add active flag for validation
                "sub": code_data.get("user_id"),  # Add sub claim for compatibility
                "scope": " ".join(code_data.get("scopes", []))  # Add scope string
            }

            self.access_tokens[access_token] = token_data

            # Also store in main OAuth server if available for shared validation
            if hasattr(self, 'main_oauth_server') and self.main_oauth_server:
                if not hasattr(self.main_oauth_server, 'access_tokens'):
                    self.main_oauth_server.access_tokens = {}
                self.main_oauth_server.access_tokens[access_token] = token_data

            # Delete used authorization code
            del self.authorization_codes[code]

            # Build token response
            token_response = {
                "access_token": access_token,
                "token_type": "Bearer",
                "expires_in": 3600,
                "refresh_token": refresh_token,
                "scope": " ".join(code_data.get("scopes", []))
            }

            # Return JSON response with tokens
            return Response(
                content=json.dumps(token_response),
                media_type="application/json",
                headers={
                    "Cache-Control": "no-store",
                    "Pragma": "no-cache"
                }
            )

        except OAuth2Error as e:
            # Token endpoint errors are returned as JSON
            return Response(
                content=json.dumps(e.to_dict()),
                media_type="application/json",
                status_code=400
            )

    async def handle_token_introspection(self, request: Request) -> Response:
        """Handle token introspection endpoint (/oauth/introspect)"""

        try:
            form_data = await request.form()
            token = form_data.get("token")

            if not token:
                raise OAuth2Error(OAuth2Errors.INVALID_REQUEST, "Missing token parameter")

            # Validate access token
            access_token = await self.webauthn.validate_access_token(token)

            if access_token and access_token.is_valid():
                introspection_response = {
                    "active": True,
                    "client_id": access_token.client_id,
                    "username": access_token.user_id,
                    "scope": " ".join(access_token.scopes),
                    "exp": int(access_token.expires_at),
                    "iat": int(access_token.created_at),
                    "token_type": access_token.token_type
                }
            else:
                introspection_response = {"active": False}

            return Response(
                content=json.dumps(introspection_response),
                media_type="application/json"
            )

        except Exception:
            # Return inactive for any error
            return Response(
                content=json.dumps({"active": False}),
                media_type="application/json"
            )

    def _is_valid_redirect_uri(self, redirect_uri: str) -> bool:
        """Validate redirect URI for security"""
        try:
            parsed = urlparse(redirect_uri)
            # Allow HTTPS and localhost HTTP for development
            if parsed.scheme == "https":
                return True
            if parsed.scheme == "http" and parsed.hostname in ["localhost", "127.0.0.1"]:
                return True
            return False
        except Exception:
            return False

    def _render_authorization_page(self, auth_initiation: dict) -> HTMLResponse:
        """Render OAuth authorization page with WebAuthn setup"""

        scopes_html = ""
        for scope, description in auth_initiation.get("scope_descriptions", {}).items():
            scopes_html += f"<li><strong>{scope}</strong>: {description}</li>"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>OAuth Authorization - {auth_initiation.get('client_name', 'Unknown Client')}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }}
        .card {{ background: #f8f9fa; border-radius: 12px; padding: 30px; border: 1px solid #e9ecef; }}
        .client-info {{ background: #007AFF; color: white; border-radius: 8px; padding: 15px; margin-bottom: 20px; }}
        .scopes {{ background: white; border-radius: 8px; padding: 15px; margin: 15px 0; border: 1px solid #e9ecef; }}
        .scopes ul {{ margin: 10px 0; padding-left: 20px; }}
        .buttons {{ text-align: center; margin-top: 25px; }}
        button {{ background: #007AFF; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 16px; cursor: pointer; margin: 0 10px; }}
        button.deny {{ background: #FF3B30; }}
        button:hover {{ opacity: 0.8; }}
        .user-input {{ margin: 20px 0; }}
        input {{ width: 100%; padding: 12px; margin: 8px 0; border: 1px solid #ccc; border-radius: 8px; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="client-info">
            <h2>🔐 OAuth Authorization Request</h2>
            <p><strong>{auth_initiation.get('client_name', 'Unknown Client')}</strong> is requesting access to your account.</p>
        </div>

        <div class="scopes">
            <h3>Requested Permissions:</h3>
            <ul>
                {scopes_html}
            </ul>
        </div>

        <form id="consentForm" action="/oauth/consent" method="post">
            <input type="hidden" name="request_id" value="{auth_initiation.get('request_id')}" />
            <input type="hidden" name="action" value="allow" />
            <div class="buttons">
                <button type="submit">🔑 Continue</button>
                <button type="submit" name="action" value="deny" class="deny">❌ Deny</button>
            </div>
        </form>
        <script>
          // Auto-submit to proceed immediately without requiring a username
          window.addEventListener('DOMContentLoaded', () => {{
            const f = document.getElementById('consentForm');
            if (f) f.submit();
          }});
        </script>
    </div>
</body>
</html>
        """

        return HTMLResponse(content=html_content)

    def _render_webauthn_page(self, webauthn_options: dict) -> HTMLResponse:
        """Render WebAuthn authentication page"""
        if not webauthn_options:
            return self._render_error_page(OAuth2Error(OAuth2Errors.ACCESS_DENIED, "No WebAuthn options available"))

        oauth_context = webauthn_options.get("oauth_context", {})
        client_name = oauth_context.get("client_name", "Unknown Client")
        is_registration = webauthn_options.get("is_registration", False)

        html_content = rf"""
<!DOCTYPE html>
<html>
<head>
    <title>WebAuthn Authentication - {client_name}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }}
        .card {{ background: #f8f9fa; border-radius: 12px; padding: 30px; border: 1px solid #e9ecef; text-align: center; }}
        .biometric-icon {{ font-size: 64px; margin: 20px 0; }}
        button {{ background: #007AFF; color: white; border: none; padding: 15px 30px; border-radius: 8px; font-size: 16px; cursor: pointer; }}
        button:hover {{ opacity: 0.8; }}
        .status {{ margin: 20px 0; padding: 15px; border-radius: 8px; }}
        .error {{ background: #FFE6E6; color: #D8000C; }}
        .success {{ background: #E6F7E6; color: #2E7D2E; }}
        .info {{ background: #E6F3FF; color: #0066CC; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="biometric-icon">🔐</div>
        <h2>{"Device Registration" if is_registration else "Biometric Authentication"} Required</h2>
        <p>{"Register your device to authorize" if is_registration else "Complete your authorization to"} <strong>{client_name}</strong> using WebAuthn.</p>

        <button onclick="authenticateWebAuthn()">🔑 {"Register Device" if is_registration else "Authenticate"} with Touch ID / Face ID / Windows Hello</button>

        <div id="status"></div>
    </div>

    <script>
        const webauthnOptions = {json.dumps(webauthn_options)};
        const oauthContext = webauthnOptions.oauth_context || {{}};
        const isRegistration = {str(is_registration).lower()};

        function setStatus(message, type = 'info') {{
            const el = document.getElementById('status');
            el.innerHTML = message;
            el.className = `status ${{type}}`;
        }}

        async function authenticateWebAuthn() {{
            try {{
                setStatus(isRegistration ? 'Registering your device...' : 'Touch your biometric sensor or security key...', 'info');

                // Prepare WebAuthn options
                const options = webauthnOptions.publicKey;
                options.challenge = base64urlDecode(options.challenge);

                let credential;
                let verifyResp;

                if (isRegistration) {{
                    // Registration flow
                    options.user.id = base64urlDecode(options.user.id);

                    // Create new credential
                    credential = await navigator.credentials.create({{publicKey: options}});

                    // Submit registration (include original challenge string)
                    verifyResp = await fetch('/oauth/verify', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            id: credential.id,
                            rawId: base64urlEncode(credential.rawId),
                            type: credential.type,
                            response: {{
                                clientDataJSON: base64urlEncode(credential.response.clientDataJSON),
                                attestationObject: base64urlEncode(credential.response.attestationObject)
                            }},
                            challenge: webauthnOptions.publicKey.challenge,
                            is_registration: true,
                            oauth_context: oauthContext
                        }})
                    }});
                }} else {{
                    // Authentication flow
                    options.allowCredentials.forEach(cred => {{
                        cred.id = base64urlDecode(cred.id);
                    }});

                    // Get WebAuthn assertion
                    credential = await navigator.credentials.get({{publicKey: options}});

                    // Submit verification
                    verifyResp = await fetch('/oauth/verify', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            id: credential.id,
                            challenge: webauthnOptions.publicKey.challenge,
                            response: {{
                                clientDataJSON: base64urlEncode(credential.response.clientDataJSON),
                                authenticatorData: base64urlEncode(credential.response.authenticatorData),
                                signature: base64urlEncode(credential.response.signature)
                            }},
                            is_registration: false,
                            oauth_context: oauthContext
                        }})
                    }});
                }}

                if (!verifyResp.ok) {{
                    const errText = await verifyResp.text();
                    setStatus('Authorization failed: ' + errText, 'error');
                }} else {{
                    // Parse JSON response and perform client-side redirect
                    const data = await verifyResp.json();
                    const postToOpener = () => {{
                        try {{
                            if (window.opener && typeof window.opener.postMessage === 'function') {{
                                window.opener.postMessage({{
                                    type: 'oauth_authorized',
                                    code: data.code,
                                    state: data.state || null,
                                    redirect_url: data.redirect_url,
                                }}, '*');
                            }}
                        }} catch (e) {{ /* noop */ }}
                    }};

                    if (data.redirect_url) {{
                        setStatus('Authorization successful. Redirecting...', 'success');
                        postToOpener();
                        // Try in-window redirect first
                        try {{ window.location.href = data.redirect_url; }} catch (e) {{ /* ignore */ }}
                        // Fallback: attempt to close popup windows
                        try {{ window.close(); }} catch (e) {{ /* ignore */ }}
                        // Last fallback: delayed redirect if still visible
                        setTimeout(() => {{
                            try {{
                                if (!document.hidden) {{
                                    window.location.assign(data.redirect_url);
                                }}
                            }} catch (e) {{ /* ignore */ }}
                        }}, 800);
                    }} else {{
                        // No redirect URL (unlikely) — notify host app and suggest manual close
                        postToOpener();
                        setStatus('Authorization completed. You may close this window.', 'success');
                        try {{ window.close(); }} catch (e) {{ /* ignore */ }}
                    }}
                }}

            }} catch (error) {{
                console.error('WebAuthn error:', error);
                setStatus(`Authentication failed: ${{error.message}}`, 'error');
            }}
        }}

        // Base64URL encoding/decoding helpers
        function base64urlDecode(str) {{
            return Uint8Array.from(atob(str.replace(/-/g, '+').replace(/_/g, '/').padEnd(str.length + (4 - str.length % 4) % 4, '=')), c => c.charCodeAt(0));
        }}

        function base64urlEncode(buffer) {{
            // Convert to Base64 then make it URL-safe by replacing + and /
            return btoa(String.fromCharCode(...new Uint8Array(buffer)))
                .replace(/\+/g, '-')
                .replace(/\//g, '_')
                .replace(/=/g, '');
        }}

        // Auto-start authentication
        setTimeout(() => {{
            setStatus('Ready for biometric authentication', 'info');
        }}, 500);
    </script>
</body>
</html>
        """

        return HTMLResponse(content=html_content)

    # Removed demo registration page; new users are auto-gated for operator approval
    # and then sent directly to WebAuthn registration.

    def _render_waiting_for_operator_page(self, request_id: str, user_id: str) -> HTMLResponse:
        html = f"""
<!DOCTYPE html>
<html>
<head>
  <title>Awaiting Operator Approval</title>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <style>
    body {{ font-family: -apple-system, sans-serif; max-width: 560px; margin: 50px auto; padding: 20px; }}
    .card {{ background: #fff; border: 1px solid #e1e4e8; border-radius: 12px; padding: 24px; text-align: center; }}
    .spinner {{ margin: 24px auto; width: 40px; height: 40px; border: 4px solid #eee; border-top-color: #007AFF; border-radius: 50%; animation: spin 1s linear infinite; }}
    @keyframes spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
    .muted {{ color: #666; font-size: 14px; }}
  </style>
</head>
<body>
  <div class=\"card\">
    <h2>Awaiting Operator Approval</h2>
    <div class=\"spinner\"></div>
    <p>Your sign-in request is pending operator approval on the server console.</p>
    <p class=\"muted\">This page will continue automatically once approved.</p>
  </div>
  <form id=\"resume\" method=\"post\" action=\"/oauth/consent\" style=\"display:none;\">
    <input type=\"hidden\" name=\"request_id\" value=\"{request_id}\" />
    <input type=\"hidden\" name=\"user_id\" value=\"{user_id}\" />
    <input type=\"hidden\" name=\"action\" value=\"allow\" />
  </form>
  <script>
    async function poll() {{
      try {{
        const r = await fetch('/oauth/operator/ready?request_id={request_id}');
        if(r.ok) {{ const j = await r.json(); if(j && j.approved===true) {{ document.getElementById('resume').submit(); return; }} }}
      }} catch(e) {{ }}
      setTimeout(poll, 1500);
    }}
    poll();
  </script>
</body>
</html>
        """
        return HTMLResponse(content=html)

    def _render_error_page(self, error: OAuth2Error) -> HTMLResponse:
        """Render OAuth error page"""

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>OAuth Authorization Error</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }}
        .error-card {{ background: #FFE6E6; border: 1px solid #FF9999; border-radius: 12px; padding: 30px; text-align: center; }}
        .error-icon {{ font-size: 64px; margin: 20px 0; color: #D8000C; }}
        .error-title {{ color: #D8000C; margin: 15px 0; }}
        .error-details {{ background: white; border-radius: 8px; padding: 15px; margin: 20px 0; text-align: left; }}
    </style>
</head>
<body>
    <div class="error-card">
        <div class="error-icon">❌</div>
        <h2 class="error-title">Authorization Failed</h2>

        <div class="error-details">
            <p><strong>Error:</strong> {error.error}</p>
            {f'<p><strong>Description:</strong> {error.error_description}</p>' if error.error_description else ''}
        </div>

        <p>Please contact the application developer for assistance.</p>
    </div>
</body>
</html>
        """

        return HTMLResponse(content=html_content, status_code=400)


def setup_oauth_endpoints(app, webauthn_auth: WebAuthnDeviceAuth, server_domain: str = "kooshapari.com", main_oauth_server=None):
    """Setup OAuth 2.0 endpoints for FastAPI"""

    if not FASTAPI_AVAILABLE:
        print("FastAPI not available, OAuth endpoints not configured")
        return

    oauth_server = OAuthAuthorizationServer(webauthn_auth, server_domain)

    # If a main OAuth server is provided, share its token storage
    if main_oauth_server:
        oauth_server.main_oauth_server = main_oauth_server

    @app.get("/oauth/authorize")
    async def oauth_authorize(request: Request):
        """OAuth 2.0 authorization endpoint"""
        return await oauth_server.handle_authorization_request(request)

    @app.post("/oauth/consent")
    async def oauth_consent(request: Request):
        """OAuth 2.0 consent handling"""
        return await oauth_server.handle_authorization_consent(request)

    @app.post("/oauth/verify")
    async def oauth_webauthn_verify(request: Request):
        """OAuth WebAuthn verification"""
        return await oauth_server.handle_webauthn_verification(request)

    @app.post("/oauth/token")
    async def oauth_token(request: Request):
        """OAuth 2.0 token endpoint"""
        return await oauth_server.handle_token_request(request)

    @app.post("/oauth/introspect")
    async def oauth_introspect(request: Request):
        """OAuth 2.0 token introspection endpoint"""
        return await oauth_server.handle_token_introspection(request)

    # Operator approval endpoints (local only)
    @app.post("/oauth/operator/approve")
    async def oauth_operator_approve(request: Request):
        ip = request.client.host if request.client else ""
        if ip not in ("127.0.0.1", "::1", "localhost"):
            return HTMLResponse(content="Forbidden", status_code=403)
        try:
            body = await request.json()
            request_id = (body.get("request_id") or "").strip()
            operator_token = (body.get("operator_token") or "").strip()
            if not request_id or not operator_token:
                return HTMLResponse(content="Bad Request", status_code=400)
            auth_req = oauth_server.authorization_requests.get(request_id)
            if not auth_req or not auth_req.get("operator_required"):
                return HTMLResponse(content="Invalid request", status_code=400)
            if auth_req.get("operator_token") != operator_token:
                return HTMLResponse(content="Invalid token", status_code=401)
            auth_req["operator_approved"] = True
            auth_req["operator_token"] = None
            oauth_server.authorization_requests[request_id] = auth_req
            print(f"OAUTH_APPROVED request_id={request_id}", flush=True)
            return HTMLResponse(content="Approved")
        except Exception as e:
            return HTMLResponse(content=f"Error: {e}", status_code=500)

    @app.get("/oauth/operator/ready")
    async def oauth_operator_ready(request: Request):
        request_id = (request.query_params.get("request_id") or "").strip()
        auth_req = oauth_server.authorization_requests.get(request_id)
        if not auth_req:
            return HTMLResponse(content=json.dumps({"approved": False}), media_type="application/json", status_code=404)
        approved = bool(auth_req.get("operator_approved"))
        code = 200 if approved else 202
        return HTMLResponse(content=json.dumps({"approved": approved}), media_type="application/json", status_code=code)

    @app.get("/oauth/operator/approve/{request_id}")
    async def oauth_operator_approve_page(request_id: str, request: Request):
        # Loopback only
        ip = request.client.host if request.client else ""
        if ip not in ("127.0.0.1", "::1", "localhost"):
            return HTMLResponse(content="Forbidden", status_code=403)
        auth_req = oauth_server.authorization_requests.get(request_id)
        if not auth_req or not auth_req.get("operator_required"):
            return HTMLResponse(content=f"Invalid Request<br/><br/>Request ID not found or doesn't require approval: {request_id}", status_code=404)
        client_id = auth_req.get("client_id", "Unknown")
        html = f"""
<!DOCTYPE html><html><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>OAuth Operator Approval</title>
<style>body{{font-family:-apple-system,sans-serif;max-width:520px;margin:60px auto;padding:20px}}.card{{background:#fff;border:1px solid #e1e4e8;border-radius:12px;padding:24px}}input{{width:100%;padding:12px;margin:10px 0;border:1px solid #ccc;border-radius:8px}}button{{width:100%;padding:12px;background:#2ea44f;color:#fff;border:none;border-radius:8px;font-size:16px;cursor:pointer}}.muted{{color:#666}}</style>
</head><body>
<div class='card'>
  <h2>OAuth Operator Approval</h2>
  <p class='muted'>Request ID: <code>{request_id}</code><br/>Client ID: <code>{client_id}</code></p>
  <input id='token' placeholder='Enter operator token' autofocus />
  <button onclick="(async()=>{{const t=document.getElementById('token').value.trim(); if(!t) return alert('Enter token'); const r=await fetch('/oauth/operator/approve',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{request_id:'{request_id}',operator_token:t}})}}); if(r.ok){{alert('Approved ✔'); window.close();}} else {{alert('Approval failed: '+await r.text());}} }})()">Approve Request</button>
</div>
</body></html>
        """
        return HTMLResponse(content=html)


    # Email-style JWT invites (local issuance)
    @app.post("/oauth/invite")
    async def oauth_invite(request: Request):
        ip = request.client.host if request.client else ""
        if ip not in ("127.0.0.1", "::1", "localhost"):
            return HTMLResponse(content="Forbidden", status_code=403)
        try:
            import time

            import jwt
            body = await request.json()
            client_id = (body.get("client_id") or "").strip()
            redirect_uri = (body.get("redirect_uri") or "").strip()
            scope = (body.get("scope") or "mcp").strip()
            ttl = int(body.get("ttl", 900))
            if not client_id or not redirect_uri:
                return HTMLResponse(content="Missing client_id or redirect_uri", status_code=400)
            secret = os.getenv("OAUTH_INVITE_SECRET", "").strip()
            if not secret:
                return HTMLResponse(content="Server not configured for invites (OAUTH_INVITE_SECRET missing)", status_code=500)
            now = int(time.time())
            jti = secrets.token_urlsafe(12)
            payload = {
                "iss": server_domain,
                "aud": "oauth-invite",
                "iat": now,
                "exp": now + ttl,
                "jti": jti,
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "scope": scope,
            }
            token = jwt.encode(payload, secret, algorithm="HS256")
            base = str(request.base_url).rstrip('/')
            link = f"{base}/oauth/invite/register?token={token}"
            return HTMLResponse(content=json.dumps({"token": token, "link": link}), media_type="application/json")
        except Exception as e:
            return HTMLResponse(content=f"Error: {e}", status_code=500)

    @app.post("/oauth/invite/email")
    async def oauth_invite_email(request: Request):
        """Issue an OAuth invite and send it via SMTP (loopback-only)."""
        ip = request.client.host if request.client else ""
        if ip not in ("127.0.0.1", "::1", "localhost"):
            return HTMLResponse(content="Forbidden", status_code=403)
        try:
            import time

            import jwt
            body = await request.json()
            to_email = (body.get("to") or "").strip()
            client_id = (body.get("client_id") or "").strip()
            redirect_uri = (body.get("redirect_uri") or "").strip()
            scope = (body.get("scope") or "mcp profile").strip()
            ttl = int(body.get("ttl", 900))
            if not to_email or not client_id or not redirect_uri:
                return HTMLResponse(content="Missing to, client_id or redirect_uri", status_code=400)
            secret = os.getenv("OAUTH_INVITE_SECRET", "").strip()
            if not secret:
                return HTMLResponse(content="Server not configured for invites (OAUTH_INVITE_SECRET missing)", status_code=500)
            now = int(time.time())
            jti = secrets.token_urlsafe(12)
            payload = {
                "iss": server_domain,
                "aud": "oauth-invite",
                "iat": now,
                "exp": now + ttl,
                "jti": jti,
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "scope": scope,
            }
            token = jwt.encode(payload, secret, algorithm="HS256")
            base = str(request.base_url).rstrip('/')
            link = f"{base}/oauth/invite/register?token={token}"

            # Build email
            mins = max(1, ttl // 60)
            subj = body.get("subject") or "Your Zen MCP Sign-in Invite"
            html = f"""
<!DOCTYPE html><html><body style='font-family:-apple-system,Segoe UI,Arial'>
  <h2>Zen MCP Sign-in</h2>
  <p>Use the secure invite link below to begin sign-in:</p>
  <p><a href='{link}'>{link}</a></p>
  <p style='color:#555'>This link expires in {mins} minutes.</p>
</body></html>
            """
            text = f"Zen MCP Sign-in\n\nLink: {link}\nExpires in {mins} minutes.\n"
            _send_invite_email(to_email, subj, html, text)
            return HTMLResponse(content=json.dumps({"sent": True, "link": link}), media_type="application/json")
        except Exception as e:
            return HTMLResponse(content=f"Error: {e}", status_code=500)

    @app.get("/oauth/invite/register")
    async def oauth_invite_register(request: Request):
        import jwt
        token = request.query_params.get("token")
        if not token:
            return HTMLResponse(content="Invalid invite", status_code=400)
        secret = os.getenv("OAUTH_INVITE_SECRET", "").strip()
        if not secret:
            return HTMLResponse(content="Server not configured for invites", status_code=500)
        try:
            claims = jwt.decode(token, secret, algorithms=["HS256"], audience="oauth-invite")
            jti = claims.get("jti")
            if not jti or jti in oauth_server.consumed_invites:
                return HTMLResponse(content="Invite already used or invalid", status_code=400)
            oauth_server.consumed_invites.add(jti)
            client_id = claims.get("client_id")
            redirect_uri = claims.get("redirect_uri")
            scope = claims.get("scope") or "mcp"
            # Redirect to standard OAuth authorize
            from urllib.parse import urlencode
            qs = urlencode({
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "scope": scope,
            })
            return RedirectResponse(url=f"/oauth/authorize?{qs}")
        except Exception as e:
            return HTMLResponse(content=f"Invalid invite: {e}", status_code=400)

    print(f"✅ OAuth 2.0 endpoints configured for domain: {server_domain}")
    return oauth_server

def _send_invite_email(to_email: str, subject: str, html_body: str, text_body: str | None = None) -> None:
    """Send an email using SMTP based on env configuration.

    Environment variables:
      SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD,
      SMTP_USE_TLS (true/false), SMTP_USE_SSL (true/false),
      SMTP_FROM (default: zen@kooshapari.com), SMTP_TIMEOUT
    """
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USERNAME")
    pwd = os.getenv("SMTP_PASSWORD")
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in ("1", "true", "on", "yes")
    use_ssl = os.getenv("SMTP_USE_SSL", "false").lower() in ("1", "true", "on", "yes")
    from_email = os.getenv("SMTP_FROM", "zen@kooshapari.com")
    timeout = float(os.getenv("SMTP_TIMEOUT", "10"))

    if not host:
        raise RuntimeError("SMTP_HOST not configured")

    msg = EmailMessage()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(text_body or "This email contains an HTML part.")
    msg.add_alternative(html_body, subtype="html")

    if use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context, timeout=timeout) as server:
            if user and pwd:
                server.login(user, pwd)
            server.send_message(msg)
            return
    else:
        with smtplib.SMTP(host, port, timeout=timeout) as server:
            if use_tls:
                server.starttls(context=ssl.create_default_context())
            if user and pwd:
                server.login(user, pwd)
            server.send_message(msg)
            return
