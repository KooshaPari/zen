# Zen MCP Streamable HTTP Interface
Status: Implemented — endpoints and flows available; see also `STREAMING_HTTP.md`.

This document explains how to expose any stdio MCP server through an HTTP(S) endpoint with OAuth 2.0, using the Streamable HTTP pattern implemented in this repository. It covers endpoints, auth flows, required headers, dynamic client registration, WebAuthn operator gating (optional), and how to swap in your own SaaS auth (e.g., Supabase) instead of the local operator/TouchID shim.

## Overview

- **MCP endpoint:** `POST /mcp` (and `GET /mcp` for a simple readiness response). Implements the Streamable HTTP transport for MCP JSON-RPC messages.
- **Auth model:** OAuth 2.0 Authorization Code + PKCE with protected resource metadata per RFC 9728. Two server-side OAuth modes exist in this repo:
  - Full OAuth server (JWT access tokens, RS256) — unified & recommended
  - Lightweight OAuth integration (opaque tokens + WebAuthn) — legacy/optional
  
  Note: The legacy OAuth integration endpoints are disabled by default to avoid duplicate routes. To enable them for local/experimental flows, set `ENABLE_OAUTH_INTEGRATION=true`.
- **Discovery:**
  - `/.well-known/oauth-protected-resource` (+ suffix variants such as `/mcp`)
  - `/.well-known/oauth-authorization-server` (+ suffix variants such as `/mcp`)
- **Dynamic client registration (DCR):** `POST /oauth/register` (+ GET/DELETE)
- **Operator/device auth (optional):** WebAuthn + operator approval for first-time devices. You can disable this for SaaS.

## Endpoints

- **`GET /mcp`**: Lightweight check. Returns JSON with `protocol_version`, `methods`, and optionally echoes `Mcp-Session-Id` when provided.
- **`HEAD /mcp`**: Returns `Allow` and `WWW-Authenticate` headers to advertise OAuth.
- **`OPTIONS /mcp`**: CORS preflight support.
- **`POST /mcp`**: Streamable HTTP endpoint for MCP JSON-RPC. Requires a valid Bearer token when auth is enabled.
  - On the first JSON-RPC `initialize` call, the server returns an `Mcp-Session-Id` header; clients must include this on subsequent requests.
  
### Streaming

- **`GET /stream/{task_id}`**: Server-Sent Events (SSE) stream for real-time updates from tools or orchestration.
  - Returns `Content-Type: text/event-stream` with standard SSE framing (`event:`, `data:`).
  - Emits status/progress/action/file/heartbeat/completion events.
 - **`GET /events/live`**: Global SSE with all task events (dashboard-friendly).

### Messaging (blocking/resume)

- **`POST /messages/channel`**: Post to a channel. Supports fields:
  - `channel_id`, `from`, `body`, optional `mentions[]`, `artifacts[]`, `blocking` (bool), `importance`.
- **`POST /messages/dm`**: Post a direct message between two members.
- **`GET /messages/channel/{channel_id}/history`**, **`GET /messages/dm/{a}/{b}/history`**: Fetch histories.
- **`GET /inbox/messages?agent_id=…`**: Unread counts plus an optional sample.
- NEW: **`POST /messages/resume`**: Mark a blocking message resolved and optionally mark it read.
  - Body: `{ "message_id": "…", "agent_id": "…" }`.
  - Response: `{ ok: true, message: { …, resolved: true, resolved_ts } }`.
 - NEW: **`POST /threads/resume`**: Resolve by `resume_token` and optionally post a reply.
   - Body: `{ "resume_token": "…", "from": "agent-id", "reply_body": "…" }`.
   - Response: `{ ok: true, message: { … }, reply_id?: "…" }`.

### OAuth 2.0

- **Authorization Server metadata:** `GET /.well-known/oauth-authorization-server` (and `/.well-known/oauth-authorization-server/mcp`). Returns issuer, authorization endpoint, token endpoint, introspection, revocation, registration, scopes, and PKCE method support.
- **Protected Resource metadata:** `GET /.well-known/oauth-protected-resource` (and `/.well-known/oauth-protected-resource/mcp`). Indicates authorization servers for this resource.
- **Authorize:** `GET /oauth/authorize` (Authorization Code + PKCE)
- **Token:** `POST /oauth/token` (Authorization Code + PKCE, and Refresh Token)
- **Introspection:** `POST /oauth/introspect`
- **Revocation:** `POST /oauth/revoke`
- **Dynamic Client Registration:**
  - `POST /oauth/register` — create a client
  - `GET /oauth/register/{client_id}` — fetch client metadata
  - `DELETE /oauth/register/{client_id}` — delete client (requires registration access token)

### Operator Approval + WebAuthn (optional)

- **Consent:** `POST /oauth/consent` — device-based consent, with optional operator approval gate on first registration.
- **WebAuthn verify:** `POST /oauth/verify` — completes registration or authentication, returns JSON `{ code, redirect_uri, redirect_url, state? }`.
- **Operator approve (loopback only):**
  - `GET /oauth/operator/approve/{request_id}` (HTML UI)
  - `POST /oauth/operator/approve` (JSON API)
  - `GET /oauth/operator/ready?request_id=...` (polling)

## Auth Flows

### A) Full OAuth Server (JWT) — recommended for SaaS

When the integrated OAuth2 server (`auth/oauth2_server.py`) is initialized, tokens are JWTs signed with RS256 and stored for quick validation. The server advertises metadata in well-known endpoints and supports DCR.

High-level flow:
1. Client discovers metadata at `/.well-known/oauth-authorization-server`.
2. Client performs Authorization Code + PKCE:
   - `GET /oauth/authorize?response_type=code&client_id=...&redirect_uri=...&code_challenge=...&code_challenge_method=S256&scope=...`
   - Successful auth redirects with `?code=...` (+ optional `state`).
3. Client exchanges code:
   - `POST /oauth/token` with `grant_type=authorization_code`, `code`, `redirect_uri`, `client_id`, `code_verifier`.
4. Client calls `/mcp` with `Authorization: Bearer <access_token>`.

Scope rules accepted by `/mcp` (any one suffices): `mcp`, `mcp:read`, `read`, `tools`.

Dynamic client registration example:
```bash
curl -X POST https://your-host/oauth/register \
  -H 'Content-Type: application/json' \
  -d '{
        "client_name": "My MCP Client",
        "redirect_uris": ["http://localhost:3737/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_method": "none",
        "scope": "mcp mcp:read mcp:write profile"
      }'
```

The HTTP server mirrors DCR clients into the live OAuth2 server automatically, so `client_id` is valid immediately for `/oauth/authorize` and `/oauth/token`.

### B) Lightweight OAuth Integration + WebAuthn (local/desktop)

The integration in `auth/oauth_integration.py` implements OAuth endpoints that generate opaque access tokens (stored in-memory) and uses device-based auth with WebAuthn. A first-time device can be operator-approved from the local console (loopback-only) and the device credential is persisted in the OS keychain (via `utils/secure_storage.py`, falling back to `~/.config/zen-mcp/secure_store.json`).

The consent and WebAuthn pages now:
- Auto-continue consent (no username prompts)
- On success, return JSON `{ code, redirect_url, state? }`
- Post a completion message to `window.opener` (`{ type: 'oauth_authorized', code, redirect_url }`), attempt to close the popup, and try window redirects as a fallback.

To disable operator approval (e.g., for dev): set `PAIRING_REQUIRE_OPERATOR_APPROVAL=false`.

## MCP over HTTP — Requests and Sessions

- Request payloads are MCP JSON-RPC messages (e.g., `initialize`, tool calls, etc.).
- On the first `initialize`, the server returns `Mcp-Session-Id` in the HTTP response headers; include that header (`Mcp-Session-Id: <value>`) on subsequent requests. If a request references a missing/expired session, a 400 is returned.
- `GET /mcp` returns a simple JSON document and is useful for health checks.
- `HEAD /mcp` returns `WWW-Authenticate` pointing to discovery if a Bearer token is missing.

Example initialize:
```http
POST /mcp HTTP/1.1
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": { "client": "my-app", "capabilities": {} }
}
```

Server response includes an `Mcp-Session-Id` header; echo it on subsequent `/mcp` requests.

## Discovery and Metadata

- `/.well-known/oauth-authorization-server` (and `/mcp` suffix) returns:
  - `issuer`
  - `authorization_endpoint`, `token_endpoint`, `revocation_endpoint`, `introspection_endpoint`, `registration_endpoint`
  - `response_types_supported`, `grant_types_supported`, `code_challenge_methods_supported`
  - `scopes_supported`
- `/.well-known/oauth-protected-resource` (and `/mcp` suffix) points to the Authorization Server(s) for this resource.
- The server sets `Cache-Control: no-store` where relevant; prefer to disable CDN caching for `/.well-known/*` and `/oauth/*` in front of your edge.

## Scopes

The MCP resource accepts any one of these for read access:
- `mcp`
- `mcp:read`
- `read`
- `tools`

You can tighten scope enforcement by customizing the scope check in `server_mcp_http.py`.

Scope normalization:
- Requests may use either standard scopes (`read`, `write`, `tools`, …) or MCP-prefixed forms (`mcp:read`, `mcp:write`).
- The bare `mcp` meta-scope expands to the default set (`read tools`).
- Access tokens presented to `/mcp` are accepted if they include any of: `mcp`, `mcp:read`, `read`, or `tools`.

## Configuration (env vars)

- `ENABLE_OAUTH2_AUTH` (default: `true`): enable OAuth requirement on `/mcp`.
- `ENABLE_OAUTH_INTEGRATION` (default: `false`): enable legacy integration endpoints in `auth/oauth_integration.py`. When disabled, only the unified `auth/oauth2_server.py` endpoints are active.
- Tunnel/health stabilization (advanced):
  - `ENABLE_TUNNEL_HEALTH_MONITOR` (default: `true`): background health monitor for the public tunnel.
  - `TUNNEL_HEALTH_CHECK_INTERVAL` (default: `30`): seconds between checks.
  - `TUNNEL_HEALTH_FAILURE_THRESHOLD` (default: `3`): consecutive failures before attempting self-heal.
  - `PUBLIC_HEALTH_PATH` (default: `/healthz`): path to probe.
  - `PUBLIC_HEALTH_REQUEST_TIMEOUT` (default: `5.0`): seconds per request.
  - Backoff tuning for initial readiness check:
    - `PUBLIC_HEALTH_MAX_ATTEMPTS` (default: `6`)
    - `PUBLIC_HEALTH_INITIAL_DELAY` (default: `1.0`)
    - `PUBLIC_HEALTH_BACKOFF_FACTOR` (default: `2.0`)
    - `PUBLIC_HEALTH_MAX_DELAY` (default: `10.0`)
- `CORS_ALLOW_ORIGINS`: comma-separated origins for browser clients (e.g., `http://localhost:6274,https://your-host`).
- Tunnel/issuer setup happens automatically; server updates issuer based on the public URL when available.
- WebAuthn/Operator:
  - `PAIRING_REQUIRE_OPERATOR_APPROVAL` (default: `true`) — require console approval for first-time device registration.
  - `OAUTH_APPROVAL_LOG_BOX` (default: `true`) — pretty console box for operator info.
  - `OAUTH_INVITE_SECRET` — enables `/oauth/invite` flow for email-style sign-in links.
- Kafka (optional):
  - `KAFKA_BOOTSTRAP_SERVERS` — if unset/unreachable, the publisher no-ops with warnings.

### A2A ACLs

- `A2A_ALLOWED_SENDERS`: Comma-separated allowlist of `sender_id` values permitted to send A2A messages to this server.
  - When set, messages from non-allowed senders receive an A2A `error` response `{ error: "forbidden", reason: "sender_not_allowed" }`.
 - `A2A_ALLOWED_TYPES`: Comma-separated list of allowed A2A message types (e.g., `chat_request,task_request`). Others are denied.

### Tooling Toggle

- `DISABLE_LEGACY_TOOLS=1` hides individual legacy tools (chat, analyze, planner, etc.) in favor of `deploy` and the new `messaging`, `project`, and `a2a` tools.

## Desktop/Popup UX

For desktop apps (e.g., Claude Desktop) that open a popup:
- After WebAuthn verify and code issuance, the page:
  - posts `{ type: 'oauth_authorized', code, state?, redirect_url }` to `window.opener`
  - attempts to redirect in-window and then closes the popup
Your desktop app can listen for that `postMessage` to finalize the flow if redirects are blocked.

## Replacing WebAuthn with Your SaaS Auth (e.g., Supabase)

If your project already uses a hosted identity provider (Supabase, Auth0, etc.), you can bypass the local operator/TouchID shim entirely and delegate token issuance/validation to your provider.

Patterns:
1. **Use your provider’s OAuth server end-to-end:**
   - Host discovery endpoints and OAuth pages at your provider.
   - Configure your app to accept Bearer tokens from that issuer.
   - In `server_mcp_http.py`, replace/extend the token validation logic to call your provider’s introspection or to verify JWTs via JWKS.
   - Map provider claims/roles to MCP scopes (e.g., `role=viewer` → `read`, `role=editor` → `mcp:write`).

2. **Bridge mode (keep our endpoints, validate externally):**
   - Keep `/oauth/authorize` and `/oauth/token` routed to your own frontend/backend, but in the MCP server’s `mcp_endpoint` auth block, validate `Authorization: Bearer <token>` by:
     - Verifying JWT via JWKS (for RS256) or provider SDK (e.g., Supabase GoTrue JWT secret).
     - On success, construct a `token_info` dict with `active: true`, `client_id`, `sub`, and `scope` string, then pass.
   - Disable WebAuthn/device flows by setting `PAIRING_REQUIRE_OPERATOR_APPROVAL=false` and not presenting those pages.

Minimal validation hook (pseudo):
```python
# server_mcp_http.py inside mcp_endpoint(), in the auth_enabled block
token = auth_header[7:]
token_info = await self.oauth2_server.validate_bearer_token(f"Bearer {token}")
if not token_info:
    # Try external provider (e.g., Supabase JWKS/JWT secret)
    token_info = await validate_with_supabase(token)  # implement using your provider
if not token_info or not token_info.get("active", True):
    raise HTTPException(status_code=401, detail="Invalid or expired access token")
```

Scope mapping example:
```python
roles = claims.get("roles", [])
scopes = set()
if "viewer" in roles: scopes.update(["read", "tools"])  # sufficient for MCP read
if "editor" in roles: scopes.update(["mcp:write"])      # write-level operations
token_info = {"active": True, "sub": claims["sub"], "client_id": "supabase", "scope": " ".join(sorted(scopes))}
```

## Troubleshooting

- `invalid_client` at callback:
  - Ensure the client is registered. This server mirrors `POST /oauth/register` clients into the OAuth2 server automatically; use `GET /oauth/register/{client_id}` to verify.
- `Insufficient scope` on `/mcp`:
  - Ensure the access token includes one of: `mcp`, `mcp:read`, `read`, `tools`.
- `Invalid or expired access token`:
  - Confirm `POST /oauth/token` succeeded; use `POST /oauth/introspect` with `token=<access_token>` to verify it’s active.
- `Authorization request not found` on WebAuthn page:
  - Caused by a stale/missing request_id; retry from `/oauth/authorize` in a fresh tab.
- Metadata errors in clients:
  - Verify `/.well-known/oauth-authorization-server` and its `/mcp` suffix return `authorization_endpoint` and `token_endpoint`.
  - Disable CDN caching for `/.well-known/*` and `/oauth/*` paths.

## Security Notes

- Use HTTPS in production; the server supports localhost HTTP for development.
- Persist WebAuthn device credentials in OS keychain (fallback to `~/.config/zen-mcp/secure_store.json` chmod 600).
- Operator approval endpoints are loopback-only (127.0.0.1/::1) and token-gated.
- For SaaS, prefer your provider’s JWT + JWKS verification and skip the operator shim.

## File Map (relevant parts)

- `server_mcp_http.py`: HTTP server, `/mcp` endpoint, OAuth metadata, DCR, and OAuth mirroring.
- `auth/oauth2_server.py`: Full OAuth 2.0 server (JWT), validation, code/token endpoints.
- `auth/oauth_integration.py`: Lightweight OAuth + WebAuthn integration, consent and WebAuthn pages.
- `auth/webauthn_flow.py`: Device credential management, registration, authentication.
- `utils/secure_storage.py`: Keychain + secure-file fallback storage for device credentials.

## Quick Start (dev)

1. Start the HTTP server; note the printed public URL (if tunneling) and issuer.
2. Register a client with `POST /oauth/register`.
3. From your app/UI, discover metadata at `/.well-known/oauth-authorization-server`.
4. Run the Authorization Code + PKCE flow; exchange code at `/oauth/token`.
5. Call `/mcp` with `Authorization: Bearer <access_token>`; start with `initialize` and keep using the `Mcp-Session-Id` header.
