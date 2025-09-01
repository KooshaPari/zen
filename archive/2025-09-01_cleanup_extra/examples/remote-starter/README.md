# Remote Starter (Streamable HTTP + OAuth) Template

This template spins up a Streamable HTTP MCP server with:
- `/mcp` endpoint (Streamable HTTP for MCP JSON-RPC)
- OAuth 2.0 (Authorization Code + PKCE)
- Two auth modes:
  - `webauthn` (local operator approval + device biometrics, persisted to OS keychain)
  - `supabase` (accept external Supabase-issued JWTs; skip operator/TouchID shim)
- Basic tools are already exposed (listmodels, version, echo, etc.). Add more via the repo’s `tools/` package.

## Quick Start

1) Create a virtualenv and install deps (FastAPI, Uvicorn, PyJWT, cryptography, httpx):
```
pip install fastapi uvicorn pyjwt cryptography httpx
```

2) Choose auth mode via env:
- WebAuthn (default)
```
export AUTH_MODE=webauthn
export ENABLE_OAUTH2_AUTH=true
# Optional: require operator approval for first device
export PAIRING_REQUIRE_OPERATOR_APPROVAL=true
python -m examples.remote_starter.main
```
- Supabase (JWT validation)
```
export AUTH_MODE=supabase
export ENABLE_OAUTH2_AUTH=true
# Provide one of:
#   SUPABASE_JWKS_URL=https://<project>.supabase.co/auth/v1/.well-known/jwks.json
# or SUPABASE_JWT_SECRET=<service_role_or_anon_jwt_secret_for_HS256>
export SUPABASE_JWKS_URL=https://your.supabase.co/auth/v1/.well-known/jwks.json
python -m examples.remote_starter.main
```

3) Register your client (optional, for local OAuth flows) via DCR:
```
curl -X POST http://localhost:8080/oauth/register \
  -H 'Content-Type: application/json' \
  -d '{
        "client_name": "My MCP Client",
        "redirect_uris": ["http://localhost:3737/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_method": "none",
        "scope": "mcp mcp:read mcp:write profile"
      }'
```
The server mirrors DCR clients into the live OAuth server so `client_id` is immediately usable.

4) Discover OAuth metadata:
```
curl http://localhost:8080/.well-known/oauth-authorization-server
curl http://localhost:8080/.well-known/oauth-protected-resource
```

5) Call `/mcp`:
- After obtaining an access token (local OAuth or Supabase token), make requests:
```
curl -X POST http://localhost:8080/mcp \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' -i
```
Echo the returned `Mcp-Session-Id` header on subsequent requests.

## Supabase Mode

- The server accepts Supabase-issued JWTs (RS256 via JWKS, or HS256 via secret) as Bearer tokens.
- Scope mapping is derived from claims (example):
  - `role` or `roles` → `read`/`tools` (sufficient for /mcp)
  - add `mcp:write` for editor/admin roles if desired.
- Customize mapping in `examples/remote_starter/supabase_auth.py`.

## Adding Tools

This server already exposes essential tools via the repo’s `tools/` package (e.g., `listmodels`, `version`, `echo`, `get_time`, `multiply`). To add your own:
- Implement a new tool class in `tools/` following the pattern in `tools/*.py`.
- It will be discovered automatically by the HTTP server’s `get_all_tools()` registry.
- Optionally filter tools via `DISABLED_TOOLS` env var (comma-separated, preserves `version` and `listmodels`).

## Production Notes

- Put the server behind HTTPS. Localhost HTTP is allowed for dev.
- Disable/omit the operator/TouchID shim in Supabase mode; set `PAIRING_REQUIRE_OPERATOR_APPROVAL=false`.
- Avoid caching for `/.well-known/*` and `/oauth/*` at the edge.
- Ensure CORS is configured if a browser UI consumes metadata: `CORS_ALLOW_ORIGINS`.
- To silence Kafka warnings: set `DISABLE_KAFKA=true` or ensure Kafka is reachable.

For a full conceptual overview, see `docs/streamable_http_mcp.md`.

