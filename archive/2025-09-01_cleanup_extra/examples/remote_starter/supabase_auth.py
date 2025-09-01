from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

import jwt

try:
    import httpx
except Exception:  # httpx optional for JWKS
    httpx = None  # type: ignore


class SupabaseTokenValidator:
    """Validate Supabase-issued JWTs (RS256 via JWKS or HS256 via secret) and map to MCP scopes.

    Env:
      - SUPABASE_JWKS_URL: e.g., https://<project>.supabase.co/auth/v1/.well-known/jwks.json
      - SUPABASE_JWT_SECRET: HS256 secret (service role or anon)
      - SUPABASE_ROLE_SCOPE_MAP (optional): JSON mapping of roles->scopes
        e.g., '{"authenticated": ["read","tools"], "admin": ["mcp:write","profile"]}'
    """

    def __init__(self):
        self.jwks_url = os.getenv("SUPABASE_JWKS_URL", "").strip()
        self.jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
        self.role_scope_map = self._load_role_scope_map()
        self._jwks_cache: dict[str, Any] | None = None
        self._jwks_fetched_at: float = 0.0

    def _load_role_scope_map(self) -> dict[str, list[str]]:
        raw = os.getenv("SUPABASE_ROLE_SCOPE_MAP", "").strip()
        if not raw:
            # Default: any authenticated user gets read/tools
            return {"authenticated": ["read", "tools"], "admin": ["read", "tools", "mcp:write", "profile"]}
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                # normalize values to list[str]
                out: dict[str, list[str]] = {}
                for k, v in data.items():
                    if isinstance(v, list):
                        out[k] = [str(x) for x in v]
                return out
        except Exception:
            pass
        return {"authenticated": ["read", "tools"]}

    async def _get_jwks(self) -> dict[str, Any] | None:
        if not self.jwks_url or not httpx:
            return None
        # Cache for 5 minutes
        if self._jwks_cache and (time.time() - self._jwks_fetched_at) < 300:
            return self._jwks_cache
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(self.jwks_url)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and data.get("keys"):
                    self._jwks_cache = data
                    self._jwks_fetched_at = time.time()
                    return data
        except Exception:
            return None
        return None

    def _get_kid(self, token: str) -> str | None:
        try:
            header_b64 = token.split(".")[0]
            # pad base64
            pad = '=' * (-len(header_b64) % 4)
            header = json.loads(base64.urlsafe_b64decode(header_b64 + pad).decode())
            return header.get("kid")
        except Exception:
            return None

    def _map_scopes(self, claims: dict[str, Any]) -> str:
        scopes: set[str] = set()
        # Common Supabase roles places
        roles: list[str] = []
        for key in ("role", "roles", "app_metadata", "user_metadata"):
            v = claims.get(key)
            if isinstance(v, str):
                roles.append(v)
            elif isinstance(v, list):
                roles.extend([str(x) for x in v])
            elif isinstance(v, dict):
                for rk in ("role", "roles"):
                    rv = v.get(rk)
                    if isinstance(rv, str):
                        roles.append(rv)
                    elif isinstance(rv, list):
                        roles.extend([str(x) for x in rv])

        # Default grant if no roles found
        if not roles:
            roles = ["authenticated"]

        for r in roles:
            for s in self.role_scope_map.get(r, []):
                scopes.add(s)

        # Ensure minimal read capability
        if not scopes:
            scopes.update({"read", "tools"})
        return " ".join(sorted(scopes))

    async def validate(self, token: str) -> dict[str, Any] | None:
        # RS256 via JWKS
        if self.jwks_url and httpx:
            try:
                jwks = await self._get_jwks()
                if jwks and jwks.get("keys"):
                    kid = self._get_kid(token)
                    key_to_use = None
                    for k in jwks["keys"]:
                        if not kid or k.get("kid") == kid:
                            key_to_use = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(k))
                            break
                    if key_to_use is None and jwks["keys"]:
                        key_to_use = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwks["keys"][0]))
                    if key_to_use:
                        claims = jwt.decode(token, key=key_to_use, algorithms=["RS256"], options={"verify_aud": False})
                        scope = self._map_scopes(claims)
                        return {"active": True, "client_id": "supabase", "sub": claims.get("sub"), "scope": scope}
            except Exception:
                pass

        # HS256 via secret
        if self.jwt_secret:
            try:
                claims = jwt.decode(token, key=self.jwt_secret, algorithms=["HS256"], options={"verify_aud": False})
                scope = self._map_scopes(claims)
                return {"active": True, "client_id": "supabase", "sub": claims.get("sub"), "scope": scope}
            except Exception:
                pass

        return None


async def attach_supabase_validator(server_obj) -> None:
    """Monkey-patch OAuth2Server.validate_bearer_token to fall back to Supabase validation.

    Only activated when AUTH_MODE=supabase.
    """
    if not getattr(server_obj, "oauth2_server", None):
        return

    validator = SupabaseTokenValidator()
    base_validate = server_obj.oauth2_server.validate_bearer_token

    async def _patched_validate(authorization_header: str | None):  # type: ignore[override]
        info = await base_validate(authorization_header)
        if info:
            # Normalize to expected fields
            if isinstance(info, dict):
                # Convert integration token_data to standard
                scope = info.get("scope") or (" ".join(info.get("scopes", [])) if isinstance(info.get("scopes"), list) else "")
                return {
                    "active": info.get("active", True),
                    "client_id": info.get("client_id"),
                    "sub": info.get("user_id") or info.get("sub"),
                    "scope": scope,
                }
            return info

        # Fallback: Supabase token validation
        if authorization_header and authorization_header.startswith("Bearer "):
            token = authorization_header[7:]
            supa = await validator.validate(token)
            if supa:
                return supa
        return None

    server_obj.oauth2_server.validate_bearer_token = _patched_validate  # type: ignore[attr-defined]

