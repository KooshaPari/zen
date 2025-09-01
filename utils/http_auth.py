from __future__ import annotations

from typing import Any


async def validate_bearer_token(authorization: str) -> dict[str, Any] | None:
    """Validate Bearer token for server_http endpoints.

    Tries Supabase validator from examples.remote_starter.supabase_auth if configured.
    Returns dict with at least: { active: bool, sub: str, scope: ' ' } or None.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    # Try Supabase validator if env is present
    try:
        from examples.remote_starter.supabase_auth import SupabaseTokenValidator

        v = SupabaseTokenValidator()
        info = await v.validate(token)
        if info and info.get("active"):
            return info
    except Exception:
        pass
    return None


def has_any_scope(scopes_str: str | None, allowed: set[str]) -> bool:
    if not scopes_str:
        return False
    try:
        s = set(scopes_str.split())
        return bool(s & allowed)
    except Exception:
        return False

