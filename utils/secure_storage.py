"""
Secure storage helper for persisting small secrets/config.

Prefers OS keyring (via optional 'keyring' package). Falls back to a JSON file
in the user's config directory with chmod 600.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

_SERVICE = "zen-mcp-server"

def _get_config_path() -> Path:
    base = os.getenv("XDG_CONFIG_HOME") or os.path.join(Path.home(), ".config")
    d = Path(base) / "zen-mcp"
    d.mkdir(parents=True, exist_ok=True)
    return d / "secure_store.json"

def _load_file() -> dict:
    p = _get_config_path()
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_file(data: dict) -> None:
    p = _get_config_path()
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, p)
    try:
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass

def get_secret(key: str) -> str | None:
    try:
        import keyring  # type: ignore
        val = keyring.get_password(_SERVICE, key)
        if val:
            return val
    except Exception:
        pass
    # Fallback file
    data = _load_file()
    val = data.get(key)
    if isinstance(val, str):
        return val
    return None

def set_secret(key: str, value: str) -> None:
    try:
        import keyring  # type: ignore
        keyring.set_password(_SERVICE, key, value)
        return
    except Exception:
        pass
    data = _load_file()
    data[key] = value
    _save_file(data)

