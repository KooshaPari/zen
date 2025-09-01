"""
KInfra helpers (local shim) to automate named Cloudflare tunnel creation and DNS routing.

This module provides ensure_named_tunnel_autocreate() that mirrors the desired
KInfra behavior without modifying the external KInfra symlink.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path


async def _run_cmd_capture(*args: str) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        text=False,
    )
    out_b, err_b = await proc.communicate()
    stdout = out_b.decode("utf-8", errors="replace") if out_b else ""
    stderr = err_b.decode("utf-8", errors="replace") if err_b else ""
    return proc.returncode or 0, stdout, stderr


def _dns_safe_slug(value: str) -> str:
    import re

    if not value:
        return "local"
    slug = value.lower()
    slug = re.sub(r"[^a-z0-9-]", "-", slug)
    slug = re.sub(r"^-+|-+$", "", slug)
    slug = re.sub(r"--+", "-", slug)
    return slug or "local"


async def ensure_named_tunnel_autocreate(service_slug: str, domain: str, port: int) -> tuple[str, str, str]:
    """Ensure a named tunnel exists, DNS is routed, and service route is running.

    Returns (tunnel_id, hostname, config_path)
    """
    # cloudflared present
    if not shutil.which("cloudflared"):
        raise RuntimeError(
            "cloudflared not found. Install it and run 'cloudflared tunnel login' once."
        )

    # Logged in
    cert_path = Path.home() / ".cloudflared" / "cert.pem"
    if not cert_path.exists():
        raise RuntimeError("~/.cloudflared/cert.pem not found. Run: cloudflared tunnel login")

    tunnel_name = os.getenv("TUNNEL_NAME") or "zen-mcp-server"

    # List tunnels
    rc, stdout, stderr = await _run_cmd_capture("cloudflared", "tunnel", "list", "--output", "json")
    if rc != 0:
        raise RuntimeError(f"Failed to list tunnels: {stderr.strip() or stdout.strip()}")

    tunnel_id: str | None = None
    try:
        tunnels = json.loads(stdout)
        for t in tunnels:
            if str(t.get("name")) == tunnel_name:
                tunnel_id = str(t.get("id"))
                break
    except Exception:
        tunnels = []

    if not tunnel_id:
        rc, stdout, stderr = await _run_cmd_capture("cloudflared", "tunnel", "create", tunnel_name)
        if rc != 0:
            raise RuntimeError(f"Failed to create tunnel: {stderr.strip() or stdout.strip()}")
        # Re-list
        rc2, stdout2, stderr2 = await _run_cmd_capture("cloudflared", "tunnel", "list", "--output", "json")
        if rc2 == 0:
            try:
                tunnels = json.loads(stdout2)
                for t in tunnels:
                    if str(t.get("name")) == tunnel_name:
                        tunnel_id = str(t.get("id"))
                        break
            except Exception:
                pass
        if not tunnel_id:
            raise RuntimeError("Tunnel created but ID not found; verify with 'cloudflared tunnel list'.")

    # Build hostname and config
    service = _dns_safe_slug(service_slug)
    hostname = f"{service}.{domain.lower()}"
    cf_dir = Path.home() / ".cloudflared"
    creds_file = cf_dir / f"{tunnel_id}.json"
    config_path = cf_dir / f"config-{service}.yml"

    # Route DNS (idempotent)
    rc, stdout, stderr = await _run_cmd_capture(
        "cloudflared", "tunnel", "route", "dns", tunnel_id, hostname
    )
    err_text = (stdout + stderr).lower()
    if rc != 0 and "already exists" not in err_text:
        # Non fatal; still proceed
        pass

    # Write config YAML
    yaml_content = (
        f"tunnel: {tunnel_id}\n"
        f"credentials-file: {creds_file}\n"
        f"ingress:\n"
        f"  - hostname: {hostname}\n"
        f"    service: http://localhost:{port}\n"
        f"  - service: http_status:404\n"
    )
    config_path.write_text(yaml_content)

    # Start cloudflared for this service
    try:
        subprocess.Popen(
            ["cloudflared", "tunnel", "--config", str(config_path), "run"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        # Let the caller handle health checks; config exists regardless
        pass

    return tunnel_id, hostname, str(config_path)

