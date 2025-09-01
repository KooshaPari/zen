from __future__ import annotations

import asyncio
import logging
import os

import uvicorn
from examples.remote_starter.supabase_auth import attach_supabase_validator

from server_mcp_http import ZenMCPStreamableServer


async def run_server() -> None:
    # Read basic config
    host = os.getenv("STARTER_HOST", "0.0.0.0")
    try:
        port = int(os.getenv("STARTER_PORT", "8080"))
    except Exception:
        port = 8080
    enable_tunnel = os.getenv("STARTER_ENABLE_TUNNEL", "false").lower() in ("1", "true", "on", "yes")

    # Create server
    server = ZenMCPStreamableServer(host=host, port=port, enable_tunnel=enable_tunnel)

    # Initialize KInfra/tunnel + OAuth server
    await server.start_server_with_kinfra()

    # Auth mode switch (default webauthn; supabase mode patches token validation)
    auth_mode = os.getenv("AUTH_MODE", "webauthn").lower()
    if auth_mode == "supabase":
        await attach_supabase_validator(server)
        # Disable operator approval flow for supabase mode, if present
        try:
            os.environ.setdefault("PAIRING_REQUIRE_OPERATOR_APPROVAL", "false")
        except Exception:
            pass
        logging.getLogger(__name__).info("🔐 AUTH_MODE=supabase enabled (Supabase JWT validation active)")
    else:
        logging.getLogger(__name__).info("🔐 AUTH_MODE=webauthn (local device auth + operator gating)")

    # Run uvicorn
    config = uvicorn.Config(
        server.app,
        host=server.host,
        port=server.allocated_port or server.port,
        log_level="info",
        access_log=True,
    )
    http = uvicorn.Server(config)
    try:
        await http.serve()
    finally:
        await server.cleanup_tunnel()


def main() -> None:
    asyncio.run(run_server())


if __name__ == "__main__":
    main()

