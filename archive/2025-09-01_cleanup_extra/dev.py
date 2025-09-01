#!/usr/bin/env python3
"""
Development server with hot-reloading and persistent Cloudflare tunnel.

This script manages:
1. A persistent Cloudflare tunnel that survives server reloads
2. Uvicorn with hot-reloading for the MCP HTTP server
3. Fixed port allocation to maintain tunnel connection
"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_tunnel_running():
    """Check if cloudflared tunnel is already running."""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        return 'cloudflared' in result.stdout and 'config-zen.yml' in result.stdout
    except Exception:
        return False

def start_tunnel(port):
    """Start Cloudflare tunnel if not already running."""
    tunnel_config = Path.home() / ".cloudflared" / "config-zen.yml"

    if not tunnel_config.exists():
        logger.error(f"Tunnel config not found: {tunnel_config}")
        return None

    # Update config to use our dev port
    config_data = tunnel_config.read_text()
    import re
    config_data = re.sub(r'service: http://localhost:\d+', f'service: http://localhost:{port}', config_data)
    tunnel_config.write_text(config_data)
    logger.info(f"Updated tunnel config to use port {port}")

    if check_tunnel_running():
        logger.info("✅ Cloudflare tunnel already running")
        return None

    logger.info("🚇 Starting Cloudflare tunnel...")
    tunnel_proc = subprocess.Popen(
        ['cloudflared', 'tunnel', '--config', str(tunnel_config), 'run'],
        stdout=open('/tmp/cloudflared-dev.log', 'w'),
        stderr=subprocess.STDOUT
    )

    # Wait for tunnel to connect
    time.sleep(3)

    if tunnel_proc.poll() is None:
        logger.info(f"✅ Tunnel started (PID: {tunnel_proc.pid})")
        logger.info("📡 Tunnel accessible at: https://zen.kooshapari.com")
        return tunnel_proc
    else:
        logger.error("❌ Failed to start tunnel")
        return None

def main():
    # Default port for development
    port = int(os.getenv("MCP_PORT", "8080"))

    # Ensure virtual environment is activated
    venv_path = Path(__file__).parent / ".zen_venv"
    if venv_path.exists() and sys.prefix != str(venv_path):
        logger.warning("Virtual environment not activated. Please run: source .zen_venv/bin/activate")

    # Start tunnel if needed
    tunnel_proc = None
    if os.getenv("DISABLE_TUNNEL", "").lower() not in ("1", "true", "yes"):
        tunnel_proc = start_tunnel(port)

    # Setup signal handlers to cleanup tunnel on exit
    def cleanup(signum=None, frame=None):
        logger.info("\n🛑 Shutting down...")
        if tunnel_proc and tunnel_proc.poll() is None:
            logger.info("Stopping tunnel...")
            tunnel_proc.terminate()
            tunnel_proc.wait(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # Run uvicorn with hot-reloading
        logger.info(f"🔥 Starting development server on port {port} with hot-reloading...")
        logger.info(f"📍 Local: http://localhost:{port}")
        logger.info("📡 Public: https://zen.kooshapari.com")
        logger.info("👁️  Watching for file changes...")

        # Set environment variable to indicate dev mode
        os.environ["MCP_PORT"] = str(port)
        os.environ["UVICORN_RELOAD"] = "true"

        # Run uvicorn directly
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "server_mcp_http:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--reload",
            "--reload-dir", str(Path(__file__).parent),
            "--reload-include", "*.py",
            "--reload-exclude", "logs/*",
            "--reload-exclude", "*.log",
            "--reload-exclude", "__pycache__/*",
            "--reload-exclude", ".git/*",
            "--log-level", "info",
            "--access-log"
        ])
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    main()
