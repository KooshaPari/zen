#!/usr/bin/env python3
"""
Zen MCP HTTP Server Deployment Script with KInfra Integration

This script provides an easy way to deploy the Zen MCP HTTP server with
KInfra's smart port allocation and Cloudflare tunnel management.

Features:
- Smart port allocation with fallback strategies
- Automatic Cloudflare tunnel setup
- Health monitoring and status reporting
- Configuration management
- Graceful shutdown handling

Usage:
    python deploy_mcp_http.py                     # Quick start with defaults
    python deploy_mcp_http.py --tunnel            # Enable tunneling
    python deploy_mcp_http.py --domain custom.com # Custom domain
    python deploy_mcp_http.py --status            # Show server status
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our server
from server_mcp_http import ZenMCPStreamableServer

# KInfra imports
try:
    kinfra_path = Path(__file__).parent / 'KInfra' / 'libraries' / 'python'
    if kinfra_path.exists():
        sys.path.insert(0, str(kinfra_path))

    from kinfra_networking import DefaultLogger
    KINFRA_AVAILABLE = True
except ImportError:
    KINFRA_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ZenMCPDeployer:
    """Deployment orchestrator for Zen MCP HTTP Server with KInfra."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.server: Optional[ZenMCPStreamableServer] = None
        self.start_time = datetime.now(timezone.utc)
        self.deployment_info = {}
        self.kinfra_logger = DefaultLogger() if KINFRA_AVAILABLE else None

        # Configure logging
        log_level = config.get('log_level', 'INFO').upper()
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True

        if self.server:
            asyncio.create_task(self.server.cleanup_tunnel())

        # Give some time for cleanup
        time.sleep(2)
        sys.exit(0)

    async def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        logger.info("🔍 Checking prerequisites...")

        issues = []

        # Check Python version

        # Check required packages
        try:
            import fastapi  # noqa: F401
            import uvicorn  # noqa: F401
        except ImportError as e:
            issues.append(f"Missing required package: {e}")

        # Check KInfra if tunnel is enabled
        if self.config.get('enable_tunnel', False):
            if not KINFRA_AVAILABLE:
                issues.append("KInfra not available but tunnel is enabled")
            else:
                # Check for cloudflared
                import shutil
                if not shutil.which('cloudflared'):
                    issues.append("cloudflared binary not found in PATH")

        if issues:
            logger.error("❌ Prerequisites check failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False

        logger.info("✅ Prerequisites check passed")
        return True

    async def deploy(self) -> bool:
        """Deploy the Zen MCP HTTP server."""
        logger.info("🚀 Starting Zen MCP HTTP Server deployment...")

        # Check prerequisites
        if not await self.check_prerequisites():
            return False

        try:
            # Create server instance
            self.server = ZenMCPStreamableServer(
                host=self.config.get('host', '0.0.0.0'),
                port=self.config.get('port'),
                enable_tunnel=self.config.get('enable_tunnel', False),
                tunnel_domain=self.config.get('tunnel_domain', 'zen.kooshapari.com'),
                port_strategy=self.config.get('port_strategy', 'preferred')
            )

            # Start server with KInfra
            port = await self.server.start_server_with_kinfra()

            # Save deployment info
            self.deployment_info = {
                'deployment_id': f"zen-mcp-{int(time.time())}",
                'start_time': self.start_time.isoformat(),
                'host': self.server.host,
                'port': port,
                'local_url': f"http://{self.server.host}:{port}",
                'tunnel_enabled': self.server.enable_tunnel,
                'tunnel_url': self.server.tunnel_url,
                'tunnel_domain': self.server.tunnel_domain,
                'pid': os.getpid(),
                'version': '1.0.0',
                'kinfra_available': KINFRA_AVAILABLE
            }

            # Save deployment info to file
            await self.save_deployment_info()

            # Display deployment summary
            self.display_deployment_summary()

            return True

        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return False

    async def save_deployment_info(self):
        """Save deployment information to a file."""
        try:
            config_dir = Path.home() / '.zen-mcp'
            config_dir.mkdir(exist_ok=True)

            info_file = config_dir / 'deployment.json'
            with open(info_file, 'w') as f:
                json.dump(self.deployment_info, f, indent=2)

            logger.info(f"📄 Deployment info saved to: {info_file}")

        except Exception as e:
            logger.warning(f"Failed to save deployment info: {e}")

    def display_deployment_summary(self):
        """Display a nice deployment summary."""
        info = self.deployment_info

        print("\n" + "="*60)
        print("🎉 ZEN MCP HTTP SERVER DEPLOYMENT SUCCESSFUL")
        print("="*60)
        print(f"📋 Deployment ID: {info['deployment_id']}")
        print(f"🕐 Started at: {info['start_time']}")
        print(f"🏠 Host: {info['host']}")
        print(f"🔌 Port: {info['port']}")
        print(f"📍 Process ID: {info['pid']}")
        print()
        print("🔗 CONNECTION URLS:")
        print(f"   Local:      {info['local_url']}")
        print(f"   MCP:        {info['local_url']}/mcp")
        print(f"   Health:     {info['local_url']}/healthz")
        print(f"   Docs:       {info['local_url']}/docs")

        if info['tunnel_enabled'] and info['tunnel_url']:
            print()
            print("🌐 PUBLIC TUNNEL URLS:")
            print(f"   Public:     {info['tunnel_url']}")
            print(f"   MCP:        {info['tunnel_url']}/mcp")
            print(f"   Health:     {info['tunnel_url']}/healthz")
            print(f"   Domain:     {info['tunnel_domain']}")

        print()
        print("💡 USAGE TIPS:")
        print("   • Test connection: curl http://localhost:{}/healthz".format(info['port']))
        print("   • View status: python deploy_mcp_http.py --status")
        print("   • Stop server: Ctrl+C or kill {}".format(info['pid']))
        print("   • View logs: Check console output above")
        print()
        print("🔧 KINFRA STATUS:")
        print(f"   Available: {'✅ Yes' if info['kinfra_available'] else '❌ No'}")
        if info['tunnel_enabled']:
            print(f"   Tunnel: {'✅ Active' if info['tunnel_url'] else '⚠️  Starting...'}")

        print("\n" + "="*60)
        print("Server is running! Press Ctrl+C to stop.")
        print("="*60 + "\n")

    async def run_server(self):
        """Run the server after deployment."""
        if not self.server:
            logger.error("No server instance available")
            return False

        try:
            # Configure uvicorn
            import uvicorn
            config = uvicorn.Config(
                self.server.app,
                host=self.server.host,
                port=self.server.allocated_port or self.server.port,
                log_level="info",
                access_log=True
            )

            server = uvicorn.Server(config)

            # Run the server
            await server.serve()

        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
            return False
        finally:
            if self.server:
                await self.server.cleanup_tunnel()

        return True


async def show_status():
    """Show current server status."""
    try:
        config_dir = Path.home() / '.zen-mcp'
        info_file = config_dir / 'deployment.json'

        if not info_file.exists():
            print("❌ No deployment information found")
            print("   Run 'python deploy_mcp_http.py' to deploy the server")
            return

        with open(info_file) as f:
            info = json.load(f)

        print("\n" + "="*50)
        print("📊 ZEN MCP SERVER STATUS")
        print("="*50)
        print(f"Deployment ID: {info.get('deployment_id', 'Unknown')}")
        print(f"Started: {info.get('start_time', 'Unknown')}")
        print(f"PID: {info.get('pid', 'Unknown')}")
        print(f"Local URL: {info.get('local_url', 'Unknown')}")

        if info.get('tunnel_url'):
            print(f"Public URL: {info.get('tunnel_url')}")

        # Check if process is running
        pid = info.get('pid')
        if pid:
            try:
                os.kill(int(pid), 0)  # Check if process exists
                status = "🟢 Running"
            except (OSError, ValueError):
                status = "🔴 Stopped"
        else:
            status = "❓ Unknown"

        print(f"Status: {status}")

        # Try to ping health endpoint
        if info.get('local_url'):
            try:
                import aiohttp  # noqa: F401
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{info['local_url']}/healthz", timeout=5) as resp:
                        if resp.status == 200:
                            health = "🟢 Healthy"
                        else:
                            health = f"🟡 Status {resp.status}"
            except Exception:
                health = "🔴 Unreachable"

            print(f"Health: {health}")

        print("="*50 + "\n")

    except Exception as e:
        print(f"❌ Error checking status: {e}")


def create_config_from_args(args) -> dict[str, Any]:
    """Create configuration from command line arguments."""
    config = {
        'host': args.host,
        'log_level': args.log_level,
        'enable_tunnel': args.tunnel,
        'tunnel_domain': args.domain,
        'port_strategy': args.port_strategy
    }

    if args.port:
        config['port'] = args.port

    return config


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy Zen MCP HTTP Server with KInfra integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Quick start with smart port allocation
  %(prog)s --tunnel                 # Enable Cloudflare tunnel
  %(prog)s --domain custom.com      # Use custom domain
  %(prog)s --port 3000 --tunnel     # Preferred port with tunnel
  %(prog)s --status                 # Show server status
  %(prog)s --port-strategy dynamic  # Use dynamic port allocation
  %(prog)s --log-level DEBUG        # Enable debug logging

Environment Variables:
  TUNNEL_DOMAIN                     # Override default tunnel domain
  KINFRA_TUNNEL_ENABLED            # Enable tunnel (set to 'true')
  PORT                             # Override default port
        """
    )

    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, help="Preferred port (smart allocation if not specified)")
    parser.add_argument("--tunnel", action="store_true", help="Enable Cloudflare tunnel")
    parser.add_argument("--domain", default="zen.kooshapari.com", help="Tunnel domain")
    parser.add_argument("--port-strategy", choices=["preferred", "dynamic", "env"],
                       default="preferred", help="Port allocation strategy")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--status", action="store_true", help="Show server status")
    parser.add_argument("--config", help="Configuration file path")

    args = parser.parse_args()

    # Handle status command
    if args.status:
        await show_status()
        return

    # Override with environment variables
    if os.getenv("KINFRA_TUNNEL_ENABLED") == "true":
        args.tunnel = True
    if os.getenv("TUNNEL_DOMAIN"):
        args.domain = os.getenv("TUNNEL_DOMAIN")
    if os.getenv("PORT") and not args.port:
        try:
            args.port = int(os.getenv("PORT"))
        except ValueError:
            pass

    # Create configuration
    config = create_config_from_args(args)

    # Load config file if specified
    if args.config:
        try:
            with open(args.config) as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")

    # Create and run deployer
    deployer = ZenMCPDeployer(config)

    # Deploy server
    if await deployer.deploy():
        # Run server
        await deployer.run_server()
    else:
        logger.error("Deployment failed")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Deployment cancelled by user")
    except Exception as e:
        logger.error(f"Deployment error: {e}")
        sys.exit(1)
