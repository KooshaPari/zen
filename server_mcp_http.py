#!/usr/bin/env python3
"""
Zen MCP Server with Streamable HTTP Transport

This module implements the MCP (Model Context Protocol) Streamable HTTP transport
as specified in protocol version 2025-03-26. It provides remote HTTP access to
the Zen MCP Server tools and capabilities with no authentication required.

Features:
- Streamable HTTP transport (MCP spec 2025-03-26)
- FastAPI-based HTTP endpoints
- No authentication (open access)
- Session management with secure UUIDs
- Bidirectional streaming support
- Full MCP protocol compliance
- Integration with existing Zen tools
"""

import asyncio
import atexit
import contextvars
import json
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv

    # Find .env file in the same directory as this script
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
        print(f"✅ Loaded environment from {env_file}")
    else:
        # Try loading from current working directory
        load_dotenv()
        print("✅ Loaded environment variables")
except ImportError:
    # python-dotenv not available - this is fine, environment variables can still be passed directly
    print("ℹ️  python-dotenv not available, using direct environment variables")
except Exception as e:
    print(f"⚠️  Error loading .env file: {e}")
    # Continue without .env file

# KInfra imports for smart networking and tunneling
try:
    # Add KInfra to path if available
    kinfra_path = Path(__file__).parent / 'KInfra' / 'libraries' / 'python'
    if kinfra_path.exists():
        sys.path.insert(0, str(kinfra_path))

    from kinfra_networking import (
        DefaultLogger,
        allocate_free_port,
        ensure_named_tunnel_route,
        wait_for_health,
    )
    from tunnel_manager import (
        AsyncTunnelManager,
        TunnelConfig,
        TunnelType,
    )
    KINFRA_AVAILABLE = True
except ImportError:
    KINFRA_AVAILABLE = False
    # Note: logger not yet defined at import time

# KInfra named tunnel automation - direct import from KInfra library
kinfra_ensure_named_tunnel_autocreate = None
try:
    # Import directly from KInfra library
    kinfra_path = os.path.expanduser("~/KInfra/libraries/python")
    if kinfra_path not in sys.path:
        sys.path.insert(0, kinfra_path)

    from kinfra_networking import ensure_named_tunnel_autocreate as kinfra_ensure_named_tunnel_autocreate
    from kinfra_networking import ensure_named_tunnel_route
except Exception:
    # Fallback to local shim if available
    try:
        from utils.kinfra_helpers import ensure_named_tunnel_autocreate as kinfra_ensure_named_tunnel_autocreate
    except Exception:
        pass

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Core MCP and FastAPI imports
try:
    import uvicorn
    from fastapi import FastAPI, Header, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError as e:
    raise RuntimeError(
        "MCP SDK (with FastMCP) is required for the Streamable HTTP server. "
        "Install a compatible version: pip install 'mcp>=1.0.0'"
    ) from e

# Zen server imports
from tools import get_all_tools  # noqa: E402
from tools.shared.agent_models import AgentType  # noqa: E402
from utils.agent_prompts import (  # noqa: E402
    enhance_agent_message,
    format_agent_summary,
    parse_agent_output,
)
from utils.conversation_memory import ConversationMemoryManager  # noqa: E402
from utils.streaming_protocol import (  # noqa: E402
    InputTransformationPipeline,
    StreamingResponseParser,
    StreamMessageType,
    create_sse_stream,
    get_streaming_manager,
)

logger = logging.getLogger(__name__)

# Correlation ID context for logs
_cid_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="-")


class CidLogFilter(logging.Filter):
    """Inject correlation id from context into log records as record.cid."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.cid = _cid_ctx.get()
        except Exception:
            record.cid = "-"
        return True


class JSONLogFormatter(logging.Formatter):
    """Simple structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "cid": getattr(record, "cid", "-"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "pathname"):
            payload["file"] = record.pathname
        if hasattr(record, "lineno"):
            payload["line"] = record.lineno
        return json.dumps(payload, ensure_ascii=False)

# Attach rotating file logging similar to stdio server (best-effort)
try:
    from logging.handlers import RotatingFileHandler

    class LocalTimeFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            ct = self.converter(record.created)
            if datefmt:
                s = time.strftime(datefmt, ct)
            else:
                t = time.strftime("%Y-%m-%d %H:%M:%S", ct)
                s = f"{t},{record.msecs:03.0f}"
            return s

    root_logger = logging.getLogger()
    # Only add once
    if not any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        # Select formatter based on ENV
        log_format_env = os.getenv("LOG_FORMAT", "text").lower()
        if log_format_env == "json":
            fmt = JSONLogFormatter()
        else:
            fmt = LocalTimeFormatter("%(asctime)s - %(name)s - %(levelname)s - cid=%(cid)s - %(message)s")

        file_handler = RotatingFileHandler(log_dir / "mcp_http_server.log", maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8")
        file_handler.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
        file_handler.setFormatter(fmt)
        file_handler.addFilter(CidLogFilter())
        root_logger.addHandler(file_handler)

        activity = logging.getLogger("mcp_activity")
        activity_handler = RotatingFileHandler(log_dir / "mcp_activity.log", maxBytes=10 * 1024 * 1024, backupCount=2, encoding="utf-8")
        activity_handler.setLevel(logging.INFO)
        if log_format_env == "json":
            activity_handler.setFormatter(JSONLogFormatter())
        else:
            activity_handler.setFormatter(LocalTimeFormatter("%(asctime)s - cid=%(cid)s - %(message)s"))
        activity.addFilter(CidLogFilter())
        activity.addHandler(activity_handler)
except Exception:
    pass


# Local provider bootstrap (minimal parity with stdio server)
def _configure_providers_http() -> None:
    from providers.base import ProviderType
    from providers.openai_provider import OpenAIModelProvider
    from providers.openrouter import OpenRouterProvider
    from providers.registry import ModelProviderRegistry
    from providers.xai import XAIModelProvider
    try:
        from providers.gemini import GeminiModelProvider  # optional
    except Exception:
        GeminiModelProvider = None  # type: ignore
    try:
        from providers.dial import DIALModelProvider
    except Exception:
        DIALModelProvider = None  # type: ignore
    from providers.custom import CustomProvider

    has_any = False

    # Native APIs
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and gemini_key != "your_gemini_api_key_here" and GeminiModelProvider:
        ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)
        has_any = True
        logger.info("Gemini provider registered")

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key != "your_openai_api_key_here":
        ModelProviderRegistry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)
        has_any = True
        logger.info("OpenAI provider registered")

    xai_key = os.getenv("XAI_API_KEY")
    if xai_key and xai_key != "your_xai_api_key_here":
        ModelProviderRegistry.register_provider(ProviderType.XAI, XAIModelProvider)
        has_any = True
        logger.info("X.AI provider registered")

    dial_key = os.getenv("DIAL_API_KEY")
    if dial_key and dial_key != "your_dial_api_key_here" and DIALModelProvider:
        ModelProviderRegistry.register_provider(ProviderType.DIAL, DIALModelProvider)
        has_any = True
        logger.info("DIAL provider registered")

    # Custom local/self-hosted
    custom_url = os.getenv("CUSTOM_API_URL")
    if custom_url:
        def custom_factory(api_key=None):
            return CustomProvider(api_key=(api_key or os.getenv("CUSTOM_API_KEY", "")), base_url=custom_url)

        ModelProviderRegistry.register_provider(ProviderType.CUSTOM, custom_factory)
        has_any = True
        logger.info(f"Custom provider registered ({custom_url})")

    # OpenRouter last
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key and or_key != "your_openrouter_api_key_here":
        ModelProviderRegistry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)
        has_any = True
        logger.info("OpenRouter provider registered")

    if not has_any:
        raise ValueError("No AI providers configured. Set at least one API key or CUSTOM_API_URL.")

    # Load any persisted batches (best-effort)
    try:
        from utils.batch_registry import load_batches_from_disk, save_batches_to_disk
        load_batches_from_disk()
        atexit.register(lambda: save_batches_to_disk())
        logger.info("Batch registry loaded (HTTP MCP)")
    except Exception:
        pass


# Minimal prompt templates (parity for common tools)
PROMPT_TEMPLATES_HTTP = {
    "chat": {
        "name": "chat",
        "description": "Chat and brainstorm ideas",
        "template": "Chat with {model} about this",
    },
    "thinkdeep": {
        "name": "thinkdeeper",
        "description": "Deep thinking workflow",
        "template": "Start deep thinking with {model} using {thinking_mode} mode",
    },
    "planner": {
        "name": "planner",
        "description": "Create a detailed plan",
        "template": "Create a detailed plan with {model}",
    },
    "consensus": {
        "name": "consensus",
        "description": "Multi-model consensus",
        "template": "Start consensus workflow with {model}",
    },
    "codereview": {
        "name": "review",
        "description": "Perform a comprehensive code review",
        "template": "Perform a comprehensive code review with {model}",
    },
    "precommit": {
        "name": "precommit",
        "description": "Pre-commit validation",
        "template": "Start pre-commit validation with {model}",
    },
    "debug": {
        "name": "debug",
        "description": "Debug an issue",
        "template": "Help debug this issue with {model}",
    },
    "listmodels": {
        "name": "listmodels",
        "description": "List available AI models",
        "template": "List available models",
    },
    "version": {
        "name": "version",
        "description": "Show server version",
        "template": "Show server version",
    },
}


class ZenMCPStreamableServer:
    """Zen MCP Server with Streamable HTTP transport and KInfra tunnel integration."""

    def __init__(self,
                 host: str = "0.0.0.0",
                 port: Optional[int] = None,
                 enable_tunnel: Optional[bool] = None,
                 tunnel_domain: Optional[str] = None,
                 port_strategy: Optional[str] = None):
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI required for HTTP transport. Install with: pip install fastapi uvicorn")
        # MCP SDK is required for Streamable HTTP transport compliance

        self.host = host
        self.port = port or 8080  # Default port, may be replaced by smart allocation
        # Defaults: enable tunnel and dynamic port strategy unless explicitly disabled
        self.port_strategy = port_strategy or os.getenv("KINFRA_PORT_STRATEGY", "dynamic")
        env_disable = os.getenv("DISABLE_TUNNEL", "").lower() in ("1", "true", "yes")
        self.enable_tunnel = (enable_tunnel if enable_tunnel is not None else not env_disable)
        # Accept full host via env; None means auto-compute service/root later
        self.tunnel_domain = tunnel_domain or os.getenv("FULL_TUNNEL_HOST")
        self.allocated_port: Optional[int] = None
        self.tunnel_manager: Optional[AsyncTunnelManager] = None
        self.tunnel_url: Optional[str] = None
        # Tunnel health monitor
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._tunnel_recovery_lock: Optional[asyncio.Lock] = asyncio.Lock()
        self.kinfra_logger = DefaultLogger() if KINFRA_AVAILABLE else None

        # Tunnel health tracking
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._tunnel_recovery_lock: Optional[asyncio.Lock] = asyncio.Lock()
        self._tunnel_last_good_host: Optional[str] = None
        self._tunnel_last_ok_ts: Optional[float] = None
        self._tunnel_last_recovery_ts: Optional[float] = None
        self._tunnel_failures_total: int = 0
        self._tunnel_success_total: int = 0

        self.sessions: dict[str, dict[str, Any]] = {}
        self.streaming_manager = get_streaming_manager()
        self.input_transformer = InputTransformationPipeline()
        self.response_parser = StreamingResponseParser(self.streaming_manager)
        self.conversation_memory = ConversationMemoryManager()

        # Internal: helpers for rich request logging without leaking secrets
        def _mask_value(val: str) -> str:
            if not isinstance(val, str):
                return val
            if len(val) <= 8:
                return "***"
            return val[:4] + "…" + val[-4:]

        def _mask_sensitive(data):
            try:
                if isinstance(data, dict):
                    masked = {}
                    for k, v in data.items():
                        lk = str(k).lower()
                        if any(s in lk for s in ("secret", "token", "assertion", "password", "authorization", "code", "refresh")):
                            masked[k] = _mask_value(v if isinstance(v, str) else str(v))
                        else:
                            masked[k] = _mask_sensitive(v)
                    return masked
                if isinstance(data, list):
                    return [_mask_sensitive(x) for x in data]
                return data
            except Exception:
                return data

        self._mask_sensitive = _mask_sensitive
        self._cid = lambda req: getattr(req.state, "correlation_id", "-")

        # Create FastAPI app
        self.app = FastAPI(
            title="Zen MCP Streamable HTTP Server",
            description="Model Context Protocol server with Streamable HTTP transport",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        # Mount static files directory
        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
            print(f"✅ Mounted static files from {static_dir}")

        # Server-Timing details toggle (can be overridden later by CLI)
        self.server_timing_details = os.getenv("SERVER_TIMING_DETAILS", "false").lower() in ("1", "true", "yes", "on")

        # Helpers for fine-grained timing marks
        def _timing_start(req: Request, label: str) -> float:
            try:
                if not hasattr(req.state, "_timings"):
                    req.state._timings = []  # type: ignore[attr-defined]
                if not hasattr(req.state, "_timing_starts"):
                    req.state._timing_starts = {}  # type: ignore[attr-defined]
                t0 = time.perf_counter()
                req.state._timing_starts[label] = t0  # type: ignore[attr-defined]
                return t0
            except Exception:
                return time.perf_counter()

        def _timing_end(req: Request, label: str) -> None:
            try:
                t0 = req.state._timing_starts.get(label)  # type: ignore[attr-defined]
                if t0 is None:
                    return
                dur_ms = int((time.perf_counter() - t0) * 1000)
                req.state._timings.append((label, dur_ms))  # type: ignore[attr-defined]
            except Exception:
                pass

        self._timing_start = _timing_start
        self._timing_end = _timing_end

        # Correlation-ID and timing middleware
        @self.app.middleware("http")
        async def add_correlation_and_timing(request: Request, call_next):
            # Reuse inbound correlation id if provided
            corr_id = request.headers.get("X-Correlation-Id") or str(uuid.uuid4())
            request.state.correlation_id = corr_id
            # Bind correlation-id into logging context
            _tok = _cid_ctx.set(corr_id)
            start = time.perf_counter()
            try:
                response = await call_next(request)
            except Exception as e:
                duration_ms = int((time.perf_counter() - start) * 1000)
                logger.warning(f"❌ {request.method} {request.url.path} cid={corr_id} dur_ms={duration_ms} error={e}")
                raise
            duration_ms = int((time.perf_counter() - start) * 1000)
            try:
                response.headers["X-Correlation-Id"] = corr_id
                # Compose Server-Timing header with optional fine-grained marks
                timing_header = f"total;dur={duration_ms}"
                if self.server_timing_details and hasattr(request.state, "_timings"):
                    try:
                        parts = [timing_header]
                        for label, dur in getattr(request.state, "_timings", []):
                            parts.append(f"{label};dur={dur}")
                        timing_header = ", ".join(parts)
                    except Exception:
                        pass
                response.headers["Server-Timing"] = timing_header
            except Exception:
                pass
            logger.info(f"⏱ {request.method} {request.url.path} {getattr(response, 'status_code', '?')} cid={corr_id} dur_ms={duration_ms}")
            # Reset logging context
            try:
                _cid_ctx.reset(_tok)
            except Exception:
                pass
            return response

        # Configure providers and persistence parity with stdio MCP server
        _configure_providers_http()

        # Initialize OAuth 2.0 + WebAuthn authentication if enabled
        self.oauth2_server = None
        self.device_auth = None
        self.auth_enabled = os.getenv("ENABLE_OAUTH2_AUTH", "true").lower() in ("true", "1", "yes", "on")
        print(f"🔍 OAuth config: ENABLE_OAUTH2_AUTH={os.getenv('ENABLE_OAUTH2_AUTH')}, auth_enabled={self.auth_enabled}")

        if self.auth_enabled:
            try:
                # Import OAuth 2.0 components
                from auth.device_auth import UnifiedDeviceAuth, setup_device_auth_endpoints

                # Initialize WebAuthn (domain will be updated after tunnel setup)
                self.device_auth = UnifiedDeviceAuth(domain="localhost")
                print("✅ Device auth initialized")

                # OAuth 2.0 server will be created after KInfra setup with proper issuer URL
                self.oauth2_server = None
                print("✅ OAuth server will be initialized after KInfra setup")

                # Setup authentication endpoints
                auth_setup = setup_device_auth_endpoints(self.app, self.device_auth)
                self.require_auth = auth_setup["require_auth"]
                # Store the OAuth integration server for token sharing
                self.oauth_integration_server = auth_setup.get("oauth_server")

                # Setup OAuth 2.0 endpoints
                self.setup_oauth2_endpoints()

                # Initialize and mount console pairing (passwordless enrollment)
                try:
                    from auth.pairing import PairingService  # lazy import

                    ttl = int(os.getenv("PAIRING_TTL_SECONDS", "600"))
                    code_len = int(os.getenv("PAIRING_CODE_LENGTH", "6"))
                    max_attempts = int(os.getenv("PAIRING_MAX_ATTEMPTS", "5"))
                    redis_enabled = (
                        os.getenv("ZEN_STORAGE", os.getenv("ZEN_STORAGE_MODE", "memory"))
                        .lower()
                        == "redis"
                    )
                    require_operator = os.getenv("PAIRING_REQUIRE_OPERATOR_APPROVAL", "true").lower() in ("1","true","on","yes")

                    self.pairing_service = PairingService(
                        webauthn=self.device_auth.webauthn,
                        ttl_seconds=ttl,
                        code_length=code_len,
                        max_attempts=max_attempts,
                        redis_enabled=redis_enabled,
                        require_operator_approval=require_operator,
                    )
                    self.setup_pairing_endpoints()
                    logger.info("Console pairing endpoints enabled")
                except Exception as e:
                    logger.warning(f"Pairing service not initialized: {e}")

                logger.info("OAuth 2.0 + WebAuthn authentication enabled")
            except Exception as e:
                print(f"❌ OAuth 2.0 authentication setup failed: {e}")
                import traceback
                traceback.print_exc()
                logger.warning(f"OAuth 2.0 authentication setup failed: {e}")
                self.auth_enabled = False

        # Add CORS middleware for browser access (configurable)
        cors_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "*")
        cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
        # If credentials are enabled, avoid wildcard '*' to satisfy browsers
        allow_credentials = True
        if allow_credentials and cors_origins == ["*"]:
            # Sensible defaults for local dev + same-origin
            cors_origins = [
                f"http://{('localhost' if self.host in ('0.0.0.0', '::') else self.host)}:{self.allocated_port or self.port}",
                "http://localhost:6274",
                "http://127.0.0.1:6274",
            ]
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=allow_credentials,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"]
        )

        # Expose a stable local base for operator approval URLs
        try:
            self.app.state.local_base = f"http://localhost:{self.allocated_port or self.port}"
        except Exception:
            pass

        # Create MCP server instance
        self.mcp = FastMCP("ZenMCP")

        # Setup HTTP endpoints
        self.setup_endpoints()

        # Register Zen tools
        self.register_zen_tools()

        # Initialize KInfra components if available
        self._startup_time = datetime.now(timezone.utc)

    def setup_oauth2_endpoints(self):
        """Setup OAuth 2.0 endpoints."""
        print("🔧 Setting up OAuth 2.0 endpoints")
        # Note: self.oauth2_server will be initialized after tunnel setup

        @self.app.get("/oauth/authorize")
        @self.app.post("/oauth/authorize")
        async def oauth_authorize(request: Request):
            """OAuth 2.0 authorization endpoint."""
            if not self.oauth2_server:
                raise HTTPException(status_code=503, detail="OAuth 2.0 server not initialized")
            ip = request.client.host if request.client else "unknown"
            ua = request.headers.get("user-agent")
            logger.info(f"➡️  {request.method} /oauth/authorize from {ip} ua={ua} query='{request.url.query}' cid={self._cid(request)}")
            try:
                resp = await self.oauth2_server.authorization_endpoint(request)
                logger.info(f"✅ /oauth/authorize completed cid={self._cid(request)}")
                return resp
            except Exception as e:
                logger.warning(f"❌ /oauth/authorize error: {e} cid={self._cid(request)}")
                raise

        @self.app.post("/oauth/token")
        async def oauth_token(request: Request):
            """OAuth 2.0 token endpoint."""
            if not self.oauth2_server:
                raise HTTPException(status_code=503, detail="OAuth 2.0 server not initialized")
            ip = request.client.host if request.client else "unknown"
            ua = request.headers.get("user-agent")
            logger.info(f"➡️  POST /oauth/token from {ip} ua={ua} cid={self._cid(request)}")
            try:
                if self.server_timing_details:
                    self._timing_start(request, "token_handler")
                resp = await self.oauth2_server.token_endpoint(request)
                if self.server_timing_details:
                    self._timing_end(request, "token_handler")
                logger.info(f"✅ /oauth/token completed cid={self._cid(request)}")
                return resp
            except Exception as e:
                logger.warning(f"❌ /oauth/token error: {e} cid={self._cid(request)}")
                raise

        @self.app.post("/oauth/revoke")
        async def oauth_revoke(request: Request):
            """OAuth 2.0 token revocation endpoint."""
            if not self.oauth2_server:
                raise HTTPException(status_code=503, detail="OAuth 2.0 server not initialized")
            ip = request.client.host if request.client else "unknown"
            ua = request.headers.get("user-agent")
            logger.info(f"➡️  POST /oauth/revoke from {ip} ua={ua} cid={self._cid(request)}")
            try:
                resp = await self.oauth2_server.revocation_endpoint(request)
                logger.info(f"✅ /oauth/revoke completed cid={self._cid(request)}")
                return resp
            except Exception as e:
                logger.warning(f"❌ /oauth/revoke error: {e} cid={self._cid(request)}")
                raise

        @self.app.post("/oauth/introspect")
        async def oauth_introspect(request: Request):
            """OAuth 2.0 token introspection endpoint."""
            if not self.oauth2_server:
                raise HTTPException(status_code=503, detail="OAuth 2.0 server not initialized")
            ip = request.client.host if request.client else "unknown"
            ua = request.headers.get("user-agent")
            logger.info(f"➡️  POST /oauth/introspect from {ip} ua={ua} cid={self._cid(request)}")
            try:
                resp = await self.oauth2_server.introspection_endpoint(request)
                logger.info(f"✅ /oauth/introspect completed cid={self._cid(request)}")
                return resp
            except Exception as e:
                logger.warning(f"❌ /oauth/introspect error: {e} cid={self._cid(request)}")
                raise

    def setup_pairing_endpoints(self):
        """Setup console pairing (passwordless registration) endpoints."""
        from fastapi import Body
        from fastapi.responses import HTMLResponse, JSONResponse

        enable_pairing = os.getenv("ENABLE_PAIRING_CONSOLE", "true").lower() in ("1", "true", "on", "yes")

        def _is_local_ip(request: Request) -> bool:
            ip = request.client.host if request.client else ""
            return ip in ("127.0.0.1", "::1", "localhost")

        @self.app.post("/auth/pairing/start")
        async def pairing_start(request: Request):
            if not enable_pairing:
                raise HTTPException(status_code=403, detail="Pairing disabled")
            # Basic guard: allow local console or an authenticated session
            allow_local = os.getenv("PAIRING_ALLOW_LOCALHOST", "true").lower() in ("1", "true", "on")
            session_ok = False
            try:
                sid = request.headers.get("X-Session-ID") or request.cookies.get("mcp_session")
                if sid and self.device_auth.validate_session(sid):  # type: ignore[attr-defined]
                    session_ok = True
            except Exception:
                session_ok = False

            if not session_ok and not (allow_local and _is_local_ip(request)):
                raise HTTPException(status_code=401, detail="Unauthorized")

            info = self.pairing_service.start_pairing(created_by_ip=(request.client.host if request.client else None))
            # Structured logs for operators
            base = str(request.base_url).rstrip('/')
            full_url = f"{base}{info['qr_url']}"
            logger.info(f"PAIRING_START pairing_id={info['pairing_id']} expires_at={int(info['expires_at'])}")
            logger.info(f"PAIRING_READY pairing_id={info['pairing_id']} code={info['display_code']} url={full_url}")

            # Optional pretty console box (ASCII) for operators
            if os.getenv("PAIRING_LOG_BOX", "true").lower() in ("1", "true", "on", "yes"):
                pid = info['pairing_id']
                code = info['display_code']
                url = full_url
                # Compute box width
                lines = [
                    f"Pairing ID: {pid}",
                    f"Code: {code}",
                    f"URL: {url}",
                ]
                width = max(len(s) for s in lines) + 2
                top = "+" + ("-" * (width + 2)) + "+"
                logger.info(top)
                logger.info(f"| {'PAIRING START'.ljust(width)} |")
                logger.info("+" + ("-" * (width + 2)) + "+")
                for s in lines:
                    logger.info(f"| {s.ljust(width)} |")
                logger.info(top)
            return info

        @self.app.get("/auth/register")
        async def pairing_register_page(pairing: Optional[str] = None):
            html = rf"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Register Device</title>
  <style>
    body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 640px; margin: 48px auto; padding: 0 16px; }}
    .card {{ background: #fff; border: 1px solid #e1e4e8; border-radius: 12px; padding: 24px; }}
    input, button {{ padding: 12px; border-radius: 8px; border: 1px solid #ccc; width: 100%; margin: 6px 0; }}
    button {{ background: #007AFF; color: white; border: none; cursor: pointer; }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .status {{ margin-top: 12px; }}
  </style>
  <script>
    async function claimPairing() {{
      const pairingId = document.getElementById('pairing_id').value.trim();
      const code = document.getElementById('code').value.trim();
      const userId = document.getElementById('user_id').value.trim();
      const deviceName = document.getElementById('device_name').value.trim() || 'My Device';
      if(!pairingId || !code || !userId) {{ setStatus('Please fill all fields', true); return; }}
      setStatus('Verifying code...');
      const claimResp = await fetch('/auth/pairing/claim', {{
        method: 'POST', headers: {{'Content-Type':'application/json'}},
        body: JSON.stringify({{ pairing_id: pairingId, code, user_id: userId, device_name: deviceName }})
      }});
      let data;
      if(claimResp.status === 202) {{
        setStatus('Waiting for operator approval...');
        data = await waitForReady(pairingId, userId, deviceName);
        if(!data) {{ setStatus('Approval failed or timed out', true); return; }}
      }} else if(!claimResp.ok) {{ const t = await claimResp.text(); setStatus('Claim failed: '+t, true); return; }}
      else {{ data = await claimResp.json(); }}
      const publicKey = data.publicKey; // WebAuthn options
      const claimToken = data.oauth_context.claim_token;
      // Decode base64url fields
      function b64uToBytes(s) {{ s = s.replace(/-/g,'+').replace(/_/g,'/'); s += '='.repeat((4 - s.length % 4) % 4); return Uint8Array.from(atob(s), c => c.charCodeAt(0)); }}
      publicKey.challenge = b64uToBytes(publicKey.challenge);
      if(publicKey.user && publicKey.user.id) {{ publicKey.user.id = b64uToBytes(publicKey.user.id); }}
      if(publicKey.excludeCredentials) {{ publicKey.excludeCredentials = publicKey.excludeCredentials.map(c => ({{...c, id: b64uToBytes(c.id)}})); }}
      setStatus('Create a new passkey with your authenticator...');
      try {{
        const cred = await navigator.credentials.create({{ publicKey }});
        const att = {{
          id: cred.id,
          challenge: data.publicKey.challenge_b64,
          response: {{
            clientDataJSON: btoa(String.fromCharCode(...new Uint8Array(cred.response.clientDataJSON))).replace(/\+/g,'-').replace(/\//g,'_').replace(/=/g,''),
            attestationObject: btoa(String.fromCharCode(...new Uint8Array(cred.response.attestationObject))).replace(/\+/g,'-').replace(/\//g,'_').replace(/=/g,''),
          }},
          pairing_id: pairingId,
          claim_token: claimToken,
          user_id: userId,
        }};
        const compResp = await fetch('/auth/pairing/complete', {{ method:'POST', headers: {{'Content-Type':'application/json'}}, body: JSON.stringify(att) }});
        if(!compResp.ok) {{ const t = await compResp.text(); setStatus('Complete failed: '+t, true); return; }}
        setStatus('✅ Device registered successfully! You may close this page.');
      }} catch(e) {{ setStatus('WebAuthn error: '+e.message, true); }}
    }}
    function setStatus(msg, err=false) {{ const el = document.getElementById('status'); el.textContent = msg; el.style.color = err?'#D8000C':'#0066CC'; }}
    async function waitForReady(pairingId, userId, deviceName) {{
      for(let i=0;i<200;i++) {{
        const r = await fetch(`/auth/pairing/ready?pairing_id=${{encodeURIComponent(pairingId)}}&user_id=${{encodeURIComponent(userId)}}&device_name=${{encodeURIComponent(deviceName)}}`);
        if(r.status === 200) return await r.json();
        if(r.status !== 202) return null;
        await new Promise(res => setTimeout(res, 1500));
      }}
      return null;
    }}
    window.addEventListener('DOMContentLoaded', () => {{
      const p = new URLSearchParams(window.location.search).get('pairing');
      if(p) document.getElementById('pairing_id').value = p;
    }});
  </script>
  </head>
  <body>
    <div class='card'>
      <h2>Register Your Device</h2>
      <div class='row'>
        <div>
          <label>Pairing ID</label>
          <input id='pairing_id' placeholder='Paste or scan to auto-fill' value='{pairing or ''}' />
        </div>
        <div>
          <label>Code</label>
          <input id='code' placeholder='6-8 digit code' />
        </div>
      </div>
      <label>User ID</label>
      <input id='user_id' placeholder='email or handle' />
      <label>Device Name (optional)</label>
      <input id='device_name' placeholder='e.g., MacBook Pro' />
      <button onclick='claimPairing()'>Claim and Register</button>
      <div id='status' class='status'></div>
    </div>
  </body>
</html>
            """
            return HTMLResponse(content=html)

        @self.app.post("/auth/pairing/claim")
        async def pairing_claim(request: Request, payload: dict = Body(...)):
            pairing_id = str(payload.get("pairing_id") or "").strip()
            code = str(payload.get("code") or "").strip()
            user_id = str(payload.get("user_id") or "").strip()
            device_name = str(payload.get("device_name") or "").strip() or f"Device-{user_id}"
            if not pairing_id or not code or not user_id:
                raise HTTPException(status_code=400, detail="pairing_id, code, and user_id are required")

            claim_info = self.pairing_service.claim_pairing(pairing_id, code, user_id)
            logger.info(f"PAIRING_CLAIMED pairing_id={pairing_id} user_id={user_id} device_name={device_name}")

            # If operator approval is required and pending, instruct client to wait
            status = self.pairing_service.get_status(pairing_id)
            if status.get("operator_required") and not status.get("operator_approved"):
                # Print the operator token ONLY to server stdout via logs
                op_token = self.pairing_service.get_operator_token_for_logging(pairing_id)
                base = str(request.base_url).rstrip('/')
                approve_url = f"{base}/auth/pairing/approve"  # local-only endpoint
                logger.info(f"APPROVAL_REQUIRED pairing_id={pairing_id} operator_token={op_token} approve_with=POST {approve_url}")
                return JSONResponse({"status": "waiting_for_operator_approval"}, status_code=202)

            # Otherwise, return WebAuthn options immediately
            options = await self.device_auth.webauthn.initiate_registration(user_id, device_name)  # type: ignore[attr-defined]
            pk = options.get("publicKey", {})
            challenge_b64 = pk.get("challenge")
            options["oauth_context"] = {
                "pairing_id": pairing_id,
                "claim_token": claim_info["claim_token"],
                "user_id": user_id,
            }
            options["publicKey"]["challenge_b64"] = challenge_b64
            return options

        @self.app.post("/auth/pairing/complete")
        async def pairing_complete(request: Request, payload: dict = Body(...)):
            pairing_id = str(payload.get("pairing_id") or "").strip()
            claim_token = str(payload.get("claim_token") or "").strip()
            if not pairing_id or not claim_token:
                raise HTTPException(status_code=400, detail="pairing_id and claim_token required")

            ok = await self.device_auth.webauthn.complete_registration(payload)  # type: ignore[attr-defined]
            if not ok:
                raise HTTPException(status_code=400, detail="WebAuthn registration failed")

            rec = self.pairing_service.complete_pairing(pairing_id, claim_token)
            logger.info(f"PAIRING_COMPLETED pairing_id={pairing_id} user_id={rec.user_id}")
            # Create authenticated session for the user and set cookie
            from fastapi import Response
            session = self.device_auth.create_session(rec.user_id or "unknown", method="webauthn", device_info={"type": "webauthn", "pairing_id": pairing_id})  # type: ignore[attr-defined]
            resp_body = {
                "success": True,
                "user_id": session.user_id,
                "session_id": session.session_id,
                "expires_at": session.expires_at,
            }
            resp = Response(content=json.dumps(resp_body), media_type="application/json")
            try:
                resp.set_cookie(
                    key="mcp_session",
                    value=session.session_id,
                    max_age=self.device_auth.session_timeout,  # type: ignore[attr-defined]
                    secure=True,
                    httponly=True,
                    samesite="strict",
                )
            except Exception:
                pass
            # Fire optional webhook (async, non-blocking)
            try:
                webhook_url = os.getenv("PAIRING_WEBHOOK_URL", "").strip()
                if webhook_url:
                    async def _notify():
                        import hashlib
                        import hmac

                        import httpx
                        body = json.dumps({
                            "event": "pairing_completed",
                            "pairing_id": pairing_id,
                            "user_id": session.user_id,
                            "session_id": session.session_id,
                            "expires_at": session.expires_at,
                            "ip": request.client.host if request.client else None,
                        }).encode()
                        headers = {"Content-Type": "application/json", "X-Pairing-Event": "pairing_completed"}
                        secret = os.getenv("PAIRING_WEBHOOK_SECRET", "").encode()
                        if secret:
                            sig = hmac.new(secret, body, hashlib.sha256).hexdigest()
                            headers["X-Pairing-Signature"] = f"sha256={sig}"
                        method = os.getenv("PAIRING_WEBHOOK_METHOD", "POST").upper()
                        timeout = float(os.getenv("PAIRING_WEBHOOK_TIMEOUT", "5.0"))
                        async with httpx.AsyncClient(timeout=timeout) as client:
                            if method == "POST":
                                r = await client.post(webhook_url, content=body, headers=headers)
                            else:
                                r = await client.request(method, webhook_url, content=body, headers=headers)
                            logger.info(f"PAIRING_WEBHOOK status={r.status_code}")
                    asyncio.create_task(_notify())
            except Exception as e:
                logger.warning(f"PAIRING_WEBHOOK error: {e}")
            return resp

        @self.app.get("/auth/pairing/status/{pairing_id}")
        async def pairing_status(pairing_id: str):
            return self.pairing_service.get_status(pairing_id)

        @self.app.get("/auth/pairing/events/{pairing_id}")
        async def pairing_events(pairing_id: str):
            """Server-Sent Events stream for real-time pairing updates.

            Emits a JSON status object whenever the pairing status changes.
            """
            from fastapi.responses import StreamingResponse

            async def event_generator():
                import json as _json
                last = None
                while True:
                    try:
                        status = self.pairing_service.get_status(pairing_id)
                    except HTTPException:
                        yield "event: end\n" + "data: {\"error\": \"not_found\"}\n\n"
                        break
                    curr = status.get("status")
                    if curr != last:
                        payload = _json.dumps(status)
                        yield f"data: {payload}\n\n"
                        last = curr
                        if curr in ("completed", "expired"):
                            yield "event: end\n" + f"data: {{\"final\": true, \"status\": \"{curr}\"}}\n\n"
                            break
                    await asyncio.sleep(1.0)

            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
            return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)

        # WebSocket events (alternative to SSE)
        from fastapi import WebSocket

        @self.app.websocket("/ws/auth/pairing/{pairing_id}")
        async def pairing_ws(websocket: WebSocket, pairing_id: str):
            await websocket.accept()
            last = None
            try:
                while True:
                    try:
                        status = self.pairing_service.get_status(pairing_id)
                    except HTTPException:
                        await websocket.send_json({"error": "not_found"})
                        break
                    curr = status.get("status")
                    if curr != last:
                        await websocket.send_json(status)
                        last = curr
                        if curr in ("completed", "expired"):
                            break
                    await asyncio.sleep(1.0)
            except Exception:
                # On disconnect or error, just exit gracefully
                pass
            finally:
                try:
                    await websocket.close()
                except Exception:
                    pass

        @self.app.post("/auth/pairing/approve")
        async def pairing_approve(request: Request):
            # Only allow loopback for approval calls
            ip = request.client.host if request.client else ""
            if ip not in ("127.0.0.1", "::1", "localhost"):
                raise HTTPException(status_code=403, detail="Local approval only")
            body = await request.json()
            pairing_id = str(body.get("pairing_id") or "").strip()
            operator_token = str(body.get("operator_token") or "").strip()
            if not pairing_id or not operator_token:
                raise HTTPException(status_code=400, detail="pairing_id and operator_token required")
            ok = self.pairing_service.approve_pairing(pairing_id, operator_token)
            if ok:
                logger.info(f"PAIRING_APPROVED pairing_id={pairing_id}")
                return {"approved": True}
            raise HTTPException(status_code=400, detail="Approval failed")

        @self.app.get("/auth/pairing/ready")
        async def pairing_ready(pairing_id: str, user_id: Optional[str] = None, device_name: Optional[str] = None):
            status = self.pairing_service.get_status(pairing_id)
            if status.get("operator_required") and not status.get("operator_approved"):
                return JSONResponse({"status": "waiting_for_operator_approval"}, status_code=202)
            # Need user_id to build options; if not present, infer from status
            uid = user_id or status.get("user_id") or "user"
            dname = device_name or (f"Device-{uid}")
            options = await self.device_auth.webauthn.initiate_registration(uid, dname)  # type: ignore[attr-defined]
            pk = options.get("publicKey", {})
            options["publicKey"]["challenge_b64"] = pk.get("challenge")
            options["oauth_context"] = {"pairing_id": pairing_id}
            return options

        @self.app.get("/oauth/consent")
        @self.app.post("/oauth/consent")
        async def oauth_consent(request: Request):
            """OAuth 2.0 consent endpoint for user authorization approval."""
            if not self.oauth2_server:
                raise HTTPException(status_code=503, detail="OAuth 2.0 server not initialized")

            ip = request.client.host if request.client else "unknown"
            ua = request.headers.get("user-agent")
            method = request.method
            logger.info(f"➡️  {method} /oauth/consent from {ip} ua={ua} cid={self._cid(request)}")

            try:
                # For GET requests, return the consent form
                if request.method == "GET":
                    # Get query parameters
                    params = dict(request.query_params)
                    client_id = params.get("client_id")
                    scope = params.get("scope", "")

                    if not client_id:
                        raise HTTPException(status_code=400, detail="Missing client_id parameter")

                    # Get client information
                    from auth.oauth2_dcr import get_dcr_manager
                    dcr_manager = get_dcr_manager()
                    client = await dcr_manager.get_client(client_id)

                    if not client:
                        raise HTTPException(status_code=400, detail="Invalid client_id")

                    # Return consent form HTML
                    consent_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>OAuth Consent</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 500px; margin: 50px auto; padding: 20px; }}
        .consent-card {{ background: white; border: 1px solid #e1e4e8; border-radius: 12px; padding: 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .app-info {{ background: #f6f8fa; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .permissions {{ margin: 20px 0; }}
        .permission-item {{ padding: 10px 0; border-bottom: 1px solid #e1e4e8; }}
        .buttons {{ display: flex; gap: 10px; margin-top: 30px; }}
        button {{ flex: 1; padding: 12px; border-radius: 8px; font-size: 16px; cursor: pointer; border: none; }}
        .approve {{ background: #28a745; color: white; }}
        .deny {{ background: #dc3545; color: white; }}
    </style>
</head>
<body>
    <div class="consent-card">
        <h2>Authorization Request</h2>
        <div class="app-info">
            <strong>{client.client_name or 'Application'}</strong> is requesting access to your account
        </div>
        <div class="permissions">
            <h3>Requested Permissions:</h3>
            {"<br>".join(f"<div class='permission-item'>✓ {s}</div>" for s in scope.split())}
        </div>
        <form method="post" action="/oauth/consent">
            <input type="hidden" name="client_id" value="{client_id}">
            <input type="hidden" name="scope" value="{scope}">
            <input type="hidden" name="consent_id" value="{params.get('consent_id', '')}">
            <input type="hidden" name="response_type" value="{params.get('response_type', 'code')}">
            <input type="hidden" name="redirect_uri" value="{params.get('redirect_uri', '')}">
            <input type="hidden" name="state" value="{params.get('state', '')}">
            <input type="hidden" name="code_challenge" value="{params.get('code_challenge', '')}">
            <input type="hidden" name="code_challenge_method" value="{params.get('code_challenge_method', 'S256')}">
            <div class="buttons">
                <button type="submit" name="consent" value="approve" class="approve">Approve</button>
                <button type="submit" name="consent" value="deny" class="deny">Deny</button>
            </div>
        </form>
    </div>
</body>
</html>"""
                    logger.info(f"✅ GET /oauth/consent rendered form cid={self._cid(request)}")
                    return HTMLResponse(content=consent_html)

                # For POST requests, process the consent decision
                else:
                    form = await request.form()
                    consent_decision = form.get("consent")
                    client_id = form.get("client_id")
                    scope = form.get("scope")
                    consent_id = form.get("consent_id")

                    if consent_decision == "approve":
                        # Store consent approval
                        logger.info(f"✅ POST /oauth/consent approved for {client_id} cid={self._cid(request)}")

                        # Get OAuth parameters from form
                        redirect_uri = form.get("redirect_uri", "")
                        state = form.get("state", "")
                        code_challenge = form.get("code_challenge", "")
                        code_challenge_method = form.get("code_challenge_method", "S256")

                        # Generate authorization code directly after consent
                        if self.oauth2_server and redirect_uri:
                            from urllib.parse import urlencode
                            # After device auth, user is authenticated as "device"
                            auth_code = self.oauth2_server._generate_authorization_code(
                                client_id=str(client_id),
                                user_id="device",  # Device-authenticated user
                                redirect_uri=str(redirect_uri),
                                scope=str(scope or "mcp:read"),
                                code_challenge=str(code_challenge) if code_challenge else None,
                                code_challenge_method=str(code_challenge_method) if code_challenge_method else None
                            )

                            # Build callback URL with authorization code
                            callback_params = {"code": auth_code.code}
                            if state:
                                callback_params["state"] = str(state)

                            callback_url = f"{redirect_uri}?{urlencode(callback_params)}"
                            logger.info(f"✅ OAuth flow complete, redirecting to {callback_url[:50]}... cid={self._cid(request)}")
                            return RedirectResponse(url=callback_url, status_code=303)
                        else:
                            # Fallback to old behavior
                            redirect_url = f"/oauth/authorize?consent=approved&consent_id={consent_id}&client_id={client_id}&scope={scope}"
                            return RedirectResponse(url=redirect_url, status_code=303)
                    else:
                        # Consent denied
                        logger.info(f"❌ POST /oauth/consent denied for {client_id} cid={self._cid(request)}")
                        return HTMLResponse(content="""<!DOCTYPE html>
<html>
<head>
    <title>Authorization Denied</title>
    <meta charset="utf-8">
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 500px; margin: 50px auto; padding: 20px; text-align: center; }
        .error-card { background: #ffe6e6; border: 1px solid #ff9999; border-radius: 12px; padding: 30px; }
    </style>
</head>
<body>
    <div class="error-card">
        <h2>Authorization Denied</h2>
        <p>You have denied the authorization request.</p>
    </div>
</body>
</html>""")

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"❌ /oauth/consent error: {e} cid={self._cid(request)}")
                raise HTTPException(status_code=500, detail=str(e))

        # Operator approval endpoints (local only)
        @self.app.post("/oauth/operator/approve")
        async def oauth_operator_approve(request: Request):
            """Approve pending OAuth authorization requests (local only)."""
            ip = request.client.host if request.client else ""
            if ip not in ("127.0.0.1", "::1", "localhost"):
                return HTMLResponse(content="Forbidden", status_code=403)
            if not self.oauth2_server:
                return HTMLResponse(content="OAuth server not initialized", status_code=503)
            try:
                body = await request.json()
                request_id = (body.get("request_id") or "").strip()
                operator_token = (body.get("operator_token") or "").strip()
                if not request_id or not operator_token:
                    return HTMLResponse(content="Bad Request", status_code=400)
                # The authorization_requests dict is in the OAuth integration module
                auth_req = getattr(self.oauth2_server, 'authorization_requests', {}).get(request_id)
                if not auth_req or not auth_req.get("operator_required"):
                    return HTMLResponse(content="Invalid request", status_code=400)
                if auth_req.get("operator_token") != operator_token:
                    return HTMLResponse(content="Invalid token", status_code=401)
                auth_req["operator_approved"] = True
                auth_req["operator_token"] = None
                # Update the authorization request
                if hasattr(self.oauth2_server, 'authorization_requests'):
                    self.oauth2_server.authorization_requests[request_id] = auth_req
                logger.info(f"OAUTH_APPROVED request_id={request_id}")
                return HTMLResponse(content="Approved")
            except Exception as e:
                return HTMLResponse(content=f"Error: {e}", status_code=500)

        @self.app.get("/oauth/operator/approve/{request_id}")
        async def oauth_operator_approve_page(request_id: str, request: Request, token: str = None):
            """Simple approval page with request ID in URL - just enter token or auto-approve with token param."""
            ip = request.client.host if request.client else ""
            if ip not in ("127.0.0.1", "::1", "localhost"):
                return HTMLResponse(content="Forbidden", status_code=403)

            if not self.oauth2_server:
                return HTMLResponse(content="OAuth server not initialized", status_code=503)

            # Check if this request exists
            auth_req = getattr(self.oauth2_server, 'authorization_requests', {}).get(request_id)
            if not auth_req or not auth_req.get("operator_required"):
                return HTMLResponse(content=f"""
                    <html><body>
                    <h2>Invalid Request</h2>
                    <p>Request ID not found or doesn't require approval: {request_id}</p>
                    </body></html>
                """, status_code=404)

            # If token is provided in URL, auto-approve
            if token:
                if auth_req.get("operator_token") == token:
                    auth_req["operator_approved"] = True
                    auth_req["operator_token"] = None
                    # Update the authorization request
                    if hasattr(self.oauth2_server, 'authorization_requests'):
                        self.oauth2_server.authorization_requests[request_id] = auth_req
                    logger.info(f"OAUTH_AUTO_APPROVED request_id={request_id}")
                    return HTMLResponse(content="""
                        <html>
                        <head>
                            <title>OAuth Approval Success</title>
                            <style>
                                body { font-family: -apple-system, system-ui, sans-serif; padding: 2rem; max-width: 600px; margin: 0 auto; }
                                h2 { color: #28a745; }
                                .info { background: #d4edda; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
                            </style>
                        </head>
                        <body>
                            <h2>✅ Auto-Approved!</h2>
                            <div class="info">
                                <p>The OAuth request has been automatically approved.</p>
                                <p>The user can now continue with authentication.</p>
                            </div>
                        </body>
                        </html>
                    """)
                else:
                    return HTMLResponse(content="""
                        <html>
                        <head>
                            <title>OAuth Approval Failed</title>
                            <style>
                                body { font-family: -apple-system, system-ui, sans-serif; padding: 2rem; max-width: 600px; margin: 0 auto; }
                                h2 { color: #dc3545; }
                                .error { background: #f8d7da; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
                            </style>
                        </head>
                        <body>
                            <h2>❌ Invalid Token</h2>
                            <div class="error">
                                <p>The provided token is invalid.</p>
                                <p><a href="javascript:history.back()">Go back</a> and try again.</p>
                            </div>
                        </body>
                        </html>
                    """, status_code=401)

            # Show simple form with just token input
            return HTMLResponse(content=f"""
                <html>
                <head>
                    <title>OAuth Operator Approval</title>
                    <style>
                        body {{ font-family: -apple-system, system-ui, sans-serif; padding: 2rem; max-width: 600px; margin: 0 auto; }}
                        h2 {{ color: #333; }}
                        .info {{ background: #f5f5f5; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
                        .form {{ margin: 2rem 0; }}
                        input {{ padding: 0.5rem; font-size: 16px; width: 300px; }}
                        button {{ padding: 0.5rem 1rem; font-size: 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }}
                        button:hover {{ background: #0056b3; }}
                        .error {{ color: red; margin: 1rem 0; }}
                    </style>
                </head>
                <body>
                    <h2>OAuth Operator Approval Required</h2>
                    <div class="info">
                        <p><strong>Request ID:</strong> {request_id}</p>
                        <p><strong>Client:</strong> {auth_req.get('client_name', 'Unknown')}</p>
                        <p><strong>User:</strong> {auth_req.get('subject', 'Unknown')}</p>
                    </div>
                    <div class="form">
                        <p>Enter the operator token to approve this request:</p>
                        <form method="POST" action="/oauth/operator/approve">
                            <input type="hidden" name="request_id" value="{request_id}">
                            <input type="text" name="operator_token" placeholder="Enter operator token" required autofocus>
                            <button type="submit">Approve</button>
                        </form>
                    </div>
                    <div id="error" class="error"></div>
                    <script>
                        document.querySelector('form').addEventListener('submit', async (e) => {{
                            e.preventDefault();
                            const formData = new FormData(e.target);
                            const response = await fetch('/oauth/operator/approve', {{
                                method: 'POST',
                                headers: {{ 'Content-Type': 'application/json' }},
                                body: JSON.stringify({{
                                    request_id: formData.get('request_id'),
                                    operator_token: formData.get('operator_token')
                                }})
                            }});
                            if (response.ok) {{
                                document.body.innerHTML = '<h2>✅ Approved!</h2><p>The OAuth request has been approved. The user can now continue with authentication.</p>';
                            }} else {{
                                document.getElementById('error').textContent = 'Invalid token or error approving request';
                            }}
                        }});
                    </script>
                </body>
                </html>
            """)


        @self.app.post("/oauth/register")
        async def oauth_register(request: Request):
            """Dynamic Client Registration endpoint.

            Available even if the OAuth 2.0 server hasn't been initialized,
            since DCR does not require the authorization server to be active.
            """
            try:
                from auth.oauth2_dcr import get_dcr_manager

                # Parse request body
                if self.server_timing_details:
                    self._timing_start(request, "dcr_parse")
                body = await request.json()
                if self.server_timing_details:
                    self._timing_end(request, "dcr_parse")
                ip = request.client.host if request.client else "unknown"
                ua = request.headers.get("user-agent")
                logger.info(f"➡️  POST /oauth/register from {ip} ua={ua} cid={self._cid(request)} body={self._mask_sensitive(body)}")

                # Build client metadata and register client via DCR manager
                from auth.oauth2_dcr import ClientMetadata
                dcr_manager = get_dcr_manager()

                # Build metadata dict, excluding None values to allow defaults
                metadata_kwargs = {}

                # Add optional fields only if present in request
                if "client_name" in body:
                    metadata_kwargs["client_name"] = body["client_name"]
                if "client_uri" in body:
                    metadata_kwargs["client_uri"] = body["client_uri"]
                if "redirect_uris" in body:
                    metadata_kwargs["redirect_uris"] = body["redirect_uris"]
                if "client_type" in body:
                    metadata_kwargs["client_type"] = body["client_type"]

                # OAuth parameters with defaults
                metadata_kwargs["token_endpoint_auth_method"] = body.get("token_endpoint_auth_method", "client_secret_basic")
                metadata_kwargs["grant_types"] = body.get("grant_types", ["authorization_code"])
                metadata_kwargs["response_types"] = body.get("response_types", ["code"])
                metadata_kwargs["scope"] = body.get("scope", "mcp mcp:read mcp:write")

                # Optional metadata fields
                optional_fields = [
                    "logo_uri", "tos_uri", "policy_uri", "software_id", "software_version",
                    "contacts", "jwks_uri", "jwks", "sector_identifier_uri", "subject_type",
                    "id_token_signed_response_alg", "mcp_capabilities",
                    "mcp_transport_protocols", "mcp_session_timeout"
                ]

                for field in optional_fields:
                    if field in body:
                        metadata_kwargs[field] = body[field]

                metadata = ClientMetadata(**metadata_kwargs)

                if self.server_timing_details:
                    self._timing_start(request, "dcr_store")
                registered_client = await dcr_manager.register_client(
                    metadata=metadata,
                    client_ip=request.client.host if request.client else "unknown",
                    user_agent=request.headers.get("User-Agent")
                )
                if self.server_timing_details:
                    self._timing_end(request, "dcr_store")

                logger.info(f"✅ /oauth/register success client_id={registered_client.client_id} cid={self._cid(request)}")

                # Mirror registration into the live OAuth2 server (if initialized)
                try:
                    if self.oauth2_server:
                        from auth.oauth2_server import GrantType, OAuthClient

                        # Determine public vs confidential client
                        token_auth = getattr(registered_client, "token_endpoint_auth_method", None) or "none"
                        is_public = (token_auth == "none")

                        # Collect redirect URIs and scopes
                        redirect_uris = list(getattr(registered_client, "redirect_uris", []) or [])
                        scope_str = getattr(registered_client, "scope", "mcp mcp:read mcp:write") or ""
                        allowed_scopes = set(scope_str.split()) if isinstance(scope_str, str) else set()

                        # Grant types
                        gts = set()
                        for gt in (getattr(registered_client, "grant_types", ["authorization_code"]) or ["authorization_code"]):
                            try:
                                if gt == "authorization_code":
                                    gts.add(GrantType.AUTHORIZATION_CODE)
                                elif gt == "refresh_token":
                                    gts.add(GrantType.REFRESH_TOKEN)
                            except Exception:
                                continue

                        # Build and register client if not already present
                        if registered_client.client_id not in self.oauth2_server.clients:
                            oauth_client = OAuthClient(
                                client_id=registered_client.client_id,
                                client_secret=None,
                                redirect_uris=redirect_uris,
                                allowed_grant_types=gts or {GrantType.AUTHORIZATION_CODE},
                                allowed_scopes=allowed_scopes or {"mcp:read", "mcp:write"},
                                is_public=is_public,
                                name=(getattr(registered_client, "client_name", None) or "MCP Client")
                            )
                            self.oauth2_server.register_client(oauth_client)
                            logger.info(f"🔗 DCR client mirrored into OAuth2 server: {registered_client.client_id}")
                        else:
                            # Optionally update existing client redirect URIs and scopes
                            try:
                                oc = self.oauth2_server.clients[registered_client.client_id]
                                oc.redirect_uris = redirect_uris or oc.redirect_uris
                                oc.allowed_scopes = allowed_scopes or oc.allowed_scopes
                                oc.allowed_grant_types = gts or oc.allowed_grant_types
                                oc.is_public = is_public
                            except Exception:
                                pass
                except Exception as mirror_err:
                    logger.warning(f"Could not mirror DCR client into OAuth2 server: {mirror_err}")

                # Use model_dump with mode='json' to handle datetime serialization, exclude None values
                return JSONResponse(content=registered_client.model_dump(mode='json', exclude_none=True), status_code=201)

            except ImportError:
                raise HTTPException(status_code=501, detail="Dynamic Client Registration not available")
            except Exception as e:
                logger.error(f"❌ /oauth/register failed: {e} cid={self._cid(request)}")
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/oauth/register/{client_id}")
        async def oauth_get_client(client_id: str, request: Request):
            """Get OAuth client configuration (independent of OAuth2 server init)."""
            try:
                from auth.oauth2_dcr import get_dcr_manager

                dcr_manager = get_dcr_manager()
                if self.server_timing_details:
                    self._timing_start(request, "dcr_lookup")
                client = await dcr_manager.get_client(client_id)
                if self.server_timing_details:
                    self._timing_end(request, "dcr_lookup")

                if not client:
                    raise HTTPException(status_code=404, detail="Client not found")

                logger.info(f"✅ GET /oauth/register/{client_id} fetched cid={self._cid(request)}")

                # Mirror into OAuth2 server if missing
                try:
                    if self.oauth2_server and client_id not in self.oauth2_server.clients:
                        from auth.oauth2_server import GrantType, OAuthClient
                        token_auth = getattr(client, "token_endpoint_auth_method", None) or "none"
                        is_public = (token_auth == "none")
                        redirect_uris = list(getattr(client, "redirect_uris", []) or [])
                        scope_str = getattr(client, "scope", "mcp mcp:read mcp:write") or ""
                        allowed_scopes = set(scope_str.split()) if isinstance(scope_str, str) else {"mcp:read", "mcp:write"}
                        gts = set()
                        for gt in (getattr(client, "grant_types", ["authorization_code"]) or ["authorization_code"]):
                            if gt == "authorization_code":
                                gts.add(GrantType.AUTHORIZATION_CODE)
                            elif gt == "refresh_token":
                                gts.add(GrantType.REFRESH_TOKEN)
                        oauth_client = OAuthClient(
                            client_id=client_id,
                            client_secret=None,
                            redirect_uris=redirect_uris,
                            allowed_grant_types=gts or {GrantType.AUTHORIZATION_CODE},
                            allowed_scopes=allowed_scopes or {"mcp:read", "mcp:write"},
                            is_public=is_public,
                            name=(getattr(client, "client_name", None) or "MCP Client")
                        )
                        self.oauth2_server.register_client(oauth_client)
                        logger.info(f"🔗 Mirrored existing DCR client into OAuth2 server: {client_id}")
                except Exception as mirror_err:
                    logger.warning(f"Could not mirror DCR client on GET: {mirror_err}")

                # Use model_dump with mode='json' to handle datetime serialization, exclude None values
                return JSONResponse(content=client.model_dump(mode='json', exclude_none=True))
            except ImportError:
                raise HTTPException(status_code=501, detail="Dynamic Client Registration not available")
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"❌ GET /oauth/register/{client_id} failed: {e} cid={self._cid(request)}")
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.delete("/oauth/register/{client_id}")
        async def oauth_delete_client(client_id: str, request: Request):
            """Delete OAuth client registration (independent of OAuth2 server init)."""
            try:
                from auth.oauth2_dcr import get_dcr_manager

                dcr_manager = get_dcr_manager()

                # Extract registration token from Authorization header (Bearer token)
                auth_header = request.headers.get("Authorization")
                if not auth_header or not auth_header.startswith("Bearer "):
                    raise HTTPException(status_code=401, detail="Missing registration access token")
                registration_token = auth_header.split(" ", 1)[1]
                ip = request.client.host if request.client else "unknown"
                ua = request.headers.get("user-agent")
                logger.info(f"➡️  DELETE /oauth/register/{client_id} from {ip} ua={ua} cid={self._cid(request)}")

                if self.server_timing_details:
                    self._timing_start(request, "dcr_delete")
                success = await dcr_manager.delete_client(client_id, registration_token, request.client.host if request.client else "unknown", request.headers.get("User-Agent"))
                if self.server_timing_details:
                    self._timing_end(request, "dcr_delete")

                if not success:
                    raise HTTPException(status_code=404, detail="Client not found")

                logger.info(f"✅ DELETE /oauth/register/{client_id} deleted cid={self._cid(request)}")
                return JSONResponse(content={"message": "Client deleted successfully"})

            except ImportError:
                raise HTTPException(status_code=501, detail="Dynamic Client Registration not available")
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"❌ DELETE /oauth/register/{client_id} failed: {e}")
                raise HTTPException(status_code=400, detail=str(e))

    def setup_endpoints(self):
        """Setup HTTP endpoints for MCP protocol."""

        @self.app.get("/")
        async def root():
            """Root endpoint with server information."""
            return {
                "name": "Zen MCP Streamable HTTP Server",
                "version": "1.0.0",
                "protocol_version": "2025-03-26",
                "capabilities": {
                    "tools": True,
                    "resources": True,
                    "prompts": True,
                    "logging": True,
                    "streaming": True
                },
                "endpoints": {
                    "mcp": "/mcp",
                    "status": "/status",
                    "sessions": "/sessions",
                    "stream": "/stream/{task_id}",
                    "websocket": "/ws/{task_id}",
                    "enhance_input": "/enhance-input",
                    "dashboard": "/dashboard",
                    "docs": "/docs"
                },
                "transport": "streamable_http",
                "authentication": "oauth2" if self.auth_enabled else "none"
            }

        # OAuth discovery endpoints for clients probing capabilities
        @self.app.get("/.well-known/oauth-authorization-server")
        async def oauth_authorization_server_metadata(request: Request):
            """Expose minimal OAuth Authorization Server metadata.

            Note: This server primarily uses device-based auth; these endpoints
            are provided for discovery to avoid 404s and to advertise where
            auth-related flows live when enabled.
            """
            # Prefer OAuth server issuer when available to ensure consistency; else tunnel or local
            base = (
                getattr(self.oauth2_server, "issuer", None)
                or self.tunnel_url
                or f"http://{('localhost' if self.host in ('0.0.0.0', '::') else self.host)}:{self.allocated_port or self.port}"
            )

            # Debug OAuth metadata generation
            ip = request.client.host if request.client else "unknown"
            ua = request.headers.get("user-agent")
            logger.info(f"🔍 GET /.well-known/oauth-authorization-server from {ip} ua={ua} base={base} cid={self._cid(request)}")

            # Always include endpoints when auth is enabled to satisfy strict clients
            if self.auth_enabled:
                payload: dict[str, Any] = {
                    "issuer": base,
                    "authorization_endpoint": f"{base}/oauth/authorize",
                    "token_endpoint": f"{base}/oauth/token",
                    "revocation_endpoint": f"{base}/oauth/revoke",
                    "introspection_endpoint": f"{base}/oauth/introspect",
                    "registration_endpoint": f"{base}/oauth/register",
                    "response_types_supported": ["code"],
                    "grant_types_supported": ["authorization_code", "refresh_token"],
                    "response_modes_supported": ["query", "form_post"],
                    "scopes_supported": ["mcp", "mcp:read", "mcp:write", "mcp:admin", "profile"],
                    "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post", "none"],
                    "code_challenge_methods_supported": ["S256"],
                    "subject_types_supported": ["public"],
                    "id_token_signing_alg_values_supported": ["RS256", "HS256"],
                    "token_endpoint_auth_signing_alg_values_supported": ["RS256", "HS256"],
                    "revocation_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
                    "introspection_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
                }
            else:
                payload = {"issuer": base}

            logger.info("✅ Served oauth-authorization-server metadata")
            logger.info(f"✅ Served oauth-authorization-server metadata cid={self._cid(request)}")
            return JSONResponse(payload)

        # Minimal dynamic client registration stub
        @self.app.post("/register")
        async def dynamic_client_registration(request: Request):
            """Accept dynamic client registration requests and return a public client.

            This stub is provided to satisfy clients that attempt RFC 7591 dynamic
            registration. It issues a non-secret public client and does not persist state.
            """
            try:
                body = await request.json()
            except Exception:
                body = {}
            ip = request.client.host if request.client else "unknown"
            ua = request.headers.get("user-agent")
            logger.info(f"➡️  POST /register (stub) from {ip} ua={ua} body={self._mask_sensitive(body)}")

            issued_at = int(datetime.now(timezone.utc).timestamp())
            client_id = body.get("client_name") or "zen-mcp-public-client"
            redirect_uris = body.get("redirect_uris") or []

            response = {
                "client_id": client_id,
                "client_id_issued_at": issued_at,
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "redirect_uris": redirect_uris,
            }

            logger.info(f"✅ /register (stub) issued client_id={client_id}")
            return JSONResponse(response, status_code=200)

        # MCP-Required: OAuth 2.0 Protected Resource Metadata (RFC 9728)
        @self.app.get("/.well-known/oauth-protected-resource")
        async def oauth_protected_resource_metadata(request: Request):
            """OAuth 2.0 Protected Resource Metadata as required by MCP specification.

            Per MCP specification: 'MCP servers MUST implement OAuth 2.0 Protected
            Resource Metadata (RFC9728) to indicate the locations of authorization servers.'
            """
            # Prefer OAuth server issuer when available to ensure consistency; else tunnel or local
            base = (
                getattr(self.oauth2_server, "issuer", None)
                or self.tunnel_url
                or f"http://{('localhost' if self.host in ('0.0.0.0', '::') else self.host)}:{self.allocated_port or self.port}"
            )
            ip = request.client.host if request.client else "unknown"
            ua = request.headers.get("user-agent")
            logger.info(f"🔍 GET /.well-known/oauth-protected-resource from {ip} ua={ua} base={base} cid={self._cid(request)}")

            payload = {
                "resource": base,
                # Per RFC 9728, this should list AS issuer URIs, not the metadata URL
                "authorization_servers": [
                    base
                ],
                "resource_scopes_supported": [
                    "mcp",
                    "mcp:read",
                    "mcp:write",
                    "mcp:admin",
                    "profile"
                ],
                "bearer_methods_supported": [
                    "header"
                ],
                "resource_documentation": f"{base}/.well-known/oauth-authorization-server"
            }

            logger.info(f"✅ Served oauth-protected-resource metadata cid={self._cid(request)}")
            return JSONResponse(payload)

        # Tolerate clients that append a suffix to the well-known path (e.g., /.well-known/oauth-protected-resource/mcp)
        @self.app.get("/.well-known/oauth-protected-resource/{suffix:path}")
        async def oauth_protected_resource_metadata_with_suffix(suffix: str, request: Request):
            # Use same metadata but reflect the specific resource identifier in the 'resource' field
            base = (
                getattr(self.oauth2_server, "issuer", None)
                or self.tunnel_url
                or f"http://{('localhost' if self.host in ('0.0.0.0', '::') else self.host)}:{self.allocated_port or self.port}"
            )
            ip = request.client.host if request.client else "unknown"
            ua = request.headers.get("user-agent")
            logger.info(f"🔍 GET /.well-known/oauth-protected-resource/{suffix} from {ip} ua={ua} base={base} cid={self._cid(request)}")
            # Normalize resource to base + "/" + suffix without leading slash duplication
            resource = f"{base}/{suffix.lstrip('/')}" if suffix else base
            payload = {
                "resource": resource,
                "authorization_servers": [base],
                "resource_scopes_supported": [
                    "mcp",
                    "mcp:read",
                    "mcp:write",
                    "mcp:admin",
                    "profile"
                ],
                "bearer_methods_supported": ["header"],
                "resource_documentation": f"{base}/.well-known/oauth-authorization-server"
            }
            logger.info(f"✅ Served oauth-protected-resource metadata (suffix variant) cid={self._cid(request)}")
            return JSONResponse(payload)

        # Alias handler for clients that mistakenly duplicate the path
        @self.app.get("/.well-known/oauth-authorization-server/.well-known/oauth-authorization-server")
        async def oauth_authorization_server_metadata_alias(request: Request):
            logger.info(f"🔁 Alias hit for oauth-authorization-server metadata cid={self._cid(request)}")
            return await oauth_authorization_server_metadata(request)

        # Accept suffix variants like /.well-known/oauth-authorization-server/mcp
        @self.app.get("/.well-known/oauth-authorization-server/{suffix:path}")
        async def oauth_authorization_server_metadata_with_suffix(suffix: str, request: Request):
            return await oauth_authorization_server_metadata(request)

        @self.app.get("/status")
        async def status():
            """Server status endpoint."""
            tunnel_info = {}
            if self.tunnel_manager:
                tunnel_status = self.tunnel_manager.get_status()
                tunnel_info = {
                    "tunnel_enabled": True,
                    "tunnel_status": tunnel_status.value,
                    "tunnel_url": self.tunnel_url,
                    "tunnel_domain": self.tunnel_domain
                }

            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "active_sessions": len(self.sessions),
                "uptime_seconds": 0,  # TODO: Track actual uptime
                "host": self.host,
                "port": self.allocated_port or self.port,
                "local_url": f"http://{self.host}:{self.allocated_port or self.port}",
                **tunnel_info
            }

        @self.app.get("/status/cors")
        async def status_cors(request: Request):
            """CORS debug endpoint to inspect allowed origins and the caller's Origin header."""
            origin = request.headers.get("origin")
            cors_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "*")
            origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
            allow_credentials = True
            if allow_credentials and origins == ["*"]:
                origins = [
                    f"http://{('localhost' if self.host in ('0.0.0.0', '::') else self.host)}:{self.allocated_port or self.port}",
                    "http://localhost:6274",
                    "http://127.0.0.1:6274",
                ]
            return JSONResponse({
                "origin_header": origin,
                "allow_origins": origins,
                "allow_credentials": allow_credentials,
                "note": "Set CORS_ALLOW_ORIGINS in .env to include your UI origin(s).",
            })

        @self.app.get("/healthz")
        async def health_check():
            """Health check endpoint for KInfra monitoring."""
            return {
                "status": "ok",
                "service": "zen-mcp-server",
                "device_auth_enabled": self.auth_enabled,
                "tunnel_url": self.tunnel_url
            }

        @self.app.get("/health/kafka")
        async def kafka_health_check():
            """Kafka connectivity health check.

            - Attempts to (re)connect to Kafka if not connected
            - Optionally publishes a tiny test event to verify end-to-end
            """
            try:
                # Lazy import to avoid hard dependency when Kafka isn't used
                from utils.kafka_events import (
                    KAFKA_AVAILABLE,
                    AgentEvent,
                    EventMetadata,
                    EventType,
                    get_event_publisher,
                )

                publisher = await get_event_publisher()
                # If previous connect failed, try again quickly
                if not getattr(publisher, "_is_connected", False):
                    await publisher.connect()

                connected = bool(getattr(publisher, "_is_connected", False))
                publish_ok = False

                # Attempt a lightweight publish when connected
                if connected:
                    evt = AgentEvent(
                        event_type=EventType.PERFORMANCE_METRIC,
                        aggregate_id="kafka-health",
                        aggregate_type="system",
                        payload={"check": "kafka-health", "timestamp": datetime.now(timezone.utc).isoformat()},
                        metadata=EventMetadata()
                    )
                    publish_ok = await publisher.publish_event(evt, topic="system-events")

                return JSONResponse({
                    "kafka_available": bool(KAFKA_AVAILABLE),
                    "bootstrap_servers": getattr(publisher, "bootstrap_servers", None),
                    "connected": connected,
                    "publish_ok": publish_ok
                })
            except Exception as e:
                logger.error(f"Kafka health check error: {e}")
                return JSONResponse({
                    "kafka_available": False,
                    "connected": False,
                    "publish_ok": False,
                    "error": str(e)
                }, status_code=500)


        @self.app.get("/sessions")
        async def list_sessions():
            """List active sessions."""
            return {
                "active_sessions": len(self.sessions),
                "sessions": [
                    {
                        "session_id": sid,
                        "created_at": session.get("created_at"),
                        "last_activity": session.get("last_activity"),
                        "client_info": session.get("client_info", {})
                    }
                    for sid, session in self.sessions.items()
                ]
            }

        # Lightweight HEAD/OPTIONS handlers for clients probing endpoint capability
        @self.app.head("/mcp")
        async def mcp_head(request: Request):
            base = (
                getattr(self.oauth2_server, "issuer", None)
                or self.tunnel_url
                or f"http://{('localhost' if self.host in ('0.0.0.0', '::') else self.host)}:{self.allocated_port or self.port}"
            )
            headers = {
                "Allow": "GET,POST,DELETE,OPTIONS,HEAD",
                "WWW-Authenticate": f"Bearer realm=\"MCP API\", authorization_uri=\"{base}/.well-known/oauth-authorization-server\"",
            }
            return Response(status_code=200, headers=headers)

        @self.app.options("/mcp")
        async def mcp_options():
            headers = {
                "Allow": "GET,POST,DELETE,OPTIONS,HEAD",
                "Access-Control-Allow-Methods": "GET,POST,DELETE,OPTIONS,HEAD",
                "Access-Control-Allow-Headers": "*",
            }
            return Response(status_code=204, headers=headers)

        # MCP Streamable HTTP endpoint (POST and GET support)
        @self.app.post("/mcp")
        @self.app.get("/mcp")
        @self.app.delete("/mcp")
        @self.app.post("/api/mcp")
        @self.app.get("/api/mcp")
        @self.app.delete("/api/mcp")
        async def mcp_endpoint(
            request: Request,
            mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id")
        ):
            """MCP Streamable HTTP protocol endpoint with OAuth 2.0 Bearer token authentication."""

            # Enforce OAuth 2.0 Bearer token authentication for MCP endpoint
            if self.auth_enabled and self.oauth2_server:
                # Extract Bearer token from Authorization header
                auth_header = request.headers.get("Authorization", "")
                if not auth_header.startswith("Bearer "):
                    base = (
                        getattr(self.oauth2_server, "issuer", None)
                        or self.tunnel_url
                        or f"http://{('localhost' if self.host in ('0.0.0.0', '::') else self.host)}:{self.allocated_port or self.port}"
                    )
                    raise HTTPException(
                        status_code=401,
                        detail="Authentication required - OAuth 2.0 Bearer token required",
                        headers={"WWW-Authenticate": f"Bearer realm=\"MCP API\", authorization_uri=\"{base}/.well-known/oauth-authorization-server\""}
                    )

                access_token = auth_header[7:]  # Remove "Bearer " prefix

                try:
                    # Validate the access token using OAuth 2.0 server
                    if self.server_timing_details:
                        self._timing_start(request, "auth_validation")
                    token_info = await self.oauth2_server.validate_bearer_token(f"Bearer {access_token}")
                    if self.server_timing_details:
                        self._timing_end(request, "auth_validation")

                    # Treat non-None as active for unified OAuth2Server which
                    # returns a dict without an 'active' flag. If an 'active'
                    # key is present (e.g., from integration/introspection), honor it.
                    if not token_info or ("active" in token_info and not token_info.get("active")):
                        base = (
                            getattr(self.oauth2_server, "issuer", None)
                            or self.tunnel_url
                            or f"http://{('localhost' if self.host in ('0.0.0.0', '::') else self.host)}:{self.allocated_port or self.port}"
                        )
                        raise HTTPException(
                            status_code=401,
                            detail="Invalid or expired access token",
                            headers={"WWW-Authenticate": f"Bearer realm=\"MCP API\", authorization_uri=\"{base}/.well-known/oauth-authorization-server\""}
                        )

                    # Check required scopes for MCP access
                    scopes = token_info.get("scope", "").split()
                    # Accept either "mcp"/"mcp:read" (full OAuth server) or
                    # StandardScopes like "read"/"tools" (integration server)
                    allowed_read_scopes = {"mcp", "mcp:read", "read", "tools"}
                    if not (allowed_read_scopes & set(scopes)):
                        base = (
                            getattr(self.oauth2_server, "issuer", None)
                            or self.tunnel_url
                            or f"http://{('localhost' if self.host in ('0.0.0.0', '::') else self.host)}:{self.allocated_port or self.port}"
                        )
                        raise HTTPException(
                            status_code=403,
                            detail="Insufficient scope - require one of: mcp, mcp:read, read, tools",
                            headers={"WWW-Authenticate": f"Bearer realm=\"MCP API\" scope=\"mcp read tools\", authorization_uri=\"{base}/.well-known/oauth-authorization-server\""}
                        )

                    # Add authenticated user context for audit logging
                    request.state.authenticated_user = token_info.get("sub") or token_info.get("user_id") or "unknown"
                    request.state.client_id = token_info.get("client_id", "unknown")
                    request.state.scopes = scopes
                    logger.info(f"MCP request authenticated: user={token_info.get('sub')}, client={token_info.get('client_id')}, scopes={scopes}")

                except Exception as e:
                    logger.warning(f"Token validation failed: {e}")
                    base = (
                        getattr(self.oauth2_server, "issuer", None)
                        or self.tunnel_url
                        or f"http://{('localhost' if self.host in ('0.0.0.0', '::') else self.host)}:{self.allocated_port or self.port}"
                    )
                    raise HTTPException(
                        status_code=401,
                        detail="Token validation failed",
                        headers={"WWW-Authenticate": f"Bearer realm=\"MCP API\", authorization_uri=\"{base}/.well-known/oauth-authorization-server\""}
                    )

            elif self.auth_enabled:
                # Auth is enabled but not properly configured
                raise HTTPException(
                    status_code=503,
                    detail="OAuth 2.0 authentication service unavailable"
                )

            # Handle GET request for connection test
            if request.method == "GET":
                return JSONResponse({
                    "message": "Zen MCP Streamable HTTP endpoint active",
                    "protocol_version": "2025-03-26",
                    "methods": ["GET", "POST"],
                    "session_id": mcp_session_id
                })

            # Handle DELETE request for session cleanup (best-effort)
            if request.method == "DELETE":
                sid = mcp_session_id
                if sid and sid in self.sessions:
                    try:
                        del self.sessions[sid]
                    except Exception:
                        pass
                return JSONResponse({"status": "ok"}, status_code=200)

            # Handle POST request for MCP messages
            try:
                # Parse JSON-RPC request
                if request.headers.get("content-type") == "application/json":
                    body = await request.json()
                else:
                    body = await request.body()
                    if isinstance(body, bytes):
                        body = json.loads(body.decode("utf-8"))

                # Validate JSON-RPC structure
                if not isinstance(body, dict) or "jsonrpc" not in body:
                    raise HTTPException(status_code=400, detail="Invalid JSON-RPC request")

                # Handle session management
                session_id = mcp_session_id
                if body.get("method") == "initialize":
                    session_id = self.create_session(body)
                    # Persist OAuth scopes into session for per-tool enforcement
                    try:
                        if hasattr(request.state, "scopes"):
                            self.sessions[session_id]["scopes"] = list(getattr(request.state, "scopes", []) or [])
                    except Exception:
                        pass
                elif session_id and session_id not in self.sessions:
                    raise HTTPException(status_code=400, detail="Invalid session ID")

                # Process MCP request
                response = await self.handle_mcp_request(body, session_id)

                # Create HTTP response
                headers = {}
                if session_id and body.get("method") == "initialize":
                    headers["Mcp-Session-Id"] = session_id

                return JSONResponse(response, headers=headers)

            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON")
            except Exception as e:
                logger.error(f"MCP endpoint error: {e}")
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,  # Internal error
                            "message": str(e)
                        },
                        "id": body.get("id") if 'body' in locals() else None
                    },
                    status_code=500
                )

        # SSE Streaming endpoint for real-time updates
        @self.app.get("/stream/{task_id}")
        async def sse_stream(task_id: str):
            """Server-Sent Events stream for real-time task updates.

            Uses standard SSE content type and recommends disabling proxy buffering
            for low-latency delivery.
            """
            return StreamingResponse(
                create_sse_stream(task_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Vary": "Accept-Encoding",
                    "X-Accel-Buffering": "no",
                }
            )

        # Global SSE stream for all task events (parity with server_http)
        @self.app.get("/events/live")
        async def sse_events_live():
            sm = self.streaming_manager

            async def generator():
                conn_id = str(uuid.uuid4())
                queue = await sm.register_sse_connection(conn_id, None)
                try:
                    # Initial hello event
                    hello = {
                        "id": str(uuid.uuid4()),
                        "event": "status_update",
                        "data": {
                            "status": "connected",
                            "message": "Global stream started"
                        }
                    }
                    yield f"id: {hello['id']}\nevent: {hello['event']}\ndata: {json.dumps(hello['data'])}\n\n"

                    while True:
                        try:
                            msg = await asyncio.wait_for(queue.get(), timeout=10.0)
                            # msg is a StreamMessage; reuse SSE serialization
                            yield msg.to_sse_format()
                        except asyncio.TimeoutError:
                            # heartbeat pulse
                            pulse = {"status": "alive"}
                            yield f"event: pulse\ndata: {json.dumps(pulse)}\n\n"
                finally:
                    try:
                        await sm.unregister_connection(conn_id)
                    except Exception:
                        pass

            return StreamingResponse(
                generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Vary": "Accept-Encoding",
                    "X-Accel-Buffering": "no",
                }
            )

        # Input enhancement endpoint
        @self.app.post("/enhance-input")
        async def enhance_input(request: Request):
            """Transform user input with intent recognition and structured prompting."""
            try:
                body = await request.json()
                user_input = body.get("input", "")
                context = body.get("context", {})
                agent_type = body.get("agent_type", "claude")

                # Transform input
                enhanced_input = self.input_transformer.transform_input(user_input, context)

                # Add agent-specific protocol
                agent_enum = getattr(AgentType, agent_type.upper(), AgentType.CLAUDE)
                enhanced_message = enhance_agent_message(user_input, agent_enum)

                return {
                    "original_input": user_input,
                    "enhanced_input": enhanced_input,
                    "enhanced_message": enhanced_message,
                    "agent_type": agent_type
                }

            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        # Dashboard endpoint
        @self.app.get("/dashboard")
        async def dashboard():
            """Serve the progress dashboard."""
            return {
                "message": "Dashboard endpoint - static files would be served here",
                "streaming_stats": self.streaming_manager.get_connection_stats(),
                "active_sessions": len(self.sessions)
            }

        # WebSocket endpoint for bidirectional real-time communication
        @self.app.websocket("/ws/{task_id}")
        async def websocket_endpoint(websocket: WebSocket, task_id: str):
            """WebSocket endpoint for real-time bidirectional communication."""
            await websocket.accept()
            connection_id = str(uuid.uuid4())

            try:
                # Register WebSocket connection with streaming manager
                await self.streaming_manager.register_websocket_connection(
                    connection_id, websocket, [task_id]
                )

                # Send initial connection message
                await websocket.send_json({
                    "type": "connection",
                    "connection_id": connection_id,
                    "task_id": task_id,
                    "message": "WebSocket connected successfully",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                # Handle incoming messages
                while True:
                    try:
                        data = await websocket.receive_json()
                        message_type = data.get("type")

                        if message_type == "ping":
                            await websocket.send_json({"type": "pong"})

                        elif message_type == "subscribe":
                            # Subscribe to additional task
                            new_task_id = data.get("task_id")
                            if new_task_id:
                                if new_task_id not in self.streaming_manager.task_subscribers:
                                    self.streaming_manager.task_subscribers[new_task_id] = set()
                                self.streaming_manager.task_subscribers[new_task_id].add(connection_id)

                                await websocket.send_json({
                                    "type": "subscription_confirmed",
                                    "task_id": new_task_id
                                })

                        elif message_type == "execute_tool":
                            # Execute tool request via WebSocket
                            tool_name = data.get("tool_name")
                            arguments = data.get("arguments", {})

                            if tool_name:
                                try:
                                    # Create a temporary request structure
                                    temp_params = {"name": tool_name, "arguments": arguments}
                                    result = await self.handle_tools_call(temp_params, connection_id)

                                    await websocket.send_json({
                                        "type": "tool_result",
                                        "tool_name": tool_name,
                                        "result": result,
                                        "task_id": task_id
                                    })
                                except Exception as e:
                                    await websocket.send_json({
                                        "type": "error",
                                        "error": str(e),
                                        "tool_name": tool_name
                                    })

                    except json.JSONDecodeError:
                        await websocket.send_json({
                            "type": "error",
                            "error": "Invalid JSON message"
                        })

            except WebSocketDisconnect:
                logger.info(f"WebSocket {connection_id} disconnected")
            except Exception as e:
                logger.error(f"WebSocket error for {connection_id}: {e}")
            finally:
                # Unregister connection
                await self.streaming_manager.unregister_connection(connection_id)

    def create_session(self, init_request: dict[str, Any]) -> str:
        """Create a new MCP session."""
        session_id = str(uuid.uuid4())

        self.sessions[session_id] = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_activity": datetime.now(timezone.utc).isoformat(),
            "client_info": init_request.get("params", {}).get("clientInfo", {}),
            "capabilities": init_request.get("params", {}).get("capabilities", {}),
            "protocol_version": init_request.get("params", {}).get("protocolVersion", "2025-03-26")
        }

        logger.info(f"Created MCP session: {session_id}")
        return session_id

    async def handle_mcp_request(self, request: dict[str, Any], session_id: Optional[str] = None) -> dict[str, Any]:
        """Handle MCP JSON-RPC request."""

        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        # Update session activity
        if session_id and session_id in self.sessions:
            self.sessions[session_id]["last_activity"] = datetime.now(timezone.utc).isoformat()

        try:
            # Handle different MCP methods
            if method == "initialize":
                result = await self.handle_initialize(params)
            elif method == "notifications/initialized":
                result = None  # No response needed for notification
            elif method == "tools/list":
                result = await self.handle_tools_list()
            elif method == "tools/call":
                result = await self.handle_tools_call(params, session_id)
            elif method == "resources/list":
                result = await self.handle_resources_list()
            elif method == "resources/read":
                result = await self.handle_resources_read(params)
            elif method == "prompts/list":
                result = await self.handle_prompts_list()
            elif method == "prompts/get":
                result = await self.handle_prompts_get(params)
            elif method == "completion/complete":
                result = await self.handle_completion(params)
            elif method == "logging/setLevel":
                result = await self.handle_logging_set_level(params)
            else:
                # Method not supported
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,  # Method not found
                        "message": f"Method not found: {method}"
                    },
                    "id": request_id
                }

            # Create successful response
            if method.startswith("notifications/"):
                return {}  # No response for notifications
            else:
                return {
                    "jsonrpc": "2.0",
                    "result": result,
                    "id": request_id
                }

        except Exception as e:
            logger.error(f"Error handling MCP method {method}: {e}")
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,  # Internal error
                    "message": str(e)
                },
                "id": request_id
            }

    async def handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle MCP initialize request."""
        # Log client info if provided
        try:
            ci = params.get("clientInfo", {}) if isinstance(params, dict) else {}
            cname = ci.get("name")
            cver = ci.get("version")
            if cname:
                logger.info(f"MCP HTTP initialize from {cname} v{cver or 'unknown'}")
                logging.getLogger("mcp_activity").info(f"MCP_CLIENT_INFO: {cname} v{cver or 'unknown'}")
        except Exception:
            pass

        return {
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {},
                "logging": {},
                "experimental": {
                    "streaming": True
                }
            },
            "serverInfo": {
                "name": "ZenMCP",
                "version": "1.0.0"
            }
        }

    async def handle_tools_list(self) -> dict[str, Any]:
        """Handle tools/list request."""
        # Get all available Zen tools
        zen_tools = get_all_tools()

        tools = []
        for tool_name, tool_info in zen_tools.items():
            # Convert to MCP tool schema
            # Inject required work_dir field into input schema
            _schema = tool_info.get("input_schema", {}) or {"type": "object", "properties": {}, "required": []}
            _props = dict(_schema.get("properties", {}))
            _req = list(_schema.get("required", []))
            if "work_dir" not in _props:
                _props["work_dir"] = {
                    "type": "string",
                    "description": "REQUIRED: Repo-relative working directory that scopes data access"
                }
            if "work_dir" not in _req:
                _req.append("work_dir")
            _schema["properties"] = _props
            _schema["required"] = _req

            tools.append({
                "name": tool_name,
                "description": tool_info.get("description", f"{tool_name} tool"),
                "inputSchema": _schema,
                "annotations": tool_info.get("annotations") or {}
            })

        # Add some demo tools for testing
        tools.extend([
            {
                "name": "echo",
                "description": "Echo back the input text",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to echo"}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "get_time",
                "description": "Get current UTC time",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "multiply",
                "description": "Multiply two numbers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"}
                    },
                    "required": ["a", "b"]
                }
            }
        ])

        return {"tools": tools}

    async def handle_tools_call(self, params: dict[str, Any], session_id: Optional[str] = None) -> dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        # Determine OAuth scopes for this session (if any)
        session_scopes = set()
        if session_id and session_id in self.sessions:
            try:
                session_scopes = set(self.sessions.get(session_id, {}).get("scopes", []) or [])
            except Exception:
                session_scopes = set()

        def _has_read_scope(scopes: set[str]) -> bool:
            allowed = {"mcp", "mcp:read", "read", "tools"}
            return bool(allowed & scopes)

        def _has_write_scope(scopes: set[str]) -> bool:
            allowed = {"mcp", "mcp:write", "write"}
            return bool(allowed & scopes)

        # Enforce fine-grained scopes for specific tools
        try:
            if tool_name in ("messaging", "project", "a2a"):
                action = (arguments.get("action") or "").lower()
                need_write = False
                if tool_name == "messaging":
                    need_write = action in ("channel_post", "dm_post", "resume")
                elif tool_name == "project":
                    need_write = action in ("create", "add_artifact")
                elif tool_name == "a2a":
                    need_write = action in ("advertise", "message")

                if need_write:
                    if not _has_write_scope(session_scopes):
                        return {"content": [{"type": "text", "text": "Insufficient scope: write required"}], "isError": True}
                else:
                    if not _has_read_scope(session_scopes):
                        return {"content": [{"type": "text", "text": "Insufficient scope: read required"}], "isError": True}
        except Exception:
            pass

        # Send streaming update if we have a session
        if session_id:
            await self.streaming_manager.broadcast_message(
                f"session-{session_id}",
                StreamMessageType.ACTIVITY_UPDATE,
                {"content": f"Calling tool: {tool_name}"},
                "mcp-server"
            )

            # Enforce work_dir presence and validity; inject scope_context (warn-only by default)
            import os as _os

            from utils.scope_utils import (
                ScopeContext,
                WorkDirError,
                create_default_scope_context,
                get_repo_root,
                inject_scope_context,
                validate_and_normalize_work_dir,
            )

            _enforce = _os.getenv("ZEN_ENFORCE_WORKDIR", "0").lower() in ("1", "true", "yes")
            _work_dir = arguments.get("work_dir")

            # Create or extract scope context
            if "_scope_context" in arguments and isinstance(arguments["_scope_context"], ScopeContext):
                scope_context = arguments["_scope_context"]
            else:
                # Create new scope context
                agent_id = _os.getenv("ZEN_AGENT_ID", "zen-mcp-http-server")
                scope_context = create_default_scope_context(
                    agent_id=agent_id,
                    work_dir=_work_dir or "",
                    session_id=session_id or str(uuid.uuid4())
                )

                # Add additional context from environment
                scope_context.org_id = _os.getenv("ZEN_ORG_ID", "default")
                scope_context.project_id = _os.getenv("ZEN_PROJECT_ID", "default")
                scope_context.source_tool = tool_name

            try:
                _norm_rel, _abs_path = validate_and_normalize_work_dir(_work_dir)
                scope_context.work_dir = _norm_rel
                scope_context.repo_root = get_repo_root()

                # Inject scope context into arguments
                arguments = inject_scope_context(arguments, scope_context)
                arguments["_work_dir_abs"] = _abs_path

            except WorkDirError as e:
                warn_msg = f"work_dir validation warning: {e} — defaulting to repo root '.' in warn-only mode"
                logger.warning(warn_msg)
                if _enforce:
                    return {"content": [{"type": "text", "text": f"OUT_OF_SCOPE: {e}"}]}

                # Warn-only default: use repo root
                scope_context.work_dir = ""
                scope_context.repo_root = get_repo_root()
                arguments = inject_scope_context(arguments, scope_context)
                arguments["_work_dir_abs"] = get_repo_root()

        try:
            # Handle built-in demo tools
            if tool_name == "echo":
                result = arguments.get("text", "")

            elif tool_name == "get_time":
                result = datetime.now(timezone.utc).isoformat()

            elif tool_name == "multiply":
                a = arguments.get("a", 0)
                b = arguments.get("b", 0)
                result = a * b

            else:
                # Try to execute Zen tool with XML communication protocol
                zen_tools = get_all_tools()
                if tool_name in zen_tools:
                    tool_info = zen_tools[tool_name]
                    tool_func = tool_info.get("function")

                    if tool_func:
                        # Generate task ID for streaming
                        task_id = f"tool-{tool_name}-{uuid.uuid4().hex[:8]}"

                        # Enhance prompt with XML communication protocol if this is a text-based tool
                        enhanced_arguments = arguments.copy()
                        if "prompt" in enhanced_arguments or "question" in enhanced_arguments or "code" in enhanced_arguments:
                            # Determine agent type based on tool
                            agent_type = AgentType.CLAUDE  # Default to Claude

                            # Enhance the main input field with XML protocol
                            for field in ["prompt", "question", "code", "message", "text"]:
                                if field in enhanced_arguments:
                                    original_input = enhanced_arguments[field]
                                    enhanced_input = enhance_agent_message(original_input, agent_type)
                                    enhanced_arguments[field] = enhanced_input
                                    break

                        # Broadcast start of execution
                        if session_id:
                            await self.streaming_manager.broadcast_message(
                                task_id,
                                StreamMessageType.ACTIVITY_UPDATE,
                                {"content": f"Starting {tool_name} with enhanced protocol"},
                                "zen-tool"
                            )

                        # Execute the tool function
                        if asyncio.iscoroutinefunction(tool_func):
                            raw_result = await tool_func(**enhanced_arguments)
                        else:
                            raw_result = tool_func(**enhanced_arguments)

                        # Parse XML response if present
                        if isinstance(raw_result, str):
                            parsed_response = parse_agent_output(raw_result)

                            # Send structured updates based on parsed XML
                            if session_id:
                                # Status update
                                if parsed_response.status:
                                    await self.streaming_manager.broadcast_message(
                                        task_id,
                                        StreamMessageType.STATUS_UPDATE,
                                        {"status": parsed_response.status, "summary": parsed_response.summary},
                                        "zen-tool"
                                    )

                                # Progress update
                                if parsed_response.progress:
                                    await self.streaming_manager.broadcast_message(
                                        task_id,
                                        StreamMessageType.PROGRESS_UPDATE,
                                        {"progress": parsed_response.progress},
                                        "zen-tool"
                                    )

                                # File operations
                                if parsed_response.files_created or parsed_response.files_modified:
                                    await self.streaming_manager.broadcast_message(
                                        task_id,
                                        StreamMessageType.FILE_UPDATE,
                                        {
                                            "files_created": parsed_response.files_created,
                                            "files_modified": parsed_response.files_modified
                                        },
                                        "zen-tool"
                                    )

                                # Warnings
                                if parsed_response.warnings:
                                    await self.streaming_manager.broadcast_message(
                                        task_id,
                                        StreamMessageType.WARNING,
                                        {"warnings": parsed_response.warnings},
                                        "zen-tool"
                                    )

                            # Format the result with structured XML data
                            formatted_result = format_agent_summary(raw_result)
                            result = formatted_result
                        else:
                            result = raw_result
                    else:
                        raise Exception(f"Tool {tool_name} has no executable function")
                else:
                    raise Exception(f"Unknown tool: {tool_name}")

            # Send completion update
            if session_id:
                await self.streaming_manager.broadcast_message(
                    f"session-{session_id}",
                    StreamMessageType.COMPLETION,
                    {
                        "status": "completed",
                        "tool": tool_name,
                        "result": str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                    },
                    "mcp-server"
                )

            # Normalize result to include JSON ToolOutput with continuation_id when needed
            try:
                from utils.mcp_normalize import normalize_tool_result
                normalized_result = normalize_tool_result(tool_name, arguments, result)
            except Exception:
                logger.debug("[MCP_NORMALIZE][HTTP] Failed to normalize tool result; using raw result", exc_info=True)
                normalized_result = result

            # Prepare text payload for HTTP response
            final_text = ""
            try:
                if isinstance(normalized_result, list) and normalized_result and hasattr(normalized_result[0], "text"):
                    final_text = normalized_result[0].text or ""
                elif isinstance(normalized_result, dict):
                    final_text = json.dumps(normalized_result)
                else:
                    final_text = str(normalized_result)
            except Exception:
                final_text = ""

            # INFO log for diagnostics
            try:
                is_json = False
                try:
                    _tmp = json.loads(final_text) if final_text else None
                    is_json = isinstance(_tmp, (dict, list))
                except Exception:
                    is_json = False
                logger.info(f"[MCP_HTTP] Tool '{tool_name}' outgoing first content is_json={is_json} len={len(final_text)}")
            except Exception:
                pass

            return {
                "content": [
                    {
                        "type": "text",
                        "text": final_text
                    }
                ]
            }

        except Exception as e:
            logger.error(f"Tool execution error for {tool_name}: {e}")

            # Send error update
            if session_id:
                await self.streaming_manager.broadcast_message(
                    f"session-{session_id}",
                    StreamMessageType.ERROR,
                    {"error": str(e), "tool": tool_name},
                    "mcp-server"
                )

            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error executing {tool_name}: {str(e)}"
                    }
                ],
                "isError": True
            }
        finally:
            try:
                logging.getLogger("mcp_activity").info(f"TOOL_COMPLETED: {tool_name}")
            except Exception:
                pass

    async def handle_resources_list(self) -> dict[str, Any]:
        """Handle resources/list request."""
        return {
            "resources": [
                {
                    "uri": "zen://server/info",
                    "name": "Server Information",
                    "description": "Basic information about the Zen MCP server"
                },
                {
                    "uri": "zen://tools/list",
                    "name": "Available Tools",
                    "description": "List of all available tools"
                },
                {
                    "uri": "zen://sessions/active",
                    "name": "Active Sessions",
                    "description": "Information about active MCP sessions"
                }
            ]
        }

    async def handle_resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri")

        if uri == "zen://server/info":
            content = json.dumps({
                "name": "Zen MCP Server",
                "version": "1.0.0",
                "transport": "streamable_http",
                "protocol_version": "2025-03-26",
                "capabilities": ["tools", "resources", "prompts", "streaming"],
                "active_sessions": len(self.sessions)
            }, indent=2)

        elif uri == "zen://tools/list":
            zen_tools = get_all_tools()
            content = json.dumps({
                "zen_tools": list(zen_tools.keys()),
                "total_tools": len(zen_tools)
            }, indent=2)

        elif uri == "zen://sessions/active":
            content = json.dumps({
                "active_sessions": len(self.sessions),
                "sessions": [
                    {
                        "id": sid,
                        "created_at": session["created_at"],
                        "client": session.get("client_info", {}).get("name", "Unknown")
                    }
                    for sid, session in self.sessions.items()
                ]
            }, indent=2)

        else:
            raise Exception(f"Unknown resource URI: {uri}")

        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": content
                }
            ]
        }

    async def handle_prompts_list(self) -> dict[str, Any]:
        """Handle prompts/list request (augmented with per-tool prompts)."""
        prompts = [
            {
                "name": "zen_help",
                "description": "Get help with Zen MCP server usage",
                "arguments": [
                    {"name": "topic", "description": "Help topic (tools, resources, etc.)", "required": False}
                ],
            },
            {
                "name": "zen_tool_guide",
                "description": "Generate a guide for using a specific tool",
                "arguments": [
                    {"name": "tool_name", "description": "Name of the tool to explain", "required": True}
                ],
            },
        ]

        # Add stdio-style prompts
        for _tool_name, t in PROMPT_TEMPLATES_HTTP.items():
            prompts.append({
                "name": t.get("name", _tool_name),
                "description": t.get("description", f"Use {_tool_name}"),
                "arguments": [],
            })
        prompts.append({"name": "continue", "description": "Continue previous conversation", "arguments": []})

        return {"prompts": prompts}

    async def handle_prompts_get(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle prompts/get request (augmented with per-tool prompts)."""
        prompt_name = params.get("name")
        arguments = params.get("arguments", {})

        if prompt_name == "zen_help":
            topic = arguments.get("topic", "general")

            if topic == "tools":
                zen_tools = get_all_tools()
                tool_list = "\n".join([f"- {name}: {info.get('description', 'No description')}"
                                     for name, info in zen_tools.items()])
                content = f"# Zen MCP Server Tools\n\nAvailable tools:\n{tool_list}"

            elif topic == "resources":
                content = """# Zen MCP Server Resources

Available resources:
- zen://server/info - Server information and status
- zen://tools/list - List of available tools
- zen://sessions/active - Active session information"""

            else:
                content = """# Zen MCP Server Help

This is the Zen MCP (Model Context Protocol) server with Streamable HTTP transport.

**Available capabilities:**
- Tools: Execute various AI and development tools
- Resources: Access server information and data
- Prompts: Get help and guidance templates
- Streaming: Real-time updates and progress tracking

**Usage:**
- Use tools/list to see available tools
- Use tools/call to execute tools
- Use resources/list to see available resources
- Use resources/read to access resource data

**Endpoints:**
- POST/GET /mcp - Main MCP protocol endpoint
- GET / - Server information
- GET /status - Server health status
- GET /docs - API documentation"""

        elif prompt_name == "zen_tool_guide":
            tool_name = arguments.get("tool_name")
            zen_tools = get_all_tools()

            if tool_name and tool_name in zen_tools:
                tool_info = zen_tools[tool_name]
                content = f"""# Tool Guide: {tool_name}

**Description:** {tool_info.get('description', 'No description available')}

**Usage:** Call this tool using the tools/call method with the tool name "{tool_name}"

**Parameters:** {json.dumps(tool_info.get('input_schema', {}), indent=2)}

**Example:** Use this tool to {tool_info.get('description', 'perform its intended function')}."""

            else:
                available_tools = list(zen_tools.keys())
                content = f"""# Tool Guide Error

Tool "{tool_name}" not found.

Available tools: {', '.join(available_tools)}

Use the "zen_help" prompt with topic "tools" to see all available tools."""

        else:
            # Map incoming name to a tool template (parity with stdio server)
            tool_name = None
            template_info = None

            # Match by template name first
            for t_name, t in PROMPT_TEMPLATES_HTTP.items():
                if t.get("name") == prompt_name:
                    tool_name = t_name
                    template_info = t
                    break

            # Direct tool name
            if not tool_name and prompt_name in PROMPT_TEMPLATES_HTTP:
                tool_name = prompt_name
                template_info = PROMPT_TEMPLATES_HTTP[prompt_name]

            # Special continue prompt
            if prompt_name == "continue":
                template_info = {
                    "name": "continue",
                    "description": "Continue the previous conversation",
                    "template": "Continue the conversation",
                }
                tool_name = "chat"

            if not template_info or not tool_name:
                raise Exception(f"Unknown prompt: {prompt_name}")

            template = template_info.get("template", f"Use {tool_name}")
            final_model = arguments.get("model", "auto") if arguments else "auto"
            thinking_mode = arguments.get("thinking_mode", "medium") if arguments else "medium"

            # Format template safely
            try:
                content = template.format(model=final_model, thinking_mode=thinking_mode)
            except Exception:
                content = template

            if prompt_name == "continue":
                content = (
                    f"Continue the previous conversation using the {tool_name} tool. "
                    "CRITICAL: You must provide the continuation_id from the previous response to maintain context."
                )

        return {
            "description": f"Generated prompt: {prompt_name}",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": content
                    }
                }
            ]
        }

    async def handle_completion(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle completion/complete request."""
        # This would typically integrate with an LLM
        # For now, return a simple completion
        return {
            "completion": {
                "values": ["example_completion"],
                "total": 1,
                "hasMore": False
            }
        }

    async def handle_logging_set_level(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle logging/setLevel request."""
        level = params.get("level", "INFO")

        # Set logging level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logging.getLogger().setLevel(numeric_level)

        logger.info(f"Logging level set to: {level}")
        return {}

    async def allocate_smart_port(self) -> int:
        """Allocate a port using KInfra smart allocation."""
        if not KINFRA_AVAILABLE:
            logger.warning("KInfra not available, using default port")
            return self.port

        try:
            # Determine preferred port from various sources
            preferred_port = None

            if self.port_strategy == "preferred":
                preferred_port = self.port
            elif self.port_strategy == "env":
                preferred_port = int(os.getenv("PORT", self.port))

            # Use KInfra smart port allocation
            allocated_port = allocate_free_port(
                preferred=preferred_port,
                logger=self.kinfra_logger
            )

            self.allocated_port = allocated_port
            # Update local_base with the actually allocated port
            if hasattr(self, 'app'):
                self.app.state.local_base = f"http://localhost:{allocated_port}"
            logger.info(f"KInfra allocated port: {allocated_port}")
            return allocated_port

        except Exception as e:
            logger.warning(f"Smart port allocation failed: {e}, using default port {self.port}")
            return self.port

    async def setup_tunnel(self) -> Optional[str]:
        """Setup KInfra tunnel if enabled, preferring KInfra's named-tunnel automation when available."""
        if not self.enable_tunnel or not KINFRA_AVAILABLE:
            return None

        try:
            port = self.allocated_port or self.port

            # Derive service slug and root domain
            env_service = os.getenv("SRVC") or os.getenv("SERVICE_SLUG") or os.getenv("SERVICE_NAME") or "zen"
            full_host = self.tunnel_domain or os.getenv("FULL_TUNNEL_HOST")
            root_env = os.getenv("TUNNEL_DOMAIN")

            service_name = env_service
            root_domain = root_env or "kooshapari.com"
            if full_host:
                parts = full_host.split('.')
                # Only override if we have a subdomain (3+ parts like zen.kooshapari.com)
                # For apex domains (kooshapari.com), keep the env_service
                if len(parts) >= 3:
                    service_name = parts[0]
                    root_domain = ".".join(parts[1:])
                elif len(parts) == 2:
                    # Apex domain like kooshapari.com - use as root_domain but keep service_name
                    root_domain = full_host
                else:
                    # Single word - treat as root domain
                    root_domain = full_host

            logger.info(f"Setting up tunnel for port {port} to {service_name}.{root_domain}")

            # 1) Preferred path: Fully automated named tunnel via KInfra
            try:
                if kinfra_ensure_named_tunnel_autocreate:
                    tunnel_id, host, config_path = await kinfra_ensure_named_tunnel_autocreate(service_name, root_domain, port, self.kinfra_logger)
                    self.tunnel_url = f"https://{host}"
                    logger.info(f"Auto-created persistent tunnel: {self.tunnel_url} (tunnel_id={tunnel_id}, config={config_path})")
                    return self.tunnel_url
            except Exception as e:
                logger.warning(f"KInfra auto-create failed, will try named route fallback: {e}")

            # 1b) Fallback: Named tunnel route when TUNNEL_ID provided
            tunnel_id = os.getenv("TUNNEL_ID")
            if tunnel_id:
                try:
                    expected_host, config_path = await ensure_named_tunnel_route(port, self.kinfra_logger)
                    self.tunnel_url = f"https://{expected_host}"
                    logger.info(f"Persistent tunnel ready via named route: {self.tunnel_url} (config={config_path})")
                    return self.tunnel_url
                except Exception as e:
                    logger.warning(f"Named tunnel route failed, falling back to manager: {e}")

            # 2) Fallback: Use tunnel_manager for quick/persistent if credentials provided
            desired_type = TunnelType.PERSISTENT if self.tunnel_domain else TunnelType.QUICK

            # Resolve credentials for persistent tunnels if requested
            credentials_path = None
            tunnel_token = None
            if desired_type == TunnelType.PERSISTENT:
                tunnel_token = os.getenv("CLOUDFLARE_TUNNEL_TOKEN") or os.getenv("TUNNEL_TOKEN")
                creds_env = os.getenv("TUNNEL_CREDENTIALS_FILE")
                if creds_env:
                    credentials_path = Path(creds_env)
                elif tunnel_id:
                    credentials_path = Path.home() / ".cloudflared" / f"{tunnel_id}.json"

                if not tunnel_token and not (credentials_path and credentials_path.exists()):
                    logger.error(
                        "Persistent tunnel requested but missing credentials. Provide one of: "
                        "CLOUDFLARE_TUNNEL_TOKEN/TUNNEL_TOKEN or TUNNEL_CREDENTIALS_FILE or TUNNEL_ID with ~/.cloudflared/<ID>.json"
                    )
                    return None

            config = TunnelConfig(
                name=service_name,
                local_url=f"http://localhost:{port}",
                hostname=self.tunnel_domain if desired_type == TunnelType.PERSISTENT else None,
                tunnel_type=desired_type,
                log_level="info",
                credentials_file=credentials_path,
                tunnel_token=tunnel_token
            )

            # Create async tunnel manager
            self.tunnel_manager = AsyncTunnelManager(config)
            success = await self.tunnel_manager.start()

            if not success:
                logger.error("Failed to start tunnel")
                return None

            # Obtain public URL if available
            pub_url: Optional[str] = None
            try:
                tunnel_info = await self.tunnel_manager.get_info()
                for key in ("url", "public_url", "publicUrl", "hostname", "host"):
                    val = getattr(tunnel_info, key, None)
                    if isinstance(val, str) and val:
                        pub_url = val
                        break
                if pub_url and not pub_url.startswith("http"):
                    pub_url = f"https://{pub_url}"
            except Exception:
                pub_url = None

            if not pub_url and desired_type == TunnelType.PERSISTENT and self.tunnel_domain:
                pub_url = f"https://{self.tunnel_domain}"

            self.tunnel_url = pub_url
            if self.tunnel_url:
                logger.info(f"Tunnel ready: {self.tunnel_url}")
            else:
                logger.info("Quick tunnel started. Public URL may be shown in cloudflared output.")

            return self.tunnel_url

        except Exception as e:
            logger.error(f"Tunnel setup failed: {e}")
            return None

    async def cleanup_tunnel(self):
        """Clean up tunnel on server shutdown."""
        if self.tunnel_manager:
            try:
                logger.info("Stopping tunnel...")
                await self.tunnel_manager.stop()
                logger.info("Tunnel stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping tunnel: {e}")

    def register_zen_tools(self):
        """Register Zen tools with the MCP server."""
        # Note: This integrates with the existing Zen tool system
        # The actual tool execution is handled in handle_tools_call
        zen_tools = get_all_tools()
        logger.info(f"Registered {len(zen_tools)} Zen tools for MCP access")

    async def start_server_with_kinfra(self):
        """Start server with KInfra smart networking and tunneling."""

        # Allocate smart port
        port = await self.allocate_smart_port()
        self.port = port  # Update the port for uvicorn

        # Setup tunnel if enabled
        if self.enable_tunnel:
            tunnel_url = await self.setup_tunnel()
            if tunnel_url:
                logger.info(f"🌐 Tunnel active: {tunnel_url}")

        # Log connection information
        logger.info("🚀 Zen MCP Streamable HTTP server starting")
        logger.info(f"📍 Local: http://{self.host}:{port}")
        logger.info(f"🔗 MCP endpoint: http://{self.host}:{port}/mcp")
        logger.info(f"📚 API docs: http://{self.host}:{port}/docs")
        logger.info(f"❤️  Health check: http://{self.host}:{port}/healthz")

        if self.tunnel_url:
            logger.info(f"🌍 Public URL: {self.tunnel_url}")
            logger.info(f"🔗 Public MCP: {self.tunnel_url}/mcp")
            logger.info(f"❤️  Public Health: {self.tunnel_url}/healthz")

            # Update WebAuthn domain for the tunnel URL
            if self.auth_enabled and self.device_auth:
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(self.tunnel_url)
                    tunnel_domain = parsed_url.hostname

                    # Update the domain in the existing auth manager
                    self.device_auth.update_domain(tunnel_domain)
                    logger.info(f"🔐 Device authentication updated for domain: {tunnel_domain}")

                    # Initialize OAuth 2.0 server with correct issuer URL (tunnel)
                    logger.info(f"🔍 Ensuring OAuth server initialization (tunnel present), auth_enabled={self.auth_enabled}")
                    if not self.oauth2_server:
                        try:
                            from auth.oauth2_server import create_oauth2_server
                            issuer = self.tunnel_url
                            self.oauth2_server = create_oauth2_server(
                                issuer=issuer,
                                webauthn_auth=self.device_auth
                            )
                            # Share the OAuth integration server for token validation
                            if hasattr(self, 'oauth_integration_server') and self.oauth_integration_server:
                                self.oauth2_server.oauth_integration_server = self.oauth_integration_server
                            print(f"🔒 OAuth 2.0 server initialized with issuer: {issuer}")
                            logger.info(f"🔒 OAuth 2.0 server initialized with issuer: {issuer}")
                            
                            # Sync any existing DCR clients to OAuth2Server
                            try:
                                from auth.oauth2_dcr import get_dcr_manager
                                from auth.oauth2_server import GrantType, OAuthClient as OAuth2Client
                                dcr_manager = get_dcr_manager()
                                
                                # Get all DCR clients
                                dcr_clients = await dcr_manager.list_clients()
                                synced_count = 0
                                
                                for dcr_client in dcr_clients:
                                    if dcr_client.client_id not in self.oauth2_server.clients:
                                        # Mirror DCR client to OAuth2Server
                                        token_auth = getattr(dcr_client, "token_endpoint_auth_method", None) or "none"
                                        is_public = (token_auth == "none")
                                        redirect_uris = list(getattr(dcr_client, "redirect_uris", []) or [])
                                        scope_str = getattr(dcr_client, "scope", "mcp mcp:read mcp:write") or ""
                                        allowed_scopes = set(scope_str.split()) if isinstance(scope_str, str) else {"mcp:read", "mcp:write"}
                                        
                                        oauth_client = OAuth2Client(
                                            client_id=dcr_client.client_id,
                                            client_secret=None,
                                            redirect_uris=redirect_uris,
                                            allowed_grant_types={GrantType.AUTHORIZATION_CODE, GrantType.REFRESH_TOKEN},
                                            allowed_scopes=allowed_scopes,
                                            is_public=is_public,
                                            name=getattr(dcr_client, "client_name", None) or "MCP Client"
                                        )
                                        self.oauth2_server.register_client(oauth_client)
                                        synced_count += 1
                                
                                if synced_count > 0:
                                    logger.info(f"📋 Synced {synced_count} DCR clients to OAuth2Server")
                                    print(f"📋 Synced {synced_count} DCR clients to OAuth2Server")
                            except Exception as sync_error:
                                logger.warning(f"Failed to sync DCR clients: {sync_error}")
                        except Exception as oauth_error:
                            logger.warning(f"Failed to initialize OAuth 2.0 server (tunnel): {oauth_error}")

                except Exception as e:
                    logger.warning(f"Failed to update device auth domain: {e}")

            # Wait briefly for public health endpoint to become reachable
            try:
                health_ok = await wait_for_health(self.tunnel_url, 20, self.kinfra_logger)
                if health_ok:
                    logger.info("✅ Public endpoint is reachable")
                else:
                    logger.warning("⏳ Public endpoint health check timed out; switching to backoff retries")
                    # Exponential backoff health-check as a stabilizer
                    backoff_ok = await self._public_health_backoff(self.tunnel_url)
                    if backoff_ok:
                        logger.info("✅ Public endpoint became reachable after backoff")
                    else:
                        logger.warning("⏳ Public endpoint still not reachable after backoff; continuing startup")
            except Exception as e:
                logger.warning(f"⚠️ Public health readiness check failed: {e}")
            # Start background tunnel health monitor
            if os.getenv("ENABLE_TUNNEL_HEALTH_MONITOR", "true").lower() in ("1","true","on","yes"):
                try:
                    if self._health_monitor_task:
                        self._health_monitor_task.cancel()
                    self._health_monitor_task = asyncio.create_task(self._run_public_health_monitor())
                    logger.info("🩺 Tunnel health monitor started")
                except Exception as e:
                    logger.warning(f"Failed to start tunnel health monitor: {e}")

        # Ensure OAuth server is initialized even without a tunnel
        if self.auth_enabled and not self.oauth2_server:
            try:
                from auth.oauth2_server import create_oauth2_server
                issuer = self.tunnel_url or f"http://localhost:{self.allocated_port or self.port}"
                self.oauth2_server = create_oauth2_server(
                    issuer=issuer,
                    webauthn_auth=self.device_auth
                )
                # Share the OAuth integration server for token validation
                if hasattr(self, 'oauth_integration_server') and self.oauth_integration_server:
                    self.oauth2_server.oauth_integration_server = self.oauth_integration_server
                print(f"🔒 OAuth 2.0 server initialized with issuer: {issuer}")
                logger.info(f"🔒 OAuth 2.0 server initialized with issuer: {issuer}")
                
                # Sync any existing DCR clients to OAuth2Server
                try:
                    from auth.oauth2_dcr import get_dcr_manager
                    from auth.oauth2_server import GrantType, OAuthClient as OAuth2Client
                    dcr_manager = get_dcr_manager()
                    
                    # Get all DCR clients
                    dcr_clients = await dcr_manager.list_clients()
                    synced_count = 0
                    
                    for dcr_client in dcr_clients:
                        if dcr_client.client_id not in self.oauth2_server.clients:
                            # Mirror DCR client to OAuth2Server
                            token_auth = getattr(dcr_client, "token_endpoint_auth_method", None) or "none"
                            is_public = (token_auth == "none")
                            redirect_uris = list(getattr(dcr_client, "redirect_uris", []) or [])
                            scope_str = getattr(dcr_client, "scope", "mcp mcp:read mcp:write") or ""
                            allowed_scopes = set(scope_str.split()) if isinstance(scope_str, str) else {"mcp:read", "mcp:write"}
                            
                            oauth_client = OAuth2Client(
                                client_id=dcr_client.client_id,
                                client_secret=None,
                                redirect_uris=redirect_uris,
                                allowed_grant_types={GrantType.AUTHORIZATION_CODE, GrantType.REFRESH_TOKEN},
                                allowed_scopes=allowed_scopes,
                                is_public=is_public,
                                name=getattr(dcr_client, "client_name", None) or "MCP Client"
                            )
                            self.oauth2_server.register_client(oauth_client)
                            synced_count += 1
                    
                    if synced_count > 0:
                        logger.info(f"📋 Synced {synced_count} DCR clients to OAuth2Server")
                        print(f"📋 Synced {synced_count} DCR clients to OAuth2Server")
                except Exception as sync_error:
                    logger.warning(f"Failed to sync DCR clients: {sync_error}")
            except Exception as oauth_error:
                logger.warning(f"Failed to initialize OAuth 2.0 server (no tunnel): {oauth_error}")

        return port

    async def _run_public_health_monitor(self):
        """Continuously check tunnel health and attempt self-healing on failures."""
        interval = float(os.getenv("TUNNEL_HEALTH_CHECK_INTERVAL", "30"))
        failure_threshold = int(os.getenv("TUNNEL_HEALTH_FAILURE_THRESHOLD", "3"))
        timeout = float(os.getenv("PUBLIC_HEALTH_REQUEST_TIMEOUT", "5.0"))
        path = os.getenv("PUBLIC_HEALTH_PATH", "/healthz").strip() or "/healthz"
        summary_interval = float(os.getenv("TUNNEL_HEALTH_SUMMARY_INTERVAL", "300"))
        next_summary = time.time() + summary_interval

        consecutive_failures = 0
        while True:
            try:
                base = (self.tunnel_url or "").rstrip("/")
                if not base:
                    await asyncio.sleep(interval)
                    continue
                url = f"{base}{path if path.startswith('/') else '/' + path}"
                ok = False
                try:
                    import httpx  # type: ignore
                    async with httpx.AsyncClient(timeout=timeout) as client:  # type: ignore
                        r = await client.request("HEAD", url)
                        ok = 200 <= r.status_code < 300
                        if not ok:
                            r = await client.get(url)
                            ok = 200 <= r.status_code < 300
                except Exception:
                    ok = False

                if ok:
                    if consecutive_failures:
                        logger.info("🩺 Tunnel health restored")
                    consecutive_failures = 0
                    self._tunnel_success_total += 1
                    self._tunnel_last_ok_ts = time.time()
                    # Track last-known good host
                    try:
                        from urllib.parse import urlparse
                        host = urlparse(self.tunnel_url or "").hostname
                        if host:
                            self._tunnel_last_good_host = host
                    except Exception:
                        pass
                else:
                    consecutive_failures += 1
                    logger.debug(f"Tunnel health failure {consecutive_failures}/{failure_threshold}")
                    self._tunnel_failures_total += 1
                    # DNS resolution hint for diagnostics
                    try:
                        import socket
                        from urllib.parse import urlparse
                        host = urlparse(self.tunnel_url or "").hostname
                        if host:
                            socket.getaddrinfo(host, None)
                    except Exception as e:
                        logger.debug(f"DNS resolution failed for tunnel host: {e}")
                    if consecutive_failures >= failure_threshold:
                        consecutive_failures = 0
                        await self._attempt_tunnel_recovery()
                # Periodic health summary
                now = time.time()
                if now >= next_summary:
                    lg = self._tunnel_last_good_host or "?"
                    last_ok = int(now - self._tunnel_last_ok_ts) if self._tunnel_last_ok_ts else -1
                    logger.info(
                        f"HEALTH tunnel host={lg} ok_total={self._tunnel_success_total} "
                        f"fail_total={self._tunnel_failures_total} last_ok_ago_s={last_ok}"
                    )
                    next_summary = now + summary_interval
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Health monitor error: {e}")
                await asyncio.sleep(interval)

    async def _attempt_tunnel_recovery(self):
        """Try to recover the public tunnel via re-ensure or manager restart."""
        if not (self.enable_tunnel and KINFRA_AVAILABLE):
            return
        if self._tunnel_recovery_lock is None:
            self._tunnel_recovery_lock = asyncio.Lock()
        async with self._tunnel_recovery_lock:
            try:
                # Recovery cooldown
                cooldown = float(os.getenv("TUNNEL_RECOVERY_COOLDOWN", "120"))
                now = time.time()
                if self._tunnel_last_recovery_ts and (now - self._tunnel_last_recovery_ts) < cooldown:
                    logger.info("⏸️  Skipping tunnel recovery due to cooldown window")
                    return
                logger.warning("🔁 Attempting tunnel recovery")
                # First, try named tunnel re-route if available
                try:
                    expected_host, _ = await ensure_named_tunnel_route(self.allocated_port or self.port, self.kinfra_logger)
                    new_url = f"https://{expected_host}"
                    if new_url != self.tunnel_url:
                        self.tunnel_url = new_url
                        logger.info(f"🔗 Updated tunnel URL to {self.tunnel_url}")
                        await self._post_tunnel_change()
                        self._tunnel_last_recovery_ts = now
                        return
                except Exception as e:
                    logger.debug(f"ensure_named_tunnel_route failed: {e}")

                # Otherwise, restart quick/persistent tunnel manager if present
                if self.tunnel_manager:
                    try:
                        await self.tunnel_manager.stop()
                    except Exception:
                        pass
                    success = await self.tunnel_manager.start()
                    if success:
                        try:
                            info = await self.tunnel_manager.get_info()
                            host = getattr(info, 'url', None) or getattr(info, 'public_url', None) or getattr(info, 'hostname', None)
                            if host and not str(host).startswith("http"):
                                host = f"https://{host}"
                            if host and host != self.tunnel_url:
                                self.tunnel_url = host
                                logger.info(f"🔗 Updated tunnel URL to {self.tunnel_url}")
                                await self._post_tunnel_change()
                                self._tunnel_last_recovery_ts = now
                                return
                        except Exception:
                            pass
                logger.warning("Tunnel recovery attempt finished; URL unchanged")
                self._tunnel_last_recovery_ts = now
            except Exception as e:
                logger.warning(f"Tunnel recovery failed: {e}")

    async def _post_tunnel_change(self):
        """Apply updates after tunnel URL/domain change (WebAuthn domain, OAuth issuer link)."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.tunnel_url or "")
            tunnel_domain = parsed.hostname
            if tunnel_domain and self.device_auth:
                self.device_auth.update_domain(tunnel_domain)
                logger.info(f"🔐 Device authentication updated for domain: {tunnel_domain}")
            if self.oauth2_server and self.tunnel_url:
                try:
                    self.oauth2_server.issuer = self.tunnel_url
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Post-tunnel-change update failed: {e}")

    async def _public_health_backoff(self, base_url: str) -> bool:
        """Exponential backoff health check for the public tunnel endpoint.

        Respects optional env vars:
        - PUBLIC_HEALTH_MAX_ATTEMPTS (default 6)
        - PUBLIC_HEALTH_INITIAL_DELAY (seconds, default 1.0)
        - PUBLIC_HEALTH_BACKOFF_FACTOR (default 2.0)
        - PUBLIC_HEALTH_MAX_DELAY (seconds, default 10.0)
        - PUBLIC_HEALTH_PATH (default "/healthz")
        """
        try:
            import httpx  # type: ignore
        except Exception:
            httpx = None  # type: ignore

        max_attempts = int(os.getenv("PUBLIC_HEALTH_MAX_ATTEMPTS", "6"))
        delay = float(os.getenv("PUBLIC_HEALTH_INITIAL_DELAY", "1.0"))
        factor = float(os.getenv("PUBLIC_HEALTH_BACKOFF_FACTOR", "2.0"))
        max_delay = float(os.getenv("PUBLIC_HEALTH_MAX_DELAY", "10.0"))
        path = os.getenv("PUBLIC_HEALTH_PATH", "/healthz").strip() or "/healthz"

        # Normalize URL
        base = (base_url or "").rstrip("/")
        url = f"{base}{path if path.startswith('/') else '/' + path}"

        for attempt in range(1, max_attempts + 1):
            try:
                if httpx is None:
                    # Fallback using stdlib urllib
                    import urllib.request
                    with urllib.request.urlopen(url, timeout=5) as resp:  # nosec B310
                        ok = 200 <= getattr(resp, 'status', 0) < 300
                else:
                    timeout = float(os.getenv("PUBLIC_HEALTH_REQUEST_TIMEOUT", "5.0"))
                    async with httpx.AsyncClient(timeout=timeout) as client:  # type: ignore
                        # Try HEAD then GET
                        r = await client.request("HEAD", url)
                        ok = 200 <= r.status_code < 300
                        if not ok:
                            r = await client.get(url)
                            ok = 200 <= r.status_code < 300
                if ok:
                    return True
            except Exception as e:
                logger.debug(f"Public health attempt {attempt}/{max_attempts} failed: {e}")

            # Backoff with jitter
            if attempt < max_attempts:
                jitter = random.uniform(0, min(0.3 * delay, 1.0))
                sleep_for = min(delay + jitter, max_delay)
                try:
                    await asyncio.sleep(sleep_for)
                except Exception:
                    time.sleep(sleep_for)
                delay = min(delay * factor, max_delay)

        return False

    def run(self):
        """Run the MCP HTTP server with KInfra integration."""
        async def run_with_setup():
            port = await self.start_server_with_kinfra()

            # Configure uvicorn
            config = uvicorn.Config(
                self.app,
                host=self.host,
                port=port,
                log_level="info",
                access_log=True
            )

            server = uvicorn.Server(config)

            # Setup cleanup on shutdown
            import signal
            def cleanup_handler(signum, frame):
                logger.info("Shutting down server...")
                asyncio.create_task(self.cleanup_tunnel())

            signal.signal(signal.SIGINT, cleanup_handler)
            signal.signal(signal.SIGTERM, cleanup_handler)

            try:
                await server.serve()
            finally:
                await self.cleanup_tunnel()

        # Run the async server
        try:
            asyncio.run(run_with_setup())
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
            sys.exit(1)


def create_zen_mcp_app(host: str = "0.0.0.0",
                       port: Optional[int] = None,
                       enable_tunnel: bool = False,
                       tunnel_domain: str = "zen.kooshapari.com") -> FastAPI:
    """Create and return the FastAPI app for external hosting."""
    server = ZenMCPStreamableServer(
        host=host,
        port=port,
        enable_tunnel=enable_tunnel,
        tunnel_domain=tunnel_domain
    )
    return server.app


def main():
    """Main entry point for the server."""
    import argparse

    parser = argparse.ArgumentParser(description="Zen MCP Streamable HTTP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, help="Preferred port to bind to (default: smart allocation)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--log-format", choices=["text", "json"], help="Console log format (default: env LOG_FORMAT or text)")
    parser.add_argument("--server-timing-details", action="store_true", help="Include fine-grained Server-Timing marks")
    # Flags are optional; defaults enable tunnel + dynamic allocation automatically
    parser.add_argument("--tunnel", action="store_true", help="Enable Cloudflare tunnel (defaults to enabled)")
    parser.add_argument("--domain", help="Full hostname (e.g., zen.kooshapari.com). If omitted, uses SRVC + TUNNEL_DOMAIN")
    parser.add_argument("--port-strategy", choices=["preferred", "dynamic", "env"],
                       help="Port allocation strategy (default: dynamic)")
    parser.add_argument("--dev", action="store_true", help="Development mode with hot-reloading (maintains tunnel)")

    args = parser.parse_args()

    # Configure logging
    desired_format = (args.log_format or os.getenv("LOG_FORMAT", "text")).lower()
    text_fmt = '%(asctime)s - %(name)s - %(levelname)s - cid=%(cid)s - %(message)s'
    if desired_format == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(JSONLogFormatter())
        handler.addFilter(CidLogFilter())
        root = logging.getLogger()
        root.setLevel(getattr(logging, args.log_level.upper()))
        # Replace existing stream handler formatter if present; otherwise add
        has_stream = False
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setFormatter(JSONLogFormatter())
                h.addFilter(CidLogFilter())
                has_stream = True
        if not has_stream:
            root.addHandler(handler)
    else:
        logging.basicConfig(
            level=getattr(logging, args.log_level.upper()),
            format=text_fmt
        )
        # Ensure cid filter is attached to root handlers
        for h in logging.getLogger().handlers:
            h.addFilter(CidLogFilter())

    # Align any existing handlers (e.g., rotating file) with selected format
    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        try:
            if desired_format == "json":
                h.setFormatter(JSONLogFormatter())
            else:
                if not isinstance(h.formatter, JSONLogFormatter):
                    h.setFormatter(logging.Formatter(text_fmt))
            h.addFilter(CidLogFilter())
        except Exception:
            pass

    # Create and run server
    try:
        # In dev mode with hot-reloading
        if args.dev:
            logger.info("🔥 Development mode enabled with hot-reloading")

            # Use a fixed port for development to maintain tunnel connection
            dev_port = args.port or int(os.getenv("MCP_PORT", "8080"))

            # Run uvicorn directly with reload for hot-reloading
            # The tunnel will be maintained separately
            import uvicorn

            # Ensure tunnel is set up if needed
            if args.tunnel or os.getenv("DISABLE_TUNNEL", "").lower() not in ("1", "true", "yes"):
                # Start tunnel in a separate process that persists across reloads
                import atexit
                from subprocess import Popen

                tunnel_config = Path.home() / ".cloudflared" / "config-zen.yml"
                if tunnel_config.exists():
                    logger.info(f"🚇 Using existing tunnel config: {tunnel_config}")
                    # Update config to use our dev port
                    config_data = tunnel_config.read_text()
                    import re
                    config_data = re.sub(r'service: http://localhost:\d+', f'service: http://localhost:{dev_port}', config_data)
                    # Inject recommended settings if missing
                    if 'originRequest:' not in config_data:
                        config_data += "\noriginRequest:\n  connectTimeout: 10s\n  keepAliveTimeout: 30s\n  tcpKeepAlive: 30s\n  http2Origin: true\n  noHappyEyeballs: true\n"
                    if 'loglevel:' not in config_data:
                        config_data += "\nloglevel: info\n"
                    if 'transport-loglevel:' not in config_data:
                        config_data += "transport-loglevel: warn\n"
                    if 'retries:' not in config_data:
                        config_data += "retries: 5\n"
                    if 'graceful-shutdown-seconds:' not in config_data:
                        config_data += "graceful-shutdown-seconds: 5\n"
                    if 'edge-ip-version:' not in config_data:
                        config_data += "edge-ip-version: auto\n"
                    tunnel_config.write_text(config_data)

                    # Check if tunnel is already running
                    tunnel_running = False
                    try:
                        import psutil
                        tunnel_running = any('cloudflared' in ' '.join(p.cmdline()) and 'config-zen.yml' in ' '.join(p.cmdline())
                                           for p in psutil.process_iter(['cmdline']) if p.info['cmdline'])
                    except ImportError:
                        # psutil not available, check using ps command
                        import subprocess
                        try:
                            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                            tunnel_running = 'cloudflared' in result.stdout and 'config-zen.yml' in result.stdout
                        except Exception:
                            pass

                    if not tunnel_running:
                        logger.info("🚇 Starting Cloudflare tunnel...")
                        tunnel_proc = Popen(['cloudflared', 'tunnel', '--config', str(tunnel_config), 'run'],
                                           stdout=open('/tmp/cloudflared-dev.log', 'w'),
                                           stderr=open('/tmp/cloudflared-dev.log', 'w'))
                        atexit.register(lambda: tunnel_proc.terminate())
                        logger.info(f"✅ Tunnel started (PID: {tunnel_proc.pid})")
                    else:
                        logger.info("✅ Tunnel already running")

            # Run uvicorn with reload
            uvicorn.run(
                "server_mcp_http:app",  # Use module:app format for reload to work
                host=args.host,
                port=dev_port,
                reload=True,
                reload_dirs=[str(Path(__file__).parent)],
                reload_includes=["*.py"],
                reload_excludes=["logs/*", "*.log", "__pycache__/*", ".git/*"],
                log_level=args.log_level.lower(),
                access_log=True
            )
        else:
            # Normal mode without hot-reloading
            # Compute smart defaults: tunnel enabled unless disabled via env, dynamic port strategy
            env_disable = os.getenv("DISABLE_TUNNEL", "").lower() in ("1", "true", "yes")
            enable_tunnel = True if args.tunnel or not env_disable else False
            server = ZenMCPStreamableServer(
                host=args.host,
                port=args.port,
                enable_tunnel=enable_tunnel,
                tunnel_domain=args.domain,
                port_strategy=args.port_strategy,
            )
            # Override server timing details if CLI flag is set
            if args.server_timing_details:
                server.server_timing_details = True
            server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


# Module-level app instance for hot-reload development mode
# This gets created when the module is imported with uvicorn reload
app = None
server_instance = None

# Check if we're running in dev/reload mode
# This happens when uvicorn imports the module with reload enabled
if "uvicorn" in sys.modules:
    # We're being imported by uvicorn with reload, create the app instance
    try:
        # Configure logging for dev mode if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - cid=- - %(message)s'
            )

        # Use fixed port for development
        dev_port = int(os.getenv("MCP_PORT", "8080"))

        # Create full server instance with all features
        # The server __init__ already sets up the app with all endpoints
        server_instance = ZenMCPStreamableServer(
            host="0.0.0.0",
            port=dev_port,
            enable_tunnel=False,  # Tunnel is managed separately in dev mode
            port_strategy="preferred"
        )

        # Get the initialized app
        app = server_instance.app

        logger.info(f"🔥 Hot-reload app instance created on port {dev_port} with full features")
    except Exception as e:
        logger.error(f"Failed to create hot-reload app: {e}")
        import traceback
        traceback.print_exc()
        # Create minimal app as fallback
        from fastapi import FastAPI
        app = FastAPI(title="Zen MCP Server (Error)")
        err_text = str(e)

        @app.get("/")
        def error_message():
            return {"error": f"Failed to initialize: {err_text}"}

if __name__ == "__main__":
    main()
# Test hot-reload comment
# Test hot-reload - Sun Aug 31 17:48:51 MST 2025
