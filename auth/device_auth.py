"""
Unified device-based OAuth authentication for MCP endpoints
Supports WebAuthn, macOS Keychain/Touch ID, and Windows Hello
"""

import json
import os
import secrets
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

try:
    from fastapi import HTTPException, Request
    from fastapi.responses import HTMLResponse
except ImportError:
    # Fallback if FastAPI not available
    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail

    class Request:
        pass

    class HTMLResponse:
        def __init__(self, content):
            self.content = content


class AuthMethod(Enum):
    WEBAUTHN = "webauthn"
    MACOS_TOUCHID = "macos_touchid"
    MACOS_KEYCHAIN = "macos_keychain"
    WINDOWS_HELLO = "windows_hello"
    DEVICE_TOKEN = "device_token"


@dataclass
class DeviceAuthSession:
    """Active authentication session"""
    session_id: str
    user_id: str
    auth_method: AuthMethod
    created_at: float
    last_used: float
    expires_at: float
    device_info: dict[str, Any]


@dataclass
class DeviceRegistration:
    """Device registration information"""
    device_id: str
    user_id: str
    device_name: str
    auth_method: AuthMethod
    registered_at: float
    last_used: float
    public_key: Optional[str] = None
    credential_data: Optional[dict] = None


class UnifiedDeviceAuth:
    """Unified device authentication system"""

    def __init__(self, domain: str = "kooshapari.com"):
        self.domain = domain
        self.sessions: dict[str, DeviceAuthSession] = {}
        self.devices: dict[str, DeviceRegistration] = {}
        self.session_timeout = 3600 * 8  # 8 hours

        # Initialize OAuth consent manager if available
        self.consent_manager = None
        if OAUTH_INTEGRATION_AVAILABLE:
            self.consent_manager = ConsentManager()

        # Initialize platform-specific auth systems
        self.webauthn = None
        self.macos_auth = None
        self.windows_auth = None

        self._initialize_platform_auth()

    def _initialize_platform_auth(self):
        """Initialize platform-specific authentication systems"""

        # WebAuthn (works on all platforms)
        try:
            # Try relative import first, then absolute
            try:
                from .webauthn_flow import WebAuthnDeviceAuth
            except ImportError:
                from auth.webauthn_flow import WebAuthnDeviceAuth

            self.webauthn = WebAuthnDeviceAuth(rp_id=self.domain, rp_name="Zen MCP Server")
            print(f"✅ WebAuthn initialized for domain: {self.domain}")
        except Exception as e:
            print(f"❌ WebAuthn initialization failed: {e}")

        # Continue with other platform auth initialization...

        # macOS specific
        if sys.platform == "darwin":
            try:
                try:
                    from .macos_keychain import MacOSKeychainAuth, NativeTouchIDAuth
                except ImportError:
                    from auth.macos_keychain import MacOSKeychainAuth, NativeTouchIDAuth

                self.macos_keychain = MacOSKeychainAuth("zen-mcp-server")
                self.macos_touchid = NativeTouchIDAuth()
                print("✅ macOS Touch ID initialized")
            except Exception as e:
                print(f"❌ macOS auth initialization failed: {e}")

        # Windows specific
        if sys.platform == "win32":
            try:
                try:
                    from .windows_hello import WindowsHelloAuth
                except ImportError:
                    from auth.windows_hello import WindowsHelloAuth

                self.windows_hello = WindowsHelloAuth("ZenMCPServer")
                print("✅ Windows Hello initialized")
            except Exception as e:
                print(f"❌ Windows Hello initialization failed: {e}")

        # Debug info
        print(f"Platform: {sys.platform}")
        print(f"WebAuthn available: {self.webauthn is not None}")
        if sys.platform == "darwin":
            print(f"macOS Touch ID available: {hasattr(self, 'macos_touchid') and self.macos_touchid is not None}")
        if sys.platform == "win32":
            print(f"Windows Hello available: {hasattr(self, 'windows_hello') and self.windows_hello is not None}")

    def update_domain(self, new_domain: str):
        """Update the domain for WebAuthn authentication"""
        self.domain = new_domain
        if self.webauthn:
            self.webauthn.update_domain(new_domain)
        print(f"✅ Device authentication domain updated to: {new_domain}")

    def get_available_methods(self) -> dict[str, dict]:
        """Get available authentication methods for current platform"""

        methods = {}

        # WebAuthn (universal)
        if self.webauthn:
            methods["webauthn"] = {
                "name": "WebAuthn (Touch ID, Face ID, Security Keys)",
                "platform": "universal",
                "biometric": True,
                "available": True
            }

        # macOS methods
        if sys.platform == "darwin":
            if hasattr(self, 'macos_touchid') and self.macos_touchid:
                methods["macos_touchid"] = {
                    "name": "macOS Touch ID",
                    "platform": "macos",
                    "biometric": True,
                    "available": True
                }

            if hasattr(self, 'macos_keychain') and self.macos_keychain:
                methods["macos_keychain"] = {
                    "name": "macOS Keychain",
                    "platform": "macos",
                    "biometric": False,
                    "available": True
                }

        # Windows methods
        if sys.platform == "win32":
            if hasattr(self, 'windows_hello') and self.windows_hello:
                try:
                    hello_info = self.windows_hello.get_hello_info()
                    methods["windows_hello"] = {
                        "name": "Windows Hello",
                        "platform": "windows",
                        "biometric": True,
                        "available": hello_info.get("available", False),
                        "capabilities": {
                            "face": hello_info.get("face_recognition", False),
                            "fingerprint": hello_info.get("fingerprint", False),
                            "pin": hello_info.get("pin", False)
                        }
                    }
                except Exception as e:
                    print(f"Windows Hello info failed: {e}")
                    methods["windows_hello"] = {
                        "name": "Windows Hello",
                        "platform": "windows",
                        "biometric": True,
                        "available": False,
                        "error": str(e)
                    }

        return methods

    async def initiate_registration(self, user_id: str, device_name: str, method: AuthMethod) -> dict:
        """Initiate device registration"""

        if method == AuthMethod.WEBAUTHN and self.webauthn:
            return await self.webauthn.initiate_registration(user_id, device_name)

        elif method == AuthMethod.MACOS_TOUCHID and self.macos_touchid:
            # For Touch ID, we create a device token
            device_token = secrets.token_urlsafe(32)
            device_id = f"macos_{user_id}_{secrets.token_hex(8)}"

            # Store in keychain
            if self.macos_keychain and self.macos_keychain.store_credential(
                f"{user_id}_device_token", device_token, require_touch_id=True
            ):
                return {
                    "method": "macos_touchid",
                    "device_id": device_id,
                    "device_token": device_token,
                    "challenge": secrets.token_urlsafe(32)
                }

        elif method == AuthMethod.WINDOWS_HELLO and self.windows_hello:
            credential_id = self.windows_hello.create_windows_hello_credential(user_id)
            if credential_id:
                return {
                    "method": "windows_hello",
                    "credential_id": credential_id,
                    "challenge": secrets.token_urlsafe(32)
                }

        raise HTTPException(status_code=400, detail=f"Registration method {method.value} not available")

    async def complete_registration(self, user_id: str, device_name: str, method: AuthMethod, registration_data: dict) -> bool:
        """Complete device registration"""

        device_id = secrets.token_hex(16)

        try:
            if method == AuthMethod.WEBAUTHN and self.webauthn:
                success = await self.webauthn.complete_registration(registration_data)
                if success:
                    device_id = registration_data.get("id", device_id)

            elif method == AuthMethod.MACOS_TOUCHID:
                # Verify the device token was stored
                device_token = registration_data.get("device_token")
                if device_token and self.macos_keychain:
                    stored_token = self.macos_keychain.retrieve_credential_with_biometric(f"{user_id}_device_token")
                    success = stored_token and device_token in stored_token
                    device_id = registration_data.get("device_id", device_id)
                else:
                    success = False

            elif method == AuthMethod.WINDOWS_HELLO:
                # For Windows Hello, credential creation is the registration
                success = registration_data.get("credential_id") is not None
                device_id = registration_data.get("credential_id", device_id)

            else:
                success = False

            if success:
                # Store device registration
                self.devices[device_id] = DeviceRegistration(
                    device_id=device_id,
                    user_id=user_id,
                    device_name=device_name,
                    auth_method=method,
                    registered_at=time.time(),
                    last_used=time.time(),
                    credential_data=registration_data
                )
                print(f"✅ Stored device registration: device_id={device_id}, user_id={user_id}, method={method.value}")

            return success

        except Exception as e:
            print(f"Registration completion failed: {e}")
            return False

    async def initiate_authentication(self, user_id: str, method: Optional[AuthMethod] = None) -> dict:
        """Initiate authentication flow"""

        # Debug logging
        print(f"🔍 Auth initiation for user: {user_id}")
        print(f"🔍 Registered devices: {list(self.devices.keys())}")
        print(f"🔍 All users with devices: {[d.user_id for d in self.devices.values()]}")

        # Find user's devices
        user_devices = [
            device for device in self.devices.values()
            if device.user_id == user_id and (method is None or device.auth_method == method)
        ]

        print(f"🔍 Found {len(user_devices)} devices for user {user_id}")

        if not user_devices:
            raise HTTPException(status_code=404, detail="No registered devices found")

        # Use the most recently used device
        device = max(user_devices, key=lambda d: d.last_used)

        if device.auth_method == AuthMethod.WEBAUTHN and self.webauthn:
            return await self.webauthn.initiate_authentication(user_id)

        elif device.auth_method == AuthMethod.MACOS_TOUCHID:
            return {
                "method": "macos_touchid",
                "device_id": device.device_id,
                "challenge": secrets.token_urlsafe(32),
                "message": "Touch ID authentication required"
            }

        elif device.auth_method == AuthMethod.WINDOWS_HELLO:
            return {
                "method": "windows_hello",
                "device_id": device.device_id,
                "challenge": secrets.token_urlsafe(32),
                "message": "Windows Hello authentication required"
            }

        raise HTTPException(status_code=400, detail="Authentication method not available")

    async def verify_authentication(self, user_id: str, auth_data: dict) -> Optional[DeviceAuthSession]:
        """Verify authentication and create session"""

        method_str = auth_data.get("method")
        if not method_str:
            return None

        try:
            method = AuthMethod(method_str)
        except ValueError:
            return None

        success = False
        device_info = {}

        if method == AuthMethod.WEBAUTHN and self.webauthn:
            verified_user = await self.webauthn.verify_authentication(auth_data)
            success = verified_user == user_id
            device_info = {"type": "webauthn", "credential_id": auth_data.get("id")}

        elif method == AuthMethod.MACOS_TOUCHID and self.macos_keychain:
            device_id = auth_data.get("device_id")
            if device_id and device_id in self.devices:
                # Touch ID verification happens during keychain access
                stored_token = self.macos_keychain.retrieve_credential_with_biometric(f"{user_id}_device_token")
                success = stored_token is not None
                device_info = {"type": "macos_touchid", "device_id": device_id}

        elif method == AuthMethod.WINDOWS_HELLO and self.windows_hello:
            reason = f"Authenticate {user_id} for Zen MCP Server"
            success = self.windows_hello.authenticate_with_windows_hello(user_id, reason)
            device_info = {"type": "windows_hello", "device_id": auth_data.get("device_id")}

        if success:
            # Create session
            session_id = secrets.token_urlsafe(32)
            session = DeviceAuthSession(
                session_id=session_id,
                user_id=user_id,
                auth_method=method,
                created_at=time.time(),
                last_used=time.time(),
                expires_at=time.time() + self.session_timeout,
                device_info=device_info
            )

            self.sessions[session_id] = session
            return session

        return None

    def create_session(self, user_id: str, method: str | AuthMethod = AuthMethod.WEBAUTHN, device_info: Optional[dict] = None) -> DeviceAuthSession:
        """Create an authenticated session (used after successful registration/pairing).

        Args:
            user_id: The user identifier
            method: Authentication method (string or AuthMethod)
            device_info: Optional device metadata

        Returns:
            DeviceAuthSession: Active session
        """
        try:
            if isinstance(method, str):
                method_enum = AuthMethod(method)
            else:
                method_enum = method
        except Exception:
            method_enum = AuthMethod.WEBAUTHN

        session_id = secrets.token_urlsafe(32)
        session = DeviceAuthSession(
            session_id=session_id,
            user_id=user_id,
            auth_method=method_enum,
            created_at=time.time(),
            last_used=time.time(),
            expires_at=time.time() + self.session_timeout,
            device_info=device_info or {},
        )
        self.sessions[session_id] = session
        return session

    def validate_session(self, session_id: str) -> Optional[DeviceAuthSession]:
        """Validate an existing session"""

        session = self.sessions.get(session_id)
        if not session:
            return None

        if time.time() > session.expires_at:
            del self.sessions[session_id]
            return None

        # Update last used
        session.last_used = time.time()
        return session

    def revoke_session(self, session_id: str) -> bool:
        """Revoke a session"""
        return self.sessions.pop(session_id, None) is not None

    def get_user_devices(self, user_id: str) -> list[DeviceRegistration]:
        """Get all registered devices for a user"""
        return [
            device for device in self.devices.values()
            if device.user_id == user_id
        ]

    def revoke_device(self, device_id: str) -> bool:
        """Revoke a device registration"""
        return self.devices.pop(device_id, None) is not None


# OAuth integration imports (lazy loaded)
try:
    from .oauth_integration import setup_oauth_endpoints
    # Temporarily skip ConsentManager due to syntax errors
    # from .oauth_consent import ConsentManager
    OAUTH_INTEGRATION_AVAILABLE = True
except ImportError:
    OAUTH_INTEGRATION_AVAILABLE = False

# Temporary stub for ConsentManager
class ConsentManager:
    def __init__(self):
        pass


# FastAPI integration
def setup_device_auth_endpoints(app, auth_manager: UnifiedDeviceAuth):
    """Setup device authentication endpoints for FastAPI"""

    @app.get("/auth/methods")
    async def get_auth_methods():
        """Get available authentication methods"""
        return auth_manager.get_available_methods()

    @app.post("/auth/signup")
    async def signup_initiate(request: Request):
        """Initiate signup flow - temporary authentication for account creation"""

        data = await request.json()
        user_id = data.get("user_id")  # Email or username

        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")

        # For signup, we create a temporary registration token
        # This will be stored securely (keychain) and linked to the user
        import secrets
        signup_token = secrets.token_urlsafe(32)

        # Store signup token in system keychain (secure storage)
        try:
            if sys.platform == "darwin":
                # Store in macOS Keychain
                if hasattr(auth_manager, 'macos_keychain') and auth_manager.macos_keychain:
                    auth_manager.macos_keychain.store_credential(
                        f"signup_token_{user_id}",
                        signup_token,
                        require_touch_id=True
                    )
                else:
                    raise Exception("macOS Keychain not available")
            else:
                # For other platforms, store in memory temporarily
                # In production, you'd want to use platform-specific secure storage
                if not hasattr(auth_manager, '_signup_tokens'):
                    auth_manager._signup_tokens = {}
                auth_manager._signup_tokens[user_id] = signup_token
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to store signup token: {e}")

        return {
            "success": True,
            "message": "Signup initiated - proceed with WebAuthn registration",
            "user_id": user_id,
            "signup_token": signup_token  # Return for immediate WebAuthn registration
        }

    @app.post("/auth/register/initiate")
    async def initiate_device_registration(request: Request):
        """Initiate device registration"""

        data = await request.json()
        user_id = data.get("user_id")
        device_name = data.get("device_name", "Unknown Device")
        method_str = data.get("method", "webauthn")

        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")

        try:
            method = AuthMethod(method_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid authentication method")

        options = await auth_manager.initiate_registration(user_id, device_name, method)
        return options

    @app.post("/auth/register/complete")
    async def complete_device_registration(request: Request):
        """Complete device registration - validates signup token"""

        data = await request.json()
        user_id = data.get("user_id")
        device_name = data.get("device_name", "Unknown Device")
        method_str = data.get("method", "webauthn")
        registration_data = data.get("registration_data", {})
        signup_token = data.get("signup_token")  # Required for new registrations

        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")

        # Validate signup token for new user registration
        if signup_token:
            try:
                if sys.platform == "darwin":
                    # Verify token from macOS Keychain
                    if hasattr(auth_manager, 'macos_keychain') and auth_manager.macos_keychain:
                        stored_token = auth_manager.macos_keychain.retrieve_credential_with_biometric(
                            f"signup_token_{user_id}"
                        )
                        if not stored_token or signup_token not in stored_token:
                            raise HTTPException(status_code=401, detail="Invalid signup token")
                    else:
                        raise Exception("macOS Keychain not available")
                else:
                    # Check memory storage
                    if not hasattr(auth_manager, '_signup_tokens'):
                        raise HTTPException(status_code=401, detail="Invalid signup token")
                    if auth_manager._signup_tokens.get(user_id) != signup_token:
                        raise HTTPException(status_code=401, detail="Invalid signup token")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Token validation failed: {e}")

        try:
            method = AuthMethod(method_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid authentication method")

        success = await auth_manager.complete_registration(user_id, device_name, method, registration_data)

        # Clean up signup token after successful registration
        if success and signup_token:
            try:
                if sys.platform == "darwin" and hasattr(auth_manager, 'macos_keychain'):
                    # Remove from keychain
                    pass  # Keep token for security audit trail
                else:
                    # Remove from memory
                    if hasattr(auth_manager, '_signup_tokens'):
                        auth_manager._signup_tokens.pop(user_id, None)
            except Exception:
                pass  # Non-critical cleanup failure

        return {"success": success}

    @app.post("/auth/authenticate/initiate")
    async def initiate_authentication(request: Request):
        """Initiate authentication"""

        data = await request.json()
        user_id = data.get("user_id")
        method_str = data.get("method")

        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")

        method = None
        if method_str:
            try:
                method = AuthMethod(method_str)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid authentication method")

        options = await auth_manager.initiate_authentication(user_id, method)
        return options

    @app.post("/auth/login")
    async def login_verify(request: Request):
        """Complete login flow - verify WebAuthn authentication"""

        data = await request.json()
        user_id = data.get("user_id")
        auth_data = data.get("auth_data", {})

        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")

        session = await auth_manager.verify_authentication(user_id, auth_data)

        if session:
            response = {
                "success": True,
                "session_id": session.session_id,
                "user_id": session.user_id,
                "auth_method": session.auth_method.value,
                "expires_at": session.expires_at,
                "message": "Login successful - MCP access granted"
            }

            # Set secure session cookie for MCP access
            from fastapi import Response
            resp = Response(content=json.dumps(response), media_type="application/json")
            resp.set_cookie(
                key="mcp_session",
                value=session.session_id,
                max_age=auth_manager.session_timeout,
                secure=True,
                httponly=True,
                samesite="strict"
            )
            return resp
        else:
            raise HTTPException(status_code=401, detail="Login failed")

    @app.post("/auth/authenticate/verify")
    async def verify_authentication(request: Request):
        """Legacy endpoint - redirects to login"""
        return await login_verify(request)

    @app.get("/auth/session")
    async def get_session_info(request: Request):
        """Get current session information"""

        session_id = request.headers.get("X-Session-ID") or request.cookies.get("mcp_session")

        if not session_id:
            raise HTTPException(status_code=401, detail="No session")

        session = auth_manager.validate_session(session_id)

        if not session:
            raise HTTPException(status_code=401, detail="Invalid session")

        return {
            "user_id": session.user_id,
            "auth_method": session.auth_method.value,
            "created_at": session.created_at,
            "expires_at": session.expires_at,
            "device_info": session.device_info
        }

    @app.post("/auth/session/revoke")
    async def revoke_session(request: Request):
        """Revoke current session"""

        session_id = request.headers.get("X-Session-ID") or request.cookies.get("mcp_session")

        if not session_id:
            raise HTTPException(status_code=401, detail="No session")

        success = auth_manager.revoke_session(session_id)
        return {"success": success}

    @app.get("/auth/devices")
    async def get_user_devices(request: Request):
        """Get user's registered devices"""

        # This endpoint requires authentication
        session_id = request.headers.get("X-Session-ID") or request.cookies.get("mcp_session")

        if not session_id:
            raise HTTPException(status_code=401, detail="No session")

        session = auth_manager.validate_session(session_id)

        if not session:
            raise HTTPException(status_code=401, detail="Invalid session")

        devices = auth_manager.get_user_devices(session.user_id)

        return {
            "devices": [
                {
                    "device_id": device.device_id,
                    "device_name": device.device_name,
                    "auth_method": device.auth_method.value,
                    "registered_at": device.registered_at,
                    "last_used": device.last_used
                }
                for device in devices
            ]
        }

    @app.delete("/auth/devices/{device_id}")
    async def revoke_device(device_id: str, request: Request):
        """Revoke a device"""

        # This endpoint requires authentication
        session_id = request.headers.get("X-Session-ID") or request.cookies.get("mcp_session")

        if not session_id:
            raise HTTPException(status_code=401, detail="No session")

        session = auth_manager.validate_session(session_id)

        if not session:
            raise HTTPException(status_code=401, detail="Invalid session")

        # Verify the device belongs to the authenticated user
        device = auth_manager.devices.get(device_id)
        if not device or device.user_id != session.user_id:
            raise HTTPException(status_code=404, detail="Device not found")

        success = auth_manager.revoke_device(device_id)
        return {"success": success}

    # Middleware for protected endpoints
    def require_device_auth(handler):
        """Decorator to require device authentication"""
        async def wrapper(request: Request):
            session_id = request.headers.get("X-Session-ID") or request.cookies.get("mcp_session")

            if not session_id:
                raise HTTPException(status_code=401, detail="Authentication required")

            session = auth_manager.validate_session(session_id)

            if not session:
                raise HTTPException(status_code=401, detail="Invalid session")

            # Add session info to request state
            request.state.session = session
            request.state.user_id = session.user_id

            return await handler(request)

        return wrapper

    # Setup legacy OAuth integration endpoints (optional)
    oauth_server = None
    enable_integration = os.getenv("ENABLE_OAUTH_INTEGRATION", "false").lower() in ("1", "true", "on", "yes")
    if enable_integration and OAUTH_INTEGRATION_AVAILABLE and auth_manager.webauthn:
        try:
            oauth_server = setup_oauth_endpoints(app, auth_manager.webauthn, auth_manager.domain)
            print(f"✅ Legacy OAuth integration endpoints configured for domain: {auth_manager.domain}")
        except Exception as e:
            print(f"❌ Legacy OAuth integration endpoints setup failed: {e}")
    else:
        print("ℹ️ Legacy OAuth integration disabled; unified OAuth2Server endpoints will be used")

    return {
        "auth_manager": auth_manager,
        "require_auth": require_device_auth,
        "oauth_server": oauth_server
    }
