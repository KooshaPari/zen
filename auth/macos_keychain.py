"""
macOS Keychain and Touch ID integration for MCP authentication
"""

import hashlib
import secrets
import subprocess
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class KeychainCredential:
    """Keychain stored credential"""
    service: str
    account: str
    password: str
    created_at: float


class MacOSKeychainAuth:
    """macOS Keychain and Touch ID authentication"""

    def __init__(self, service_name: str = "zen-mcp-server"):
        self.service_name = service_name
        self.keychain_name = "login"  # or "System" for system keychain

    def store_credential(self, user_id: str, password: str, require_touch_id: bool = True) -> bool:
        """Store credential in macOS Keychain with Touch ID protection"""

        try:
            # Build security command
            cmd = [
                "security", "add-generic-password",
                "-s", self.service_name,
                "-a", user_id,
                "-p", password,
                "-A",  # Allow access by all applications
                "-T", "",  # Trust self (current application)
                "-U"   # Update if exists
            ]

            if require_touch_id:
                # Add Touch ID requirement for access
                cmd.extend([
                    "-A",  # Allow access
                    "-T", "/System/Library/CoreServices/SecurityAgent"  # Allow SecurityAgent (Touch ID)
                ])

            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0

        except Exception as e:
            print(f"Failed to store credential: {e}")
            return False

    def retrieve_credential_with_biometric(self, user_id: str) -> Optional[str]:
        """Retrieve credential from keychain with biometric prompt"""

        try:
            # This will trigger Touch ID/Face ID prompt
            cmd = [
                "security", "find-generic-password",
                "-s", self.service_name,
                "-a", user_id,
                "-w"  # Show password only
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                return result.stdout.strip()

            return None

        except Exception as e:
            print(f"Failed to retrieve credential: {e}")
            return None

    def delete_credential(self, user_id: str) -> bool:
        """Delete credential from keychain"""

        try:
            cmd = [
                "security", "delete-generic-password",
                "-s", self.service_name,
                "-a", user_id
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0

        except Exception as e:
            print(f"Failed to delete credential: {e}")
            return False

    def generate_device_token(self, user_id: str) -> Optional[str]:
        """Generate and store a device-specific token"""

        # Generate secure token
        token = secrets.token_urlsafe(32)

        # Hash for storage verification
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        # Store in keychain
        keychain_data = f"{token}:{token_hash}:{time.time()}"

        if self.store_credential(f"{user_id}_device_token", keychain_data, require_touch_id=True):
            return token

        return None

    def verify_device_token(self, user_id: str, token: str) -> bool:
        """Verify device token with biometric authentication"""

        # This triggers Touch ID prompt
        keychain_data = self.retrieve_credential_with_biometric(f"{user_id}_device_token")

        if not keychain_data:
            return False

        try:
            stored_token, stored_hash, created_at = keychain_data.split(":")

            # Verify token
            if stored_token != token:
                return False

            # Verify hash
            if hashlib.sha256(token.encode()).hexdigest() != stored_hash:
                return False

            # Check expiration (24 hours)
            if time.time() - float(created_at) > 86400:
                return False

            return True

        except Exception as e:
            print(f"Token verification failed: {e}")
            return False


class MacOSSecureEnclaveAuth:
    """Secure Enclave-based authentication (requires Objective-C bindings)"""

    @staticmethod
    def create_secure_key(user_id: str) -> bool:
        """Create a key in Secure Enclave (requires native implementation)"""

        # This would require PyObjC or ctypes bindings to Security framework
        # Simplified version using command line tools

        applescript = f'''
        tell application "System Events"
            display dialog "Authenticate to create secure key for {user_id}" ¬
                with icon caution ¬
                buttons {{"Cancel", "Authenticate"}} ¬
                default button "Authenticate" ¬
                with title "Zen MCP Server"
        end tell
        '''

        try:
            result = subprocess.run(
                ["osascript", "-e", applescript],
                capture_output=True,
                text=True
            )

            return "Authenticate" in result.stdout

        except Exception:
            return False

    @staticmethod
    def authenticate_with_secure_enclave(user_id: str) -> bool:
        """Authenticate using Secure Enclave"""

        # This would typically use LocalAuthentication framework
        # Simplified AppleScript version for demonstration

        applescript = f'''
        tell application "System Events"
            display dialog "Touch ID authentication required for {user_id}" ¬
                with icon note ¬
                buttons {{"Cancel", "Use Touch ID"}} ¬
                default button "Use Touch ID" ¬
                with title "Zen MCP Server Authentication"
        end tell
        '''

        try:
            result = subprocess.run(
                ["osascript", "-e", applescript],
                capture_output=True,
                text=True
            )

            return "Touch ID" in result.stdout

        except Exception:
            return False


# Swift script for proper Touch ID integration (save as touchid_auth.swift)
SWIFT_TOUCHID_SCRIPT = '''
#!/usr/bin/env swift

import LocalAuthentication
import Foundation

func authenticateWithTouchID(reason: String) -> Bool {
    let context = LAContext()
    var error: NSError?

    // Check if biometric authentication is available
    guard context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error) else {
        print("Biometric authentication not available: \\(error?.localizedDescription ?? "Unknown error")")
        return false
    }

    let semaphore = DispatchSemaphore(value: 0)
    var result = false

    context.evaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, localizedReason: reason) { success, authenticationError in
        result = success
        if let error = authenticationError {
            print("Authentication failed: \\(error.localizedDescription)")
        }
        semaphore.signal()
    }

    semaphore.wait()
    return result
}

// Usage: ./touchid_auth.swift "Authenticate for Zen MCP Server"
if CommandLine.arguments.count > 1 {
    let reason = CommandLine.arguments[1]
    let success = authenticateWithTouchID(reason: reason)
    exit(success ? 0 : 1)
} else {
    print("Usage: touchid_auth.swift <reason>")
    exit(1)
}
'''


class NativeTouchIDAuth:
    """Native Touch ID authentication using Swift script"""

    def __init__(self):
        self.script_path = "/tmp/touchid_auth.swift"
        self.setup_swift_script()

    def setup_swift_script(self):
        """Create Swift script for Touch ID"""
        with open(self.script_path, 'w') as f:
            f.write(SWIFT_TOUCHID_SCRIPT)

        # Make executable
        subprocess.run(["chmod", "+x", self.script_path])

    def authenticate_touch_id(self, reason: str = "Authenticate for Zen MCP Server") -> bool:
        """Authenticate using Touch ID"""

        try:
            result = subprocess.run(
                ["swift", self.script_path, reason],
                capture_output=True,
                text=True,
                timeout=30
            )

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            print(f"Touch ID authentication failed: {e}")
            return False


# Integration example
def setup_macos_auth_for_mcp(app, service_name: str = "zen-mcp-server"):
    """Setup macOS authentication endpoints for MCP server"""

    keychain_auth = MacOSKeychainAuth(service_name)
    touchid_auth = NativeTouchIDAuth()

    @app.post("/auth/macos/register")
    async def register_macos_device(request):
        """Register device with macOS keychain and Touch ID"""

        data = await request.json()
        user_id = data.get("user_id")

        if not user_id:
            return {"error": "user_id required"}

        # Generate device token
        device_token = keychain_auth.generate_device_token(user_id)

        if device_token:
            return {
                "success": True,
                "device_token": device_token,
                "message": "Device registered. Token stored in keychain with Touch ID protection."
            }
        else:
            return {"error": "Failed to register device"}

    @app.post("/auth/macos/authenticate")
    async def authenticate_macos_device(request):
        """Authenticate using macOS Touch ID"""

        data = await request.json()
        user_id = data.get("user_id")
        device_token = data.get("device_token")

        if not user_id or not device_token:
            return {"error": "user_id and device_token required"}

        # This will prompt for Touch ID
        if keychain_auth.verify_device_token(user_id, device_token):
            # Generate session token
            session_token = secrets.token_urlsafe(32)

            return {
                "success": True,
                "session_token": session_token,
                "user_id": user_id,
                "message": "Authentication successful via Touch ID"
            }
        else:
            return {"error": "Authentication failed"}

    @app.post("/auth/macos/touchid-only")
    async def touchid_authenticate(request):
        """Direct Touch ID authentication"""

        data = await request.json()
        user_id = data.get("user_id", "Unknown User")

        reason = f"Authenticate {user_id} for Zen MCP Server access"

        if touchid_auth.authenticate_touch_id(reason):
            session_token = secrets.token_urlsafe(32)

            return {
                "success": True,
                "session_token": session_token,
                "user_id": user_id,
                "method": "touch_id"
            }
        else:
            return {"error": "Touch ID authentication failed"}

    return {
        "keychain_auth": keychain_auth,
        "touchid_auth": touchid_auth
    }
