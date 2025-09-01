# ruff: noqa: W293

"""
Windows Hello integration for MCP authentication
Supports fingerprint, face recognition, and PIN
"""

import secrets
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class WindowsHelloCredential:
    """Windows Hello credential information"""
    user_id: str
    credential_id: str
    created_at: float


class WindowsHelloAuth:
    """Windows Hello authentication manager"""

    def __init__(self, app_name: str = "ZenMCPServer"):
        self.app_name = app_name
        self.credentials: dict[str, WindowsHelloCredential] = {}

    def is_windows_hello_available(self) -> bool:
        """Check if Windows Hello is available"""

        if sys.platform != "win32":
            return False

        try:
            # PowerShell script to check Windows Hello availability
            ps_script = '''
            $capability = Get-WindowsOptionalFeature -Online -FeatureName "WindowsHelloFace"
            if ($capability -and $capability.State -eq "Enabled") {
                Write-Output "face_available"
            }

            $biometric = Get-WmiObject -Class Win32_BiometricDevice
            if ($biometric) {
                Write-Output "fingerprint_available"
            }

            # Check for PIN
            $pin = Get-ItemProperty "HKLM:\\SOFTWARE\\Microsoft\\PolicyManager\\current\\device\\DeviceLock" -Name "DevicePasswordEnabled" -ErrorAction SilentlyContinue
            if ($pin -and $pin.DevicePasswordEnabled -eq 1) {
                Write-Output "pin_available"
            }
            '''

            result = subprocess.run([
                "powershell", "-Command", ps_script
            ], capture_output=True, text=True, timeout=10)

            output = result.stdout.lower()
            return "face_available" in output or "fingerprint_available" in output or "pin_available" in output

        except Exception:
            return False

    def create_windows_hello_credential(self, user_id: str) -> Optional[str]:
        """Create Windows Hello credential"""

        try:
            # PowerShell script for credential creation
            credential_id = secrets.token_urlsafe(16)

            ps_script = f'''
            # Load Windows Hello API
            Add-Type -AssemblyName System.Runtime.WindowsRuntime

            # Create credential request
            $credential_name = "{self.app_name}_{user_id}"
            $credential_id = "{credential_id}"

            try {{
                # Use Windows.Security.Credentials.UI namespace
                [Windows.Security.Credentials.UI.UserConsentVerifier,Windows.Security.Credentials.UI,ContentType=WindowsRuntime] | Out-Null

                $availability = [Windows.Security.Credentials.UI.UserConsentVerifier]::CheckAvailabilityAsync()
                $availability.Wait()

                if ($availability.Result -eq "Available") {{
                    $consent = [Windows.Security.Credentials.UI.UserConsentVerifier]::RequestVerificationAsync("Create credential for {self.app_name}")
                    $consent.Wait()

                    if ($consent.Result -eq "Verified") {{
                        Write-Output "credential_created:$credential_id"
                    }} else {{
                        Write-Output "credential_failed:user_cancelled"
                    }}
                }} else {{
                    Write-Output "credential_failed:not_available"
                }}
            }} catch {{
                Write-Output "credential_failed:$($_.Exception.Message)"
            }}
            '''

            result = subprocess.run([
                "powershell", "-Command", ps_script
            ], capture_output=True, text=True, timeout=30)

            output = result.stdout.strip()

            if output.startswith("credential_created:"):
                return credential_id

            return None

        except Exception as e:
            print(f"Windows Hello credential creation failed: {e}")
            return None

    def authenticate_with_windows_hello(self, user_id: str, reason: str = None) -> bool:
        """Authenticate using Windows Hello"""

        if not reason:
            reason = f"Authenticate {user_id} for {self.app_name}"

        try:
            ps_script = f'''
            Add-Type -AssemblyName System.Runtime.WindowsRuntime

            try {{
                [Windows.Security.Credentials.UI.UserConsentVerifier,Windows.Security.Credentials.UI,ContentType=WindowsRuntime] | Out-Null

                $availability = [Windows.Security.Credentials.UI.UserConsentVerifier]::CheckAvailabilityAsync()
                $availability.Wait()

                if ($availability.Result -eq "Available") {{
                    $consent = [Windows.Security.Credentials.UI.UserConsentVerifier]::RequestVerificationAsync("{reason}")
                    $consent.Wait()

                    if ($consent.Result -eq "Verified") {{
                        Write-Output "auth_success"
                    }} else {{
                        Write-Output "auth_failed:$($consent.Result)"
                    }}
                }} else {{
                    Write-Output "auth_failed:not_available"
                }}
            }} catch {{
                Write-Output "auth_failed:$($_.Exception.Message)"
            }}
            '''

            result = subprocess.run([
                "powershell", "-Command", ps_script
            ], capture_output=True, text=True, timeout=30)

            return result.stdout.strip() == "auth_success"

        except Exception as e:
            print(f"Windows Hello authentication failed: {e}")
            return False

    def get_hello_info(self) -> dict:
        """Get Windows Hello capability information"""

        try:
            ps_script = '''
            $info = @{}

            # Check face recognition
            $face = Get-WindowsOptionalFeature -Online -FeatureName "WindowsHelloFace" -ErrorAction SilentlyContinue
            $info.face_recognition = ($face -and $face.State -eq "Enabled")

            # Check fingerprint
            $fingerprint = Get-WmiObject -Class Win32_BiometricDevice -ErrorAction SilentlyContinue
            $info.fingerprint = ($fingerprint -ne $null)

            # Check PIN
            $pin = Get-ItemProperty "HKLM:\\SOFTWARE\\Microsoft\\PolicyManager\\current\\device\\DeviceLock" -Name "DevicePasswordEnabled" -ErrorAction SilentlyContinue
            $info.pin = ($pin -and $pin.DevicePasswordEnabled -eq 1)

            # Check general availability
            Add-Type -AssemblyName System.Runtime.WindowsRuntime
            [Windows.Security.Credentials.UI.UserConsentVerifier,Windows.Security.Credentials.UI,ContentType=WindowsRuntime] | Out-Null

            $availability = [Windows.Security.Credentials.UI.UserConsentVerifier]::CheckAvailabilityAsync()
            $availability.Wait()

            $info.available = ($availability.Result -eq "Available")
            $info.status = $availability.Result

            $info | ConvertTo-Json
            '''

            result = subprocess.run([
                "powershell", "-Command", ps_script
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and result.stdout:
                import json
                return json.loads(result.stdout)

            return {"available": False, "error": "Failed to get info"}

        except Exception as e:
            return {"available": False, "error": str(e)}


# C# alternative for deeper integration (save as WindowsHelloAuth.cs)
CSHARP_WINDOWS_HELLO = '''
using System;
using System.Threading.Tasks;
using Windows.Security.Credentials.UI;
using System.Runtime.InteropServices;

namespace ZenMCPWindowsHello
{
    public class WindowsHelloAuth
    {
        public static async Task<bool> IsAvailableAsync()
        {
            try
            {
                var availability = await UserConsentVerifier.CheckAvailabilityAsync();
                return availability == UserConsentVerifierAvailability.Available;
            }
            catch
            {
                return false;
            }
        }

        public static async Task<bool> AuthenticateAsync(string reason)
        {
            try
            {
                var availability = await UserConsentVerifier.CheckAvailabilityAsync();

                if (availability == UserConsentVerifierAvailability.Available)
                {
                    var consent = await UserConsentVerifier.RequestVerificationAsync(reason);
                    return consent == UserConsentVerificationResult.Verified;
                }

                return false;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Authentication failed: {ex.Message}");
                return false;
            }
        }

        // Command line interface
        public static async Task Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("Usage: WindowsHelloAuth.exe <reason>");
                Environment.Exit(1);
            }

            string reason = string.Join(" ", args);
            bool success = await AuthenticateAsync(reason);

            Environment.Exit(success ? 0 : 1);
        }
    }
}
'''


def setup_windows_hello_for_mcp(app):
    """Setup Windows Hello authentication endpoints"""

    hello_auth = WindowsHelloAuth()

    @app.get("/auth/windows/info")
    async def get_windows_hello_info():
        """Get Windows Hello availability and capabilities"""
        return hello_auth.get_hello_info()

    @app.post("/auth/windows/register")
    async def register_windows_hello_device(request):
        """Register device with Windows Hello"""

        data = await request.json()
        user_id = data.get("user_id")

        if not user_id:
            return {"error": "user_id required"}

        if not hello_auth.is_windows_hello_available():
            return {"error": "Windows Hello not available"}

        credential_id = hello_auth.create_windows_hello_credential(user_id)

        if credential_id:
            return {
                "success": True,
                "credential_id": credential_id,
                "message": "Device registered with Windows Hello"
            }
        else:
            return {"error": "Failed to create Windows Hello credential"}

    @app.post("/auth/windows/authenticate")
    async def authenticate_with_windows_hello(request):
        """Authenticate using Windows Hello"""

        data = await request.json()
        user_id = data.get("user_id")
        reason = data.get("reason")

        if not user_id:
            return {"error": "user_id required"}

        auth_reason = reason or f"Authenticate {user_id} for Zen MCP Server"

        if hello_auth.authenticate_with_windows_hello(user_id, auth_reason):
            session_token = secrets.token_urlsafe(32)

            return {
                "success": True,
                "session_token": session_token,
                "user_id": user_id,
                "method": "windows_hello"
            }
        else:
            return {"error": "Windows Hello authentication failed"}

    return hello_auth
