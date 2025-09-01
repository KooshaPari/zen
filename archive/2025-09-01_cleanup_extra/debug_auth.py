#!/usr/bin/env python3
"""
Debug script to test authentication setup
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_auth_imports():
    """Test authentication module imports"""
    print("🔍 Testing authentication imports...")

    try:
        print("Testing device_auth import...")
        from auth.device_auth import UnifiedDeviceAuth
        print("✅ device_auth imported successfully")

        print("Testing UnifiedDeviceAuth initialization...")
        auth = UnifiedDeviceAuth(domain="localhost")
        print("✅ UnifiedDeviceAuth initialized")

        print("Testing get_available_methods...")
        methods = auth.get_available_methods()
        print(f"✅ Available methods: {list(methods.keys())}")

        for method_name, method_info in methods.items():
            print(f"  📱 {method_name}: {method_info.get('name', 'Unknown')}")
            print(f"     Platform: {method_info.get('platform', 'unknown')}")
            print(f"     Available: {method_info.get('available', False)}")
            print(f"     Biometric: {method_info.get('biometric', False)}")

        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_webauthn():
    """Test WebAuthn module specifically"""
    print("\n🔍 Testing WebAuthn module...")

    try:
        from auth.webauthn_flow import WebAuthnDeviceAuth
        WebAuthnDeviceAuth(rp_id="localhost", rp_name="Test")
        print("✅ WebAuthn module working")
        return True
    except Exception as e:
        print(f"❌ WebAuthn failed: {e}")
        return False

def test_platform_auth():
    """Test platform-specific auth modules"""
    print(f"\n🔍 Testing platform-specific auth (platform: {sys.platform})...")

    if sys.platform == "darwin":
        try:
            from auth.macos_keychain import MacOSKeychainAuth
            MacOSKeychainAuth("test-service")
            print("✅ macOS Keychain auth working")
        except Exception as e:
            print(f"❌ macOS auth failed: {e}")

    elif sys.platform == "win32":
        try:
            from auth.windows_hello import WindowsHelloAuth
            WindowsHelloAuth("TestApp")
            print("✅ Windows Hello auth working")
        except Exception as e:
            print(f"❌ Windows Hello failed: {e}")

    else:
        print(f"ℹ️  Platform {sys.platform} - only WebAuthn available")

def test_fastapi_integration():
    """Test FastAPI integration"""
    print("\n🔍 Testing FastAPI integration...")

    try:
        from fastapi import FastAPI

        from auth.device_auth import UnifiedDeviceAuth, setup_device_auth_endpoints

        app = FastAPI()
        auth = UnifiedDeviceAuth(domain="localhost")
        setup_device_auth_endpoints(app, auth)

        print("✅ FastAPI integration working")
        print(f"   Available routes: {len(app.routes)} routes")

        # List auth routes
        auth_routes = [route for route in app.routes if hasattr(route, 'path') and '/auth' in route.path]
        for route in auth_routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                print(f"   🛣️  {list(route.methods)[0]} {route.path}")

        return True

    except Exception as e:
        print(f"❌ FastAPI integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all authentication tests"""
    print("🚀 Zen MCP Authentication Debug Script")
    print("=" * 50)

    results = []

    results.append(("Auth Imports", test_auth_imports()))
    results.append(("WebAuthn", test_webauthn()))
    results.append(("Platform Auth", test_platform_auth()))
    results.append(("FastAPI Integration", test_fastapi_integration()))

    print("\n" + "=" * 50)
    print("📊 Test Results:")

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False

    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

    if not all_passed:
        print("\n💡 Troubleshooting:")
        print("   1. Make sure FastAPI is installed: pip install fastapi")
        print("   2. Check if auth/ directory has __init__.py file")
        print("   3. Verify all auth modules are present")
        print("   4. Run with: ENABLE_DEVICE_AUTH=true python debug_auth.py")

if __name__ == "__main__":
    main()
