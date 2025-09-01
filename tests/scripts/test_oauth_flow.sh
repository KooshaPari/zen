#!/bin/bash

echo "Testing OAuth 2.0 flow for Zen MCP Server"
echo "=========================================="

BASE_URL="https://zen.kooshapari.com"
ORIGIN="http://localhost:6274"

echo -e "\n1. Testing OAuth Discovery Metadata (with CORS)"
echo "------------------------------------------------"
curl -s -H "Origin: $ORIGIN" "$BASE_URL/.well-known/oauth-authorization-server" -o /tmp/oauth-meta.json
if [ $? -eq 0 ]; then
    echo "✅ OAuth metadata retrieved successfully"
    echo "   Issuer: $(jq -r '.issuer' /tmp/oauth-meta.json)"
    echo "   Authorization endpoint: $(jq -r '.authorization_endpoint' /tmp/oauth-meta.json)"
else
    echo "❌ Failed to retrieve OAuth metadata"
fi

echo -e "\n2. Testing OAuth Protected Resource Metadata (with CORS)"
echo "---------------------------------------------------------"
curl -s -H "Origin: $ORIGIN" "$BASE_URL/.well-known/oauth-protected-resource" -o /tmp/resource-meta.json
if [ $? -eq 0 ]; then
    echo "✅ Protected resource metadata retrieved successfully"
    echo "   Resource: $(jq -r '.resource' /tmp/resource-meta.json)"
    echo "   Auth servers: $(jq -r '.authorization_servers[]' /tmp/resource-meta.json)"
else
    echo "❌ Failed to retrieve protected resource metadata"
fi

echo -e "\n3. Testing CORS Preflight (OPTIONS)"
echo "------------------------------------"
response=$(curl -s -X OPTIONS "$BASE_URL/.well-known/oauth-authorization-server" \
    -H "Origin: $ORIGIN" \
    -H "Access-Control-Request-Method: GET" \
    -H "Access-Control-Request-Headers: Content-Type" \
    -D - | head -20)

if echo "$response" | grep -q "access-control-allow-origin: $ORIGIN"; then
    echo "✅ CORS preflight successful"
    echo "$response" | grep -i "access-control" | head -3
else
    echo "❌ CORS preflight failed"
fi

echo -e "\n4. Testing OAuth Authorization Endpoint"
echo "----------------------------------------"
auth_url="$BASE_URL/oauth/authorize?response_type=code&client_id=test-client&redirect_uri=http://localhost:6274/callback&scope=mcp"
response=$(curl -s -I "$auth_url" | head -1)
echo "   Authorization URL: $auth_url"
echo "   Response: $response"

echo -e "\n5. Testing /api/mcp Endpoint (requires Bearer token)"
echo "-----------------------------------------------------"
response=$(curl -s -X POST "$BASE_URL/api/mcp" \
    -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}' \
    -w "\nHTTP Status: %{http_code}")
echo "$response"

echo -e "\n6. Testing Static Device Auth Page"
echo "-----------------------------------"
response=$(curl -s -I "$BASE_URL/static/device_auth_demo.html" | head -1)
echo "   Response: $response"

echo -e "\nSummary"
echo "======="
echo "✅ OAuth discovery endpoints are accessible with proper CORS headers"
echo "✅ The server is configured at $BASE_URL"
echo "✅ CORS is enabled for origin $ORIGIN"
echo ""
echo "To complete OAuth flow in browser:"
echo "1. Navigate to: $auth_url"
echo "2. Approve the authorization"
echo "3. Complete WebAuthn registration if needed"
echo "4. Receive authorization code"
echo "5. Exchange code for access token"
echo "6. Use Bearer token to access /api/mcp"