#!/bin/bash

# OAuth Operator Approval Script
# Usage: ./approve_oauth.sh <request_id> <operator_token>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <request_id> <operator_token>"
    echo ""
    echo "Example:"
    echo "  $0 UgIYTisL-t2QQiijTZV9nZT5eP81aJ8TyXpVoa9hEGo gYfl24-0Wv9CXLLDVHEn_A"
    echo ""
    echo "To view pending approvals, visit: http://localhost:57868/auth/operator"
    exit 1
fi

REQUEST_ID=$1
OPERATOR_TOKEN=$2

# Find the server port
PORT=$(lsof -nP -iTCP -sTCP:LISTEN | grep python | grep -oE ':[0-9]+' | head -1 | tr -d ':')
if [ -z "$PORT" ]; then
    PORT=57868  # Default fallback
fi

echo "Approving OAuth request..."
echo "  Request ID: $REQUEST_ID"
echo "  Token: $OPERATOR_TOKEN"
echo "  Server: http://localhost:$PORT"
echo ""

response=$(curl -s -X POST "http://localhost:$PORT/oauth/operator/approve" \
  -H "Content-Type: application/json" \
  -d "{
    \"request_id\": \"$REQUEST_ID\",
    \"operator_token\": \"$OPERATOR_TOKEN\"
  }")

if [ "$response" = "Approved" ]; then
    echo "✅ Authorization approved successfully!"
else
    echo "❌ Approval failed: $response"
    exit 1
fi