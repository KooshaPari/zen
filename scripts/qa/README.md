Quickstart
- MCP HTTP smoke: `bash scripts/qa/curl_api_smoke.sh` (requires Streamable MCP HTTP on :8080)
- MCP stdio smoke: `python scripts/qa/mcp_stdio_ping.py`

Notes
- Export `API_BASE` to point curls at a non-default host/port.
- Scripts are non-destructive and suitable for CI gating.
