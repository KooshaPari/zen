# Legacy OpenAPI Spec

This directory contains the legacy REST OpenAPI specification and JSON schema used by the old minimal HTTP server (`server_http.py`).

Status: Deprecated. The active server is the Streamable HTTP MCP server exposed at `/mcp` (see `docs/streamable_http_mcp.md`).

Files:
- `openapi.json` — Legacy REST API spec (e.g., /tasks, /messages)
- `tools.schema.json` — Legacy tools JSON schema

These remain for reference and tooling compatibility. New development should target the MCP HTTP transport.

