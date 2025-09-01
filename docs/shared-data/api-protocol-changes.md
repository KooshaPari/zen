# API and Protocol Changes

## MCP Tool Contract
- All tools MUST accept `work_dir` (repo-relative) in arguments
- Optional `scope_context` accepted but set by server: {org, proj, team, agent, repo, work_dir}
- Tools MUST propagate `scope_context` to downstream services (KG, Vector, LSP/QA, Memory)

## Server Enforcement
- server.py and server_mcp_http.py:
  - On tool discovery, annotate tool descriptions to include `work_dir` requirement
  - On tools/call, validate and inject scope_context from session claims
  - Deny requests without valid scope, with actionable error

## Identity & Sessions
- HTTP MCP: OAuth2 bearer tokens or mTLS; map to session with claims
- STDIO: API key + signed JWT handoff; short-lived session tokens

## Tool Descriptions
- Extend Tool metadata (ToolAnnotations or description) to say: "Requires work_dir (repo-relative)"
- Provide example arguments in `list_tools` output

## Resource Scoping
- resources/list and resources/read MUST enforce work_dir and tenant policies

## Backward Compatibility
- Phase 1: Warning headers in responses when `work_dir` missing; return suggested path
- Phase 2: Soft errors with remediation messages
- Phase 3: Hard enforcement

## Example Tool Arg Schema (JSON)
{
  "name": "analyze",
  "arguments": {
    "code": "string",
    "work_dir": "string (required)",
    "top_k": 8,
    "filters": {"lang": "python"}
  }
}

## Error Format
{
  "error": "OUT_OF_SCOPE",
  "message": "work_dir 'frontend/ui' is not permitted for agent X",
  "allowed": ["frontend/", "api/"]
}

