# Identity, Scoping, and Access Control for Zen Shared Data

## Objectives
- Uniquely and cryptographically identify every agent/session
- Enforce least-privilege scoped access to data/workspaces
- Support multi-org, multi-project, team subdivisions, and per-work_dir scoping

## Identity Model
- Principals: Agent, HumanUser, Service, Team, Org, Project
- Credential options:
  - OAuth2/OIDC (JWT access tokens) with audience=zen-mcp, scopes, and claims
  - mTLS client certs mapping to subjects in an identity directory
  - API keys only for dev; wrap into short-lived signed tokens server-side

### Token/Claim Schema (JWT)
- sub: principal id (agent:{uuid})
- org: org id
- team: team id(s)
- proj: project id
- repo: repo id or URL
- work_dir: repo-relative directory prefix
- roles: ["admin","maintainer","developer","viewer"]
- perms: fine-grained privileges
- exp/iat/nbf: standard claims
- kid/iss: key and issuer identifiers

## Session Binding
- On connect (stdio or HTTP MCP), server authenticates and creates a session record:
  - session_id, principal, org/team/proj, allowed work_dirs, effective roles
  - For HTTP MCP (server_mcp_http.py), include session identity in StreamingManager context

## Scope Enforcement
- All tool calls must include work_dir. Enforcement steps:
  1) Validate work_dir path is canonical, exists, and inside repo
  2) Check against session.allowed_work_dirs and policies
  3) Derive scope_context = {org, proj, team, agent, repo, work_dir}
  4) Attach to tool call and downstream service requests

### Partitioning Policy
- Backend/frontend partial isolation: allow separate work_dirs for `backend/` and `frontend/` with shared `api/` or `contracts/`
- Cross-scope reads: allowed only to shared interfaces (e.g., OpenAPI specs), denied for impl details
- Write operations: allowed only within caller's work_dir partitions unless role permits shared areas

## Authorization Policy Engine
- Static RBAC + doc-driven ABAC rules
- Policy inputs: identity claims, tool name, resource type, scope_context
- Deny by default; explicit allow rules for shared interfaces

## Data Plane Scoping
- VectorDB/RAG namespaces: `{org}/{proj}/{repo}/{work_dir}` + sub-collections for `code`, `knowledge`, `other`
- Knowledge Graph: tenant-aware labels/attributes and relationship filters with query guards
- LSP/QA caches: keys include commit_sha and work_dir
- Memory: key prefixing `{org}:{proj}:{agent}:{type}` with scope in metadata

## Auditing & Observability
- Log all access decisions and tool calls with scope_context
- Emit events to bus (e.g., Kafka) for security analytics and simulator tests

## Backward Compatibility Strategy
- Phase 1: Log-only enforcement (warn when work_dir missing or out-of-scope)
- Phase 2: Soft-fail with suggestions to provide work_dir
- Phase 3: Hard enforcement; tools refuse without valid scope

## Implementation Hooks in Repo
- server.py and server_mcp_http.py: insert identity/session middlewares, require work_dir in handle_call_tool/handle_tools_call
- tools/shared/base_tool.py: base argument contract to include work_dir and scope_context
- utils/agent_memory.py & utils/conversation_memory.py: extend to store identity+scope metadata

## Open Questions
- Do we require mutual TLS for intra-datacenter traffic?
- Should we support per-branch scopes in addition to work_dir?
- Versioning of tokens and policies across teams?

