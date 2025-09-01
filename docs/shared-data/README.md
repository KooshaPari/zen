# Zen Shared Data Systems: Architecture Overview

## Goals
- Enable hundreds of agents across orgs/teams to securely share core data systems
- Provide scoped access using cryptographic identity and repo work_dir boundaries
- Centralize knowledge structures: project knowledge graph, RAG/CAG vector stores, LSP/QA services, hierarchical memory
- Maintain multi-tenancy, privacy, and least-privilege by default

## Scope at a Glance
- Identity & Scoping: Agent/Org/Team/User identities, OAuth2/mTLS keys, signed claims; work_dir enforced at tool boundary
- Shared Stores:
  - Knowledge Graph: project entities, code artifacts, APIs, tests, tasks, decisions
  - VectorDB RAG/CAG: code, knowledge, other corpora; unified semantic search
  - LSP/QA Service: code intelligence, diagnostics, test hints; shared cache
  - Hierarchical Memory: by agent, team, org, project; integrates with conversation + agent memory
- MCP Protocol Changes: All tools must accept and propagate work_dir; server enforces scope on resources

## Current Code Anchors (as of repo)
- Conversation memory: utils/conversation_memory.py (thread context, TTL, cross-tool)
- Agent memory: utils/agent_memory.py (typed memories, vector similarity)
- HTTP MCP server: server_mcp_http.py (sessions, tools/resources/prompts, future enforcement point)
- Core server: server.py (tools registry, call dispatcher)

## High-Level Architecture
1) Identity & Access
- Each agent has a cryptographic identity (OIDC/JWT or mTLS). Sessions bind: {agent_id, org_id, team_id, project_id}
- Authorization policy derives allowed resources: work_dir, resources, tools, data partitions

2) Scoping via work_dir
- Every tool request includes required work_dir (repo-relative path). Server validates:
  - work_dir is within repo, matches policy, and is permitted for agent/team
  - Non-overlapping partitions for backend/frontend; shared API interfaces allowed

3) Shared Data Planes
- Knowledge Graph (KG): Graph DB (ArangoDB primary; alternatives include Postgres+Apache AGE, Neo4j CE). Entities: CodeUnit, API, Test, Doc, Ticket, Decision, Build, Dep, Service
- Vector RAG/CAG: Dedicated service with namespaces per {org, project, repo, work_dir}; collections: code, knowledge, other
- LSP/QA: Shared LSP cluster + QA analysis; caches per commit/work_dir; emits diagnostics events
- Hierarchical Memory: Built atop agent_memory + conversation_memory with persistent backing (Redis/Postgres + vector index)

4) Event Bus & Audit
- All shared services publish audit and lineage events (Kafka/NATS compatible). Used for memory consolidation, observability, and test sims

## Data Partitioning & Multi-Tenancy
- Tenants: org -> project -> team -> agent
- Resource keys include tenant context and cryptographic subject
- Workload isolation via namespaces/collections/DB-per-tenant where required

## RAG/CAG Strategy
- Code: code embeddings (e.g., small/fast model), with tree-sitter chunks, repo+path metadata, line ranges, symbol IDs
- Knowledge: docs, decisions, ADRs, PRs, tickets; chunked with references into KG
- CAG (Cite-Answer-Grade): retrieval with grounding and self-grading; store graded passages and citations for reuse

## Knowledge Graph Strategy
- Importers: parser pipeline for code, API schemas, tests, PRs, issues, logs
- Relations: CALLS, DEFINES, TESTS, IMPLEMENTS, DEPENDS_ON, EXPOSES_API, MENTIONS, RESOLVES
- Supports cross-cutting queries (e.g., which tests cover this API?) and agent planning

## LSP/QA Shared Service
- Central LSP farm that indexes per-commit; agents query diagnostics/symbols/refactor actions via API
- QA layer runs linters/tests and provides hints; results cached

## Hierarchical Memory
- Layers: short-term, working, long-term, shared; plus procedural/episodic/semantic
- Index memories in vector store; link to KG nodes for grounding
- Access governed by identity and work_dir scope

## MCP Protocol & Tooling Changes (summary)
- All tools: require work_dir in arguments
- Server validates and annotates requests with identity and scope context
- Tools pass scope to providers/services (KG, Vector, LSP/QA, Memory)

## Security & Compliance (summary)
- Mutual auth (mTLS) or OAuth2; signed session tokens including tenant + scope
- Row/graph-level security; deny-by-default policies
- Comprehensive audit trail and data retention policies

## Implementation Phases (see roadmap doc)
1) Identity + work_dir enforcement + shared namespaces
2) Vector service (code/knowledge), minimal KG schema, LSP cache passthrough
3) Full KG ingestion + CAG + QA integrations
4) Hierarchical memory persistence + cross-agent workflows

## Testing & Validation (see testing doc)
- Unit, integration, simulator tests (communication_simulator_test.py)
- Access control tests per tenant/work_dir
- Performance/load tests for shared services

