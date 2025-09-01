# Implementation Roadmap

## Phase 0 – Foundations (Week 0)
- Align stakeholders, finalize data plane choices (Qdrant/Weaviate, Neo4j, Redis/Postgres)
- Decide auth mode (OIDC vs mTLS) and token schema
- Define policy model (RBAC/ABAC) and initial rules

## Phase 1 – Identity & Scope Enforcement (Weeks 1–2)
- Require work_dir in all MCP tool calls (warn-only)
- Inject scope_context on server; extend base_tool arg contract
- Add session store and claims parsing; log audit events
- Unit/integration tests for scope and path guards

## Phase 2 – Vector Service (Weeks 2–4)
- Stand up dev vector DB; implement ingest and search tools
- Embed providers via providers/registry.py; chunkers for code/docs
- Namespaces by tenant/work_dir; hybrid search
- Simulator tests for RAG and scope

## Phase 3 – Knowledge Graph (Weeks 4–6)
- Minimal schema (CodeUnit, API, TESTS/CALLS/DEPENDS_ON)
- Ingest from repo + OpenAPI; link to vector chunks
- Canned queries and API

## Phase 4 – LSP/QA (Weeks 6–8)
- Proxy to local LSPs with commit/work_dir cache
- QA linters and test hints; emit events

## Phase 5 – Hierarchical Memory (Weeks 8–10)
- Add scope_context to agent_memory & conversation_memory
- Persist long-term/shared memories; vector index for retrieval

## Phase 6 – CAG & Cross-Service Refinements (Weeks 10–12)
- Add citation enforcement and grading persistence
- Integrate KG in RAG results; enrich planning tools

## Phase 7 – Hard Enforcement & SLOs (Weeks 12–13)
- Switch to hard enforcement for work_dir
- Quotas, rate limits, and dashboards

## Deliverables
- Docs in docs/shared-data/*
- Tools: semtools.rag_search, semtools.ingest, lsp.*, qa.*
- Config and policy examples
- Test suites and simulator scenarios

