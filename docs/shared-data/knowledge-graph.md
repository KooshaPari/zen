# Shared Project Knowledge Graph

## Purpose
- Provide a unified graph representation of the project across code, APIs, tests, docs, tasks, and decisions
- Enable cross-cutting queries to support agents' planning, impact analysis, and QA

## Graph Store Choice
- ArangoDB Community (primary): multi-model (documents + graphs) with AQL, supports DB-per-tenant in OSS
- Alternatives: Postgres + Apache AGE (single database stack with openCypher); Neo4j CE (mature, Cypher, but multi-DB limitations), TypeDB (rich schema/logic; higher complexity)
- Multi-tenant strategies: DB-per-tenant (preferred), schema/graph prefixing when needed

## Core Entities
- CodeUnit (file/module/class/function)
- API (endpoint, schema, RPC)
- Test (unit/integration/e2e)
- Service (backend/frontend/lib)
- Build (artifact, pipeline, status)
- Dependency (package, version)
- Doc (ADR, README, design)
- Ticket (issue/PR/task)
- Decision (ADR/outcome)

## Relationships
- CALLS, DEFINES, TESTS, IMPLEMENTS, DEPENDS_ON, EXPOSES_API, CONSUMES_API, MENTIONS, RESOLVES, PRODUCES, OWNED_BY

## Identifiers & Linking
- Stable IDs built from repo path + symbol signature + commit
- Link to Vector chunks by chunk_id
- Attach scope_context (org, proj, repo, work_dir) to nodes and edges

## Ingestion Sources
- Git (commits/diffs), AST parsers, OpenAPI/GraphQL schemas, CI pipelines, PR metadata
- Test frameworks to capture coverage and flakiness

## Query Examples
- Impact of changing function X? -> Affected tests, services, and API endpoints
- Which tests cover API Y? -> Test nodes via TESTS -> API
- Unused endpoints? -> EXPOSES_API without CONSUMES_API

## APIs
- Write: upsert_node, upsert_edge, batch ingest
- Read: cypher-like query API + canned queries for common tasks
- Security: enforce scope_context; deny cross-tenant traversals except shared nodes (e.g., third-party deps)

## Integration
- RAG: store KG refs in vector metadata and return back-references
- Planning tools: use KG for dependency-aware task ordering
- QA tools: fetch related tests and owners

## Roadmap
- Phase 1: Minimal schema (CodeUnit, API, TESTS/CALLS/DEPENDS_ON) on ArangoDB
- Phase 2: CI/test coverage, Decisions, Ticket linking; evaluate AGE adapter parity
- Phase 3: Ownership models, runtime signals, and design docs ingestion

