# Shared LSP/QA Service

## Goals
- Provide centralized code intelligence and quality analysis accessible to all agents with scoped access
- Reduce duplicated work by caching and sharing analysis per commit/work_dir

## LSP Architecture
- LSP cluster with language servers: typescript-language-server (TS/JS), pyright (Python), gopls (Go), rust-analyzer (Rust), clangd (C/C++), jdt.ls (Java)
- Indexing per tenant repo and commit; shards by work_dir
- API endpoints: symbols, references, definitions, diagnostics, code actions, rename, format
- Caching: key on {org, proj, repo, work_dir, commit_sha, file_path}

## QA Layer
- Aggregates linters, static analyzers, and test runners; prefer oxlint for JS/TS (faster than ESLint), Ruff for Python, golangci-lint for Go, cargo clippy for Rust, clang-tidy/cppcheck for C/C++, SpotBugs/Checkstyle for Java
- Exposes: lint_diagnostics, test_suggestions, coverage hints, complexity metrics
- Optional: run-on-demand tests for a path with sandboxed environment

## Access Control
- Requests include scope_context; service validates allowed work_dirs
- Cross-area access:
  - Backend vs Frontend: allow shared API specs; deny internals across boundaries
  - Elevated roles may request broader reads for refactor/impact analysis

## Data Flow
1) Agent requests diagnostics for a file in work_dir
2) LSP checks cache; if miss, analyze using language server
3) QA layer augments with lints/tests; store results and emit events
4) Response includes references and suggestions; IDs link to KG and vector chunks

## Integration Points
- server.py/server_mcp_http.py: enrich tool calls with scope_context; add endpoints for LSP/QA proxy
- tools: new tools lsp.symbols, lsp.references, qa.lint, qa.test_hints
- utils: cache helpers; metrics exporters

## Observability
- Metrics: analysis time, cache hit rate, error rates, queue depth
- Tracing across ingestion and retrieval
- Auditable events for accesses and code actions

## Roadmap
- Phase 1: Proxy to local LSPs with caching; minimal QA (linters)
- Phase 2: Distributed LSP pool; test hints from coverage
- Phase 3: On-demand tests; deep semantics and refactors

