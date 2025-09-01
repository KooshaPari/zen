# Testing and Validation Plan

## Principles
- Shift-left access control and scope validation
- Deterministic and reproducible ingestion/retrieval
- Performance baselines with regression alarms

## Unit Tests
- Scope guards: work_dir path canonicalization, allowed prefixes
- Memory APIs: scope_context propagation, TTLs, vector retrieval filters
- Vector ingestion: chunking correctness, idempotent upserts
- KG upserts and common queries

## Integration Tests
- HTTP MCP: session auth, work_dir enforcement, error formats
- RAG search over namespaces with filters
- LSP diagnostics cache per commit/work_dir
- QA lint hints surfaced via tools

## Simulator Tests (extend communication_simulator_test.py)
- cross_tool_continuation with scope_context
- token_allocation_validation with scoped resources
- planner_validation using KG relationships
- codereview_validation retrieving from vector service

## Load/Perf Tests
- Vector QPS, p95/p99 latencies, recall proxy (golden queries)
- LSP cache hit-rates, analysis latency per language
- Memory store throughput and eviction behavior

## Security Tests
- Adversarial: cross-work_dir access attempts -> denied
- Prompt injection resilience with CAG citation checks
- Token rotation and session invalidation flows

## Tooling
- code_quality_checks.sh remains the gate
- Add scripts: run_shared_services.sh (dev), seed_vector_kg.sh (dev data)

## Release Criteria
- All tests green; SLOs met for p95 latencies
- Audit logs verified; no scope violations in logs
- Backward compatibility warnings resolved or documented

