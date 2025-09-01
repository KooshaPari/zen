# Technology Choices (Final)

This document selects concrete technologies for the shared data systems and supporting infrastructure, with pragmatic, ops-friendly defaults and open-source first.

## Summary of Decisions
- Vector DB: Postgres + pgvector (primary), Qdrant optional for high-scale or complex filtering; Chroma (dev-only fallback)
- Knowledge Graph DB: ArangoDB Community (multi-model graph, DB-per-tenant)
- Event Bus: Redpanda (Kafka API-compatible), Kafka (managed) as alternative
- Identity & Auth: Keycloak (OIDC/OAuth2), mTLS for service-to-service
- Cache & Persistence: Redis (fast TTL/cache) + Postgres (durable metadata/state)
- Embeddings: Local bge-small (SentenceTransformers via TEI) by default; optional embeddings via OpenRouter-supported providers (e.g., OpenAI) when configured
- Reranking: bge-reranker-base (local via TEI); Cohere Rerank optional
- LSPs: Standard language servers (pyright for Python, typescript-language-server for TS/JS, gopls for Go, rust-analyzer for Rust, clangd for C/C++, jdt.ls for Java) orchestrated centrally
- QA: Ruff (Python), oxlint (JS/TS; faster alternative to ESLint), pytest + coverage; extendable with flake8, mypy, golangci-lint, cargo clippy, clang-tidy/cppcheck, SpotBugs/Checkstyle
- Observability: OpenTelemetry traces, Prometheus metrics, Grafana dashboards, Loki logs
- Orchestration: Docker Compose for dev; Kubernetes for production
- Job Orchestration: Temporal (durable, language-agnostic workflows)
- API Gateway & mTLS Termination: Envoy (with cert-manager)
- Object Storage (artifacts/cache): MinIO/S3-compatible
- Search (optional hybrid BM25/ANN): OpenSearch (KNN) with strict licensing review
- Schema Registry (events): Karapace (OSS alternative to Confluent)


## Rationale & Trade-offs
### Vector DB — Postgres + pgvector
- Pros: Single database stack, easy ops, transactional updates, good enough performance for small-to-medium scale, integrates with existing Postgres tooling
- Alternatives: Qdrant (vector-native, named vectors, strong payload filters, hybrid sparse+dense), Weaviate/Milvus (heavier ops)
- Tenancy: Schema/table-per-namespace or row-level filtering using `{org}/{proj}/{repo}/{work_dir}` convention

### Knowledge Graph — ArangoDB
- Pros: True multi-model (documents + graphs + key/value), strong AQL query language, OSS-friendly multi-database support, Foxx microservices when needed
- Tenancy: Database-per-tenant (preferred) or graph-level scoping; straightforward backups and restores per-tenant
- Alternatives: Neo4j (mature, Cypher, great tooling but multi-DB in OSS limited), TypeDB (powerful schema/logic; higher complexity)

### Event Bus — Redpanda
- Pros: Kafka-compatible API, single binary, low-latency and simpler ops for self-hosting; durable audit/event streams
- Alternatives: Managed Kafka (Confluent) for enterprise; NATS for lightweight pub/sub (less durable log semantics)

### Identity — Keycloak + mTLS
- Pros: OSS, realms/clients/groups, flexible claims; integrates with OIDC flows; supports multi-tenant patterns
- Service-to-service mTLS via Envoy/cert-manager or native TLS where supported

### Cache/Persistence — Redis + Postgres
- Redis for TTL, queues, ephemeral state; Postgres for durable metadata/state and transactional needs
- Well-supported in Python ecosystem and easy to operate locally and in K8s

### Embeddings & Reranking
- Default local stack via Text-Embeddings-Inference (TEI): `bge-small-en` for embeddings, `bge-reranker-base` for rerank
- Optional embeddings via OpenRouter-supported providers (e.g., OpenAI, Mistral, Google) when API keys present; Cohere Rerank optional
- Provides privacy & cost control with smooth upgrade path for quality

### LSP/QA
- Central LSP pool using standard language servers; cache by {org, proj, repo, work_dir, commit_sha, path}
- QA uses linters and test frameworks already common in repos; easy to extend toolchain

### Observability
- OTel for traces across server/tools/services; Prometheus + Grafana for metrics; Loki for logs
- Ensures we can benchmark recall/latency, detect regressions, and audit access

## Implementation Notes
- Dev environment: Docker Compose bringing up Postgres (pgvector), ArangoDB, Redis, Postgres (core), Redpanda, TEI services
- Production: K8s with helm charts/operators where available (Postgres/pgvector operators, ArangoDB, Redpanda, Bitnami Redis/Postgres)
- Security: Keycloak for token issuance; sidecar/ingress for mTLS; secrets via Vault/SealedSecrets

## Mapping to Architecture
- Vector RAG/CAG: Postgres + pgvector tables/partitions per tenant/work_dir; hybrid retrieval via external BM25 (e.g., OpenSearch) or SQL BM25 extensions; optional rerank
- Knowledge Graph: ArangoDB graphs with scope attributes and query guards; link vector chunk IDs
- LSP/QA: Central services with commit/work_dir caches, guarded by scope_context
- Qdrant trade-offs: Excellent for high-scale vector-native features (payload filters, named vectors, hybrid). For simpler ops and single-DB footprints, pgvector keeps complexity low. Choose Qdrant when multi-tenant RAG grows beyond Postgres constraints.

- Hierarchical Memory: Redis+Postgres persistence with vector index in pgvector; scope-aware keys

## Risks & Mitigations
- Multi-tenancy in Neo4j Community: mitigate with strict query guards + automated tests; upgrade path to Enterprise/Aura if needed
- Local model quality: allow optional OpenAI/Cohere for quality-sensitive tasks
- Ops complexity: provide reference Compose and Helm manifests; observability from day one

## Next Steps
- Add compose stack to bootstrap local services
- Implement Phase 1 enforcement and tests using these choices
- Instrument observability and audit events across services

