# Shared Vector RAG/CAG Service

## Objectives
- Provide a unified semantic retrieval plane for code, knowledge, and other data
- Support RAG (Retrieve-Answer-Ground) and CAG (Cite-Answer-Grade) workflows
- Partition by tenant and work_dir while enabling shared interfaces

## Store Options
- Primary vector DB: Postgres + pgvector (single database stack, simpler ops). Qdrant is optional for high-scale, vector-native features (named vectors, advanced payload filters, hybrid sparse+dense).
- Alt/local: Chroma or SQLite+Faiss for dev
- Embedding models: small-fast for code (e.g., all-MiniLM-L6-v2 or bge-small equivalents), larger for knowledge; pluggable via providers; optional via OpenRouter-supported providers when configured

## Namespaces & Collections
- Namespace key: `{org}/{proj}/{repo}/{work_dir}`
- Collections:
  - `code`: files, symbols, chunks, line ranges, language
  - `knowledge`: docs, ADRs, PRs, issues, decisions
  - `other`: data files, logs, configs

## Chunking & Metadata
- Code: tree-sitter structural chunks (function/class/module) + sliding window for context
- Knowledge: 1–2k tokens with overlap, preserve headings and references
- Metadata fields: path, symbol, lang, commit_sha, line_start/end, tags, scope_context, citations

## Ingestion Pipeline
1) Discover files inside allowed work_dirs
2) Extract symbols/AST via tree-sitter; compute hashes and commit boundaries
3) Chunk + embed concurrently; batch upserts with idempotency
4) Link chunks to KG nodes (when available) using stable IDs
5) Emit ingestion events (success/failure, counts, latency)

## Retrieval API
- Inputs: query, top_k, filters (namespace, collection, lang, tag, path prefix), work_dir (required)
- Response: items with score, text, metadata, citations, and KG references
- Hybrid retrieval: BM25 + vector; for pgvector, pair with an external BM25 (e.g., OpenSearch or Tantivy-based service) or SQL BM25 extensions; Qdrant’s built-in hybrid usable when Qdrant is selected
- Rerank optional for higher quality

## CAG Workflow
1) Retrieve top_k candidates
2) Ask model to answer with citations; enforce quote spans (path + line ranges)
3) Self-grade: verify factuality/coverage; record grade
4) Store graded answer + citations for reuse (per namespace)

## Operational Concerns
- TTL for stale commits; garbage collect orphaned chunks
- Backpressure and rate limits per tenant
- Observability: QPS, tail latencies, recall proxy metrics, cache hit-rate

## Integration Points in Repo
- tools to add: semtools.rag_search, semtools.ingest
- providers: model embeddings via providers/registry.py
- server enforcement: require work_dir and check allowed namespace before reads/writes

## Roadmap
- Phase 1: Dev-local vector DB + minimal ingestion + search
- Phase 2: Deploy pgvector-backed namespaces + external BM25; optionally add Qdrant for vector-native hybrid
- Phase 3: CAG persistence + KG linking + advanced filters

