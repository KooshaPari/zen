import json
import logging
import os
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# Minimal scaffolding for pgvector-based ingest and RAG search.
# NOTE: This is a planning stub. Real implementation should handle pooling, retries, and SQL injection safety.

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover
    psycopg = None
    dict_row = None


def _get_pg_dsn() -> str:
    return os.getenv("ZEN_PG_DSN", "postgresql://postgres:postgres@localhost:5432/zen")


def _connect():
    if not psycopg:
        raise RuntimeError("psycopg is required for pgvector semtools. Install psycopg[binary].")
    return psycopg.connect(_get_pg_dsn())


def _get_vector_dim() -> int:
    try:
        return int(os.getenv("RAG_VECTOR_DIM", "384"))
    except Exception:
        return 384


def ensure_schema():
    """Create minimal tables for namespaces and vectors if not exists."""
    with _connect() as conn, conn.cursor() as cur:
        # Ensure required extensions (pgvector for vector, pg_trgm if we add trigram later)
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception:
            # Extension may not be installed on standard images
            pass
        dim = _get_vector_dim()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_chunks (
              id TEXT PRIMARY KEY,
              namespace TEXT NOT NULL,
              collection TEXT NOT NULL,
              text TEXT NOT NULL,
              metadata JSONB NOT NULL,
              embedding VECTOR(%s) -- set via RAG_VECTOR_DIM
            )
            """,
            (dim,)
        )
        # Create FTS index for BM25-like ranking when using Postgres backend
        try:
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_fts
                ON rag_chunks
                USING GIN (to_tsvector('english', text));
                """
            )
        except Exception:
            pass
        conn.commit()


def upsert_chunk(id: str, namespace: str, collection: str, text: str, metadata: dict[str, Any], embedding: list[float]):
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO rag_chunks (id, namespace, collection, text, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
              namespace = EXCLUDED.namespace,
              collection = EXCLUDED.collection,
              text = EXCLUDED.text,
              metadata = EXCLUDED.metadata,
              embedding = EXCLUDED.embedding
            """,
            (id, namespace, collection, text, json.dumps(metadata), embedding)
        )
        conn.commit()


def search(namespace: str, collection: str, query_emb: list[float], top_k: int = 8, filters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    filters = filters or {}
    where = ["namespace = %s", "collection = %s"]
    params: list[Any] = [namespace, collection]

    # Extremely simple payload filter support (exact match); extend for ranges/arrays
    for key, val in filters.items():
        where.append("metadata ->> %s = %s")
        params.extend([key, str(val)])

    where_sql = " AND ".join(where)
    sql = f"""
        SELECT id, text, metadata, 1 - (embedding <=> %s::vector) AS score
        FROM rag_chunks
        WHERE {where_sql}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    with _connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        params2 = params.copy()
        params2.insert(0, query_emb)
        params2.append(query_emb)
        params2.append(top_k)
        cur.execute(sql, params2)
        rows = cur.fetchall()
        return rows




def _fts_search_postgres(namespace: str, collection: str, query: str, top_k: int = 20, filters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    filters = filters or {}
    where = ["namespace = %s", "collection = %s", "to_tsvector('english', text) @@ plainto_tsquery('english', %s)"]
    params: list[Any] = [namespace, collection, query]
    for key, val in filters.items():
        where.append("metadata ->> %s = %s")
        params.extend([key, str(val)])
    where_sql = " AND ".join(where)
    sql = f"""
        SELECT id, text, metadata,
               ts_rank_cd(to_tsvector('english', text), plainto_tsquery('english', %s)) AS score
        FROM rag_chunks
        WHERE {where_sql}
        ORDER BY score DESC
        LIMIT %s
    """
    with _connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        params2 = [query] + params + [top_k]
        cur.execute(sql, params2)
        rows = cur.fetchall()
        return rows


def _rrf_fuse(dense: list[dict[str, Any]], sparse: list[dict[str, Any]], k: int = 60, top_k: int = 10) -> list[dict[str, Any]]:
    def to_rank_map(items: list[dict[str, Any]]):
        return {it["id"]: idx for idx, it in enumerate(items)}
    dmap = to_rank_map(dense)
    smap = to_rank_map(sparse)
    ids = set(dmap.keys()) | set(smap.keys())
    scored = []
    for _id in ids:
        rd = dmap.get(_id)
        rs = smap.get(_id)
        s = 0.0
        if rd is not None:
            s += 1.0 / (k + rd + 1)
        if rs is not None:
            s += 1.0 / (k + rs + 1)
        payload = next((it for it in dense if it["id"] == _id), None) or next((it for it in sparse if it["id"] == _id), None)
        if payload:
            scored.append({"id": _id, "score": s, "text": payload.get("text"), "metadata": payload.get("metadata", {})})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

def _embed_texts_tei(texts: list[str], tei_url: str) -> list[list[float]]:
    resp = requests.post(f"{tei_url.rstrip('/')}/embed", json={"inputs": texts})
    resp.raise_for_status()
    data = resp.json()
    return data


def _embed_texts_ollama(texts: list[str], base_url: str, model: str) -> list[list[float]]:
    base = base_url.rstrip("/")
    out: list[list[float]] = []
    for t in texts:
        resp = requests.post(
            f"{base}/api/embeddings",
            json={"model": model, "prompt": t},
            timeout=60,
        )
        resp.raise_for_status()
        emb = resp.json().get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError("Invalid embedding response from Ollama")
        out.append(emb)
    return out


def _embed_texts_openrouter(texts: list[str], api_key: str, model: str, base_url: str = "https://openrouter.ai/api/v1") -> list[list[float]]:
    url = f"{base_url.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # OpenRouter recommends these headers; optional here
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "Zen MCP"),
    }
    # OpenAI-compatible schema supports batching via input list
    payload = {"model": model, "input": texts}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    j = resp.json()
    data = j.get("data") or []
    # Each item has { index, embedding }
    out = []
    for item in data:
        emb = item.get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError("Invalid embedding from OpenRouter")
        out.append(emb)
    if len(out) != len(texts):
        raise RuntimeError("Mismatched embeddings count from OpenRouter")
    return out


def embed_texts(texts: list[str], tei_url: str | None = None) -> list[list[float]]:
    """Embed texts using configured provider.

    Selection order:
    - EMBEDDINGS_PROVIDER=ollama → OLLAMA_URL + OLLAMA_EMBED_MODEL
    - EMBEDDINGS_PROVIDER=openrouter → OPENROUTER_API_KEY + EMBEDDINGS_MODEL_OPENROUTER
    - default/tei → TEI_URL or provided tei_url
    """
    provider = (os.getenv("EMBEDDINGS_PROVIDER", "tei").strip().lower())

    if provider == "ollama":
        base = os.getenv("OLLAMA_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        return _embed_texts_ollama(texts, base, model)

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY required for openrouter embeddings")
        model = os.getenv("EMBEDDINGS_MODEL_OPENROUTER", "text-embedding-3-small")
        base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return _embed_texts_openrouter(texts, api_key, model, base)

    # default: TEI-compatible endpoint
    url = tei_url or os.getenv("TEI_URL", "http://localhost:8082")
    return _embed_texts_tei(texts, url)


def namespace_from_scope(scope: dict[str, Any]) -> str:
    parts = [scope.get(k) for k in ("org", "proj", "repo", "work_dir") if scope.get(k)]
    return "/".join(parts) if parts else "default"


def ingest_documents(docs: list[tuple[str, str, dict[str, Any]]], scope: dict[str, Any], collection: str = "code", tei_url: str = "http://localhost:8082", index_bm25: bool = True):
    """
    docs: list of (id, text, metadata)
    """
    ensure_schema()
    texts = [t for _, t, _ in docs]
    embs = embed_texts(texts, tei_url)
    ns = namespace_from_scope(scope)
    for (doc_id, text, meta), emb in zip(docs, embs):
        upsert_chunk(doc_id, ns, collection, text, meta, emb)

    if index_bm25:
        try:
            from tools.semtools_indexer import index_ingested_docs
            doc_objs = [{"id": i, "text": t, "metadata": m} for (i, t, m) in docs]
            index_ingested_docs(doc_objs, scope, collection=collection)
        except Exception:
            # OpenSearch not available; skip
            pass


def rag_search(query: str, scope: dict[str, Any], collection: str = "code", tei_url: str = "http://localhost:8082", top_k: int = 8, rerank: bool = False) -> list[dict[str, Any]]:
    # Dense
    qvec = embed_texts([query], tei_url)[0]
    ns = namespace_from_scope(scope)
    dense = search(ns, collection, qvec, top_k=top_k*2)

    # Sparse (optional)
    backend = os.getenv("BM25_BACKEND", os.getenv("SEARCH_BACKEND", "postgres")).lower()
    sparse: list[dict[str, Any]] = []
    try:
        if backend == "postgres":
            sparse = _fts_search_postgres(ns, collection, query, top_k=top_k*2)
        elif backend == "meili":
            from tools.semtools_indexer import os_index_name
            url = os.getenv("MEILI_URL", "http://localhost:7700").rstrip("/")
            key = os.getenv("MEILI_API_KEY") or os.getenv("MEILI_MASTER_KEY") or "dev-master-key"
            index = os_index_name(ns, collection)
            resp = requests.post(f"{url}/indexes/{index}/search", headers={"X-Meili-API-Key": key}, json={"q": query, "limit": top_k*2}, timeout=15)
            if resp.status_code == 200:
                hits = resp.json().get("hits", [])
                sparse = [{"id": h.get("id"), "text": h.get("text"), "metadata": h.get("metadata", {})} for h in hits]
    except Exception:
        sparse = []

    if sparse:
        return _rrf_fuse(dense, sparse, top_k=top_k)
    return dense[:top_k]
