"""
Comprehensive indexer for pgvector + OpenSearch hybrid.

- Creates/updates OpenSearch indices per namespace/collection
- Indexes documents (id, text, metadata)
- Provides sync_index(ns, collection) helpers to mirror pgvector -> OpenSearch

This is a synchronous baseline; production should add retries, backoff, and bulk.
"""
from __future__ import annotations

import json
import os
from typing import Any

from tools.semtools import _connect, ensure_schema, namespace_from_scope
from tools.semtools_bm25 import get_os_client
from tools.semtools_meili import ensure_index as meili_ensure_index
from tools.semtools_meili import index_documents as meili_index_documents


def os_index_name(namespace: str, collection: str) -> str:
    safe_ns = namespace.replace("/", "-").replace("_", "-")
    return f"zen-{safe_ns}-{collection}"


def ensure_os_index(index: str):
    client = get_os_client()
    if not client.indices.exists(index=index):
        client.indices.create(index=index, body={
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "metadata": {"type": "object", "enabled": True}
                }
            }
        })


def index_documents(namespace: str, collection: str, docs: list[dict[str, Any]]):
    backend = os.getenv("BM25_BACKEND", os.getenv("SEARCH_BACKEND", "postgres")).lower()
    index = os_index_name(namespace, collection)
    if backend == "meili":
        meili_ensure_index(index)
        meili_index_documents(index, docs)
        return
    if backend == "opensearch":
        client = get_os_client()
        ensure_os_index(index)
        for d in docs:
            _id = d.get("id")
            body = {"text": d.get("text", ""), "metadata": d.get("metadata", {})}
            client.index(index=index, id=_id, body=body, refresh=True)
        return
    # postgres backend: no external indexing required (use FTS at query time)
    return


def sync_index(namespace: str, collection: str = "code", limit: int = 1000):
    ensure_schema()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, text, metadata FROM rag_chunks WHERE namespace=%s AND collection=%s LIMIT %s",
            (namespace, collection, limit)
        )
        rows = cur.fetchall()
        docs = []
        for r in rows:
            _id, text, meta_json = r
            try:
                meta = json.loads(meta_json) if isinstance(meta_json, str) else meta_json
            except Exception:
                meta = {}
            docs.append({"id": _id, "text": text, "metadata": meta})
        index_documents(namespace, collection, docs)


def index_ingested_docs(docs: list[dict[str, Any]], scope: dict[str, Any], collection: str = "code"):
    ns = namespace_from_scope(scope)
    index_documents(ns, collection, docs)
