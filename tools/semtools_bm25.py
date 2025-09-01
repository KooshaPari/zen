"""
Optional BM25 pairing (OpenSearch) for hybrid retrieval with pgvector.

This module provides a thin client for OpenSearch and hybrid fusion utilities
(e.g., RRF) to combine pgvector dense results with BM25 sparse results.

Guard usage behind RUN_OS_TESTS or config flags in production.
"""
from __future__ import annotations

import os
from typing import Any

try:
    from opensearchpy import OpenSearch
except Exception:  # pragma: no cover
    OpenSearch = None


def get_os_client() -> OpenSearch:
    if not OpenSearch:
        raise RuntimeError("opensearch-py is required for hybrid BM25 support")
    host = os.getenv("OS_HOST", "localhost")
    port = int(os.getenv("OS_PORT", "9200"))
    return OpenSearch(hosts=[{"host": host, "port": port}])


def bm25_search(index: str, query: str, top_k: int = 20, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    client = get_os_client()
    musts = [{"multi_match": {"query": query, "fields": ["text^2", "metadata^1"]}}]
    # Very simple filters (term); extend as needed
    if filters:
        for k, v in filters.items():
            musts.append({"term": {f"metadata.{k}": v}})
    body = {"query": {"bool": {"must": musts}}}
    resp = client.search(index=index, body=body, size=top_k)
    hits = resp.get("hits", {}).get("hits", [])
    out = []
    for h in hits:
        src = h.get("_source", {})
        out.append({
            "id": h.get("_id"),
            "score": h.get("_score"),
            "text": src.get("text"),
            "metadata": src.get("metadata", {})
        })
    return out


def rrf_fuse(dense: list[dict[str, Any]], sparse: list[dict[str, Any]], k: int = 60, top_k: int = 10) -> list[dict[str, Any]]:
    # Reciprocal Rank Fusion
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
        # pick a representative payload (prefer dense)
        payload = next((it for it in dense if it["id"] == _id), None) or next((it for it in sparse if it["id"] == _id), None)
        if payload:
            scored.append({"id": _id, "score": s, "text": payload.get("text"), "metadata": payload.get("metadata", {})})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

