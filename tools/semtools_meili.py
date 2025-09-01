"""
Meilisearch BM25/lightweight search backend for semtools.

Provides minimal index creation and document indexing. Searching is handled from
semtools via direct HTTP calls when BM25_BACKEND=meili.
"""
from __future__ import annotations

import os
from typing import Any

import requests


def _base() -> tuple[str, dict[str, str]]:
    url = os.getenv("MEILI_URL", "http://localhost:7700").rstrip("/")
    key = os.getenv("MEILI_API_KEY") or os.getenv("MEILI_MASTER_KEY") or "dev-master-key"
    headers = {"X-Meili-API-Key": key}
    return url, headers


def ensure_index(index: str) -> None:
    url, headers = _base()
    r = requests.get(f"{url}/indexes/{index}", headers=headers, timeout=10)
    if r.status_code == 200:
        return
    # Create index with primary key 'id'
    resp = requests.post(f"{url}/indexes", headers=headers, json={"uid": index, "primaryKey": "id"}, timeout=10)
    resp.raise_for_status()


def index_documents(index: str, docs: list[dict[str, Any]]) -> None:
    url, headers = _base()
    # Ensure index exists
    ensure_index(index)
    # Add docs
    payload = []
    for d in docs:
        payload.append({
            "id": d.get("id"),
            "text": d.get("text", ""),
            "metadata": d.get("metadata", {}),
        })
    resp = requests.post(f"{url}/indexes/{index}/documents", headers=headers, json=payload, timeout=30)
    resp.raise_for_status()

