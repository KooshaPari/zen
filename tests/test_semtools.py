import os

import pytest

PG_DSN = os.getenv("ZEN_PG_DSN", "postgresql://postgres:postgres@localhost:5432/zen")

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_PG_TESTS") != "1",
    reason="Set RUN_PG_TESTS=1 to run pgvector semtools tests"
)


def test_semtools_ingest_and_search(monkeypatch):
    # Lazy import inside test to avoid hard dependency when skipped
    from tools.semtools import ensure_schema, ingest_documents, rag_search

    scope = {"org": "acme", "proj": "zen", "repo": "zen-mcp-server", "work_dir": "."}
    docs = [
        ("doc1", "hello world", {"path": "README.md"}),
        ("doc2", "vector search test", {"path": "docs/plan.md"}),
    ]

    ensure_schema()
    ingest_documents([(i, t, m) for i, t, m in docs], scope)

    res = rag_search("hello", scope, top_k=3)
    assert isinstance(res, list)

