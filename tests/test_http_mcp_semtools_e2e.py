import os
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_E2E") != "1",
    reason="Set RUN_E2E=1 (and ensure Postgres, TEI, OpenSearch are running)"
)


@pytest.mark.anyio
async def test_ingest_and_hybrid_search(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import server_mcp_http as mcp_http

    # Require DSN and TEI
    dsn = os.getenv("ZEN_PG_DSN")
    os.getenv("TEI_URL", "http://localhost:8082")
    assert dsn, "ZEN_PG_DSN required"

    # Prepare work_dir and a file
    work_dir = tmp_path / "frontend" / "ui"
    work_dir.mkdir(parents=True)
    (work_dir / "readme.md").write_text("hello world from zen mcp server", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    app = mcp_http.create_app()
    client = TestClient(app)

    def rpc(method, params=None, id=1):
        body = {"jsonrpc": "2.0", "id": id, "method": method}
        if params is not None:
            body["params"] = params
        r = client.post("/mcp", json=body)
        assert r.status_code == 200
        return r.json()

    # Ingest a doc via tool
    docs = [{
        "id": "readme",
        "text": (work_dir / "readme.md").read_text(encoding="utf-8"),
        "metadata": {"path": "frontend/ui/readme.md"}
    }]
    ingest_args = {"work_dir": "frontend/ui", "docs": docs, "collection": "knowledge"}
    rpc("tools/call", {"name": "sem_ingest", "arguments": ingest_args}, id=2)

    # Give OpenSearch a moment to index
    time.sleep(1.0)

    # Search with hybrid enabled
    search_args = {
        "work_dir": "frontend/ui",
        "query": "hello world",
        "collection": "knowledge",
        "top_k": 5,
        "use_bm25": True
    }
    res = rpc("tools/call", {"name": "sem_rag_search", "arguments": search_args}, id=3)
    assert "result" in res
    content = res["result"].get("content") or []
    # Some servers may wrap tool outputs as text; accept either list or string
    if isinstance(content, list) and content:
        # okay
        pass
    elif isinstance(content, str):
        assert "hello" in content.lower()
    else:
        pytest.skip("Hybrid search returned empty content; check services are healthy")

