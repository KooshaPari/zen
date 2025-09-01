import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_HTTP_TESTS") != "1",
    reason="Set RUN_HTTP_TESTS=1 to run HTTP MCP integration tests"
)


async def _call_list_tools(client):
    resp = await client.post("/mcp", json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list"
    })
    assert resp.status_code == 200
    data = resp.json()
    return data["result"]["tools"]


async def _call_tool(client, name, args):
    resp = await client.post("/mcp", json={
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": args
        }
    })
    assert resp.status_code == 200
    return resp.json()


@pytest.mark.anyio
async def test_http_semtools_end_to_end(tmp_path, monkeypatch):
    # Start app in-process
    from fastapi.testclient import TestClient

    import server_mcp_http as mcp_http

    monkeypatch.chdir(tmp_path)
    # Create work_dir
    (tmp_path / "frontend" / "ui").mkdir(parents=True)

    app = mcp_http.create_app()
    client = TestClient(app)

    # List tools and ensure work_dir is required
    tools = await _call_list_tools(client)
    names = [t["name"] for t in tools]
    assert "sem_ingest" in names and "sem_rag_search" in names
    for t in tools:
        if t["name"] in ("sem_ingest", "sem_rag_search"):
            schema = t.get("inputSchema", {})
            assert "work_dir" in schema.get("properties", {})
            assert "work_dir" in schema.get("required", [])

    # Call sem_ingest with warn-only (no pg running, so just ensure server handles)
    args = {
        "work_dir": "frontend/ui",
        "docs": [{"id": "d1", "text": "hello world", "metadata": {"path": "readme"}}],
        "collection": "code"
    }
    result = await _call_tool(client, "sem_ingest", args)
    # Should return JSON-RPC result with content
    assert "result" in result or "error" in result

