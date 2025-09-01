import pytest
from aiohttp.test_utils import TestClient, TestServer
import pytest as _pytest

# Mark as integration to avoid running in unit-only mode
_pytestmark = _pytest.mark.integration
from server_http import build_app


@pytest.mark.asyncio
async def test_models_catalog_endpoint(monkeypatch):
    app = build_app()
    server = TestServer(app)
    client = TestClient(server)
    async with server, client:
        resp = await client.get("/models/catalog")
        assert resp.status == 200
        data = await resp.json()

        assert "models" in data
        assert isinstance(data["models"], list)

        # Basic shape: entries with key/model_id or name
        if data["models"]:
            entry = data["models"][0]
            assert isinstance(entry, dict)
            assert any(k in entry for k in ("key", "model_id", "name"))
