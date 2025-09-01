import pytest
import pytest as _pytest
from aiohttp.test_utils import TestClient

# Mark as integration to avoid unit-only runs
_pytestmark = _pytest.mark.integration


@pytest.mark.asyncio
async def test_http_tasks_csv_empty_list(aiohttp_client):
    from server_http import build_app
    app = build_app()
    client: TestClient = await aiohttp_client(app)

    resp = await client.get('/tasks.csv')
    assert resp.status == 200
    assert 'text/csv' in resp.headers.get('Content-Type', '')
    etag = resp.headers.get('ETag')
    text = await resp.text()
    # Expect header row present even when no tasks
    assert 'task_id' in text.splitlines()[0]

    # ETag cache validation
    resp2 = await client.get('/tasks.csv', headers={'If-None-Match': etag})
    assert resp2.status == 304
