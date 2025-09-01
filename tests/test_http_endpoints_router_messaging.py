import pytest
import pytest as _pytest
from aiohttp.test_utils import TestClient

# Mark as integration to avoid unit-only runs
_pytestmark = _pytest.mark.integration


@pytest.mark.asyncio
async def test_http_router_decide_endpoint(aiohttp_client):
    from server_http import build_app
    app = build_app()
    client: TestClient = await aiohttp_client(app)

    resp = await client.post('/router/decide', json={"message": "route this", "task_type": "quick_qa"})
    assert resp.status == 200
    data = await resp.json()
    assert 'decision' in data
    assert data['decision'].get('chosen_model')
    assert isinstance(data['decision'].get('budgets'), dict)
    assert isinstance(data['decision'].get('plan'), dict)


@pytest.mark.asyncio
async def test_http_messages_resume_flow(aiohttp_client, monkeypatch):
    # Force memory storage
    monkeypatch.setenv('ZEN_STORAGE_MODE', 'memory')
    from server_http import build_app
    app = build_app()
    client: TestClient = await aiohttp_client(app)

    # Create channel
    resp = await client.post('/channels', json={"project_id": "proj:test", "name": "general", "created_by": "tester"})
    assert resp.status == 201
    ch = (await resp.json())["channel"]
    cid = ch["id"]

    # Post blocking message mentioning alice
    payload = {"channel_id": cid, "from": "system", "body": "Please review", "mentions": ["alice"], "blocking": True}
    resp = await client.post('/messages/channel', json=payload)
    assert resp.status == 201
    msg = (await resp.json())["message"]
    mid = msg["id"]

    # Unread before
    resp = await client.get('/inbox/messages', params={"agent_id": "alice"})
    assert resp.status == 200
    unread_before = (await resp.json())["unread"]["total"]
    assert unread_before >= 1

    # Resume
    resp = await client.post('/messages/resume', json={"message_id": mid, "agent_id": "alice"})
    assert resp.status == 200
    data = await resp.json()
    assert data.get("ok") is True
    assert data.get("message", {}).get("resolved") in (True, "True")

    # Unread after
    resp = await client.get('/inbox/messages', params={"agent_id": "alice"})
    assert resp.status == 200
    unread_after = (await resp.json())["unread"]["total"]
    assert unread_after <= unread_before - 1


@pytest.mark.asyncio
async def test_http_tasks_llm_stream_router_included(aiohttp_client, monkeypatch):
    # Enable router auto-select
    monkeypatch.setenv('ZEN_ROUTER_ENABLE', '1')
    # Ensure memory storage to avoid Redis dependency
    monkeypatch.setenv('ZEN_STORAGE_MODE', 'memory')

    # Patch provider registry to avoid real API keys
    from providers.base import ProviderType
    from providers.registry import ModelProviderRegistry

    class DummyProvider:
        def get_provider_type(self):
            return ProviderType.CUSTOM

    monkeypatch.setattr(ModelProviderRegistry, 'get_provider_for_model', classmethod(lambda cls, m: DummyProvider()))

    from server_http import build_app
    app = build_app()
    client: TestClient = await aiohttp_client(app)

    # Create LLM task in streaming mode with model auto-select
    resp = await client.post('/tasks', json={
        'agent_type': 'llm',
        'message': 'Route me please',
        'model': 'auto',
        'stream_mode': True
    })
    assert resp.status in (200, 202)
    data = await resp.json()
    # Router decision should be included in the response body for streaming mode
    assert 'router' in data and isinstance(data['router'], dict)
    assert data['router'].get('chosen_model')


@pytest.mark.asyncio
async def test_http_a2a_acl_denied(aiohttp_client, monkeypatch):
    # Allow only a specific sender
    monkeypatch.setenv('A2A_ALLOWED_SENDERS', 'allowed-agent')
    from server_http import build_app
    app = build_app()
    client: TestClient = await aiohttp_client(app)

    import datetime
    now = datetime.datetime.utcnow().isoformat() + 'Z'
    msg = {
        'message_id': 'm1',
        'sender_id': 'blocked-agent',
        'receiver_id': None,
        'message_type': 'chat_request',
        'timestamp': now,
        'payload': {'text': 'hi'},
        'ttl_seconds': 3600,
        'priority': 5
    }
    resp = await client.post('/a2a/message', json=msg)
    assert resp.status == 200
    body = await resp.json()
    assert body.get('message_type') == 'error'
    assert body.get('payload', {}).get('error') == 'forbidden'
    assert body.get('payload', {}).get('reason') == 'sender_not_allowed'


@pytest.mark.asyncio
async def test_http_a2a_type_acl_denied(aiohttp_client, monkeypatch):
    # Only allow 'chat_request', deny 'task_request'
    monkeypatch.setenv('A2A_ALLOWED_TYPES', 'chat_request')
    from server_http import build_app
    app = build_app()
    client: TestClient = await aiohttp_client(app)

    import datetime
    now = datetime.datetime.utcnow().isoformat() + 'Z'
    msg = {
        'message_id': 'm2',
        'sender_id': 'agent-x',
        'receiver_id': None,
        'message_type': 'task_request',
        'timestamp': now,
        'payload': {'text': 'hi'},
        'ttl_seconds': 3600,
        'priority': 5
    }
    resp = await client.post('/a2a/message', json=msg)
    assert resp.status == 200
    body = await resp.json()
    assert body.get('message_type') == 'error'
    assert body.get('payload', {}).get('error') == 'forbidden'
    assert body.get('payload', {}).get('reason') == 'type_not_allowed'


@pytest.mark.asyncio
async def test_http_threads_resume_by_token(aiohttp_client, monkeypatch):
    monkeypatch.setenv('ZEN_STORAGE_MODE', 'memory')
    from server_http import build_app
    app = build_app()
    client: TestClient = await aiohttp_client(app)

    # Create channel
    resp = await client.post('/channels', json={"project_id": "proj:t3", "name": "general", "created_by": "tester"})
    assert resp.status == 201
    ch = (await resp.json())["channel"]
    cid = ch["id"]

    # Post blocking message and capture resume_token
    resp = await client.post('/messages/channel', json={"channel_id": cid, "from": "system", "body": "Block", "mentions": ["carol"], "blocking": True})
    assert resp.status == 201
    msg = (await resp.json())["message"]
    token = msg.get('resume_token')
    assert token

    # Resolve by token
    resp = await client.post('/threads/resume', json={"resume_token": token, "from": "carol", "reply_body": "ack"})
    assert resp.status == 200
    j = await resp.json()
    assert j.get('ok') is True
    assert 'reply_id' in j
