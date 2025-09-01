import os

import pytest

pytestmark = pytest.mark.asyncio


async def _start_responder(session, base):
    async with session.post(f"{base}/a2a/test/rpc-responder/start", json={"task_id": "TCHAT"}) as r:
        text = await r.text()
        assert r.status == 200, text


async def _invoke_rpc(session, base):
    async with session.post(f"{base}/a2a/test/rpc", json={"task_id": "TCHAT", "method": "ping", "params": {"x": 1}, "timeout": 5}) as r:
        assert r.status == 200
        data = await r.json()
        assert data["result"]["ok"] is True


async def _advertise(session, base, agent_id="agent-alpha"):
    payload = {
        "agent_card": {
            "agent_id": agent_id,
            "name": "Alpha",
            "version": "1.0.0",
            "endpoint_url": base,
            "capabilities": [
                {"name": "chat", "description": "Chat capability", "category": "nlp", "input_schema": {}, "output_schema": {}}
            ],
            "last_seen": "2025-01-01T00:00:00Z",
        }
    }
    async with session.post(f"{base}/a2a/advertise", json=payload) as r:
        assert r.status == 200


async def _discover(session, base):
    async with session.post(f"{base}/a2a/discover", json={"capability_filter": "chat", "max_results": 10}) as r:
        assert r.status == 200
        data = await r.json()
        assert "agents" in data


async def _chat_blocking(mgr):
    # Use A2A manager directly for blocking chat
    res = await mgr.chat_send("agent-alpha", "hello", {"topic": "greeting"}, timeout_seconds=5)
    assert res and res.get("ok") is True
    assert res.get("echo") == "hello"


@pytest.mark.integration
async def test_a2a_chat_flow():
    base = os.environ.get("TEST_SERVER_BASE", "http://127.0.0.1:8080")

    import aiohttp
    async with aiohttp.ClientSession() as session:
        # Health check
        async with session.get(f"{base}/health") as r:
            assert r.status == 200

        # Start RPC responder for the chat task id (not required for chat, but validates RPC endpoint)
        await _start_responder(session, base)

        # Advertise and discover an agent with chat capability (optional for chat loopback)
        await _advertise(session, base)
        await _discover(session, base)

        # Use A2A manager to perform blocking chat to our own agent_id so the default handler replies
        os.environ["A2A_ENABLED"] = "1"
        os.environ["ZEN_EVENT_BUS"] = "nats"
        os.environ["NATS_SERVERS"] = "nats://127.0.0.1:4223"
        from utils.a2a_protocol import get_a2a_manager
        mgr = get_a2a_manager()
        await mgr._ensure_nats()
        # Override chat target to our own agent id
        res = await mgr.chat_send(mgr.agent_id, "hello", {"topic": "greeting"}, timeout_seconds=5)
        assert res and res.get("ok") is True and res.get("echo") == "hello"

