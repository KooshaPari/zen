import asyncio
import os

import pytest

pytestmark = pytest.mark.asyncio


@pytest.mark.integration
async def test_a2a_reply_routed_over_out_channel():
    # Enable NATS
    os.environ["A2A_ENABLED"] = "1"
    os.environ["ZEN_EVENT_BUS"] = "nats"
    os.environ["NATS_SERVERS"] = "nats://127.0.0.1:4223"

    from utils.a2a_protocol import get_a2a_manager
    from utils.nats_communicator import get_nats_communicator

    mgr = get_a2a_manager()
    await mgr._ensure_nats()
    comm = await get_nats_communicator(None)

    got = asyncio.Event()
    received = {}

    # Subscribe to this agent's out channel to capture chat responses
    subj = f"a2a.agent.{mgr.agent_id}.out"

    async def handler(data):
        # Expect a chat_response with correlation_id and payload.ok True
        if data.get("message_type") == "chat_response" and data.get("correlation_id"):
            payload = data.get("payload", {})
            if payload.get("ok"):
                received.update(payload)
                got.set()

    ok = await comm.subscribe(subj, handler, use_jetstream=True, durable_name=f"out-{mgr.agent_id[:6]}")
    if not ok:
        ok = await comm.subscribe(subj, handler, use_jetstream=False)
        assert ok

    # Send chat to self with blocking to ensure a response is produced
    res = await mgr.chat_send(mgr.agent_id, "route-check", {"test": True}, timeout_seconds=5)
    assert res and res.get("ok") is True

    # Wait for publish on out channel
    try:
        await asyncio.wait_for(got.wait(), timeout=5)
    except asyncio.TimeoutError:
        pytest.fail("Did not observe response on out channel")

    assert received.get("echo") == "route-check"

