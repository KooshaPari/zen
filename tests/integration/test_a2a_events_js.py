import asyncio
import os
import uuid

import pytest

pytestmark = pytest.mark.asyncio


@pytest.mark.integration
async def test_a2a_task_events_js_or_fallback():
    # Ensure NATS setup
    os.environ["A2A_ENABLED"] = "1"
    os.environ["ZEN_EVENT_BUS"] = "nats"
    os.environ["NATS_SERVERS"] = "nats://127.0.0.1:4223"

    from utils.nats_communicator import get_nats_communicator

    comm = await get_nats_communicator(None)
    assert comm.connected is True

    subj = f"a2a.task.{uuid.uuid4().hex[:8]}.events"
    got = asyncio.Event()
    received = {}

    async def handler(data):
        received.update(data)
        got.set()

    # Try JetStream durable first, fallback to core
    js_ok = await comm.subscribe(subj, handler, use_jetstream=True, durable_name=f"durable-{uuid.uuid4().hex[:6]}")
    if not js_ok:
        # Fall back to core NATS subscription
        core_ok = await comm.subscribe(subj, handler, use_jetstream=False)
        assert core_ok is True

    # Publish depending on subscription capability
    payload = {"event": "demo", "value": 42}
    ok = await comm.publish(subj, payload, use_jetstream=bool(js_ok))
    assert ok is True

    # Wait for receipt
    try:
        await asyncio.wait_for(got.wait(), timeout=5)
    except asyncio.TimeoutError:
        pytest.fail("Did not receive event on subject")

    assert received.get("event") == "demo"
    assert received.get("value") == 42

