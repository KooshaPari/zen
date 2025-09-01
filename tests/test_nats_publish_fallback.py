from types import SimpleNamespace

import pytest

from utils.nats_communicator import NATSCommunicator, NATSConfig


@pytest.mark.asyncio
async def test_publish_fallback_counters(monkeypatch):
    # Create communicator with fake nc/js
    comm = NATSCommunicator(NATSConfig())
    comm.connected = True

    # Fake nc with async publish/flush
    published_core = {}

    async def fake_nc_publish(subject, data):
        published_core[subject] = data

    async def fake_nc_flush(timeout=None):
        return None

    comm.nc = SimpleNamespace(publish=fake_nc_publish, flush=fake_nc_flush)

    class FakeJS:
        async def publish(self, subject, data):
            raise RuntimeError("force_js_failure")

    comm.js = FakeJS()

    # Ensure counters zeroed
    assert comm.publish_js_attempts == 0
    assert comm.publish_js_fallbacks == 0
    assert comm.publish_core_success == 0

    ok = await comm.publish("metrics.nats", {"x": 1}, use_jetstream=True)
    assert ok is True

    # JS attempted then fell back to core, and core publish succeeded
    assert comm.publish_js_attempts == 1
    assert comm.publish_js_fallbacks == 1
    assert comm.publish_core_success == 1
    assert "metrics.nats" in published_core

    # Metrics surface these counters
    m = await comm.get_performance_metrics()
    assert m["publish_js_attempts"] >= 1
    assert m["publish_js_fallbacks"] >= 1
    assert m["publish_core_success"] >= 1

