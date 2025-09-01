import json

import pytest

from utils.router_service import RouterInput, RouterService


class _FakeRedisConn:
    def __init__(self):
        self.store = {}

    def get(self, key):
        val = self.store.get(key)
        if val is None:
            return None
        # Return JSON string as Redis would
        return json.dumps(val)

    def setex(self, key, ttl, value):
        # Value is JSON string per RouterService
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = value
        self.store[key] = parsed


class _FakeRedisManager:
    def __init__(self):
        self.conn = _FakeRedisConn()

    def get_connection(self, db: int):
        return self.conn


class _FakeYamlRouter:
    def __init__(self, chosen="google:gemini-2.5-flash"):
        self._chosen = chosen
        self.config = {"caching": {"ttls_seconds": {"router_default": 60}}}

    def decide(self, task_type, signals=None, est_tokens=None, allow_long_context=True):
        return {
            "chosen_model": self._chosen,
            "candidates": [self._chosen],
            "tier": "simple",
        }


@pytest.mark.asyncio
async def test_routerservice_cache_miss_then_hit(monkeypatch):
    svc = RouterService()

    # Monkeypatch Redis manager and YAML router; disable OpenRouter refinement
    fake_mgr = _FakeRedisManager()
    monkeypatch.setattr(svc, "redis", fake_mgr)
    monkeypatch.setattr(svc, "router", _FakeYamlRouter())

    from providers import registry as prov_registry
    monkeypatch.setattr(prov_registry.ModelProviderRegistry, "get_provider", staticmethod(lambda *_args, **_kw: None))

    rin = RouterInput(task_type="quick_qa", prompt="Hello world")

    d1 = svc.decide(rin)
    assert d1["cache_hit"] is False
    assert d1["chosen_model"]

    d2 = svc.decide(rin)
    assert d2["cache_hit"] is True
    assert d2["chosen_model"] == d1["chosen_model"]

    # Ensure stored structure round-trips via cache
    assert d2["candidates"] and isinstance(d2["candidates"], list)

