import asyncio
import hashlib
import hmac
import json
import os
import time

import pytest

pytestmark = pytest.mark.asyncio

BASE = os.environ.get("TEST_SERVER_BASE", "http://127.0.0.1:8080")
PATH = "/a2a/chat"
URL = f"{BASE}{PATH}"
SECRET = "testsecret"


def _sig(secret: str, method: str, path: str, ts: str, body: bytes) -> str:
    msg = f"{method}\n{path}\n{ts}\n".encode() + body
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


@pytest.mark.integration
async def test_a2a_chat_hmac_success_and_failures():
    os.environ["HTTP_HMAC_SECRET"] = SECRET

    import aiohttp
    async with aiohttp.ClientSession() as session:
        # Health
        async with session.get(f"{BASE}/health") as r:
            assert r.status == 200

        # Missing headers → 401
        body = json.dumps({"to_agent": "dummy", "text": "hi"}).encode()
        async with session.post(URL, data=body, headers={"Content-Type": "application/json"}) as r:
            assert r.status == 401

        # Bad signature → 401
        ts = str(int(time.time()))
        headers = {
            "Content-Type": "application/json",
            "X-Timestamp": ts,
            "X-Signature": "deadbeef",
        }
        async with session.post(URL, data=body, headers=headers) as r:
            assert r.status == 401

        # Skewed timestamp → 401
        ts_old = str(int(time.time() - 10000))
        headers = {
            "Content-Type": "application/json",
            "X-Timestamp": ts_old,
            "X-Signature": _sig(SECRET, "POST", PATH, ts_old, body),
        }
        async with session.post(URL, data=body, headers=headers) as r:
            assert r.status == 401

        # Good signature → 200 (reset bucket by using unique client id)
        ts = str(int(time.time()))
        headers = {
            "Content-Type": "application/json",
            "X-Client-Id": "hmac-test-1",
            "X-Timestamp": ts,
            "X-Signature": _sig(SECRET, "POST", PATH, ts, body),
        }
        async with session.post(URL, data=body, headers=headers) as r:
            assert r.status == 200
            data = await r.json()
            assert "accepted" in data or "results" in data


@pytest.mark.integration
async def test_a2a_chat_rate_limit_token_bucket():
    # Enable HMAC and rate limiting
    os.environ["HTTP_HMAC_SECRET"] = SECRET
    os.environ["HTTP_RATE_LIMIT_RPS"] = "1"
    os.environ["HTTP_RATE_LIMIT_BURST"] = "1"

    import aiohttp
    async with aiohttp.ClientSession() as session:
        # Health
        async with session.get(f"{BASE}/health") as r:
            assert r.status == 200

        # First request should pass
        body = json.dumps({"to_agent": "dummy", "text": "hi"}).encode()
        ts1 = str(int(time.time()))
        headers = {
            "Content-Type": "application/json",
            "X-Client-Id": "ratelimit-test",
            "X-Timestamp": ts1,
            "X-Signature": _sig(SECRET, "POST", PATH, ts1, body),
        }
        async with session.post(URL, data=body, headers=headers) as r:
            assert r.status == 200

        # Second immediate request should be 429
        ts2 = str(int(time.time()))
        headers["X-Timestamp"] = ts2
        headers["X-Signature"] = _sig(SECRET, "POST", PATH, ts2, body)
        async with session.post(URL, data=body, headers=headers) as r:
            assert r.status == 429
            assert r.headers.get("Retry-After") is not None

        # Wait a bit then should succeed again
        await asyncio.sleep(1.1)
        ts3 = str(int(time.time()))
        headers["X-Timestamp"] = ts3
        headers["X-Signature"] = _sig(SECRET, "POST", PATH, ts3, body)
        async with session.post(URL, data=body, headers=headers) as r:
            assert r.status in (200, 201)

