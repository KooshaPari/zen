import asyncio
import os

import aiohttp

BASE = os.environ.get("TEST_SERVER_BASE", "http://127.0.0.1:8080")


async def main():
    async with aiohttp.ClientSession() as s:
        # health
        async with s.get(f"{BASE}/health") as r:
            print("/health:", r.status, await r.text())

        # Start RPC responder for demo task
        async with s.post(f"{BASE}/a2a/test/rpc-responder/start", json={"task_id": "TDEMO"}) as r:
            print("start responder:", r.status, await r.text())

        # Advertise agent
        payload = {
            "agent_card": {
                "agent_id": "agent-demo",
                "name": "DemoAgent",
                "version": "0.1.0",
                "endpoint_url": BASE,
                "capabilities": [
                    {"name":"chat","description":"Chat demo","category":"nlp","input_schema":{},"output_schema":{}}
                ],
                "last_seen": "2025-01-01T00:00:00Z",
            }
        }
        async with s.post(f"{BASE}/a2a/advertise", json=payload) as r:
            print("advertise:", r.status, await r.text())

        # Discover agents
        async with s.post(f"{BASE}/a2a/discover", json={"capability_filter": "chat", "max_results": 5}) as r:
            print("discover:", r.status, await r.text())

        # Blocking chat to self via HTTP wrapper
        async with s.post(f"{BASE}/a2a/chat", json={"to_agent": "agent-demo", "text": "hello", "timeout": 5}) as r:
            print("chat blocking:", r.status, await r.text())

        # RPC via helper
        async with s.post(f"{BASE}/a2a/test/rpc", json={"task_id":"TDEMO","method":"ping","params":{"x":1},"timeout":5}) as r:
            print("rpc:", r.status, await r.text())

        # Non-blocking chat
        async with s.post(f"{BASE}/a2a/chat", json={"to_agent": "agent-demo", "text": "fire-and-forget"}) as r:
            print("chat nb:", r.status, await r.text())

if __name__ == "__main__":
    asyncio.run(main())

