#!/usr/bin/env python3
"""
Simple SSE soak test for the streamable HTTP server.

Spawns N concurrent clients connected to /events/live (global) or /stream/{task_id}.

Usage:
  python scripts/soak_sse.py --host http://localhost:8080 --clients 50 --duration 30 --mode global

Modes:
  - global: connect to /events/live
  - task:   connect to /stream/{task_id} (requires --task-id)
"""

import argparse
import asyncio
import sys

import aiohttp


async def sse_reader(session: aiohttp.ClientSession, url: str, idx: int, duration: int) -> int:
    messages = 0
    try:
        async with session.get(url, timeout=None) as resp:
            if resp.status != 200:
                print(f"[{idx}] HTTP {resp.status} for {url}")
                return 0
            start = asyncio.get_event_loop().time()
            async for line in resp.content:
                if not line:
                    continue
                messages += 1
                now = asyncio.get_event_loop().time()
                if now - start > duration:
                    break
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[{idx}] Error: {e}")
    return messages


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8080")
    ap.add_argument("--clients", type=int, default=50)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--mode", choices=["global", "task"], default="global")
    ap.add_argument("--task-id", default=None)
    args = ap.parse_args()

    if args.mode == "task" and not args.task_id:
        print("--task-id is required for mode=task", file=sys.stderr)
        return 2

    url = f"{args.host}/events/live" if args.mode == "global" else f"{args.host}/stream/{args.task_id}"

    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(sse_reader(session, url, i, args.duration)) for i in range(args.clients)]
        results = await asyncio.gather(*tasks)
    total = sum(results)
    print(f"Clients: {args.clients}, Duration: {args.duration}s, Total messages: {total}")
    return 0


if __name__ == "__main__":
    try:
        exit(asyncio.run(main()))
    except KeyboardInterrupt:
        pass
