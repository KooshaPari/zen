"""
Minimal A2A handler example: listens for delegate requests and logs them.

Usage:
  export NATS_URL=nats://localhost:4222
  python examples/a2a_handler_demo.py --agent-id agent:builder
"""
import argparse
import asyncio
import json
import os

try:
    from nats.aio.client import Client as NATS
    from nats.js.api import AckPolicy, DeliverPolicy
except Exception:
    raise SystemExit("Please install nats-py: pip install nats-py")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--durable", default="a2a_handler")
    args = parser.parse_args()

    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    nc = NATS()
    await nc.connect(servers=[nats_url])
    js = nc.jetstream()

    subject = f"a2a.agent.{args.agent_id}.in"
    sub = await js.subscribe(subject, durable=args.durable, deliver_policy=DeliverPolicy.All, ack_policy=AckPolicy.Explicit)

    print(f"Listening on {subject} durable={args.durable}")
    try:
        while True:
            msg = await sub.next_msg(timeout=30)
            if not msg:
                continue
            try:
                data = json.loads(msg.data.decode("utf-8"))
            except Exception:
                data = {"raw": msg.data.decode("utf-8", "ignore")}
            print("A2A HANDLER RECEIVED", data)
            await msg.ack()
    except asyncio.TimeoutError:
        pass
    finally:
        await nc.drain()


if __name__ == "__main__":
    asyncio.run(main())

