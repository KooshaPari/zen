"""
Minimal NATS JetStream durable consumer example for task lifecycle events.

Usage:
  export NATS_URL=nats://localhost:4222
  python examples/nats_durable_consumer.py --subject tasks.completed --durable my_durable

This script prints received messages and acks them, supporting replay after restart.
"""
import argparse
import asyncio
import json
import os

try:
    from nats.aio.client import Client as NATS
    from nats.js.api import AckPolicy, DeliverPolicy
except Exception:  # pragma: no cover
    raise SystemExit("Please install NATS: pip install nats-py")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="tasks.completed")
    parser.add_argument("--durable", default="zen_consumer")
    args = parser.parse_args()

    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    nc = NATS()
    await nc.connect(servers=[nats_url])

    js = nc.jetstream()
    sub = await js.subscribe(
        args.subject,
        durable=args.durable,
        deliver_policy=DeliverPolicy.All,
        ack_policy=AckPolicy.Explicit,
    )

    print(f"Listening on {args.subject} with durable={args.durable}")
    try:
        while True:
            msg = await sub.next_msg(timeout=30)
            if not msg:
                continue
            try:
                data = json.loads(msg.data.decode("utf-8"))
            except Exception:
                data = {"raw": msg.data.decode("utf-8", "ignore")}
            print("EVENT", data)
            await msg.ack()
    except asyncio.TimeoutError:
        pass
    finally:
        await nc.drain()


if __name__ == "__main__":
    asyncio.run(main())

