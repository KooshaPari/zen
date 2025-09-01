"""
A2A delegate demo: publish a 'delegate' intent to a target agent.

Usage:
  export NATS_URL=nats://localhost:4222
  python examples/a2a_delegate_demo.py --task-id <task_id> --to agent:builder --payload '{"subtask":"implement feature X"}'
"""
import argparse
import asyncio
import json
import os
from datetime import datetime, timezone

try:
    from nats.aio.client import Client as NATS
except Exception:
    raise SystemExit("Please install nats-py: pip install nats-py")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--to", required=True)
    parser.add_argument("--payload", default='{}')
    args = parser.parse_args()

    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    nc = NATS()
    await nc.connect(servers=[nats_url])

    subject = f"a2a.agent.{args.to}.in"
    env = {
        "spec": "a2a/1",
        "type": "request",
        "id": f"dlg-{int(datetime.now(timezone.utc).timestamp()*1000)}",
        "intent": "delegate",
        "context": {"task_id": args.task_id},
        "payload": json.loads(args.payload),
    }
    await nc.publish(subject, json.dumps(env).encode("utf-8"))
    await nc.drain()


if __name__ == "__main__":
    asyncio.run(main())

