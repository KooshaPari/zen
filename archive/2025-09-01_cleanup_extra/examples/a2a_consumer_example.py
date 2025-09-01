"""
Simple A2A consumer example.

Usage:
  export NATS_URL=nats://localhost:4222
  python examples/a2a_consumer_example.py
"""
import asyncio
import os


async def main():
    try:
        from utils.nats_communicator import get_nats_communicator
    except Exception:
        raise SystemExit("Please install nats-py to run this example: pip install nats-py")

    nats_url = os.getenv("NATS_URL", "nats://127.0.0.1:4222")
    print(f"Connecting to NATS at {nats_url}…")
    comm = await get_nats_communicator(None)

    async def on_evt(msg: dict):
        print("A2A EVENT:", msg)

    # Subscribe to all task events (JetStream optional)
    ok = await comm.subscribe("a2a.task.*.events", on_evt, use_jetstream=True, durable_name="a2a-example-consumer")
    print("Subscribed:", ok)

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Exiting")


if __name__ == "__main__":
    asyncio.run(main())

