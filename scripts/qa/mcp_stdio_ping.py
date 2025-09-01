#!/usr/bin/env python3
"""
Minimal MCP stdio smoke: initialize and list tools.

Starts `./zen-mcp-server`, performs JSON-RPC framed requests over stdio:
- initialize
- tools/list

Prints a short summary and exits. Intended for CI smoke.
"""
import json
import os
import subprocess
import time


def frame(msg: dict) -> bytes:
    data = json.dumps(msg).encode("utf-8")
    return f"Content-Length: {len(data)}\r\n\r\n".encode() + data


def read_message(proc, timeout=5.0):
    proc.stdout.flush()
    start = time.time()
    header = b""
    while b"\r\n\r\n" not in header:
        if time.time() - start > timeout:
            raise TimeoutError("Timed out waiting for header")
        chunk = proc.stdout.read(1)
        if not chunk:
            continue
        header += chunk
    # Parse Content-Length
    header_text = header.decode("utf-8", errors="ignore")
    length = 0
    for line in header_text.split("\r\n"):
        if line.lower().startswith("content-length:"):
            length = int(line.split(":", 1)[1].strip())
            break
    if length <= 0:
        raise ValueError(f"Bad Content-Length in header: {header_text!r}")
    body = proc.stdout.read(length)
    return json.loads(body.decode("utf-8"))


def main():
    env = os.environ.copy()
    env.setdefault("LOG_LEVEL", "WARNING")

    proc = subprocess.Popen(
        ["./zen-mcp-server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        bufsize=0,
    )

    try:
        # initialize
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "capabilities": {
                    "experimental": {},
                }
            },
        }
        proc.stdin.write(frame(init_req))
        proc.stdin.flush()
        init_resp = read_message(proc, timeout=10.0)

        # tools/list
        list_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
        proc.stdin.write(frame(list_req))
        proc.stdin.flush()
        list_resp = read_message(proc, timeout=10.0)

        tools = list_resp.get("result", {}).get("tools", [])
        print(json.dumps({
            "initialize_ok": "result" in init_resp,
            "tool_count": len(tools),
            "sample_tools": [t.get("name") for t in tools[:5]],
        }))
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()

