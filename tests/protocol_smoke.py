"""End-to-end wire test: real serve.py process, a fake evaluator, and a mock OpenAI endpoint.

Run: .venv/bin/python tests/protocol_smoke.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import pathlib
import socket
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import websockets

HARNESS = pathlib.Path(__file__).resolve().parents[1]
PY = HARNESS / ".venv" / "bin" / "python"

SCRIPT = [
    {"tool": "bash", "args": {"command": "mkdir -p output && printf '42\\n' > output/answer.txt && cat output/answer.txt"}},
    {"tool": "read_file", "args": {"path": "output/answer.txt"}},
    {"tool": "finish", "args": {"summary": "Wrote output/answer.txt with 42."}},
]
SEEN_BODIES: list[dict] = []


class Mock(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN_BODIES.append(body)
        step = len(SEEN_BODIES)
        item = SCRIPT[min(step - 1, len(SCRIPT) - 1)]
        message = {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"call_{step}", "type": "function",
            "function": {"name": item["tool"], "arguments": json.dumps(item["args"])}}]}
        payload = {"id": f"x{step}", "object": "chat.completion", "created": 0, "model": body["model"],
                   "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
                   "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def evaluator(port: int, workdir: str) -> str:
    uri = f"ws://127.0.0.1:{port}"
    ws = None
    for _ in range(200):
        try:
            ws = await websockets.connect(uri, max_size=16 * 2**20, ping_interval=None)
            break
        except OSError:
            await asyncio.sleep(0.1)
    assert ws is not None, "harness did not start"
    async with ws:
        await ws.send(json.dumps({"type": "task", "protocol": 1, "session": "smoke",
                                  "instruction": "Write output/answer.txt containing 42.",
                                  "workdir": workdir, "limits": {"agent_timeout_sec": 600},
                                  "mcp_servers": []}))
        async for raw in ws:
            msg = json.loads(raw)
            kind = msg["type"]
            if kind == "done":
                return msg["final_message"]
            if kind == "exec":
                proc = subprocess.run(["bash", "-c", msg["command"]], cwd=workdir, capture_output=True, text=True)
                reply = {"type": "result", "id": msg["id"], "exit_code": proc.returncode,
                         "output": proc.stdout + proc.stderr, "truncated": False, "timed_out": False}
            elif kind == "write_file":
                target = pathlib.Path(workdir) / msg["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                data = base64.b64decode(msg["content_b64"])
                target.write_bytes(data)
                reply = {"type": "result", "id": msg["id"], "ok": True, "size": len(data)}
            elif kind == "read_file":
                data = (pathlib.Path(workdir) / msg["path"]).read_bytes()
                reply = {"type": "result", "id": msg["id"], "ok": True,
                         "content_b64": base64.b64encode(data).decode(), "size": len(data), "truncated": False}
            else:
                continue  # log, usage
            await ws.send(json.dumps(reply))
    raise RuntimeError("connection closed without done")


def main() -> None:
    mock_port = free_port()
    httpd = HTTPServer(("127.0.0.1", mock_port), Mock)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    harness_port = free_port()
    env = dict(os.environ, PORT=str(harness_port), MODEL_NAME="test/mock-model",
               MODEL_BASE_URL=f"http://127.0.0.1:{mock_port}/v1", MODEL_API_KEY="test-key")
    proc = subprocess.Popen([str(PY), str(HARNESS / "serve.py")], env=env, cwd=HARNESS,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        with tempfile.TemporaryDirectory() as workdir:
            final = asyncio.run(asyncio.wait_for(evaluator(harness_port, workdir), 90))
            answer = (pathlib.Path(workdir) / "output/answer.txt").read_text().strip()
        print("final_message:", final)
        print("answer.txt:", answer)
        print("model calls:", len(SEEN_BODIES), "| tools offered:", [t["function"]["name"] for t in SEEN_BODIES[0]["tools"]])
        assert answer == "42"
        assert final.startswith("Wrote output/answer.txt")
        print("PROTOCOL SMOKE: PASS")
    finally:
        proc.terminate()
        out = proc.communicate(timeout=10)[0]
        httpd.shutdown()
        if "Traceback" in out:
            print(out)


if __name__ == "__main__":
    main()
