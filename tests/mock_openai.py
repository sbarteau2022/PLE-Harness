"""Scripted OpenAI-compatible mock for kit checks without a live model. Usage: mock_openai.py PORT

It answers the kit's smoke prompt (hello.txt) and MCP prompt (mcp.txt) by following the
tool results it receives, so it exercises the harness's bash and MCP tool paths.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


def _call(name, arguments, call_id):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def decide(messages):
    user = next((m["content"] for m in messages if m["role"] == "user"), "") or ""
    tools_used = [m for m in messages if m["role"] == "tool"]
    last = tools_used[-1]["content"] if tools_used else ""
    calls = sum(1 for m in messages if m["role"] == "assistant" and m.get("tool_calls"))
    cid = f"call_{calls + 1}"
    if "mcp.txt" in user:
        if not tools_used:
            return _call("mcp_call", {"server": "apps", "tool": "secret_word", "arguments": {}}, cid)
        if "secret_word" not in messages[-1].get("content", "") and "mcp_call" in str(messages[-2:]) and not any(
                "printf" in str(m.get("tool_calls", "")) for m in messages):
            word = last.strip().splitlines()[0] if last.strip() else "unknown"
            return _call("bash", {"command": f"printf '%s\\n' '{word}' > mcp.txt && cat mcp.txt"}, cid)
        return _call("finish", {"summary": "Wrote mcp.txt with the secret word."}, cid)
    if "hello.txt" in user:
        if not tools_used:
            return _call("bash", {"command": "printf 'HELLO\\n' > hello.txt && cat hello.txt"}, cid)
        return _call("finish", {"summary": "done"}, cid)
    return _call("finish", {"summary": "Nothing to do."}, cid)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self._send(json.dumps({"object": "list", "data": [{"id": "mock", "object": "model"}]}).encode())

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if request.get("tools"):
            message = {"role": "assistant", "content": None, "tool_calls": [decide(request["messages"])]}
        else:
            message = {"role": "assistant", "content": "OK"}
        payload = {"id": "m", "object": "chat.completion", "created": 0, "model": request.get("model", "mock"),
                   "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}}
        self._send(json.dumps(payload).encode())

    def _send(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
