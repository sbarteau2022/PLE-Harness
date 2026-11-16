from __future__ import annotations

import base64
import contextlib
import http.server
import json
import os
import pathlib
import subprocess
import threading


TOOL = pathlib.Path(__file__).parents[1] / "ablations" / "bridge" / "ple_tool.py"


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        if "/mcp/" in self.path:
            raw = self.rfile.read(length)
            if "/missing/" in self.path:
                self.send_response(404)
                self.send_header("Content-Length", str(len(b"no such MCP server")))
                self.end_headers()
                self.wfile.write(b"no such MCP server")
                return
            reply = b"mcp-echo:" + raw
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)
            return
        body = json.loads(self.rfile.read(length))
        if self.path.endswith("/exec"):
            value = {"ok": True, "output": body["command"], "exit_code": 7, "timed_out": False}
        elif self.path.endswith("/read_file"):
            value = {"ok": True, "content_b64": base64.b64encode(body["path"].encode()).decode()}
        else:
            value = {"ok": True, "size": len(base64.b64decode(body["content_b64"]))}
        raw = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@contextlib.contextmanager
def server():
    instance = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{instance.server_port}/token"
    finally:
        instance.shutdown()
        thread.join()


def call(url: str, *args: str, input_text: str | None = None):
    return subprocess.run(
        [str(TOOL), *args], input=input_text, text=True, capture_output=True,
        env={**os.environ, "TASK_RELAY_URL": url}, timeout=3,
    )


def test_exec_preserves_output_and_exit_code() -> None:
    with server() as url:
        result = call(url, "exec", "hello")
    assert result.stdout == "hello" and result.returncode == 7


def test_read_and_write_round_trip_contract() -> None:
    with server() as url:
        assert call(url, "read", "/work/a").stdout == "/work/a"
        result = call(url, "write", "/work/a", "-", input_text="three")
    assert result.returncode == 0 and result.stdout.strip() == "5"


def test_rejects_untrusted_relay_and_bad_timeout() -> None:
    assert call("https://example.com/token", "read", "x").returncode == 125
    with server() as url:
        assert call(url, "exec", "x", "--timeout", "0").returncode == 125


def test_mcp_forwards_body_and_reports_server_errors() -> None:
    with server() as url:
        ok = call(url, "mcp", "calc", "POST", "mcp/", "-", input_text='{"a":1}')
        assert ok.returncode == 0 and ok.stdout == 'mcp-echo:{"a":1}'
        missing = call(url, "mcp", "missing", "POST", "/", "-", input_text="{}")
        assert missing.returncode == 126 and "no such MCP server" in missing.stdout


def test_mcp_rejects_get_and_bad_names() -> None:
    with server() as url:
        assert call(url, "mcp", "calc", "GET", "mcp/", "-", input_text="").returncode == 125
        assert call(url, "mcp", "a/b", "POST", "mcp/", "-", input_text="").returncode == 125
