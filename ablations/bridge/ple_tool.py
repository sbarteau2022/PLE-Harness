#!/usr/bin/env python3
"""Call the live PLE task relay from an external harness subprocess."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class RelayError(RuntimeError):
    pass


def _base() -> str:
    value = os.environ.get("TASK_RELAY_URL", "").rstrip("/")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or not parsed.path.strip("/"):
        raise RelayError("TASK_RELAY_URL must be the authenticated loopback relay URL")
    return value


def _request(route: str, body: dict[str, Any], timeout: int = 610) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{_base()}/{route}",
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(64 << 20)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RelayError(f"relay request failed: {exc}") from exc
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RelayError("relay returned malformed JSON") from exc
    if not isinstance(value, dict):
        raise RelayError("relay response must be an object")
    if value.get("ok") is False or value.get("error"):
        raise RelayError(str(value.get("error") or "task operation failed"))
    return value


def execute(command: str, timeout: int) -> int:
    result = _request("exec", {"command": command, "timeout_sec": timeout}, timeout=timeout + 10)
    output = result.get("output", "")
    if not isinstance(output, str):
        raise RelayError("exec output is not text")
    sys.stdout.write(output)
    if result.get("timed_out") is True:
        return 124
    code = result.get("exit_code")
    if isinstance(code, bool) or not isinstance(code, int):
        raise RelayError("exec response has no integer exit_code")
    return code


def read_file(path: str) -> int:
    result = _request("read_file", {"path": path})
    encoded = result.get("content_b64")
    if not isinstance(encoded, str):
        raise RelayError("read response has no content_b64")
    try:
        data = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise RelayError("read response contains invalid base64") from exc
    sys.stdout.buffer.write(data)
    return 0


def write_file(path: str, source: str, append: bool) -> int:
    data = sys.stdin.buffer.read() if source == "-" else open(source, "rb").read()
    result = _request("write_file", {
        "path": path,
        "content_b64": base64.b64encode(data).decode(),
        "append": append,
    })
    size = result.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise RelayError("write response has no valid size")
    print(size)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="operation", required=True)
    run = commands.add_parser("exec")
    run.add_argument("command")
    run.add_argument("--timeout", type=int, default=300)
    read = commands.add_parser("read")
    read.add_argument("path")
    write = commands.add_parser("write")
    write.add_argument("path")
    write.add_argument("source", help="local source file, or - for stdin")
    write.add_argument("--append", action="store_true")
    mcp = commands.add_parser("mcp", help="POST/PUT/etc a body to a task MCP server via the relay proxy")
    mcp.add_argument("server", help="MCP server name from the task session")
    mcp.add_argument("method", help="HTTP method, e.g. POST (GET is refused: SSE streams need no single response)")
    mcp.add_argument("path", help="subpath under the server, e.g. mcp/ or messages/?session_id=..")
    mcp.add_argument("source", help="local body file, or - for stdin (may be empty)")
    mcp.add_argument("--content-type", default="application/json")
    mcp.add_argument("--timeout", type=int, default=300)
    return root


def mcp_call(server: str, method: str, path: str, body: bytes, content_type: str, timeout: int) -> int:
    method = method.upper()
    if method == "GET" or not method.isalpha():
        raise RelayError("mcp method must be a non-GET alpha token (POST, PUT, DELETE, ..)")
    if not server or "/" in server or not path:
        raise RelayError("mcp needs a server name without '/' and a nonempty subpath")
    if len(body) > 8 << 20:
        raise RelayError("mcp body exceeds the 8 MiB relay cap")
    target = f"{_base()}/mcp/{urllib.parse.quote(server, safe='')}/{path.lstrip('/')}"
    request = urllib.request.Request(target, data=body or b"",
                                     headers={"Content-Type": content_type}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            sys.stdout.buffer.write(response.read(64 << 20))
    except urllib.error.HTTPError as exc:
        detail = exc.read(64 << 20)
        try:
            sys.stdout.buffer.write(detail)
        except OSError:
            pass
        print(f"ple-tool: mcp {server} answered HTTP {exc.code}", file=sys.stderr)
        return 126
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RelayError(f"mcp request failed: {exc}") from exc
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.operation == "exec":
            if args.timeout < 1 or args.timeout > 600:
                raise RelayError("timeout must be in 1..600")
            return execute(args.command, args.timeout)
        if args.operation == "read":
            return read_file(args.path)
        if args.operation == "mcp":
            if args.timeout < 1 or args.timeout > 600:
                raise RelayError("timeout must be in 1..600")
            data = sys.stdin.buffer.read() if args.source == "-" else open(args.source, "rb").read()
            return mcp_call(args.server, args.method, args.path, data, args.content_type, args.timeout)
        return write_file(args.path, args.source, args.append)
    except (OSError, RelayError) as exc:
        print(f"ple-tool: {exc}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())

