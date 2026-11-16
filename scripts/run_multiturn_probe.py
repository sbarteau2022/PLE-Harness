#!/usr/bin/env python3
"""Run the controlled live 27-call probe without Harbor or PLE dev tasks."""
from __future__ import annotations

import argparse
import asyncio
import base64
import datetime as dt
import hashlib
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ALLOWED = {
    "pwd",
    "find inputs -type f | sort",
    "wc -l inputs/*.txt",
    "awk '{print NF}' inputs/fact-1.txt",
    "sort inputs/fact-2.txt",
    "uniq -c inputs/fact-3.txt",
    "shasum -a 256 inputs/fact-4.txt",
    "shasum -a 256 output/summary.md",
}
REPEAT = "shasum -a 256 output/summary.md"
TASK = """Complete this controlled tool-use task. Use one tool call per numbered item, in order; do not combine calls.

1. bash: pwd
2. bash: find inputs -type f | sort
3-10. read_file each of inputs/fact-1.txt through inputs/fact-8.txt, separately.
11. bash: wc -l inputs/*.txt
12. bash: awk '{print NF}' inputs/fact-1.txt
13. bash: sort inputs/fact-2.txt
14. bash: uniq -c inputs/fact-3.txt
15. bash: shasum -a 256 inputs/fact-4.txt
16. write_file output/summary.md with a concise synthesis containing every FACT-ID.
17. write_file output/matrix.json with a JSON object mapping every FACT-ID to its value.
18. write_file output/receipt.txt listing the three output paths.
19-24. Call bash with exactly `shasum -a 256 output/summary.md` six separate times. These repetitions are intentional; observe every result before the next call.
25. read_file output/summary.md
26. read_file output/matrix.json
27. call finish alone with the three paths and checksum agreement. Do not finish beside another tool call.
"""


class ProbeSession:
    mcp_servers: list[dict[str, Any]] = []

    def __init__(self, workdir: pathlib.Path, seconds: int = 1800):
        self.root = workdir.resolve(); self.workdir = str(self.root)
        self.instruction = TASK; self.deadline = time.monotonic() + seconds
        self.events: list[dict[str, Any]] = []

    def time_left(self) -> float: return self.deadline - time.monotonic()

    def _path(self, value: str) -> pathlib.Path:
        target = (self.root / value).resolve()
        if target != self.root and self.root not in target.parents: raise ValueError("path escapes probe workspace")
        return target

    def _event(self, kind: str, **fields: Any) -> None:
        self.events.append({"at": dt.datetime.now(dt.timezone.utc).isoformat(), "kind": kind, **fields})

    async def exec(self, command: str, timeout_sec: int = 120) -> dict[str, Any]:
        started = time.monotonic()
        if command not in ALLOWED:
            self._event("tool", name="bash", command=command, ok=False, error="command not admitted")
            return {"exit_code": 126, "output": "command not admitted", "timed_out": False}
        try:
            done = await asyncio.to_thread(subprocess.run, ["/bin/bash", "-lc", command], cwd=self.root,
                                           capture_output=True, text=True, timeout=min(timeout_sec, 120), check=False)
            output = done.stdout + done.stderr
            self._event("tool", name="bash", command=command, ok=done.returncode == 0,
                        exit_code=done.returncode, ms=round((time.monotonic()-started)*1000))
            return {"exit_code": done.returncode, "output": output, "timed_out": False}
        except subprocess.TimeoutExpired:
            self._event("tool", name="bash", command=command, ok=False, timed_out=True)
            return {"exit_code": 124, "output": "timed out", "timed_out": True}

    async def read_file(self, path: str) -> bytes:
        data = self._path(path).read_bytes(); self._event("tool", name="read_file", path=path, ok=True, bytes=len(data)); return data

    async def write_file(self, path: str, data: bytes | str, append: bool = False) -> int:
        target = self._path(path); target.parent.mkdir(parents=True, exist_ok=True)
        raw = data.encode() if isinstance(data, str) else bytes(data)
        with open(target, "ab" if append else "wb") as handle: handle.write(raw)
        self._event("tool", name="write_file", path=path, ok=True, bytes=len(raw)); return len(raw)

    async def log(self, text: str) -> None: self._event("harness_log", text=text)
    async def usage(self, model: str, input_tokens: int = 0, output_tokens: int = 0, cache_read_tokens: int = 0) -> None:
        self._event("usage", model=model, input_tokens=input_tokens, output_tokens=output_tokens, cache_read_tokens=cache_read_tokens)


def seed(root: pathlib.Path) -> None:
    (root / "inputs").mkdir(parents=True); (root / "output").mkdir()
    for index in range(1, 9):
        (root / "inputs" / f"fact-{index}.txt").write_text(f"FACT-{index:02d}: value-{index * 17}\n", encoding="utf-8")


def digest(path: pathlib.Path) -> dict[str, Any]:
    raw = path.read_bytes(); return {"path": str(path.name), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


async def main_async(args: argparse.Namespace) -> int:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = ROOT / "ablations" / "probes" / args.harness / args.seat / stamp
    workspace = run / "workspace"; workspace.mkdir(parents=True); seed(workspace)
    session = ProbeSession(workspace); meter: dict[str, Any] = {}; started = time.monotonic()
    model = args.model
    if args.harness == "generic":
        from ablations.generic.agent import Config, run as execute
        summary = await execute(session, Config(model, args.base_url, args.api_key, max_steps=30, max_tokens=2048), meter)
    elif args.harness == "ple":
        import agent
        summary = await agent.run(session, agent.Config(model, args.base_url, args.api_key, max_steps=30,
                                  max_output_tokens=2048,
                                  checkpoint_every=0, compact_at_tokens=90000), telemetry=meter)
    else:
        raise ValueError("harness must be generic or ple")
    tools = [x for x in session.events if x["kind"] == "tool"]
    repeated = [x for x in tools if x.get("name") == "bash" and x.get("command") == REPEAT]
    outputs = []
    for rel in ("output/summary.md", "output/matrix.json", "output/receipt.txt"):
        path = workspace / rel
        if path.is_file(): outputs.append({"path": rel, **{k:v for k,v in digest(path).items() if k != "path"}})
    closed = bool(meter.get("response_ids")) and 20 <= len(tools) <= 30 and 5 <= len(repeated) <= 8 and len(outputs) == 3 and meter.get("by_name", {}).get("finish", 0) >= 1
    result = {"schema_version":1,"harness":args.harness,"seat":args.seat,"model":model,"live":True,
              "closed":closed,"started_at":session.events[0]["at"] if session.events else None,
              "ended_at":dt.datetime.now(dt.timezone.utc).isoformat(),"wall_seconds":round(time.monotonic()-started,3),
              "requested_calls":27,"actual_tool_calls":len(tools),"prescribed_repeated_calls":len(repeated),
              "deliverables":outputs,"summary":summary,"telemetry":meter,"events_path":"trajectory.jsonl"}
    (run / "trajectory.jsonl").write_text("".join(json.dumps(x, sort_keys=True)+"\n" for x in session.events), encoding="utf-8")
    tmp = run / ".results.tmp"; tmp.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n"); os.replace(tmp, run / "results.json")
    print(run / "results.json"); return 0 if closed else 1


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--harness", required=True, choices=("generic","ple")); p.add_argument("--seat", required=True)
    p.add_argument("--model", required=True); p.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    p.add_argument("--api-key", default=os.environ.get("NVIDIA_API_KEY") or os.environ.get("MODEL_API_KEY")); args = p.parse_args()
    if not args.api_key: p.error("NVIDIA_API_KEY or MODEL_API_KEY is required")
    return args


if __name__ == "__main__": raise SystemExit(asyncio.run(main_async(parse())))
