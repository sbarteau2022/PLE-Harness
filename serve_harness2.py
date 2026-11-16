"""PLE serve-protocol entry point using Google's pinned Harness2 method."""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import os
import pathlib

import taskbridge
from ablations.normalize import (MARKER, parse_opencode_tool_calls,
                                 response_ids_from_events, sha256_file)


ROOT = pathlib.Path(__file__).resolve().parent


def _configure() -> str:
    model = os.environ.get("MODEL_NAME", "")
    base = os.environ.get("MODEL_BASE_URL", "")
    key = os.environ.get("MODEL_API_KEY", "")
    if not model or not base or not key:
        raise RuntimeError("MODEL_NAME, MODEL_BASE_URL, and MODEL_API_KEY are required")
    routed = model if "/" in model and model.split("/", 1)[0] in {"openai", "nvidia"} else f"nvidia/{model}"
    os.environ.update({
        "NVIDIA_API_KEY": key,
        "OPENAI_API_KEY": key,
        "OPENAI_BASE_URL": base,
        "HARNESS2_SOLVER_MODEL": routed,
        "HARNESS2_IMPROVER_MODEL": routed,
        "HARNESS2_JUDGE_MODEL": model,
        "HARNESS2_JUDGE_BASE_URL": base,
        "HARNESS2_JUDGE_API_KEY": key,
    })
    return routed


async def solve(session: taskbridge.Session) -> str:
    started = dt.datetime.now(dt.timezone.utc)
    model = _configure()
    from harness2.core import schedule
    from harness2.core.proposer import Proposer
    from ablations.harness2_ple import PleAdapter

    digest = hashlib.sha256(session.instruction.encode()).hexdigest()[:16]
    run_root = ROOT / "ablations" / "runs" / "harness2" / model.replace("/", "__") / digest
    run_root.mkdir(parents=True, exist_ok=True)
    remaining = session.time_left() or 3600
    per_call = max(120, min(900, int(max(120, remaining - 180) / 7)))
    async with taskbridge.Relay(session) as relay:
        adapter = PleAdapter(session.instruction, relay.url, run_root, model, timeout=per_call)
        proposer = Proposer(adapter.spec.channels["proposer"], audit_dir=run_root / "audit", attempts=2)
        result = await asyncio.to_thread(
            schedule.run_task,
            adapter.task_id,
            1,
            adapter=adapter,
            prop=proposer,
            runs_root=run_root / "adapt",
            modes=("agg",),
            seed=20261115,
            pool_policy="finals",
        )
    record = result.to_dict()
    (run_root / "harness2-result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    selection = record.get("selections", {}).get("agg", {})
    if record.get("status") != "ok" or not selection.get("chosen_run_dir"):
        raise RuntimeError(f"Harness2 did not close the task: {record.get('errors')}")
    meter: dict = {"chosen_label": selection.get("chosen_label"), "executions": []}
    adapt = run_root / "adapt"
    if adapt.is_dir():
        for stream in sorted(adapt.rglob("events.jsonl")):
            try:
                digest, size = sha256_file(stream)
                counts = parse_opencode_tool_calls(stream)
                meter["executions"].append({
                    "path": str(stream.relative_to(run_root)),
                    "sha256": digest, "bytes": size,
                    "response_ids": response_ids_from_events(stream), **counts,
                })
            except OSError:
                continue
    ended = dt.datetime.now(dt.timezone.utc)
    print(f"{MARKER} " + json.dumps({
        "harness": "harness2", "started_at": started.isoformat(), "ended_at": ended.isoformat(),
        "telemetry": meter,
    }), flush=True)
    return "Harness2 completed Propose-Probe-Compose; adopted live run " + str(selection["chosen_label"])


if __name__ == "__main__":
    # Harness2 is pinned under this tree and never imported from a global install.
    import sys
    sys.path.insert(0, str(ROOT / "ablations" / "vendor" / "harness2"))
    taskbridge.run(solve)

