"""Campaign normalization: hashes, trajectories, receipts, and results table.

All inputs are explicit parsed values; nothing here estimates a measurement.
Unavailable provider fields stay ``None`` with a reason in ``unavailable``.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
from typing import Any

from ablations.receipts import validate_receipt


def sha256_file(path: pathlib.Path) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def hash_deliverables(root: pathlib.Path, relpaths: list[str]) -> list[dict[str, Any]]:
    items = []
    for rel in relpaths:
        target = root / rel
        if not target.is_file():
            raise ValueError(f"deliverable {rel} is missing under {root}")
        digest, size = sha256_file(target)
        items.append({"path": rel, "sha256": digest, "bytes": size})
    if not items:
        raise ValueError("at least one deliverable is required")
    return items


_TOOL_TYPES = {"tool", "tool_use", "tool_call", "tool-invocation"}


def parse_opencode_tool_calls(events_path: pathlib.Path) -> dict[str, Any]:
    """Count tool calls by name in an OpenCode `--format json` JSONL stream."""
    by_name: dict[str, int] = {}
    calls, errors = 0, 0
    with open(events_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            part = event.get("part") if isinstance(event.get("part"), dict) else event
            if part.get("type", event.get("type")) not in _TOOL_TYPES:
                continue
            name = (part.get("tool") or event.get("tool") or part.get("name")
                    or (part.get("toolInvocation") or {}).get("toolName") or "tool")
            by_name[str(name)] = by_name.get(str(name), 0) + 1
            calls += 1
            state = part.get("state") if isinstance(part.get("state"), dict) else part
            if state.get("status") == "error" or state.get("error"):
                errors += 1
    return {"calls": calls, "errors": errors, "by_name": by_name}


MARKER = "PLE-ABLATION-RUN"


def response_ids_from_events(events_path: pathlib.Path, cap: int = 400) -> list[str]:
    """Collect distinct live turn IDs from an OpenCode JSONL stream, in order."""
    seen: list[str] = []
    with open(events_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            part = event.get("part") if isinstance(event.get("part"), dict) else {}
            for key in ("id", "messageId", "message_id", "sessionId", "session_id"):
                value = part.get(key) or event.get(key)
                if isinstance(value, str) and value and value not in seen:
                    seen.append(value)
                    break
            if len(seen) >= cap:
                break
    return seen


def extract_markers(text: str) -> list[dict[str, Any]]:
    """Parse every ``PLE-ABLATION-RUN {...}`` marker from captured output."""
    found = []
    for line in text.splitlines():
        prefix = MARKER + " "
        if not line.startswith(prefix):
            continue
        try:
            value = json.loads(line[len(prefix):])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            found.append(value)
    return found


def record_trajectory(events_path: pathlib.Path, dest_dir: pathlib.Path, run_id: str) -> dict[str, Any]:
    """Copy a trajectory stream into the run archive and hash it."""
    if not events_path.is_file() or not events_path.stat().st_size:
        raise ValueError("trajectory stream is missing or empty")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{run_id}-trajectory.jsonl"
    shutil.copy2(events_path, dest)
    digest, size = sha256_file(dest)
    counts = parse_opencode_tool_calls(dest)
    return {"path": dest.name, "sha256": digest, "bytes": size, **counts}


def write_json_atomic(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def build_receipt(*, run_id: str, campaign: str, harness_id: str, harness_version: str,
                  harness_digest: str, model_id: str, endpoint_class: str,
                  response_ids: list[str], task: str, seed: int, config: dict[str, Any],
                  started_at: str, ended_at: str, wall_seconds: float,
                  judges: list[dict[str, Any]], deliverables: list[dict[str, Any]],
                  calls: dict[str, Any], tools: dict[str, Any], tokens: dict[str, Any],
                  latency: dict[str, Any], memory: dict[str, Any],
                  peak_rss_bytes: int | None, provider_cost: float | None,
                  unavailable: dict[str, str],
                  finish_refusals: int = 0) -> dict[str, Any]:
    receipt = {
        "schema_version": 1, "run_id": run_id, "campaign": campaign, "task": task, "seed": seed,
        "harness": {"id": harness_id, "version": harness_version, "source_digest": harness_digest},
        "model": {"id": model_id, "endpoint_class": endpoint_class, "response_ids": response_ids},
        "config": config,
        "timing": {"started_at": started_at, "ended_at": ended_at, "wall_seconds": wall_seconds},
        "live": True, "exit": {"state": "complete", "unresolved_error": None,
                               "finish_refusals": finish_refusals},
        "judges": judges, "deliverables": deliverables, "calls": calls, "tools": tools,
        "tokens": tokens, "latency": latency, "memory": memory,
        "process": {"peak_rss_bytes": peak_rss_bytes},
        "provider_cost": provider_cost, "unavailable": unavailable,
    }
    validate_receipt(receipt)
    return receipt


CELL_STATES = ("complete", "incomplete", "failed", "excluded")


def build_results(*, campaign: str, seed: int, cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the canonical machine-readable outcome table."""
    for cell in cells:
        if cell.get("state") not in CELL_STATES:
            raise ValueError(f"cell {cell} has no valid state")
        if not isinstance(cell.get("evidence"), list) or not cell["evidence"]:
            raise ValueError(f"cell {cell.get('harness')}:{cell.get('model')}:{cell.get('task')} lacks evidence paths")
    return {"schema_version": 1, "campaign": campaign, "seed": seed, "cells": cells}
