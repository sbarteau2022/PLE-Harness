"""Strict, model-neutral admission checks for live ablation receipts."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import pathlib
import re
from typing import Any


SHA256 = re.compile(r"^[0-9a-f]{64}$")
ATOM = re.compile(r"^atom:[A-Za-z0-9_-]+$")
TRAJECTORY = re.compile(r"^distilled_trajectory:[A-Za-z0-9_-]+$")


class ReceiptError(ValueError):
    """A receipt is incomplete, incomparable, or not live evidence."""


def _require(mapping: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ReceiptError(f"{where} missing fields: {', '.join(missing)}")


def _timestamp(value: Any, where: str) -> dt.datetime:
    if not isinstance(value, str):
        raise ReceiptError(f"{where} must be an ISO-8601 string")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReceiptError(f"{where} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ReceiptError(f"{where} must include a timezone")
    return parsed


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_receipt(receipt: dict[str, Any], *, require_mind: bool = True) -> None:
    """Reject anything that cannot enter the final live-results table."""
    _require(receipt, (
        "schema_version", "run_id", "campaign", "harness", "model", "task",
        "seed", "config", "timing", "live", "exit", "judges", "deliverables",
        "calls", "tools", "tokens", "latency", "memory", "process",
        "provider_cost", "unavailable",
    ), "receipt")
    if receipt["schema_version"] != 1:
        raise ReceiptError("unsupported receipt schema")
    for name in ("run_id", "campaign", "task"):
        if not isinstance(receipt[name], str) or not receipt[name].strip():
            raise ReceiptError(f"{name} must be nonempty")

    harness, model = receipt["harness"], receipt["model"]
    if not isinstance(harness, dict) or not isinstance(model, dict):
        raise ReceiptError("harness and model must be objects")
    _require(harness, ("id", "version", "source_digest"), "harness")
    _require(model, ("id", "endpoint_class", "response_ids"), "model")
    if not SHA256.fullmatch(str(harness["source_digest"])):
        raise ReceiptError("harness source_digest must be SHA-256")
    if model["endpoint_class"] not in {"nvidia-api", "local"}:
        raise ReceiptError("unknown endpoint_class")
    if not isinstance(model["response_ids"], list) or not model["response_ids"]:
        raise ReceiptError("live response_ids are required")
    if any(not isinstance(value, str) or not value for value in model["response_ids"]):
        raise ReceiptError("response_ids must be nonempty strings")

    timing = receipt["timing"]
    _require(timing, ("started_at", "ended_at", "wall_seconds"), "timing")
    started = _timestamp(timing["started_at"], "timing.started_at")
    ended = _timestamp(timing["ended_at"], "timing.ended_at")
    if ended < started or not isinstance(timing["wall_seconds"], (int, float)) or timing["wall_seconds"] < 0:
        raise ReceiptError("timing is inconsistent")
    if receipt["live"] is not True:
        raise ReceiptError("scripted, replayed, or simulated runs are inadmissible")

    exit_state = receipt["exit"]
    _require(exit_state, ("state", "unresolved_error", "finish_refusals"), "exit")
    if exit_state["state"] != "complete" or exit_state["unresolved_error"] is not None:
        raise ReceiptError("run did not close cleanly")

    judges = receipt["judges"]
    if not isinstance(judges, list) or not judges:
        raise ReceiptError("at least one judge is required")
    for judge in judges:
        _require(judge, ("id", "passed", "score", "receipt_digest"), "judge")
        if judge["passed"] is not True or not SHA256.fullmatch(str(judge["receipt_digest"])):
            raise ReceiptError("every judge must pass with a receipt digest")

    deliverables = receipt["deliverables"]
    if not isinstance(deliverables, list) or not deliverables:
        raise ReceiptError("deliverable hashes are required")
    for item in deliverables:
        _require(item, ("path", "sha256", "bytes"), "deliverable")
        if not SHA256.fullmatch(str(item["sha256"])) or not isinstance(item["bytes"], int) or item["bytes"] < 0:
            raise ReceiptError("invalid deliverable receipt")

    for name, fields in {
        "calls": ("steps", "retries", "no_progress_turns", "replans"),
        "tools": ("offered", "calls", "successes", "errors", "by_name", "repeated_cycles"),
        "tokens": ("prompt", "output", "cached", "uncached", "max_context"),
        "latency": ("ttft_ms", "e2e_ms", "decode_tokens_per_second"),
        "memory": ("enabled", "recalls", "atom_reads", "minted_atoms", "trajectory_reads", "minted_trajectory", "causal_traces"),
        "process": ("peak_rss_bytes",),
    }.items():
        value = receipt[name]
        if not isinstance(value, dict):
            raise ReceiptError(f"{name} must be an object")
        _require(value, fields, name)
    if receipt["tools"]["calls"] < 1:
        raise ReceiptError("at least one real tool execution is required")

    memory = receipt["memory"]
    if require_mind:
        if not memory["minted_atoms"] or any(not ATOM.fullmatch(str(atom)) for atom in memory["minted_atoms"]):
            raise ReceiptError("validated minted atoms are required")
        if not TRAJECTORY.fullmatch(str(memory["minted_trajectory"])):
            raise ReceiptError("validated minted trajectory is required")
    if receipt["provider_cost"] is None and "provider_cost" not in receipt["unavailable"]:
        raise ReceiptError("unavailable provider cost requires a reason")


def parity_projection(receipt: dict[str, Any]) -> dict[str, Any]:
    """Fields that must be identical across harnesses for one comparison cell."""
    return copy.deepcopy({
        "campaign": receipt.get("campaign"),
        "model": receipt.get("model", {}).get("id"),
        "endpoint_class": receipt.get("model", {}).get("endpoint_class"),
        "task": receipt.get("task"),
        "seed": receipt.get("seed"),
        "config": receipt.get("config"),
    })


def assert_parity(receipts: list[dict[str, Any]]) -> None:
    if len(receipts) < 2:
        raise ReceiptError("parity requires at least two receipts")
    expected = canonical_digest(parity_projection(receipts[0]))
    if any(canonical_digest(parity_projection(receipt)) != expected for receipt in receipts[1:]):
        raise ReceiptError("manifest parity failed")


def assert_mind_ab(off: dict[str, Any], on: dict[str, Any]) -> None:
    """The only configured difference may be the mind flag; on must rescue off."""
    left, right = parity_projection(off), parity_projection(on)
    left_config, right_config = left["config"], right["config"]
    if not isinstance(left_config, dict) or not isinstance(right_config, dict):
        raise ReceiptError("mind A/B config is missing")
    left_config.pop("mind_enabled", None)
    right_config.pop("mind_enabled", None)
    if canonical_digest(left) != canonical_digest(right):
        raise ReceiptError("mind A/B changed fields other than mind availability")
    if off.get("exit", {}).get("state") == "complete" and all(j.get("passed") for j in off.get("judges", [])):
        raise ReceiptError("mind-off run did not fail; this is not a rescue")
    validate_receipt(on, require_mind=True)
    traces = on["memory"]["causal_traces"]
    if not isinstance(traces, list) or not traces:
        raise ReceiptError("mind rescue requires a causal trace")
    for trace in traces:
        _require(trace, ("source_id", "read_step", "action_step", "artifact", "judge_id"), "causal trace")
        if trace["read_step"] >= trace["action_step"]:
            raise ReceiptError("causal source must be read before the influenced action")


def load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReceiptError("receipt root must be an object")
    return value

