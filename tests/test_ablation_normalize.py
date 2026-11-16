from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from ablations.normalize import (MARKER, build_receipt, build_results,
                                 extract_markers, hash_deliverables,
                                 parse_opencode_tool_calls, record_trajectory,
                                 response_ids_from_events, write_json_atomic)
from ablations.receipts import ReceiptError


def test_hash_deliverables_measures_bytes(tmp_path: pathlib.Path) -> None:
    (tmp_path / "a.txt").write_text("hey", encoding="utf-8")
    items = hash_deliverables(tmp_path, ["a.txt"])
    assert items[0]["sha256"] == hashlib.sha256(b"hey").hexdigest()
    assert items[0]["bytes"] == 3
    with pytest.raises(ValueError, match="missing"):
        hash_deliverables(tmp_path, ["gone"])
    with pytest.raises(ValueError, match="at least one"):
        hash_deliverables(tmp_path, [])


def test_parse_counts_tools_and_errors(tmp_path: pathlib.Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text("\n".join([
        json.dumps({"part": {"type": "text", "text": "hi"}}),
        json.dumps({"part": {"type": "tool", "tool": "bash", "state": {"status": "ok"}}}),
        json.dumps({"type": "tool_call", "tool": "bash", "state": {"status": "error", "error": "x"}}),
        json.dumps({"part": {"type": "tool", "name": "read", "state": {}}}),
        "not json",
    ]), encoding="utf-8")
    counts = parse_opencode_tool_calls(events)
    assert counts == {"calls": 3, "errors": 1, "by_name": {"bash": 2, "read": 1}}


def test_record_trajectory_copies_and_hashes(tmp_path: pathlib.Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps({"part": {"type": "tool", "tool": "bash", "state": {}}}) + "\n",
                      encoding="utf-8")
    record = record_trajectory(events, tmp_path / "runs", "r1")
    assert record["path"] == "r1-trajectory.jsonl" and record["calls"] == 1
    assert (tmp_path / "runs" / "r1-trajectory.jsonl").is_file()
    with pytest.raises(ValueError, match="missing or empty"):
        record_trajectory(tmp_path / "gone.jsonl", tmp_path / "runs", "r2")


def _receipt_kwargs() -> dict:
    return {
        "run_id": "r1", "campaign": "c", "harness_id": "harness2",
        "harness_version": "6868817", "harness_digest": "a" * 64,
        "model_id": "m", "endpoint_class": "nvidia-api", "response_ids": ["live-1"],
        "task": "t", "seed": 7,
        "config": {"budgets": {}, "tools": [], "judges": ["j"], "sampling": {}},
        "started_at": "2026-11-15T00:00:00Z", "ended_at": "2026-11-15T00:00:01Z",
        "wall_seconds": 1.0,
        "judges": [{"id": "j", "passed": True, "score": 1, "receipt_digest": "b" * 64}],
        "deliverables": [{"path": "a", "sha256": "c" * 64, "bytes": 1}],
        "calls": {"steps": 1, "retries": 0, "no_progress_turns": 0, "replans": 0},
        "tools": {"offered": ["bash"], "calls": 1, "successes": 1, "errors": 0,
                  "by_name": {"bash": 1}, "repeated_cycles": 0},
        "tokens": {"prompt": 1, "output": 1, "cached": 0, "uncached": 1, "max_context": 2},
        "latency": {"ttft_ms": [1], "e2e_ms": [2], "decode_tokens_per_second": [1]},
        "memory": {"enabled": True, "recalls": 0, "atom_reads": [], "minted_atoms": ["atom:1"],
                   "trajectory_reads": [], "minted_trajectory": "distilled_trajectory:t",
                   "causal_traces": []},
        "peak_rss_bytes": None, "provider_cost": None,
        "unavailable": {"provider_cost": "not reported", "process.peak_rss_bytes": "not measured"},
    }


def test_build_receipt_validates_and_rejects_replay() -> None:
    receipt = build_receipt(**_receipt_kwargs())
    assert receipt["live"] is True and receipt["exit"]["state"] == "complete"
    bad = _receipt_kwargs()
    bad["response_ids"] = []
    with pytest.raises(ReceiptError):
        build_receipt(**bad)


def test_write_json_atomic_round_trips(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "sub" / "results.json"
    write_json_atomic(target, {"b": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"b": 1}
    assert list(tmp_path.rglob("*.tmp")) == []


def test_response_ids_and_markers_round_trip(tmp_path: pathlib.Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text("\n".join([
        json.dumps({"id": "sess-1", "part": {"type": "text", "id": "msg-1"}}),
        json.dumps({"part": {"type": "tool", "tool": "bash", "messageId": "msg-2"}}),
        json.dumps({"id": "sess-1", "part": {"type": "text", "id": "msg-1"}}),
    ]), encoding="utf-8")
    assert response_ids_from_events(events) == ["msg-1", "msg-2"]
    text = f"noise\n{MARKER} " + '{"a": 1}\n' + f"{MARKER} " + "broken\n"
    assert extract_markers(text) == [{"a": 1}]


def test_build_results_requires_state_and_evidence() -> None:
    table = build_results(campaign="c", seed=1, cells=[
        {"harness": "h", "model": "m", "task": "t", "state": "complete", "evidence": ["a/b"]},
        {"harness": "h", "model": "m", "task": "t2", "state": "failed", "evidence": ["a/c"]},
    ])
    assert [c["state"] for c in table["cells"]] == ["complete", "failed"]
    with pytest.raises(ValueError, match="valid state"):
        build_results(campaign="c", seed=1, cells=[{"state": "done", "evidence": ["x"]}])
    with pytest.raises(ValueError, match="evidence"):
        build_results(campaign="c", seed=1, cells=[
            {"harness": "h", "model": "m", "task": "t", "state": "complete", "evidence": []}])


def test_classify_run_states(tmp_path: pathlib.Path) -> None:
    import sys
    sys.path.insert(0, "scripts")
    import write_results
    run = tmp_path / "run1"
    run.mkdir()
    assert write_results.classify_run(run, tmp_path)[0] == "incomplete"
    (run / "results.json").write_text('{"tasks": [{"id": "t", "measured": false}]}', encoding="utf-8")
    assert write_results.classify_run(run, tmp_path)[0] == "failed"
    (run / "results.json").write_text('{"tasks": [{"id": "t", "measured": true}]}', encoding="utf-8")
    assert write_results.classify_run(run, tmp_path)[0] == "incomplete"
