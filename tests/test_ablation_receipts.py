from __future__ import annotations

import copy

import pytest

from ablations.receipts import ReceiptError, assert_mind_ab, assert_parity, validate_receipt


def valid() -> dict:
    return {
        "schema_version": 1, "run_id": "r1", "campaign": "c", "task": "t", "seed": 7,
        "harness": {"id": "ple", "version": "v", "source_digest": "a" * 64},
        "model": {"id": "m", "endpoint_class": "local", "response_ids": ["live-1"]},
        "config": {"budgets": {"steps": 2}, "tools": ["bash"], "judges": ["j"], "sampling": {}, "mind_enabled": True},
        "timing": {"started_at": "2026-11-15T00:00:00Z", "ended_at": "2026-11-15T00:00:01Z", "wall_seconds": 1},
        "live": True, "exit": {"state": "complete", "unresolved_error": None, "finish_refusals": 0},
        "judges": [{"id": "j", "passed": True, "score": 1, "receipt_digest": "b" * 64}],
        "deliverables": [{"path": "answer", "sha256": "c" * 64, "bytes": 1}],
        "calls": {"steps": 2, "retries": 0, "no_progress_turns": 0, "replans": 0},
        "tools": {"offered": ["bash"], "calls": 1, "successes": 1, "errors": 0, "by_name": {"bash": 1}, "repeated_cycles": 0},
        "tokens": {"prompt": 10, "output": 2, "cached": 0, "uncached": 10, "max_context": 10},
        "latency": {"ttft_ms": [1], "e2e_ms": [2], "decode_tokens_per_second": [2]},
        "memory": {"enabled": True, "recalls": 1, "atom_reads": ["atom:old"], "minted_atoms": ["atom:new"], "trajectory_reads": [], "minted_trajectory": "distilled_trajectory:walk", "causal_traces": []},
        "process": {"peak_rss_bytes": 1}, "provider_cost": None,
        "unavailable": {"provider_cost": "provider did not report cost"},
    }


def test_valid_receipt_is_admitted() -> None:
    validate_receipt(valid())


@pytest.mark.parametrize("change", ["replay", "unfinished", "judge", "usage", "trajectory", "tool"])
def test_incomplete_evidence_is_rejected(change: str) -> None:
    receipt = valid()
    if change == "replay": receipt["live"] = False
    if change == "unfinished": receipt["exit"]["state"] = "timeout"
    if change == "judge": receipt["judges"][0]["passed"] = False
    if change == "usage": del receipt["tokens"]["cached"]
    if change == "trajectory": receipt["memory"]["minted_trajectory"] = None
    if change == "tool": receipt["tools"]["calls"] = 0
    with pytest.raises(ReceiptError):
        validate_receipt(receipt)


def test_cross_harness_parity_rejects_budget_change() -> None:
    first, second = valid(), valid()
    second["harness"]["id"] = "harness2"
    assert_parity([first, second])
    second["config"]["budgets"]["steps"] = 3
    with pytest.raises(ReceiptError, match="parity"):
        assert_parity([first, second])


def test_mind_rescue_requires_prior_trace_and_exact_pair() -> None:
    off, on = valid(), valid()
    off["config"]["mind_enabled"] = False
    off["exit"]["state"] = "budget_exhausted"
    off["judges"][0]["passed"] = False
    on["memory"]["causal_traces"] = [{"source_id": "atom:old", "read_step": 1, "action_step": 2, "artifact": "answer", "judge_id": "j"}]
    assert_mind_ab(off, on)
    broken = copy.deepcopy(on)
    broken["memory"]["causal_traces"][0]["action_step"] = 1
    with pytest.raises(ReceiptError, match="before"):
        assert_mind_ab(off, broken)
