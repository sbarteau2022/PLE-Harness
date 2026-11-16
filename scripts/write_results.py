#!/usr/bin/env python3
"""Assemble ablations/api/results.json from campaign.json and the runs tree.

Cell states:
- complete: a closed run with a validating admitted receipt.
- incomplete: no closed admitted run yet (UNMET), or a run closed without one.
- failed: the latest attempt errored (evidence points at the failed run).
- excluded: the cell was attempted and ledgered as inadmissible.

Only `complete` cells may enter the final admitted table.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ablations.normalize import build_results, write_json_atomic  # noqa: E402
from ablations.receipts import ReceiptError, validate_receipt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def classify_run(run_dir: pathlib.Path, root: pathlib.Path = ROOT) -> tuple[str, list[str], dict | None]:
    """Classify one kit output directory; return (state, evidence, receipt)."""
    evidence = [str(run_dir.relative_to(root))]
    results_path = run_dir / "results.json"
    if not results_path.is_file():
        return "incomplete", evidence, None
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "failed", evidence, None
    tasks = results.get("tasks", [])
    if not tasks or any(not t.get("measured") for t in tasks):
        return "failed", evidence, None
    receipt_path = run_dir / "receipt.json"
    if not receipt_path.is_file():
        return "incomplete", evidence, None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        validate_receipt(receipt)
    except (json.JSONDecodeError, UnicodeDecodeError, ReceiptError):
        return "failed", evidence, None
    return "complete", evidence + [str(receipt_path.relative_to(root))], receipt


def main() -> int:
    campaign = json.loads((ROOT / "ablations" / "campaign.json").read_text(encoding="utf-8"))
    tasks = campaign["task_suite"]["tasks"]
    seats = {m["seat"]: m["model"] for m in campaign["api_models"]}
    cells: list[dict] = []
    ledger = "ablations/FAILURES.md"
    for harness in campaign["harness_order"]:
        for seat, model in seats.items():
            for task in tasks:
                runs_root = ROOT / "ablations" / "runs" / harness / seat
                runs = sorted(p for p in runs_root.glob("*") if p.is_dir()) if runs_root.is_dir() else []
                task_runs = [p for p in runs if task in _tasks_of(p)]
                if not task_runs:
                    cells.append({"harness": harness, "model": model, "seat": seat, "task": task,
                                  "state": "incomplete", "detail": "UNMET: no attempt",
                                  "evidence": ["ablations/PREREGISTRATION.md"]})
                    continue
                latest = task_runs[-1]
                state, evidence, receipt = classify_run(latest)
                cell: dict = {"harness": harness, "model": model, "seat": seat, "task": task,
                              "state": state, "run": latest.name, "evidence": evidence}
                if state != "complete":
                    cell["detail"] = "UNMET" if state == "incomplete" else "see failure ledger"
                    cell["evidence"] = evidence + [ledger]
                else:
                    assert receipt is not None
                    cell["receipt_digest"] = receipt["judges"][0]["receipt_digest"]
                    cell["minted_atoms"] = receipt["memory"]["minted_atoms"]
                    cell["minted_trajectory"] = receipt["memory"]["minted_trajectory"]
                cells.append(cell)
    table = build_results(campaign=campaign["campaign"], seed=campaign["seed"], cells=cells)
    write_json_atomic(ROOT / "ablations" / "api" / "results.json", table)
    states = [c["state"] for c in cells]
    print(f"cells={len(cells)} complete={states.count('complete')} "
          f"incomplete={states.count('incomplete')} failed={states.count('failed')} "
          f"excluded={states.count('excluded')}")
    return 0


def _tasks_of(run_dir: pathlib.Path) -> list[str]:
    try:
        results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    return [t.get("id", "") for t in results.get("tasks", [])]


if __name__ == "__main__":
    raise SystemExit(main())
