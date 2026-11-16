#!/usr/bin/env python3
"""Build the canonical compact table from closed multi-turn probe runs."""
from __future__ import annotations
import json, pathlib, statistics, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

def latest(harness: str, seat: str):
    root = ROOT / "ablations" / "probes" / harness / seat
    rows = []
    for path in sorted(root.glob("*/results.json")) if root.is_dir() else []:
        try: value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError): continue
        rows.append((path, value))
    return rows[-1] if rows else None

def main() -> int:
    cells=[]
    for seat in ("mistral","step5","nemotron-ultra"):
        for harness in ("generic","ple"):
            found=latest(harness, seat)
            if not found:
                cells.append({"harness":harness,"seat":seat,"state":"incomplete"}); continue
            path,d=found; t=d["telemetry"]; latency=t.get("e2e_ms",[])
            observed_errors = 0
            trajectory = path.parent / "trajectory.jsonl"
            if trajectory.is_file():
                for line in trajectory.read_text().splitlines():
                    try: event=json.loads(line)
                    except json.JSONDecodeError: continue
                    if event.get("kind") == "tool" and event.get("ok") is False: observed_errors += 1
            cells.append({"harness":harness,"seat":seat,"state":"complete" if d.get("closed") else "excluded","evidence":str(path.relative_to(ROOT)),
              "wall_seconds":d["wall_seconds"],"steps":t.get("steps"),"tool_calls":d["actual_tool_calls"],
              "repeated_calls":d["prescribed_repeated_calls"],"tool_errors":observed_errors,
              "finish_refusals":t.get("finish_refusals"),"prompt_tokens":t.get("prompt"),
              "output_tokens":t.get("output"),"cached_tokens":t.get("cached"),"max_context":t.get("max_context"),
              "median_e2e_ms":statistics.median(latency) if latency else None,"max_e2e_ms":max(latency) if latency else None,
              "deliverables":d["deliverables"],"trajectory":str((path.parent/'trajectory.jsonl').relative_to(ROOT)),
              "minted_atoms":d.get("mint",{}).get("atoms",[]),
              "minted_trajectory":d.get("mint",{}).get("trajectory"),
              "mint_readback":bool(d.get("mint",{}).get("readback"))})
    out={"schema_version":1,"probe":"live-multiturn-27-call-v1","cells":cells}
    target=ROOT/"ablations"/"probes"/"results.json"; tmp=target.with_suffix(".tmp")
    tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); tmp.replace(target)
    print(target); return 0

if __name__ == "__main__": raise SystemExit(main())
