#!/usr/bin/env python3
"""Mint and verify harness-side atoms for one closed live probe result."""
from __future__ import annotations
import json, os, pathlib, sys

ROOT=pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from ablations.mint import mint_run, verify_readback
from ablations.normalize import write_json_atomic
from frontier import surreal

def main(path: pathlib.Path) -> int:
    data=json.loads(path.read_text())
    if not data.get("closed"): raise SystemExit("refusing to mint an unclosed probe")
    if data.get("mint",{}).get("readback"):
        print(data["mint"]["trajectory"]); return 0
    run_id=path.parent.name
    result=mint_run(lambda: surreal.Mind("nvidia",data["model"],f"probe://{data['harness']}/{data['seat']}/{run_id}"),
      campaign="live-multiturn-27-call-v1",harness=data["harness"],model=data["model"],task="multiturn-27-call",
      run_id=run_id,summary=f"closed={data['closed']} calls={data['actual_tool_calls']} repeats={data['prescribed_repeated_calls']}",
      deliverables=data["deliverables"])
    readback=verify_readback(surreal.atom_get,surreal.trajectory_get,result)
    data["mint"]={"kind":"harness-side-observation","atoms":list(result.atoms),"trajectory":result.trajectory,
                  "admission":result.admission,"readback":readback}
    write_json_atomic(path,data); print(result.trajectory); return 0

if __name__=="__main__":
    if len(sys.argv)!=2: raise SystemExit("usage: mint_multiturn_probe.py PATH/results.json")
    raise SystemExit(main(pathlib.Path(sys.argv[1]).resolve()))
