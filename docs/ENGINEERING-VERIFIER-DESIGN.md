# Engineering verification tool design

Date: 2026-11-16. Status: first implementation contract.

## Goal and boundary

Expose one model-neutral `engineering_verify` tool in the PLE driver and frontier pipeline. It consumes a bounded, inline engineering manifest and returns deterministic findings. It must not claim IFC schema compliance, CAD solid validity, fluid dynamics, or engineering approval. Those require separate task-local exporters and solvers. The tool runs in the harness; it receives no grader paths, network access, or project files.

## Manifest v1

`units` is one of `mm`, `m`, `in`, `ft`. `components` have unique IDs, an axis-aligned box `{x,y,z,dx,dy,dz}`, and optional `source` (drawing reference). `connections` identify component endpoints with `from`, `to`, and optional `flow` (signed volume per unit time). `states` are ordered records of `time` and `flows` keyed by connection ID. The manifest names `time_unit` and `flow_unit` when states are present. Optional `clearances` name two components, an axis, and a minimum gap. Inputs are capped to avoid tool-output and compute abuse.

## Checks and receipts

Reject malformed, nonfinite, unbounded, duplicate, or unknown-reference input. Check positive dimensions and source references, pairwise box overlap, requested axis clearances, graph reachability for named required routes, strictly increasing time, exact flow IDs per state, and flow conservation at declared junctions. Do not treat overlapping boxes as a universal failure: report them as findings; task constraints decide whether overlap is allowed. A pass means only the declared machine-checkable invariants passed. Return JSON with `passed`, `checks`, `findings`, input SHA-256, and limits. Findings have stable codes and IDs. The result must never be silently truncated to an apparent pass.

## Verification plan

Test through `agent.ToolRunner` and frontier registration. Cover valid graph, missing source, overlap, gap, unknown endpoint, disconnected route, nonfinite/negative geometry, non-monotonic time, missing flow, imbalance, size limits, and deterministic hashes. Preserve the existing PLE protocol and model API. Add the tool to README and engineering guide.
