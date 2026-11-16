# Cross-harness live ablation design

Date: 2026-11-15. Status: PREREGISTERED.

## Research question

[PROPOSED] For the same task packet, model endpoint, tool surface, budgets, and
completion judges, how do Harness2, DeepSeek Harness, and the PLE harness differ
in task success, tool reliability, token/context efficiency, latency,
throughput, recovery, and admitted atom/trajectory production? This decides
which mechanisms should remain in the PLE harness and supports a later harness
write-up only if live evidence favors it.

## Boundary

[PROPOSED] All executable adapters, manifests, receipts, logs, and reports live
under this `harness/` tree. External source may be read for provenance, but a
run is admitted only from a pinned copy or installed package identified in its
receipt. No model-specific layer count, tensor geometry, prompt rewrite, or
Glimmer-only path is part of this comparison.

[PROPOSED] API phase order is Harness2, DeepSeek Harness, then PLE, with NVIDIA
Mistral Large 4, Step 5, and Nemotron Ultra 550B. The local phase repeats the
same harness order one model at a time for Qwen 3.6 Cyber, Nemotron Omni 30B,
and Glimmer 30B. A model begins only after the preceding model has complete
admitted runs.

## Local multi-model-mind A/B

[PROPOSED] A second research run is local-model-only and compares the PLE
harness with `FRONTIER_MIND=off` against the same harness with the shared mind
enabled. Model artifact, inference server, task state, tools, budgets, seed,
sampling controls, and judges remain fixed. Each model runs one task at a time.

[PROPOSED] The primary endpoint is rescue: a mind-enabled trial passes every
task judge on a task where a matched mind-off trial, or a compatible preserved
historical trial, verifiably failed. Historical failures are admissible only
when model identity, task revision, tool surface, and budgets match; otherwise a
fresh mind-off baseline is required.

[PROPOSED] Secondary endpoints are artifact-quality score, valid and successful
tool-call rates, attempts and retries, repeated-call cycles, no-progress turns,
premature finishes, replan/stop events, wall time, and tokens. Mind-enabled
runs additionally record recalled atom IDs, atom reads, in-motion mints with
step/timestamp, trajectory reads/writes, and the final minted trajectory.

[PROPOSED] A mint count is not evidence of useful learning. A claimed mind
benefit requires traceable use: the recalled atom or trajectory must predate the
decision it allegedly influenced, its content must be relevant to an action or
artifact change, and the same claim must survive the task judge. Thrashing is a
predeclared repeated-state/tool-call pattern or exhaustion without judge-passing
progress; ordinary iterative correction is not labeled thrashing.

## Common observation contract

[PROPOSED] Every admitted run records harness/model identity and digest,
endpoint class, task and seed, budgets/configuration, start/end timestamps,
exit state, judge outcomes, deliverable hashes, calls/steps, tool calls by name,
tool successes/errors/retries, premature-finish refusals, prompt/output/cached/
uncached tokens, maximum context, TTFT and end-to-end latency distributions,
decode throughput, wall time, archive/recall activity, atoms, trajectories,
process peak RSS, and provider-reported cost when available. Missing provider
fields are `null`, never estimated as measured values.

[PROPOSED] Quality and task completion are primary. Efficiency comparisons are
valid only among runs that pass the same task judges. Synthetic, replay-only,
partial, manually repaired, or judge-incomplete runs are excluded from the
final table but retained in an failures ledger.

## Evidence labels

[PROPOSED] Findings use VERIFIED for direct measurements/reads, REPORTED for a
source claim, INFERRED with an explicit evidence boundary, or PROPOSED for a
plan. Corrections are appended; measurements are never silently overwritten.

## Amendment 2026-11-16: generic baseline replaces Harness2

[VERIFIED] Harness2 did not pass its frozen dependency packaging gate and
produced no admitted live task. The user replaced that arm with a generic
baseline. This correction is appended; prior attempts remain excluded.

[PROPOSED] Revised cross-harness order is `generic`, `deepseek-harness`, `ple`.
The existing common observation, fairness, admission, model, task, tool,
budget, judge, atom, and trajectory contracts above remain unchanged. The
separate local PLE mind A/B remains unchanged. This amendment changes only the
name and implementation of the superseded first arm; it defines no new study.

