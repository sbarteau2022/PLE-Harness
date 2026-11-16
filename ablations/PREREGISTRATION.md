# Cross-harness live ablation preregistration

Recorded: 2026-11-15 before source acquisition and campaign calls in this run.

## Acceptance criteria

1. [PROPOSED] Source identity: each harness has a provenance row containing
   URL/local origin, exact version or digest, license, and the relied-on claim.
2. [PROPOSED] Baseline: the PLE offline suite and protocol smoke pass before
   campaign execution; adapter tests cover malformed, timed-out, premature
   finish, missing usage, and incomplete trajectory cases.
3. [PROPOSED] Fairness: a machine-readable manifest pins the same task packet,
   tool contracts, budgets, seeds, endpoint model IDs, and judges per comparison.
4. [PROPOSED] Live-only admission: the endpoint must return live model output,
   tools must actually execute, all judges must pass, and the receipt must close
   normally. Replay and scripted demos cannot enter the final table.
5. [PROPOSED] Research state: each admitted research run records atom IDs and a
   trajectory ID. A claimed mint must be readable from the nested Surreal store
   and its receipt/digest must validate.
6. [PROPOSED] Completeness: every requested harness-model cell has a complete
   admitted run or remains explicitly UNMET; failed attempts stay in a separate
   immutable ledger and are rerun after the fault is fixed.
7. [PROPOSED] Reporting: one full table uses only admitted runs and separates
   observed values from derived values. No write-up claim upgrades component or
   simulation evidence into live comparative evidence.
8. [PROPOSED] Local mind A/B: each pair has identical model/task/runtime/tools/
   budgets/seed/sampling/judges and differs only in mind availability. A rescue
   claim requires a judge-passing mind-on run and a verified matched mind-off
   failure; otherwise it is UNMET.
9. [PROPOSED] Causal trace: every claimed useful recall names the atom or
   trajectory, its read step, the later changed action/artifact, and the judge
   it helped pass. Post-hoc or unused recall is reported but earns no benefit.

## Deterministic judges

- `source_receipt`: required provenance fields nonempty and digest format valid.
- `manifest_parity`: canonicalized task/tool/budget/seed/judge blocks identical.
- `run_closed`: exit state `complete`, no unresolved error, timestamps ordered.
- `task_passed`: every declared task judge passes and deliverable hashes exist.
- `telemetry_complete`: all common fields present; unavailable values explicit
  `null` with reason.
- `mind_closed`: minted atom roster nonempty, trajectory ID present, trajectory
  readback matches the roster and receipt digest.
- `final_table_closed`: exactly one admitted row for every requested cell; no
  failed or partial run ID appears in the table.
- `mind_ab_parity`: canonicalized pair fields are identical after removing only
  the mind-enabled flag and its resulting telemetry.
- `mind_rescue`: matched off-run fails, on-run passes, causal trace validates,
  and neither run violates its budget or completion contract.

## Confounds and falsifiers

[PROPOSED] Provider load, caching, model nondeterminism, task-order learning,
different tool implementations, and local thermal state are confounds and are
recorded or controlled. The PLE advantage is falsified if it does not improve a
predeclared primary outcome on judge-passing runs, or if any apparent gain
depends on unequal budgets/tools, missing telemetry, a model-specific path, or
excluded failures.

## Source provenance

| ID | URL / local origin | Version | License | Relied-on claim |
|---|---|---|---|---|
| S1 | https://github.com/google-research/harness2 ; `ablations/vendor/harness2` | `6868817518876de8fd170c6a8354daca273b6191` | Apache-2.0, license SHA-256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` | [VERIFIED] Harness2 implements Propose-Probe-Compose through a `BenchmarkAdapter` and OpenCode/Codex substrate; custom Codex Responses endpoints are supported. |
| S2 | https://github.com/deepseek-ai/deepseek-harness ; `ablations/vendor/deepseek-harness` | `39163dd30f703a79079e2d2446545cff3c453738` | MIT, license SHA-256 `ebb4f09972aee8608be255debaf78451a68e95c290f55c240dec2ecfa16ea6be` | [VERIFIED] DSH is a plugin harness whose pi-ai provider accepts declared OpenAI-compatible gateways and self-hosted routes. |
| S3 | `/Users/stewartbarteau/Desktop/PhyEngLastExam/ple-dev-set/tasks` | local 29-task snapshot observed 2026-11-15 | mixed task assets; not redistributed by this campaign | [VERIFIED] The campaign task roster contains 29 task directories named in `campaign.json`. |

[INFERRED] Harness2 does not natively expose the PLE kit protocol: its shipped
adapters target LAB, JobBench, and WorkBuddy. The evidence boundary is the
pinned adapter registry and CLI inspected at S1. A PLE adapter is therefore a
campaign integration, not an upstream Harness2 feature.

## Criteria disposition

Pending. Each criterion will be marked MET or UNMET with receipt paths.

