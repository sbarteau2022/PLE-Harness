# Recall-first forward and attention economics preregistration

Date: 2026-11-16. Status: FROZEN PROPOSAL before implementation or live policy runs.
Design: `DESIGN.md`. Any change after first live run is an appended amendment
with a new arm ID; earlier receipts remain immutable.

## Hypotheses and decision

[PROPOSED] H1: recall-first increases independently judged task success or
artifact quality against the existing PLE path at matched model, task,
sampling, tool surface, and budget. H2: after counting scout and retrieval,
recall-first reduces total uncached prompt tokens or total prefill time without
violating H1's noninferiority gate. H3 (local branch-capable runtimes only):
clean task-prefix branch continuation is numerically equivalent to clean
re-prefill and reduces repeated task-prefix prefill time. H4 (separate sparse
attention arm only): measured attention work falls with context length while
quality remains within its preregistered margin.

## Arms and units

[PROPOSED] A0 is current PLE task-first prefill with existing optional recall.
A1 is scout → bounded Surreal recall → main prefill, with no branch reuse. A2
is A1 with a clean local KV/state branch when the runtime passes P3. A3 is a
separate sparse-page attention candidate and cannot be mixed into A1/A2.
Pair arms within a model, task revision, seed, serving build, sampling profile,
tools, and judge. Order pairs AB/BA by a recorded deterministic seed. Run three
replicates per pair. Report each model separately before pooling; models with
no qualified local runtime cannot enter A2/A3.

[PROPOSED] Select 12 tasks before the first run: four short/no-recall controls,
four long-history retrieval tasks, and four tasks with a predeclared relevant
atom/trajectory roster. Freeze task bytes and Surreal snapshot digest. For
quality claims, tasks must have independent graders or blinded rubric judges;
the 27-call tool probe is only a protocol check. A user may supply a different
frozen task roster before execution, with an appended preregistration version.

## Limits held fixed

[PROPOSED] Per task: 4 scout queries maximum, 16 candidate atoms maximum, 8
admitted atoms maximum, 4,000 admitted text tokens maximum, 512 scout output
tokens maximum, 30-minute task timeout, and 400 agent steps. These are design
caps, not measured values. A0 receives the same total task time and model-call
budget; A1/A2 scout calls count against them. Match main generation limit and
sampling settings. Capture all runtime/model identities and hardware state.

## Primary and secondary endpoints

[PROPOSED] Primary quality endpoint: fraction of paired tasks passing all
independent judges. Primary economics endpoint: total uncached prompt tokens
per completed task for API seats, and measured end-to-end wall time per
completed task for local seats. Neither endpoint is evaluated only on lucky
successes: failures count in quality and their consumed resources are reported.
Secondary endpoints: judge score, deliverable hashes, total token/call count,
scout and main phase times, prefill kernel time, decode time, tool time,
retrieval time, cache hit/miss, GPU/CPU utilization, peak memory, provider cost,
model-reported cache reads, page bytes, and joules only when a calibrated meter
and idle subtraction receipt exist.

[PROPOSED] For each task and arm, count success when at least two of its three
replicates pass; retain all 36 trial outcomes in the report. A1 passes H1 if
its 12-task success count does not fall below A0 by more than one task per
model and no critical safety or provenance failure occurs. H2 passes only if
H1 passes and median paired
uncached-token or local wall-time ratio is ≤0.90 with a bootstrap 95% interval
whose upper bound is <1.00. A quality-superiority claim requires at least two
additional task passes per model with no regression on controls. H3
requires exact generated IDs and max absolute logit error ≤1e-5 in f32 or a
runtime-declared tolerance justified before testing for lower precision, plus
median task-prefix prefill time ratio ≤0.90. H4 requires an explicit operation
counter or kernel trace showing fewer key reads than exact dense attention
over at least four preregistered increasing context lengths, quality within
H1's margin, and end-to-end timing. Finite-length scaling cannot prove an
asymptotic bound; otherwise H4 remains a component claim.

## Deterministic judges and adversarial gates

| ID | Gate | Failure disposition |
|---|---|---|
| J1 | Frozen task/model/tool/seed/limit and Surreal snapshot digests match pair | Exclude pair |
| J2 | Every admitted atom is active, same-workspace, digest-valid, taint-allowed, and cited by ID | Reject candidate/run |
| J3 | Scout JSON obeys `RecallPlanV1`; 0–4 queries and bounded output | Fail closed to A0-compatible task-only path; count failure |
| J4 | No scout tool/database call or raw SurrealQL reaches model-controlled execution | Reject run |
| J5 | Local branch receipt covers every cache/state buffer and exact task prefix | Refuse branch; use measured re-prefill arm |
| J6 | Fork and clean re-prefill compare logits/IDs after restart, changed prefix/weights/tokenizer/position, corrupt and missing state | A2 blocked on any mismatch |
| J7 | Phase counters reconcile to wall time; token counters to provider/runtime receipts; absent fields are null with reason | Exclude economics claim |
| J8 | Independent quality judges and deliverable hashes close; all failed attempts remain in ledger | Exclude incomplete run |
| J9 | Paper figures regenerate from immutable run receipts, with environment and source provenance | No paper promotion |

## Analysis and confounds

[PROPOSED] Use paired differences and per-model confidence intervals, report
all attempts, and keep excluded runs visible. Record cold/warm provider cache,
network and server load, local thermal state, quantization, prompt template,
tool nondeterminism, retrieval index state, and atom availability. No
post-hoc change of the relevance threshold, task roster, or success margin.
Report H1–H4 as MET, UNMET, or NOT RUN with receipt paths.

[PROPOSED] The attention-economics paper will include definitions, exact
accounting equations, task/model/hardware roster, paired results, quality
outcomes, ablations, error bars, and limitations. It will distinguish API
token transport, runtime prefill, dense attention operations, sparse page
selection, and energy. It may report a negative result.

## Current disposition

[VERIFIED] No recall-first scout or KV branch has been implemented or run as
of this preregistration. All H1–H4 are NOT RUN. Existing exact-prefix tests,
synthetic restarts, and the prior 27-call tool probe are component evidence
only. No attention-economics paper result is established.

## Amendment A, 2026-11-16: FEP question before first policy run

[PROPOSED] `FEP-INTEGRATION.md` identifies the missing active-inference
contract. If an FEP claim is pursued, freeze F0 (deterministic cost-only
controller) and F1 (specified, calibrated active-inference controller) before
the first live policy run. Pair F0/F1 on the same tasks, candidate atoms,
model, budget, and seed as A1. Report quality, total resource cost,
calibration, unsafe-atom rejection, and loop termination. This amendment does
not change H1–H4 or confer FEP qualification. F0/F1 are NOT RUN.
