# Next tasks beyond the existing PLE handoff

Date: 2026-11-16. Scope: `harness/` only. This packet is **additive** to [HANDOFF.md](HANDOFF.md). It does not replace the frozen [cross-harness campaign](../ablations/PREREGISTRATION.md) or [recall-first study](recall-first-forward/PREREGISTRATION.md). As of this audit, `ablations/api/results.json` has no admitted full campaign cells. Do not present the following as completed capabilities.

## 1. Make documentation and evidence navigable

**Why now:** the operator handoff contains older dated sections whose statements are superseded by later live probes; an ignored `attention-economics/` tree duplicated tracked docs and figures. **Work:** maintain [the documentation index](README.md), mark dated handoff sections as historical, check repository links and evidence paths in CI, and keep one canonical copy of each design. **Done when:** a reader can reach every governing design, preregistration, run receipt, failure ledger, and current status from README without relying on files outside the Git repo. Historical snapshots stay labeled as history. Do not silently rewrite frozen preregistrations.

## 2. Make research ingestion atomic for new preprints

**Current gap:** `Rag.ingest` indexes extracted text; `scripts/migrate_research_library_to_edge.py` archives original bytes. A frontier research call does not yet prove both occurred for each newly relied-on paper. **Work:** build a single model-neutral ingestion transaction that verifies the primary source, stores exact original bytes and full extractable text in the existing Edge shard, reopens both, checks SHA-256 and live BM25 retrieval, and writes a receipt. Preserve dense vectors and shard locking. **Done when:** interrupted/duplicate/corrupt-source tests fail closed and a new verified public preprint is restorable and searchable after restart. Do not ingest task/private material.

## 3. Graduate the engineering verifier from inline manifest to real deliverables

**Current gap:** `engineering_verify` checks only declared boxes/routes/clearances/flow states. It does not extract geometry from files or validate IFC. **Work:** design typed adapters for task-provided 2D drawings, IFC4/STEP, DXF, and schedule tables; carry source-page, units, transforms, and uncertainty into a provenance manifest; independently reopen exported deliverables. **Done when:** a drawing-to-IFC task can produce a machine-readable reconciliation report, schema/geometry findings, and hashes that correspond to the actual exported file. Test scale ambiguity, mixed units, unconnected routes, self-intersections, missing properties, and missing source dimensions. Do not label a bounding-box pass as CAD validity.

## 4. Add a checked 4D systems-engineering path

**Current gap:** `artifact_render` draws time-indexed graphs but has no hydraulic or thermal solver. **Work:** design a model-neutral component graph and state schema linked to CAD object IDs, with pump/valve/tank boundary conditions, timestep, schedule events, and units. Compute network flow/head and optional temperature using a declared solver; render only those computed states. **Done when:** mass balance, resource limits, monotonic schedule, energy balance where claimed, zero-flow/closed-valve cases, and deterministic replay pass. Compare calculations against an independent reference. Keep CFD and field-calibration claims separate.

## 5. Build the Meta Cognitive Automaton as an outside verifier

**Current gap:** finish ordering and repeated-state guards exist, but no typed external controller judges whether new work, recall, or another call is justified. **Work:** define a deterministic event schema over response IDs, tool receipts, atom provenance, task gates, repeated actions, time/token budgets, and deliverable inspection. It may request evidence, revise, stop, or reject a claim with reason codes. **Done when:** adversarial trajectories show it catches unseen-output claims, repeated no-progress calls, unverified atom use, invalid deliverables, and budget overrun without suppressing a genuinely improving run. Compare it against the current guard on matched tasks; freeze thresholds first. This is not a prompt persona or a substitute for independent task judges.

## 6. Add optional HyperDX observability without moving authority

**Current gap:** PLE receipts measure calls and tokens locally; there is no verified PLE HyperDX export/readback. **Work:** map existing immutable run IDs to OpenTelemetry traces for model, tool, retrieval, engineering-check, and judge phases; redact credentials/task-private content and keep local receipt files authoritative. **Done when:** trace spans reconcile to the run receipt, missing export is explicit, and a live readback confirms the same IDs and timing fields. No physical-energy field without a calibrated meter.

## 7. Turn these into product-development qualification tasks

Select public engineering tasks spanning drawing interpretation, IFC generation, routing, calculation, schedule, and deliverable revision. Freeze task bytes and judge rules before a new ablation. Compare current PLE against each new capability one at a time, then test combined operation. Report completed-task rate, hard-constraint failures, tool errors, interventions, attempts, latency, tokens, and cost by model and task. A claim that the multimodel mind rescued a task requires a verified mind-off failure, mind-on completion, and a cited atom/trajectory read that changed the action. Preserve every incomplete or failed trial in a separate ledger.

## Execution order

Start with documentation/evidence integrity and atomic research ingestion; then file-based engineering verification; then the 4D path and outside verifier. Instrumentation must precede any comparison that uses it. The existing campaign and recall-first packets remain their own preregistered tracks. Update README and tests with every implementation. No cross-model, cost, quality, or thermodynamic claim advances from component tests alone.
