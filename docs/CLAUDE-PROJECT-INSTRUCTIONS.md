# Claude project instructions: Physical Engineering's Last Exam Harness

This is the expanded operating reference. For Claude's project-instructions field,
paste [`CLAUDE-PROJECT-INSTRUCTIONS-PASTE.md`](CLAUDE-PROJECT-INSTRUCTIONS-PASTE.md).
The canonical repository is `https://github.com/sbarteau2022/PLE-Harness`; locally it is `/Users/stewartbarteau/Desktop/PhyEngLastExam/harness`. Start each coding session with `docs/HANDOFF.md`, then the relevant design, preregistration, README, source, and tests. Everything that runs or is newly implemented belongs under `harness/`. Do not modify the pinned Google PLE kit, external hackathon harnesses, or task graders. Keep instructions and results model-agnostic; discover runtime capabilities rather than assuming model names, layer counts, or K/V shapes. Do not spawn subagents without explicit owner approval.

## Operating method and proof labels

Before implementation, write or update a design and preregistration with hypotheses, baselines, metrics, falsifiers, scope, budgets, and quality gates. Preserve the frozen version; amendments get new arm IDs. Work one change at a time. After implementation, update README, run focused and full suites, pressure-test edge cases, perform an adversarial review, and inspect end-to-end receipts. Distinguish **PROPOSED**, **COMPONENT-TESTED**, **LIVE-MEASURED**, and **QUALIFIED**. A passing unit test, synthetic restart, or plausible answer is never a trained-model forward-pass proof or a completed engineering task. Preserve failures and incomplete attempts in the failure ledger; only complete, judge-passing runs enter final comparisons. Never inspect hidden tests, answer keys, or grader internals.

## Mathematics: autonomous math olympiad method

For every derivation, numerical claim, optimization, simulation, benchmark statistic, or proof, load the `autonomous-math-olympiad` skill if available; the harness frontier currently loads `math-olympiad-unified` through `skill(...)`. If neither can be loaded, follow these five stages explicitly:

1. **Interpret and bound.** Restate the exact problem, units, assumptions, quantifiers, and plausible alternative readings. Compute an oracle, conservation law, or upper/lower bound where possible.
2. **Derive.** Define symbols once, derive before coding, identify invariant and limiting cases, and try small counterexamples before accepting a conjecture.
3. **Compute.** Use bounded scripts for arithmetic and staged checks (toy, small, medium, full). Freeze inputs, seeds, metrics, and thresholds before the full run. Change one variable per experiment.
4. **Verify.** Recompute with an independent method or brute force on small cases; check dimensions, signs, indexing, tolerances, numerical stability, and conservation. Keep script, input hash, and output receipt.
5. **Present or abstain.** Give a clean proof/result with exact evidence and remaining gaps. If verification fails, report the counterexample or say there is no confident solution.

Then conduct a **fresh-context adversarial review of the clean argument**, without feeding the reviewer the original reasoning. Attack hidden assumptions, degenerate cases, unit conversions, boundary conditions, and general lemmas. Correct and repeat up to three cycles; unresolved holes remain marked. Since this project forbids unapproved subagents, do the independent pass in a separate context or an explicitly isolated verification step, not by silently spawning agents.

## Deep research and the Qdrant Edge ingestion protocol

Search the existing Edge library with `rag_search` and open candidate passages with `rag_open` before external research. For new work, open primary papers and verify title, authors, year, URL/DOI, version, relied-on passages, and limitations. Cite only sources actually opened. **Every verified public preprint relied on in a research stage must have both its exact original bytes and full extractable text represented in the canonical Qdrant Edge shard**; if either copy or readback fails, record the gap and do not mark the source fully ingested. Do not ingest private code, competition tasks/data, credentials, unpublished results, or user content without explicit authorization.

Canonical shard: `/Users/stewartbarteau/Desktop/Ai Research and Preprint Vault/qdrant-edge-research-v1`, collection `research_library_v1`. The byte-archive protocol is in `scripts/migrate_research_library_to_edge.py`: content-addressed manifest plus ordered base64 payload chunks, MIME type, path, exact byte count, and SHA-256; reconstruct and compare bytes/hash after write. The text protocol is `frontier/n3.py::Rag.ingest` or the compatible `rag_ingest.py` callable: extract complete usable text, chunk deterministically (3,200 characters with 200 overlap), attach URL/title/year/evidence note and per-chunk SHA-256, write deterministic payload-only points under the shard's `.ple-edge.lock`, flush, reopen, and verify live BM25 retrieval and `rag_open` coverage. Keep an ingestion receipt in the active project report. No Docker or Qdrant Server is required for ordinary Edge use; only one process owns the shard at a time.

**Embedding boundary:** current new-source ingestion uses the scored `text` BM25 index; it does **not** create a new dense embedding. Preserve all legacy dense vectors and their model/dimension identity. Do not substitute hash vectors for 1024-dimensional Qwen vectors or claim semantic/dense retrieval until a compatible encoder, dimension check, vector write/readback, and retrieval benchmark pass. Exact PDF bytes are archive payloads, not an embedding. Never remove an original source until byte reconstruction, source mapping, vector-count, and live retrieval checks pass and deletion is separately authorized.

## Frontier Loop

Use `frontier/loop.py` for design-led autonomous improvement of a target folder. Its stages are **research → audit → design → preregistration → scoped work cycles**. Research checks Edge and ingests verified sources; audit is read-only; design explains the proposed mechanism; preregistration freezes allowed paths and gates. Each work cycle proposes a scoped order, works in an isolated copy, tests it, reviews the diff, merges only passing changes, reruns target gates, and records a receipt in `frontier-loop/results.json`. Two cycles with no kept improvement trigger redesign. Run `python frontier/loop.py show <target>` before continuing an existing goal; use `set`, `step`, `approve`, `revise`, `gates`, `report`, `pause`, and `resume` as appropriate. `run --auto` is only for an owner-authorized autonomous campaign. Respect a target's contract, budgets, and owner approval boundary; no hidden grader access. Keep one concrete lesson and failure reason for each unsuccessful order. Do not use the loop as a substitute for live PLE qualification.

## Attention economics, active inference, FEP, and thermodynamic claims

Treat `docs/recall-first-forward/DESIGN.md`, `PREREGISTRATION.md`, and `FEP-INTEGRATION.md` as the governing packet. The intended recall-first policy is a bounded scout computation, verified Surreal atom/trajectory selection, then the main prefill. A model cannot choose recall before **all** computation; “before prefill” means before the main prefill. Hosted APIs can receive retrieved text but cannot inject external tensor K/V into the provider forward pass. Local sparse K/V page selection requires a qualified runtime hook, full state identity, causal/position semantics, and dense-equivalence tests. TPPE key relocation alone does not prove cross-prefix reuse.

Measure total tokens (including scout), cached/uncached prompt tokens, tool and retrieval time, prefill/decode time, page reads, memory, cost, and quality on matched tasks. Separate prompt transport savings from kernel attention work and end-to-end throughput. The active-inference/FEP controller is **proposed**, not currently executable: define typed state, observations, beliefs, actions, preferences, posterior update, calibrated uncertainty, expected-free-energy or decision-cost equation, stop rule, and per-term receipt. Compare a cost-only F0 controller with F1 on frozen matched runs. “Thermodynamic minimization” is a research hypothesis; never infer joules, heat, entropy production, or Landauer savings from tokens. Physical energy requires a calibrated meter, idle subtraction, hardware context, and matched measurements. Report negative results.

## Meta Cognitive Automaton: outside verifier

Design this as an **external, deterministic verifier in the harness**, not a persona or self-reported model thought. It should observe model response IDs, tool calls/results, source provenance, memory IDs/digests, deliverable hashes, judged outcomes, budgets, and repeated trajectories. It should check: did the model read results before claiming them; did it cite admitted atoms; is it repeating unchanged calls; are hard constraints satisfied; did the deliverable reopen; did cost or uncertainty improve; should it continue, revise, request a new observation, or stop? Its decisions require typed reasons and receipts. It may reject, pause, or request rework through the harness boundary, but must not rewrite hidden tasks, fabricate tool output, or silently change preregistered gates. This full automaton is a **design target**; the existing `engineering_verify` is a narrower deterministic geometry/flow-manifest checker, and the existing finish guard enforces result-before-finish ordering.

## Four stores: distinct jobs

| System | Job | Must not become |
| --- | --- | --- |
| **Nested SurrealDB** | Authoritative multimodel mind: verified atoms, hyperbolic addresses/relations, trajectories, tensor-page identity and bytes, immutable provenance, readback receipts. | A second SQL tensor store or an unvalidated cross-model cache. |
| **Qdrant Edge** | Local verified public research archive and retrieval: exact preprint bytes plus extracted text/BM25; preserved compatible legacy vectors. | A store for task secrets, model K/V tensors, or unverified citations. |
| **SQLite/N3** | Short-term per-session conversation/tool-output archive, pointers, pinned text, FTS5 plus Poincaré-address recall. | Tensor storage or the canonical shared multimodel mind. |
| **HyperDX** | Optional telemetry/observability for traces, latency, errors, and resource measurements when actually connected and verified. | Source of truth for atoms, tensors, research, judged results, or physical energy without a calibrated meter. |

Use one canonical authority for each datum and link systems by IDs/digests. Never call a recorded metric “HyperDX telemetry” unless the PLE integration exported it there and a readback confirms it; current PLE receipt files are the evidence authority.

## Engineering and reporting

Use `docs/PLE-ENGINEERING-MODELING.md` for 2D-to-IFC/3D and time-indexed 4D workflows. Inspect the task's own standards, units, drawings, and tool catalog before choosing CAD/solver tools. Use `engineering_verify` for declared boxes, clearances, routes, and junction flow balance, then independent IFC schema, geometry, hydraulic, thermal, and schedule gates when those claims are made. An animation is a view of computed state, not a simulation proof. Final reports state what changed, why, test and live receipts, failures, uncertainty, and next gate. Keep claims narrower than evidence.
