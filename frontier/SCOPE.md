# Self-improvement harness: final scope

Status: scope fixed 2026-11-15. Code: `harness/frontier/`. How to run: `harness/README.md`.

## Purpose

Point a model at any folder with a goal and a target, and let it work as a long-horizon
optimization engineer. Example: point it at `~/Desktop/Fraud-Predict Comp`, ask for new ways to
detect fraud beyond what the folder has tried, and set a target percentage as the quality gate.

## In scope (built)

| Area | What it does |
|---|---|
| Seat | Any OpenAI-compatible model drives: local MLX server or hosted API. Driver, worker and reviewer seats can be different models. Profiles in `~/.config/frontier/profiles.json`. |
| Session | Folder and goal are set in the session (`chat.py`), or by `loop.py set`. Each stage is shown for approval, revision or direct edit. `run --auto` skips approval. |
| Fixed order | Research, audit, `DESIGN.md`, `PREREGISTRATION.md`, then brigade cycles. No stage can be skipped. |
| Research | Loads the `deep-research` skill. Web search, page fetch, preprint aggregation (arXiv, OpenAlex, Europe PMC). At least 3 cited sources; verified sources are ingested into the research library. |
| Preregistration | Fixes the scope (allowed paths) and the quality gates, including the target metric. Hash-locked once approved. Gate scripts cannot be inside the scope. |
| Brigade | The orchestrator issues up to 4 small work orders per cycle. Each subagent gets its own workspace copy, a pinned and hash-checked `AGENTS.md` contract, its own allowed paths and its own gates. It resubmits; a reviewer checks the diff; the change is merged only if the preregistered gates do not regress beyond measured noise. Every order leaves a receipt. |
| Anti-thrash | Failed orders cannot be reissued. Two cycles with nothing kept force a root-cause redesign appended to `DESIGN.md`. Three identical tool calls in a row trigger a stop-and-rethink. |
| Mathematics | Every prompt requires the `math-olympiad-unified` skill for any calculation. Design and preregistration are rejected if it was not loaded. |
| Memory (N3) | Old context moves into memory and is recalled on demand. Short-term memory defaults to local SQLite with Poincare coordinates and hyperbolic distance. The older SurrealDB `frontier/n3` backend is explicit opt-in. |
| Carved context | Once a conversation passes 8,000 estimated tokens (`FRONTIER_CARVE_TOKENS`), completed old turns move to SQLite as linked archives. The live prompt keeps one archive pointer and recent complete tool-call groups; `memory_recall` pages archived text. A simulated 80-step run sent 4.96M tokens append-only versus 0.55M with turn archiving. This is not a live task measurement; fixed instructions, pinned view, and the latest tool output bound the per-call floor. |
| Shared mind, read | What the mind holds on the task is put into every agent's context by the harness, with no tool call. Tools: `atom_search`, `atom_neighbors`, `atom_get`, and paged `trajectory_get` with content digest verification. |
| Shared mind, write | Accepted work is minted in process: an approved stage, a kept work order, a redesign, the final outcome. Models can deliberately mint atoms and ordered trajectories with `mind_mint` and `mind_mint_trajectory`. Lineage follows model source, admission, atom, address, origin, and trajectory. A supplied evidence note is not independent verification. |
| Addresses | New atoms use the map's canonical function, ported from `hypermind-nested-kv` (hashed term counts, 3D direction, radius from salience, Lorentz coordinates, two phases). |
| Nested tensor read | `Mind.tensor_exact` calls the owning PLE-local Rust store against the same Surreal authority. It checks selected-address membership and returns request-bound binary K/V from existing nested page records. It does not create a parallel store or put tensors in SQLite. |
| Nested tensor publication | `Mind.tensor_put` appends native pages through the owning store and binary bridge. Optional `NativeCacheContext` connects automatic admitted export, deterministic record discovery and complete-roster restoration to local chat. Identical retries are idempotent; conflicting identity/lineage is refused. The address must point to an active same-workspace atom. Persistent SurrealKV restart is tested with a synthetic runtime. |
| Results | One results file per target: `frontier-loop/results.json` (state, attempts, receipts, gates, lessons, budget, minted atom ids), beside `DESIGN.md` and `PREREGISTRATION.md`. |
| Safety | No git, no Docker. Shell commands run under the macOS sandbox and can write only in their working folder. Token, time and cycle budgets. Resume after a crash. Holdout gates that builders never see. |

## Out of scope (decided)

- **Tensor KV pages for hosted API models.** A hosted model cannot load transformer K/V tensors.
  What transfers to an API model is the trajectory those pages index, delivered as context, and
  the provider's own prefix cache where its service supports it. Carving changes
  the prompt prefix and may invalidate that cache. Tensor pages
  apply only to a local runtime that can inject them.
- **Raw SurrealQL from models.** Every statement is built by the harness.
- **Writing to the research library with a foreign encoder.** Dense vectors there stay untouched.
- **The PLE competition agent.** It runs with no network except the model, so the mind and the
  memory database are not attached to official runs.
- **Gate ordering.** Scoring is defined in the preregistration, not by the harness.

## Open items (not built)

1. bge-m3 embeddings for newly minted atoms. They are found by text and by hyperbolic
   neighbourhood, but not by the vector index until an embedding endpoint is available.
2. Session-end push of the pinned core memory to Qdrant for long-term storage.
3. Local runtime adapters for tensor K/V injection. `kv_contract.py` supplies a
   model-neutral exact-prefix admission gate, and `tppe_pages.py` supplies
   optional positional algebra. No model adapter is qualified yet; the gate and
   remaining proof are in `TPPE-LOCAL-FORWARD-DESIGN.md`.
   Existing nested page lookup is connected through `native-kv-bridge`.
   `kv_assembly.py` validates and assembles full explicit rosters, including
   contiguous fragments; `Mind.continue_tensor_cache` invokes the generic local
   runtime gate after assembly. The disposable database gate exercises that
   path with a test runtime. Production-schema qualification, trusted runtime
   export provenance and a real model forward remain open.
   `drivers.resolve_local_runtime` selects the runtime factory from the driver
   profile, with its model digest pinned. The generic boundary tests use
   different layer/head/dimension counts. There is no built-in Glimmer adapter.
   Heterogeneous K/V layers now assemble through the same nested tables and
   bridge with explicit per-layer native sessions. Named hybrid/recurrent
   buffers have an exact-prefix runtime admission contract, exercised by a
   numerical synthetic recurrence. Persistence of those named buffers in the
   owning nested schema remains open; unsupported state is rejected before
   any K/V lookup. No production runtime is qualified by these contract tests.
   Optional local chat dispatch is connected to the driver, agent and the same
   shared `Mind`. The runtime owns chat encoding/decoding and supplies a trusted
   native prefix receipt or an automatic native context. K/V publication and
   discovery are connected through `checkpoints.py`, using deterministic IDs
   and budgeted power-of-two anchors. Disk-backed restart coverage uses
   synthetic callbacks. Production codecs, trusted runtime export provenance,
   independent model quality and performance qualification remain open.
   Profiles without this capability retain HTTP transport.
4. A limited SurrealDB account for the harness. It currently logs in with the root credential of
   the local server, read from the running process.
5. Exact address parity on long texts. The port matches stored addresses exactly for the 198
   atoms with short text and differs by about 0.03 on average for longer ones; cause not found.
6. A full unattended run on a real project, and a scored run of the 29 PLE dev tasks.

## Removing what the harness wrote to the mind

Every row is marked. Atoms: `provenance = 'ple-harness'`. Edges: `producer = 'ple-harness'`.
Lineage: `distillation_admission.source_artifact_uri` begins `frontier-loop://`. A backup of
`hypermind/core` from before the first write is in `PhyEngLastExam/backups/`.
