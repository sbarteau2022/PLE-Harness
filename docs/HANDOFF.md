# PLE harness handoff

Claude project setup text is in `docs/CLAUDE-PROJECT-INSTRUCTIONS.md`. It
preserves the distinction between implemented PLE mechanisms and proposed
recall-first, FEP, sparse-attention, thermodynamic, and Meta Cognitive
Automaton research.

## Engineering modeling reference, 2026-11-16

`engineering_verify` is now available to every PLE model seat and frontier
worker. It checks bounded, inline manifests for geometry boxes, source tags,
connection routes, clearances, time order, and declared junction flow balance.
Its contract and limits are in `docs/ENGINEERING-VERIFIER-DESIGN.md`; it is not
an IFC validator or hydraulic solver.

`docs/PLE-ENGINEERING-MODELING.md` maps the official PLE kit protocol and
public engineering briefs to available tools, 2D-to-IFC reconstruction,
time-indexed cooling-system visualization, verification loops, and model-neutral
implementation gaps. `artifact_render` remains an explanatory graph renderer,
not CAD or a physical solver.

## Artifact authoring capability, 2026-11-16

`docs/DOCUMENTATION_STANDARDS.md` now governs PLE-authored reports and visuals.
`skills/ple-artifacts/SKILL.md` is found by the existing frontier skill loader.
The `artifact_render` frontier tool writes portable text, Markdown, PDF, DOCX,
SVG mind maps/DAGs, and interactive 3D/4D HTML inside the active workspace.
Zed's `ple-harness` task uses this same frontier pipeline. The renderer is
model-neutral and returns byte counts and SHA-256 receipts. Read
`docs/ARTIFACTS-DESIGN.md` for limits and verification boundaries.

## OPERATOR HANDOFF — 2026-11-15 23:30 CST

**Current truth:** no paid NVIDIA model call has been made. The only runnable
campaign cell is Harness2 + Mistral. Step 5, Nemotron Ultra, DeepSeek Harness,
PLE campaign normalization, local-model cells, and mind A/B are **not wired**
and must not be represented as runnable or complete.

### Resume exactly here

From `/Users/stewartbarteau/Desktop/PhyEngLastExam/harness`:

```sh
colima status || colima start --cpu 6 --memory 12 --disk 100
docker context use colima
scripts/run_ablation.sh harness2 check
```

The 2026-11-15 check was manually interrupted after about two minutes. It had
passed submission/file validation and installed the Python dependencies, then
stopped advancing at OpenCode's `bun install ... --frozen-lockfile` with
`Resolving dependencies`. This was a **stalled/incomplete gate**, not a test
failure and not a pass. Evidence is in
`ablations/work/harness2-mistral/build.log`; the earlier missing-Docker attempt
is in `builder.log`.

If the check stalls again, inspect without starting a model call:

```sh
docker ps --filter name=plebench-build
tail -80 ablations/work/harness2-mistral/build.log
docker stats --no-stream
```

Fix the frozen image install, rerun `check`, and require listener success.
Then, before any paid run, wire and test: normalized receipt emission,
trajectory capture, nested-Surreal atom mint/readback, deliverable hashes, and
judge admission. Cross-harness minting must be labeled harness-side; it is not
evidence that a model minted or used an atom in motion.

Only after those gates pass, run one confirmed no-MCP pilot:

```sh
scripts/run_ablation.sh harness2 run wind-diaphragm-shear
```

The bridge currently exposes exec/read/write only. Do not run or claim parity
for tasks requiring MCP until MCP transport is implemented and tested. The
pilot output lands under `ablations/runs/harness2/mistral/<timestamp>/`.

### Campaign contract

- Frozen design: `ablations/DESIGN.md`, `ablations/PREREGISTRATION.md`,
  `ablations/campaign.json`.
- Order: Harness2, DeepSeek Harness, PLE; within each, Mistral Large 4, Step 5,
  Nemotron Ultra 550B. Then local Qwen, Nemotron Omni, and Glimmer one at a time.
- Local PLE models also require matched `FRONTIER_MIND=off/on` pairs. Hold all
  other fields fixed. Rescue requires a verified off failure, on completion,
  causal atom/trajectory read-to-action trace, and passing judges.
- Failed/partial attempts go only to `ablations/FAILURES.md`. The final table
  admits closed full runs only. Never silently substitute a smaller suite.
- All code, generated entries, work, receipts, reports, and handoffs remain
  inside this `harness/` directory. SQLite holds session memory only; tensor
  state remains in the nested Surreal store. No Glimmer-specific assumptions.

### Implementation map

- launcher: `scripts/run_ablation.sh`
- generated Harness2 entry: `scripts/prepare_harness2_entry.sh`
- kit server: `serve_harness2.py`
- Harness2 adapter: `ablations/harness2_ple.py`
- PLE relay tool: `ablations/bridge/ple_tool.py`
- admission logic: `ablations/receipts.py`
- generated/work/run roots: `ablations/{entries,work,runs}`

Do not start all 29 tasks until the single pilot closes and validates. Do not
publish a comparative harness conclusion unless the preregistered full table
closes and the live results support it.

## 2026-11-15 cross-harness live ablation campaign

- Frozen gates: `ablations/DESIGN.md`, `ablations/PREREGISTRATION.md`, and
  `ablations/campaign.json`. All new executables and artifacts stay here.
- Harness2 is pinned at `6868817518876de8fd170c6a8354daca273b6191`
  (`ablations/vendor/harness2`, Apache-2.0). Its suite passes 70 tests and 73
  subtests using the tested PLE-local macOS `timeout` boundary; upstream is
  unmodified.
- DeepSeek Harness is pinned at
  `39163dd30f703a79079e2d2446545cff3c453738`
  (`ablations/vendor/deepseek-harness`, MIT). Frozen install passed its
  supply-chain policy, `build:artifacts` passed 8/8, and `build:check` passed 2/2.
- API order: Harness2, DeepSeek Harness, PLE for Mistral Large 4, Step 5, and
  Nemotron Ultra 550B. Local order repeats one model at a time for Qwen 3.6
  Cyber, Nemotron Omni 30B, and Glimmer 30B. Only closed, live, judge-passing
  runs enter the final table; excluded attempts stay in `ablations/FAILURES.md`.
- A separate local PLE mind A/B holds every field fixed except mind availability.
  It measures verified task rescue, judge quality, tool behavior, attempts,
  retries, no-progress loops, stop/replan behavior, recall, in-motion atoms, and
  trajectories. Mint counts alone are not benefit; rescue requires a causal
  recall-to-artifact trace and a judge-passing result.
- No paid campaign call has run. Next: common PLE adapters and telemetry
  normalizer, keyless dry runs, then Harness2/Mistral on one task before all 29.

## Layout (single source of truth, reorganized 2026-11-16)

Everything that runs lives under `harness/`. Nothing in it depends on another harness: the research harness, `reference-harness` and `submission` were removed on 2026-11-16 (in `~/.Trash/ple-outside-harnesses-20261116`), and the shared-mind schema is vendored. The Meta Hackathon checkout is not used.

| Path | What |
|---|---|
| `serve.py`, `agent.py`, `taskbridge.py`, `submission.yaml` | PLE competition harness |
| `frontier/` | Self-improvement loop, mind, memory, tensor-page contracts, and their design docs |
| `native-kv-bridge/` | Rust bridge to the nested SurrealDB tensor store |
| `runtime/globalmind-runtime`, `runtime/core_lm` | PLE-owned runtime crates (moved from `PhyEngLastExam/`) |
| `runtime/hypermind-schema` | Vendored SurrealDB schema of the shared mind; the runtime's tests read it here |
| `goals/` | Goal files for the loop (moved from `PhyEngLastExam/frontier-goals`) |
| `docs/` | Handoffs and cross-cutting design notes |
| `reports/` | Run receipts |
| `scripts/`, `tests/` | Tools and the test suite (`.venv/bin/python -m pytest tests -q`) |

Outside `harness/`: `PhyEngLastExam/ple` (Google's kit, unmodified), `ple-dev-set` (task data),
`runs/` (run outputs), `backups/`, and `_archive/` (earlier attempts, not maintained).
`~/.config/zed/harnesses/ple-harness` is a symlink to `harness/`, not a copy.

Status: handoff prepared 2026-11-15. The overall goal is incomplete and active.
Implementation belongs to `harness/` and PLE-local owning runtime dependencies;
the Meta Hackathon checkout is reference only. Do not introduce a Glimmer
default. Models differ in layer count, per-layer cache shape and state type.

## Read first

- `harness/README.md`
- `harness/frontier/TPPE-LOCAL-FORWARD-DESIGN.md`
- `harness/frontier/MIND-TRAJECTORY-DESIGN.md`
- `harness/frontier/SCOPE.md`

The nested tensors and multimodel knowledge share the existing Surreal `Mind`
authority/workspace. SurrealQL and native Surreal byte fields are permitted.
SQLite is hyperbolic short-term session memory only; never put tensors there.
No parallel tensor store or SQL/base64 bundle table. No subagents without
explicit user approval. Update the README for every implementation, reference
the design, and run adversarial and comprehensive checks.

## Verified before the current publication change

The full Python suite passed 167 tests with the Rust bridge and disposable
Surreal integration enabled; 2 bridge Rust tests and the PLE protocol smoke
passed. See `harness/reports/local-chat-dispatch-2026-11-15.md` and the earlier
`model-neutral-state-2026-11-15.md` receipt.

`drivers.resolve` carries optional trusted runtime configuration into
`agent.ModelClient`. The pipeline passes the SAME existing `Mind` into the
agent. `frontier/local_chat.py` dispatches through a runtime-owned tokenizer,
template and decoder. It checks typed prefix receipts and uses the existing
native reader before continuation. Synthetic tests cover agent/tool cycles;
they do not qualify any trained model or establish universal model support.

`kv_contract.py` admits uniform or heterogeneous K/V and named hybrid/recurrent
state in process. `kv_assembly.py` restores supported explicit rank-two K/V
rosters from nested Surreal. Named-state persistence remains open. Hosted APIs
use text/trajectory tools and prompt archiving; external tensors cannot alter
their internal attention runtime.

## Native publication

Native append-only single-page publication is implemented in the owning PLE-local
`harness/runtime/globalmind-runtime/src/kv_identity.rs` and `harness/native-kv-bridge` (`put`
mode, raw binary input), exposed by `Mind.tensor_put` and
`frontier/nested_kv.py::publish_exact`.

It validates dense f32/f16/bf16 shapes, immutable bytes/digests, finite values,
geometry existence, active same-workspace address/atom linkage, exact native
session identity and page lineage receipt. Identical retries succeed;
conflicting sessions/pages/lineage are refused without overwrite. Pages and
sessions are created through the owning typed store. No live database writes
or production schema migration were performed. A multi-page upload is NOT
atomic; the coordinator below refuses incomplete rosters and resumes identical
uploads without overwrite.

Production schema discovery: `transformer_kv_session_identity` is UNIQUE over
workspace/model/request. Separate per-layer sessions therefore need distinct
request digests. Layered assembly now permits that while retaining immutable
session fields within each layer/session. The earlier minimal disposable
fixture did not enforce this index. The latest disposable test applies copied
production 0004/0009 SCHEMAFULL definitions, READONLY fields and UNIQUE indexes.
It demonstrates the duplicate-request failure and successful distinct requests
while restoring heterogeneous f32/f16 layers. The copied files are in
`harness/native-kv-bridge/schema/`; never run these scripts on the live database
as a test setup (0004 contains legacy table removal statements).

Writer framing: `put` stdin = `PKP1` + three u32LE lengths (metadata/K/V) + JSON
metadata + raw K + raw V. Metadata is the exact reader witness plus one
`trajectory_atoms` ID and `receipt_digest`. Maximum metadata 64 KiB; K+V 16 MiB.
Success = `PWA1` + SHA-256 of the entire input. The existing reader remains
`PKV1` and its request-bound receipt. Tensors never enter model text tools.

## Latest coordinator and restart evidence

`harness/frontier/checkpoints.py` implements `NativeCacheContext`, position
contracts, deterministic native session/page IDs, budgeted strict power-of-two
prefix anchors, complete metadata discovery, and admitted export publication.
`PreparedChat(..., context=...)` opts into this automatically. Native position
version/config must match `native_position_contract`; a raw position digest is
not interchangeable. Explicit manual prefix receipts are still supported, but
cannot be combined with an automatic context on one request.

IDs bind workspace, model label, full weights/tokenizer/adapter/state identity,
selected native context, prefix hash and fragmentation version. Prefix hashing
is O(L); anchor probes number O(log L). Metadata is fetched only by exact record
IDs in batches of 128, never by scanning cache history. Payload verification
still costs O(B). Neither total inference nor geometric neighbor lookup has
been proved O(log N). No eviction or radial probability policy was added.

After generation, the largest eligible anchor is exported and admitted before
publication. A complete matching anchor is reused without republishing. Partial
uploads are misses; corrupt metadata is an error. There is no side catalog or
SQLite tensor state. A new client discovers from the database alone.

`tests/test_checkpoint_restart.py` uses actual disk-backed SurrealKV, strict
copied native schema, a process restart, new client/runtime/Mind instances,
heterogeneous f32/f16 K/V and the actual Rust bridge. It verifies cold export
then automatic warm restoration. Its model callbacks are synthetic: this is
not trained-model restart, logits, quality or speed evidence.

## Immediate resume point

Extend the SAME nested native schema/owning runtime for named recurrent and
packed state (currently rejected by the persistence path), and qualify real
runtime plugins across independent architectures. Keep the generic dispatcher;
do not reintroduce Glimmer-specific setup as proof of agnosticism. Inspect
available runtimes/checkpoints before selecting an opt-in plugin. Preserve
unrelated servers and port 8123. Bind export provenance and compare real
uninterrupted/imported/restarted logits and token IDs before claiming reuse.

## Remaining full-goal work

### Next implementation packet: recall-first forward policy

Resume with `docs/recall-first-forward/HANDOFF.md`. Its governing packet is:

- `docs/recall-first-forward/DESIGN.md`
- `docs/recall-first-forward/PREREGISTRATION.md`
- `docs/recall-first-forward/FEP-INTEGRATION.md`

Implement this work in the PLE harness. The attention-economics paper measures
the resulting policy and runtime behavior; it is not a separate harness.
Recall-first/FEP arms, clean-prefix branching, and sparse-page attention remain
unimplemented and must retain the preregistered evidence boundaries.

1. Production qualification of automatic export/publication/discovery. The
   coordinator and synthetic persistent restart work; trusted actual runtime
   export and independent trained-model behavior remain unqualified.
2. Extend the SAME owning nested schema for named recurrent/packed state.
3. Qualify independent real-model plugins, including restart/logit/token parity,
   changed identity, missing/corrupt state, matched prefill latency and memory.
   No production plugin is qualified. Historical Glimmer experiments do not
   establish model independence and the Glimmer adapter was removed.
4. Establish indexed hyperbolic retrieval bounds, probability/radial reuse
   policy without eviction, and measured attention economics. FEP/energy/heat
   claims are unproven. Do not infer physical savings from token counts.
5. TPPE remains optional position algebra. Re-addressing keys does not remove
   dependence of deeper K/V on the causal prefix. Cross-prefix tensor reuse
   and trained-model TPPE quality remain unqualified.

## Validation commands

Latest final validation: **168 Python tests passed**, including the strict
disposable Surreal publication/heterogeneous restore checks; **3 Rust bridge
tests passed**; the PLE protocol smoke passed. No tests are currently running.
The bridge binary at the path below was rebuilt against the changed owning
runtime. Detailed receipt: `harness/reports/native-publication-2026-11-15.md`.

From `harness/`:

```sh
cargo build --offline --manifest-path native-kv-bridge/Cargo.toml --target-dir /Users/stewartbarteau/.cargo-target-ple-nested-kv
PLE_NESTED_KV_BRIDGE=/Users/stewartbarteau/.cargo-target-ple-nested-kv/debug/ple-nested-kv-bridge ./.venv/bin/python -m pytest tests -q
./.venv/bin/python tests/protocol_smoke.py
cargo test --offline --manifest-path native-kv-bridge/Cargo.toml --target-dir /Users/stewartbarteau/.cargo-target-ple-nested-kv
```

Tests must use their own disposable Surreal instance, never mutate the live
database as a fixture. Preserve unrelated serving processes and port 8123.
No owned Glimmer server remains running from the rejected experiment.


## Session addendum (2026-11-15, self-improvement loop and TPPE)

Read in this order: `frontier/SCOPE.md` (what the loop is and is not), this file, then
`frontier/TPPE-LOCAL-FORWARD-DESIGN.md`, `frontier/MIND-TRAJECTORY-DESIGN.md`,
`frontier/HYPERBOLIC-BOUNDS.md`, and `docs/TPPE-FORWARD-HANDOFF-2026-11-15.md`.

Several agents edited `harness/frontier/` in parallel today. The test suite is the arbiter:
`.venv/bin/python -m pytest tests -q` gave 203 passed, 2 skipped at the end of the session.

## State that is verified

| Item | Evidence |
|---|---|
| PLE competition harness (`serve.py`, `agent.py`) | Live `kit check` passes on Mistral Large 4 via NVIDIA. One official graded run: wind-diaphragm-shear scored 0.0 because no deliverable was written; draft and wrap-up reminders were added afterwards and have not been rerun on a real task. |
| Any model in the seat reads the mind | `scripts/mind_read_check.py step5 "<question>"`: step5 searched the atoms and answered correctly with citations. step5 native tool calling works (4 of 4 probes); its earlier failure was one bad reply. |
| Minting in process | Mistral smoke on a toy folder: research and audit each minted a stage atom plus 6 to 7 extracted atoms linked `part_of`, and a trajectory over 18 atoms was minted after the preregistration. The mind holds 2,097 atoms, 18 with provenance `ple-harness`, 1 trajectory, 30 edges. |
| Scope check | The first smoke blocked because Mistral copied the prompt's example scope (`src/features.py`). Two fixes now exist: a goal-file rule (another agent) and "at least one allowed path must match a real file" (`scope_matches_folder`). |
| Carved context | Simulated 80 steps: 4.96M tokens append-only, 0.86M carved. Not measured live. |
| TPPE findings | `linear_log_v1` fails closed at 624 positions; the two-angle arm aliases far positions; relocation of K pages is exact for position. Rust tests in `globalmind-runtime` pass. |

## In flight at hand-off

The Mistral smoke was resumed from the design stage after the scope fix:

- folder: `/private/tmp/claude-501/-Users-stewartbarteau-Desktop-PhyEngLastExam/1cbfd258-325a-4237-a43e-959905a1f44f/scratchpad/smoke/median-fix`
- log: same folder's parent, `run2.log`; state before resuming: `results-before-resume.json`
- success means: brigade cycle keeps an order, `check.py` prints `score=1.00`, status `complete`,
  and a second trajectory is minted. Check with `.venv/bin/python frontier/loop.py report <folder>`.

## Next steps, in order

1. Read the smoke's outcome. If it blocks again, the report's `blocked_reason` says where.
2. Run the loop on a real project (the fraud folder) with `frontier/chat.py`, approving each stage.
3. Rerun one PLE dev task with the draft and wrap-up reminders, to confirm a deliverable is written.
4. TPPE: follow `docs/TPPE-FORWARD-HANDOFF-2026-11-15.md` (new contract identity for `multiscale_spiral_v3`, radial
   law decision for >624 positions, bakeoff once the Qwen3.6 checkpoint is available, Glimmer
   page-relocation error measurement).
5. Open items from `frontier/SCOPE.md`: bge-m3 embeddings for new atoms, session-end core push
   to Qdrant, limited SurrealDB account, long-text address parity.

## Removing what this session wrote to the mind

Atoms `provenance = 'ple-harness'`, edges `producer = 'ple-harness'`, lineage whose
`source_artifact_uri` begins `frontier-loop://`. Backup from before the first write:
`PhyEngLastExam/backups/hypermind-core-20261115-105040.surql`.

## Addendum 2026-11-16: models and token measurement

- Every model brings its own tokenizer: `runtime/globalmind-runtime/src/model_select.rs` picks the
  active manifest (`GLOBALMIND_MODEL_MANIFEST`, else the first manifest whose files are all on disk)
  and uses the tokenizer that manifest lists. `scripts/make_model_manifest.py <dir> <name>` writes a
  manifest for any local model. Manifests now exist for Qwen3.6Cyber (`~/Desktop/Qwen3.6Cyber`),
  Nemotron-3-Nano-Omni-30B (`~/models/Nemotron-3-Nano-Omni-30B-A3B-MXFP4`) and Glimmer; the tokenizer,
  ablation, smoke and selection tests pass under each (15 of 15 per model).
- API and local seats: the PLE serve agent reads `MODEL_*` from the kit, so the competition's
  `google-cloud/<id>` models need no code change. The frontier loop takes any OpenAI-compatible seat
  by profile (`~/.config/frontier/profiles.json`) or `<base_url>|<model>`.
- Carved context, replayed on the real 30-step PLE run (wind-diaphragm-shear, 40 calls, 1,906,548
  prompt tokens measured by the kit): about 323,677 tokens carved, 5.9x fewer; the largest single call
  drops from about 75k to 10k tokens. The earlier synthetic 80-step figure was 4.96M to 0.86M (5.8x).
  The replay reuses the run's own tool calls; a live A/B run has not been done.

## Addendum 2026-11-16: live API and local runs, carving policy, Zed

Benchmark: `scripts/live_bench.py <trajectory> <seat>` replays the real wind-diaphragm-shear PLE run
(recorded tool calls and outputs, with re-issued earlier calls) against a live model, two arms on the
same conversation: regular (append-only) and PLE harness (carved). Each arm starts with a cold provider
cache (a random run id leads the prompt). Raw data in `reports/live-bench-2026-11-16-*.json`.

API, Nemotron Ultra 550B on NVIDIA, 15 steps, cold cache (`live-bench-2026-11-16-cold-percall.json`):

| | Regular | PLE harness (carve every call above 8k) |
|---|---|---|
| Prompt tokens | 435,544 | 121,838 (3.6x fewer) |
| Provider-cached | 280,576 (64%) | 8,768 (7%) |
| Uncached prompt tokens | 154,968 | 113,070 (1.4x fewer) |
| TTFT median | 1.72 s | 1.40 s |
| Largest prompt | 66,555 | about 21k |

Hysteresis (carve only above 24k) on the same provider sent 186,992 uncached against 185,701, no gain,
so API seats carve every call (default `FRONTIER_CARVE_HIGH` = floor). One carved call errored and is
recorded. Mistral Large 4 was queueing at 170 to 290 s first-token on NVIDIA, so it was not used.

Local, Qwen3.6Cyber via MLX (`mlx_lm.server`, port 8001; MLX reuses the exact prompt KV prefix):

| | Regular | PLE harness |
|---|---|---|
| 9 steps, carve every call above 8k: prompt / uncached | 120,791 / 32,849 | 64,124 / 42,472 |
| 15 steps, carve above 48k (profile `carve_high`): prompt / uncached | 423,374 / 64,869 | same (threshold not reached) |
| Local KV prefix reuse (cached share) | 85% | 85% |
| TTFT median / decode | 1.37 s / 68 tok/s | 1.30 s / 67 tok/s |

So on a local model with KV prefix reuse, carving early costs prefill; the `qwen` profile now carves only
above 48k estimated tokens and stays append-only (full KV reuse) below it. Note: the harness's token
estimate (characters / 4) undercounts Qwen's tokens by about 1.8x; calibrating it per tokenizer is open.
Start the local server with the command in the `qwen` profile's `note`.

Zed: profile id `ple-harness`, name "PLE Harness", is Zed's default agent profile. Harness code is reached
only through `~/.config/zed/harnesses/ple-harness` (symlink to this folder), so every change here is live
in Zed with nothing to copy. `~/.config/zed/tasks.json` has launch tasks that run through the symlink
(frontier chat on Mistral or local Qwen, report, tests, kit check). Backup: `settings.json.bak-20261116-ple-profile`.

### Rerun 2026-11-16: Nemotron Ultra 550B, three arms in one run, fixed prefix primed

All arms started with a cold cache except for the fixed prefix (run id, system prompt, task), which was
sent once before step 1 so every arm began with the same warm fixed prefix
(`reports/live-bench-2026-11-16-nemotron-3arm.json`, `--prime --stable 24000`).

| Arm | Prompt tokens | Provider-cached | Uncached | TTFT median | Decode median | E2E total |
|---|---|---|---|---|---|---|
| Regular (append-only) | 435,634 | 249,888 (57%) | 185,746 | 1.99 s | 82 tok/s | 70.2 s |
| PLE, carve every call above 8k | 129,781 | 8,768 (7%) | 121,013 (1.53x fewer) | 1.22 s | 76 tok/s | 61.5 s |
| PLE, prefix-stable (carve above 24k) | 283,395 | 118,368 (42%) | 165,027 (1.13x fewer) | 1.38 s | 111 tok/s | 50.3 s |

Carving every call sends the fewest uncached tokens and stays the API default. The prefix-stable arm
keeps more of the provider cache and had the shortest end-to-end time in this run, but sends more
uncached tokens. Single run per arm; timing differences are within network noise.
