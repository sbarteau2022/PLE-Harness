# PLE harness

Start here: `docs/HANDOFF.md` (layout, state, next steps). Everything that runs is under this folder.

The preregistered live cross-harness campaign is defined in
[`ablations/DESIGN.md`](ablations/DESIGN.md) and
[`ablations/PREREGISTRATION.md`](ablations/PREREGISTRATION.md). Only complete,
judge-passing live runs are admitted to its final table; failed and partial
attempts remain in a separate ledger.
The pinned upstream sources remain unmodified. macOS campaign processes prepend
`ablations/bin` for the tested GNU `timeout` compatibility boundary required by
Harness2's JobBench runner.
Every candidate result is normalized and fail-closed by
`ablations/receipts.py`; live response IDs, tool execution, complete judges,
deliverable hashes, telemetry, and validated atom/trajectory receipts are
required before a row can enter the final table.
External harness subprocesses reach the live evaluator through the authenticated
loopback relay and `ablations/bridge/ple_tool.py`; they do not receive copied
task workspaces or simulated shell results.
`ablations/harness2_ple.py` implements Harness2's native `BenchmarkAdapter`
against that relay while retaining its Propose-Probe-Compose core and editable
eight-component OpenCode surface.
`serve_harness2.py` is the PLE protocol entry point for that adapter;
`submission.harness2.yaml` keeps the ablation separate from the production PLE
submission.
Use `scripts/run_ablation.sh harness2 check` for the keyless container/listener
gate and `scripts/run_ablation.sh harness2 run <task>` for a one-task live cell.
The launcher retrieves the existing NVIDIA credential without printing it and
keeps all work and run artifacts under `harness/ablations/`.
`scripts/prepare_harness2_entry.sh` generates a minimal kit entry from pinned
source, excluding Git history, dependency caches, downloaded benchmark data,
and the unrelated DeepSeek tree. The production `submission.yaml` is never
replaced.
Campaign seats are `mistral`, `step5`, and `nemotron-ultra` (`MODEL=<seat>`);
Harness2 and PLE entries exist per seat, while the DeepSeek adapter is still
an UNMET scaffold (`ablations/deepseek-scaffold.md`).
Both serve entries print a `PLE-ABLATION-RUN {...}` telemetry marker;
`ablations/normalize.py` parses markers and OpenCode trajectories, hashes
deliverables, and builds validating receipts, `ablations/mint.py` mints and
readback-verifies the harness-side atoms/trajectory, and
`scripts/write_results.py` atomically assembles `ablations/api/results.json`
with complete/incomplete/failed/excluded cells plus evidence paths.
`ple_tool.py` covers `exec`, `read`, `write`, and task-MCP forwarding.
`run-local` serves the harness from this Mac (same packet/tools/budgets/judges)
so only the small task/verifier environments use colima; use it for campaign
cells, `run` only to revalidate the container path.

An autonomous agent harness for Physical-Engineering's Last Exam (PLE). It is a
WebSocket server that speaks the kit's serve protocol and runs a tool-calling loop
against any OpenAI-compatible model.

| File | Purpose |
|---|---|
| `serve.py` | Entry point. `taskbridge.run(solve)` listens on `$PORT`, one session per task |
| `agent.py` | Agent loop: system prompt, tools, retries, context compaction, time budget |
| `taskbridge.py` | Protocol plumbing and MCP client, vendored unchanged from the kit (Apache 2.0) |
| `submission.yaml` | Install/serve commands and model; `env` holds non-secret settings |
| `scripts/run_local.sh` | Local validation on NVIDIA's API (`check` or `dev`) |
| `frontier/` | Self-improvement loop: research, audit, design, preregistration, brigade of subagents |
| `tests/` | Offline unit tests, a wire-level smoke test, and a mock model for `kit check` |

## Tools the model gets

- `bash`, `read_file`, `write_file`, `finish` on every task.
- `rag_search`, `rag_open` when the local Edge shard is available; these read
  archived research from the host without a Qdrant Server endpoint.
- `mcp_list`, `mcp_schema`, `mcp_call` when the task serves MCP apps (17 of 29 dev
  tasks do: excel, word, pdfs, freecad, mail, code execution and so on). These are
  meta-tools, so a catalog of hundreds of MCP tools never floods the context.

## Settings (environment)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_NAME`, `MODEL_BASE_URL`, `MODEL_API_KEY` | set by the evaluator | Model endpoint |
| `MAX_OUTPUT_TOKENS` | 16000 | Reply cap per call |
| `COMPACT_AT_TOKENS` | 90000 | Old tool outputs are elided above this estimate |
| `MAX_STEPS` | 400 | Model calls per task |
| `RESERVE_SECONDS` | 240 | When time left drops below this, the agent is told to finish |
| `FRONTIER_EDGE_SHARD` | local research vault Edge shard | Canonical in-process research library |
| `FRONTIER_RAG_LIBRARY` | `research_library_v1` | Name shown in RAG citations |
| `FRONTIER_RAG_TEXT_FIELD` | `text` | Edge BM25 payload-text index |

## Model

Local validation default: `openai/gpt-oss-120b` on `https://integrate.api.nvidia.com/v1`.
Change it with `MODEL=<nvidia model id> scripts/run_local.sh ...`, which also rewrites
`model_api` in `submission.yaml`. Competition entries must use a `google-cloud/<id>`
model (`ple/docs/models.md`); switch `model_api` to e.g. `google-cloud/gpt-oss-120b`
and remove `model_base_url` before submitting.

## Run it

```bash
cd /Users/stewartbarteau/Desktop/PhyEngLastExam/harness
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt pytest
```

```bash
.venv/bin/python -m pytest tests -q && .venv/bin/python tests/protocol_smoke.py
```

Put `NVIDIA_API_KEY=...` in `harness/.env` (gitignored) or export it, then:

```bash
scripts/run_local.sh check
```

```bash
scripts/run_local.sh dev roof-deck-design,wind-diaphragm-shear
```

```bash
scripts/run_local.sh dev
```

Results land in `../runs/ple-harness-<time>/` (`results.json` holds per-task rewards).
Rubric and checks-judge tasks are graded by `JUDGE_MODEL` (default: the same NVIDIA
model); the leaderboard uses Claude Opus 5.5, so local rubric scores are indicative.

Without a key, `kit check` can still be exercised end to end with the mock model:

```bash
.venv/bin/python tests/mock_openai.py 8099 &
cd ../ple && uv run python -m kit check ../harness --env-file <file with MODEL_API_KEY=test> --model-base-url http://localhost:8099/v1
```

## Self-improvement loop (frontier)

Final scope, what is deliberately out, and open items: `frontier/SCOPE.md`. SurrealDB is native to the
loop (`frontier/surreal.py`): the shared mind is recalled into every agent's context, and accepted work
is distilled into atoms and trajectories as it happens. Short-term memory defaults to SQLite;
`FRONTIER_MIND=off` disables the shared mind, while `FRONTIER_N3=surreal` explicitly selects the
older Surreal short-term backend. The trajectory tool contract is in
`frontier/MIND-TRAJECTORY-DESIGN.md`.

The shared text/trajectory path works for every driver seat. Local tensor
continuation is an optional, model-neutral admission contract in
`frontier/kv_contract.py`; no runtime adapter is admitted yet. TPPE relocation
is a configurable reference in `frontier/tppe_pages.py`, and the exact-prefix
gate is described in `frontier/TPPE-LOCAL-FORWARD-DESIGN.md`. A different causal
prefix invalidates deeper-layer K/V even if a key's position is relocated.

Tensor state belongs to the existing nested SurrealDB substrate. SQLite holds
short-term memory only. `Mind.tensor_exact` now calls the owning Rust runtime
through `native-kv-bridge`, using this mind's existing Surreal endpoint,
namespace, database and workspace. It requires the tensor address to be in the
selected recall. The runtime checks the complete stored page identity and hashes;
raw K/V returns in a bounded binary frame bound to the request. Tensors never
enter a model's text tools. No parallel tensor table or separate store is created.
The model-neutral PLE admission contract also checks complete rosters and finite
values; qualification with a real local forward adapter remains open.
Build and verification instructions: [native-kv-bridge/README.md](native-kv-bridge/README.md).
`Mind.tensor_put` now publishes a single immutable native page through that
same authority and bridge, using raw binary input. The owning PLE-local Rust
store validates dense shape/digests/finite values, selected address/atom linkage,
session identity and lineage receipt. Identical publication is idempotent;
conflicting records are refused without overwrite. The strict disposable test
uses copied production SCHEMAFULL definitions and UNIQUE indexes. Separate
per-layer sessions require distinct request digests under that schema.
Automatic admitted K/V checkpoint export, publication and discovery are
connected through the coordinator described below. Named-state persistence
and production runtime qualification remain open.
Verification: [native publication receipt](reports/native-publication-2026-11-15.md).
`frontier/kv_assembly.py` validates a complete explicit native roster before
reading tensors, assembles contiguous fragments into every layer/head, and
calls the model-neutral runtime gate only after all sizes, hashes and finite
values pass. It rejects inconsistent sessions, missing heads, changed prefixes and
addresses outside the selected recall. Runtime tokenizer/adapter provenance
must come from a trusted export receipt; the native page schema alone does
not attest it. A production local adapter and live forward qualification are
still required.
The same `Mind` exposes `tensor_cache(...)` for complete assembly and
`continue_tensor_cache(...)` for admission and continuation. Native position
version/configuration must match the runtime's `native_position_contract`
digest. These methods require an explicit trusted page roster and opt-in local
runtime; the hosted driver loop continues to use knowledge and memory tools.
Continuation arguments are checked before fetching pages. Runtime output IDs
and generation limits are checked before results are returned to the caller.

`drivers.resolve_local_runtime(seat)` selects an optional runtime plugin from
that driver's `local_runtime` profile entry: `factory` is a Python
`module:callable`, `model_digest` pins the model artifact, and `options` supplies
runtime-specific settings. The factory receives the selected driver model.
There is no built-in model or serving-wire fallback. Layer count, KV head count,
head width, dtype, layout and position contract come from the runtime identity.
The wrapper pins that full identity across export and continuation. Profiles
without this capability keep the shared text/trajectory/SQLite path.
The removed Glimmer-wire experiment is not evidence of support for other
models. No production runtime plugin is qualified yet.

`TensorSpec` is a uniform K/V shorthand. `LayeredTensorSpec` supplies a separate
head count, width, dtype, layout and position contract for each attention layer.
The same nested Surreal tables and Rust bridge assemble heterogeneous layers
from explicit per-layer sessions; no tensor schema or second store is created.
`StateSpec` and `ExactStateSet` also admit named buffers for hybrid/recurrent
state, window-limited attention, counters and opaque packed encodings. The
runtime declares the complete schema; the gate checks it, the causal prefix,
payload sizes and digests before forwarding. Aggregate component/byte budgets
replace a fixed model's layer-count bound. Assembly inspects each fragment once
and reads/verifies its payload bytes; it does not rescan all layers per fragment.

Named state currently supports local export/import admission only: the existing
nested schema can persist rank-two K/V, so recurrent-state persistence fails
before lookup. Extending that owning schema and qualifying independent real
runtimes are still required. Synthetic hybrid continuation and disposable
Surreal heterogeneous-layer tests establish contract behavior, not universal
model or TPPE quality. A plugin must check the active model atomically inside
its forward operation; harness identity checks cannot prevent a server race.
Verification: [model-neutral state receipt](reports/model-neutral-state-2026-11-15.md).

The driver profile now carries its runtime configuration into `agent.ModelClient`.
The pipeline passes its existing shared `Mind`; local requests run through
`frontier/local_chat.py` outside the async event loop. A chat-capable plugin
implements `prepare_chat(messages, tools, mind)`, `generate_tokens(prompt_ids,
max_tokens)`, and `decode_chat(output_ids)`, alongside the exact-state runtime
methods. It owns its tokenizer, chat template and output decoding.

Preparation returns `PreparedChat` with token IDs and optionally a trusted
`NestedPrefix` receipt. PLE checks that receipt's identity and strict causal
prefix, then reads and admits the full native roster through that same `Mind` before
continuing with the remaining token suffix. Without a candidate it generates
locally from the full prompt. Invalid candidates fail locally. Profiles without
`local_runtime` use the existing HTTP client. Native tools and the shared text
tool protocol both work through this dispatch.

For example, a trusted local profile can supply:

```json
{
  "model": "your-selected-model",
  "context": 32768,
  "max_output": 1024,
  "local_runtime": {
    "factory": "your_runtime:create",
    "model_digest": "<64 lowercase SHA-256 hex characters>",
    "bridge": "/absolute/path/to/ple-nested-kv-bridge",
    "options": {}
  }
}
```

This is a plugin contract, not a bundled production adapter. Local generation
currently has a 4,096-token admission limit. The plugin must supply trusted
native page witnesses, or opt into automatic coordination with a typed context.
Completion usage reports the admitted cached-prefix token count; frontier saves
it as `budget.tokens_cached` while total prompt counts remain in `tokens_in`.
These counters do not establish latency, energy or monetary savings.
`PreparedChat(..., context=NativeCacheContext(...))` opts into automatic K/V
checkpoint coordination in `frontier/checkpoints.py`. The context supplies an
existing selected atom address and native geometry/position contracts. PLE
checks eligible power-of-two strict prefixes through exact record metadata
reads, restores the longest complete match, and automatically publishes an
admitted `export_prefix` at the largest eligible anchor after a successful
completion. No hand-built page roster or in-process checkpoint catalog is
needed after a client/database restart. Partial uploads remain misses until
the full expected roster exists; identical retries resume without overwrite.

The record identity binds workspace, model label, weights, tokenizer, adapter,
state schema, native context, prefix digest and fragmentation policy. Native
metadata reads are limited to 128 exact record IDs per request. Anchor hashing
is O(L), candidate count is O(log L), and metadata/payload work scales with the
selected fragments and bytes. No global history scan is used. This does not
prove O(log N) total inference or indexed hyperbolic neighborhood bounds.
The restart gate uses real disk-backed SurrealKV and a synthetic runtime;
trained-model restart/logit quality and performance remain unqualified.
Dispatch verification: [local chat receipt](reports/local-chat-dispatch-2026-11-15.md).
Session continuation: [project handoff](../HANDOFF.md).

Point a model at any folder with a goal and let it work as a long-horizon optimization
engineer. It ports `~/.claude/frontier-loop/frontier_goal.py` (goal, state, cap, complete,
block, pause, resume) into a standalone system. No git, no Docker, no Colima. Any
OpenAI-compatible model can drive: a local MLX server or a hosted API.

| File | Role |
|---|---|
| `frontier/chat.py` | Interactive session: set the folder and goal in chat, review every step |
| `frontier/loop.py` | CLI: `set`, `step`, `approve`, `revise`, `run`, `show`, `report`, `gates`, `pause`, `resume` |
| `frontier/pipeline.py` | Stages with approval, brigade orchestrator, subagents, reviewer, redesign |
| `frontier/n3.py` | N3 memory: SQLite store addressed in a 3D hyperbolic ball, Qdrant RAG |
| `frontier/research.py` | Web search, page fetch, preprint aggregation |
| `frontier/skills.py` | Loads the global skills (`~/.agents/skills`) |
| `frontier/drivers.py` | Model profiles and context-window sizing |
| `frontier/core.py` | Results file, gates, noise, snapshots, scope, sandboxed shell, budget |

Everything is saved in the folder it is pointed at, under `frontier-loop/`: `results.json`
(the single results file: state, stage attempts, receipts, gates, lessons, budget),
`DESIGN.md`, `PREREGISTRATION.md`, and `n3.sqlite` (the memory store).

### Flow

1. **Research** (uses the `deep-research` skill): web and preprint search (arXiv, OpenAlex
   covering engrXiv/SSRN/bioRxiv/medRxiv/ChemRxiv/TechRxiv, Europe PMC). Notes must cite at
   least 3 sources, and verified sources must be ingested into Qdrant with `rag_ingest`.
2. **Audit**: read-only review of what the folder has tried and how it scored.
3. **Design**: writes `frontier-loop/DESIGN.md`.
4. **Preregistration**: writes `frontier-loop/PREREGISTRATION.md` with the scope
   (`allowed_paths`) and the quality gates, including the goal's target as a metric gate.
   It is hash-locked once approved. Gate scripts may not sit inside the scope.
   When the goal names an existing implementation file, approval rejects a scope
   that excludes it. The prompt uses no sample paths that could be copied into
   an unrelated task. A scope already approved before this check remains locked;
   start a new run for that target rather than editing its preregistration.
5. **Brigade cycles**: the orchestrator issues up to 4 small work orders, each with its own
   allowed paths and gates. Each subagent works in its own workspace copy under a pinned,
   hash-checked `AGENTS.md` contract, then resubmits. Its gates run, a reviewer checks the
   diff, the change is merged, and the preregistered gates decide: a regression beyond
   measured noise is rolled back. Every order leaves a receipt.

Each stage waits for your approval (`approve`, or `revise --feedback`); `run --auto` skips
that. Design and preregistration are rejected unless the agent loaded the
`math-olympiad-unified` skill; every prompt carries the rule that all math goes through it.

### What keeps it from thrashing

- A failed order's signature is remembered; the orchestrator cannot issue it again.
- Two cycles with nothing kept force a **redesign**: root cause, new approach, an amendment
  appended to `DESIGN.md`. If the scope or gates make the goal unreachable it blocks and asks.
- Inside a run, three identical tool calls with the same result trigger a stop-and-rethink.
- Lessons (`record_lesson`, plus one per failed order) are shown to every later agent.

### Memory instead of compaction (N3)

Completed old turns, including their native tool calls and results, move into
`n3.sqlite` and leave one archive pointer in the prompt. Nothing is dropped;
`memory_recall(id, start, chars)` reads an archive in bounded pages. A view of
the 20 newest pinned entries stays in context; older pinned entries remain
searchable in SQLite. Entries are addressed in the Poincare 3-ball (direction =
topic, radius = depth, children placed outward from their parent) and recall blends lexical
match with hyperbolic distance. Depths through seven keep the original radius
law; deeper notes grow logarithmically in hyperbolic radius instead of being
clipped to one Poincare radius. A missing parent is rejected. `rag_search`
queries the ranked BM25 text index in the
local Qdrant Edge shard, `rag_open` pages through a source's indexed passages, and
`rag_ingest` adds verified text to the same shard. The harness opens and closes the
shard under a file lock for each operation, allowing different chat models to share
one local research store without a separate embedding endpoint. Existing `text_rank0`
and `text_rank1` vectors remain stored for an admitted dense query encoder; this
harness does not reinterpret them. Source paths in `rag_search` are archive keys;
use `rag_open` to read further passages because the original files were removed.
The shared mind also exposes `atom_search`, `atom_neighbors`, `atom_get`,
`trajectory_get`, `mind_mint`, and `mind_mint_trajectory`. A named trajectory is
read in bounded pages with atom text digest checks; a model can record a walk
over existing atoms with an evidence note. The note is not an independent truth
check. Pushing pinned SQLite memory to Qdrant at session end remains open.

### Local Qdrant Edge research library

The copy protocol and gates are in [docs/RESEARCH_LIBRARY_EDGE_DESIGN.md](docs/RESEARCH_LIBRARY_EDGE_DESIGN.md).
`qdrant-edge-py==0.9.0` is installed in the harness environment. Qdrant Edge is
beta and keeps one shard on local disk. The default shard path is
`/Users/stewartbarteau/Desktop/Ai Research and Preprint Vault/qdrant-edge-research-v1`.
It holds the original Qdrant points and vectors, a scored text index, and exact
source bytes as verified manifest and chunk records. The 30 preprints that had
no old text points gained 920 searchable text chunks.

Before retiring an old server collection, use the migration script's `sync` and
`verify` phases. Condensation uses `scripts/condense_research_library.py` to
verify exact originals, stage them away from their original paths, smoke Edge
retrieval without those paths, and release the staged bytes. To restore one
archived original to a new path:

```bash
.venv/bin/python scripts/migrate_research_library_to_edge.py --phase restore \
  --restore-source '/absolute/original/source/path.pdf' --output '/new/path.pdf'
```

Cutover receipt (2026-11-15): the Edge shard has 61,520 points and 96,288
indexed vectors. Its 1,929 file manifests independently reconstruct
2,037,891,309 bytes. The 1,929 verified original files were removed after an
Edge-only runtime smoke. The old `research_library_v1` and
`frontier_research_hash384_v1` server collections were deleted; three legacy
verified sources from the latter were first copied into Edge as text payloads.
The server's unrelated session-memory collections were left untouched. With
Colima stopped and the original PDF path absent, Edge search, `rag_open`, and
PDF restoration still worked. The full offline suite passed (35 tests) and the
protocol smoke passed with `rag_search` and `rag_open` offered by the PLE task
agent. At the first verification, 503 old local source paths
and three web URLs did not point to local files; their indexed passages remain
in Edge, but unavailable original bytes cannot be certified.

This is a storage and runtime cutover. BM25 is the model-independent retrieval
baseline; dense, visual, and reranking lanes still need a PLE-specific relevance
benchmark before promotion.

### Gates

JSON entries: `name`, `cmd`, optional `metric` (regex capturing one number) with `min`
(the target), `repeats` (measure noise; a noisy gate must pass on average), `holdout: true`
(reported, never shown to builders, never decides), `evidence` (glob of log files whose
tail is shown when the gate fails), `timeout_sec`. You may also put a `gates` block in the
goal text; those always apply.

For the PLE harness itself, `scripts/score.sh` is the score gate (official graders, needs
Docker): it prints `score=`, writes the agent's step log for the worst task to
`frontier-loop/evidence/score.txt` for failure analysis, and `scripts/score.sh --holdout`
scores a separate held-out task set (`FRONTIER_TASKS`, `FRONTIER_HOLDOUT_TASKS`).
See `goals/harness-quality/GOAL.md`.

### Models

Profiles live in `~/.config/frontier/profiles.json` (`qwen`, `nemo`, `gemma`, `mistral`,
`step5` are set up). A seat is a profile name or `"<base_url>|<model>"`. The loop sizes
context and output limits from the profile or the server. Servers that reject native tool
calls are handled automatically with a text tool-call protocol (`"tool_mode": "text"` forces it).

```bash
.venv/bin/python frontier/chat.py --driver qwen --worker nemo --reviewer mistral
```

```bash
.venv/bin/python frontier/loop.py set "$HOME/Desktop/Fraud-Predict Comp" --goal "Find new ways to detect fraud beyond what the folder has tried. Target: validation AUC of at least 0.95." --max-hours 12
```

```bash
.venv/bin/python frontier/loop.py step "$HOME/Desktop/Fraud-Predict Comp" --driver qwen
```

```bash
.venv/bin/python frontier/loop.py approve "$HOME/Desktop/Fraud-Predict Comp"
```

`report` prints the summary (gates baseline to latest, work orders, redesigns, lessons).
A crash or restart resumes: stages already approved are kept, an unfinished cycle continues
with its remaining orders, and each agent resumes from its last saved step.

### Safety

Agent shell commands run under the macOS sandbox (`sandbox-exec`): they can write only
inside their working folder and temp folders (`FRONTIER_SANDBOX=off` disables this). File
tools cannot leave the working folder. Task `tests/` and `artifacts/` are refused, and
`web_fetch` refuses local addresses. Budgets: `--max-tokens`, `--max-hours`, `--max-rounds`.

## Mac Docker setup (Colima, Apple Silicon)

- The VM needs room for one task at a time: 4 CPUs and 8 GB per task, plus the serve
  and verifier containers. Use `colima start --cpu 8 --memory 24` (default 2/2 fails with
  "range of CPUs is from 0.01 to 2.00").
- The egress sidecar needs netlink under linux/amd64, which qemu emulation lacks.
  Install Rosetta (`softwareupdate --install-rosetta --agree-to-license`) and run
  Colima with `--vm-type vz --vz-rosetta`.
- `~/.docker/cli-plugins/*` must point at `/Applications/Docker.app/...`; stale links
  make `docker compose` disappear ("unknown flag: --project-name").
- Harbor builds its egress sidecar for the daemon's arch (arm64) while the kit runs
  linux/amd64. If a run fails with "does not provide the specified platform
  (linux/amd64)", rebuild the sidecar under the same tag:
  `docker buildx build --platform linux/amd64 --load -t <tag from the error> ~/.local/share/uv/tools/harbor/lib/python3.13/site-packages/harbor/environments/docker/harbor-docker-egress-control-sidecar`

## Fair play

The harness never reads task `tests/`, graders, or `artifacts/`, and the system
prompt tells the model not to look for them. No task-specific branching.
# Cross-harness baseline runner

The existing preregistered campaign now runs the independent generic baseline,
DeepSeek Harness, then the one PLE harness. The replacement baseline changes no
models, tasks, tools, budgets, judges, receipts, atoms, trajectories, or final
admission rules. Validate or run its current seat with:

```sh
MODEL_SEAT=mistral scripts/run_generic_baseline.sh check
MODEL_SEAT=mistral scripts/run_generic_baseline.sh run wind-diaphragm-shear
```

The same baseline is available through the common launcher as
`scripts/run_ablation.sh generic check` and
`scripts/run_ablation.sh generic run <task>`.

For the lightweight live behavior probe (27 requested calls: 20 distinct, six
intentional checksum repetitions, one finish), use
`scripts/run_multiturn_probe.sh <generic|ple> <mistral|step5|nemotron-ultra>`.
It bypasses Harbor and PLE-dev tasks and writes an atomic `results.json` plus
full observer `trajectory.jsonl` under `ablations/probes/`. This probe measures
tool/loop behavior only, not engineering-task quality.
Run `scripts/summarize_multiturn_probes.py` to atomically update the compact
cross-arm table at `ablations/probes/results.json`.
For each closed run, `scripts/mint_multiturn_probe.py <run>/results.json` mints
and reads back harness-side observation atoms and one trajectory in the nested
Surreal mind. These are labeled observer records, not model learning.

## Recall-first forward and attention economics

The [receipt-backed graphs](reports/attention-economics/README.md) show the
current generic-versus-PLE probe and a separate Nemotron carving replay.

The [active inference and FEP integration audit](docs/recall-first-forward/FEP-INTEGRATION.md)
lists the missing controller contract and forward-pass gates. It is a design
audit, not a qualified FEP implementation.

The implementation design, frozen preregistration, and resume instructions are
in `docs/recall-first-forward/`. The proposed policy uses a bounded scout
inference to select verified nested-Surreal atoms before the main prefill. A
local runtime may reuse a clean task-prefix branch only after full-state and
logit parity; hosted seats use separate scout and main calls. Attention costs,
quality, latency, and physical energy are separate measured endpoints. This
policy and the attention-economics paper results are not implemented yet.
