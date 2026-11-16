# PLE nested mind / tensor forward handoff

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
`globalmind-runtime/src/kv_identity.rs` and `harness/native-kv-bridge` (`put`
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
