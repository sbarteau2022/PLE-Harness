# PLE nested tensor bridge

This bridge reads and appends existing `transformer_kv_session` and `transformer_kv_page`
records through the owning PLE-local `globalmind-runtime` crate. SQLite is not
in this path. No tensor store, table, model, or positional transform is created.

Input is one bounded JSON `ExactKvIdentity` on stdin; record IDs omit table
prefixes, as in the runtime's MCP interface. Output is `PKV1`, two little-endian
u32 lengths, raw K and V bytes, then SHA-256 of request bytes plus preceding
output bytes. Maximum input is 64 KiB; combined tensor output is at most 16 MiB.
Errors return a nonzero status and stderr, with no tensor output.

`put` mode accepts `PKP1`, three little-endian u32 lengths (metadata, K, V),
bounded JSON identity metadata, then raw K/V bytes. Metadata additionally names
one `trajectory_atoms` record (the atom at the selected address) and a
`receipt_digest`. Combined K/V is at most 16 MiB; metadata is at most 64 KiB.
Success returns `PWA1` plus SHA-256 of the entire input. The owning runtime
checks payloads, active same-workspace address/atom linkage, session identity,
and immutable page lineage. Identical publication is idempotent, conflicts
are refused, and no existing page is overwritten. Multi-page publication is
not atomic; complete-checkpoint coordination/discovery remains open.
`Mind.tensor_put` exposes this only as a runtime binary API, never a text tool.
The `schema/` files are read-only copies of the Meta reference's production
0004/0009 definitions, used only in a fresh disposable test database. They are
not automatically applied to production. Tests exercise the SCHEMAFULL fields
and unique indexes, including separate per-layer request identities.

`Mind.tensor_exact(identity, executable, recalled_addresses=...)` pins the bridge
to the same Surreal authority and workspace as the shared mind. Identity
checks include model, prefix, TPPE/native transform, tensor contract, address,
layer/head/token interval, shape, dtype, revision and expected payload digests.
The selected address membership is checked before launching the bridge.
This reads one head page, not a complete model cache. A local runtime adapter
must admit every required head before a forward call. PLE's `Mind.tensor_cache`
now assembles and verifies complete explicit rosters, and
`Mind.continue_tensor_cache` routes them through the model-neutral admission
gate. The disposable integration test exercises this path with a test runtime;
it is not real-model inference evidence.

From the PLE harness directory:

```sh
cargo build --offline --manifest-path native-kv-bridge/Cargo.toml
cargo test --offline --manifest-path native-kv-bridge/Cargo.toml
PLE_NESTED_KV_BRIDGE="$PWD/native-kv-bridge/target/debug/ple-nested-kv-bridge" \
  .venv/bin/python -m pytest tests -q
```

The integration gate starts and stops its own disposable SurrealDB. It seeds
native byte fields in the existing nested schema's table names, reads actual
2x4 K/V payloads through Rust into PLE, and rejects changed model identity,
foreign workspace, address mismatch and corrupt bytes. Its fixture is minimal;
it does not certify the complete production schema, restart persistence,
real-model inference, asymptotic retrieval, or a trained model's forward pass.
The same gate now also exercises the actual PLE agent through a selected
synthetic chat codec, its shared `Mind`, and this Rust bridge into the disposable
database. This establishes the dispatch wiring, not production model readiness.

The runtime's existing authentication distinguishes root from namespace/database
accounts; the disposable test uses root. Its quarantine guard for `core` remains
active unless `GLOBALMIND_ALLOW_QUARANTINED_CORE=1` is explicitly configured.
