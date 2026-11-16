# HyperMind schema (vendored)

The SurrealDB schema of the shared mind (`hypermind/core`): atoms, addresses, lineage,
trajectories, tensor KV pages. Copied unchanged on 2026-11-16 from the hypermind-nested-kv
project so the runtime's tests do not reach outside `harness/`. The live database is the
authority; these files describe it. Do not edit them to change the live schema.

The harness's own additive migrations for named model state live in
`../../native-kv-bridge/schema/` (their 0010/0011 numbers are a separate series).
