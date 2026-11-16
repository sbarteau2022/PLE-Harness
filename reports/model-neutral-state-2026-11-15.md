# PLE model-neutral state verification — 2026-11-15

Historical state-boundary receipt. Subsequent chat dispatch is recorded in
[local-chat-dispatch-2026-11-15.md](local-chat-dispatch-2026-11-15.md).

Scope: PLE harness only. No Meta Hackathon source changes, production database
writes, or production runtime plugin qualification in this change.

## Implemented

- Uniform K/V shorthand and independently described attention layers share the
  existing exact-prefix admission gate. Runtime shape is not fixed to Glimmer
  or to 256 layers; a 384-layer fixture is admitted within aggregate budgets.
- Named runtime state covers fixed recurrent tensors, windowed token axes,
  integer scalar counters and explicitly contracted opaque packed bytes.
  The selected runtime supplies the complete state schema; no cache conversion
  between models is claimed.
- Heterogeneous attention layers assemble from explicit native per-layer
  sessions using the existing nested SurrealDB tables and binary Rust bridge.
  The disposable database test exercises different head counts, dimensions,
  dtypes and position contracts in one assembled checkpoint.
- In-process synthetic hybrid recurrence has equal uninterrupted and resumed
  generated IDs for prefixes of length 1, 3 and 32. This checks buffer admission
  and continuation semantics, not any trained model's numerical quality.

## Adversarial review and fixes

- Every layer's metadata is validated before the first tensor fetch. Missing
  state, wrong shapes/dtypes/positions, incomplete heads, swapped buffers,
  mixed requests, altered contracts, overlap and changed sessions are refused.
- Payload mutation, non-finite declared float state, changed causal prefixes,
  mutable buffers and output token/limit errors are rejected.
- Identity changes during continuation discard returned output. Runtime plugins
  must still check the selected model atomically inside their forward operation.
- Assembly formerly expanded the complete uniform layer list for every page
  fragment. It now expands once per assembly phase; a schema-access regression
  check bounds the number of full-schema passes independently of fragments.
- Aggregate component and byte limits are checked before payload admission.
  Work remains proportional to metadata components/fragments plus payload bytes;
  this receipt does not establish indexed hyperbolic retrieval or O(log N)
  total forward work.

## Validation

- Full Python suite, with `PLE_NESTED_KV_BRIDGE` set to the built PLE bridge:
  **133 passed**, including the disposable SurrealDB integration (not skipped).
- PLE serve protocol smoke: **passed**.
- Rust binary bridge: **2 tests passed** (offline build/test).

## Remaining limits

The native nested schema persists rank-two K/V with equal K/V head dimensions.
Named recurrent/packed-state persistence is not implemented and fails before
lookup. It is not encoded as fake K/V or written to SQLite. The generic local
runtime boundary can carry that state in process, but production plugins and
independent real-model restart/logit parity tests remain open. The agent chat
loop still uses HTTP; local checkpoint export and cache-backed chat dispatch
are explicit runtime APIs, not automatic chat behavior. TPPE relocation does
not make unrelated causal prefixes reusable. Hosted model attention is still
owned by its provider.
