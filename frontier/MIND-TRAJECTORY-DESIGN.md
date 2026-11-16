# Shared mind trajectory interface

Status: implementation candidate, 2026-11-15. Owner: PLE harness.

## Purpose

The PLE harness currently mints stage trajectories automatically, but a model
cannot read a named trajectory page or deliberately mint a walk over atoms it
has verified. The same harness also defaults its short-term N3 store to a
separate SurrealDB namespace, although the intended short-term store is SQLite.

## Contract

- `trajectory_get(id, start, limit)` reads a `distilled_trajectory` by its exact
  record ID in `hypermind/core`. It returns ordered atom content and a next
  offset. The page is bounded to 16 atoms; a missing/foreign trajectory, missing
  atom, changed workspace, or text digest mismatch fails closed. Pagination
  follows the stored array order; it does not infer an order from graph edges.
- `mind_mint_trajectory(atoms, evidence)` accepts 2–32 explicit atom IDs. The
  harness checks every atom exists in the shared workspace and verifies inline
  text hashes before the existing lineage writer mints the walk and `next`
  edges. The evidence text is hashed into the judge receipt; this is a
  structural admission, not independent proof of the model's claim.
- Automatic recall names the trajectory ID so the model can open it. Models
  never send raw SurrealQL. The shared database remains `hypermind/core`, where
  both knowledge atoms and model-bound tensor page metadata live.
- `open_memory` defaults to local SQLite for short-term notes and carved tool
  output. An explicitly selected Surreal memory backend remains available for
  older sessions; this change does not migrate or delete its data.
- SQLite's Poincare radius follows the earlier `tanh(depth/2)` law through
  depth 7, then grows logarithmically in hyperbolic radius. This avoids the
  previous 0.999 radius clamp, which merged deeper generations. Recall uses the
  exact Poincare distance identity in a numerically stable `asinh` form and
  rejects points outside the open ball. Existing stored points remain valid;
  their coordinates are not rewritten. A missing parent is an error rather
  than a silent root placement.

## Verification and limits

Run the harness test suite and protocol smoke, then page through an existing
real trajectory in the live database without writing to it. Test malformed
IDs, page bounds, foreign or missing atoms, and tampered content. Exact record
lookup bounds traversal work by page size, but discovering trajectories by
substring search and nearest-neighbor retrieval still lack an O(log n) proof.
This change does not inject tensor K/V into a hosted model or qualify TPPE as a
replacement for its trained attention transform.

SQLite recall uses an indexed FTS5 candidate set and an exact Poincare metric
on that set. When FTS5 is unavailable, it examines at most the 256 newest row
IDs, so older lexical matches may be missed instead of triggering a full-table
LIKE scan. Its pinned prompt view uses an index and a 20-entry limit; the full
pin set remains retrievable by ID.

## Hosted conversation budget

Old complete turns are serialized to SQLite as linked conversation archives.
The live prompt keeps one pointer, the system/task instructions, a bounded core
memory view, and the recent complete tool-call groups. Archive IDs can be read
in bounded character pages, including the pointer to an earlier archive. An
assistant tool call and its results are always moved together; the latest
in-flight group stays intact. This limits transmitted history independently
of the number of completed steps, provided the fixed prompt and recent window
are themselves bounded. It does not change the provider's internal attention
or guarantee any specific latency or energy saving without a live benchmark.

Verification on 2026-11-15: 96 Python tests (including the explicitly enabled
disposable nested-SurrealDB gate), two Rust bridge tests, and the protocol smoke
passed after the nested read bridge was added. A synthetic 80-step tool-history simulation
sent 4,955,523 estimated tokens append-only and 546,019 with carving. One live
Mistral transport probe used a synthetic history, carved its estimate from
42,869 to 6,231, and received a native `finish` call; the provider reported
2,594 prompt tokens. Repetitive probe text explains why this provider count
differs from the character estimate. This is not a live project benchmark.
