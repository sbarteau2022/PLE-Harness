# PLE native single-page publication — 2026-11-15

Implemented `Mind.tensor_put` -> raw binary `put` bridge -> owning PLE-local
`GlobalMindStore.publish_tensor_page_exact`. It uses the existing nested native
session/page tables and the same Surreal authority as the knowledge mind.
SQLite and model text tools never carry tensor state. No production database
writes, production schema migrations, or Meta reference source edits occurred.

The owning store checks dense shape, byte digests, finite values, geometry
existence, active same-workspace address/atom grounding, exact session/page
identity and immutable lineage receipts. Identical retries succeed; conflicts
are refused without overwrite. A single publication can create its session
and page through the owning typed methods; a complete multi-page checkpoint
is not atomic and still needs coordination.

The disposable integration applies copied production SCHEMAFULL 0004/0009
definitions, including READONLY fields and UNIQUE indexes. It verifies native
publication/readback, idempotent retries, conflicting payload/receipt refusal,
bad shapes, missing geometry/address, foreign atom, non-finite payloads and
changed model rejection. Distinct per-layer request digests satisfy the real
workspace/model/request unique index; duplicate requests fail. Published
heterogeneous f32/f16 layers restore through the exact PLE admission gate.
The earlier minimal read fixture did not exercise that index. Layered assembly
was corrected to allow separate immutable per-session request identities.

Final validation: **168 Python tests passed** with the native bridge enabled;
**3 Rust bridge tests passed** offline; **PLE protocol smoke passed**. The
disposable servers were stopped by their test cleanup. Production-schema
coverage here concerns the native tensor tables, not the entire knowledge
schema or runtime deployment.

Automatic checkpoint export/publication, deterministic restart discovery,
named-state persistence, independent real-model qualification, radial reuse
policy, indexed hyperbolic bounds and physical attention/energy claims remain
open. Power-of-two prefix anchors and a typed native context are a documented
plan, not implemented APIs. The exact next work is in `../../HANDOFF.md`.
