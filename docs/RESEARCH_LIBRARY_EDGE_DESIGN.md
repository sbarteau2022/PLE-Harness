# Research library Edge copy design

## Scope

PLE harness owns the migration code. The existing Qdrant Server collection
`research_library_v1` and filesystem preprints remain readable during migration.
The target is one local Qdrant Edge shard. Its original paper bytes and search
indexes live in the same shard directory; no separate embedding endpoint is
required for the copy.

## Copy protocol

1. Download a server **shard** snapshot to a temporary file. Unpack it into a
   new, empty target directory with Qdrant Edge's `unpack_snapshot`. Preserve
   point IDs, payload, dense/sparse vectors, and payload indexes.
2. Compare source and target point counts and inspect lexical retrieval. Abort
   without modifying the source if these checks fail.
3. For each regular PDF or Markdown file under the preprint root, write a
   content-addressed manifest and base64 chunks as payload-only Edge points.
   Preserve the relative path, exact byte count, SHA-256, and MIME type. Skip
   symlinks. A repeated run overwrites the same IDs.
4. Traverse the copied index's `source_path` values and archive every other
   existing local file it names, including files outside the vault. Record
   unresolved paths as migration gaps. Use disjoint archive keys for preprints,
   other vault files, and external files.
5. Copy server points added or changed during the archive pass and check that
   every current server point ID and payload is represented in Edge. This is a catch-up pass;
   final cutover still requires a stable source or a final sync.
6. Reconstruct every archived file through Edge reads and compare exact bytes
   and SHA-256. Keep a resumable progress report outside the shard directory.
7. Only after all counts, files, retrieval paths, and application integration
   pass will a later cutover replace server access. Deletion of originals is
   a separate, explicit post-verification step.

## Invariants and gates

- Snapshot creation never overwrites an existing target shard.
- The copy must contain at least the source's exact point count at snapshot time.
- Every archived source path must have one manifest and all ordered chunks.
- A failed or interrupted copy is resumable without deleting source data.
- A byte round trip is necessary but does not establish retrieval quality;
  the BM25/dense/visual benchmark remains a separate admission gate.
- Qdrant Edge 0.9.0 is beta and a shard directory has one process owner.

## Runtime cutover and condensation

- `frontier.n3.Rag` opens the Edge shard for each operation under an advisory
  lock in the shard directory, then closes it. This lets independent model
  sessions share one local store without an embedding HTTP service. A missing
  shard or incompatible text index fails closed.
- `serve.solve` exposes read-only `rag_search` and `rag_open` to the PLE task
  agent when that shard directory exists. Competition environments without
  a mounted shard retain the original task tool set. Ingestion stays within
  the frontier research workflow.
- Search reads the existing scored `text` payload index. Ingestion writes
  payload-only points to that same shard and flushes before returning. The
  stored legacy dense vectors stay available for a future admitted query
  encoder; no model is allowed to reinterpret them.
- Reingestion replaces deterministic text chunks for the same URL and removes
  any surplus old chunks. It rejects a point ID collision with a dense-vector
  record rather than clearing those vectors.
- Preprints with exact bytes but no old text points are extracted to text and
  indexed in Edge. If text extraction fails, the filename remains searchable
  and the original PDF bytes remain restorable.
- Before source removal, verify the reopened Edge shard against every existing
  local source byte, server point IDs and payloads, vector count, and a runtime
  search/ingest smoke. Wait for active server-backed frontier runs to end.
- The removal list is built from the verified archive manifests, restricted
  to exact regular files. Remove only those verified originals and the exact
  server collection after a final sync and verification. The Edge shard itself
  is never a deletion target. Record the receipt outside the shard.
- A separate legacy hash384 research collection may contain payload-only
  verified sources. Copy its text and provenance to Edge without treating its
  384-dimensional hash vectors as the 1024-dimensional Qwen vectors.
- After condensation, scan every Edge archive manifest and reconstruct each
  file internally, then check vector count and live BM25 retrieval without
  any original or server dependency.

## Operational note

The requested `Desktop/OPS_BOARD/DOCUMENTATION STANDARDS.MD` was not present
at that path during this run. This document records assumptions, protocol,
invariants, and gates so the implementation remains reviewable.
