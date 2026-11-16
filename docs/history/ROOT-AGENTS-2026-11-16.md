# PLE Exam project instructions

Follow the global Codex instructions in `/Users/stewartbarteau/.codex/AGENTS.md`.

## Verified research memory

- The canonical research library is the local Qdrant Edge shard at `/Users/stewartbarteau/Desktop/Ai Research and Preprint Vault/qdrant-edge-research-v1`.
- Query it before repeating literature searches. After primary-source verification, ingest sources through `harness/frontier/n3.py` (`Rag.ingest`) or `/Users/stewartbarteau/Desktop/Fraud-Predict Comp/src/rag_ingest.py`.
- Preserve the scored `text` index and all existing dense vectors. Use payload-only deterministic points, the `.ple-edge.lock`, post-write hash checks, and live retrieval verification.
- Store only verified public research by default. Do not ingest exam data, run artifacts, private code, credentials, or unpublished findings without explicit approval.
- Record source URLs, evidence notes, and an ingestion receipt. Research claims remain invalid without opened primary-source evidence.
