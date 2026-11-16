# PLE Harness documentation

This directory is the documentation home for the [PLE Harness repository](../README.md). The pinned Google PLE kit (`../ple` on the original Mac) is an external dependency and is not edited here.

## Current operating documents

| Read for | Canonical document |
| --- | --- |
| State, limits, and existing campaign work | [Handoff](HANDOFF.md) |
| Additive work beyond that handoff | [Next tasks](NEXT-TASKS-BEYOND-HANDOFF.md) |
| Claude setup | [Paste-ready instructions](CLAUDE-PROJECT-INSTRUCTIONS-PASTE.md), [expanded instructions](CLAUDE-PROJECT-INSTRUCTIONS.md) |
| API/local comparison | [Ablation design](../ablations/DESIGN.md), [frozen preregistration](../ablations/PREREGISTRATION.md), [failure ledger](../ablations/FAILURES.md) |
| Forward-policy experiment | [Recall-first packet](recall-first-forward/README.md), [TPPE/local forward contract](../frontier/TPPE-LOCAL-FORWARD-DESIGN.md) |
| Model-neutral engineering outputs | [Engineering modeling](PLE-ENGINEERING-MODELING.md), [verifier design](ENGINEERING-VERIFIER-DESIGN.md), [artifact design](ARTIFACTS-DESIGN.md) |
| Research source migration | [Qdrant Edge copy design](RESEARCH_LIBRARY_EDGE_DESIGN.md) |
| Writing and publishing | [Documentation standards](DOCUMENTATION_STANDARDS.md) |

## Historical records

The [history directory](history/README.md) holds exact copies of older project-authored files that had been outside this Git repository. They are snapshots, not current instructions. The root `HANDOFF.md` only points to the canonical harness handoff. The ignored `attention-economics/` folder has identical copies of the four tracked recall-first documents and duplicate figures; the tracked packet and `reports/attention-economics/` are authoritative.

The upstream PLE kit, public dev task inputs, generated run directories, private research shard, and third-party paper PDF are not copied into this documentation tree. Their source links and evidence paths are recorded where used.
