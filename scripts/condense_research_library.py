"""Remove exact verified originals after the Edge library is operational.

Plan, stage, smoke, then commit. Staging moves only files whose bytes already
round-trip through Edge. Commit releases the staged bytes after another check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from qdrant_edge import EdgeShard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontier.n3 import Rag  # noqa: E402
from scripts.migrate_research_library_to_edge import (  # noqa: E402
    DEFAULT_PREPRINTS,
    DEFAULT_TARGET,
    archive_key,
    indexed_source_paths,
    restore_file,
)


def _plan_path(quarantine: Path) -> Path:
    return quarantine / "plan.json"


def _load_plan(quarantine: Path) -> dict:
    plan = json.loads(_plan_path(quarantine).read_text(encoding="utf-8"))
    if plan.get("version") != 1 or not isinstance(plan.get("files"), list):
        raise ValueError("invalid condensation plan")
    return plan


def _files(target: Path, preprints: Path) -> list[Path]:
    shard = EdgeShard.load(str(target))
    try:
        sources = indexed_source_paths(shard)
    finally:
        shard.close()
    files = {path for path in preprints.rglob("*") if path.suffix.lower() in {".pdf", ".md"}}
    files.update(Path(source) for source in sources if Path(source).is_absolute() and Path(source).is_file())
    if any(path.is_symlink() or not path.is_file() or path.is_relative_to(target) for path in files):
        raise ValueError("condensation set contains an unsafe or missing file")
    return sorted(files)


def plan_condensation(target: Path, preprints: Path, quarantine: Path,
                      expected_files: int, expected_bytes: int) -> dict:
    if quarantine.exists():
        raise FileExistsError(f"quarantine already exists: {quarantine}")
    files = _files(target, preprints)
    if len(files) != expected_files:
        raise ValueError(f"file count changed: expected {expected_files}, got {len(files)}")
    shard = EdgeShard.load(str(target))
    rows = []
    total_bytes = 0
    try:
        for path in files:
            original = path.read_bytes()
            key = archive_key(preprints, path)
            if restore_file(shard, key) != original:
                raise ValueError(f"Edge bytes differ from original: {path}")
            digest = hashlib.sha256(original).hexdigest()
            rows.append({"path": str(path), "key": key, "sha256": digest,
                         "bytes": len(original),
                         "quarantine_name": hashlib.sha256(str(path).encode()).hexdigest()})
            total_bytes += len(original)
    finally:
        shard.close()
    if total_bytes != expected_bytes:
        raise ValueError(f"byte count changed: expected {expected_bytes}, got {total_bytes}")
    quarantine.mkdir(parents=True)
    plan = {"version": 1, "target": str(target), "preprints": str(preprints),
            "files": rows, "file_count": len(rows), "source_bytes": total_bytes,
            "phase": "planned"}
    _plan_path(quarantine).write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return {"files": len(rows), "bytes": total_bytes, "quarantine": str(quarantine)}


def stage_condensation(quarantine: Path) -> dict:
    plan = _load_plan(quarantine)
    target = Path(plan["target"])
    shard = EdgeShard.load(str(target))
    try:
        for row in plan["files"]:
            path = Path(row["path"])
            destination = quarantine / row["quarantine_name"]
            if path.is_symlink() or not path.is_file() or destination.exists():
                raise ValueError(f"source changed or destination exists: {path}")
            source = path.read_bytes()
            if (len(source) != row["bytes"] or hashlib.sha256(source).hexdigest() != row["sha256"] or
                    restore_file(shard, row["key"]) != source):
                raise ValueError(f"source changed after plan: {path}")
    finally:
        shard.close()
    moved = []
    try:
        for row in plan["files"]:
            source = Path(row["path"])
            destination = quarantine / row["quarantine_name"]
            current = source.read_bytes()
            if len(current) != row["bytes"] or hashlib.sha256(current).hexdigest() != row["sha256"]:
                raise ValueError(f"source changed while staging: {source}")
            os.replace(source, destination)
            moved.append((source, destination))
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                os.replace(destination, source)
        raise
    plan["phase"] = "staged"
    _plan_path(quarantine).write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return {"staged_files": len(moved), "staged_bytes": plan["source_bytes"]}


def smoke_staged(quarantine: Path, query: str = "hyperbolic geometry",
                 allow_committed: bool = False) -> dict:
    plan = _load_plan(quarantine)
    if plan["phase"] != "staged":
        raise ValueError("condensation has not been staged")
    rag = Rag(plan["target"])
    hits = rag.search(query, 3)
    if not hits or hits[0].get("from") == "error":
        raise ValueError("Edge RAG search failed after source paths were removed")
    source = next((hit["source"] for hit in hits if hit.get("source") and
                   rag.open_document(hit["source"], 0, 1)), None)
    if source is None:
        raise ValueError("Edge RAG could not open a retrieved source")
    shard = EdgeShard.load(plan["target"])
    try:
        for row in plan["files"]:
            original = quarantine / row["quarantine_name"]
            if Path(row["path"]).exists():
                raise ValueError(f"source path was recreated after staging: {row['path']}")
            if original.is_file() and not original.is_symlink():
                staged_bytes = original.read_bytes()
            elif allow_committed and not original.exists() and not Path(row["path"]).exists():
                staged_bytes = restore_file(shard, row["key"])
            else:
                raise ValueError(f"staged file is missing: {original}")
            if (len(staged_bytes) != row["bytes"] or
                    hashlib.sha256(staged_bytes).hexdigest() != row["sha256"] or
                    restore_file(shard, row["key"]) != staged_bytes):
                raise ValueError(f"Edge archive differs from staged original: {row['path']}")
    finally:
        shard.close()
    return {"search_hits": len(hits), "opened_source": source,
            "verified_staged_files": len(plan["files"]), "verified_bytes": plan["source_bytes"]}


def commit_condensation(quarantine: Path, query: str = "hyperbolic geometry") -> dict:
    smoke = smoke_staged(quarantine, query, allow_committed=True)
    plan = _load_plan(quarantine)
    receipt = {"edge_shard": plan["target"], "removed_files": plan["file_count"],
               "removed_source_bytes": plan["source_bytes"], "smoke": smoke}
    receipt_path = quarantine.parent / "qdrant-edge-research-v1.condensation.json"
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing.get("edge_shard") != plan["target"] or existing.get("removed_files") != plan["file_count"]:
            raise FileExistsError(f"different condensation receipt exists: {receipt_path}")
    for row in plan["files"]:
        (quarantine / row["quarantine_name"]).unlink(missing_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    _plan_path(quarantine).unlink()
    quarantine.rmdir()
    return receipt


def rollback_condensation(quarantine: Path) -> int:
    plan = _load_plan(quarantine)
    restored = 0
    for row in plan["files"]:
        original = Path(row["path"])
        staged = quarantine / row["quarantine_name"]
        if staged.exists():
            if original.exists():
                raise FileExistsError(f"original path was recreated: {original}")
            original.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, original)
            restored += 1
    _plan_path(quarantine).unlink()
    quarantine.rmdir()
    return restored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "stage", "smoke", "commit", "rollback"))
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--preprints", type=Path, default=DEFAULT_PREPRINTS)
    parser.add_argument("--quarantine", type=Path, default=DEFAULT_TARGET.parent / ".edge-source-quarantine-v1")
    parser.add_argument("--expected-files", type=int)
    parser.add_argument("--expected-bytes", type=int)
    parser.add_argument("--query", default="hyperbolic geometry")
    args = parser.parse_args()
    if args.phase == "plan":
        if args.expected_files is None or args.expected_bytes is None:
            parser.error("plan requires --expected-files and --expected-bytes")
        result = plan_condensation(args.target, args.preprints, args.quarantine,
                                   args.expected_files, args.expected_bytes)
    elif args.phase == "stage":
        result = stage_condensation(args.quarantine)
    elif args.phase == "smoke":
        result = smoke_staged(args.quarantine, args.query)
    elif args.phase == "commit":
        result = commit_condensation(args.quarantine, args.query)
    else:
        result = {"restored_files": rollback_condensation(args.quarantine)}
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
