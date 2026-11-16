"""Copy the canonical Qdrant shard and exact preprint bytes into Qdrant Edge.

The source server and original files are never modified. See
RESEARCH_LIBRARY_EDGE_DESIGN.md for the admission and cutover gates.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request
import uuid

from qdrant_edge import EdgeShard, Point, QueryRequest, ScrollRequest, TextQuery, UpdateOperation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontier.n3 import Rag, research_chunks, research_point_id  # noqa: E402


ARCHIVE_VERSION = "ple-edge-source-v1"
CHUNK_BYTES = 256 * 1024
SOURCE_URL = "http://127.0.0.1:6333"
COLLECTION = "research_library_v1"
DEFAULT_TARGET = Path("/Users/stewartbarteau/Desktop/Ai Research and Preprint Vault/qdrant-edge-research-v1")
DEFAULT_PREPRINTS = Path("/Users/stewartbarteau/Desktop/Ai Research and Preprint Vault/preprints")


def _json(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _point_id(relative_path: str, digest: str, kind: str, index: int | None = None) -> str:
    key = f"{ARCHIVE_VERSION}:{relative_path}:{digest}:{kind}:{index if index is not None else ''}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _manifest_id(relative_path: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{ARCHIVE_VERSION}:{relative_path}:manifest"))


def _records(shard: EdgeShard, ids: list[str]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for start in range(0, len(ids), 32):
        for record in shard.retrieve(ids[start:start + 32], True, False):
            result[str(record.id)] = record.payload or {}
    return result


def restore_file(shard: EdgeShard, relative_path: str) -> bytes:
    manifest_id = _manifest_id(relative_path)
    manifest = _records(shard, [manifest_id]).get(manifest_id)
    if not manifest or manifest.get("record_kind") != "source_manifest":
        raise ValueError(f"missing manifest: {relative_path}")
    if manifest.get("archive_version") != ARCHIVE_VERSION or manifest.get("artifact_path") != relative_path:
        raise ValueError(f"incompatible manifest: {relative_path}")
    digest = manifest["artifact_sha256"]
    count = manifest["artifact_chunks"]
    size = manifest["artifact_size"]
    if not isinstance(count, int) or count < 1 or not isinstance(size, int) or size < 0:
        raise ValueError(f"invalid manifest dimensions: {relative_path}")
    ids = [_point_id(relative_path, digest, "chunk", i) for i in range(count)]
    chunks = _records(shard, ids)
    if len(chunks) != count:
        raise ValueError(f"missing chunks: {relative_path}")
    data = bytearray()
    for index, point_id in enumerate(ids):
        payload = chunks[point_id]
        if (payload.get("record_kind") != "source_chunk" or
                payload.get("artifact_sha256") != digest or
                payload.get("artifact_path") != relative_path or
                payload.get("artifact_chunk_index") != index):
            raise ValueError(f"invalid chunk {index}: {relative_path}")
        data.extend(base64.b64decode(payload["artifact_b64"], validate=True))
    restored = bytes(data)
    if len(restored) != size or hashlib.sha256(restored).hexdigest() != digest:
        raise ValueError(f"byte verification failed: {relative_path}")
    return restored


def archive_file(shard: EdgeShard, root: Path, path: Path, key: str | None = None) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular file: {path}")
    relative_path = key if key is not None else path.relative_to(root).as_posix()
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    existing = _records(shard, [_manifest_id(relative_path)]).get(_manifest_id(relative_path))
    if existing and existing.get("artifact_sha256") == digest:
        if restore_file(shard, relative_path) != source:
            raise ValueError(f"existing archive differs: {relative_path}")
        return {"path": relative_path, "bytes": len(source), "sha256": digest, "status": "verified-existing"}
    chunks = [source[i:i + CHUNK_BYTES] for i in range(0, len(source), CHUNK_BYTES)] or [b""]
    for start in range(0, len(chunks), 4):
        points = []
        for index in range(start, min(start + 4, len(chunks))):
            points.append(Point(
                id=_point_id(relative_path, digest, "chunk", index), vector={},
                payload={"archive_version": ARCHIVE_VERSION, "record_kind": "source_chunk",
                         "artifact_path": relative_path, "artifact_sha256": digest,
                         "artifact_chunk_index": index,
                         "artifact_b64": base64.b64encode(chunks[index]).decode("ascii")},
            ))
        shard.update(UpdateOperation.upsert_points(points))
    manifest = {"archive_version": ARCHIVE_VERSION, "record_kind": "source_manifest",
                "artifact_path": relative_path, "artifact_sha256": digest,
                "artifact_size": len(source), "artifact_chunks": len(chunks),
                "artifact_source_path": str(path),
                "artifact_mime": mimetypes.guess_type(path.name)[0] or "application/octet-stream"}
    shard.update(UpdateOperation.upsert_points([Point(id=_manifest_id(relative_path), vector={}, payload=manifest)]))
    if restore_file(shard, relative_path) != source:
        raise ValueError(f"new archive differs: {relative_path}")
    return {"path": relative_path, "bytes": len(source), "sha256": digest, "status": "verified-new"}


def copy_snapshot(source_url: str, collection: str, target: Path) -> int:
    if target.exists():
        raise FileExistsError(f"target already exists; refusing overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source_root = f"{source_url.rstrip('/')}/collections/{collection}"
    before = _json(f"{source_root}/points/count", {"exact": True})["result"]["count"]
    snapshot_url = f"{source_root}/shards/0/snapshot"
    with tempfile.NamedTemporaryFile(prefix="ple-edge-shard-", suffix=".snapshot",
                                     dir=target.parent, delete=False) as temp:
        snapshot_path = Path(temp.name)
    try:
        with urllib.request.urlopen(snapshot_url, timeout=600) as response, snapshot_path.open("wb") as output:
            while chunk := response.read(4 * 1024 * 1024):
                output.write(chunk)
        print(f"downloaded snapshot: {snapshot_path.stat().st_size} bytes", flush=True)
        target.mkdir()
        EdgeShard.unpack_snapshot(str(snapshot_path), str(target))
        shard = EdgeShard.load(str(target))
        try:
            copied = shard.info().points_count
            if copied != before:
                raise ValueError(f"point count mismatch: source {before}, Edge {copied}")
            hits = shard.query(QueryRequest(query=TextQuery("text", "hyperbolic"), limit=3,
                                            with_payload=True))
            if not hits or not any((hit.payload or {}).get("text") for hit in hits):
                raise ValueError("BM25 text retrieval smoke test returned no passages")
        finally:
            shard.close()
        print(f"snapshot verified: {copied} points; BM25 returned {len(hits)} passages", flush=True)
        return copied
    finally:
        snapshot_path.unlink(missing_ok=True)


def archive_preprints(target: Path, preprints: Path) -> tuple[int, int]:
    if not target.is_dir() or not preprints.is_dir():
        raise FileNotFoundError("Edge shard and preprint directory must both exist")
    paths = sorted(path for path in preprints.rglob("*") if path.suffix.lower() in {".pdf", ".md"})
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("preprint set contains a symlink or non-file with a paper suffix")
    shard = EdgeShard.load(str(target))
    report_path = target.parent / f"{target.name}.migration.jsonl"
    total_bytes = 0
    try:
        with report_path.open("a", encoding="utf-8") as report:
            for number, path in enumerate(paths, 1):
                entry = archive_file(shard, preprints, path)
                total_bytes += entry["bytes"]
                report.write(json.dumps(entry, sort_keys=True) + "\n")
                report.flush()
                if number % 25 == 0 or number == len(paths):
                    print(f"archived {number}/{len(paths)} files ({total_bytes} source bytes)", flush=True)
        shard.flush()
    finally:
        shard.close()
    return len(paths), total_bytes


def indexed_source_paths(shard: EdgeShard) -> set[str]:
    paths: set[str] = set()
    offset = None
    while True:
        records, offset = shard.scroll(ScrollRequest(limit=1000, offset=offset,
                                                     with_payload=["source_path"], with_vector=False))
        for record in records:
            source = (record.payload or {}).get("source_path")
            if isinstance(source, str) and source:
                paths.add(source)
        if offset is None:
            return paths


def drifted_server_ids(shard: EdgeShard, source_url: str, collection: str) -> list[int | str]:
    """Find server points absent from Edge or carrying different payloads."""
    root = f"{source_url.rstrip('/')}/collections/{collection}"
    drifted: list[int | str] = []
    offset = None
    while True:
        request = {"limit": 256, "with_payload": True, "with_vector": False}
        if offset is not None:
            request["offset"] = offset
        result = _json(f"{root}/points/scroll", request)["result"]
        source_points = result["points"]
        ids = [point["id"] for point in source_points]
        copied = {str(record.id): record.payload or {}
                  for record in shard.retrieve(ids, True, False)}
        drifted.extend(point["id"] for point in source_points
                       if copied.get(str(point["id"])) != (point.get("payload") or {}))
        offset = result.get("next_page_offset")
        if offset is None:
            return drifted


def sync_server_updates(target: Path, source_url: str, collection: str) -> int:
    """Catch up inserted or updated server payloads; never deletes Edge data."""
    shard = EdgeShard.load(str(target))
    try:
        drifted = drifted_server_ids(shard, source_url, collection)
        root = f"{source_url.rstrip('/')}/collections/{collection}"
        for start in range(0, len(drifted), 32):
            batch = drifted[start:start + 32]
            result = _json(f"{root}/points", {"ids": batch, "with_payload": True,
                                               "with_vector": True})["result"]
            if {str(point["id"]) for point in result} != {str(point_id) for point_id in batch}:
                raise ValueError("source point changed during incremental copy")
            shard.update(UpdateOperation.upsert_points([
                Point(point["id"], point.get("vector") or {}, point.get("payload") or {})
                for point in result
            ]))
        shard.flush()
        return len(drifted)
    finally:
        shard.close()


def archive_key(preprints: Path, path: Path) -> str:
    if path.is_relative_to(preprints):
        return path.relative_to(preprints).as_posix()
    vault = preprints.parent
    if path.is_relative_to(vault):
        return "vault/" + path.relative_to(vault).as_posix()
    path_hash = hashlib.sha256(str(path).encode()).hexdigest()
    return f"external/{path_hash}/{path.name}"


def archive_indexed_sources(target: Path, preprints: Path) -> dict:
    """Archive each available local file named in the copied index."""
    shard = EdgeShard.load(str(target))
    vault = preprints.parent
    report_path = target.parent / f"{target.name}.migration.jsonl"
    try:
        source_paths = indexed_source_paths(shard)
        existing: list[Path] = []
        missing: list[str] = []
        for source in sorted(source_paths):
            path = Path(source)
            if not path.is_absolute() or path.is_symlink() or not path.is_file():
                missing.append(source)
            elif not path.is_relative_to(preprints):
                existing.append(path)
        total_bytes = 0
        with report_path.open("a", encoding="utf-8") as report:
            for number, path in enumerate(existing, 1):
                entry = archive_file(shard, vault, path, key=archive_key(preprints, path))
                total_bytes += entry["bytes"]
                report.write(json.dumps(entry, sort_keys=True) + "\n")
                report.flush()
                if number % 25 == 0 or number == len(existing):
                    print(f"archived indexed sources {number}/{len(existing)} "
                          f"({total_bytes} source bytes)", flush=True)
            for source in missing:
                report.write(json.dumps({"source_path": source, "status": "missing-local-source"}) + "\n")
        shard.flush()
        return {"indexed_paths": len(source_paths), "additional_files": len(existing),
                "additional_bytes": total_bytes, "missing_local_paths": len(missing)}
    finally:
        shard.close()


def index_missing_preprints(target: Path, preprints: Path) -> dict:
    """Make archived preprints absent from the old index searchable in Edge."""
    rag = Rag(target)
    indexed = 0
    chunks_written = 0
    extraction_failures: set[str] = set()
    with rag._open() as shard:
        known = indexed_source_paths(shard)
        for path in sorted(preprints.rglob("*")):
            if path.suffix.lower() not in {".pdf", ".md"}:
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"preprint is not a regular file: {path}")
            if str(path) in known:
                continue
            if restore_file(shard, archive_key(preprints, path)) != path.read_bytes():
                raise ValueError(f"preprint archive differs before indexing: {path}")
            if path.suffix.lower() == ".pdf":
                extraction = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                                            capture_output=True, timeout=120, check=False)
                if extraction.returncode != 0:
                    extraction_failures.add(str(path))
                    text = ""
                else:
                    text = extraction.stdout.decode("utf-8", errors="replace")
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                extraction_failures.add(str(path))
                text = path.stem.replace("-", " ").replace("_", " ")
            points = []
            for index, chunk in enumerate(research_chunks(text)):
                points.append(Point(id=research_point_id(str(path), index), vector={}, payload={
                    "source_path": str(path), "document_id": path.stem, "title": path.stem,
                    "source_type": path.suffix.lower().lstrip("."),
                    "content_hash": hashlib.sha256(chunk.encode()).hexdigest(),
                    "language": "und", "research_domain": "preprints", "tenant_id": "local",
                    "silo": "research", "chunk_index": index, "text": chunk,
                    "archive_key": archive_key(preprints, path),
                    "archive_manifest_id": _manifest_id(archive_key(preprints, path)),
                    "retrieval_mode": "qdrant-edge-bm25-text-v1",
                }))
            for start in range(0, len(points), 32):
                shard.update(UpdateOperation.upsert_points(points[start:start + 32]))
            indexed += 1
            chunks_written += len(points)
        shard.flush()
    return {"indexed_preprints": indexed, "text_chunks": chunks_written,
            "extraction_failures": sorted(extraction_failures)}


def verify_copy(target: Path, preprints: Path, source_url: str, collection: str) -> dict:
    """Reopen and verify search plus every available source file without writing."""
    source_root = f"{source_url.rstrip('/')}/collections/{collection}"
    source_points = _json(f"{source_root}/points/count", {"exact": True})["result"]["count"]
    source_vectors = _json(source_root)["result"]["indexed_vectors_count"]
    shard = EdgeShard.load(str(target))
    try:
        info = shard.info()
        if info.points_count < source_points or info.indexed_vectors_count != source_vectors:
            raise ValueError("copied point or vector count differs from the source")
        drifted_ids = drifted_server_ids(shard, source_url, collection)
        if drifted_ids:
            raise ValueError(f"{len(drifted_ids)} server points are absent or changed in Edge")
        hits = shard.query(QueryRequest(query=TextQuery("text", "hyperbolic geometry"),
                                        limit=3, with_payload=True))
        if not hits or not all((hit.payload or {}).get("text") for hit in hits):
            raise ValueError("Edge BM25 query failed after reopening")
        preprint_paths = sorted(path for path in preprints.rglob("*")
                                if path.suffix.lower() in {".pdf", ".md"})
        source_paths = indexed_source_paths(shard)
        all_paths = set(preprint_paths)
        missing: list[str] = []
        for source in source_paths:
            path = Path(source)
            if path.is_absolute() and not path.is_symlink() and path.is_file():
                all_paths.add(path)
            else:
                missing.append(source)
        verified_bytes = 0
        for number, path in enumerate(sorted(all_paths), 1):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"source changed or became a symlink: {path}")
            source_bytes = path.read_bytes()
            if restore_file(shard, archive_key(preprints, path)) != source_bytes:
                raise ValueError(f"source and Edge bytes differ: {path}")
            verified_bytes += len(source_bytes)
            if number % 250 == 0 or number == len(all_paths):
                print(f"verified after reopen {number}/{len(all_paths)} files", flush=True)
        return {"source_points": source_points, "source_indexed_vectors": source_vectors,
                "edge_points": info.points_count, "edge_indexed_vectors": info.indexed_vectors_count,
                "bm25_hits": len(hits), "verified_files": len(all_paths),
                "verified_bytes": verified_bytes, "missing_local_paths": len(missing)}
    finally:
        shard.close()


def restore_to_path(target: Path, key: str, output: Path) -> int:
    """Restore exact archived bytes to a new file; never overwrite a file."""
    shard = EdgeShard.load(str(target))
    try:
        data = restore_file(shard, key)
    finally:
        shard.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        destination.write(data)
        destination.flush()
        os.fsync(destination.fileno())
    return len(data)


def copy_legacy_hash_sources(target: Path, source_url: str,
                             collection: str = "frontier_research_hash384_v1") -> int:
    """Preserve older verified-source text without its incompatible hash vectors."""
    root = f"{source_url.rstrip('/')}/collections/{collection}"
    offset = None
    copied = 0
    rag = Rag(target)
    with rag._open() as shard:
        while True:
            request = {"limit": 256, "with_payload": True, "with_vector": False}
            if offset is not None:
                request["offset"] = offset
            result = _json(f"{root}/points/scroll", request)["result"]
            for item in result["points"]:
                payload = dict(item.get("payload") or {})
                source = payload.get("url") or payload.get("source_path")
                text = payload.get("text")
                if not isinstance(source, str) or not isinstance(text, str) or not text:
                    raise ValueError(f"legacy source lacks URL or text: {item['id']}")
                payload.update({"source_path": source,
                                "document_id": payload.get("title") or source,
                                "source_type": "verified-web",
                                "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                                "retrieval_mode": "qdrant-edge-bm25-text-v1",
                                "legacy_collection": collection,
                                "legacy_point_id": str(item["id"])})
                payload.pop("encoder", None)
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
                                          f"ple-edge-legacy:{collection}:{item['id']}"))
                previous = shard.retrieve([point_id], True, False)
                if previous:
                    if (previous[0].payload or {}) != payload:
                        raise ValueError(f"legacy point ID already exists with different payload: {point_id}")
                    continue
                shard.update(UpdateOperation.upsert_points([Point(point_id, {}, payload)]))
                copied += 1
            offset = result.get("next_page_offset")
            if offset is None:
                break
        shard.flush()
    return copied


def verify_edge_internal(target: Path, expected_files: int, expected_bytes: int,
                         expected_vectors: int) -> dict:
    """Verify archived bytes and retrieval after originals and server are gone."""
    shard = EdgeShard.load(str(target))
    try:
        info = shard.info()
        if info.indexed_vectors_count != expected_vectors:
            raise ValueError("Edge indexed-vector count changed")
        manifests: dict[str, dict] = {}
        offset = None
        while True:
            records, offset = shard.scroll(ScrollRequest(
                limit=1000, offset=offset,
                with_payload=["record_kind", "artifact_path", "artifact_size", "artifact_sha256"],
                with_vector=False))
            for record in records:
                payload = record.payload or {}
                if payload.get("record_kind") == "source_manifest":
                    path = payload.get("artifact_path")
                    if not isinstance(path, str) or path in manifests:
                        raise ValueError("missing or duplicate archive path")
                    manifests[path] = payload
            if offset is None:
                break
        if len(manifests) != expected_files:
            raise ValueError(f"archive manifest count {len(manifests)} != {expected_files}")
        total_bytes = 0
        for path, manifest in manifests.items():
            data = restore_file(shard, path)
            if (len(data) != manifest["artifact_size"] or
                    hashlib.sha256(data).hexdigest() != manifest["artifact_sha256"]):
                raise ValueError(f"archive bytes failed internal check: {path}")
            total_bytes += len(data)
        if total_bytes != expected_bytes:
            raise ValueError(f"archive byte count {total_bytes} != {expected_bytes}")
        hits = shard.query(QueryRequest(query=TextQuery("text", "hyperbolic geometry"),
                                        limit=3, with_payload=True))
        if not hits or not (hits[0].payload or {}).get("text"):
            raise ValueError("Edge retrieval smoke failed")
        return {"edge_points": info.points_count, "indexed_vectors": info.indexed_vectors_count,
                "archive_files": len(manifests), "archive_bytes": total_bytes,
                "bm25_hits": len(hits)}
    finally:
        shard.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("snapshot", "archive", "index-preprints", "indexed",
                                            "sync", "legacy-hash", "verify", "verify-edge",
                                            "restore", "all"), default="all")
    parser.add_argument("--source-url", default=os.environ.get("FRONTIER_QDRANT_URL", SOURCE_URL))
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--preprints", type=Path, default=DEFAULT_PREPRINTS)
    parser.add_argument("--restore-source", type=Path)
    parser.add_argument("--restore-key")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-files", type=int)
    parser.add_argument("--expected-bytes", type=int)
    parser.add_argument("--expected-vectors", type=int)
    args = parser.parse_args()
    if args.phase == "restore":
        if args.output is None or (args.restore_source is None) == (args.restore_key is None):
            parser.error("restore requires --output and exactly one of --restore-source or --restore-key")
        key = args.restore_key or archive_key(args.preprints, args.restore_source)
        size = restore_to_path(args.target, key, args.output)
        print(f"restored {size} bytes to {args.output}", flush=True)
        return
    if args.phase in {"snapshot", "all"}:
        copy_snapshot(args.source_url, args.collection, args.target)
    if args.phase in {"archive", "all"}:
        count, size = archive_preprints(args.target, args.preprints)
        print(f"archive complete: {count} files, {size} exact source bytes", flush=True)
    if args.phase in {"index-preprints", "all"}:
        summary = index_missing_preprints(args.target, args.preprints)
        print("new preprint text index: " + json.dumps(summary, sort_keys=True), flush=True)
    if args.phase in {"indexed", "all"}:
        summary = archive_indexed_sources(args.target, args.preprints)
        print("indexed source archive: " + json.dumps(summary, sort_keys=True), flush=True)
    if args.phase in {"sync", "all"}:
        copied = sync_server_updates(args.target, args.source_url, args.collection)
        print(f"server points copied or refreshed: {copied}", flush=True)
    if args.phase == "legacy-hash":
        copied = copy_legacy_hash_sources(args.target, args.source_url)
        print(f"legacy verified sources copied: {copied}", flush=True)
    if args.phase in {"verify", "all"}:
        summary = verify_copy(args.target, args.preprints, args.source_url, args.collection)
        print("copy verification: " + json.dumps(summary, sort_keys=True), flush=True)
    if args.phase == "verify-edge":
        if None in (args.expected_files, args.expected_bytes, args.expected_vectors):
            parser.error("verify-edge requires --expected-files, --expected-bytes, and --expected-vectors")
        summary = verify_edge_internal(args.target, args.expected_files,
                                       args.expected_bytes, args.expected_vectors)
        print("Edge-only verification: " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
