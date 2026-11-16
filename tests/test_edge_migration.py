"""Exact-byte and failure-path checks for the Edge copy migration."""

from pathlib import Path
import uuid

import pytest
from qdrant_edge import (Distance, EdgeConfig, EdgeShard, EdgeVectorParams, Point,
                         TextIndexParams, UpdateOperation)

from scripts import migrate_research_library_to_edge as migration
from scripts.migrate_research_library_to_edge import (
    _manifest_id,
    _point_id,
    archive_file,
    archive_indexed_sources,
    archive_preprints,
    copy_legacy_hash_sources,
    index_missing_preprints,
    restore_file,
    restore_to_path,
    sync_server_updates,
    verify_edge_internal,
)


@pytest.fixture
def shard(tmp_path: Path):
    db = EdgeShard.create(str(tmp_path / "edge"), EdgeConfig(
        vectors={"text_rank0": EdgeVectorParams(1024, Distance.Cosine)}))
    try:
        yield db
    finally:
        db.close()


def test_archive_exact_binary_and_idempotent(shard, tmp_path: Path):
    root = tmp_path / "preprints"
    root.mkdir()
    path = root / "paper.pdf"
    data = b"%PDF-1.7\x00\xff" + bytes(range(256)) * 2500
    path.write_bytes(data)
    first = archive_file(shard, root, path)
    assert first["status"] == "verified-new"
    assert restore_file(shard, "paper.pdf") == data
    assert archive_file(shard, root, path)["status"] == "verified-existing"


def test_archive_detects_missing_or_corrupt_chunk(shard, tmp_path: Path):
    root = tmp_path / "preprints"
    root.mkdir()
    path = root / "paper.md"
    path.write_text("formula λ/μ and text\n" * 20000, encoding="utf-8")
    entry = archive_file(shard, root, path)
    chunk_id = _point_id("paper.md", entry["sha256"], "chunk", 0)
    shard.update(UpdateOperation.upsert_points([Point(chunk_id, {}, {
        "record_kind": "source_chunk", "artifact_sha256": entry["sha256"],
        "artifact_path": "paper.md", "artifact_chunk_index": 0,
        "artifact_b64": "AA==",
    })]))
    with pytest.raises(ValueError, match="verification failed"):
        restore_file(shard, "paper.md")
    with pytest.raises(ValueError, match="existing archive differs|verification failed"):
        archive_file(shard, root, path)


def test_archive_zero_byte_and_symlink_refusal(shard, tmp_path: Path):
    root = tmp_path / "preprints"
    root.mkdir()
    empty = root / "empty.md"
    empty.write_bytes(b"")
    assert archive_file(shard, root, empty)["bytes"] == 0
    assert restore_file(shard, "empty.md") == b""
    link = root / "link.pdf"
    link.symlink_to(empty)
    with pytest.raises(ValueError, match="regular file"):
        archive_file(shard, root, link)


def test_archive_preprints_resumes_and_rejects_symlink(tmp_path: Path):
    root = tmp_path / "preprints"
    root.mkdir()
    (root / "a.md").write_text("one", encoding="utf-8")
    (root / "b.pdf").write_bytes(b"%PDF-1.4")
    target = tmp_path / "edge"
    db = EdgeShard.create(str(target), EdgeConfig(
        vectors={"text_rank0": EdgeVectorParams(1024, Distance.Cosine)}))
    db.close()
    assert archive_preprints(target, root) == (2, 11)
    assert archive_preprints(target, root) == (2, 11)
    (root / "bad.pdf").symlink_to(root / "b.pdf")
    with pytest.raises(ValueError, match="symlink"):
        archive_preprints(target, root)
    db = EdgeShard.load(str(target))
    try:
        assert db.retrieve([_manifest_id("a.md")], True, False)
    finally:
        db.close()


def test_indexed_sources_archives_vault_and_external_and_reports_missing(tmp_path: Path):
    vault = tmp_path / "vault"
    preprints = vault / "preprints"
    preprints.mkdir(parents=True)
    inside = vault / "Deep Research" / "paper.pdf"
    inside.parent.mkdir()
    inside.write_bytes(b"%PDF-in-vault")
    external = tmp_path / "staging" / "outside.md"
    external.parent.mkdir()
    external.write_text("external paper", encoding="utf-8")
    missing = vault / "gone.pdf"
    target = vault / "edge"
    db = EdgeShard.create(str(target), EdgeConfig(
        vectors={"text_rank0": EdgeVectorParams(1024, Distance.Cosine)}))
    db.update(UpdateOperation.upsert_points([
        Point(1, {}, {"source_path": str(inside)}),
        Point(2, {}, {"source_path": str(external)}),
        Point(3, {}, {"source_path": str(missing)}),
    ]))
    db.close()
    result = archive_indexed_sources(target, preprints)
    assert result == {"indexed_paths": 3, "additional_files": 2,
                      "additional_bytes": len(inside.read_bytes()) + len(external.read_bytes()),
                      "missing_local_paths": 1}
    db = EdgeShard.load(str(target))
    try:
        assert restore_file(db, "vault/Deep Research/paper.pdf") == inside.read_bytes()
        assert any("outside.md" in line for line in
                   (vault / "edge.migration.jsonl").read_text().splitlines())
    finally:
        db.close()


def test_incremental_sync_only_copies_missing_server_points(tmp_path: Path, monkeypatch):
    target = tmp_path / "edge"
    db = EdgeShard.create(str(target), EdgeConfig(
        vectors={"text_rank0": EdgeVectorParams(1024, Distance.Cosine)}))
    db.update(UpdateOperation.upsert_points([Point(10, {}, {"text": "old"})]))
    db.close()
    state = {10: "old", 20: "new"}

    def fake_json(url, body=None):
        if url.endswith("/points/scroll"):
            return {"result": {"points": [{"id": 10, "payload": {"text": state[10]}},
                                          {"id": 20, "payload": {"text": state[20]}}],
                               "next_page_offset": None}}
        assert url.endswith("/points")
        return {"result": [{"id": point_id, "vector": {}, "payload": {"text": state[point_id]}}
                           for point_id in body["ids"]]}

    monkeypatch.setattr(migration, "_json", fake_json)
    assert sync_server_updates(target, "http://localhost:6333", "research_library_v1") == 1
    assert sync_server_updates(target, "http://localhost:6333", "research_library_v1") == 0
    state[10] = "changed"
    assert sync_server_updates(target, "http://localhost:6333", "research_library_v1") == 1
    db = EdgeShard.load(str(target))
    try:
        assert {str(record.id) for record in db.retrieve([10, 20], True, False)} == {"10", "20"}
        assert db.retrieve([10], True, False)[0].payload["text"] == "changed"
    finally:
        db.close()


def test_missing_preprint_becomes_searchable_and_restorable(tmp_path: Path):
    root = tmp_path / "preprints"
    root.mkdir()
    paper = root / "new.md"
    paper.write_text("the novel neutron-flux calibration method\n" * 20, encoding="utf-8")
    target = tmp_path / "edge"
    db = EdgeShard.create(str(target), EdgeConfig(
        vectors={"text_rank0": EdgeVectorParams(1024, Distance.Cosine)}))
    db.update(UpdateOperation.create_field_index("text", TextIndexParams(scoring=True)))
    archive_file(db, root, paper)
    db.close()
    result = index_missing_preprints(target, root)
    assert result["indexed_preprints"] == 1 and result["text_chunks"] == 1
    assert index_missing_preprints(target, root)["indexed_preprints"] == 0
    from frontier.n3 import Rag
    rag = Rag(target)
    assert rag.search("neutron-flux calibration")[0]["source"] == str(paper)
    assert "neutron-flux" in rag.open_document(str(paper))[0]["text"]
    restored = tmp_path / "restored.md"
    assert restore_to_path(target, "new.md", restored) == len(paper.read_bytes())
    assert restored.read_bytes() == paper.read_bytes()
    with pytest.raises(FileExistsError):
        restore_to_path(target, "new.md", restored)


def test_legacy_hash_payload_copies_without_incompatible_vector(tmp_path: Path, monkeypatch):
    target = tmp_path / "edge"
    db = EdgeShard.create(str(target), EdgeConfig(
        vectors={"text_rank0": EdgeVectorParams(1024, Distance.Cosine)}))
    db.update(UpdateOperation.create_field_index("text", TextIndexParams(scoring=True)))
    db.close()
    payload = {"url": "https://example.test/old", "title": "Old paper", "text": "old research",
               "verified": True, "encoder": "hash384"}

    def fake_json(url, body=None):
        return {"result": {"points": [{"id": 10, "payload": payload}], "next_page_offset": None}}

    monkeypatch.setattr(migration, "_json", fake_json)
    assert copy_legacy_hash_sources(target, "http://localhost:6333") == 1
    assert copy_legacy_hash_sources(target, "http://localhost:6333") == 0
    db = EdgeShard.load(str(target))
    try:
        new_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "ple-edge-legacy:frontier_research_hash384_v1:10"))
        record = db.retrieve([new_id], True, True)[0]
        assert record.vector in ({}, None)
        assert record.payload["source_path"] == payload["url"]
        assert record.payload["legacy_point_id"] == "10"
        assert "encoder" not in record.payload
    finally:
        db.close()


def test_edge_only_verification_after_original_disappears(tmp_path: Path):
    root = tmp_path / "preprints"
    root.mkdir()
    paper = root / "paper.md"
    paper.write_text("hyperbolic geometry source", encoding="utf-8")
    target = tmp_path / "edge"
    db = EdgeShard.create(str(target), EdgeConfig(
        vectors={"text_rank0": EdgeVectorParams(1024, Distance.Cosine)}))
    db.update(UpdateOperation.create_field_index("text", TextIndexParams(scoring=True)))
    archive_file(db, root, paper)
    db.update(UpdateOperation.upsert_points([Point(1, {}, {"text": paper.read_text()})]))
    db.close()
    size = paper.stat().st_size
    paper.unlink()
    assert verify_edge_internal(target, 1, size, 0)["archive_bytes"] == size
    with pytest.raises(ValueError, match="manifest count"):
        verify_edge_internal(target, 2, size, 0)
