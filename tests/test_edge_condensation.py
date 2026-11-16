"""Condensation gates with a real temporary Edge shard and reversible files."""

from pathlib import Path

import pytest
from qdrant_edge import (Distance, EdgeConfig, EdgeShard, EdgeVectorParams, PayloadSchemaType,
                         Point, TextIndexParams, UpdateOperation)

from scripts.condense_research_library import (
    commit_condensation,
    plan_condensation,
    rollback_condensation,
    smoke_staged,
    stage_condensation,
)
from scripts.migrate_research_library_to_edge import archive_file, restore_file


def _fixture(tmp_path: Path):
    vault = tmp_path / "vault"
    preprints = vault / "preprints"
    preprints.mkdir(parents=True)
    paper = preprints / "paper.md"
    paper.write_text("neutronflux calibration paper", encoding="utf-8")
    other = vault / "other.md"
    other.write_text("supporting notes", encoding="utf-8")
    target = vault / "edge"
    shard = EdgeShard.create(str(target), EdgeConfig(
        vectors={"text_rank0": EdgeVectorParams(1024, Distance.Cosine)}))
    shard.update(UpdateOperation.create_field_index("text", TextIndexParams(scoring=True)))
    shard.update(UpdateOperation.create_field_index("source_path", PayloadSchemaType.Keyword))
    archive_file(shard, preprints, paper)
    archive_file(shard, vault, other, key="vault/other.md")
    shard.update(UpdateOperation.upsert_points([
        Point(1, {}, {"source_path": str(paper), "document_id": "paper", "chunk_index": 0,
                      "text": "neutronflux calibration paper"}),
        Point(2, {}, {"source_path": str(other), "document_id": "other", "chunk_index": 0,
                      "text": "supporting notes"}),
    ]))
    shard.close()
    return preprints, target, paper, other, vault / ".quarantine"


def test_stage_smoke_rollback_and_commit(tmp_path: Path):
    preprints, target, paper, other, quarantine = _fixture(tmp_path)
    total = len(paper.read_bytes()) + len(other.read_bytes())
    with pytest.raises(ValueError, match="file count changed"):
        plan_condensation(target, preprints, quarantine, 3, total)
    assert not quarantine.exists()
    assert plan_condensation(target, preprints, quarantine, 2, total)["files"] == 2
    assert stage_condensation(quarantine)["staged_files"] == 2
    assert not paper.exists() and not other.exists()
    assert smoke_staged(quarantine, "neutronflux")["verified_staged_files"] == 2
    assert rollback_condensation(quarantine) == 2
    assert paper.exists() and other.exists()

    plan_condensation(target, preprints, quarantine, 2, total)
    stage_condensation(quarantine)
    receipt = commit_condensation(quarantine, "neutronflux")
    assert receipt["removed_files"] == 2
    assert not quarantine.exists() and not paper.exists() and not other.exists()
    shard = EdgeShard.load(str(target))
    try:
        assert restore_file(shard, "paper.md") == b"neutronflux calibration paper"
        assert restore_file(shard, "vault/other.md") == b"supporting notes"
    finally:
        shard.close()
