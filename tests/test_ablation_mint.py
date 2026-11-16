from __future__ import annotations

import pytest

from ablations.mint import MintResult, mint_run, run_artifact_uri, verify_readback


class FakeMind:
    def __init__(self):
        self.artifact = ""
        self.atoms: list[str] = []
        self.edges: list[tuple[str, str, str]] = []

    def mint_atom(self, text, domain="general", salience=0.5, step=""):
        atom_id = f"atom:{len(self.atoms) + 1}"
        self.atoms.append(atom_id)
        return atom_id

    def relate(self, source, target, kind="part_of"):
        self.edges.append((source, kind, target))

    def mint_trajectory(self, atoms, receipt):
        return "distilled_trajectory:t1" if len(atoms) >= 2 else None

    def admission(self):
        return "distillation_admission:1"


def test_mint_labels_harness_side_and_links_parts() -> None:
    mind = FakeMind()
    result = mint_run(lambda: mind, campaign="c", harness="harness2", model="m",
                      task="t", run_id="r1", summary="closed, score 1",
                      deliverables=[{"path": "a", "sha256": "b" * 64, "bytes": 3}])
    assert result.atoms == ("atom:1", "atom:2")
    assert result.trajectory == "distilled_trajectory:t1"
    assert mind.artifact == run_artifact_uri("c", "harness2", "m", "t", "r1")
    assert ("atom:2", "part_of", "atom:1") in mind.edges


def test_mint_refuses_trajectory_without_two_atoms() -> None:
    mind = FakeMind()
    with pytest.raises(ValueError, match="at least two atoms"):
        mint_run(lambda: mind, campaign="c", harness="h", model="m",
                 task="t", run_id="r", summary="s", deliverables=[])


def _reads(roster):
    def atom_get(atom_id):
        return {"id": atom_id, "domain": "S_workflow", "content_digest": "d"} if atom_id in roster else None

    def trajectory_get(trajectory_id, start=0, limit=16):
        assert trajectory_id == "distilled_trajectory:t1"
        steps = [{"id": a} for a in roster[start:start + limit]]
        rest = start + len(steps)
        return {"id": trajectory_id, "steps": steps,
                "next_start": rest if rest < len(roster) else None}
    return atom_get, trajectory_get


def test_verify_readback_matches_roster_and_pages() -> None:
    roster = tuple(f"atom:{i}" for i in range(1, 20))
    result = MintResult(atoms=roster, trajectory="distilled_trajectory:t1", admission="a")
    atom_get, trajectory_get = _reads(list(roster))
    evidence = verify_readback(atom_get, trajectory_get, result)
    assert evidence["steps"] == 19


def test_verify_readback_rejects_missing_atom_and_short_walk() -> None:
    result = MintResult(atoms=("atom:1", "atom:2"), trajectory="distilled_trajectory:t1", admission="a")
    atom_get, trajectory_get = _reads(["atom:1"])
    with pytest.raises(ValueError, match="not readable"):
        verify_readback(atom_get, trajectory_get, result)
    atom_get, _ = _reads(["atom:1", "atom:2"])
    _, short_walk = _reads(["atom:1"])
    with pytest.raises(ValueError, match="roster"):
        verify_readback(atom_get, short_walk, result)
