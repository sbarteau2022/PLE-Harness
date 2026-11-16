"""Harness-side campaign minting into the nested Surreal store.

Every campaign run mints a small harness-side record (never model output
presented as model learning): one stage atom summarizing the closed run plus
one atom per deliverable, linked ``part_of``, and one trajectory over the
roster. ``verify_readback`` re-reads every ID and checks digests, so a
receipt's mint claim is always backed by a live store read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


def run_artifact_uri(campaign: str, harness: str, model: str, task: str, run_id: str) -> str:
    return f"ablation://{campaign}/{harness}/{model}/{task}/{run_id}"


@dataclass(frozen=True)
class MintResult:
    atoms: tuple[str, ...]
    trajectory: str
    admission: str


def mint_run(open_mind: Callable[[], Any], *, campaign: str, harness: str,
             model: str, task: str, run_id: str, summary: str,
             deliverables: list[dict[str, Any]]) -> MintResult:
    """Mint the harness-side record for one closed run and return its IDs."""
    mind = open_mind()
    uri = run_artifact_uri(campaign, harness, model, task, run_id)
    if hasattr(mind, "artifact"):
        mind.artifact = uri
    stage = mind.mint_atom(
        f"[ablation-harness-side:{harness}:{model}:{task}:{run_id}] {summary.strip()[:2000]}",
        "S_workflow", 0.5, f"ablation-harness-side:{run_id}")
    roster = [stage]
    for item in deliverables:
        atom = mind.mint_atom(
            f"[ablation-harness-side:{run_id}] deliverable {item.get('path')} "
            f"sha256={item.get('sha256')} bytes={item.get('bytes')}",
            "S_workflow", 0.5, f"ablation-harness-side:{run_id}:part")
        mind.relate(atom, stage, "part_of")
        if atom not in roster:
            roster.append(atom)
    trajectory = mind.mint_trajectory(roster, f"ablation-harness-side:{uri}")
    if not trajectory:
        raise ValueError("campaign mint requires at least two atoms for a trajectory")
    admission = mind.admission() if hasattr(mind, "admission") else ""
    return MintResult(atoms=tuple(roster), trajectory=trajectory, admission=admission)


def read_full_walk(trajectory_get: Callable[..., dict | None], trajectory_id: str) -> list[dict]:
    """Page a stored walk to its full ordered steps; raise when unreadable."""
    steps, start = [], 0
    while True:
        page = trajectory_get(trajectory_id, start, 16)
        if page is None:
            raise ValueError(f"minted trajectory {trajectory_id} is not readable")
        if str(page.get("id", trajectory_id)) != trajectory_id:
            raise ValueError("trajectory page identity mismatch")
        steps.extend(page.get("steps", []))
        if page.get("next_start") is None:
            return steps
        start = int(page["next_start"])


def verify_readback(atom_get: Callable[[str], dict | None],
                    trajectory_get: Callable[..., dict | None],
                    result: MintResult) -> dict[str, Any]:
    """Re-read every minted ID; return the evidence or raise on any mismatch."""
    atoms = []
    for atom_id in result.atoms:
        row = atom_get(atom_id)
        if row is None or str(row.get("id", atom_id)) != atom_id:
            raise ValueError(f"minted atom {atom_id} is not readable")
        atoms.append({"id": atom_id, "domain": row.get("domain"),
                      "content_digest": row.get("content_digest")})
    steps = read_full_walk(trajectory_get, result.trajectory)
    if [str(step.get("id")) for step in steps] != list(result.atoms):
        raise ValueError("trajectory roster does not match the minted atoms")
    return {"atoms": atoms, "trajectory": result.trajectory, "steps": len(steps)}
