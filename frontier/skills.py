"""Skill loader: the harness reads the same global skills every other agent uses.

Skills are folders holding SKILL.md. Search order: FRONTIER_SKILLS_DIR, the harness's own
`skills/` folder, ~/.agents/skills (the shared home), ~/.claude/skills.
"""

from __future__ import annotations

import os
import pathlib
import re
from typing import Any

RESEARCH_SKILL = "deep-research"
MATH_SKILL = "math-olympiad-unified"
SKILL_DIRS = [pathlib.Path(p).expanduser() for p in filter(None, [
    os.environ.get("FRONTIER_SKILLS_DIR"),
    str(pathlib.Path(__file__).resolve().parents[1] / "skills"),
    "~/.agents/skills", "~/.claude/skills"])]

MATH_RULE = f"""Mathematics rule (mandatory): before any calculation, derivation, statistical argument or
numeric claim, call skill("{MATH_SKILL}") and follow it. In short: write the derivation before
code; compute with code, never in your head; compute the oracle or upper bound first; change one
variable per experiment; verify formulas against brute force on small cases; then try to falsify
your own result before you state it."""

TOOL: dict[str, Any] = {"type": "function", "function": {
    "name": "skill",
    "description": "Load a skill's instructions by name. Call with no name to list the skills.",
    "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}}}


def find(name: str) -> pathlib.Path | None:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name or ""):
        return None
    for folder in SKILL_DIRS:
        path = folder / name / "SKILL.md"
        if path.is_file():
            return path
    return None


def load(name: str, limit: int = 14000) -> str:
    """Returns the skill body (front matter removed), clipped for small context windows."""
    path = find(name)
    if path is None:
        raise KeyError(f"skill {name!r} not found in {[str(d) for d in SKILL_DIRS]}")
    body = re.sub(r"(?s)\A---\n.*?\n---\n", "", path.read_text())
    if len(body) > limit:
        body = body[:limit] + f"\n\n[skill clipped at {limit} characters; full text: {path}]"
    return body.strip()


def names() -> list[str]:
    found = set()
    for folder in SKILL_DIRS:
        if folder.is_dir():
            found.update(p.parent.name for p in folder.glob("*/SKILL.md"))
    return sorted(found)


def make_handler(used: list[str], limit: int = 14000):
    """Returns the `skill` tool handler; it records which skills a run loaded in `used`."""
    async def handler(args: dict[str, Any]) -> str:
        name = args.get("name")
        if not name:
            return "skills: " + ", ".join(names())
        try:
            text = load(name, limit)
        except KeyError as exc:
            return f"error: {exc.args[0]}"
        used.append(name)
        return text
    return handler
