"""The frontier pipeline: gated stages with owner approval, then a brigade of scoped subagents.

Order of work (nothing can be skipped):
  research -> audit -> design -> preregistration -> brigade cycles

Each stage leaves its result pending until the owner approves it (or `auto` is on).
The preregistration fixes the scope and the quality gates, including the target metric.

Brigade cycle (after the preregistration is approved):
  1. The orchestrator plans work orders: small objectives, each with its own allowed paths
     (inside the preregistered scope) and its own quality gates.
  2. Each order runs in its own workspace copy with a pinned AGENTS.md contract. No contract,
     or a contract whose hash changed, means the agent's work is void.
  3. The worker resubmits. Its gates run, a reviewer checks the diff against the contract,
     and only then is the change merged into the target.
  4. The preregistered gates run on the target. A merge that regresses beyond measured noise
     is rolled back. Every order leaves a receipt in results.json.
  5. Two cycles with nothing kept trigger a redesign: root cause, new approach, an amendment
     to DESIGN.md. Orders that already failed are not issued again.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import agent  # noqa: E402
from frontier import artifacts, core, n3, research, skills, surreal  # noqa: E402

STAGES = ["research", "audit", "design", "preregistration"]
DOCS = {"design": "DESIGN.md", "preregistration": "PREREGISTRATION.md"}
MIN_SOURCES = 3
MAX_ORDERS = 4
STRIKES_BEFORE_REDESIGN = 2
CONTRACT = "AGENTS.md"
MIN_EXTRACT_CHARS = 200   # shorter work is minted whole, not distilled

BASE_RULES = """Rules:
- Never weaken, delete or special-case a gate or a test. Never edit the goal or the preregistration.
- Never read grading files, task tests, answer keys or artifacts, and never hard-code answers.
- Commands run in fresh shells in the working folder; cd and export do not persist.
- Memory: deposit decisions, results and dead ends with memory_deposit (pin what must stay in
  view). Old tool output is moved to memory, not lost; read it back with memory_recall.
- Record what a later round must know with record_lesson (one sentence, concrete).
- The shared hyperbolic mind holds knowledge minted by earlier runs and other models. What it
  holds on your task is already in the user message. Search further with atom_search and
  atom_neighbors; open a named walk with trajectory_get. When you establish something durable
  and verified, mint it with mind_mint and connect steps with mind_mint_trajectory.
- If only the owner can unblock you, finish with a summary that starts with "BLOCKED:" and
  states the exact decision needed.

""" + skills.MATH_RULE

PROMPTS = {
    "research": """You are in the RESEARCH stage. Do not touch the folder. Follow the deep-research skill below.
Search the web and the preprint servers for the current state of the art that bears on the goal,
open and verify the primary sources, and ingest every source you verified with rag_ingest so later
stages can retrieve it. Check rag_search first: the local research library may already hold
relevant papers. Finish with the research notes the skill specifies, citing at least %d distinct
source URLs.

# deep-research skill
{skill}""" % MIN_SOURCES,
    "audit": """You are in the AUDIT stage. The research notes are in the user message. Review the target folder
read-only: what has been implemented and tried, the results so far, how success is measured, and
where it falls short of the goal in light of the research. Do not modify any file. Finish with
the audit: concrete findings with file paths, what has already been tried and how it scored, and
the gaps ranked by expected effect on the goal.""",
    "design": """You are in the DESIGN stage. Using the research notes and the audit, write the design to
`frontier-loop/DESIGN.md`: the way forward you think will produce new insight, each proposed change
and why it should move the goal's metric (cite the research), alternatives rejected, risks, and
the order of work. Write only that file. Finish with a summary.""",
    "preregistration": """You are in the PREREGISTRATION stage. Before any change is made, write
`frontier-loop/PREREGISTRATION.md`. After the owner approves it, it is locked. Include: hypotheses;
how each will be judged; what counts as success and as failure; what is out of scope; and two
fenced blocks.

The scope: the only paths later work may create or modify (relative, * wildcards allowed, as
narrow as the design allows). Include every implementation file explicitly named in the goal
and design. Do not copy paths from another task. Write one fenced `scope` JSON block with
an `allowed_paths` list containing the actual target-relative paths.

The quality gates: commands run in the target folder that decide success. Put the goal's target
here as a metric gate ("metric" is a regex capturing one number from the output, "min" is the
target). Use "repeats" (2 to 5) on a noisy metric so noise is measured. Mark a gate
"holdout": true to keep it for reporting only; it is never shown to the builders. Gate scripts
must already exist or be outside the scope, so builders cannot edit them:

```gates
[{"name": "tests", "cmd": "python -m pytest -q"},
 {"name": "auc", "cmd": "python evaluate.py", "metric": "AUC=([0-9.]+)", "min": 0.95, "repeats": 3}]
```

Write only that file. Finish with a summary.""",
    "plan": """You are the ORCHESTRATOR of a brigade. Research, audit, design and preregistration are in the user
message, with the current gate results, lessons and receipts of earlier work orders. Inspect the
folder read-only if you need to, then decompose the next step into at most %d small work orders
for subagents. Each order must be narrow enough for one focused agent, have its own allowed
paths (inside the preregistered scope) and its own quality gates that prove it did its job.
Do not re-issue an order that already failed; change the approach instead. Do not modify files.

Finish with a summary that contains exactly one fenced block:

```orders
[{"id": "short-id", "objective": "one precise objective",
  "allowed_paths": ["path/inside/scope.py"],
  "gates": [{"name": "unit", "cmd": "python -m pytest tests/test_x.py -q"}],
  "context": "the facts, file paths and research findings this agent needs, nothing more"}]
```""" % MAX_ORDERS,
    "worker": """You are a brigade subagent with one work order. Your contract is AGENTS.md at the top of your
working folder: read it first. It is your whole brief. Stay inside its allowed paths; work outside
them is discarded. Do not edit AGENTS.md. Run your gates yourself before you finish. Finish with a
summary for the orchestrator: what you changed, what you measured, and anything it should know.""",
    "review": """You are the REVIEWER. A subagent resubmitted work for one work order. Judge the diff against the
contract and the preregistration: does it meet the objective, stay honest (no weakened tests or
gates, no hard-coded answers, no leakage of evaluation data into training), and fit the design?
Finish with a summary that is exactly one JSON object: {"approve": true or false, "reason": "..."}""",
    "redesign": """The brigade is not making progress: recent work orders failed or were rolled back. Stop and
redesign. Using the receipts and lessons in the user message, state the root cause of the
failures, what was wrong with the approach (not just the execution), and a new plan of attack
that differs from what was tried. Research again if the failures show a gap in knowledge. Do not
modify files. If the preregistered scope or gates make the goal unreachable, finish with
"BLOCKED:" and the exact change the owner must approve. Otherwise finish with the redesign.""",
}

PROMPTS["extract"] = """Extract the durable knowledge from the work in the user message as atoms for a shared memory
that other models will read without this context. Each atom is one self-contained statement
(1 to 3 sentences) that names its subject explicitly, is supported by the text, and will still be
useful later: a verified finding with its source URL, a method and its measured result, a decision
and its reason, or a dead end and why it failed. Leave out instructions, chatter and anything
unverified. Finish with a summary that is exactly one JSON list of at most 8 objects:
[{"text": "...", "domain": "S_research or S_code or S_workflow or S_memory or general", "salience": 0.0 to 1.0}]"""

LESSON_TOOL = {"type": "function", "function": {
    "name": "record_lesson",
    "description": "Record one concrete lesson for later rounds (what was tried, what happened, what to do instead).",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}
FINISH = [t for t in agent.TOOLS if t["function"]["name"] == "finish"]
READ = [t for t in agent.TOOLS if t["function"]["name"] == "read_file"]
COMMON = n3.MEMORY_TOOLS + [n3.RAG_TOOLS[0], n3.RAG_TOOLS[2], skills.TOOL, LESSON_TOOL, artifacts.TOOL] + surreal.MIND_TOOLS
MIND_TOOL_NAMES = {t["function"]["name"] for t in surreal.MIND_TOOLS}
PHASE_TOOLS = {
    "research": research.TOOLS + n3.TOOLS + [skills.TOOL, LESSON_TOOL, artifacts.TOOL] + surreal.MIND_TOOLS + READ + FINISH,
    "audit": agent.TOOLS + COMMON,
    "design": agent.TOOLS + COMMON + research.TOOLS,
    "preregistration": agent.TOOLS + COMMON,
    "plan": agent.TOOLS + COMMON,
    "worker": agent.TOOLS + COMMON + research.TOOLS,
    "review": READ + FINISH,
    "extract": FINISH,
    "redesign": agent.TOOLS + COMMON + research.TOOLS + [n3.RAG_TOOLS[1]],
}


@dataclass
class Options:
    driver: agent.Config
    worker: agent.Config
    reviewer: agent.Config
    round_minutes: float = 30
    parallel: int = 1
    auto: bool = False


@dataclass
class Run:
    session: core.LocalSession
    summary: str
    skills_used: list
    ingested: list


def say(text: str) -> None:
    print(text, flush=True)


# ---------- running one agent ----------

async def run_agent(target: pathlib.Path, state: dict, phase: str, instruction: str, config: agent.Config,
                    minutes: float, name: str, root: pathlib.Path | None = None) -> Run:
    """Runs one agent with the phase's prompt and tools. Usage is added to the budget."""
    root = root or target
    mind = open_mind(target, state, config)
    if mind is not None and phase not in ("review", "extract"):
        try:  # the harness recalls for the model; no tool call is needed to benefit from the mind
            recalled = await asyncio.to_thread(surreal.mind_recall, mind_query(state, phase, instruction))
        except surreal.SurrealError:
            recalled = ""
        if recalled:
            instruction += "\n\n# From the shared hyperbolic mind (retrieved for you)\n" + core.clip(recalled, 5000)
    session = core.LocalSession(root, instruction, minutes)
    memory = surreal.open_memory(target, session=f"rev{state.get('rev', 0)}")
    used, ingested = [], []

    async def record_lesson(args):
        add_lesson(state, str(args.get("text", ""))[:500], phase)
        return "recorded"

    handlers = dict(research.HANDLERS)
    handlers.update(n3.make_handlers(memory, n3.Rag(), ingested, fetch=research.HANDLERS["web_fetch"]))
    handlers.update({"skill": skills.make_handler(used, limit=min(14000, config.compact_at_tokens)),
                     "record_lesson": record_lesson,
                     "artifact_render": artifacts.make_handler(session)})
    handlers.update(surreal.make_handlers(mind))
    tools = [t for t in PHASE_TOOLS[phase] if mind is not None or t["function"]["name"] not in MIND_TOOL_NAMES]
    prompt = PROMPTS[phase]
    if phase == "research":
        prompt = prompt.replace("{skill}", skills.load(skills.RESEARCH_SKILL))
        used.append(skills.RESEARCH_SKILL)
    try:
        summary = await agent.run(
            session, config, system_prompt=prompt + "\n\n" + BASE_RULES, tools=tools,
            handlers=handlers, memory=Carved(memory, CARVE_TOKENS, config.carve_high or None), mind=mind,
            transcript=core.loop_dir(target) / ".checkpoints" / f"{name}.json")
    except Exception as exc:  # a model or network failure ends this attempt, not the loop
        summary = f"attempt failed: {type(exc).__name__}: {exc}"
    core.add_usage(state, session.tokens, session.elapsed(), session.cache_read_tokens)
    if mind is not None and mind.minted:  # atoms the model minted itself join this goal's walk
        state.setdefault("mind", {}).setdefault("atoms", []).extend(mind.minted)
    if mind is not None and mind.minted_trajectories:
        walks = state.setdefault("mind", {}).setdefault("trajectories", [])
        walks.extend(w for w in mind.minted_trajectories if w not in walks)
    core.save(target, state)
    return Run(session, summary, used, ingested)


class Carved:
    """Archives old completed turns into memory once the prompt reaches a floor.

    The live prompt keeps the fixed instructions, bounded core view, one archive
    pointer, and recent complete tool-call groups. Individual outputs remain
    recallable through archive pages.

    `high` (FRONTIER_CARVE_HIGH, default = floor) sets hysteresis: nothing is touched until the
    prompt passes `high`, then it is carved down to `floor`, so between carves the prefix is
    identical for a provider's prompt cache. Measured on Nemotron Ultra 550B with a cold cache
    (reports/live-ab-nemotron-ultra-2026-11-16.md): carving every call above 8k sent 113k
    uncached prompt tokens against 155k append-only; hysteresis to 24k sent 187k against 186k.
    So the default carves every call; raise `high` for a provider whose cache rewards it.
    """

    def __init__(self, memory: Any, floor: int, high: int | None = None) -> None:
        self.memory, self.floor = memory, floor
        self.high = high if high is not None else int(os.environ.get("FRONTIER_CARVE_HIGH", str(floor)))
        self.carves = 0

    def manage(self, messages: list[dict[str, Any]], budget: int) -> None:
        if not self.floor:
            self.memory.manage(messages, budget)
            return
        if n3._estimate(messages) <= min(budget, self.high):
            return  # append-only between carves: the prompt prefix stays identical
        self.carves += 1
        self.memory.manage(messages, min(budget, self.floor))


CARVE_TOKENS = int(os.environ.get("FRONTIER_CARVE_TOKENS", "8000"))   # 0 turns carving off


# ---------- the shared mind ----------

def open_mind(target: pathlib.Path, state: dict, config: agent.Config | None = None,
              provider: str = "", model: str = "") -> surreal.Mind | None:
    if not surreal.mind_enabled():
        return None
    if config is not None:
        provider, model = re.sub(r"^https?://|/.*$", "", config.base_url or "local"), config.model
    return surreal.Mind(provider or "unknown", model or "unknown",
                        f"frontier-loop://{surreal.scope_for(target)}/rev{state.get('rev', 0)}")


def mind_query(state: dict, phase: str, instruction: str) -> str:
    if phase == "worker":
        found = re.search(r"## Objective\n(.+?)\n\n", instruction, re.S)
        return (found.group(1) if found else instruction)[:400]
    return state["goal"][:400]


def mint(target: pathlib.Path, state: dict, mind: surreal.Mind | None, label: str, text: str,
         domain: str = "S_workflow") -> None:
    """Mints accepted work into the shared mind. Only gate-backed results reach here: an approved
    stage, a kept work order, a redesign. A database failure never stops the loop."""
    if mind is None:
        return
    try:
        atom = mind.mint_atom(f"[frontier:{target.name}:{label}] {core.clip(text, 6000)}", domain, 0.5, label)
    except surreal.SurrealError as exc:
        say(f"  mind: could not mint {label}: {exc}")
        return
    record = state.setdefault("mind", {})
    if atom not in record.setdefault("atoms", []):
        record["atoms"].append(atom)
    say(f"  mind: minted {atom} ({label})")


async def extract_atoms(target: pathlib.Path, state: dict, config: agent.Config, label: str, text: str) -> list[dict]:
    """Asks the model in the seat to distil accepted work into discrete atoms. Returns candidates;
    nothing is minted here. An unusable answer yields no atoms, never an error."""
    if not surreal.mind_enabled() or len(text.strip()) < MIN_EXTRACT_CHARS:
        return []
    run = await run_agent(target, state, "extract", core.clip(text, 14000), config, 6,
                          f"extract-{label}-{len(state.get('attempts', []))}-{len(state.get('receipts', []))}")
    try:
        found = agent.repair_json(run.summary)
    except json.JSONDecodeError:
        return []
    atoms = []
    for item in found if isinstance(found, list) else []:
        if isinstance(item, dict) and isinstance(item.get("text"), str) and len(item["text"].strip()) >= 30:
            try:
                salience = min(max(float(item.get("salience", 0.5)), 0.0), 1.0)
            except (TypeError, ValueError):
                salience = 0.5
            atoms.append({"text": item["text"].strip()[:1500], "domain": str(item.get("domain", "general")),
                          "salience": salience})
    return atoms[:8]


def mint_parts(target: pathlib.Path, state: dict, mind: surreal.Mind | None, label: str, parts: list[dict]) -> None:
    """Mints extracted atoms and links each to the atom of the work it came from (`part_of`)."""
    record = state.get("mind", {})
    if mind is None or not parts or not record.get("atoms"):
        return
    parent, minted = record["atoms"][-1], 0
    for part in parts:
        try:
            atom = mind.mint_atom(part["text"], part["domain"], part["salience"], f"{label}:part")
            mind.relate(atom, parent, "part_of")
        except surreal.SurrealError as exc:
            say(f"  mind: skipped one extracted atom: {exc}")
            continue
        if atom not in record["atoms"]:
            record["atoms"].append(atom)
            minted += 1
    say(f"  mind: minted {minted} extracted atom(s) under {parent} ({label})")


def mint_walk(target: pathlib.Path, state: dict, mind: surreal.Mind | None, receipt: str) -> None:
    """Mints the goal's ordered atoms as a trajectory, so any model can replay the walk."""
    atoms = state.get("mind", {}).get("atoms", [])
    if mind is None or len(atoms) < 2:
        return
    try:
        walk = mind.mint_trajectory(atoms, receipt)
    except surreal.SurrealError as exc:
        say(f"  mind: could not mint the trajectory: {exc}")
        return
    if walk and walk not in state["mind"].setdefault("trajectories", []):
        state["mind"]["trajectories"].append(walk)
        say(f"  mind: minted {walk} over {len(atoms)} atoms")


def add_lesson(state: dict, text: str, source: str) -> None:
    if text and text not in [l["text"] for l in state.setdefault("lessons", [])]:
        state["lessons"].append({"text": text, "from": source, "time": core.now()})


def lessons_text(state: dict) -> str:
    items = state.get("lessons", [])[-15:]
    return "\n".join(f"- {l['text']}" for l in items) or "(none yet)"


def context(state: dict, target: pathlib.Path) -> str:
    """What stage agents and the orchestrator see: the goal plus the approved work so far."""
    parts = [f"# Goal\n{state['goal']}"]
    done = {s["stage"]: s for s in state.get("stages", [])}
    if "research" in done:
        parts.append("# Research notes\n" + core.clip(done["research"]["summary"], 12000))
    if "audit" in done:
        parts.append("# Audit\n" + core.clip(done["audit"]["summary"], 10000))
    if "design" in done:
        parts.append("# frontier-loop/DESIGN.md\n" + core.clip(core.read_doc(target, "DESIGN.md"), 10000))
    if "preregistration" in done:
        parts.append("# frontier-loop/PREREGISTRATION.md\n" + core.clip(core.read_doc(target, "PREREGISTRATION.md"), 8000))
    parts.append("# Lessons so far\n" + lessons_text(state))
    return "\n\n".join(parts)


def stop(target: pathlib.Path, state: dict, status: str, key: str, note: str) -> str:
    state["status"] = status
    state[key] = note
    if status in ("complete", "capped"):
        mind = open_mind(target, state, provider="harness", model="frontier-loop")
        mint(target, state, mind, status, f"Goal: {state['goal'][:1500]}\nOutcome: {status}. {note}")
        mint_walk(target, state, mind, f"{status}: {note}")
    core.save(target, state)
    say(f"{status}: {note}")
    return status


# ---------- stages ----------

def owner_gates(state: dict) -> list[dict]:
    try:
        return core.parse_block(state["goal"], "gates") or []
    except json.JSONDecodeError:
        return []


def scope_matches_folder(paths: list[str], target: pathlib.Path) -> str:
    """At least one allowed path must cover a file that exists in the folder. New files and new
    folders may be listed too, but a scope where nothing is real (one copied from an example)
    is refused."""
    files = list(core.tree_hashes(target))
    if any(core.in_scope(f, [path]) for path in paths for f in files):
        return ""
    return (f"none of {paths} matches a file in this folder; name the real files the design "
            "changes (new files may be listed alongside them)")


def read_preregistration(state: dict, target: pathlib.Path) -> tuple[list[str], list[dict], str]:
    """Returns (scope, gates, problem). The owner's own gates always apply and win by name."""
    text = core.read_doc(target, "PREREGISTRATION.md")
    try:
        scope = core.parse_block(text, "scope")
        gates = core.parse_block(text, "gates") or []
    except json.JSONDecodeError as exc:
        return [], [], f"a fenced block is not valid JSON: {exc}"
    paths = scope.get("allowed_paths") if isinstance(scope, dict) else None
    problem = core.validate_paths(paths)
    if problem:
        return [], [], "scope: " + problem
    mine = owner_gates(state)
    merged = mine + [g for g in gates if isinstance(g, dict) and g.get("name") not in {m["name"] for m in mine}]
    problem = core.validate_gates(merged)
    if problem:
        return [], [], "gates: " + problem
    if not any(not g.get("holdout") for g in merged):
        return [], [], "gates: at least one gate must not be a holdout"
    for gate in merged:
        for token in re.findall(r"[\w./\-]+", gate["cmd"]):
            if (target / token).is_file() and core.in_scope(token, paths):
                return [], [], f"gate {gate['name']} runs {token}, which is inside the scope; builders could edit it"
    # A goal that names its implementation file must not be locked to an
    # unrelated example scope. This is a narrow check; it does not infer paths
    # from tests, output files, or arbitrary mentions elsewhere in the goal.
    goal_paths = re.findall(
        r"\b(?:in|improving|fixing)\s+[`\"']?([\w./-]+\.(?:py|rs|c|cpp|h|js|ts|tsx|md|txt))\b",
        state["goal"], re.IGNORECASE,
    )
    for path in goal_paths:
        if (target / path).is_file() and not core.in_scope(path, paths):
            return [], [], f"scope: goal implementation file {path} is outside allowed_paths {paths}"
    problem = scope_matches_folder(paths, target)
    if problem:
        return [], [], "scope: " + problem
    return paths, merged, ""


def check_stage(stage: str, state: dict, target: pathlib.Path, run: Run, changed: list[str]) -> str:
    """Returns "" when the stage met its exit condition, else the reason it did not."""
    if run.summary.startswith("attempt failed"):
        return run.summary
    if stage == "research":
        urls = set(re.findall(r"https?://[^\s)\]>\"']+", run.summary))
        if not any(run.session.tool_calls.count(t) for t in research.HANDLERS):
            return "no web or preprint tool was used"
        if len(urls) < MIN_SOURCES:
            return f"the notes cite {len(urls)} source URLs; at least {MIN_SOURCES} are required"
        if not run.ingested and "rag_ingest" not in run.session.tool_calls:
            return "no verified source was ingested with rag_ingest"
    if stage in ("research", "audit") and changed:
        return f"this stage is read-only but files changed: {changed[:5]}"
    if stage in DOCS:
        if changed:
            return f"only frontier-loop/{DOCS[stage]} may be written in this stage; also changed: {changed[:5]}"
        if len(core.read_doc(target, DOCS[stage]).strip()) < 200:
            return f"frontier-loop/{DOCS[stage]} is missing or too short"
        if skills.MATH_SKILL not in run.skills_used:
            return f'the math skill was not loaded; call skill("{skills.MATH_SKILL}") and apply it to every number'
    if stage == "preregistration":
        return read_preregistration(state, target)[2]
    return ""


async def run_stage(stage: str, state: dict, target: pathlib.Path, opts: Options, feedback: str = "") -> str:
    """Runs one stage (three attempts). Leaves the result pending approval. Returns the loop status."""
    note = f"\n\nThe owner asked for this revision: {feedback}" if feedback else ""
    problem = ""
    for attempt in range(1, 4):
        if core.over_budget(state):
            return stop(target, state, "capped", "capped_reason", core.over_budget(state))
        say(f"\n=== stage: {stage} (attempt {attempt}) ===")
        snapshot = core.Snapshot(target)
        instruction = context(state, target) + f"\n\nStage now: {stage}." + note
        run = await run_agent(target, state, stage, instruction, opts.driver, opts.round_minutes,
                              f"stage-{stage}-r{len(state.get('attempts', []))}")
        changed = snapshot.changed()
        blocked = run.summary.startswith("BLOCKED:")
        problem = "" if blocked else check_stage(stage, state, target, run, changed)
        if blocked or problem:
            snapshot.restore()
        snapshot.close()
        state.setdefault("attempts", []).append({
            "stage": stage, "attempt": attempt, "ok": not problem and not blocked, "problem": problem,
            "time": core.now(), "tokens": run.session.tokens, "summary": run.summary})
        if blocked:
            return stop(target, state, "blocked", "blocked_reason", run.summary[8:].strip())
        if not problem:
            body = core.read_doc(target, DOCS[stage]) if stage in DOCS else run.summary
            extracted = await extract_atoms(target, state, opts.driver, stage, body)
            state["pending"] = {"stage": stage, "summary": run.summary, "doc": DOCS.get(stage), "atoms": extracted,
                                "model": opts.driver.model, "time": core.now(),
                                "provider": re.sub(r"^https?://|/.*$", "", opts.driver.base_url or "local"),
                                "sources": run.ingested, "skills": sorted(set(run.skills_used))}
            core.save(target, state)
            say(f"stage {stage}: ready for approval")
            return approve(target, state) if opts.auto else "pending"
        core.save(target, state)
        say(f"stage {stage}: not accepted: {problem}")
        note = f"\n\nYour previous attempt at this stage was not accepted: {problem}. Fix that."
    return stop(target, state, "blocked", "blocked_reason", f"stage {stage} failed three attempts: {problem}")


def approve(target: pathlib.Path, state: dict) -> str:
    """Owner approval of the pending stage. The preregistration is validated and locked here."""
    pending = state.get("pending")
    if not pending:
        return "nothing is pending"
    entry = {k: pending[k] for k in ("stage", "summary", "model", "sources", "skills")}
    entry["approved"] = core.now()
    if pending["stage"] == "preregistration":
        scope, gates, problem = read_preregistration(state, target)
        if problem:
            return f"cannot approve: {problem}"
        text = core.read_doc(target, "PREREGISTRATION.md")
        entry.update(scope=scope, gates=gates, sha1=hashlib.sha1(text.encode()).hexdigest())
    state.setdefault("stages", []).append(entry)
    state.pop("pending")
    mind = open_mind(target, state, provider=pending.get("provider", ""), model=pending.get("model", ""))
    body = core.read_doc(target, pending["doc"]) if pending.get("doc") else entry["summary"]
    mint(target, state, mind, entry["stage"], body, "S_research" if entry["stage"] == "research" else "S_workflow")
    mint_parts(target, state, mind, entry["stage"], pending.get("atoms") or [])
    if entry["stage"] == "preregistration":
        mint_walk(target, state, mind, f"stages approved for rev {state.get('rev', 0)}")
    core.save(target, state)
    say(f"stage {entry['stage']}: approved")
    return "approved"


# ---------- brigade ----------

def signature(order: dict) -> str:
    text = re.sub(r"\W+", " ", order["objective"].lower()).strip() + "|" + ",".join(sorted(order["allowed_paths"]))
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def extract_orders(summary: str):
    """Finds the work orders in a plan, tolerating the ways models format them: a block tagged
    `orders`, a `json` block or bare JSON list of order objects, or one object for a single order."""
    found = core.parse_block(summary, "orders")
    if found:
        return found
    candidates = re.findall(r"```(?:json)?\s*\n(.*?)```", summary, re.S)
    start, end = summary.find("["), summary.rfind("]")
    if 0 <= start < end:
        candidates.append(summary[start:end + 1])
    for text in candidates:
        try:
            data = agent.repair_json(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("orders"), list):
            data = data["orders"]
        if isinstance(data, dict) and data.get("objective"):
            data = [data]
        if isinstance(data, list) and data and all(isinstance(o, dict) and o.get("objective") for o in data):
            return data
    return found


def validate_plan(orders, scope: list[str], failed: set[str], parallel: int) -> str:
    if not isinstance(orders, list) or not orders:
        return "the orders block must be a non-empty JSON list"
    if len(orders) > MAX_ORDERS:
        return f"at most {MAX_ORDERS} orders per cycle"
    seen, taken = set(), []
    for order in orders:
        if not isinstance(order, dict) or not all(order.get(k) for k in ("id", "objective", "allowed_paths", "gates")):
            return "each order needs id, objective, allowed_paths and gates"
        if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", str(order["id"])) or order["id"] in seen:
            return f"order ids must be unique short names: {order['id']!r}"
        seen.add(order["id"])
        problem = core.validate_paths(order["allowed_paths"]) or core.validate_gates(order["gates"])
        if problem:
            return f"order {order['id']}: {problem}"
        for path in order["allowed_paths"]:
            if not (path in scope or core.in_scope(path, scope)):
                return f"order {order['id']}: {path} is outside the preregistered scope {scope}"
            if parallel > 1 and any(core.in_scope(path, [t]) or core.in_scope(t, [path]) for t in taken):
                return f"order {order['id']}: {path} overlaps another order; parallel orders need disjoint paths"
        taken += order["allowed_paths"]
        if signature(order) in failed:
            return f"order {order['id']} repeats an order that already failed; change the approach"
    return ""


def contract_text(order: dict, state: dict, minutes: float, steps: int) -> str:
    gates = "\n".join(f"- {g['name']}: `{g['cmd']}`" + (f" (metric {g['metric']} >= {g['min']})" if g.get("metric") else "")
                      for g in order["gates"])
    return f"""# AGENTS.md: work order {order['id']} (cycle {state['round']})

This file is your contract. It is pinned and hash-checked: if it is edited or missing, your work is void.

## Objective
{order['objective']}

## Allowed paths (everything else is read-only)
{chr(10).join('- ' + p for p in order['allowed_paths'])}

## Acceptance gates (all must pass in this folder before you finish)
{gates}

## Context from the orchestrator
{order.get('context', '(none)')}

## Budget
{steps} steps, {minutes:g} minutes.

## Invariants
{BASE_RULES}

## Refusal conditions
If the objective cannot be met inside the allowed paths, or needs a decision only the owner can
make, do not widen the scope: finish with "BLOCKED: <what is needed>".
"""


async def run_order(order: dict, state: dict, target: pathlib.Path, opts: Options) -> dict:
    """One subagent, one workspace, one contract. Returns the receipt (not yet merged)."""
    workspace = core.loop_dir(target) / ".work" / f"c{state['round']}-{order['id']}"
    resume = (core.loop_dir(target) / ".checkpoints" / f"order-c{state['round']}-{order['id']}.json").exists()
    if not (resume and workspace.exists()):
        shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True)
        core.copy_tree(target, workspace)
    base = core.tree_hashes(target)
    own = (target / CONTRACT).read_text() if (target / CONTRACT).exists() else ""
    contract = contract_text(order, state, opts.round_minutes, opts.worker.max_steps)
    (workspace / CONTRACT).write_text(contract + ("\n\n---\n\n" + own if own else ""))
    pinned = hashlib.sha1((workspace / CONTRACT).read_bytes()).hexdigest()
    receipt = {"order": order["id"], "cycle": state["round"], "objective": order["objective"],
               "allowed_paths": order["allowed_paths"], "signature": signature(order),
               "model": opts.worker.model, "contract_sha1": pinned, "time": core.now()}
    say(f"  order {order['id']}: subagent started ({opts.worker.model})")
    run = await run_agent(target, state, "worker", contract, opts.worker, opts.round_minutes,
                          f"order-c{state['round']}-{order['id']}", root=workspace)
    path = workspace / CONTRACT
    intact = path.exists() and hashlib.sha1(path.read_bytes()).hexdigest() == pinned
    changed = [p for p in core.diff_paths(base, core.tree_hashes(workspace)) if p != CONTRACT]
    outside = [p for p in changed if not core.in_scope(p, order["allowed_paths"])]
    receipt.update(summary=run.summary, tokens=run.session.tokens, changed=changed, out_of_scope=outside,
                   skills=sorted(set(run.skills_used)))
    if run.summary.startswith("BLOCKED:"):
        receipt["verdict"] = "blocked: " + run.summary[8:].strip()
    elif run.summary.startswith("attempt failed"):
        receipt["verdict"] = "failed: " + run.summary
    elif not intact:
        receipt["verdict"] = "void: the AGENTS.md contract was changed or removed"
    elif outside:
        receipt["verdict"] = f"void: changed paths outside the order: {outside[:5]}"
    elif not changed:
        receipt["verdict"] = "failed: no change submitted"
    else:
        results = await asyncio.to_thread(core.run_gates, order["gates"], workspace)
        receipt["order_gates"] = core.slim(results)
        failing = [r for r in results if not r["passed"]]
        receipt["verdict"] = ("failed: order gates: " + core.gate_report(failing)[:600]) if failing else "submitted"
    receipt["workspace"] = str(workspace)
    return receipt


async def review(receipt: dict, order: dict, state: dict, target: pathlib.Path, opts: Options) -> tuple[bool, str]:
    workspace = pathlib.Path(receipt["workspace"])
    diff = core.text_diff(target, workspace, receipt["changed"])
    instruction = (f"# Contract\n{(workspace / CONTRACT).read_text()[:5000]}\n\n"
                   f"# Preregistration\n{core.clip(core.read_doc(target, 'PREREGISTRATION.md'), 5000)}\n\n"
                   f"# Subagent summary\n{receipt['summary'][:2000]}\n\n# Diff\n{diff}")
    for attempt in range(2):
        run = await run_agent(target, state, "review", instruction, opts.reviewer, min(opts.round_minutes, 10),
                              f"review-c{state['round']}-{order['id']}-{attempt}", root=workspace)
        try:
            verdict = agent.repair_json(run.summary)
            if isinstance(verdict, dict) and isinstance(verdict.get("approve"), bool):
                return verdict["approve"], str(verdict.get("reason", ""))[:600]
        except json.JSONDecodeError:
            pass
    return True, "the reviewer gave no usable verdict; accepted on gates alone"


def merge(receipt: dict, target: pathlib.Path) -> core.Snapshot:
    snapshot = core.Snapshot(target)
    workspace = pathlib.Path(receipt["workspace"])
    for rel in receipt["changed"]:
        source, dest = workspace / rel, target / rel
        if source.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
        else:
            dest.unlink(missing_ok=True)
    return snapshot


async def make_plan(state: dict, target: pathlib.Path, opts: Options, gates_now: list[dict],
                    scope: list[str]) -> list[dict] | None:
    failed = set(state.get("failed_signatures", []))
    recent = "\n".join(f"- cycle {r['cycle']} order {r['order']} ({r['verdict'][:120]}): {r['objective'][:200]}"
                       for r in state.get("receipts", [])[-10:]) or "(none yet)"
    redesign = state.get("redesigns", [])[-1]["text"] if state.get("redesigns") else ""
    base = (context(state, target) + f"\n\n# Scope (allowed paths)\n{json.dumps(scope)}\n\n"
            f"# Gate results now\n{core.gate_report(gates_now, state.get('noise'))}\n\n"
            f"# Receipts of earlier work orders\n{recent}\n\n"
            + (f"# Latest redesign (follow it)\n{core.clip(redesign, 5000)}\n\n" if redesign else "")
            + f"This is brigade cycle {state['round']} of {state['max_rounds']}. Plan the work orders.")
    note = ""
    for attempt in range(1, 4):
        snapshot = core.Snapshot(target)
        run = await run_agent(target, state, "plan", base + note, opts.driver, opts.round_minutes,
                              f"plan-c{state['round']}-{attempt}")
        if snapshot.changed():
            snapshot.restore()
        snapshot.close()
        if run.summary.startswith("BLOCKED:"):
            stop(target, state, "blocked", "blocked_reason", run.summary[8:].strip())
            return None
        try:
            orders = extract_orders(run.summary)
            problem = validate_plan(orders, scope, failed, opts.parallel)
        except json.JSONDecodeError as exc:
            orders, problem = None, f"the orders block is not valid JSON: {exc}"
        if problem and "non-empty JSON list" in problem:
            problem += (". Your summary must end with a fenced block tagged orders that holds a JSON list "
                        "of order objects, for example ```orders\n[{...}]\n```")
        state.setdefault("plan_attempts", []).append(
            {"cycle": state["round"], "attempt": attempt, "problem": problem, "summary": run.summary[-3000:]})
        if not problem:
            return orders
        say(f"  plan not accepted: {problem}")
        note = f"\n\nYour previous plan was not accepted: {problem}. Fix that."
    stop(target, state, "blocked", "blocked_reason", f"the orchestrator could not produce a valid plan: {problem}")
    return None


async def redesign(state: dict, target: pathlib.Path, opts: Options) -> str:
    say("\n=== redesign: no progress in recent cycles ===")
    failures = "\n\n".join(
        f"cycle {r['cycle']} order {r['order']}: {r['objective']}\nverdict: {r['verdict']}\nsummary: {r['summary'][:800]}"
        for r in state.get("receipts", [])[-8:] if r["verdict"] != "kept")
    snapshot = core.Snapshot(target)
    run = await run_agent(target, state, "redesign", context(state, target) + f"\n\n# Recent failures\n{failures}",
                          opts.driver, opts.round_minutes, f"redesign-{len(state.get('redesigns', []))}")
    if snapshot.changed():
        snapshot.restore()
    snapshot.close()
    if run.summary.startswith("BLOCKED:"):
        return stop(target, state, "blocked", "blocked_reason", run.summary[8:].strip())
    number = len(state.setdefault("redesigns", [])) + 1
    state["redesigns"].append({"number": number, "time": core.now(), "after_cycle": state["round"], "text": run.summary})
    with open(core.loop_dir(target) / "DESIGN.md", "a") as handle:
        handle.write(f"\n\n## Amendment {number} ({core.now()}): redesign after cycle {state['round']}\n\n{run.summary}\n")
    state["strikes"] = 0
    mint(target, state, open_mind(target, state, opts.driver), f"redesign-{number}", run.summary)
    core.save(target, state)
    return "active"


async def cycle(state: dict, target: pathlib.Path, opts: Options) -> str:
    """One brigade cycle. Returns the loop status afterwards."""
    prereg = next(s for s in state["stages"] if s["stage"] == "preregistration")
    text = core.read_doc(target, "PREREGISTRATION.md")
    if hashlib.sha1(text.encode()).hexdigest() != prereg["sha1"]:
        return stop(target, state, "blocked", "blocked_reason", "PREREGISTRATION.md changed after it was approved")
    scope, gates = prereg["scope"], prereg["gates"]
    if "baseline" not in state:
        say("\nmeasuring the baseline gates and their noise")
        state["baseline"] = core.slim(await asyncio.to_thread(core.run_gates, gates, target))
        state["noise"] = await asyncio.to_thread(core.measure_noise, gates, target)
        core.save(target, state)
    before = await asyncio.to_thread(core.run_gates, gates, target)
    if core.all_pass(before):
        return stop(target, state, "complete", "complete_note",
                    "every preregistered gate passes: " + ", ".join(r["name"] for r in before if not r["holdout"]))
    if state["round"] >= state["max_rounds"]:
        return stop(target, state, "capped", "capped_reason", "cycle limit reached")
    if core.over_budget(state):
        return stop(target, state, "capped", "capped_reason", core.over_budget(state))
    if state.get("strikes", 0) >= STRIKES_BEFORE_REDESIGN:
        if await redesign(state, target, opts) != "active":
            return state["status"]

    active = state.get("active")
    if not active:
        state["round"] += 1
        say(f"\n=== brigade cycle {state['round']} of {state['max_rounds']} ===\n{core.gate_report(before, state.get('noise'))}")
        orders = await make_plan(state, target, opts, before, scope)
        if orders is None:
            return state["status"]
        active = state["active"] = {"cycle": state["round"], "orders": orders, "done": []}
        core.save(target, state)
    else:
        say(f"\n=== resuming brigade cycle {state['round']} ===")
    todo = [o for o in active["orders"] if o["id"] not in active["done"]]
    kept = 0
    for start in range(0, len(todo), max(1, opts.parallel)):
        batch = todo[start:start + max(1, opts.parallel)]
        receipts = await asyncio.gather(*(run_order(order, state, target, opts) for order in batch))
        for order, receipt in zip(batch, receipts):
            if receipt["verdict"] == "submitted":
                approved, reason = await review(receipt, order, state, target, opts)
                receipt["review"] = {"approve": approved, "reason": reason, "model": opts.reviewer.model}
                if not approved:
                    receipt["verdict"] = "rejected by the reviewer: " + reason
                else:
                    snapshot = merge(receipt, target)
                    after = await asyncio.to_thread(core.run_gates, gates, target)
                    after = await asyncio.to_thread(core.confirm_passes, gates, before, after, target)
                    reason = core.regressed(before, after, state.get("noise", {}))
                    receipt["gates"] = core.slim(after)
                    if reason:
                        snapshot.restore()
                        receipt["verdict"] = "rolled back: " + reason
                    else:
                        receipt["verdict"], before, kept = "kept", after, kept + 1
                        mint(target, state, open_mind(target, state, opts.worker), f"order-{order['id']}",
                             f"{order['objective']}\nResult: {receipt['summary']}\nGates: "
                             + ", ".join(f"{g['name']}={g['value'] if g['value'] is not None else g['passed']}"
                                         for g in receipt["gates"] if not g["holdout"]), "S_code")
                        mint_parts(target, state, open_mind(target, state, opts.worker), f"order-{order['id']}",
                                   await extract_atoms(target, state, opts.worker, f"order-{order['id']}",
                                                       f"{order['objective']}\n{receipt['summary']}"))
                    snapshot.close()
            if receipt["verdict"] != "kept":
                state.setdefault("failed_signatures", []).append(receipt["signature"])
                add_lesson(state, f"Order '{order['objective'][:140]}' ended as: {receipt['verdict'][:200]}", "brigade")
            shutil.rmtree(receipt.pop("workspace"), ignore_errors=True)
            state.setdefault("receipts", []).append(receipt)
            active["done"].append(order["id"])
            core.save(target, state)
            say(f"  order {order['id']}: {receipt['verdict'][:160]}")
    state["strikes"] = 0 if kept else state.get("strikes", 0) + 1
    state.setdefault("cycles", []).append({
        "cycle": state["round"], "time": core.now(), "orders": [o["id"] for o in active["orders"]],
        "kept": kept, "gates": core.slim(before)})
    state.pop("active")
    core.save(target, state)
    if core.all_pass(before):
        return stop(target, state, "complete", "complete_note",
                    "every preregistered gate passes: " + ", ".join(r["name"] for r in before if not r["holdout"]))
    return "active"


# ---------- driving ----------

def next_stage(state: dict) -> str | None:
    done = {s["stage"] for s in state.get("stages", [])}
    return next((s for s in STAGES if s not in done), None)


async def step(target: pathlib.Path, opts: Options, feedback: str = "") -> str:
    """Does the next piece of work: one stage (then waits for approval) or one brigade cycle."""
    state = core.require(target)
    if state["status"] != "active":
        return state["status"]
    if state.get("pending") and not feedback:
        return "pending"
    if feedback:
        stage = (state.pop("pending", None) or {}).get("stage") or next_stage(state)
        return await run_stage(stage, state, target, opts, feedback) if stage else "active"
    stage = next_stage(state)
    if stage:
        return await run_stage(stage, state, target, opts)
    return await cycle(state, target, opts)


async def run(target: pathlib.Path, opts: Options) -> str:
    """Runs until something needs the owner (approval, block), the goal is met, or a cap is hit."""
    while True:
        status = await step(target, opts)
        if status not in ("active", "approved"):
            return status


def report(target: pathlib.Path) -> str:
    state = core.require(target)
    budget = state.get("budget", {})
    lines = [f"# Frontier report: {target.name}", "",
             f"Status: **{state['status']}** " + (state.get("blocked_reason") or state.get("complete_note") or state.get("capped_reason") or ""),
             f"Cycles: {state.get('round', 0)} of {state.get('max_rounds')}. Tokens in/out: "
             f"{budget.get('tokens_in', 0)}/{budget.get('tokens_out', 0)}. Agent time: {budget.get('agent_seconds', 0) / 3600:.2f} h.", "",
             "## Stages"]
    lines += [f"- {s['stage']}: approved {s.get('approved', '')} ({s.get('model', '')})" for s in state.get("stages", [])]
    if state.get("pending"):
        lines.append(f"- {state['pending']['stage']}: **waiting for your approval**")
    lines += ["", "## Gates (baseline -> latest)", "", "| Gate | Target | Baseline | Latest | Passing | Holdout |", "|---|---|---|---|---|---|"]
    latest = (state.get("cycles") or [{}])[-1].get("gates") or state.get("baseline") or []
    base = {g["name"]: g for g in state.get("baseline", [])}
    for gate in latest:
        b = base.get(gate["name"], {})
        lines.append(f"| {gate['name']} | {gate.get('min', '')} | {b.get('value', b.get('passed', ''))} | "
                     f"{gate.get('value', gate['passed'])} | {'yes' if gate['passed'] else 'no'} | {'yes' if gate['holdout'] else ''} |")
    lines += ["", "## Work orders"]
    lines += [f"- cycle {r['cycle']} `{r['order']}`: {r['verdict'][:140]}. {r['objective'][:160]}" for r in state.get("receipts", [])] or ["(none yet)"]
    if state.get("redesigns"):
        lines += ["", "## Redesigns"] + [f"- after cycle {r['after_cycle']}: {r['text'][:300]}" for r in state["redesigns"]]
    lines += ["", "## Lessons"] + ([f"- {l['text']}" for l in state.get("lessons", [])[-15:]] or ["(none yet)"])
    return "\n".join(lines)
