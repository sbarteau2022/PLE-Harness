#!/usr/bin/env python3
"""Frontier loop CLI: point a model at a folder with a goal and let it improve the work.

Any OpenAI-compatible model can drive (local MLX server or hosted API). No git, no Docker.
Everything is saved in `<target>/frontier-loop/`: results.json (the single results file),
DESIGN.md, PREREGISTRATION.md, and n3.sqlite (the memory store).

  set <target> --goal "text" | --goal-file FILE [--max-rounds N] [--max-tokens N] [--max-hours H]
  step <target> [seats]         do the next stage (then wait for approval) or one brigade cycle
  approve <target>              approve the pending stage
  revise <target> --feedback T  redo the pending stage with your feedback
  run <target> [seats] [--auto] keep going until approval is needed, blocked, capped or complete
  show | report | gates | pause | resume | clear <target>
  block <target> --reason T     complete <target> --note T

Seats: --driver, --worker, --reviewer take a profile name (~/.config/frontier/profiles.json)
or "<base_url>|<model>". --base-url and --model set the driver directly. Worker and reviewer
default to the driver. For an interactive session use frontier/chat.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from frontier import core, drivers, pipeline  # noqa: E402

DEFAULT_CAP = 20
RESUME_ROUNDS = 20


def target_of(args) -> pathlib.Path:
    target = pathlib.Path(args.target).expanduser().resolve()
    if not target.is_dir():
        sys.exit(f"target folder not found: {target}")
    return target


def options(args) -> pipeline.Options:
    seat = args.driver or (f"{args.base_url}|{args.model}" if args.base_url and args.model else None)
    driver = drivers.resolve(seat, args.max_steps)
    worker = drivers.resolve(args.worker, args.max_steps) if args.worker else driver
    reviewer = drivers.resolve(args.reviewer, 12) if args.reviewer else driver
    return pipeline.Options(driver=driver, worker=worker, reviewer=reviewer, round_minutes=args.round_minutes,
                            parallel=args.parallel, auto=getattr(args, "auto", False))


def set_goal(target: pathlib.Path, goal: str, max_rounds: int = DEFAULT_CAP,
             max_tokens: int | None = None, max_hours: float | None = None) -> dict:
    if not goal.strip():
        sys.exit("the goal is empty")
    old = core.load(target)
    if old.get("status") not in (None, "none"):
        shutil.copy2(core.results_path(target), core.results_path(target).with_name(f"results.rev{old.get('rev', 0)}.json"))
        for name in ("DESIGN.md", "PREREGISTRATION.md"):
            doc = core.loop_dir(target) / name
            if doc.exists():
                doc.rename(doc.with_name(f"{doc.stem}.rev{old.get('rev', 0)}.md"))
        shutil.rmtree(core.loop_dir(target) / ".checkpoints", ignore_errors=True)
    state = {"status": "active", "goal": goal.strip(), "target": str(target), "round": 0, "max_rounds": max_rounds,
             "rev": old.get("rev", 0) + 1, "started": core.now(), "strikes": 0,
             "budget": {"max_tokens": max_tokens, "max_hours": max_hours, "tokens_in": 0, "tokens_out": 0, "agent_seconds": 0},
             "stages": [], "attempts": [], "lessons": [], "receipts": [], "cycles": [], "redesigns": []}
    core.save(target, state)
    return state


def cmd_set(args) -> None:
    target = target_of(args)
    goal = pathlib.Path(args.goal_file).read_text() if args.goal_file else (args.goal or "")
    state = set_goal(target, goal, args.max_rounds, args.max_tokens, args.max_hours)
    print(f"goal set for {target} (rev {state['rev']}, cap {args.max_rounds} cycles)\nresults: {core.results_path(target)}")


def show_pending(target: pathlib.Path) -> None:
    pending = core.load(target).get("pending")
    if not pending:
        return
    print(f"\n----- {pending['stage']} is waiting for your approval -----")
    if pending.get("doc"):
        print(f"[{core.loop_dir(target) / pending['doc']}]\n" + core.read_doc(target, pending["doc"]))
    print("\nSummary:\n" + pending["summary"])
    if pending.get("sources"):
        print("\nSources ingested: " + ", ".join(s["title"] for s in pending["sources"]))
    print(f"\nNext: approve {target}   or   revise {target} --feedback \"...\"")


def cmd_step(args) -> None:
    target = target_of(args)
    status = asyncio.run(pipeline.step(target, options(args)))
    print(f"\nstatus: {status}")
    show_pending(target)


def cmd_run(args) -> None:
    target = target_of(args)
    status = asyncio.run(pipeline.run(target, options(args)))
    print(f"\nstatus: {status}")
    show_pending(target)


def cmd_approve(args) -> None:
    target = target_of(args)
    print(pipeline.approve(target, core.require(target)))


def cmd_revise(args) -> None:
    target = target_of(args)
    status = asyncio.run(pipeline.step(target, options(args), feedback=args.feedback))
    print(f"\nstatus: {status}")
    show_pending(target)


def cmd_show(args) -> None:
    state = core.load(target_of(args))
    brief = {k: v for k, v in state.items()
             if k not in ("stages", "attempts", "receipts", "cycles", "baseline", "lessons", "redesigns", "pending", "active")}
    brief["stages_approved"] = [s["stage"] for s in state.get("stages", [])]
    brief["pending"] = (state.get("pending") or {}).get("stage")
    brief["orders"] = [(r["cycle"], r["order"], r["verdict"][:60]) for r in state.get("receipts", [])]
    print(json.dumps(brief, indent=2))


def cmd_gates(args) -> None:
    target = target_of(args)
    state = core.require(target)
    prereg = next((s for s in state.get("stages", []) if s["stage"] == "preregistration"), None)
    gates = prereg["gates"] if prereg else pipeline.owner_gates(state)
    if not gates:
        sys.exit("no gates yet: they are set in the preregistration")
    results = core.run_gates(gates, target)
    print(core.gate_report(results) + "\n" + "\n".join(
        f"- {r['name']} (holdout): {'PASS' if r['passed'] else 'FAIL'} value={r['value']}" for r in results if r["holdout"]))


def set_status(args, status: str, key: str | None = None, note: str | None = None) -> None:
    target = target_of(args)
    state = core.require(target)
    state["status"] = status
    if key:
        state[key] = note
    if status == "active":
        state["max_rounds"] = state.get("max_rounds", DEFAULT_CAP) + RESUME_ROUNDS
        state["strikes"] = 0
    core.save(target, state)
    print(f"status: {status}" + (f" (cap now {state['max_rounds']} cycles)" if status == "active" else ""))


def add_seats(p) -> None:
    p.add_argument("--driver"); p.add_argument("--worker"); p.add_argument("--reviewer")
    p.add_argument("--base-url"); p.add_argument("--model")
    p.add_argument("--round-minutes", type=float, default=30)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--parallel", type=int, default=1)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, fn, seats=False):
        p = sub.add_parser(name)
        p.add_argument("target")
        p.set_defaults(fn=fn)
        if seats:
            add_seats(p)
        return p

    p = add("set", cmd_set)
    p.add_argument("--goal"); p.add_argument("--goal-file")
    p.add_argument("--max-rounds", type=int, default=DEFAULT_CAP)
    p.add_argument("--max-tokens", type=int); p.add_argument("--max-hours", type=float)
    add("step", cmd_step, seats=True)
    add("run", cmd_run, seats=True).add_argument("--auto", action="store_true",
                                                 help="approve stages automatically (no owner review)")
    add("approve", cmd_approve)
    add("revise", cmd_revise, seats=True).add_argument("--feedback", required=True)
    add("show", cmd_show); add("gates", cmd_gates)
    add("report", lambda a: print(pipeline.report(target_of(a))))
    add("pause", lambda a: set_status(a, "paused")); add("resume", lambda a: set_status(a, "active"))
    add("clear", lambda a: (core.save(target_of(a), {"status": "none"}), print("goal cleared")))
    add("block", lambda a: set_status(a, "blocked", "blocked_reason", a.reason)).add_argument("--reason", required=True)
    add("complete", lambda a: set_status(a, "complete", "complete_note", a.note)).add_argument("--note", required=True)
    args = parser.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
