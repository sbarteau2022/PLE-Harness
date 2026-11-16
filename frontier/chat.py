#!/usr/bin/env python3
"""Interactive frontier session: set the folder and the goal in chat, review every step.

  .venv/bin/python frontier/chat.py [--driver PROFILE | --base-url URL --model NAME]
                                    [--worker SEAT] [--reviewer SEAT] [--target DIR]

You are asked for the folder and the goal, then each stage (research, audit, design,
preregistration) is shown for your approval. At a prompt:
  a            approve and continue
  r <text>     revise: redo the stage with your feedback
  e            open the stage's document in $EDITOR, then approve what you saved
  s            show status      p   print the report      q   quit (resume later; state is saved)
After the preregistration is approved the brigade runs; you are asked before each cycle
unless you answer `go` (run to the end).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from frontier import core, loop, pipeline  # noqa: E402


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


def multiline(prompt: str) -> str:
    print(prompt + " (finish with an empty line)")
    lines = []
    while True:
        line = ask("  ")
        if not line or line == "q":
            return "\n".join(lines)
        lines.append(line)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target")
    loop.add_seats(parser)
    args = parser.parse_args(argv)
    args.auto = False
    opts = loop.options(args)
    print(f"driver: {opts.driver.model} @ {opts.driver.base_url}\nworker: {opts.worker.model}   reviewer: {opts.reviewer.model}")

    target = pathlib.Path(args.target or ask("Folder to work on: ")).expanduser().resolve()
    if not target.is_dir():
        sys.exit(f"folder not found: {target}")
    state = core.load(target)
    if state.get("status") in (None, "none") or ask(
            f"A goal already exists here (status {state['status']}). Continue it? [Y/n] ").lower() == "n":
        goal = multiline("Goal (say what to improve and the target that counts as success)")
        cap = ask("Cycle cap [20]: ") or "20"
        loop.set_goal(target, goal, int(cap))
    elif state["status"] != "active":
        print(f"status is {state['status']}: {state.get('blocked_reason') or state.get('capped_reason') or ''}")
        if ask("Resume? [y/N] ").lower() != "y":
            return 0
        state["status"] = "active"; state["strikes"] = 0
        state["max_rounds"] = state.get("max_rounds", 20) + 20
        core.save(target, state)

    free_run = False
    while True:
        state = core.load(target)
        pending = state.get("pending")
        if pending:
            loop.show_pending(target)
            answer = ask("\n[a]pprove  [r <feedback>]  [e]dit  [s]tatus  [p]rint report  [q]uit > ")
            if answer == "a":
                print(pipeline.approve(target, state))
            elif answer.startswith("r"):
                asyncio.run(pipeline.step(target, opts, feedback=answer[1:].strip() or multiline("Feedback")))
            elif answer == "e" and pending.get("doc"):
                subprocess.call([os.environ.get("EDITOR", "nano"), str(core.loop_dir(target) / pending["doc"])])
                print(pipeline.approve(target, core.load(target)))
            elif answer == "s":
                loop.cmd_show(argparse.Namespace(target=str(target)))
            elif answer == "p":
                print(pipeline.report(target))
            elif answer == "q":
                return 0
            continue
        if state["status"] != "active":
            print("\n" + pipeline.report(target))
            return 0
        if pipeline.next_stage(state) is None and not free_run:
            answer = ask(f"\nBrigade cycle {state['round'] + 1}: [Enter] run one cycle  [go] run to the end  "
                         "[p]rint report  [q]uit > ")
            if answer == "q":
                return 0
            if answer == "p":
                print(pipeline.report(target))
                continue
            free_run = answer == "go"
        asyncio.run(pipeline.step(target, opts))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
