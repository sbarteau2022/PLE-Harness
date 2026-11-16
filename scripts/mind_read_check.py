#!/usr/bin/env python3
"""Checks that a model in the seat can read the shared mind.

  .venv/bin/python scripts/mind_read_check.py <seat> "<question>" [--no-recall]

The harness recalls from the mind for the model (as the loop does), and the model also has
the atom tools. It must answer from the mind and cite atom ids. `--no-recall` withholds the
automatic recall, so the model has to use the tools itself.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import agent  # noqa: E402
from frontier import core, drivers, surreal  # noqa: E402

PROMPT = """Answer the question using only the shared hyperbolic mind. Atoms retrieved for you may be in the
user message; search further with atom_search, atom_neighbors and atom_get if you need to. Finish
with a short answer that cites the atom ids you used (for example atom:12). If the mind does not
hold the answer, say so."""


async def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seat, question = args[0], args[1]
    config = drivers.resolve(seat, max_steps=8)
    recalled = "" if "--no-recall" in sys.argv else surreal.mind_recall(question)
    instruction = question + ("\n\n# From the shared hyperbolic mind (retrieved for you)\n" + recalled if recalled else "")
    session = core.LocalSession(pathlib.Path(tempfile.mkdtemp()), instruction, 10)
    finish = [t for t in agent.TOOLS if t["function"]["name"] == "finish"]
    answer = await agent.run(session, config, system_prompt=PROMPT, tools=surreal.TOOLS + finish,
                             handlers=surreal.HANDLERS)
    print(f"\nseat: {config.model} (tool mode {config.tool_mode})\ntools used: {session.tool_calls}\nanswer: {answer}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
