"""PLE harness entry point: serves the agent loop over the evaluator's WebSocket protocol.

The evaluator runs `serve` from submission.yaml, which starts this file. Each task
arrives on its own connection; `taskbridge.run` builds a Session for it and calls `solve`.
"""

from __future__ import annotations

import datetime as dt
import json

import agent
import taskbridge
from ablations.normalize import MARKER
from frontier import n3


async def solve(session: taskbridge.Session) -> str:
    tools = agent.TOOLS + (agent.MCP_TOOLS if session.mcp_servers else [])
    handlers = {}
    if n3.EDGE_SHARD.is_dir():
        rag = n3.Rag()
        tools += [tool for tool in n3.RAG_TOOLS
                  if tool["function"]["name"] in {"rag_search", "rag_open"}]
        handlers = {name: handler for name, handler in n3.make_handlers(None, rag).items()
                    if name in {"rag_search", "rag_open"}}
    started = dt.datetime.now(dt.timezone.utc)
    meter: dict = {}
    try:
        return await agent.run(session, agent.Config.from_env(), tools=tools,
                               handlers=handlers, telemetry=meter)
    finally:
        ended = dt.datetime.now(dt.timezone.utc)
        print(f"{MARKER} " + json.dumps({
            "harness": "ple", "started_at": started.isoformat(), "ended_at": ended.isoformat(),
            "telemetry": meter,
        }), flush=True)


if __name__ == "__main__":
    taskbridge.run(solve)
