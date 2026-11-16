"""Serve-protocol entry for the independent generic baseline arm."""
from __future__ import annotations
import datetime as dt
import json
import taskbridge
from ablations.generic.agent import Config, run

MARKER = "PLE-ABLATION-RUN"

async def solve(session: taskbridge.Session) -> str:
    started = dt.datetime.now(dt.timezone.utc); meter: dict = {}
    try:
        return await run(session, Config.from_env(), meter)
    finally:
        print(f"{MARKER} " + json.dumps({"harness":"generic","started_at":started.isoformat(),
              "ended_at":dt.datetime.now(dt.timezone.utc).isoformat(),"telemetry":meter}), flush=True)

if __name__ == "__main__": taskbridge.run(solve)
