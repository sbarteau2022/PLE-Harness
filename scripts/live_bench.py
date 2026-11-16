#!/usr/bin/env python3
"""Live multi-step benchmark over a recorded tool-call trajectory.

Replays the tool calls and real tool outputs of an earlier PLE run, step by step, against a
live model, and measures every model call. Some steps repeat an earlier tool call with its
same output (a re-read), so prompt reuse can be seen. Two arms per model:

  append-only   the conversation grows as the agent normally sends it
  carved        frontier.pipeline.Carved moves old tool output to memory above 8k tokens

Per call it records: time to first token, decode speed, end-to-end time, prompt and output
tokens, provider-reported cached prompt tokens (the provider's KV cache, if it reports it),
and client-side prefix reuse (share of this prompt identical to the start of the previous one).

  .venv/bin/python scripts/live_bench.py <trajectory.json> <seat> [<seat> ...] [--steps 12] [--out file.json]

The model's own replies are measured, then discarded: the recorded tool call is appended
instead, so every model sees the same conversation and arms are comparable.
"""

from __future__ import annotations

import asyncio
import copy
import json
import pathlib
import statistics
import sys
import tempfile
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import agent  # noqa: E402
import openai  # noqa: E402
from frontier import drivers, n3, pipeline  # noqa: E402

REPEATS = {5: 2, 9: 4, 12: 3, 14: 7}   # after recorded step k, re-issue the call from step v


def load_steps(path: str, count: int) -> tuple[str, list[tuple[dict, list[dict]]]]:
    data = json.loads(pathlib.Path(path).read_text())
    instruction = next((s["message"] for s in data["steps"] if s["source"] == "user"), "")
    recorded = []
    for s in data["steps"]:
        if s["source"] != "agent" or not s.get("tool_calls"):
            continue
        calls = [{"id": c["tool_call_id"], "type": "function",
                  "function": {"name": "bash" if c["function_name"] == "exec" else c["function_name"],
                               "arguments": json.dumps({k: v for k, v in c["arguments"].items() if k == "command"})}}
                 for c in s["tool_calls"]]
        results = [r.get("content") or "" for r in (s.get("observation") or {}).get("results") or []]
        recorded.append((calls, results))
    recorded = recorded[:count]
    steps, n = [], 0
    for index, (calls, results) in enumerate(recorded, 1):
        steps.append((calls, results, False))
        if index in REPEATS and REPEATS[index] <= len(recorded):
            again_calls, again_results = recorded[REPEATS[index] - 1]
            steps.append((again_calls, again_results, True))
    out = []
    for calls, results, repeat in steps:
        n += 1
        ids = [f"call_{n}_{i}" for i in range(len(calls))]
        assistant = {"role": "assistant", "content": "",
                     "tool_calls": [dict(c, id=i) for c, i in zip(calls, ids)]}
        tools = [{"role": "tool", "tool_call_id": i, "content": agent._clip(r, 20000)} for i, r in zip(ids, results)]
        out.append((assistant, tools, repeat))
    return instruction, out


def common_prefix(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    lo, hi = 0, limit
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


async def measure(client, model: str, messages: list[dict]) -> dict:
    started = time.perf_counter()
    first = None
    usage = None
    stream = await client.chat.completions.create(
        model=model, messages=messages, tools=agent.TOOLS, tool_choice="auto", max_tokens=200,
        temperature=0, stream=True, stream_options={"include_usage": True})
    async for chunk in stream:
        if first is None and chunk.choices:
            delta = chunk.choices[0].delta
            if (delta.content or delta.tool_calls or getattr(delta, "reasoning_content", None)
                    or getattr(delta, "reasoning", None)):
                first = time.perf_counter()
        if getattr(chunk, "usage", None):
            usage = chunk.usage
    end = time.perf_counter()
    prompt = getattr(usage, "prompt_tokens", None) or 0
    output = getattr(usage, "completion_tokens", None) or 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details is not None else None
    ttft = (first or end) - started
    decode = output / (end - first) if first and end > first and output > 1 else None
    return {"ttft": ttft, "e2e": end - started, "decode_tps": decode, "prompt_tokens": prompt,
            "output_tokens": output, "cached_tokens": cached}


async def run_arm(config, instruction: str, steps, high: int | None) -> list[dict]:
    carved = high is not None
    client = openai.AsyncOpenAI(base_url=config.base_url, api_key=config.api_key, max_retries=3, timeout=600)
    # A fresh run id at the very start of the prompt makes every arm start with a cold provider
    # cache, so earlier benchmark runs cannot lend one arm cache hits the other does not get.
    nonce = f"Run id: {uuid.uuid4().hex}\n\n"
    messages = [{"role": "system", "content": nonce + agent.SYSTEM_PROMPT}, {"role": "user", "content": instruction}]
    manager = pipeline.Carved(n3.N3(pathlib.Path(tempfile.mkdtemp()) / "m.sqlite"), 8000, high or None) if carved else None
    if "--prime" in sys.argv:
        # Put the fixed prefix (run id, system prompt, task) in the provider cache before step 1,
        # so every arm starts with the same warm fixed prefix and nothing else.
        try:
            await measure(client, config.model, messages)
        except Exception:
            pass
    rows, previous = [], ""
    for number, (assistant, tools, repeat) in enumerate(steps, 1):
        if manager is not None:
            manager.manage(messages, 200000)
        text = json.dumps(messages, ensure_ascii=False)
        reuse = common_prefix(text, previous) / len(text) if previous else 0.0
        try:
            row = await measure(client, config.model, messages)
        except Exception as exc:  # recorded, not fatal
            row = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        row.update(step=number, repeat=repeat, prefix_reuse=reuse, est_prompt_tokens=len(text) // 4)
        rows.append(row)
        print(f"  {config.model.split('/')[-1]:28} {'carved' if carved else 'append':6} step {number:2}"
              f"{' (repeat)' if repeat else '         '} prompt {row.get('prompt_tokens', '-'):>6} "
              f"ttft {row.get('ttft', 0):6.2f}s e2e {row.get('e2e', 0):6.2f}s cached {row.get('cached_tokens')}"
              f" reuse {reuse:.0%} {row.get('error', '')}", flush=True)
        previous = text
        messages.append(copy.deepcopy(assistant))
        messages.extend(copy.deepcopy(tools))
    return rows


def summarize(rows: list[dict]) -> dict:
    good = [r for r in rows if "error" not in r]
    def med(key):
        values = [r[key] for r in good if r.get(key) is not None]
        return statistics.median(values) if values else None
    cached = [r["cached_tokens"] for r in good if r.get("cached_tokens") is not None]
    prompt = sum(r["prompt_tokens"] for r in good)
    return {"calls": len(rows), "errors": len(rows) - len(good), "prompt_tokens": prompt,
            "output_tokens": sum(r["output_tokens"] for r in good),
            "ttft_median": med("ttft"), "decode_tps_median": med("decode_tps"), "e2e_median": med("e2e"),
            "e2e_total": sum(r["e2e"] for r in good),
            "cached_reported": bool(cached), "cached_tokens": sum(cached) if cached else None,
            "cache_hit_rate": (sum(cached) / prompt) if cached and prompt else None,
            "prefix_reuse_mean": statistics.mean(r["prefix_reuse"] for r in rows[1:]) if len(rows) > 1 else 0.0,
            "repeat_steps": sum(1 for r in rows if r["repeat"])}


async def main(argv: list[str]) -> int:
    path, rest = argv[0], argv[1:]
    count = int(rest[rest.index("--steps") + 1]) if "--steps" in rest else 12
    out = pathlib.Path(rest[rest.index("--out") + 1]) if "--out" in rest else None
    seats = [a for i, a in enumerate(rest) if not a.startswith("--") and (i == 0 or rest[i - 1] not in ("--steps", "--out", "--stable"))]
    instruction, steps = load_steps(path, count)
    print(f"{len(steps)} steps ({sum(1 for s in steps if s[2])} repeats) from {path}")
    report = {"trajectory": path, "steps": len(steps), "models": {}}
    for seat in seats:
        config = drivers.resolve(seat, max_steps=1)
        arms = {}
        arm_plan = [("regular_append_only", None), ("ple_carve_every_call", 8000)]
        if "--stable" in rest:
            arm_plan.append(("ple_carve_prefix_stable", int(rest[rest.index("--stable") + 1])))
        for name, high in arm_plan:
            rows = await run_arm(config, instruction, steps, high)
            arms[name] = {"summary": summarize(rows), "calls": rows}
        report["models"][config.model] = arms
        if out:
            out.write_text(json.dumps(report, indent=2))
    print(json.dumps({m: {a: v["summary"] for a, v in arms.items()} for m, arms in report["models"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
