"""Independent OpenAI-compatible tool loop used as the generic baseline."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import time
from typing import Any

import openai
import taskbridge

SYSTEM = """You are an autonomous agent in a task workspace. Read the task and its
files, create the requested deliverable at the exact requested path, verify it,
and call finish alone after all other tool results have been observed. Do not
inspect graders, tests, answer keys, or verifier code."""

TOOLS = [
    {"type":"function","function":{"name":"bash","description":"Run a shell command in the task workspace.","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout_sec":{"type":"integer"}},"required":["command"]}}},
    {"type":"function","function":{"name":"read_file","description":"Read a UTF-8 text file.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"write_file","description":"Write a UTF-8 text file, overwriting it.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
    {"type":"function","function":{"name":"finish","description":"Finish after the deliverable is written and checked.","parameters":{"type":"object","properties":{"summary":{"type":"string"}},"required":["summary"]}}},
]
MCP_TOOLS = [
    {"type":"function","function":{"name":"mcp_list","description":"List task MCP tools.","parameters":{"type":"object","properties":{"server":{"type":"string"},"contains":{"type":"string"}}}}},
    {"type":"function","function":{"name":"mcp_schema","description":"Read one MCP tool schema.","parameters":{"type":"object","properties":{"server":{"type":"string"},"tool":{"type":"string"}},"required":["server","tool"]}}},
    {"type":"function","function":{"name":"mcp_call","description":"Call one task MCP tool.","parameters":{"type":"object","properties":{"server":{"type":"string"},"tool":{"type":"string"},"arguments":{"type":"object"}},"required":["server","tool"]}}},
]

@dataclasses.dataclass(frozen=True)
class Config:
    model: str
    base_url: str | None
    api_key: str
    max_steps: int = 400
    max_tokens: int = 16000
    reserve_seconds: float = 240
    max_tool_chars: int = 20000

    @classmethod
    def from_env(cls) -> "Config":
        return cls(os.environ.get("MODEL_NAME", ""), os.environ.get("MODEL_BASE_URL") or None,
                   os.environ.get("MODEL_API_KEY") or "none", int(os.environ.get("MAX_STEPS", "400")),
                   int(os.environ.get("MAX_OUTPUT_TOKENS", "16000")),
                   float(os.environ.get("RESERVE_SECONDS", "240")))

def clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    half = limit // 2
    return value[:half] + f"\n...[{len(value)-2*half} chars omitted]...\n" + value[-half:]

class Mcp:
    def __init__(self, session: taskbridge.Session, limit: int):
        self.session, self.limit = session, limit
        self.clients: dict[str, taskbridge.MCPClient] = {}
        self.cache: dict[str, list[dict[str, Any]]] = {}

    async def client(self, name: str) -> taskbridge.MCPClient:
        if name not in self.clients:
            rows = [x for x in self.session.mcp_servers if x.get("name") == name]
            if not rows:
                raise ValueError(f"unknown MCP server {name!r}")
            client = taskbridge.MCPClient(self.session, rows[0]["url"])
            await client.initialize()
            self.clients[name] = client
        return self.clients[name]

    async def tools(self, name: str) -> list[dict[str, Any]]:
        if name not in self.cache:
            self.cache[name] = await (await self.client(name)).list_tools()
        return self.cache[name]

    async def run(self, name: str, args: dict[str, Any]) -> str:
        if name == "mcp_list":
            names = [args["server"]] if args.get("server") else [x.get("name", "") for x in self.session.mcp_servers]
            rows = []
            for server in names:
                for tool in await self.tools(server):
                    row = f"{server}.{tool.get('name')}: {' '.join(str(tool.get('description','')).split())[:120]}"
                    if not args.get("contains") or args["contains"].lower() in row.lower(): rows.append(row)
            return clip("\n".join(rows) or "no matching MCP tools", self.limit)
        if name == "mcp_schema":
            for tool in await self.tools(args["server"]):
                if tool.get("name") == args["tool"]:
                    return clip(json.dumps(tool.get("inputSchema", {})), self.limit)
            return "error: MCP tool not found"
        result = await (await self.client(args["server"])).call_tool(args["tool"], args.get("arguments") or {})
        text = "\n".join(x.get("text", "") for x in result.get("content", []) if x.get("type") == "text")
        return ("error: " if result.get("isError") else "") + clip(text or json.dumps(result), self.limit)

async def execute(session: taskbridge.Session, mcp: Mcp, name: str, args: dict[str, Any], limit: int) -> str:
    try:
        if name == "bash":
            left = session.time_left()
            timeout = min(int(args.get("timeout_sec", 300)), 600, max(1, int(left)-5) if left else 600)
            result = await session.exec(args["command"], timeout_sec=timeout)
            return f"exit_code={result.get('exit_code')}\n" + clip(result.get("output", ""), limit)
        if name == "read_file":
            data = await session.read_file(args["path"])
            return clip(data.decode("utf-8", "replace"), limit)
        if name == "write_file":
            size = await session.write_file(args["path"], args["content"])
            return f"ok: wrote {size} bytes"
        if name.startswith("mcp_"):
            return await mcp.run(name, args)
        return f"error: unknown tool {name}"
    except (KeyError, TypeError, ValueError, taskbridge.TaskError, taskbridge.MCPError) as exc:
        return f"error: {type(exc).__name__}: {exc}"

def assistant_message(message: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"role":"assistant", "content":message.content or ""}
    if message.tool_calls:
        out["tool_calls"] = [{"id":c.id,"type":"function","function":{"name":c.function.name,"arguments":c.function.arguments}} for c in message.tool_calls]
    return out

async def run(session: taskbridge.Session, config: Config, telemetry: dict[str, Any]) -> str:
    client = openai.AsyncOpenAI(base_url=config.base_url, api_key=config.api_key, max_retries=0, timeout=600)
    tools = TOOLS + (MCP_TOOLS if session.mcp_servers else [])
    messages: list[dict[str, Any]] = [{"role":"system","content":SYSTEM}, {"role":"user","content":f"Working directory: {session.workdir}\n\n{session.instruction}"}]
    meter = {"steps":0,"response_ids":[],"prompt":0,"output":0,"cached":0,"max_context":0,
             "e2e_ms":[],"by_name":{},"successes":0,"errors":0,"finish_refusals":0,"no_progress_turns":0}
    mcp, idle, summary = Mcp(session, config.max_tool_chars), 0, ""
    for step in range(1, config.max_steps + 1):
        left = session.time_left()
        if left is not None and left <= config.reserve_seconds:
            messages.append({"role":"user","content":"Time is nearly exhausted. Write the deliverable now, verify it, then call finish alone."})
        started = time.monotonic()
        response = await client.chat.completions.create(model=config.model, messages=messages, tools=tools,
                                                        tool_choice="auto", max_tokens=config.max_tokens)
        meter["steps"] += 1; meter["e2e_ms"].append(round((time.monotonic()-started)*1000))
        if getattr(response, "id", None): meter["response_ids"].append(response.id)
        if response.usage:
            cached = getattr(getattr(response.usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            meter["prompt"] += response.usage.prompt_tokens or 0; meter["output"] += response.usage.completion_tokens or 0
            meter["cached"] += cached; meter["max_context"] = max(meter["max_context"], response.usage.prompt_tokens or 0)
            await session.usage(config.model, response.usage.prompt_tokens or 0, response.usage.completion_tokens or 0, cached)
        message = response.choices[0].message; calls = message.tool_calls or []
        messages.append(assistant_message(message))
        if not calls:
            idle += 1; meter["no_progress_turns"] += 1; summary = message.content or summary
            if idle >= 3: break
            messages.append({"role":"user","content":"Continue with a tool call, or call finish alone."}); continue
        idle = 0; finished = False
        for call in calls:
            try: args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError: args, output = {}, "error: invalid JSON arguments"
            else:
                if call.function.name == "finish" and len(calls) > 1:
                    output = "not finished: finish accompanied other tools; read their results and call finish alone"
                    meter["finish_refusals"] += 1
                elif call.function.name == "finish":
                    summary, output, finished = str(args.get("summary", "")), "ok", True
                else: output = await execute(session, mcp, call.function.name, args, config.max_tool_chars)
            meter["by_name"][call.function.name] = meter["by_name"].get(call.function.name, 0) + 1
            if output.startswith("error:"): meter["errors"] += 1
            else: meter["successes"] += 1
            await session.log(f"generic step {step}: {call.function.name} -> {output[:160]!r}")
            messages.append({"role":"tool","tool_call_id":call.id,"content":output})
        if finished: break
    telemetry.update(meter); telemetry["uncached"] = meter["prompt"] - meter["cached"]
    return summary or "Stopped at the step or time limit; inspect deliverables on disk."
