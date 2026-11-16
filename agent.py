"""Tool-calling agent loop for Physical-Engineering's Last Exam (PLE) tasks.

The loop is model-agnostic: it talks to any OpenAI-compatible Chat Completions
endpoint that supports tool calling. Configuration is read from the environment
(see `Config.from_env`), so the same code runs locally against NVIDIA-hosted
models and inside the competition's serve container.

Tools exposed to the model:
  bash        run a shell command in the task container (fresh shell per call)
  read_file   read a text file from the task container
  write_file  write a text file in the task container
  finish      end the task with a final summary
  rag_search  search the local research library, when an Edge shard is available
  rag_open    read more passages of a returned source from that shard
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import json
import os
import pathlib
import re
import time
import types
from typing import Any

import engineering_verify

import openai

import taskbridge

SYSTEM_PROMPT = """You are an autonomous engineering agent working inside a project workspace.
You act as the licensed engineer of record for the task in the user message.

How to work:
- Start by listing and reading the project documents the task points to. Read the
  task's own brief file first if it names one.
- Produce the deliverable at the exact path the task names. Write a complete,
  valid first version early, then improve it in place. An imperfect file on disk
  earns more than no file.
- Check your numbers: recompute key results with a script, check units, and
  check every constraint the documents impose. State assumptions explicitly in the
  deliverable.
- Use bash for Python, spreadsheets, CAD and BIM scripts, and unzipping. Prefer
  plain text, .csv, .json, .md, .py, .docx, .xlsx or .pdf deliverables.
- Do not open, list or search for grading files, test folders, answer keys or
  verifier code. Work only from the project workspace and its documents.
- Keep tool outputs small (use head, tail, grep). Commands run in fresh shells:
  cd and export do not persist, but files do.
- When available, use rag_search for relevant external research and rag_open to
  read further passages. A returned source path is an archive key, not a live
  filesystem path; the task's own brief and project files take precedence.

When the deliverable is complete and checked, call `finish` with a concise summary
that lists each deliverable path, the key results, and any open issues. Do not call
`finish` before a deliverable exists on disk."""

MCP_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "mcp_list",
            "description": (
                "List the tools of the task's MCP servers (project applications: documents, "
                "spreadsheets, CAD, mail, code execution). Pass `server` to list one server, "
                "and `contains` to filter by name or description."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "server": {"type": "string"},
                    "contains": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mcp_schema",
            "description": "Show the full input schema of one MCP tool before calling it.",
            "parameters": {
                "type": "object",
                "properties": {"server": {"type": "string"}, "tool": {"type": "string"}},
                "required": ["server", "tool"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mcp_call",
            "description": "Call one MCP tool and return its text result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server": {"type": "string"},
                    "tool": {"type": "string"},
                    "arguments": {"type": "object", "description": "Arguments per the tool schema."},
                },
                "required": ["server", "tool"],
            },
        },
    },
]

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a bash command in the task container, starting in the task working "
                "directory. Output (stdout and stderr) is truncated past the limit, keeping "
                "the head and tail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command passed to bash -c."},
                    "timeout_sec": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 300, max 600).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file. Long files are truncated (head and tail kept).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a UTF-8 text file, creating parent directories. Overwrites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "End the task. Call only after the deliverable is on disk.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
]
TOOLS.insert(-1, engineering_verify.TOOL)


@dataclasses.dataclass(frozen=True)
class Config:
    """HTTP settings from the environment; local plugins require trusted profile configuration."""

    model: str
    base_url: str | None
    api_key: str
    max_output_tokens: int = 16000
    compact_at_tokens: int = 90000
    max_steps: int = 400
    reserve_seconds: float = 240.0
    max_tool_output_chars: int = 20000
    max_retries: int = 6
    checkpoint_every: int = 25
    tool_mode: str = "auto"  # "auto" (native, falling back to text) or "text"
    carve_high: int = 0  # carve only above this many tokens; 0 = the carve floor (see pipeline.Carved)
    local_runtime: dict[str, Any] | None = dataclasses.field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> "Config":
        env = os.environ
        return cls(
            model=env.get("MODEL_NAME", ""),
            base_url=env.get("MODEL_BASE_URL") or None,
            api_key=env.get("MODEL_API_KEY") or "none",
            max_output_tokens=int(env.get("MAX_OUTPUT_TOKENS", "16000")),
            compact_at_tokens=int(env.get("COMPACT_AT_TOKENS", "90000")),
            max_steps=int(env.get("MAX_STEPS", "400")),
            reserve_seconds=float(env.get("RESERVE_SECONDS", "240")),
            checkpoint_every=int(env.get("CHECKPOINT_EVERY", "25")),
        )


def _clip(text: str, limit: int) -> str:
    """Keeps the head and tail of `text` when it exceeds `limit` characters."""
    if len(text) <= limit:
        return text
    half = limit // 2
    dropped = len(text) - 2 * half
    return f"{text[:half]}\n... [{dropped} characters truncated] ...\n{text[-half:]}"


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate (about four characters per token)."""
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) // 4


def compact(messages: list[dict[str, Any]], budget: int, keep_recent: int = 12) -> None:
    """Shrinks old tool results in place until the estimate fits `budget`.

    Keeps the system prompt, the task, and the most recent `keep_recent` messages
    intact. Older tool outputs longer than 400 characters are replaced by a short
    marker, so the model can re-run a command if it needs the text again.
    """
    if _estimate_tokens(messages) <= budget:
        return
    cutoff = max(2, len(messages) - keep_recent)
    for index in range(2, cutoff):
        message = messages[index]
        if message.get("role") != "tool":
            continue
        content = message.get("content") or ""
        if len(content) > 400:
            message["content"] = f"[earlier tool output elided: {len(content)} characters]"
        if _estimate_tokens(messages) <= budget:
            return


TEXT_TOOL_PROTOCOL = """You have tools. To call one, reply with exactly one block and nothing after it:

<tool_call>{"name": "<tool name>", "arguments": {<arguments as JSON>}}</tool_call>

The result arrives in the next message. Tools:
"""


def repair_json(text: str) -> Any:
    """Parses JSON that small models tend to emit: fenced, single-quoted, trailing commas."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    for candidate in (text, re.sub(r",\s*([}\]])", r"\1", text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        inner = re.sub(r",\s*([}\]])", r"\1", text[start : end + 1])
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(inner)  # Python-style dicts with single quotes
            except (ValueError, SyntaxError):
                pass
    raise json.JSONDecodeError("unrepairable JSON", text, 0)


def text_tool_calls(content: str, names: set[str]) -> list[Any]:
    """Finds tool calls a model wrote as text instead of using the tool-call API."""
    if not content:
        return []
    chunks = re.findall(r"(?s)<tool_call>(.*?)(?:</tool_call>|$)", content)
    chunks += re.findall(r"(?s)```(?:json|tool_call)?\s*(\{.*?\})\s*```", content)
    if not chunks and content.strip().startswith("{"):
        chunks = [content]
    calls = []
    for index, chunk in enumerate(chunks):
        try:
            data = repair_json(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "function" in data and isinstance(data["function"], dict):
            data = data["function"]
        name = data.get("name") if isinstance(data, dict) else None
        if name not in names:
            continue
        arguments = data.get("arguments", data.get("parameters", {}))
        if isinstance(arguments, str):
            try:
                arguments = repair_json(arguments)
            except json.JSONDecodeError:
                arguments = {}
        calls.append(types.SimpleNamespace(
            id=f"text_{index}_{abs(hash(chunk)) % 10**8}",
            function=types.SimpleNamespace(name=name, arguments=json.dumps(arguments))))
    return calls


def to_text_messages(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrites a tool-call conversation for servers that do not accept `tools`."""
    listing = "\n".join(
        f"- {t['function']['name']}: {t['function']['description']} "
        f"Arguments schema: {json.dumps(t['function']['parameters'])}" for t in tools)
    out = []
    for message in messages:
        role, content = message["role"], message.get("content") or ""
        if role == "system":
            out.append({"role": "system", "content": f"{content}\n\n{TEXT_TOOL_PROTOCOL}{listing}"})
        elif role == "assistant":
            for call in message.get("tool_calls", []):
                body = {"name": call["function"]["name"]}
                try:
                    body["arguments"] = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    body["arguments"] = {}
                content += f"\n<tool_call>{json.dumps(body)}</tool_call>"
            out.append({"role": "assistant", "content": content.strip() or "(no text)"})
        elif role == "tool":
            out.append({"role": "user", "content": f"Tool result:\n{content}"})
        else:
            out.append({"role": "user", "content": content})
    merged: list[dict[str, Any]] = []
    for message in out:  # many local chat templates reject two user turns in a row
        if merged and merged[-1]["role"] == message["role"] == "user":
            merged[-1]["content"] += "\n\n" + message["content"]
        else:
            merged.append(message)
    return merged


class ModelClient:
    """Chat completion dispatch through HTTP or an explicitly selected local runtime.

    Uses native tool calling when the server supports it. If the server rejects the `tools`
    parameter (common for local MLX and llama.cpp servers), it switches to a text protocol
    where the model writes <tool_call> blocks.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self.text_mode = "text" in (config.tool_mode, os.environ.get("TOOL_MODE", "auto"))
        self._local = None
        if config.local_runtime is not None:
            from frontier.local_chat import LocalChatClient
            self._local = LocalChatClient(config)
            self._client = None
            return
        self._client = openai.AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            max_retries=0,
            timeout=600.0,
        )

    def bind_mind(self, mind: Any) -> None:
        if self._local is not None:
            self._local.bind_mind(mind)

    async def _create(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        if self.text_mode:
            return await self._client.chat.completions.create(
                model=self._config.model, messages=to_text_messages(messages, tools),
                max_tokens=self._config.max_output_tokens)
        return await self._client.chat.completions.create(
            model=self._config.model, messages=messages, tools=tools, tool_choice="auto",
            max_tokens=self._config.max_output_tokens)

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> Any:
        """Dispatch locally, or retry eligible HTTP rate limits, timeouts and server errors."""
        tools = tools or TOOLS
        if self._local is not None:
            encoded_messages = to_text_messages(messages, tools) if self.text_mode else messages
            self.attempts = getattr(self, "attempts", 0) + 1
            return await asyncio.to_thread(self._local.complete, encoded_messages, [] if self.text_mode else tools)
        delay = 2.0
        for attempt in range(self._config.max_retries):
            try:
                self.attempts = getattr(self, "attempts", 0) + 1
                return await self._create(messages, tools)
            except (openai.BadRequestError, openai.UnprocessableEntityError, openai.NotFoundError) as exc:
                if self.text_mode or not re.search(r"tool|function|template|unsupported", str(exc), re.I):
                    raise
                self.text_mode = True  # the server cannot take `tools`; fall back for the rest of the run
            except (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            ):
                if attempt == self._config.max_retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise RuntimeError("the model call kept failing")


class McpRouter:
    """Lazily connects to the task's MCP servers and caches their tool lists."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._clients: dict[str, Any] = {}
        self._tools: dict[str, list[dict[str, Any]]] = {}

    def server_names(self) -> list[str]:
        return [s.get("name", "") for s in self._session.mcp_servers]

    async def _client(self, server: str) -> Any:
        if server not in self._clients:
            matches = [s for s in self._session.mcp_servers if s.get("name") == server]
            if not matches:
                raise KeyError(f"unknown MCP server {server!r}; known: {self.server_names()}")
            client = taskbridge.MCPClient(self._session, matches[0]["url"])
            await client.initialize()
            self._clients[server] = client
        return self._clients[server]

    async def tools(self, server: str) -> list[dict[str, Any]]:
        if server not in self._tools:
            self._tools[server] = await (await self._client(server)).list_tools()
        return self._tools[server]

    async def list(self, server: str | None, contains: str | None, limit: int) -> str:
        lines = []
        for name in [server] if server else self.server_names():
            for tool in await self.tools(name):
                desc = " ".join(str(tool.get("description", "")).split())[:120]
                line = f"{name}.{tool.get('name')}: {desc}"
                if contains and contains.lower() not in line.lower():
                    continue
                lines.append(line)
        if not lines:
            return "no matching MCP tools"
        return _clip("\n".join(lines), limit)

    async def schema(self, server: str, tool: str, limit: int) -> str:
        for item in await self.tools(server):
            if item.get("name") == tool:
                return _clip(json.dumps(item.get("inputSchema", {}), indent=1), limit)
        return f"error: no tool {tool!r} on server {server!r}"

    async def call(self, server: str, tool: str, arguments: Any, limit: int) -> str:
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        result = await (await self._client(server)).call_tool(tool, arguments or {})
        parts = []
        for block in result.get("content", []) or []:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(f"[{block.get('type', 'binary')} content omitted]")
        text = "\n".join(parts) or json.dumps(result)[:limit]
        prefix = "error: tool reported an error\n" if result.get("isError") else ""
        return prefix + _clip(text, limit)


class ToolRunner:
    """Executes the model's tool calls against the task container and its MCP servers."""

    def __init__(self, session: Any, config: Config, handlers: dict[str, Any] | None = None) -> None:
        self._session = session
        self._limit = config.max_tool_output_chars
        self._mcp = McpRouter(session)
        self._handlers = handlers or {}  # extra tools: name -> async fn(args) -> str

    async def run(self, name: str, args: dict[str, Any]) -> str:
        try:
            if name == "engineering_verify":
                result = engineering_verify.run(args)
                return result if len(result) <= self._limit else "error: engineering verification report exceeds tool output limit; reduce manifest size"
            if name in self._handlers:
                return _clip(str(await self._handlers[name](args)), self._limit)
            if name == "bash":
                return await self._bash(args)
            if name == "read_file":
                return await self._read(args)
            if name == "write_file":
                return await self._write(args)
            if name == "mcp_list":
                return await self._mcp.list(args.get("server"), args.get("contains"), self._limit)
            if name == "mcp_schema":
                return await self._mcp.schema(args["server"], args["tool"], self._limit)
            if name == "mcp_call":
                return await self._mcp.call(
                    args["server"], args["tool"], args.get("arguments"), self._limit
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, taskbridge.MCPError,
                taskbridge.TaskError) as exc:
            return f"error: bad arguments for {name}: {exc}"
        return f"error: unknown tool {name}"

    async def _bash(self, args: dict[str, Any]) -> str:
        timeout = min(int(args.get("timeout_sec") or 300), 600)
        left = self._session.time_left()
        if left is not None:
            timeout = max(1, min(timeout, int(left) - 5))
        result = await self._session.exec(args["command"], timeout_sec=timeout)
        flag = " (timed out)" if result.get("timed_out") else ""
        output = _clip(result.get("output", ""), self._limit)
        return f"exit_code={result.get('exit_code')}{flag}\n{output}"

    async def _read(self, args: dict[str, Any]) -> str:
        data = await self._session.read_file(args["path"])
        if b"\x00" in data[:4096]:
            return "error: binary file; inspect it with bash (e.g. python or unzip)."
        return _clip(data.decode("utf-8", errors="replace"), self._limit)

    async def _write(self, args: dict[str, Any]) -> str:
        size = await self._session.write_file(args["path"], args["content"])
        return f"ok: wrote {size} bytes to {args['path']}"


async def run(
    session: Any,
    config: Config | None = None,
    system_prompt: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    handlers: dict[str, Any] | None = None,
    transcript: str | os.PathLike | None = None,
    memory: Any | None = None,
    mind: Any | None = None,
    telemetry: dict[str, Any] | None = None,
) -> str:
    """Runs the agent on the task held by `session` and returns the final message.

    `transcript` names a JSON file the conversation is saved to after every step. If it
    exists when the run starts, the run resumes from it, so a crash loses at most one step.

    `memory` is an object with `manage(messages, budget)`. When given, it manages the context
    (moving old tool output into a store the model can recall from) instead of compaction.
    `mind` is the existing shared mind used by an optional local runtime's nested cache path.
    `telemetry`, when given, is filled with measured run counters (steps, live
    response IDs, tokens, per-call latency, tool counts); it never changes behavior.
    """
    config = config or Config.from_env()
    model = ModelClient(config)
    if config.local_runtime is not None:
        model.bind_mind(mind)
    runner = ToolRunner(session, config, handlers)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Working directory: {session.workdir}\n\n{session.instruction}",
        },
    ]
    tool_specs = tools or (TOOLS + (MCP_TOOLS if session.mcp_servers else []))
    tool_names = {t["function"]["name"] for t in tool_specs}
    summary = ""
    idle_turns = 0
    repeats: list[str] = []  # fingerprints of recent tool calls, to catch thrashing
    first_step = 1
    saved = pathlib.Path(transcript) if transcript else None
    if saved and saved.exists():
        try:
            state = json.loads(saved.read_text())
            messages, first_step = state["messages"], state["step"] + 1
            await session.log(f"resumed from step {state['step']}")
        except (json.JSONDecodeError, KeyError):
            pass
    total_time = session.time_left()
    slowest_step = 0.0
    drafted = wrapping_up = False
    meter = {"steps": 0, "response_ids": [], "prompt": 0, "output": 0, "cached": 0,
             "max_context": 0, "e2e_ms": [], "by_name": {}, "successes": 0,
             "errors": 0, "finish_refusals": 0, "no_progress_turns": 0}
    for step in range(first_step, config.max_steps + 1):
        step_started = time.monotonic()
        left = session.time_left()
        steps_left = config.max_steps - step + 1
        # Keep enough time for a few more steps at this model's observed pace.
        out_of_time = left is not None and left < max(config.reserve_seconds, 3 * slowest_step)
        used = 1 - left / total_time if left is not None and total_time else 0.0
        if out_of_time:
            await session.log("time nearly up; asking for a final summary")
            messages.append({
                "role": "user",
                "content": "Time is nearly up. Call finish now with your summary of the "
                           "deliverables on disk.",
            })
        elif config.checkpoint_every and steps_left <= 5 and not wrapping_up:
            wrapping_up = True
            messages.append({
                "role": "user",
                "content": f"Only {steps_left} steps remain. Stop analysing. Write or update the deliverable "
                           "at the exact path the task names with your best current results, check it "
                           "opens, then call finish.",
            })
        elif config.checkpoint_every and not drafted and (step >= max(8, config.max_steps // 4) or used >= 0.25):
            drafted = True
            messages.append({
                "role": "user",
                "content": "A quarter of your budget is used. If no complete deliverable exists on disk yet, "
                           "write a first complete version now from what you know, even if rough, before "
                           "any further analysis. A missing file scores zero.",
            })
        elif config.checkpoint_every and step % config.checkpoint_every == 0:
            messages.append({
                "role": "user",
                "content": f"Checkpoint (step {step}): make sure the deliverable exists at the exact "
                           "path the task names and reflects your current best results. If it does "
                           "not, write or update it now, then continue improving it.",
            })
        if memory is not None:
            memory.manage(messages, config.compact_at_tokens)
        else:
            compact(messages, config.compact_at_tokens)
        call_started = time.monotonic()
        response = await model.complete(messages, tool_specs)
        meter["steps"] += 1
        meter["e2e_ms"].append(round((time.monotonic() - call_started) * 1000))
        response_id = getattr(response, "id", None)
        if isinstance(response_id, str) and response_id:
            meter["response_ids"].append(response_id)
        if response.usage is not None:
            details = getattr(response.usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
            prompt_tokens = response.usage.prompt_tokens or 0
            meter["prompt"] += prompt_tokens
            meter["output"] += response.usage.completion_tokens or 0
            meter["cached"] += cached
            meter["max_context"] = max(meter["max_context"], prompt_tokens)
            await session.usage(
                config.model,
                input_tokens=prompt_tokens,
                output_tokens=response.usage.completion_tokens or 0,
                cache_read_tokens=cached,
            )
        message = response.choices[0].message
        calls = message.tool_calls or text_tool_calls(message.content or "", tool_names)
        messages.append(_assistant_entry(message, calls))

        if not calls:
            idle_turns += 1
            meter["no_progress_turns"] += 1
            if idle_turns >= 3 or out_of_time:
                summary = message.content or summary
                break
            messages.append({"role": "user", "content": "Continue. Use a tool, or call finish."})
            continue
        idle_turns = 0

        finished = False
        for call in calls:
            try:
                args = repair_json(call.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise json.JSONDecodeError("arguments must be an object", "", 0)
            except json.JSONDecodeError:
                args = {}
                output = "error: tool arguments were not valid JSON"
            else:
                if call.function.name == "finish" and len(calls) > 1:
                    # Some models call finish in the same turn as other tools and describe
                    # results they have not seen yet. Make them look first.
                    output = ("not finished: you called finish in the same turn as other tools, before "
                              "seeing their results. Read the results, then call finish on its own.")
                elif call.function.name == "finish":
                    summary = str(args.get("summary", "")) or summary
                    finished = True
                    output = "ok"
                else:
                    output = await runner.run(call.function.name, args)
            await session.log(f"step {step}: {call.function.name} -> {output[:160]!r}")
            meter["by_name"][call.function.name] = meter["by_name"].get(call.function.name, 0) + 1
            if output.startswith("not finished:"):
                meter["finish_refusals"] += 1
            elif output.startswith("error:"):
                meter["errors"] += 1
            else:
                meter["successes"] += 1
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
            repeats.append(f"{call.function.name}|{call.function.arguments}|{hash(output)}")
        if len(repeats) >= 3 and len(set(repeats[-3:])) == 1 and not finished:
            repeats.clear()
            messages.append({
                "role": "user",
                "content": "You have repeated the same action three times with the same result. Stop. "
                           "State the root cause in one sentence, then take a different approach.",
            })
        slowest_step = max(slowest_step, time.monotonic() - step_started)
        if saved:
            saved.parent.mkdir(parents=True, exist_ok=True)
            tmp = saved.with_suffix(".tmp")
            tmp.write_text(json.dumps({"step": step, "messages": messages}))
            os.replace(tmp, saved)
        if finished:
            break
    if saved:
        saved.unlink(missing_ok=True)
    if telemetry is not None:
        attempts = getattr(model, "attempts", meter["steps"])
        telemetry.update(meter)
        telemetry["retries"] = max(0, attempts - meter["steps"])
        telemetry["uncached"] = meter["prompt"] - meter["cached"]
    return summary or "Stopped at the step or time limit. See the deliverables on disk."


def _assistant_entry(message: Any, calls: list[Any] | None = None) -> dict[str, Any]:
    """Converts an SDK message into a plain dict for the next request."""
    calls = calls if calls is not None else (message.tool_calls or [])
    content = message.content or ""
    if calls and not message.tool_calls:
        content = re.sub(r"(?s)<tool_call>.*", "", content).strip()  # calls were parsed from the text
    entry: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        entry["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in calls
        ]
    return entry


__all__ = ["Config", "SYSTEM_PROMPT", "TOOLS", "MCP_TOOLS", "compact", "run"]
