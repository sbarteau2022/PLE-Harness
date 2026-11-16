"""Offline tests: scripted model replies and a fake task container. Run: .venv/bin/python -m pytest tests -q"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import agent  # noqa: E402


def _tool_call(call_id, name, args):
    return types.SimpleNamespace(
        id=call_id, function=types.SimpleNamespace(name=name, arguments=json.dumps(args))
    )


def _reply(content=None, tool_calls=None):
    msg = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)


class ScriptedModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    async def complete(self, messages, tools=None):
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0)


class FakeSession:
    """Mimics taskbridge.Session over a local temp directory."""

    def __init__(self, root, time_left=None):
        self.root = pathlib.Path(root)
        self.workdir = str(self.root)
        self.instruction = "Write out/result.txt containing 42."
        self.mcp_servers = []
        self._left = time_left
        self.logs, self.usages, self.final = [], [], None

    def time_left(self):
        return self._left

    async def exec(self, command, timeout_sec=120):
        import subprocess

        proc = subprocess.run(["bash", "-c", command], cwd=self.root, capture_output=True, text=True)
        return {"exit_code": proc.returncode, "output": proc.stdout + proc.stderr,
                "truncated": False, "timed_out": False}

    async def read_file(self, path):
        return (self.root / path).read_bytes()

    async def write_file(self, path, data, append=False):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = data.encode() if isinstance(data, str) else data
        target.write_bytes(payload)
        return len(payload)

    async def log(self, text):
        self.logs.append(text)

    async def usage(self, model, input_tokens=0, output_tokens=0, cache_read_tokens=0):
        self.usages.append((model, input_tokens, output_tokens))


def _config(**kw):
    base = dict(model="test-model", base_url=None, api_key="none", max_steps=20)
    base.update(kw)
    return agent.Config(**base)


def test_bash_write_then_finish(tmp_path, monkeypatch):
    session = FakeSession(tmp_path)
    script = ScriptedModel([
        _reply(tool_calls=[_tool_call("c1", "bash", {"command": "mkdir -p out && echo hi > out/a.txt && cat out/a.txt"})]),
        _reply(tool_calls=[_tool_call("c2", "write_file", {"path": "out/result.txt", "content": "42"})]),
        _reply(tool_calls=[_tool_call("c3", "finish", {"summary": "wrote out/result.txt"})]),
    ])
    monkeypatch.setattr(agent, "ModelClient", lambda cfg: script)
    final = asyncio.run(agent.run(session, _config()))
    assert final == "wrote out/result.txt"
    assert (tmp_path / "out/result.txt").read_text() == "42"
    assert "hi" in script.seen[1][-1]["content"]
    assert session.usages == [("test-model", 10, 5)] * 3


def test_text_only_replies_nudge_then_stop(tmp_path, monkeypatch):
    session = FakeSession(tmp_path)
    script = ScriptedModel([_reply(content="thinking"), _reply(content="still thinking"), _reply(content="done")])
    monkeypatch.setattr(agent, "ModelClient", lambda cfg: script)
    final = asyncio.run(agent.run(session, _config()))
    assert final == "done"
    assert script.seen[1][-1]["content"].startswith("Continue.")


def test_time_nearly_up_requests_finish(tmp_path, monkeypatch):
    session = FakeSession(tmp_path, time_left=10)
    script = ScriptedModel([_reply(tool_calls=[_tool_call("c1", "finish", {"summary": "partial"})])])
    monkeypatch.setattr(agent, "ModelClient", lambda cfg: script)
    assert asyncio.run(agent.run(session, _config(reserve_seconds=240))) == "partial"
    assert "Time is nearly up" in script.seen[0][-1]["content"]


def test_bad_arguments_and_unknown_tool_are_reported(tmp_path, monkeypatch):
    session = FakeSession(tmp_path)
    bad = types.SimpleNamespace(id="c1", function=types.SimpleNamespace(name="bash", arguments="{not json"))
    script = ScriptedModel([
        _reply(tool_calls=[bad]),
        _reply(tool_calls=[_tool_call("c2", "nope", {})]),
        _reply(tool_calls=[_tool_call("c3", "finish", {"summary": "ok"})]),
    ])
    monkeypatch.setattr(agent, "ModelClient", lambda cfg: script)
    asyncio.run(agent.run(session, _config()))
    tool_msgs = [m["content"] for m in script.seen[2] if m["role"] == "tool"]
    assert tool_msgs[0].startswith("error: tool arguments")
    assert tool_msgs[1] == "error: unknown tool nope"


def test_compact_elides_old_tool_output():
    big = "x" * 5000
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for i in range(30):
        messages.append({"role": "tool", "tool_call_id": str(i), "content": big})
    agent.compact(messages, budget=2000, keep_recent=12)
    assert messages[2]["content"].startswith("[earlier tool output elided")
    assert messages[-1]["content"] == big


def test_clip_keeps_head_and_tail():
    text = "A" * 100 + "B" * 100
    clipped = agent._clip(text, 50)
    assert clipped.startswith("A" * 25) and clipped.endswith("B" * 25)
    assert "truncated" in clipped


def test_checkpoint_reminder(tmp_path, monkeypatch):
    session = FakeSession(tmp_path)
    replies = [_reply(tool_calls=[_tool_call(f"c{i}", "bash", {"command": "true"})]) for i in range(2)]
    replies.append(_reply(tool_calls=[_tool_call("c9", "finish", {"summary": "ok"})]))
    script = ScriptedModel(replies)
    monkeypatch.setattr(agent, "ModelClient", lambda cfg: script)
    asyncio.run(agent.run(session, _config(checkpoint_every=3)))
    assert script.seen[2][-1]["content"].startswith("Checkpoint (step 3)")


def test_text_tool_calls_and_json_repair(tmp_path, monkeypatch):
    session = FakeSession(tmp_path)
    script = ScriptedModel([
        _reply(content='I will write it.\n<tool_call>{"name": "write_file", "arguments": {"path": "a.txt", "content": "hi",}}</tool_call>'),
        _reply(content="```json\n{'name': 'finish', 'arguments': {'summary': 'done via text'}}\n```"),
    ])
    monkeypatch.setattr(agent, "ModelClient", lambda cfg: script)
    assert asyncio.run(agent.run(session, _config())) == "done via text"
    assert (tmp_path / "a.txt").read_text() == "hi"
    assert script.seen[1][-2]["tool_calls"][0]["function"]["name"] == "write_file"


def test_to_text_messages_merges_and_describes_tools():
    messages = [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "a.txt"}, {"role": "user", "content": "Continue."},
    ]
    out = agent.to_text_messages(messages, agent.TOOLS)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user"]
    assert "<tool_call>" in out[2]["content"] and "bash" in out[0]["content"]
    assert "a.txt" in out[3]["content"] and "Continue." in out[3]["content"]


def test_transcript_resume(tmp_path, monkeypatch):
    session = FakeSession(tmp_path)
    path = tmp_path / "t.json"

    class Crash(Exception):
        pass

    class Crashing(ScriptedModel):
        async def complete(self, messages, tools=None):
            if not self.replies:
                raise Crash()
            return await super().complete(messages, tools)

    script = Crashing([_reply(tool_calls=[_tool_call("c1", "bash", {"command": "echo one > one.txt"})])])
    monkeypatch.setattr(agent, "ModelClient", lambda cfg: script)
    try:
        asyncio.run(agent.run(session, _config(), transcript=path))
    except Crash:
        pass
    assert path.exists()
    script2 = ScriptedModel([_reply(tool_calls=[_tool_call("c2", "finish", {"summary": "resumed"})])])
    monkeypatch.setattr(agent, "ModelClient", lambda cfg: script2)
    assert asyncio.run(agent.run(session, _config(), transcript=path)) == "resumed"
    assert any(m.get("tool_call_id") == "c1" for m in script2.seen[0]) and not path.exists()


def test_draft_and_wrap_up_reminders(tmp_path, monkeypatch):
    session = FakeSession(tmp_path)
    replies = [_reply(tool_calls=[_tool_call(f"c{i}", "bash", {"command": "true"})]) for i in range(11)]
    replies.append(_reply(tool_calls=[_tool_call("c99", "finish", {"summary": "ok"})]))
    script = ScriptedModel(replies)
    monkeypatch.setattr(agent, "ModelClient", lambda cfg: script)
    asyncio.run(agent.run(session, _config(max_steps=12, checkpoint_every=25)))
    users = [m["content"] for m in script.seen[-1] if m["role"] == "user"]
    assert any(u.startswith("A quarter of your budget") for u in users)
    assert any(u.startswith("Only 5 steps remain") for u in users)


def test_telemetry_records_counts_and_tokens(tmp_path, monkeypatch):
    session = FakeSession(tmp_path)
    script = ScriptedModel([
        _reply(tool_calls=[_tool_call("c1", "bash", {"command": "echo ok"})]),
        _reply(tool_calls=[_tool_call("c2", "finish", {"summary": "done"})]),
    ])
    monkeypatch.setattr(agent, "ModelClient", lambda cfg: script)
    meter: dict = {}
    assert asyncio.run(agent.run(session, _config(), telemetry=meter)) == "done"
    assert meter["steps"] == 2 and meter["prompt"] == 20 and meter["output"] == 10
    assert meter["uncached"] == 20 and meter["max_context"] == 10
    assert meter["by_name"] == {"bash": 1, "finish": 1}
    assert meter["successes"] == 2 and meter["errors"] == 0 and meter["retries"] == 0
    assert len(meter["e2e_ms"]) == 2
