"""Real agent dispatch with synthetic codecs; no trained-model support claim."""

import asyncio
from dataclasses import replace
import sys
import threading
import time
import types

import pytest

import agent
from frontier import drivers, pipeline, core, n3
from frontier.kv_assembly import continue_nested
from frontier.local_chat import NestedPrefix, PreparedChat
from test_kv_assembly import fixture, Mind as PageSource


@pytest.fixture
def setup(monkeypatch):
    who, prefix, roster, payloads = fixture()

    class Mind(PageSource):
        minted = []
        minted_trajectories = []

        def continue_tensor_cache(self, runtime, prefix, roster, suffix, limit, executable, *, recalled_addresses):
            return continue_nested(runtime, self, prefix, roster, suffix, limit, executable,
                                   recalled_addresses=recalled_addresses)

    mind = Mind(payloads)
    cache = NestedPrefix(who, prefix, tuple(roster), frozenset({"a1"}))

    class Runtime:
        plan = PreparedChat(prefix + (40,), cache)
        output = (50,)
        decoded = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "finish1", "type": "function", "function": {"name": "finish", "arguments": '{"summary":"cached result"}'}}]}
        current = who

        def __init__(self):
            self.calls, self.minds, self.threads = [], [], []

        def identity(self):
            return self.current

        def prepare_chat(self, messages, tools, shared_mind):
            self.calls.append("prepare")
            self.minds.append(shared_mind)
            self.threads.append(threading.get_ident())
            return self.plan

        def generate_tokens(self, prompt, limit):
            self.calls.append("cold")
            return self.output

        def continue_from_pages(self, pages, suffix, limit):
            self.calls.append("warm")
            assert pages.identity == who and pages.prefix_ids == prefix and suffix == (40,)
            return self.output

        def decode_chat(self, output):
            self.calls.append("decode")
            assert output == self.output
            return self.decoded

    raw = Runtime()
    module = types.ModuleType("test_chat_plugin")
    selected = []

    def create(**kwargs):
        selected.append(kwargs)
        return raw

    module.create = create
    monkeypatch.setitem(sys.modules, module.__name__, module)
    profile = {"model": "fixture-seat", "context": 32000, "max_output": 2,
               "local_runtime": {"factory": "test_chat_plugin:create", "model_digest": who.model_digest,
                                 "bridge": "bridge", "options": {"marker": "selected"}}}
    monkeypatch.setattr(drivers, "load_profiles", lambda: {"seat": profile})

    def no_http(*args, **kwargs):
        raise AssertionError("local driver silently attempted HTTP")

    monkeypatch.setattr(agent.openai, "AsyncOpenAI", no_http)
    return types.SimpleNamespace(raw=raw, mind=mind, who=who, prefix=prefix, cache=cache, profile=profile, selected=selected)


def test_driver_profile_reaches_agent_runtime_and_shared_nested_mind(setup, tmp_path):
    from test_agent import FakeSession
    config = drivers.resolve("seat", max_steps=2)
    session = FakeSession(tmp_path)
    main_thread = threading.get_ident()
    result = asyncio.run(agent.run(session, config, mind=setup.mind))
    assert result == "cached result"
    assert setup.raw.calls == ["prepare", "warm", "decode"]
    assert setup.raw.minds == [setup.mind] and setup.raw.threads[0] != main_thread
    assert setup.mind.calls == 8
    assert setup.selected == [{"model": "fixture-seat", "marker": "selected"}]
    assert session.usages == [("fixture-seat", 4, 1)]


def test_agent_reads_tool_result_before_warm_cached_finish(setup, tmp_path):
    from test_agent import FakeSession
    (tmp_path / "data.txt").write_text("42")
    prepare, decode = setup.raw.prepare_chat, setup.raw.decode_chat
    prompts, replies = [], []

    def prepare_turn(messages, tools, mind):
        if not prompts:
            setup.raw.plan = PreparedChat(setup.prefix)
        else:
            assert messages[-1]["role"] == "tool" and "42" in messages[-1]["content"]
            setup.raw.plan = PreparedChat(setup.prefix + (40,), setup.cache)
        prompts.append(messages)
        return prepare(messages, tools, mind)

    def decode_turn(output):
        result = decode(output)
        if not replies:
            result = {"role": "assistant", "tool_calls": [{"id": "read", "type": "function",
                      "function": {"name": "read_file", "arguments": '{"path":"data.txt"}'}}]}
        replies.append(result)
        return result

    setup.raw.prepare_chat, setup.raw.decode_chat = prepare_turn, decode_turn
    final = asyncio.run(agent.run(FakeSession(tmp_path), drivers.resolve("seat", max_steps=2), mind=setup.mind))
    assert final == "cached result" and len(prompts) == 2
    assert setup.raw.calls == ["prepare", "cold", "decode", "prepare", "warm", "decode"]


@pytest.mark.parametrize("plan", [None, {"prompt_ids": (1,)}, PreparedChat([1]), PreparedChat((True,))])
def test_bad_preparation_never_reaches_forward(setup, plan):
    setup.raw.plan = plan
    client = agent.ModelClient(drivers.resolve("seat"))
    with pytest.raises(ValueError):
        asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
    assert setup.raw.calls == ["prepare"]


@pytest.mark.parametrize("name", ["prepare_chat", "generate_tokens", "decode_chat"])
def test_incomplete_chat_plugin_fails_without_http_fallback(setup, name):
    setattr(setup.raw, name, None)
    with pytest.raises(ValueError, match=name):
        agent.ModelClient(drivers.resolve("seat"))
    assert setup.raw.calls == []


def test_cold_path_and_usage_are_local_without_cache_candidate(setup):
    setup.raw.plan = replace(setup.raw.plan, cache=None)
    client = agent.ModelClient(drivers.resolve("seat"))
    response = asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
    assert setup.raw.calls == ["prepare", "cold", "decode"]
    assert setup.mind.calls == 0
    assert response.usage.prompt_tokens == 4 and response.usage.prompt_tokens_details.cached_tokens == 0


def test_warm_path_reports_reused_tokens(setup):
    client = agent.ModelClient(drivers.resolve("seat"))
    client.bind_mind(setup.mind)
    response = asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
    assert response.usage.prompt_tokens == 4 and response.usage.prompt_tokens_details.cached_tokens == 3
    assert response.usage.total_tokens == 5


@pytest.mark.parametrize("change", ["prefix", "full-prefix", "identity", "addresses", "roster", "missing-mind", "missing-bridge", "untyped"])
def test_invalid_cache_candidate_never_reaches_forward_or_http(setup, change):
    cache = setup.cache
    if change == "prefix": cache = replace(cache, prefix_ids=(10, 20, 31))
    elif change == "full-prefix": cache = replace(cache, prefix_ids=setup.raw.plan.prompt_ids)
    elif change == "identity": cache = replace(cache, identity=replace(setup.who, adapter_digest="f" * 64))
    elif change == "addresses": cache = replace(cache, recalled_addresses=frozenset({"other"}))
    elif change == "roster": cache = replace(cache, witnesses=cache.witnesses[:-1])
    elif change == "missing-bridge": setup.profile["local_runtime"].pop("bridge")
    elif change == "untyped": cache = {}
    setup.raw.plan = replace(setup.raw.plan, cache=cache)
    client = agent.ModelClient(drivers.resolve("seat"))
    if change != "missing-mind": client.bind_mind(setup.mind)
    with pytest.raises(ValueError):
        asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
    assert setup.raw.calls == ["prepare"] and setup.mind.calls == 0


@pytest.mark.parametrize("output", [(), (True,), (2**32,), (1, 2, 3), [1]])
def test_bad_cold_runtime_output_never_reaches_decoder(setup, output):
    setup.raw.plan = replace(setup.raw.plan, cache=None)
    setup.raw.output = output
    client = agent.ModelClient(drivers.resolve("seat"))
    with pytest.raises(ValueError):
        asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
    assert setup.raw.calls == ["prepare", "cold"]


@pytest.mark.parametrize("decoded", [[1], {"role": "user", "content": "bad"},
                                      {"role": "assistant", "tool_calls": [{"type": "unknown"}]}])
def test_invalid_decoder_message_fails_locally(setup, decoded):
    setup.raw.plan = replace(setup.raw.plan, cache=None)
    setup.raw.decoded = decoded
    client = agent.ModelClient(drivers.resolve("seat"))
    with pytest.raises(ValueError):
        asyncio.run(client.complete([{"role": "user", "content": "hi"}]))


def test_local_profile_has_safe_default_output_and_no_network_probe(setup, monkeypatch):
    setup.profile.pop("max_output")
    monkeypatch.setattr(drivers, "probe_context", lambda *args: pytest.fail("local driver probed HTTP"))
    assert drivers.resolve("seat").max_output_tokens == 4096


def test_text_tool_protocol_still_works_with_runtime_owned_tokenization(setup, tmp_path):
    from test_agent import FakeSession
    setup.raw.plan = replace(setup.raw.plan, cache=None)
    setup.raw.decoded = {"role": "assistant", "content":
                         '<tool_call>{"name":"finish","arguments":{"summary":"text result"}}</tool_call>'}
    prepare = setup.raw.prepare_chat
    seen = []

    def inspect(messages, tools, mind):
        seen.append((messages, tools))
        return prepare(messages, tools, mind)

    setup.raw.prepare_chat = inspect
    config = replace(drivers.resolve("seat", max_steps=1), tool_mode="text")
    assert asyncio.run(agent.run(FakeSession(tmp_path), config)) == "text result"
    assert seen[0][1] == []
    assert "<tool_call>" in str(seen[0][0])


@pytest.mark.parametrize("stage", ["prepare_chat", "generate_tokens", "decode_chat"])
def test_identity_changes_inside_plugin_stages_are_rejected(setup, stage):
    setup.raw.plan = replace(setup.raw.plan, cache=None)
    original = getattr(setup.raw, stage)

    def switch(*args):
        result = original(*args)
        setup.raw.current = replace(setup.who, adapter_digest="f" * 64)
        return result

    setattr(setup.raw, stage, switch)
    client = agent.ModelClient(drivers.resolve("seat"))
    with pytest.raises(ValueError, match="identity changed"):
        asyncio.run(client.complete([{"role": "user", "content": "hi"}]))


def test_one_client_serializes_concurrent_plugin_calls(setup):
    setup.raw.plan = replace(setup.raw.plan, cache=None)
    prepare, decode = setup.raw.prepare_chat, setup.raw.decode_chat
    active = []

    def start(*args):
        assert not active
        active.append(True)
        time.sleep(0.02)
        return prepare(*args)

    def finish(*args):
        result = decode(*args)
        active.clear()
        return result

    setup.raw.prepare_chat, setup.raw.decode_chat = start, finish
    client = agent.ModelClient(drivers.resolve("seat"))

    async def parallel():
        return await asyncio.gather(*(client.complete([{"role": "user", "content": "hi"}]) for _ in range(2)))

    assert len(asyncio.run(parallel())) == 2
    assert setup.raw.calls == ["prepare", "cold", "decode"] * 2


def test_pipeline_passes_its_existing_mind_to_local_dispatch(setup, tmp_path, monkeypatch):
    target = tmp_path / "project"
    target.mkdir()
    state = {"goal": "test local dispatch", "rev": 1}
    monkeypatch.setattr(pipeline, "open_mind", lambda *args, **kwargs: setup.mind)
    monkeypatch.setattr(n3, "Rag", lambda: types.SimpleNamespace())
    config = drivers.resolve("seat", max_steps=1)
    run = asyncio.run(pipeline.run_agent(target, state, "extract", "test", config, 1, "local"))
    assert run.summary == "cached result"
    assert setup.raw.minds == [setup.mind] and setup.raw.calls == ["prepare", "warm", "decode"]
    assert state["budget"]["tokens_in"] == 4
    assert state["budget"]["tokens_cached"] == 3
