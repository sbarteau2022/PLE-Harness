"""Offline tests of the shared-mind wiring: automatic recall, extraction, minting, trajectory.

The database is replaced by a fake; the canonical address port is checked against a known atom.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import test_frontier as tf
from test_frontier import env  # noqa: F401  (pytest fixture)

from frontier import core, pipeline, surreal


class FakeMind:
    def __init__(self):
        self.atoms, self.edges, self.walks, self.minted, self.minted_trajectories = [], [], [], [], []

    def mint_atom(self, text, domain="general", salience=0.5, step=""):
        if text not in self.atoms:
            self.atoms.append(text)
        return f"atom:{self.atoms.index(text) + 1}"

    def relate(self, source, target, kind="part_of"):
        self.edges.append((source, kind, target))

    def mint_trajectory(self, atoms, receipt):
        self.walks.append(list(atoms))
        return f"distilled_trajectory:t{len(self.walks)}"


def test_recall_is_injected_and_work_is_extracted_and_minted(env, monkeypatch):  # noqa: F811
    mind = FakeMind()
    monkeypatch.setattr(surreal, "mind_enabled", lambda: True)
    monkeypatch.setattr(surreal, "mind_recall", lambda query, k=4: "atom:7 [S_research] Medians of even lists average the two middle values.")
    monkeypatch.setattr(pipeline, "open_mind", lambda *a, **k: mind)
    monkeypatch.setattr(pipeline, "MIN_EXTRACT_CHARS", 0)
    parts = tf.fin(json.dumps([{"text": "The median of an even-length list is the mean of its two middle values.",
                                "domain": "S_research", "salience": 0.7},
                               {"text": "too short"}]))
    none = tf.fin("no atoms")
    stages = (tf.STAGES[:3] + [parts] + tf.STAGES[3:5] + [none] + tf.STAGES[5:8] + [none] + tf.STAGES[8:] + [none])
    worker = [tf.call("write_file", {"path": "value.txt", "content": "score=0.9\n"}), tf.fin("raised the score")]
    script = env.start(stages + [tf.orders(tf.order("raise", "raise the score in value.txt"))] + worker + [tf.APPROVE, none])
    seen = []
    original = script.complete

    async def spy(messages, tools=None):
        seen.append((messages[1]["content"], [t["function"]["name"] for t in tools]))
        return await original(messages, tools)
    script.complete = spy
    env.opts.auto = True
    assert asyncio.run(pipeline.run(env.target, env.opts)) == "complete"

    assert "From the shared hyperbolic mind" in seen[0][0] and "atom:7" in seen[0][0]      # recall needs no tool call
    assert {"atom_search", "atom_neighbors", "mind_mint"} <= set(seen[0][1])               # and tools are there too
    state = core.load(env.target)
    labels = [a.split("]")[0] for a in mind.atoms if a.startswith("[frontier:")]
    assert [l.split(":")[-1] for l in labels] == ["research", "audit", "design", "preregistration", "order-raise", "complete"]
    assert "The median of an even-length list is the mean of its two middle values." in mind.atoms
    assert ("atom:2", "part_of", "atom:1") in mind.edges                                   # extracted atom under its stage
    assert len(mind.walks) == 2 and len(mind.walks[-1]) == len(state["mind"]["atoms"]) == 7
    assert state["mind"]["trajectories"] == ["distilled_trajectory:t1", "distilled_trajectory:t2"]


def test_canonical_address_matches_a_live_atom():
    # atom_address:1 in hypermind/core, salience 0.5: radius 1.35 on the hyperboloid.
    h3, t2 = surreal.canonical_address("any text at all", 0.5)
    assert abs(h3[0] ** 2 - sum(v * v for v in h3[1:]) - 1.0) < 1e-9
    assert abs(sum(v * v for v in h3[1:]) ** 0.5 - 1.35) < 1e-9
    assert surreal.canonical_tokens("Hello, World! Don't stop.") == ["hello", "world", "don", "t", "stop"]
    assert surreal.judge_text("ok")[0] is False and surreal.judge_text("A verified finding about medians.")[0]
    lorentz = surreal.poincare_to_lorentz((0.3, 0.2, 0.1))
    assert abs(lorentz[0] ** 2 - sum(v * v for v in lorentz[1:]) - 1.0) < 1e-9


def test_carving_keeps_calls_constant_size(tmp_path):
    """Completed native tool groups archive together and cumulative cost scales with steps."""
    from frontier import n3

    def total_sent(steps, manager):
        messages = [{"role": "system", "content": "s" * 2000}, {"role": "user", "content": "task " * 200}]
        sent = 0
        for i in range(steps):
            manager(messages)
            sent += n3._estimate(messages)
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": str(i), "type": "function", "function": {"name": "bash", "arguments": "{}"}}]})
            messages.append({"role": "tool", "tool_call_id": str(i), "content": f"out {i}\n" + "x" * 6000})
        return sent

    carved = pipeline.Carved(n3.N3(tmp_path / "m.sqlite"), 8000)
    plain = {n: total_sent(n, lambda m: None) for n in (40, 80)}
    kept = {n: total_sent(n, lambda m: carved.manage(m, 200000)) for n in (40, 80, 160)}
    assert plain[80] / plain[40] > 3.5            # doubling the steps roughly quadruples the cost
    assert kept[80] / kept[40] < 2.4              # carved: roughly doubles
    assert kept[160] / kept[80] < 2.3
    assert kept[80] < plain[80] / 5
    print("tokens sent, 80 steps: append-only", plain[80], "carved", kept[80])


def test_archive_preserves_multi_tool_pairs_and_pages_old_output(tmp_path):
    from frontier import n3
    memory = n3.N3(tmp_path / "archive.sqlite")
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
    for turn in range(24):
        ids = [f"{turn}-a", f"{turn}-b"]
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": "bash", "arguments": "{}"}}
            for cid in ids]})
        messages.extend({"role": "tool", "tool_call_id": cid,
                         "content": f"result {cid}: " + "x" * 3000} for cid in ids)
        memory.manage(messages, 6000, keep_recent=5)
        for index, message in enumerate(messages):
            if message.get("tool_calls"):
                expected = {call["id"] for call in message["tool_calls"]}
                actual = {m["tool_call_id"] for m in messages[index + 1:index + 1 + len(expected)]}
                assert actual == expected
    assert messages[3]["content"].startswith("[memory m")
    import re
    archive_id = int(re.search(r"memory m(\d+)", messages[3]["content"]).group(1))
    entry = memory.get(archive_id)
    assert entry["kind"] == "conversation_archive"
    handler = n3.make_handlers(memory, None)["memory_recall"]
    first = asyncio.run(handler({"id": archive_id, "chars": 300}))
    assert "next_start=300" in first and "conversation_archive" in first
    second = asyncio.run(handler({"id": archive_id, "start": 300, "chars": 300}))
    assert "chars 300:600" in second
    assert n3._estimate(messages) < 8500


def test_archive_handles_oversized_fixed_prompt_and_shrinks_recent_window(tmp_path):
    from frontier import n3
    memory = n3.N3(tmp_path / "budget.sqlite")
    fixed = [{"role": "system", "content": "s" * 40000},
             {"role": "user", "content": "task"}]
    memory.manage(fixed, 8000)
    assert len(fixed) == 3  # core view is added, but nothing fixed can be archived

    messages = [{"role": "system", "content": "s" * 1000},
                {"role": "user", "content": "task"}]
    for i in range(12):
        messages.extend([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": str(i), "type": "function",
                 "function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": str(i), "content": "x" * 9000},
        ])
    memory.manage(messages, 8000, keep_recent=10)
    assert n3._estimate(messages) <= 8000
    assert messages[3]["content"].startswith("[memory m")
    assert messages[-2]["tool_calls"][0]["id"] == "11"
    assert messages[-1]["tool_call_id"] == "11"


def test_finish_alongside_other_tools_is_refused(tmp_path, monkeypatch):
    import agent
    import test_agent as ta
    session = ta.FakeSession(tmp_path)
    script = ta.ScriptedModel([
        ta._reply(tool_calls=[ta._tool_call("c1", "bash", {"command": "echo hi"}),
                              ta._tool_call("c2", "finish", {"summary": "made up"})]),
        ta._reply(tool_calls=[ta._tool_call("c3", "finish", {"summary": "saw hi"})]),
    ])
    monkeypatch.setattr(agent, "ModelClient", lambda cfg: script)
    assert asyncio.run(agent.run(session, ta._config())) == "saw hi"


def test_trajectory_page_verifies_order_identity_and_content(monkeypatch):
    atom_ids = ["atom:10", "atom:11", "atom:12"]
    def read_sql(_ns, _db, query):
        assert query.endswith("FROM distilled_trajectory:walk1;")
        return [[{"id": "distilled_trajectory:walk1", "workspace": "workspace:hypermind",
                  "atoms": atom_ids, "trajectory_digest": surreal._sha("|".join(atom_ids))}]]
    monkeypatch.setattr(surreal, "sql", read_sql)
    monkeypatch.setattr(surreal, "atom_get", lambda aid: {
        "id": aid, "workspace": "workspace:hypermind", "domain": "S_workflow",
        "content_text": f"verified {aid}", "content_digest": surreal._sha(f"verified {aid}"),
        "content_uri": None})
    page = surreal.trajectory_get("distilled_trajectory:walk1", 1, 1)
    assert page["total"] == 3 and page["next_start"] == 2
    assert page["steps"] == [{"index": 1, "id": "atom:11", "domain": "S_workflow",
                              "text": "verified atom:11", "content_uri": None}]
    assert surreal.trajectory_get("distilled_trajectory:walk1", 3, 1)["steps"] == []
    with pytest.raises(surreal.SurrealError, match="invalid trajectory ID"):
        surreal.trajectory_get("distilled_trajectory:walk1; DELETE atom")
    with pytest.raises(surreal.SurrealError, match="page requires"):
        surreal.trajectory_get("distilled_trajectory:walk1", -1, 1)
    with pytest.raises(surreal.SurrealError, match="page requires"):
        surreal.trajectory_get("distilled_trajectory:walk1", 0, 17)
    monkeypatch.setattr(surreal, "atom_get", lambda aid: {
        "id": aid, "workspace": "workspace:hypermind", "content_text": "tampered",
        "content_digest": surreal._sha("original")})
    with pytest.raises(surreal.SurrealError, match="digest mismatch"):
        surreal.trajectory_get("distilled_trajectory:walk1")


def test_model_mints_only_validated_trajectory(monkeypatch):
    mind = FakeMind()
    atom = lambda aid: {"id": aid, "workspace": "workspace:hypermind",
                        "content_text": aid, "content_digest": surreal._sha(aid)}
    monkeypatch.setattr(surreal, "atom_get", atom)
    handler = surreal.make_handlers(mind)["mind_mint_trajectory"]
    args = {"atoms": ["atom:1", "atom:2"], "evidence": "Both steps were verified against the task output."}
    assert asyncio.run(handler(args)) == "minted distilled_trajectory:t1"
    assert mind.walks == [["atom:1", "atom:2"]]
    assert asyncio.run(handler({**args, "atoms": ["atom:1", "atom:missing"]})).startswith("error:")
    monkeypatch.setattr(surreal, "atom_get", lambda aid: None if aid == "atom:2" else atom(aid))
    assert "missing or foreign" in asyncio.run(handler(args))
    assert len(mind.walks) == 1


def test_short_term_memory_defaults_to_sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("FRONTIER_N3", raising=False)
    monkeypatch.setattr(surreal, "SurrealN3", lambda *a, **k: pytest.fail("unexpected Surreal memory"))
    memory = surreal.open_memory(tmp_path)
    assert isinstance(memory, surreal.n3.N3)
    entry = memory.deposit("A checked observation about the task", key="observation")
    assert memory.get(entry)["key"] == "observation"


def test_scope_must_name_real_paths(tmp_path):
    (tmp_path / "stats.py").write_text("x = 1\n")
    assert pipeline.scope_matches_folder(["stats.py", "notes/*"], tmp_path) == ""
    assert pipeline.scope_matches_folder(["*.py"], tmp_path) == ""
    assert "matches a file" in pipeline.scope_matches_folder(["src/features.py", "experiments/*"], tmp_path)


def test_orders_are_found_in_common_formats():
    item = {"id": "a", "objective": "fix", "allowed_paths": ["x.py"], "gates": [{"name": "t", "cmd": "true"}]}
    for summary in ("```orders\n" + json.dumps([item]) + "\n```",
                    "Plan:\n```json\n" + json.dumps([item]) + "\n```",
                    "Here: " + json.dumps([item]),
                    "```json\n" + json.dumps({"orders": [item]}) + "\n```",
                    "```json\n" + json.dumps(item) + "\n```"):
        assert pipeline.extract_orders(summary) == [item], summary
    assert pipeline.extract_orders("no plan here") is None


def test_carving_keeps_the_prefix_identical_between_carves(tmp_path):
    from frontier import n3
    carved = pipeline.Carved(n3.N3(tmp_path / "m.sqlite"), 4000, high=12000)
    messages = [{"role": "system", "content": "s" * 2000}, {"role": "user", "content": "task"}]
    previous, changed_prefix = None, 0
    for i in range(40):
        carved.manage(messages, 200000)
        text = json.dumps(messages)
        if previous is not None and not text.startswith(previous[:-1]):
            changed_prefix += 1
        previous = text
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": str(i), "type": "function", "function": {"name": "bash", "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": str(i), "content": "x" * 3000})
    assert carved.carves >= 2
    assert changed_prefix == carved.carves      # the prefix changes only when a carve happens
