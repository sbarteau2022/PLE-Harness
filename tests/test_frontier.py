"""Offline tests of the frontier pipeline: scripted models, temp folders, real gates. No network."""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import types

import pytest
from qdrant_edge import (Distance, EdgeConfig, EdgeShard, EdgeVectorParams, PayloadSchemaType,
                         Point, TextIndexParams, UpdateOperation)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import agent  # noqa: E402
import serve  # noqa: E402
from frontier import core, loop, n3, pipeline, research, skills  # noqa: E402

GOAL = "Raise the score printed by check.py to at least 0.5 by improving value.txt."
NOTES = "Findings: https://a.example/1 https://b.example/2 https://c.example/3 say X."
MATH = skills.MATH_SKILL
PREREG = ("# Preregistration\n" + "Hypothesis and success criteria. " * 10 +
          '\n```scope\n{"allowed_paths": ["value.txt", "notes/*"]}\n```\n'
          '\n```gates\n[{"name": "score", "cmd": "python3 check.py", "metric": "score=([0-9.]+)", "min": 0.5}]\n```\n')


def call(name, args):
    c = types.SimpleNamespace(id="c", function=types.SimpleNamespace(name=name, arguments=json.dumps(args)))
    msg = types.SimpleNamespace(content=None, tool_calls=[c])
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)],
                                 usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=2))


def fin(text):
    return call("finish", {"summary": text})


def orders(*items):
    return fin("Plan.\n```orders\n" + json.dumps(list(items)) + "\n```")


def order(oid, objective, paths=("value.txt",)):
    return {"id": oid, "objective": objective, "allowed_paths": list(paths),
            "gates": [{"name": "has-score", "cmd": "grep -q score= value.txt"}], "context": "value.txt holds the score"}


STAGES = [
    call("preprint_search", {"query": "x"}),
    call("rag_ingest", {"url": "https://a.example/1", "title": "A paper", "text": "verified passage " * 30}),
    fin(NOTES),
    call("bash", {"command": "ls"}), fin("Audit: value.txt is low; nothing else tried."),
    call("skill", {"name": MATH}),
    call("write_file", {"path": "frontier-loop/DESIGN.md", "content": "# Design\n" + "Plan. " * 60}), fin("design written"),
    call("skill", {"name": MATH}),
    call("write_file", {"path": "frontier-loop/PREREGISTRATION.md", "content": PREREG}), fin("preregistered"),
]
APPROVE = fin('{"approve": true, "reason": "meets the contract"}')


class Scripted:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    async def complete(self, messages, tools=None):
        self.prompts.append(messages[0]["content"][:60])
        assert self.replies, "the script ran out of replies"
        return self.replies.pop(0)


class FakeRag:
    collection = "fake"
    stored = []

    def ingest(self, url, title, text, year="", note=""):
        FakeRag.stored.append(url)
        return 1

    def search(self, query, k=5):
        return []


@pytest.fixture
def env(tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir()
    (target / "value.txt").write_text("score=0.1\n")
    (target / "keep.txt").write_text("owner file\n")
    (target / "check.py").write_text("print(open('value.txt').read())\n")
    skill_dir = tmp_path / "skills"
    for name in (skills.RESEARCH_SKILL, MATH):
        (skill_dir / name).mkdir(parents=True)
        (skill_dir / name / "SKILL.md").write_text(f"---\nname: {name}\n---\n# {name}\nDo it properly.\n")
    monkeypatch.setattr(skills, "SKILL_DIRS", [skill_dir])
    monkeypatch.setattr(n3, "Rag", FakeRag)
    monkeypatch.setattr(core, "SANDBOX", False)
    monkeypatch.setenv("FRONTIER_QUIET", "1")
    FakeRag.stored = []

    async def fake_search(args):
        return "A paper\nhttps://a.example/1\nabstract"
    monkeypatch.setitem(research.HANDLERS, "preprint_search", fake_search)

    def start(replies, max_rounds=6):
        script = Scripted(replies)
        monkeypatch.setattr(agent, "ModelClient", lambda cfg: script)
        loop.set_goal(target, GOAL, max_rounds)
        return script

    config = agent.Config(model="m", base_url="http://x/v1", api_key="none", max_steps=12,
                          compact_at_tokens=100000, checkpoint_every=0)
    opts = pipeline.Options(driver=config, worker=config, reviewer=config, round_minutes=5)
    return types.SimpleNamespace(target=target, start=start, opts=opts)


def test_stages_need_approval_then_brigade_completes(env):
    worker = [call("write_file", {"path": "value.txt", "content": "score=0.9\n"}), fin("raised the score")]
    env.start(STAGES + [orders(order("raise", "raise the score in value.txt"))] + worker + [APPROVE])
    for stage in pipeline.STAGES:
        assert asyncio.run(pipeline.step(env.target, env.opts)) == "pending"
        assert asyncio.run(pipeline.step(env.target, env.opts)) == "pending"  # waits; does not run ahead
        assert core.load(env.target)["pending"]["stage"] == stage
        assert pipeline.approve(env.target, core.load(env.target)) == "approved"
    assert asyncio.run(pipeline.run(env.target, env.opts)) == "complete"
    state = core.load(env.target)
    receipt = state["receipts"][0]
    assert receipt["verdict"] == "kept" and receipt["review"]["approve"] and receipt["changed"] == ["value.txt"]
    assert len(receipt["contract_sha1"]) == 40
    assert (env.target / "value.txt").read_text() == "score=0.9\n"
    assert not (env.target / "AGENTS.md").exists() and not (env.target / ".git").exists()
    assert FakeRag.stored == ["https://a.example/1"]
    assert state["budget"]["tokens_in"] > 0 and "kept" in pipeline.report(env.target)
    names = sorted(p.name for p in core.loop_dir(env.target).iterdir() if not p.name.startswith("."))
    assert names == ["DESIGN.md", "PREREGISTRATION.md", "n3.sqlite", "results.json"]


def test_void_rollback_no_repeat_then_redesign(env):
    replies = STAGES + [
        # cycle 1: the subagent edits a file outside its order -> void
        orders(order("o1", "raise the score")),
        call("bash", {"command": "echo hacked > keep.txt"}), fin("edited something else"),
        # cycle 2: re-issuing the failed order is refused; the new one lowers the metric -> rolled back
        orders(order("o1b", "raise the score")),
        orders(order("o2", "rewrite value.txt from scratch")),
        call("write_file", {"path": "value.txt", "content": "score=0.05\n"}), fin("rewrote"), APPROVE,
        # two strikes -> redesign, then cycle 3 succeeds
        fin("Root cause: orders were vague. New approach: set the value directly and verify."),
        orders(order("o3", "set the score to 0.9 and verify with check.py")),
        call("write_file", {"path": "value.txt", "content": "score=0.9\n"}), fin("done"), APPROVE,
    ]
    script = env.start(replies)
    env.opts.auto = True
    assert asyncio.run(pipeline.run(env.target, env.opts)) == "complete"
    state = core.load(env.target)
    verdicts = [r["verdict"].split(":")[0] for r in state["receipts"]]
    assert verdicts == ["void", "rolled back", "kept"]
    assert (env.target / "keep.txt").read_text() == "owner file\n"
    assert len(state["redesigns"]) == 1 and "Amendment 1" in core.read_doc(env.target, "DESIGN.md")
    assert len(state["lessons"]) >= 2 and not script.replies


def test_research_must_cite_and_ingest(env):
    env.start([fin("I know this already.")] * 3)
    assert asyncio.run(pipeline.step(env.target, env.opts)) == "blocked"
    state = core.load(env.target)
    assert "research" in state["blocked_reason"] and state["stages"] == [] and len(state["attempts"]) == 3


def test_design_requires_math_skill_and_revise(env):
    no_skill = [call("write_file", {"path": "frontier-loop/DESIGN.md", "content": "# Design\n" + "Plan. " * 60}), fin("d")]
    env.start(STAGES[:5] + no_skill + STAGES[5:8] + STAGES[5:8])
    env.opts.auto = True
    for _ in range(2):
        asyncio.run(pipeline.step(env.target, env.opts))
    env.opts.auto = False
    assert asyncio.run(pipeline.step(env.target, env.opts)) == "pending"   # first design attempt rejected, second accepted
    attempts = [a for a in core.load(env.target)["attempts"] if a["stage"] == "design"]
    assert "math skill" in attempts[0]["problem"] and attempts[1]["ok"]
    assert asyncio.run(pipeline.step(env.target, env.opts, feedback="add risks")) == "pending"
    assert len([a for a in core.load(env.target)["attempts"] if a["stage"] == "design"]) == 3


def test_preregistration_rules(env):
    env.start([])
    state = core.load(env.target)
    doc = core.loop_dir(env.target) / "PREREGISTRATION.md"
    doc.write_text(PREREG.replace('"value.txt", "notes/*"', '"check.py"'))
    assert "inside the scope" in pipeline.read_preregistration(state, env.target)[2]
    doc.write_text(PREREG.replace('"min": 0.5', '"holdout": true, "min": 0.5'))
    assert "holdout" in pipeline.read_preregistration(state, env.target)[2]
    doc.write_text(PREREG.replace('["value.txt", "notes/*"]', '["*"]'))
    assert "specific paths" in pipeline.read_preregistration(state, env.target)[2]
    doc.write_text(PREREG)
    assert pipeline.read_preregistration(state, env.target)[2] == ""


def test_preregistration_rejects_goal_file_outside_scope(env):
    env.start([])
    state = core.load(env.target)
    doc = core.loop_dir(env.target) / "PREREGISTRATION.md"
    doc.write_text(PREREG.replace('"value.txt", "notes/*"', '"src/features.py", "experiments/*"'))
    problem = pipeline.read_preregistration(state, env.target)[2]
    assert "value.txt is outside allowed_paths" in problem
    state["goal"] = "Make median() in stats.py correct. Do not edit check.py."
    (env.target / "stats.py").write_text("def median(values): return values[0]\n")
    problem = pipeline.read_preregistration(state, env.target)[2]
    assert "stats.py is outside allowed_paths" in problem


def test_noise_and_holdout_decisions():
    def gate(name, passed, value, holdout=False):
        return {"name": name, "passed": passed, "value": value, "holdout": holdout, "min": 0.95,
                "exit_code": 0, "tail": ""}
    before = [gate("auc", False, 0.90), gate("hold", True, 0.9, True)]
    within = [gate("auc", False, 0.895), gate("hold", False, 0.1, True)]
    assert core.regressed(before, within, {"auc": 0.01}) == ""          # inside noise; holdout never decides
    assert "fell" in core.regressed(before, within, {})
    assert "hold" not in core.gate_report(before)


def test_memory_store_and_addressing(tmp_path):
    memory = n3.N3(tmp_path / "m.sqlite")
    root = memory.deposit("Gradient boosting baseline reached AUC 0.94 on fraud data", key="baseline", pin=True)
    child = memory.deposit("Target encoding leaked labels across folds", key="leakage", parent=root)
    memory.deposit("Colima needs eight CPUs", key="docker")
    assert [e["id"] for e in memory.recall("fraud AUC leakage")][:2] == [root, child]
    point = lambda i: memory.db.execute("SELECT x, y, z FROM mem WHERE id=?", (i,)).fetchone()
    radius = lambda p: sum(v * v for v in p) ** 0.5
    assert radius(point(root)) < radius(point(child)) < 1.0            # children sit further out in the ball
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    for i in range(20):
        messages += [
            {"role": "assistant", "content": "", "tool_calls": [{"id": str(i), "type": "function",
                "function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": str(i), "content": f"out {i}\n" + "x" * 3000}]
    memory.manage(messages, 4000)
    assert messages[2]["content"].startswith("[core memory]") and "baseline" in messages[2]["content"]
    stub = messages[3]["content"]
    assert stub.startswith("[memory m")
    import re
    stored = memory.get(int(re.search(r"memory m(\d+)", stub).group(1)))
    archived = json.loads(stored["text"])
    assert stored["kind"] == "conversation_archive"
    assert archived[1]["content"].startswith("out 0") and len(stored["text"]) > 3000


def test_sqlite_hyperbolic_depth_and_invalid_parent(tmp_path):
    import math
    memory = n3.N3(tmp_path / "depth.sqlite")
    parent = memory.deposit("A verified parent observation", key="parent")
    with pytest.raises(ValueError, match="does not exist"):
        memory.deposit("An orphan observation", parent=parent + 100)
    current = parent
    radii = []
    for depth in range(1, 25):
        row = memory.db.execute("SELECT x, y, z FROM mem WHERE id=?", (current,)).fetchone()
        radii.append(math.sqrt(sum(x * x for x in row)))
        current = memory.deposit(f"Verified child at depth {depth}", parent=current)
    assert all(0 < a < b < 1 for a, b in zip(radii, radii[1:]))
    assert radii[8] > 0.999  # the old clip merged all deeper descendants
    a, b = n3.place("topic one", 3), n3.place("topic two", 5)
    reference = math.acosh(1 + 2 * sum((x - y) ** 2 for x, y in zip(a, b)) /
                           ((1 - sum(x * x for x in a)) * (1 - sum(y * y for y in b))))
    assert math.isclose(n3.hyperbolic_distance(a, b), reference, rel_tol=1e-12)
    assert math.isfinite(n3.hyperbolic_distance((0.999999, 0, 0), (0, 0, 0)))
    with pytest.raises(ValueError, match="open ball"):
        n3.hyperbolic_distance((1, 0, 0), (0, 0, 0))


def test_sqlite_core_view_and_no_fts_fallback_are_bounded(tmp_path):
    memory = n3.N3(tmp_path / "bounded.sqlite")
    for i in range(300):
        memory.deposit(f"observation {i} about safety", key=f"note-{i}", pin=True)
    assert len(memory.core(limit=20)) == 20
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    memory.manage(messages, 1000)
    assert "note-299" in messages[2]["content"] and "note-0" not in messages[2]["content"]
    assert memory.get(1)["key"] == "note-0"  # old pinned content remains retrievable
    memory.fts = False
    assert memory.recall("note-299")[0]["key"] == "note-299"
    assert memory.recall("note-0") == []  # degraded mode scans only the recent 256 rows


def _edge_rag(tmp_path, index="scored"):
    path = tmp_path / "edge"
    shard = EdgeShard.create(str(path), EdgeConfig(
        vectors={"text_rank0": EdgeVectorParams(1024, Distance.Cosine)}))
    if index == "scored":
        shard.update(UpdateOperation.create_field_index("text", TextIndexParams(scoring=True)))
    elif index == "keyword":
        shard.update(UpdateOperation.create_field_index("text", PayloadSchemaType.Keyword))
    shard.update(UpdateOperation.create_field_index("source_path", PayloadSchemaType.Keyword))
    shard.close()
    return n3.Rag(path)


def test_rag_search_uses_local_edge_text_index(tmp_path):
    rag = _edge_rag(tmp_path)
    shard = EdgeShard.load(str(rag.shard_path))
    shard.update(UpdateOperation.upsert_points([Point(1, {}, {
        "document_id": "paper-1", "source_path": "/papers/one.pdf",
        "text": "hyperbolic memory matched passage",
    })]))
    shard.close()
    hits = rag.search("hyperbolic memory", 3)
    assert len(hits) == 1
    assert hits[0]["source"] == "/papers/one.pdf"
    assert hits[0]["title"] == "paper-1"
    assert hits[0]["text"] == "hyperbolic memory matched passage"
    assert hits[0]["score"] > 0
    assert hits[0]["from"] == "research_library_v1"
    assert rag.open_document("/papers/one.pdf") == [{
        "index": 0, "title": "paper-1", "text": "hyperbolic memory matched passage"}]


def test_rag_ingest_adds_payload_only_points_to_local_edge(tmp_path):
    rag = _edge_rag(tmp_path)
    assert rag.ingest("https://example.test/paper", "Verified paper", "evidence " * 200) == 1
    shard = EdgeShard.load(str(rag.shard_path))
    record = shard.retrieve([n3.research_point_id("https://example.test/paper", 0)], True, True)[0]
    shard.close()
    assert record.vector in ({}, None)
    assert record.payload["verified"] is True
    assert record.payload["retrieval_mode"] == "qdrant-edge-bm25-text-v1"
    assert record.payload["silo"] == "frontier" and len(record.payload["content_hash"]) == 64
    assert rag.search("evidence")[0]["title"] == "Verified paper"
    assert rag.open_document("https://example.test/paper")[0]["text"].startswith("evidence")


def test_rag_reingest_removes_surplus_chunks_without_touching_dense_points(tmp_path):
    rag = _edge_rag(tmp_path)
    url = "https://example.test/revision"
    assert rag.ingest(url, "Revision", "neutronflux " * 800) > 1
    assert rag.ingest(url, "Revision", "neutronflux " * 20) == 1
    assert len(rag.open_document(url, 0, 12)) == 1
    shard = EdgeShard.load(str(rag.shard_path))
    try:
        assert shard.retrieve([n3.research_point_id(url, 1)], True, False) == []
        dense_url = "https://example.test/dense"
        dense_id = n3.research_point_id(dense_url, 0)
        shard.update(UpdateOperation.upsert_points([Point(
            dense_id, {"text_rank0": [1.0] + [0.0] * 1023}, {"text": "dense source"})]))
    finally:
        shard.close()
    with pytest.raises(ValueError, match="dense vectors"):
        rag.ingest(dense_url, "Dense", "new source " * 30)
    shard = EdgeShard.load(str(rag.shard_path))
    try:
        assert len(shard.retrieve([dense_id], False, True)[0].vector["text_rank0"]) == 1024
    finally:
        shard.close()


def test_rag_requires_existing_scored_text_index(tmp_path):
    rag = _edge_rag(tmp_path, index="none")
    assert "scored text index" in rag.search("query")[0]["text"]
    rag = n3.Rag(tmp_path / "missing")
    assert "missing" in rag.search("query")[0]["text"]


def test_rag_rejects_incompatible_index_and_bounds_queries(tmp_path):
    rag = _edge_rag(tmp_path, index="keyword")
    assert rag.search("query")[0]["from"] == "error"
    assert rag.search("   ") == []
    assert "4096" in rag.search("x" * 4097)[0]["text"]


def test_ple_serve_exposes_edge_read_tools_only_when_shard_exists(tmp_path, monkeypatch):
    rag = _edge_rag(tmp_path)
    shard = EdgeShard.load(str(rag.shard_path))
    shard.update(UpdateOperation.upsert_points([Point(1, {}, {
        "source_path": "/archived/paper.pdf", "document_id": "paper",
        "chunk_index": 0, "text": "hyperbolic geometry passage",
    })]))
    shard.close()
    monkeypatch.setattr(n3, "EDGE_SHARD", rag.shard_path)
    session = types.SimpleNamespace(mcp_servers=[])

    async def fake_run(_session, _config, tools=None, handlers=None, telemetry=None):
        names = {tool["function"]["name"] for tool in tools}
        assert {"rag_search", "rag_open"}.issubset(names)
        assert "rag_ingest" not in names
        assert "source key:" in await handlers["rag_search"]({"query": "hyperbolic geometry"})
        assert "hyperbolic geometry" in await handlers["rag_open"]({"source": "/archived/paper.pdf"})
        return "wired"

    monkeypatch.setattr(agent, "run", fake_run)
    assert asyncio.run(serve.solve(session)) == "wired"
    monkeypatch.setattr(n3, "EDGE_SHARD", tmp_path / "missing")

    async def no_rag(_session, _config, tools=None, handlers=None, telemetry=None):
        assert not ({"rag_search", "rag_open"} &
                    {tool["function"]["name"] for tool in tools})
        assert handlers == {}
        return "portable"

    monkeypatch.setattr(agent, "run", no_rag)
    assert asyncio.run(serve.solve(session)) == "portable"


def test_guards(tmp_path):
    session = core.LocalSession(tmp_path, "x", 1)
    assert asyncio.run(session.exec("cat /x/ple-dev-set/tasks/roof-deck-design/tests/a.py"))["exit_code"] == 126
    with pytest.raises(ValueError):
        asyncio.run(session.write_file("../escape.txt", "x"))
    assert "not fetched" in asyncio.run(research.web_fetch({"url": "http://127.0.0.1:8001/v1/models"}))


def test_n3_fts5_candidate_limit_is_positive_integer():
    from frontier.n3 import N3_FTS5_CANDIDATE_LIMIT
    assert isinstance(N3_FTS5_CANDIDATE_LIMIT, int)
    assert N3_FTS5_CANDIDATE_LIMIT > 0


def test_n3_fts5_fallback_limit_is_positive_integer():
    from frontier.n3 import N3_FTS5_FALLBACK_LIMIT
    assert isinstance(N3_FTS5_FALLBACK_LIMIT, int)
    assert N3_FTS5_FALLBACK_LIMIT > 0


def test_n3_exact_lexical_match_is_recalled(tmp_path):
    memory = n3.N3(tmp_path / "recall.sqlite")
    unique_keyword = "xylophonicquasargalaxy"
    entry_id = memory.deposit(f"Detail containing {unique_keyword} for testing recall bounds", key="unique_note")
    results = memory.recall(unique_keyword)
    assert any(r["id"] == entry_id for r in results)


def test_n3_pinned_view_is_bounded(tmp_path):
    memory = n3.N3(tmp_path / "pinned_bounds.sqlite")
    for i in range(25):
        memory.deposit(f"pinned item {i}", key=f"pinned-{i}", pin=True)
    pinned_core = memory.core(limit=20)
    assert len(pinned_core) == 20
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}]
    memory.manage(messages, budget=10000)
    assert messages[2]["content"].startswith("[core memory]")
    core_lines = [line for line in messages[2]["content"].splitlines() if line.startswith("m")]
    assert len(core_lines) <= 20

