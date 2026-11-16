"""Artifact output and tool-boundary qualification."""

import asyncio
import hashlib
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

import agent
from frontier import artifacts, core, pipeline, skills


def test_document_formats_through_tool_runner(tmp_path):
    session = core.LocalSession(tmp_path, "report", 1)
    config = agent.Config(model="mock", base_url=None, api_key="test")
    runner = agent.ToolRunner(session, config, {"artifact_render": artifacts.make_handler(session)})
    result = asyncio.run(runner.run("artifact_render", {
        "path": "reports/decision", "title": "Decision note", "kind": "document",
        "formats": ["txt", "md", "pdf", "docx"],
        "sections": [{"heading": "Evidence", "body": "The measured sample was 3 seconds."}],
        "note": "One sample; no throughput conclusion.",
    }))
    receipts = json.loads(result)["artifacts"]
    assert len(receipts) == 4
    for receipt in receipts:
        payload = (tmp_path / receipt["path"]).read_bytes()
        assert receipt["bytes"] == len(payload)
        assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()
    assert (tmp_path / "reports/decision.pdf").read_bytes().startswith(b"%PDF-1.4")
    with zipfile.ZipFile(tmp_path / "reports/decision.docx") as archive:
        doc = ET.fromstring(archive.read("word/document.xml"))
        assert "3 seconds" in "".join(doc.itertext())


def test_dag_and_spatial_are_accessible_and_escaped(tmp_path):
    session = core.LocalSession(tmp_path, "graph", 1)
    base = {"path": "figures/work", "title": "Plan <draft>", "kind": "dag",
            "formats": ["svg", "md"],
            "nodes": [{"id": "a", "label": "Research <script>"}, {"id": "b", "label": "Audit"}],
            "edges": [{"source": "a", "target": "b", "label": "before"}],
            "note": "A measured dependency, not a performance result."}
    artifacts.render(session, base)
    svg = (tmp_path / "figures/work.svg").read_text()
    ET.fromstring(svg)
    assert "&lt;script&gt;" in svg and "<script>" not in svg
    assert "Research <script>" in (tmp_path / "figures/work.md").read_text()
    spatial = dict(base, path="figures/space", kind="graph4d", formats=["html"])
    artifacts.render(session, spatial)
    page = (tmp_path / "figures/space.html").read_text()
    assert 'type="range"' in page and "Accessible graph data" in page
    assert "\\u003cscript>" in page and "<script>\"" not in page


def test_rejects_cycle_escape_and_bad_coordinates(tmp_path):
    session = core.LocalSession(tmp_path, "graph", 1)
    base = {"path": "map", "title": "Map", "kind": "dag", "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}]}
    with pytest.raises(ValueError, match="cycle"):
        artifacts.render(session, base)
    with pytest.raises(ValueError, match="path"):
        artifacts.render(session, dict(base, path="../outside", edges=[]))
    with pytest.raises(ValueError, match="finite"):
        artifacts.render(session, dict(base, kind="graph3d", edges=[], nodes=[{"id": "a", "x": float("nan")}]))
    assert not list(tmp_path.iterdir())


def test_frontier_and_zed_skill_surface():
    assert skills.find("ple-artifacts") is not None
    assert "artifact_render" in {t["function"]["name"] for t in pipeline.PHASE_TOOLS["worker"]}
    assert "artifact_render" in {t["function"]["name"] for t in pipeline.PHASE_TOOLS["design"]}
