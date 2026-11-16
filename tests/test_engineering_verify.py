"""Contract and adversarial checks for the model-neutral engineering tool."""

import asyncio
import json
import subprocess
import sys

import pytest

import agent
from frontier import core, pipeline
from engineering_verify import verify


def sample():
    return {"units": "m", "time_unit": "s", "flow_unit": "L/s",
            "components": [
                {"id": "pump", "box": {"x": 0, "y": 0, "z": 0, "dx": 1, "dy": 1, "dz": 1}, "source": "P1"},
                {"id": "header", "box": {"x": 2, "y": 0, "z": 0, "dx": 1, "dy": 1, "dz": 1}, "source": "P1"},
                {"id": "coil", "box": {"x": 4, "y": 0, "z": 0, "dx": 1, "dy": 1, "dz": 1}, "source": "P2"}],
            "connections": [{"id": "a", "from": "pump", "to": "header"},
                            {"id": "b", "from": "header", "to": "coil"}],
            "required_routes": [{"from": "pump", "to": "coil"}],
            "clearances": [{"a": "pump", "b": "header", "axis": "x", "min": 1}],
            "junctions": ["header"],
            "states": [{"time": 0, "flows": {"a": 2, "b": 2}},
                       {"time": 1, "flows": {"a": 3, "b": 3}}]}


def codes(value):
    return {f["code"] for f in verify(value)["findings"]}


def test_clean_manifest_through_tool_runner(tmp_path):
    session = core.LocalSession(tmp_path, "engineering", 1)
    runner = agent.ToolRunner(session, agent.Config(model="mock", base_url=None, api_key="x"))
    result = json.loads(asyncio.run(runner.run("engineering_verify", {"manifest": sample()})))
    assert result["passed"] and result["checks"] == {"components": 3, "connections": 2, "states": 2, "junctions": 1}
    assert result["input_sha256"] == verify(sample())["input_sha256"]
    assert "engineering_verify" in {t["function"]["name"] for t in agent.TOOLS}
    assert "engineering_verify" in {t["function"]["name"] for t in pipeline.PHASE_TOOLS["worker"]}


def test_geometry_and_source_findings():
    data = sample()
    data["components"][0]["source"] = ""
    data["components"][1]["box"]["x"] = 0.5
    data["clearances"][0]["min"] = 2
    assert {"missing_source", "box_overlap", "insufficient_clearance"} <= codes(data)
    data["components"][0]["box"]["dx"] = -1
    assert "nonpositive_dimension" in codes(data)


def test_graph_and_time_findings():
    data = sample()
    data["connections"][1]["to"] = "missing"
    data["states"][1] = {"time": 0, "flows": {"a": 3}}
    assert {"unknown_endpoint", "disconnected_route", "nonmonotonic_time", "flow_ids_mismatch"} <= codes(data)
    data = sample()
    data["states"][0]["flows"]["b"] = 1
    assert "flow_imbalance" in codes(data)


@pytest.mark.parametrize("mutation", [
    lambda d: d["components"][0]["box"].update(x=float("nan")),
    lambda d: d["components"][0].update(id="header"),
    lambda d: d.update(units="parsec"),
    lambda d: d.update(states=[{"time": 0, "flows": {"a": float("inf"), "b": 1}}]),
    lambda d: d.update(components=d["components"] * 100),
])
def test_bad_input_rejected(mutation):
    data = sample()
    mutation(data)
    with pytest.raises(ValueError):
        verify(data)


def test_empty_manifest_and_tool_output_limit_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        verify({"units": "m"})
    session = core.LocalSession(tmp_path, "engineering", 1)
    config = agent.Config(model="mock", base_url=None, api_key="x", max_tool_output_chars=32)
    result = asyncio.run(agent.ToolRunner(session, config).run("engineering_verify", {"manifest": sample()}))
    assert result.startswith("error:")
    assert '"passed":true' not in result


def test_cli_is_a_reusable_gate(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(sample()))
    command = [sys.executable, "-m", "engineering_verify", str(path)]
    ok = subprocess.run(command, capture_output=True, text=True)
    assert ok.returncode == 0 and json.loads(ok.stdout)["passed"]
    data = sample()
    data["states"][0]["flows"]["b"] = 0
    path.write_text(json.dumps(data))
    bad = subprocess.run(command, capture_output=True, text=True)
    assert bad.returncode == 1 and "flow_imbalance" in bad.stdout
