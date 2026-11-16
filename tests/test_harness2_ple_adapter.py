from __future__ import annotations

import pathlib
import sys

import pytest


H2 = pathlib.Path(__file__).parents[1] / "ablations" / "vendor" / "harness2"
sys.path.insert(0, str(H2))

from ablations.harness2_ple import PleAdapter, make_base  # noqa: E402
from harness2.core import tree  # noqa: E402
from harness2.substrate.opencode import OpencodeSubstrate  # noqa: E402


def test_base_tree_is_real_harness2_surface(tmp_path: pathlib.Path) -> None:
    base = make_base(tmp_path / "base")
    value = tree.read_tree(OpencodeSubstrate().surface(), base)
    assert "ple_tool.py" in value["systemprompt.md"]
    assert (base / "scripts" / "ple_tool.py").is_file()


def test_adapter_rejects_unknown_task_and_removed_relay_guidance(tmp_path: pathlib.Path) -> None:
    adapter = PleAdapter("do work", "http://127.0.0.1:1/token", tmp_path, "openai/model")
    with pytest.raises(ValueError, match="unknown"):
        adapter.task_prompt("other")
    value = tree.read_tree(adapter.substrate.surface(), adapter.base_harness())
    value["systemprompt.md"] = "no bridge"
    value["guardrails.md"] = ""
    value["tool_descriptions/tool_guidance.md"] = ""
    with pytest.raises(ValueError, match="relay"):
        adapter.validate_tree(value)



def test_provider_config_registers_nvidia_model(tmp_path: pathlib.Path) -> None:
    from ablations.harness2_ple import provider_config, write_provider_config
    config = provider_config("nvidia/mistralai/m", "https://integrate.api.nvidia.com/v1")
    provider = config["provider"]["nvidia"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["env"] == ["NVIDIA_API_KEY"]
    assert provider["options"]["baseURL"].endswith("/v1")
    assert provider["models"]["mistralai/m"]["variants"] == {"high": {}}
    target = write_provider_config(tmp_path, "nvidia/mistralai/m", "https://x/v1")
    assert target == tmp_path / ".opencode" / "opencode.json" and target.is_file()
    with pytest.raises(ValueError, match="provider/model"):
        provider_config("bare", "https://x/v1")
