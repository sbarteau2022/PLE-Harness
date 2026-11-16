"""Harness2 BenchmarkAdapter that executes against one live PLE task relay."""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from typing import Any

from harness2 import config as h2_config
from harness2.benchmarks.jobbench import _render_ctx, render_opencode_tree
from harness2.core.spec import BenchPrompts, BenchSpec, ChannelSpec
from harness2.core import tree
from harness2.substrate.opencode import OpencodeSubstrate


ROOT = pathlib.Path(__file__).resolve().parent
TOOL = ROOT / "bridge" / "ple_tool.py"

BASE_SYSTEM = """You are completing a live Physical Engineering task. Inspect inputs before calculating.
Write a viable deliverable early, verify units and constraints, reopen the final artifact, and finish
with exact paths and remaining limitations. The task filesystem is remote: use
`python scripts/ple_tool.py exec '<command>'`, `read '<path>'`,
`write '<path>' <local-source-or->`, and `mcp <server> <method> <subpath> <body-file-or->`
for task MCP servers. Never claim a command result before reading its output.
"""

BASE_GUARDRAILS = """Do not inspect tests, graders, answer keys, or rubric files. Do not substitute a
local workspace file for the requested remote deliverable. A finish without a verified remote
deliverable is incomplete. If an approach repeats without new evidence, stop, diagnose, and replan.
"""


def provider_config(model: str, base_url: str, key_env: str = "NVIDIA_API_KEY") -> dict[str, Any]:
    """OpenCode `.opencode/opencode.json` registering a custom provider/model.

    `model` is `provider/model-id`; the provider uses the OpenAI-compatible
    SDK against `base_url` with its key read from `key_env`. A `high` variant
    is declared because Harness2 always executes with `--variant high`.
    """
    provider, _, model_id = model.partition("/")
    if not provider or not model_id:
        raise ValueError(f"model {model!r} must use provider/model syntax")
    return {"provider": {provider: {
        "npm": "@ai-sdk/openai-compatible",
        "name": provider,
        "env": [key_env],
        "options": {"baseURL": base_url},
        "models": {model_id: {"name": model_id, "tool_call": True, "variants": {"high": {}}}},
    }}}


def write_provider_config(workspace: pathlib.Path, model: str, base_url: str) -> pathlib.Path:
    target = workspace / ".opencode" / "opencode.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(provider_config(model, base_url), indent=2) + "\n", encoding="utf-8")
    return target


def make_base(path: pathlib.Path) -> pathlib.Path:
    surface = OpencodeSubstrate().surface()
    if path.is_dir() and tree.is_tree(surface, path):
        return path
    values = {name: "" for name in surface.always_on}
    values["systemprompt.md"] = BASE_SYSTEM
    values["guardrails.md"] = BASE_GUARDRAILS
    values["tool_descriptions/tool_guidance.md"] = (
        "All task operations go through scripts/ple_tool.py and its authenticated TASK_RELAY_URL.\n"
    )
    values["LongTermMEMORY.md"] = "No task-specific answer is preloaded.\n"
    values["scripts/ple_tool.py"] = TOOL.read_text(encoding="utf-8")
    tree.write_tree(surface, values, path, stubs=True)
    return path


class PleAdapter:
    """A single live PLE task behind Harness2's unchanged method core."""

    def __init__(self, instruction: str, relay_url: str, run_root: pathlib.Path, model: str, timeout: int = 600):
        self.instruction = instruction
        self.relay_url = relay_url
        self.run_root = run_root
        self.model = model
        self.timeout = timeout
        self.task_id = "ple-" + __import__("hashlib").sha256(instruction.encode()).hexdigest()[:16]
        self.substrate = OpencodeSubstrate()
        channel = ChannelSpec("opencode", model, effort="high", steps=120, timeout=timeout)
        self.spec = BenchSpec(
            name="ple-live",
            runs_root=run_root,
            install_mode="workspace_agents_md",
            prompts=BenchPrompts(
                work="produce a checked physical-engineering deliverable from a live project workspace",
                domain_word="engineering discipline",
                deliverable_word="engineering deliverable",
                yardstick="Does the trajectory inspect authoritative inputs, calculate reproducibly, write the requested artifact, and verify it?",
                selection_checks="Prefer complete verified artifacts, traceable calculations, correct units, and explicit assumptions.",
            ),
            channels={"proposer": channel, "selector": channel},
            state_deliverable=True,
            obs_trunc=50_000,
        )
        self._base = make_base(run_root / "base-harness")

    def tasks(self, split: str = "all") -> list[str]:
        return [self.task_id]

    def domain_of(self, task_id: str) -> str:
        self._check_task(task_id)
        return "physical-engineering"

    def task_prompt(self, task_id: str) -> str:
        self._check_task(task_id)
        return self.instruction

    def task_files(self, task_id: str) -> None:
        self._check_task(task_id)
        return None

    def base_harness(self) -> pathlib.Path:
        return self._base

    def validate_tree(self, value: tree.Tree) -> None:
        joined = "\n".join(value.values())
        if "ple_tool.py" not in joined or "TASK_RELAY_URL" not in joined:
            raise ValueError("PLE harness tree must retain the authenticated relay tool guidance")

    def execute(self, task_id: str, harness_dir: pathlib.Path | None, run_id: str, *, timeout: int | None = None) -> pathlib.Path:
        self._check_task(task_id)
        run_dir = self.run_root / "executions" / re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
        if (run_dir / "closed.json").is_file():
            return run_dir
        shutil.rmtree(run_dir, ignore_errors=True)
        workspace = run_dir / "workspace"
        workspace.mkdir(parents=True)
        render_opencode_tree(pathlib.Path(harness_dir or self._base), workspace, _render_ctx("physical engineering"))
        (workspace / "scripts").mkdir(exist_ok=True)
        shutil.copy2(TOOL, workspace / "scripts" / "ple_tool.py")
        base_url = os.environ.get("MODEL_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or ""
        if not base_url:
            raise RuntimeError("MODEL_BASE_URL (or OPENAI_BASE_URL) is required to register the model provider")
        write_provider_config(workspace, self.model, base_url)
        prompt = (
            "Complete the live task below. The actual project filesystem is available only through "
            "scripts/ple_tool.py; local files are harness support, not deliverables.\n\n" + self.instruction
        )
        stream = run_dir / "events.jsonl"
        log = run_dir / "stderr.log"
        command = [
            str(h2_config.BUN), "run", "--cwd", str(h2_config.OPENCODE_DIR / "packages" / "opencode"),
            "--conditions=browser", "src/index.ts", "run", prompt, "--dir", str(workspace),
            "--model", self.model, "--variant", "high", "--format", "json",
        ]
        env = h2_config.scrub_env(keep=h2_config.provider_keys(self.model))
        env.update({
            "NVIDIA_API_KEY": os.environ.get("NVIDIA_API_KEY", os.environ.get("MODEL_API_KEY", "")),
            "TASK_RELAY_URL": self.relay_url,
            "NO_PROXY": "127.0.0.1,localhost," + env.get("NO_PROXY", ""),
            "no_proxy": "127.0.0.1,localhost," + env.get("no_proxy", ""),
            "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": os.environ.get("MAX_OUTPUT_TOKENS", "16000"),
            "OPENCODE_PERMISSION": json.dumps({"external_directory": {"*": "deny", f"{workspace}/**": "allow"}}),
            "XDG_DATA_HOME": str(workspace / ".ocdata"),
            "XDG_STATE_HOME": str(workspace / ".ocstate"),
        })
        with stream.open("w", encoding="utf-8") as out, log.open("w", encoding="utf-8") as err:
            result = subprocess.run(command, cwd=workspace, env=env, text=True, stdout=out, stderr=err, timeout=timeout or self.timeout)
        if result.returncode != 0:
            raise RuntimeError(f"OpenCode exited {result.returncode}; see {log}")
        if not stream.stat().st_size:
            raise RuntimeError("OpenCode returned no trajectory")
        (run_dir / "closed.json").write_text(json.dumps({"returncode": 0}) + "\n", encoding="utf-8")
        return run_dir

    def delivered(self, run_dir: pathlib.Path) -> bool:
        return (pathlib.Path(run_dir) / "closed.json").is_file()

    def trajectory(self, run_dir: pathlib.Path, *, obs_trunc: int) -> str:
        return self.substrate.read_trajectory(pathlib.Path(run_dir) / "events.jsonl", obs_trunc=obs_trunc, max_chars=200_000)

    def deliverable_text(self, run_dir: pathlib.Path) -> str:
        return self.trajectory(run_dir, obs_trunc=self.spec.obs_trunc)

    def deliverable_files(self, run_dir: pathlib.Path) -> dict[str, str]:
        return {"trajectory.md": self.deliverable_text(run_dir)}

    def score(self, run_dir: pathlib.Path, task_id: str) -> None:
        self._check_task(task_id)
        return None

    def _check_task(self, task_id: str) -> None:
        if task_id != self.task_id:
            raise ValueError(f"unknown live task {task_id!r}")

