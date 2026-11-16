"""Shared machinery for the frontier loop: results file, gates, snapshots, scope, sandboxed session."""

from __future__ import annotations

import asyncio
import fnmatch
import glob
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

DIR_NAME = "frontier-loop"
GATE_TIMEOUT = int(os.environ.get("FRONTIER_GATE_TIMEOUT", "3600"))
IGNORE = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", "runs", "artifacts",
          ".DS_Store", DIR_NAME} | set(filter(None, os.environ.get("FRONTIER_IGNORE", "").split(",")))
# Fair play: no model in the loop may read graders, answer keys or exported artifacts.
FORBIDDEN = re.compile(r"ple-dev-set/tasks/[^\s/]+/tests|/artifacts/|paper_score\.py")
GATE_KEYS = ("name", "passed", "exit_code", "value", "min")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit // 2] + "\n...\n" + text[-limit // 2:]


# ---------- the single results file ----------

def loop_dir(target: pathlib.Path) -> pathlib.Path:
    return target / DIR_NAME


def results_path(target: pathlib.Path) -> pathlib.Path:
    return loop_dir(target) / "results.json"


def load(target: pathlib.Path) -> dict:
    try:
        return json.loads(results_path(target).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"status": "none"}


def save(target: pathlib.Path, state: dict) -> None:
    path = results_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, path)


def require(target: pathlib.Path) -> dict:
    state = load(target)
    if state.get("status") in (None, "none"):
        sys.exit(f"no goal is set for {target}")
    return state


def read_doc(target: pathlib.Path, name: str) -> str:
    path = loop_dir(target) / name
    return path.read_text() if path.exists() else ""


def parse_block(text: str, tag: str):
    """Returns the JSON in the first ```<tag> fenced block, or None."""
    match = re.search(r"```" + tag + r"\s*\n(.*?)```", text, re.S)
    return json.loads(match.group(1)) if match else None


# ---------- budget ----------

def add_usage(state: dict, tokens: list[int], seconds: float, cache_read_tokens: int = 0) -> None:
    budget = state.setdefault("budget", {})
    budget["tokens_in"] = budget.get("tokens_in", 0) + tokens[0]
    budget["tokens_out"] = budget.get("tokens_out", 0) + tokens[1]
    budget["tokens_cached"] = budget.get("tokens_cached", 0) + cache_read_tokens
    budget["agent_seconds"] = round(budget.get("agent_seconds", 0) + seconds, 1)


def over_budget(state: dict) -> str:
    """Returns the reason the budget is spent, or ""."""
    budget = state.get("budget", {})
    used = budget.get("tokens_in", 0) + budget.get("tokens_out", 0)
    if budget.get("max_tokens") and used >= budget["max_tokens"]:
        return f"token budget spent ({used} of {budget['max_tokens']})"
    if budget.get("max_hours") and budget.get("agent_seconds", 0) >= budget["max_hours"] * 3600:
        return f"time budget spent ({budget['agent_seconds'] / 3600:.1f} of {budget['max_hours']} h)"
    return ""


# ---------- gates ----------

def validate_gates(gates) -> str:
    if not isinstance(gates, list) or not gates:
        return "gates must be a non-empty JSON list"
    names = set()
    for gate in gates:
        if not isinstance(gate, dict) or not gate.get("name") or not gate.get("cmd"):
            return f"each gate needs name and cmd: {gate}"
        if gate["name"] in names:
            return f"duplicate gate name: {gate['name']}"
        names.add(gate["name"])
        if gate.get("metric"):
            try:
                groups = re.compile(gate["metric"]).groups
                if groups != 1:
                    return (f"gate {gate['name']}: metric must be a regex with exactly one capture group around "
                            f"the number, like \"score=([0-9.]+)\" (this one has {groups}); a pass/fail gate "
                            "needs no metric, only a command that exits 0 on success")
            except re.error as exc:
                return f"gate {gate['name']}: bad metric regex: {exc}"
            if "min" not in gate:
                return f"gate {gate['name']}: a metric gate needs min (the target)"
    return ""


def _evidence(gate: dict, cwd: pathlib.Path) -> str:
    """Tail of the newest file matching the gate's `evidence` glob (logs that explain a failure)."""
    pattern = gate.get("evidence")
    if not pattern:
        return ""
    files = [f for f in glob.glob(str(cwd / pattern), recursive=True)
             if os.path.isfile(f) and not FORBIDDEN.search(f)]
    if not files:
        return ""
    newest = max(files, key=os.path.getmtime)
    with open(newest, errors="replace") as handle:
        return f"{os.path.relpath(newest, cwd)}:\n" + handle.read()[-2500:]


def run_gate(gate: dict, cwd: pathlib.Path) -> dict:
    try:
        proc = subprocess.run(["bash", "-c", gate["cmd"]], cwd=cwd, capture_output=True, text=True,
                              timeout=gate.get("timeout_sec", GATE_TIMEOUT))
        code, output = proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        code, output = 124, "gate timed out"
    value, passed = None, code == 0
    if gate.get("metric"):
        found = re.findall(gate["metric"], output)
        try:
            value = float(found[-1]) if found else None
        except ValueError:
            value = None
        passed = passed and value is not None and value >= float(gate["min"])
    result = {"name": gate["name"], "passed": passed, "exit_code": code, "value": value,
              "min": gate.get("min"), "holdout": bool(gate.get("holdout")), "tail": output[-1500:]}
    if not passed:
        result["evidence"] = _evidence(gate, cwd)
    return result


def run_gates(gates: list[dict], cwd: pathlib.Path) -> list[dict]:
    return [run_gate(gate, cwd) for gate in gates]


def measure_noise(gates: list[dict], cwd: pathlib.Path) -> dict[str, float]:
    """Run-to-run spread of each metric gate that asks for `repeats` > 1."""
    noise = {}
    for gate in gates:
        repeats = int(gate.get("repeats", 1))
        if gate.get("metric") and repeats > 1:
            values = [v for v in (run_gate(gate, cwd)["value"] for _ in range(repeats)) if v is not None]
            noise[gate["name"]] = round(max(values) - min(values), 6) if len(values) > 1 else 0.0
    return noise


def confirm_passes(gates: list[dict], before: list[dict], after: list[dict], cwd: pathlib.Path) -> list[dict]:
    """A noisy metric gate that newly passes must pass on average over its repeats, not once."""
    was = {r["name"]: r["passed"] for r in before}
    by_name = {g["name"]: g for g in gates}
    for result in after:
        gate = by_name[result["name"]]
        repeats = int(gate.get("repeats", 1))
        if result["passed"] and not was.get(result["name"]) and gate.get("metric") and repeats > 1:
            values = [result["value"]] + [run_gate(gate, cwd)["value"] or 0.0 for _ in range(repeats - 1)]
            result["value"] = round(sum(values) / len(values), 6)
            result["passed"] = result["value"] >= float(gate["min"])
    return after


def regressed(before: list[dict], after: list[dict], noise: dict[str, float]) -> str:
    """Returns why `after` is worse than `before` beyond noise, or "". Holdout gates never decide."""
    old = {r["name"]: r for r in before if not r["holdout"]}
    for result in after:
        prev = old.get(result["name"])
        if prev is None or result["holdout"]:
            continue
        if prev["passed"] and not result["passed"]:
            return f"gate {result['name']} went from PASS to FAIL"
        if prev["value"] is not None:
            drop = prev["value"] - (result["value"] if result["value"] is not None else float("-inf"))
            if drop > noise.get(result["name"], 0.0):
                return f"gate {result['name']} fell from {prev['value']} to {result['value']} (noise {noise.get(result['name'], 0.0)})"
    return ""


def all_pass(results: list[dict]) -> bool:
    visible = [r for r in results if not r["holdout"]]
    return bool(visible) and all(r["passed"] for r in visible)


def gate_report(results: list[dict], noise: dict[str, float] | None = None) -> str:
    """What a model sees. Holdout gates are left out on purpose."""
    lines = []
    for r in (r for r in results if not r["holdout"]):
        metric = f" value={r['value']} (target >= {r['min']})" if r["min"] is not None else ""
        if noise and r["name"] in noise:
            metric += f" noise +/-{noise[r['name']]}"
        lines.append(f"- {r['name']}: {'PASS' if r['passed'] else 'FAIL'} exit={r['exit_code']}{metric}")
        if not r["passed"]:
            lines.append("  output tail:\n" + "\n".join("    " + l for l in r["tail"].splitlines()[-25:]))
            if r.get("evidence"):
                lines.append("  evidence:\n" + "\n".join("    " + l for l in r["evidence"].splitlines()[-40:]))
    return "\n".join(lines) or "(no gates)"


def slim(results: list[dict]) -> list[dict]:
    return [{k: r[k] for k in GATE_KEYS} | {"holdout": r["holdout"]} for r in results]


# ---------- snapshots and workspaces (no git) ----------

def tree_hashes(root: pathlib.Path) -> dict[str, str]:
    hashes = {}
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in IGNORE]
        for name in files:
            path = pathlib.Path(folder) / name
            if name in IGNORE or path.is_symlink() or not path.is_file():
                continue
            hashes[str(path.relative_to(root))] = hashlib.sha1(path.read_bytes()).hexdigest()
    return hashes


def copy_tree(source: pathlib.Path, dest: pathlib.Path) -> None:
    for rel in tree_hashes(source):
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, out)


def diff_paths(base: dict[str, str], other: dict[str, str]) -> list[str]:
    return sorted(p for p in set(base) | set(other) if base.get(p) != other.get(p))


class Snapshot:
    """A copy of a folder (minus ignored folders) that it can be rolled back to."""

    def __init__(self, target: pathlib.Path) -> None:
        import tempfile
        self.target = target
        self.store = pathlib.Path(tempfile.mkdtemp(prefix="frontier-snap-"))
        self.before = tree_hashes(target)
        copy_tree(target, self.store)

    def changed(self) -> list[str]:
        return diff_paths(self.before, tree_hashes(self.target))

    def restore(self) -> None:
        for rel in self.changed():
            path = self.target / rel
            if rel in self.before:
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.store / rel, path)
            else:
                path.unlink(missing_ok=True)

    def close(self) -> None:
        shutil.rmtree(self.store, ignore_errors=True)


def in_scope(path: str, allowed: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) or path.startswith(pattern.rstrip("/") + "/") for pattern in allowed)


def validate_paths(paths) -> str:
    if not isinstance(paths, list) or not paths:
        return "allowed_paths must be a non-empty list"
    for path in paths:
        if not isinstance(path, str) or not path or path.startswith(("/", "..", "~")) or ".." in path.split("/"):
            return f"paths must be relative to the target folder: {path!r}"
        if path.strip("*/.") == "":
            return "the scope must name specific paths, not everything"
        if path.split("/")[0] == DIR_NAME:
            return f"{DIR_NAME}/ is managed by the loop and cannot be in scope"
    return ""


def text_diff(base: pathlib.Path, other: pathlib.Path, paths: list[str], limit: int = 12000) -> str:
    import difflib
    chunks = []
    for rel in paths:
        def lines(root):
            path = root / rel
            if not path.exists():
                return []
            try:
                return path.read_text().splitlines(keepends=True)
            except UnicodeDecodeError:
                return ["<binary>\n"]
        chunks.append("".join(difflib.unified_diff(lines(base), lines(other), f"a/{rel}", f"b/{rel}", n=2)))
    return clip("\n".join(chunks), limit)


# ---------- the session an agent drives ----------

def sandbox_profile(root: pathlib.Path) -> str:
    """macOS seatbelt profile: commands may write only inside `root` and temp folders."""
    allowed = [str(root.resolve()), "/private/tmp", "/tmp", "/private/var/folders", "/dev"]
    writes = " ".join(f'(subpath "{p}")' for p in allowed)
    return ('(version 1)(allow default)'
            '(deny file-write* (subpath "/"))'
            f'(allow file-write* {writes})'
            '(deny file-read* (regex #"ple-dev-set/tasks/[^/]+/tests"))')


SANDBOX = os.environ.get("FRONTIER_SANDBOX", "on") != "off" and os.path.exists("/usr/bin/sandbox-exec")


class LocalSession:
    """Gives agent.run the Session interface over a local folder (no Docker, no evaluator)."""

    mcp_servers: list = []

    def __init__(self, root: pathlib.Path, instruction: str, minutes: float) -> None:
        self.root, self.workdir, self.instruction = root, str(root), instruction
        self._deadline = time.monotonic() + minutes * 60
        self.started = time.monotonic()
        self.tokens = [0, 0]
        self.cache_read_tokens = 0
        self.tool_calls: list[str] = []

    def time_left(self) -> float:
        return self._deadline - time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def _path(self, path: str) -> pathlib.Path:
        full = (self.root / path).resolve()
        if FORBIDDEN.search(str(full)):
            raise ValueError("that path is off limits (grader or artifact)")
        if self.root.resolve() not in (full, *full.parents):
            raise ValueError("paths must stay inside the working folder")
        return full

    async def exec(self, command: str, timeout_sec: int = 120) -> dict:
        if FORBIDDEN.search(command):
            return {"exit_code": 126, "output": "refused: grading files and artifacts are off limits",
                    "truncated": False, "timed_out": False}
        argv = ["bash", "-c", command]
        if SANDBOX:
            argv = ["/usr/bin/sandbox-exec", "-p", sandbox_profile(self.root), *argv]
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=self.root, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout_sec)
            return {"exit_code": proc.returncode, "output": out.decode(errors="replace"),
                    "truncated": False, "timed_out": False}
        except asyncio.TimeoutError:
            proc.kill()
            return {"exit_code": 124, "output": "", "truncated": False, "timed_out": True}

    async def read_file(self, path: str) -> bytes:
        return self._path(path).read_bytes()

    async def write_file(self, path: str, data, append: bool = False) -> int:
        full = self._path(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        payload = data.encode() if isinstance(data, str) else bytes(data)
        with open(full, "ab" if append else "wb") as handle:
            handle.write(payload)
        return full.stat().st_size

    async def log(self, text: str) -> None:
        match = re.match(r"step \d+: (\w+)", text)
        if match:
            self.tool_calls.append(match.group(1))
        if os.environ.get("FRONTIER_QUIET") != "1":
            print("    " + text[:180], flush=True)

    async def usage(self, model, input_tokens=0, output_tokens=0, cache_read_tokens=0) -> None:
        self.tokens[0] += input_tokens
        self.tokens[1] += output_tokens
        self.cache_read_tokens += cache_read_tokens
