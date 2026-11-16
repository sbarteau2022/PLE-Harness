"""Opt-in real SurrealKV restart; the model codec/forward fixture is synthetic."""

import asyncio
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import types
import urllib.request

import pytest

import agent
from frontier import surreal
from frontier.local_chat import PreparedChat
from test_checkpoints import fixture, checkpoint


def test_automatic_checkpoint_survives_real_database_and_client_restart(tmp_path, monkeypatch):
    binary = os.environ.get("PLE_NESTED_KV_BRIDGE")
    if not binary or not Path(binary).is_file() or not shutil.which("surreal"):
        pytest.skip("set PLE_NESTED_KV_BRIDGE and install surreal")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log = (tmp_path / "surreal.log").open("w")
    process = None

    def start():
        nonlocal process
        process = subprocess.Popen([shutil.which("surreal"), "start", "--bind", f"127.0.0.1:{port}",
            "--user", "root", "--pass", "disposable", f"surrealkv://{tmp_path / 'state'}"], stdout=log, stderr=log)
        deadline = time.monotonic() + 10
        while True:
            try:
                with urllib.request.urlopen(url + "/health", timeout=0.3): return
            except OSError:
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail("disposable persistent SurrealDB did not start")
                time.sleep(0.05)

    def stop():
        nonlocal process
        if process is not None:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            process = None

    try:
        start()
        monkeypatch.setattr(surreal, "URL", url)
        monkeypatch.setattr(surreal, "CORE_NS", "test")
        monkeypatch.setattr(surreal, "CORE_DB", "persistent")
        monkeypatch.setattr(surreal, "credentials", lambda: ("root", "disposable"))
        surreal.sql(None, None, "DEFINE NAMESPACE test; USE NS test; DEFINE DATABASE persistent;")
        mind = surreal.Mind("test", "fixture-neutral", "test://checkpoint-restart")
        schema = Path(__file__).resolve().parents[1] / "native-kv-bridge" / "schema"
        for name in ("0004_transformer_kv_tensor_pages.surql", "0009_immutable_tensor_kv.surql"):
            mind._core((schema / name).read_text())
        mind._core("CREATE workspace:hypermind; CREATE geometry_space:h3t2; "
                   "CREATE atom:a1 SET workspace = workspace:hypermind, active = true; "
                   "CREATE atom_address:a1 SET workspace = workspace:hypermind, atom = atom:a1, active = true;")
        queries = []
        original_core = mind._core

        def traced(query, variables=None):
            queries.append(query)
            return original_core(query, variables)

        mind._core = traced
        assert mind.tensor_metadata("transformer_kv_page", tuple(f"missing{i}" for i in range(300))) == {}
        assert len(queries) == 3 and all("FROM transformer_kv_page:" in query for query in queries)
        who, context = fixture(layered=True)
        prompt = tuple(range(513))
        instances = []

        class Runtime:
            def __init__(self): self.cold = self.warm = self.exports = 0
            def identity(self): return who
            def prepare_chat(self, messages, tools, shared):
                assert shared.workspace == "workspace:hypermind"
                return PreparedChat(prompt, context=context)
            def generate_tokens(self, tokens, limit):
                self.cold += 1
                return (60,)
            def export_prefix(self, prefix):
                self.exports += 1
                return checkpoint(who, prefix)
            def continue_from_pages(self, pages, suffix, limit):
                self.warm += 1
                assert pages == checkpoint(who, prompt[:512]) and suffix == prompt[512:]
                return (60,)
            def decode_chat(self, output): return {"role": "assistant", "content": "restored"}

        def create(**kwargs):
            raw = Runtime()
            instances.append(raw)
            return raw

        module = types.ModuleType("test_persistent_runtime")
        module.create = create
        monkeypatch.setitem(sys.modules, module.__name__, module)
        config = agent.Config(model=mind.model, base_url=None, api_key="none", max_output_tokens=1,
            local_runtime={"factory": "test_persistent_runtime:create", "model_digest": who.model_digest, "bridge": binary})
        first = agent.ModelClient(config)
        first.bind_mind(mind)
        reply = asyncio.run(first.complete([{"role": "user", "content": "first"}]))
        assert reply.usage.prompt_tokens_details.cached_tokens == 0
        assert (instances[0].cold, instances[0].warm, instances[0].exports) == (1, 0, 1)
        stop()
        start()
        fresh_mind = surreal.Mind("test", mind.model, "test://new-client")
        second = agent.ModelClient(config)
        second.bind_mind(fresh_mind)
        reply = asyncio.run(second.complete([{"role": "user", "content": "second"}]))
        assert reply.choices[0].message.content == "restored"
        assert reply.usage.prompt_tokens_details.cached_tokens == 512
        assert (instances[1].cold, instances[1].warm, instances[1].exports) == (0, 1, 0)
    finally:
        stop()
        log.close()
