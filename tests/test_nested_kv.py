"""Binary bridge integrity and an opt-in disposable SurrealDB integration gate."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import time
import types
import urllib.request

import pytest

from frontier import nested_kv, surreal
from frontier.kv_contract import (AttentionLayerSpec, LayeredTensorSpec, RuntimeIdentity,
                                  TensorSpec, _token_bytes)
from frontier.kv_assembly import native_position_contract


def test_publication_binary_bytes_and_request_receipt(monkeypatch):
    key, value = b"\x00\xff\x01", b"\x10\x00"
    witness = {"key_digest": hashlib.sha256(key).hexdigest(), "value_digest": hashlib.sha256(value).hexdigest()}
    seen = []

    def run(args, *, input, capture_output, env, timeout):
        seen.append((args, input))
        assert input[:4] == b"PKP1"
        nm, nk, nv = struct.unpack_from("<3I", input, 4)
        assert json.loads(input[16:16 + nm]) == witness
        assert input[16 + nm:] == key + value and (nk, nv) == (3, 2)
        return types.SimpleNamespace(returncode=0, stderr=b"", stdout=b"PWA1" + hashlib.sha256(input).digest())

    monkeypatch.setattr(nested_kv.subprocess, "run", run)
    nested_kv.publish_exact(witness, key, value, executable="bridge", env={})
    assert seen[0][0] == ["bridge", "put"]
    monkeypatch.setattr(nested_kv.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(
        returncode=0, stderr=b"", stdout=b"PWA1" + bytes(32)))
    with pytest.raises(ValueError, match="receipt"):
        nested_kv.publish_exact(witness, key, value, executable="bridge", env={})
    with pytest.raises(ValueError, match="digest"):
        nested_kv.publish_exact(witness, bytes(3), value, executable="bridge", env={})
    with pytest.raises(ValueError, match="tensor bytes"):
        nested_kv.publish_exact(witness, bytearray(key), value, executable="bridge", env={})


def test_binary_receipt_rejects_corruption_and_wrong_payload_identity():
    request, key, value = b"identity", b"\x00\xff", b"\x01\x00"
    identity = {"key_digest": hashlib.sha256(key).hexdigest(),
                "value_digest": hashlib.sha256(value).hexdigest()}
    body = b"PKV1" + struct.pack("<2I", len(key), len(value)) + key + value
    frame = body + hashlib.sha256(request + body).digest()
    assert nested_kv.decode_exact(request, frame, identity) == (key, value)
    for bad in (frame[:-1], frame + b"x", frame[:-32] + bytes(32)):
        with pytest.raises(ValueError):
            nested_kv.decode_exact(request, bad, identity)
    with pytest.raises(ValueError, match="receipt"):
        nested_kv.decode_exact(b"changed", frame, identity)
    with pytest.raises(ValueError, match="digest"):
        nested_kv.decode_exact(request, frame, {**identity, "key_digest": "f" * 64})


def test_existing_nested_surreal_pages_reach_ple_as_binary(tmp_path, monkeypatch):
    binary = os.environ.get("PLE_NESTED_KV_BRIDGE")
    if not binary or not Path(binary).is_file() or not shutil.which("surreal"):
        pytest.skip("set PLE_NESTED_KV_BRIDGE to a built binary and install surreal")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log = (tmp_path / "surreal.log").open("w")
    process = subprocess.Popen([shutil.which("surreal"), "start", "--bind", f"127.0.0.1:{port}",
                                "--user", "root", "--pass", "disposable", "memory"],
                               stdout=log, stderr=log)
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                with urllib.request.urlopen(url + "/health", timeout=0.3):
                    break
            except OSError:
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail("disposable SurrealDB did not start")
                time.sleep(0.05)
        monkeypatch.setattr(surreal, "URL", url)
        monkeypatch.setattr(surreal, "CORE_NS", "test")
        monkeypatch.setattr(surreal, "CORE_DB", "nested")
        monkeypatch.setattr(surreal, "credentials", lambda: ("root", "disposable"))
        surreal.sql(None, None, "DEFINE NAMESPACE test; USE NS test; DEFINE DATABASE nested;")
        digest = lambda x: hashlib.sha256(x).hexdigest()
        key, value = struct.pack("<8f", *range(8)), struct.pack("<8f", *range(8, 16))
        q = dict(workspace="hypermind", session="s1", page="p1", model_id="any-model",
                 model_revision="frozen", model_digest=digest(b"model"), request_digest=digest(b"request"),
                 prefix_digest=digest(_token_bytes((10, 20))), tppe_version="native-transform",
                 tppe_configuration_digest=digest(b"position"), address_contract_digest=digest(b"address"),
                 tensor_contract_digest=digest(b"tensor"), geometry_space="h3t2", dtype="f32",
                 layer_count=1, kv_head_count=1, head_dim=4, layer=0, kv_head=0,
                 token_start=0, token_end=2, tensor_shape=[2, 4], layout="token_major",
                 address="a1", phase_state_digest=digest(b"phase"), revision=1,
                 key_digest=digest(key), value_digest=digest(value))
        session_names = ("model_id", "model_revision", "model_digest", "request_digest", "prefix_digest",
                         "tppe_configuration_digest", "address_contract_digest", "tensor_contract_digest",
                         "dtype", "layer_count", "kv_head_count", "head_dim")
        page_names = ("layer", "kv_head", "token_start", "token_end", "tensor_shape", "dtype", "layout",
                      "phase_state_digest", "revision", "key_digest", "value_digest")
        fields = lambda names: ", ".join(f"{name} = {json.dumps(q[name])}" for name in names)
        mind = surreal.Mind("test", "any-model", "test://nested")
        mind._core("CREATE transformer_kv_session:s1 SET workspace = workspace:hypermind, active = true, "
                   "geometry_space = geometry_space:h3t2, toroidal_phase_encoder_version = 'native-transform', "
                   + fields(session_names) + "; CREATE transformer_kv_page:p1 SET workspace = workspace:hypermind, "
                   "session = transformer_kv_session:s1, address = atom_address:a1, active = true, "
                   + fields(page_names) + f', key_tensor = b"{key.hex()}", value_tensor = b"{value.hex()}";')
        assert mind.tensor_exact(q, binary, recalled_addresses={"a1"}) == (key, value)
        runtime_identity = RuntimeIdentity(q["model_digest"], digest(b"tokenizer"), digest(b"adapter"),
                                           TensorSpec(1, 1, 4, "f32", "token_major", native_position_contract(q)))
        cache = mind.tensor_cache(runtime_identity, (10, 20), [q], binary, recalled_addresses={"a1"})
        assert cache.pages[0].keys == key and cache.pages[0].values == value
        cache.admit(runtime_identity, (10, 20))

        class LocalRuntime:
            calls = 0

            def identity(self):
                return runtime_identity

            def continue_from_pages(self, admitted, continuation_ids, max_tokens):
                self.calls += 1
                assert admitted == cache and continuation_ids == (30,) and max_tokens == 1
                return (40,)

        runtime = LocalRuntime()
        assert mind.continue_tensor_cache(runtime, (10, 20), [q], (30,), 1, binary,
                                          recalled_addresses={"a1"}) == (40,)
        assert runtime.calls == 1

        # Actual agent -> selected driver -> codec -> same Mind -> Rust -> DB.
        import agent
        from frontier import drivers
        from frontier.local_chat import PreparedChat, NestedPrefix
        from test_agent import FakeSession

        class ChatRuntime(LocalRuntime):
            def prepare_chat(self, messages, tools, shared_mind):
                assert shared_mind is mind and messages[-1]["role"] == "user"
                return PreparedChat((10, 20, 30), NestedPrefix(runtime_identity, (10, 20), (q,), frozenset({"a1"})))

            def generate_tokens(self, *args):
                raise AssertionError("native cache dispatch unexpectedly took the cold path")

            def decode_chat(self, output):
                assert output == (40,)
                return {"role": "assistant", "tool_calls": [{"id": "finish", "type": "function",
                        "function": {"name": "finish", "arguments": '{"summary":"native nested chat"}'}}]}

        chat_runtime = ChatRuntime()
        module = types.ModuleType("test_nested_chat_plugin")
        module.create = lambda **kwargs: chat_runtime
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setattr(drivers, "load_profiles", lambda: {"native": {
            "model": "any-model", "context": 1000, "max_output": 1,
            "local_runtime": {"factory": "test_nested_chat_plugin:create", "model_digest": q["model_digest"],
                              "bridge": binary}}})
        config = drivers.resolve("native", max_steps=1)
        result = asyncio.run(agent.run(FakeSession(tmp_path), config, mind=mind))
        assert result == "native nested chat" and chat_runtime.calls == 1

        # Same nested tables and bridge; each layer has its own native session
        # shape, dtype and position recipe. No Glimmer wire or model assumptions.
        witnesses, expected, layers = [], [], []
        for layer, (heads, dim, dtype) in enumerate(((1, 2, "f32"), (2, 3, "f16"))):
            layer_q = {**q, "session": f"mixed{layer}", "layer_count": 2, "kv_head_count": heads,
                       "head_dim": dim, "dtype": dtype, "layer": layer, "tensor_shape": [2, dim],
                       "tppe_configuration_digest": digest(f"position{layer}".encode())}
            layers.append(AttentionLayerSpec(heads, dim, dtype, "token_major", native_position_contract(layer_q)))
            fields = lambda names: ", ".join(f"{name} = {json.dumps(layer_q[name])}" for name in names)
            mind._core(f"CREATE transformer_kv_session:mixed{layer} SET workspace = workspace:hypermind, active = true, "
                       "geometry_space = geometry_space:h3t2, toroidal_phase_encoder_version = 'native-transform', "
                       + fields(session_names) + ";")
            for head in range(heads):
                k = struct.pack(f"<{2 * dim}{'f' if dtype == 'f32' else 'e'}", *range(2 * dim))
                v = struct.pack(f"<{2 * dim}{'f' if dtype == 'f32' else 'e'}", *range(1, 2 * dim + 1))
                layer_q = {**layer_q, "page": f"mixed{layer}_{head}", "kv_head": head,
                           "key_digest": digest(k), "value_digest": digest(v)}
                mind._core(f"CREATE transformer_kv_page:mixed{layer}_{head} SET workspace = workspace:hypermind, "
                           f"session = transformer_kv_session:mixed{layer}, address = atom_address:a1, active = true, "
                           + fields(page_names) + f', key_tensor = b"{k.hex()}", value_tensor = b"{v.hex()}";')
                witnesses.append(layer_q)
                expected.append((k, v))
        mixed_identity = RuntimeIdentity(q["model_digest"], digest(b"tokenizer"), digest(b"adapter"),
                                         LayeredTensorSpec(tuple(layers)))
        mixed = mind.tensor_cache(mixed_identity, (10, 20), witnesses, binary, recalled_addresses={"a1"})
        assert [(p.keys, p.values) for p in mixed.pages] == expected
        mixed.admit(mixed_identity, (10, 20))

        class MixedRuntime:
            def identity(self):
                return mixed_identity

            def continue_from_pages(self, admitted, ids, count):
                assert admitted == mixed and ids == (30,) and count == 1
                return (40,)

        assert mind.continue_tensor_cache(MixedRuntime(), (10, 20), witnesses, (30,), 1, binary,
                                          recalled_addresses={"a1"}) == (40,)
        with pytest.raises(ValueError, match="selected recall"):
            mind.tensor_exact(q, binary, recalled_addresses={"other"})
        with pytest.raises(RuntimeError, match="lookup refused"):
            mind.tensor_exact({**q, "model_digest": digest(b"wrong")}, binary, recalled_addresses={"a1"})
        with pytest.raises(ValueError, match="workspace"):
            mind.tensor_exact({**q, "workspace": "foreign"}, binary, recalled_addresses={"a1"})
        mind._core('UPDATE transformer_kv_page:p1 SET key_tensor = b"00000000";')
        with pytest.raises(RuntimeError, match="digest"):
            mind.tensor_exact(q, binary, recalled_addresses={"a1"})

        # Fresh database with copied owning SCHEMAFULL, immutable fields and
        # production UNIQUE indexes, rather than the minimal read fixture.
        surreal.sql(None, None, "USE NS test; DEFINE DATABASE published;")
        monkeypatch.setattr(surreal, "CORE_DB", "published")
        schema_dir = Path(__file__).resolve().parents[1] / "native-kv-bridge" / "schema"
        for filename in ("0004_transformer_kv_tensor_pages.surql", "0009_immutable_tensor_kv.surql"):
            mind._core((schema_dir / filename).read_text())
        mind._core("CREATE workspace:hypermind; CREATE geometry_space:h3t2; "
                   "CREATE atom:a1 SET workspace = workspace:hypermind, active = true; "
                   "CREATE atom_address:a1 SET workspace = workspace:hypermind, atom = atom:a1, active = true;")
        published = {**q, "trajectory_atoms": ["a1"], "receipt_digest": digest(b"publication")}
        mind.tensor_put(published, key, value, binary, recalled_addresses={"a1"})
        mind.tensor_put(published, key, value, binary, recalled_addresses={"a1"})  # Idempotent.
        assert mind.tensor_exact(q, binary, recalled_addresses={"a1"}) == (key, value)
        assert len(mind._core("SELECT VALUE id FROM transformer_kv_page;")[0]) == 1

        changed = bytes(32)
        with pytest.raises(RuntimeError, match="publication refused"):
            mind.tensor_put({**published, "key_digest": digest(changed)}, changed, value, binary,
                            recalled_addresses={"a1"})
        assert mind.tensor_exact(q, binary, recalled_addresses={"a1"}) == (key, value)
        for bad in ({**published, "tensor_shape": [2, 99]},
                    {**published, "address": "missing"},
                    {**published, "geometry_space": "missing", "session": "missing_geometry", "page": "missing_geometry"},
                    {**published, "trajectory_atoms": ["foreign"]},
                    {**published, "receipt_digest": digest(b"changed-receipt")},
                    {**published, "model_digest": digest(b"changed-model")}):
            with pytest.raises(RuntimeError):
                mind.tensor_put(bad, key, value, binary, recalled_addresses={"a1", "missing"})
        nonfinite = struct.pack("<8f", float("inf"), *range(7))
        with pytest.raises(RuntimeError, match="non-finite"):
            mind.tensor_put({**published, "key_digest": digest(nonfinite)}, nonfinite, value, binary,
                            recalled_addresses={"a1"})
        assert len(mind._core("SELECT VALUE id FROM transformer_kv_page;")[0]) == 1

        # Distinct per-layer request identities satisfy the real UNIQUE index.
        first = {**published, "session": "layer0", "page": "layer0", "layer_count": 2,
                 "request_digest": digest(b"layer0-request"), "receipt_digest": digest(b"layer0-receipt")}
        mind.tensor_put(first, key, value, binary, recalled_addresses={"a1"})
        second = {**first, "session": "layer1", "page": "layer1", "layer": 1, "dtype": "f16", "head_dim": 3,
                  "tensor_shape": [2, 3], "receipt_digest": digest(b"layer1-receipt")}
        k2, v2 = struct.pack("<6e", *range(6)), struct.pack("<6e", *range(6, 12))
        second.update(key_digest=digest(k2), value_digest=digest(v2))
        with pytest.raises(RuntimeError, match="publication refused"):
            mind.tensor_put(second, k2, v2, binary, recalled_addresses={"a1"})
        second["request_digest"] = digest(b"layer1-request")
        mind.tensor_put(second, k2, v2, binary, recalled_addresses={"a1"})
        layered = RuntimeIdentity(q["model_digest"], digest(b"tokenizer"), digest(b"adapter"),
                                  LayeredTensorSpec((
                                      AttentionLayerSpec(1, 4, "f32", "token_major", native_position_contract(first)),
                                      AttentionLayerSpec(1, 3, "f16", "token_major", native_position_contract(second)))))
        restored = mind.tensor_cache(layered, (10, 20), [first, second], binary, recalled_addresses={"a1"})
        assert [(p.keys, p.values) for p in restored.pages] == [(key, value), (k2, v2)]
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
