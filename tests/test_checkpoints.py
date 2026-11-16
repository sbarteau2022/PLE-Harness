"""Checkpoint coordination/admission; test memory store is not a production backend."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import math
import struct
import sys
import types
import asyncio

import pytest

import agent
from frontier.checkpoints import (NativeCacheContext, NativePosition, discover, plan,
                                  prefix_anchors, eligible_anchors, publish)
from frontier.kv_contract import (AttentionLayerSpec, LayeredTensorSpec, TensorSpec,
                                  RuntimeIdentity, ExactPageSet, HeadPage, _token_bytes)
from frontier.kv_assembly import assemble_exact, continue_nested, native_position_contract
from frontier.local_chat import PreparedChat


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture(layered=False):
    positions = (NativePosition("native-a", sha("position-a")), NativePosition("native-b", sha("position-b")))
    contracts = [native_position_contract({"tppe_version": p.version, "tppe_configuration_digest": p.configuration_digest}) for p in positions]
    spec = (LayeredTensorSpec((AttentionLayerSpec(2, 2, "f32", "token_major", contracts[0]),
                               AttentionLayerSpec(1, 3, "f16", "token_major", contracts[1]))) if layered
            else TensorSpec(2, 2, 2, "f32", "token_major", contracts[0]))
    context = NativeCacheContext("a1", "h3t2", sha("address"), sha("phase"), positions if layered else positions[:1])
    identity = RuntimeIdentity(sha("weights"), sha("tokenizer"), sha("adapter"), spec)
    return identity, context


def checkpoint(who, prefix):
    pages = []
    for layer, s in enumerate(who.spec.layer_specs):
        for head in range(s.kv_heads):
            size = len(prefix) * s.head_dim
            fmt = "f" if s.dtype == "f32" else "e"
            keys = struct.pack(f"<{size}{fmt}", *([0.25 + layer + head] * size))
            values = struct.pack(f"<{size}{fmt}", *([0.5 + layer + head] * size))
            pages.append(HeadPage(layer, head, keys, values))
    return ExactPageSet.seal(who, prefix, tuple(pages))


class MemoryMind:
    """Protocol double for bounded metadata and binary operations."""

    workspace, model = "workspace:hypermind", "fixture-neutral"

    def __init__(self):
        self.sessions, self.pages, self.payloads, self.lookups = {}, {}, {}, []
        self.puts = self.fetches = 0
        self.fail_at = None

    def tensor_grounding(self, address, geometry):
        assert (address, geometry) == ("a1", "h3t2")
        return "a1"

    def tensor_metadata(self, table, ids):
        self.lookups.append((table, ids))
        source = self.sessions if table == "transformer_kv_session" else self.pages
        return {key: deepcopy(source[key]) for key in ids if key in source}

    def tensor_put(self, q, key, value, executable, *, recalled_addresses):
        assert executable == "bridge" and recalled_addresses == {"a1"}
        self.puts += 1
        if self.fail_at == self.puts:
            self.fail_at = None
            raise OSError("simulated interrupted upload")
        s = {name: q[name] for name in ("model_id", "model_revision", "model_digest", "request_digest", "prefix_digest",
             "address_contract_digest", "tensor_contract_digest", "tppe_configuration_digest", "dtype", "layer_count", "kv_head_count", "head_dim")}
        s.update(id="transformer_kv_session:" + q["session"], workspace="workspace:" + q["workspace"],
                 geometry_space="geometry_space:" + q["geometry_space"], toroidal_phase_encoder_version=q["tppe_version"], active=True)
        if q["session"] in self.sessions and self.sessions[q["session"]] != s:
            raise ValueError("immutable session conflict")
        p = {name: q[name] for name in ("layer", "kv_head", "token_start", "token_end", "tensor_shape", "dtype", "layout", "phase_state_digest",
                                      "revision", "key_digest", "value_digest", "receipt_digest")}
        p.update(id="transformer_kv_page:" + q["page"], workspace="workspace:" + q["workspace"],
                 session="transformer_kv_session:" + q["session"], address="atom_address:" + q["address"],
                 trajectory_atoms=["atom:a1"], active=True)
        if q["page"] in self.pages and (self.pages[q["page"]] != p or self.payloads[q["page"]] != (key, value)):
            raise ValueError("immutable page conflict")
        self.sessions[q["session"]], self.pages[q["page"]], self.payloads[q["page"]] = s, p, (key, value)

    def tensor_exact(self, q, executable, *, recalled_addresses):
        self.fetches += 1
        return self.payloads[q["page"]]

    def continue_tensor_cache(self, runtime, prefix, roster, suffix, limit, bridge, *, recalled_addresses):
        return continue_nested(runtime, self, prefix, roster, suffix, limit, bridge, recalled_addresses=recalled_addresses)


@pytest.mark.parametrize("layered", [False, True])
def test_fragmented_publication_discovery_and_exact_restore(layered):
    who, context = fixture(layered)
    prefix = tuple(range(513))
    mind = MemoryMind()
    assert discover(mind, who, prefix, context) is None
    cache = checkpoint(who, prefix)
    roster = publish(mind, cache, context, "bridge")
    assert len(roster) == (9 if layered else 12)
    assert len({q["request_digest"] for q in roster}) == (2 if layered else 1)
    found = discover(mind, who, prefix, context)
    restored = assemble_exact(mind, who, prefix, found, "bridge", recalled_addresses={"a1"})
    assert restored == cache
    publish(mind, cache, context, "bridge")
    assert len(mind.pages) == len(roster)
    assert all(type(ids) is tuple and ids for _, ids in mind.lookups)


def test_interrupted_publication_is_a_miss_then_resumes_without_overwrite():
    who, context = fixture()
    mind = MemoryMind()
    prefix = tuple(range(513))
    cache = checkpoint(who, prefix)
    mind.fail_at = 3
    with pytest.raises(OSError):
        publish(mind, cache, context, "bridge")
    original = deepcopy(mind.payloads)
    assert discover(mind, who, prefix, context) is None and mind.fetches == 0
    publish(mind, cache, context, "bridge")
    assert all(mind.payloads[key] == value for key, value in original.items())
    assert discover(mind, who, prefix, context) is not None


@pytest.mark.parametrize("field", ["model_digest", "tokenizer_digest", "adapter_digest"])
def test_identity_changes_cannot_discover_another_runtime_cache(field):
    who, context = fixture()
    mind, prefix = MemoryMind(), (10, 20)
    publish(mind, checkpoint(who, prefix), context, "bridge")
    assert discover(mind, replace(who, **{field: sha("other")}), prefix, context) is None
    assert mind.fetches == 0


@pytest.mark.parametrize("change", ["session", "request", "prefix", "tokenizer-contract", "inactive", "shape-bool", "head", "address", "digest", "receipt", "atom"])
def test_metadata_corruption_is_not_a_silent_cache_miss(change):
    who, context = fixture()
    mind, prefix = MemoryMind(), (10, 20)
    publish(mind, checkpoint(who, prefix), context, "bridge")
    session, page = next(iter(mind.sessions.values())), next(iter(mind.pages.values()))
    if change == "session": session["model_digest"] = sha("other")
    elif change == "request": session["request_digest"] = sha("other")
    elif change == "prefix": session["prefix_digest"] = sha("other")
    elif change == "tokenizer-contract": session["tensor_contract_digest"] = sha("other")
    elif change == "inactive": page["active"] = False
    elif change == "shape-bool": page["tensor_shape"][0] = True
    elif change == "head": page["kv_head"] = True
    elif change == "address": page["address"] = "atom_address:foreign"
    elif change == "digest": page["key_digest"] = "invalid"
    elif change == "receipt": page["receipt_digest"] = sha("other")
    elif change == "atom": page["trajectory_atoms"] = ["atom:foreign"]
    with pytest.raises(ValueError):
        discover(mind, who, prefix, context)
    assert mind.fetches == 0


@pytest.mark.parametrize("length", [1, 2, 3, 17, 1025])
def test_prefix_anchor_hashes_and_logarithmic_candidate_count(length):
    ids = tuple(range(length))
    anchors = prefix_anchors(ids)
    assert len(anchors) == (math.floor(math.log2(length - 1)) + 1 if length > 1 else 0)
    for size, digest in anchors:
        assert size & (size - 1) == 0 and size < length
        assert digest == hashlib.sha256(_token_bytes(ids[:size])).hexdigest()


def test_anchor_selection_respects_byte_and_fragment_budgets(monkeypatch):
    from frontier import checkpoints
    who, _ = fixture()
    per_token = 2 * 2 * 2 * 4 * 2
    monkeypatch.setattr(checkpoints, "MAX_CACHE_BYTES", per_token * 8)
    assert [n for n, _ in eligible_anchors(who, tuple(range(65)))] == [1, 2, 4, 8]
    monkeypatch.setattr(checkpoints, "MAX_COMPONENTS", 3)
    assert eligible_anchors(who, (1, 2, 3)) == ()


@pytest.mark.parametrize("change", ["address", "geometry", "phase", "positions", "position-contract", "layout"])
def test_invalid_native_context_fails_before_metadata_or_payload_work(change):
    who, context = fixture()
    if change == "address": context = replace(context, address="a1; RETURN 1")
    elif change == "geometry": context = replace(context, geometry_space="geometry_space:h3t2")
    elif change == "phase": context = replace(context, phase_state_digest="bad")
    elif change == "positions": context = replace(context, positions=())
    elif change == "position-contract": context = replace(context, positions=(NativePosition("other", sha("other")),))
    elif change == "layout": who = replace(who, spec=replace(who.spec, layout="unknown"))
    mind = MemoryMind()
    with pytest.raises(ValueError):
        discover(mind, who, (10, 20), context)
    assert not mind.lookups and mind.fetches == mind.puts == 0


def test_corrupt_payload_is_rejected_after_discovery_before_forward():
    who, context = fixture()
    mind, prefix = MemoryMind(), (10, 20)
    publish(mind, checkpoint(who, prefix), context, "bridge")
    roster = discover(mind, who, prefix, context)
    key = roster[0]["page"]
    old_k, old_v = mind.payloads[key]
    mind.payloads[key] = (bytes(len(old_k)), old_v)

    class Runtime:
        def identity(self): return who
        def continue_from_pages(self, *args): raise AssertionError("corruption reached forward")

    with pytest.raises(ValueError, match="digest"):
        mind.continue_tensor_cache(Runtime(), prefix, roster, (30,), 1, "bridge", recalled_addresses={"a1"})


def test_local_chat_automatically_publishes_then_reuses_without_hand_built_receipt(monkeypatch):
    who, context = fixture()
    mind = MemoryMind()

    class Runtime:
        cold = warm = exports = 0

        def identity(self): return who

        def prepare_chat(self, messages, tools, shared):
            assert shared is mind
            return PreparedChat((10, 20, 30, 40, 50), context=context)

        def generate_tokens(self, prompt, limit):
            self.cold += 1
            return (60,)

        def continue_from_pages(self, pages, suffix, limit):
            self.warm += 1
            assert pages.prefix_ids == (10, 20, 30, 40) and suffix == (50,)
            return (60,)

        def export_prefix(self, prefix):
            self.exports += 1
            return checkpoint(who, prefix)

        def decode_chat(self, ids): return {"role": "assistant", "content": "done"}

    raw = Runtime()
    module = types.ModuleType("test_automatic_plugin")
    module.create = lambda **kwargs: raw
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = agent.Config(model=mind.model, base_url=None, api_key="none", max_output_tokens=1,
                          local_runtime={"factory": "test_automatic_plugin:create", "model_digest": who.model_digest, "bridge": "bridge"})
    for expected_cached in (0, 4):
        client = agent.ModelClient(config)  # New client; discovery uses only stored metadata.
        client.bind_mind(mind)
        response = asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
        assert response.usage.prompt_tokens_details.cached_tokens == expected_cached
    assert (raw.cold, raw.warm, raw.exports) == (1, 1, 1)
