"""Adversarial complete-cache assembly checks; no real-model quality claim."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import struct

import pytest

from frontier.kv_assembly import assemble_exact, continue_nested, native_position_contract
from frontier.kv_contract import (AttentionLayerSpec, LayeredTensorSpec, RuntimeIdentity,
                                  TensorSpec, _token_bytes)


def fixture(seed="model-a"):
    digest = lambda b: hashlib.sha256(b).hexdigest()
    prefix = (10, 20, 30)
    common = dict(workspace="hypermind", session="s1", model_id=seed, model_revision="frozen",
                  model_digest=digest(seed.encode()), request_digest=digest(b"request"),
                  prefix_digest=digest(_token_bytes(prefix)), tppe_version="native-v1",
                  tppe_configuration_digest=digest(b"position"), address_contract_digest=digest(b"address"),
                  tensor_contract_digest=digest(b"tensor"), geometry_space="h3t2", dtype="f32",
                  layer_count=2, kv_head_count=2, head_dim=4, layout="token_major")
    identity = RuntimeIdentity(common["model_digest"], digest(b"tokenizer"), digest(b"adapter"),
                               TensorSpec(2, 2, 4, "f32", "token_major", native_position_contract(common)))
    roster, payloads = [], {}
    for layer in range(2):
        for head in range(2):
            for start, end in ((0, 1), (1, 3)):
                name = f"p{layer}_{head}_{start}"
                size = (end - start) * 4
                k = struct.pack(f"<{size}f", *(layer + head + i / 10 for i in range(start * 4, end * 4)))
                v = struct.pack(f"<{size}f", *(10 + layer + head + i / 10 for i in range(start * 4, end * 4)))
                payloads[name] = (k, v)
                roster.append({**common, "page": name, "layer": layer, "kv_head": head,
                               "token_start": start, "token_end": end, "tensor_shape": [end - start, 4],
                               "address": "a1", "phase_state_digest": digest(b"phase"), "revision": 1,
                               "key_digest": digest(k), "value_digest": digest(v)})
    return identity, prefix, roster, payloads


def layered_fixture():
    who, prefix, old, _ = fixture()
    roster, payloads, layers = [], {}, []
    for layer, (heads, dim, dtype) in enumerate(((2, 4, "f32"), (1, 3, "f16"), (3, 2, "bf16"))):
        common = {**old[0], "session": f"session{layer}", "layer_count": 3, "kv_head_count": heads,
                  "head_dim": dim, "dtype": dtype,
                  "request_digest": hashlib.sha256(f"request{layer}".encode()).hexdigest(),
                  "tppe_configuration_digest": hashlib.sha256(f"position{layer}".encode()).hexdigest()}
        layers.append(AttentionLayerSpec(heads, dim, dtype, "token_major", native_position_contract(common)))
        for head in range(heads):
            for start, end in ((0, 1), (1, 3)):
                name, size = f"p{layer}_{head}_{start}", (end - start) * dim
                fmt, number = ("f", layer + head + 1) if dtype == "f32" else (("e", layer + head + 1) if dtype == "f16" else ("H", 0x3F80))
                k = struct.pack(f"<{size}{fmt}", *([number] * size))
                v = struct.pack(f"<{size}{fmt}", *([number + 1] * size))
                payloads[name] = (k, v)
                roster.append({**common, "page": name, "layer": layer, "kv_head": head, "token_start": start,
                               "token_end": end, "tensor_shape": [end - start, dim],
                               "key_digest": hashlib.sha256(k).hexdigest(), "value_digest": hashlib.sha256(v).hexdigest()})
    return replace(who, spec=LayeredTensorSpec(tuple(layers))), prefix, roster, payloads


class Mind:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = 0

    def tensor_exact(self, witness, executable, *, recalled_addresses):
        self.calls += 1
        assert executable == "bridge" and witness["address"] in recalled_addresses
        return self.payloads[witness["page"]]


@pytest.mark.parametrize("seed", ["model-a", "model-b"])
def test_fragmented_cache_assembled_before_runtime_continuation(seed):
    who, prefix, roster, payloads = fixture(seed)
    mind = Mind(payloads)
    pages = assemble_exact(mind, who, prefix, roster, "bridge", recalled_addresses={"a1"})
    assert len(pages.pages) == 4
    assert pages.pages[0].keys == payloads["p0_0_0"][0] + payloads["p0_0_1"][0]
    assert mind.calls == 8

    class Runtime:
        calls = 0

        def identity(self):
            return who

        def continue_from_pages(self, admitted, ids, count):
            self.calls += 1
            assert admitted == pages and ids == (40,) and count == 2
            return (50, 60)

    runtime = Runtime()
    assert continue_nested(runtime, mind, prefix, roster, (40,), 2, "bridge",
                           recalled_addresses={"a1"}) == (50, 60)
    assert runtime.calls == 1


def test_heterogeneous_native_layers_keep_their_own_shapes_and_encodings():
    who, prefix, roster, payloads = layered_fixture()
    mind = Mind(payloads)
    cache = assemble_exact(mind, who, prefix, roster, "bridge", recalled_addresses={"a1"})
    assert len(cache.pages) == 6 and mind.calls == 12
    for page in cache.pages:
        s = who.spec.layer_specs[page.layer]
        assert len(page.keys) == len(prefix) * s.head_dim * s.bytes_per_value
        assert page.keys == payloads[f"p{page.layer}_{page.head}_0"][0] + payloads[f"p{page.layer}_{page.head}_1"][0]
    cache.admit(who, prefix)


def test_assembly_does_not_expand_all_layer_descriptors_for_every_fragment():
    who, prefix, roster, payloads = fixture()
    expansions = []

    class CountedSpec(TensorSpec):
        @property
        def layer_specs(self):
            expansions.append(1)
            return super().layer_specs

    who = replace(who, spec=CountedSpec(2, 2, 4, "f32", "token_major", who.spec.position_contract))
    assemble_exact(Mind(payloads), who, prefix, roster, "bridge", recalled_addresses={"a1"})
    assert len(expansions) <= 3  # Constant number of full-schema passes, independent of fragments.


@pytest.mark.parametrize("change", ["dim", "dtype", "heads", "position", "request", "session", "contract", "extra"])
def test_heterogeneous_metadata_errors_fail_before_fetch(change):
    who, prefix, roster, payloads = layered_fixture()
    target = roster[5]  # Second fragment of the second layer.
    if change == "dim": target["head_dim"] = 4
    elif change == "dtype": target["dtype"] = "f32"
    elif change == "heads": target["kv_head_count"] = 2
    elif change == "position": target["tppe_configuration_digest"] = "f" * 64
    elif change == "request": target["request_digest"] = "f" * 64
    elif change == "session": target["session"] = "foreign"
    elif change == "contract": target["tensor_contract_digest"] = "f" * 64
    elif change == "extra": roster.append({**roster[-1], "layer": 3, "kv_head": 0, "token_start": 0})
    mind = Mind(payloads)
    with pytest.raises(ValueError):
        assemble_exact(mind, who, prefix, roster, "bridge", recalled_addresses={"a1"})
    assert mind.calls == 0


@pytest.mark.parametrize("change", ["missing", "duplicate", "gap", "overlap", "session", "contract",
                                    "model", "prefix", "position", "address", "shape", "boolean", "order"])
def test_invalid_roster_fails_before_any_tensor_is_fetched(change):
    who, prefix, roster, payloads = fixture()
    roster = deepcopy(roster)
    if change == "missing": roster.pop()
    elif change == "duplicate": roster[1]["page"] = roster[0]["page"]
    elif change == "gap": roster[1]["token_start"] = 2
    elif change == "overlap": roster[1]["token_start"] = 0
    elif change == "session": roster[-1]["session"] = "other"
    elif change == "contract": roster[-1]["tensor_contract_digest"] = "f" * 64
    elif change == "model": who = replace(who, model_digest="f" * 64)
    elif change == "prefix": prefix = (10, 20, 31)
    elif change == "position": who = replace(who, spec=replace(who.spec, position_contract="f" * 64))
    elif change == "address": roster[-1]["address"] = "other"
    elif change == "shape": roster[-1]["tensor_shape"] = [2, 8]
    elif change == "boolean": roster[-1]["token_start"] = True
    elif change == "order": roster[0], roster[1] = roster[1], roster[0]
    mind = Mind(payloads)
    with pytest.raises(ValueError):
        assemble_exact(mind, who, prefix, roster, "bridge", recalled_addresses={"a1"})
    assert mind.calls == 0


def test_corruption_fetch_failure_and_live_model_switch_never_reach_forward():
    who, prefix, roster, payloads = fixture()

    class Runtime:
        calls = 0
        identity_calls = 0

        def identity(self):
            self.identity_calls += 1
            return who if self.identity_calls == 1 else replace(who, adapter_digest="f" * 64)

        def continue_from_pages(self, *args):
            self.calls += 1
            return (1,)

    runtime = Runtime()
    with pytest.raises(ValueError, match="identity"):
        continue_nested(runtime, Mind(payloads), prefix, roster, (40,), 1, "bridge", recalled_addresses={"a1"})
    assert runtime.calls == 0
    payloads[roster[-1]["page"]] = (bytes(32), bytes(32))
    with pytest.raises(ValueError, match="digest"):
        continue_nested(runtime, Mind(payloads), prefix, roster, (40,), 1, "bridge", recalled_addresses={"a1"})
    assert runtime.calls == 0
    del payloads[roster[0]["page"]]
    with pytest.raises(KeyError):
        continue_nested(runtime, Mind(payloads), prefix, roster, (40,), 1, "bridge", recalled_addresses={"a1"})
    assert runtime.calls == 0


def test_cache_budget_rejects_before_fetch():
    who, prefix, roster, payloads = fixture()
    mind = Mind(payloads)
    with pytest.raises(ValueError, match="budget"):
        assemble_exact(mind, who, prefix, roster, "bridge", recalled_addresses={"a1"}, max_bytes=1)
    assert mind.calls == 0


@pytest.mark.parametrize("tokens,limit", [((), 1), ((True,), 1), ((40,), 0), ((40,), True)])
def test_invalid_continuation_does_not_fetch_any_page(tokens, limit):
    who, prefix, roster, payloads = fixture()
    mind = Mind(payloads)

    class Runtime:
        def identity(self):
            return who

    with pytest.raises(ValueError):
        continue_nested(Runtime(), mind, prefix, roster, tokens, limit, "bridge", recalled_addresses={"a1"})
    assert mind.calls == 0
