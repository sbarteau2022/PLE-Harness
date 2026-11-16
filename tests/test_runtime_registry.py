"""Driver selection and varied runtime-reported cache geometry; no inference claim."""

from dataclasses import replace
import hashlib
import sys
import types

import pytest

from frontier import drivers
from frontier.kv_contract import ExactPageSet, HeadPage, RuntimeIdentity, TensorSpec, continue_exact
from frontier.runtime_registry import load_runtime


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("layers,heads,dim", [(1, 1, 8), (12, 4, 16), (32, 8, 32), (80, 2, 64)])
def test_selected_driver_runtime_owns_all_cache_dimensions(monkeypatch, layers, heads, dim):
    model = f"fixture-{layers}-{heads}-{dim}"
    who = RuntimeIdentity(sha(model), sha("tokenizer-" + model), sha("adapter-" + model),
                           TensorSpec(layers, heads, dim, "f32", "token_major", sha("position-" + model)))

    class Runtime:
        def identity(self):
            return who

        def export_prefix(self, ids):
            payload = bytes(len(ids) * dim * 4)
            pages = tuple(HeadPage(layer, head, payload, payload)
                          for layer in range(layers) for head in range(heads))
            return ExactPageSet.seal(who, ids, pages)

        def continue_from_pages(self, pages, ids, count):
            assert pages.identity == who and len(pages.pages) == layers * heads
            return (100,)

    module = types.ModuleType("test_runtime_plugin")
    selected = []

    def factory(*, model, marker):
        selected.append((model, marker))
        return Runtime()

    module.create = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(drivers, "load_profiles", lambda: {
        "seat": {"model": model, "local_runtime": {"factory": "test_runtime_plugin:create",
                                                   "model_digest": who.model_digest, "options": {"marker": "selected"}}}})
    runtime = drivers.resolve_local_runtime("seat")
    pages = runtime.export_prefix((10, 20))
    assert pages.identity.spec == who.spec
    assert continue_exact(runtime, pages, (10, 20), (30,), 1) == (100,)
    assert selected == [(model, "selected")]


def test_api_seats_have_no_implicit_local_model(monkeypatch):
    monkeypatch.setattr(drivers, "load_profiles", lambda: {"api": {"model": "any-api-model"}})
    assert drivers.resolve_local_runtime("api") is None
    assert drivers.resolve_local_runtime("http://localhost:1234|another-model") is None


def test_wrong_model_and_live_shape_change_are_rejected(monkeypatch):
    who = RuntimeIdentity(sha("model"), sha("tokenizer"), sha("adapter"),
                           TensorSpec(2, 2, 4, "f32", "token_major", sha("position")))

    class Runtime:
        current = who

        def identity(self):
            return self.current

        def continue_from_pages(self, *args):
            raise AssertionError("changed model reached forward")

    raw = Runtime()
    module = types.ModuleType("test_runtime_plugin")
    module.create = lambda **kwargs: raw
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = {"model": "selected", "local_runtime": {"factory": "test_runtime_plugin:create", "model_digest": sha("wrong")}}
    with pytest.raises(ValueError, match="selected driver"):
        load_runtime(config)
    config["local_runtime"]["model_digest"] = who.model_digest
    bound = load_runtime(config)
    raw.current = replace(who, spec=replace(who.spec, layers=3))
    with pytest.raises(ValueError, match="identity changed"):
        bound.identity()


@pytest.mark.parametrize("config", [False, {}, {"factory": "not a factory"},
                                    {"factory": "test:create", "model_digest": "not-a-digest"}])
def test_invalid_runtime_configuration_has_no_fallback(config):
    with pytest.raises(ValueError):
        load_runtime({"model": "selected", "local_runtime": config})
