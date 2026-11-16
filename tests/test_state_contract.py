"""Architecture-neutral checkpoint gates; synthetic runtimes, not model qualification."""

from dataclasses import replace
import hashlib
import struct

import pytest

from frontier.kv_assembly import assemble_exact
from frontier.kv_contract import (ExactPageSet, ExactStateSet, HeadPage, RuntimeIdentity,
                                  StateSpec, StateTensor, StateTensorSpec, TensorSpec,
                                  continue_exact)
from frontier.runtime_registry import BoundRuntime


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def specification(window=2):
    def tensor(layer, name, shape, dtype="f32", limit=None):
        return StateTensorSpec(layer, name, shape, dtype, "fixture-le", sha(name), limit)

    return StateSpec(3, (
        tensor(0, "keys", ("tokens", 2), limit=window),
        tensor(0, "values", ("tokens", 2), limit=window),
        tensor(1, "recurrent", (3,)),
        tensor(2, "cursor", (), "i64"),
        tensor(2, "packed", (4,), "bytes"),
    ), sha("fixture-hybrid-recipe"))


def identity(spec=None):
    return RuntimeIdentity(sha("weights"), sha("tokenizer"), sha("adapter"), spec or specification())


def tensors(who, tokens=3):
    return tuple(StateTensor(s.layer, s.name, bytes(s.byte_length(tokens))) for s in who.spec.tensors)


def test_layer_count_is_not_a_single_model_limit():
    who = identity(TensorSpec(384, 1, 1, "f32", "token_major", sha("native-position")))
    pages = tuple(HeadPage(layer, 0, bytes(4), bytes(4)) for layer in range(384))
    cache = ExactPageSet.seal(who, (1,), pages)
    cache.admit(who, (1,))


def test_hybrid_checkpoint_resumes_a_numerical_recurrence():
    """Every declared state type contributes to the synthetic runtime's output."""
    who = identity()
    f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]

    def advance(state, ids):
        keys, values, recurrent, cursor, packed = state
        for token in ids:
            keys = (keys + [token / 2, token / 4])[-4:]
            values = (values + [token, token * 2])[-4:]
            recurrent = [f32(x * 0.75 + token * (i + 1) / 8) for i, x in enumerate(recurrent)]
            cursor += 1
            packed = [(x + token + i) % 256 for i, x in enumerate(packed)]
        return keys, values, recurrent, cursor, packed

    def generate(state, ids, count):
        state = advance(state, ids)
        result = []
        for _ in range(count):
            k, v, r, c, p = state
            token = int(sum(k + v + r + p) + c) % 32000
            result.append(token)
            state = advance(state, (token,))
        return tuple(result)

    empty = ([], [], [0, 0, 0], 0, [0, 0, 0, 0])

    class Runtime:
        def identity(self):
            return who

        def export_prefix(self, ids):
            k, v, r, c, p = advance(empty, ids)
            payloads = [struct.pack(f"<{len(x)}f", *x) for x in (k, v, r)]
            payloads += [struct.pack("<q", c), bytes(p)]
            buffers = tuple(StateTensor(s.layer, s.name, b) for s, b in zip(who.spec.tensors, payloads))
            return ExactStateSet.seal(who, ids, buffers)

        def continue_from_pages(self, checkpoint, ids, count):
            k, v, r, c, p = (tensor.payload for tensor in checkpoint.tensors)
            floats = lambda b: list(struct.unpack(f"<{len(b) // 4}f", b))
            return generate((floats(k), floats(v), floats(r), struct.unpack("<q", c)[0], list(p)), ids, count)

    runtime = BoundRuntime(Runtime(), who.model_digest)
    for prefix in ((1,), (1, 2, 3), tuple(range(1, 33))):
        checkpoint = runtime.export_prefix(prefix)
        assert continue_exact(runtime, checkpoint, prefix, (40, 50), 3) == generate(advance(empty, prefix), (40, 50), 3)
        with pytest.raises(ValueError, match="prefix"):
            continue_exact(runtime, checkpoint, prefix + (99,), (40,), 1)


@pytest.mark.parametrize("change", ["missing", "extra", "reorder", "layer", "name", "size", "mutable", "nan", "digest"])
def test_bad_hybrid_state_cannot_reach_runtime(change):
    who, prefix = identity(), (1, 2, 3)
    good = ExactStateSet.seal(who, prefix, tensors(who))
    buffers = list(good.tensors)
    if change == "missing": buffers.pop()
    elif change == "extra": buffers.append(buffers[-1])
    elif change == "reorder": buffers[0], buffers[1] = buffers[1], buffers[0]
    elif change == "layer": buffers[0] = replace(buffers[0], layer=True)
    elif change == "name": buffers[0] = replace(buffers[0], name="foreign")
    elif change == "size": buffers[2] = replace(buffers[2], payload=bytes(8))
    elif change == "mutable": buffers[2] = replace(buffers[2], payload=bytearray(12))
    elif change == "nan": buffers[2] = replace(buffers[2], payload=struct.pack("<3f", 0, float("nan"), 0))
    elif change == "digest": buffers[2] = replace(buffers[2], payload=struct.pack("<3f", 1, 2, 3))
    bad = replace(good, tensors=tuple(buffers))

    class Runtime:
        def identity(self):
            return who

        def continue_from_pages(self, *args):
            raise AssertionError("invalid state reached forward")

    with pytest.raises(ValueError):
        continue_exact(Runtime(), bad, prefix, (4,), 1)


@pytest.mark.parametrize("change", ["contract", "window", "dtype", "layers", "position"])
def test_complete_state_schema_is_pinned(change):
    who = identity()
    good = ExactStateSet.seal(who, (1, 2, 3), tensors(who))
    spec = who.spec
    if change == "contract": spec = replace(spec, state_contract=sha("other"))
    elif change == "layers": spec = replace(spec, layers=4)
    else:
        fields = {"window": {"token_limit": 3}, "dtype": {"dtype": "f16"}, "position": {"contract": sha("other-position")}}[change]
        spec = replace(spec, tensors=(replace(spec.tensors[0], **fields),) + spec.tensors[1:])
    with pytest.raises(ValueError, match="identity"):
        good.admit(replace(who, spec=spec), good.prefix_ids)


def test_runtime_swap_during_forward_discards_the_output():
    who = identity()
    checkpoint = ExactStateSet.seal(who, (1, 2, 3), tensors(who))

    class Runtime:
        current = who

        def identity(self):
            return self.current

        def continue_from_pages(self, *args):
            self.current = replace(who, model_digest=sha("changed"))
            return (10,)

    with pytest.raises(ValueError, match="during continuation"):
        continue_exact(Runtime(), checkpoint, checkpoint.prefix_ids, (4,), 1)


@pytest.mark.parametrize("buffers", [lambda t: t[::-1], lambda t: t + (t[-1],), lambda t: list(t)])
def test_invalid_declared_state_roster_is_rejected(buffers):
    who = identity()
    with pytest.raises(ValueError):
        replace(who, spec=replace(who.spec, tensors=buffers(who.spec.tensors))).validate()


@pytest.mark.parametrize("descriptor", [
    StateTensorSpec(0, "a", (True,), "f32", "le", sha("a")),
    StateTensorSpec(0, "a", ("tokens", "tokens"), "f32", "le", sha("a")),
    StateTensorSpec(0, "a", (2,), "f32", "le", sha("a"), 2),
    StateTensorSpec(0, "a", ("tokens",), "f32", "le", sha("a"), True),
    StateTensorSpec(0, "a", [2], "f32", "le", sha("a")),
    StateTensorSpec(0, "a", (2,), "unidentified", "le", sha("a")),
])
def test_malformed_state_descriptors_are_rejected(descriptor):
    with pytest.raises(ValueError):
        identity(StateSpec(1, (descriptor,), sha("state"))).validate()


def test_state_budget_and_unsupported_persistence_fail_before_reading():
    who = identity(StateSpec(1, (StateTensorSpec(0, "huge", (2**32 - 1,), "bytes", "packed", sha("packing")),), sha("state")))
    with pytest.raises(ValueError, match="budget"):
        ExactStateSet.seal(who, (1,), (StateTensor(0, "huge", b""),))

    class Mind:
        def tensor_exact(self, *args, **kwargs):
            raise AssertionError("unsupported state reached database")

    with pytest.raises(ValueError, match="cannot persist"):
        assemble_exact(Mind(), identity(), (1,), [], "bridge", recalled_addresses={"a1"})
    with pytest.raises(ValueError, match="named runtime state"):
        ExactPageSet.seal(identity(), (1,), ())
