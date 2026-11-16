"""Model-neutral page admission checks with two unrelated fake seats."""

from dataclasses import replace
import hashlib
import struct

import pytest

from frontier.kv_contract import (ExactPageSet, HeadPage, RuntimeIdentity,
                                  TensorSpec, continue_exact)


def identity(seed, heads=2):
    digest = lambda label: hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()
    return RuntimeIdentity(digest("weights"), digest("tokenizer"), digest("adapter"),
                           TensorSpec(2, heads, 4, "f32", "token-head-dim", digest("rope")))


def bundle(who):
    prefix = (10, 20, 30)
    pages = tuple(HeadPage(layer, head, bytes([layer + head]) * 48, bytes([head]) * 48)
                  for layer in range(2) for head in range(who.spec.kv_heads))
    return ExactPageSet.seal(who, prefix, pages)


@pytest.mark.parametrize("seed", ["model-a", "model-b"])
def test_any_exact_local_model_can_implement_the_same_contract(seed):
    who = identity(seed)
    pages = bundle(who)

    class Runtime:
        calls = 0

        def identity(self):
            return who

        def continue_from_pages(self, admitted, ids, count):
            self.calls += 1
            assert admitted is pages and ids == (40,) and count == 2
            return (41, 42)

    runtime = Runtime()
    assert continue_exact(runtime, pages, (10, 20, 30), (40,), 2) == (41, 42)
    assert runtime.calls == 1


def test_identity_prefix_missing_head_and_tamper_fail_before_runtime_call():
    who = identity("model-a")
    pages = bundle(who)

    class Runtime:
        calls = 0
        current = who

        def identity(self):
            return self.current

        def continue_from_pages(self, *args):
            self.calls += 1
            return (1,)

    runtime = Runtime()
    for changed in (identity("model-b"), replace(who, spec=replace(who.spec, position_contract="f" * 64))):
        runtime.current = changed
        with pytest.raises(ValueError, match="identity"):
            continue_exact(runtime, pages, (10, 20, 30), (40,), 1)
    runtime.current = who
    with pytest.raises(ValueError, match="prefix"):
        continue_exact(runtime, pages, (10, 20, 31), (40,), 1)
    with pytest.raises(ValueError, match="roster"):
        ExactPageSet.seal(who, (10, 20, 30), pages.pages[:-1])
    tampered = replace(pages, pages=(replace(pages.pages[0], keys=b"x" * 48),) + pages.pages[1:])
    with pytest.raises(ValueError, match="digest"):
        continue_exact(runtime, tampered, (10, 20, 30), (40,), 1)
    assert runtime.calls == 0


@pytest.mark.parametrize("dtype,payload", [
    ("f32", struct.pack("<f", float("nan"))),
    ("f16", struct.pack("<e", float("inf"))),
    ("bf16", struct.pack("<H", 0x7F80)),
])
def test_nonfinite_tensor_bytes_are_rejected(dtype, payload):
    who = replace(identity("finite"), spec=TensorSpec(1, 1, 1, dtype, "le", "position-v1"))
    with pytest.raises(ValueError, match="non-finite"):
        ExactPageSet.seal(who, (1,), (HeadPage(0, 0, payload, payload),))


@pytest.mark.parametrize("output", [(), (True,), (2**32,), (1, 2), [1]])
def test_runtime_token_output_is_checked(output):
    who = identity("output")
    pages = bundle(who)

    class Runtime:
        def identity(self):
            return who

        def continue_from_pages(self, *args):
            return output

    with pytest.raises(ValueError):
        continue_exact(Runtime(), pages, pages.prefix_ids, (40,), 1)
