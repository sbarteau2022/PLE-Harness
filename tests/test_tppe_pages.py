"""Algebraic and adversarial checks of the PLE TPPE boundary reference."""

import math

import pytest

from frontier.tppe_pages import Layout, Position, encode_key, relocate_key, positional_kernel


@pytest.mark.parametrize("arm", ["two_angle_v2", "multiscale_spiral_v3"])
def test_relocation_equals_fresh_position_encoding(arm):
    raw = [math.sin(i * 0.37) for i in range(80)]
    source = Position(19.3, 8.1, 0.35)
    target = Position(41.7, 16.8, 0.72)
    moved = relocate_key(encode_key(raw, source, arm), source, target, arm)
    fresh = encode_key(raw, target, arm)
    assert moved == pytest.approx(fresh, abs=1e-13)
    assert moved[64:] == raw[64:]


def test_page_attention_score_and_value_are_unchanged():
    a, b = Position(3.1, 1.7, 0.2), Position(14.9, 7.2, 0.8)
    query = encode_key([math.cos(i) for i in range(64)], b, "multiscale_spiral_v3")
    value = [math.sin(i) for i in range(64)]
    for offset in range(16):
        raw = [math.sin(i + offset) for i in range(64)]
        moved = relocate_key(encode_key(raw, a, "multiscale_spiral_v3"),
                             a, b, "multiscale_spiral_v3")
        direct = encode_key(raw, b, "multiscale_spiral_v3")
        assert sum(q * k for q, k in zip(query, moved)) == pytest.approx(
            sum(q * k for q, k in zip(query, direct)), abs=1e-12)
    assert value == [math.sin(i) for i in range(64)]


def test_bad_geometry_and_shape_fail_closed():
    valid = Position(1.0, 2.0, 0.5)
    for bad in (Position(float("nan"), 0, 0), Position(0, 0, 1), Position(0, 0, -0.1)):
        with pytest.raises(ValueError):
            relocate_key([1.0] * 64, valid, bad, "two_angle_v2")
    with pytest.raises(ValueError):
        encode_key([1.0] * 63, valid, "two_angle_v2")
    with pytest.raises(ValueError):
        encode_key([1.0] * 64, valid, "unknown")
    with pytest.raises(ValueError):
        encode_key([float("inf")] * 64, valid, "two_angle_v2")
    assert positional_kernel(valid, valid, "two_angle_v2") == pytest.approx(1)


def test_layout_is_runtime_supplied_not_model_named():
    layout = Layout(pairs=8, radial_gain=0.02, multiscale_base=1000)
    source, target = Position(2, 1, 0.1), Position(5, 3, 0.4)
    raw = [math.sin(i) for i in range(20)]
    moved = relocate_key(encode_key(raw, source, "multiscale_spiral_v3", layout),
                         source, target, "multiscale_spiral_v3", layout)
    assert moved == pytest.approx(encode_key(raw, target, "multiscale_spiral_v3", layout))
    with pytest.raises(ValueError):
        encode_key(raw, source, "two_angle_v2", Layout(pairs=11))
