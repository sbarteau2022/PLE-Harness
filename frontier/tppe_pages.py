"""PLE harness reference for TPPE key relocation; no model-forward injection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

ARMS = {"two_angle_v2", "multiscale_spiral_v3"}


@dataclass(frozen=True)
class Layout:
    pairs: int = 32
    radial_gain: float = 0.05
    multiscale_base: float = 1e4

    def validate(self) -> None:
        if not 1 <= self.pairs <= 4096:
            raise ValueError("rotary pair count out of bounds")
        if not math.isfinite(self.radial_gain) or self.radial_gain < 0:
            raise ValueError("radial gain must be finite and nonnegative")
        if not math.isfinite(self.multiscale_base) or self.multiscale_base <= 1:
            raise ValueError("multiscale base must be finite and greater than one")


@dataclass(frozen=True)
class Position:
    fast: float  # unwrapped TPPE phase
    slow: float  # unwrapped TPPE phase
    radius: float  # Poincare radius

    def validate(self) -> None:
        if not all(math.isfinite(x) for x in (self.fast, self.slow, self.radius)):
            raise ValueError("TPPE position must be finite")
        if not 0 <= self.radius < 1:
            raise ValueError("TPPE radius must be inside the Poincare ball")


def _angle(position: Position, pair: int, arm: str, layout: Layout) -> float:
    phase = position.fast if pair % 2 == 0 else position.slow
    if arm == "two_angle_v2":
        return phase % math.tau
    return phase * layout.multiscale_base ** (-(pair // 2) / ((layout.pairs + 1) // 2))


def _validate(vector: Sequence[float], arm: str, layout: Layout, *positions: Position) -> None:
    layout.validate()
    if arm not in ARMS:
        raise ValueError("unknown TPPE arm")
    if len(vector) < layout.pairs * 2 or not all(math.isfinite(x) for x in vector):
        raise ValueError("key must contain a finite rotary domain")
    for position in positions:
        position.validate()


def _rotate(vector: Sequence[float], angles: Sequence[float], scale: float, pairs: int) -> list[float]:
    result = list(vector)
    for pair, angle in enumerate(angles):
        sine, cosine = math.sin(angle), math.cos(angle)
        x, y = vector[pair], vector[pair + pairs]
        result[pair] = scale * (x * cosine - y * sine)
        result[pair + pairs] = scale * (y * cosine + x * sine)
    return result


def encode_key(raw: Sequence[float], position: Position, arm: str,
               layout: Layout = Layout()) -> list[float]:
    """Encode the positional part of an already projected and normalized Q/K."""
    _validate(raw, arm, layout, position)
    angles = [_angle(position, pair, arm, layout) for pair in range(layout.pairs)]
    return _rotate(raw, angles, 1 + layout.radial_gain * position.radius, layout.pairs)


def relocate_key(encoded: Sequence[float], source: Position, target: Position, arm: str,
                 layout: Layout = Layout()) -> list[float]:
    """Move only the position transform of a K; hidden-state context is unchanged."""
    _validate(encoded, arm, layout, source, target)
    angles = [_angle(target, pair, arm, layout) - _angle(source, pair, arm, layout)
              for pair in range(layout.pairs)]
    scale = (1 + layout.radial_gain * target.radius) / (1 + layout.radial_gain * source.radius)
    return _rotate(encoded, angles, scale, layout.pairs)


def positional_kernel(a: Position, b: Position, arm: str,
                      layout: Layout = Layout()) -> float:
    _validate([0.0] * (layout.pairs * 2), arm, layout, a, b)
    return sum(math.cos(_angle(a, i, arm, layout) - _angle(b, i, arm, layout))
               for i in range(layout.pairs)) / layout.pairs
