"""Model-neutral admission gate for exact local runtime state continuation.

The normal driver-seat conversation uses text/trajectory tools for every model.
Only a local runtime with this explicit capability may import opaque tensor pages.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Protocol

MAX_COMPONENTS = 65536
MAX_CACHE_BYTES = 256 * 1024 * 1024


def _sha(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _finite(payload: bytes, dtype: str) -> None:
    if dtype == "bf16":
        finite = all((bits & 0x7F80) != 0x7F80 for (bits,) in struct.iter_unpack("<H", payload))
    elif dtype in ("f32", "f16"):
        finite = all(math.isfinite(value) for (value,) in
                     struct.iter_unpack("<f" if dtype == "f32" else "<e", payload))
    else:
        return  # Integer/opaque packing semantics belong to the runtime.
    if not finite:
        raise ValueError("tensor payload contains non-finite values")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _token_bytes(ids: tuple[int, ...]) -> bytes:
    if type(ids) is not tuple or not ids or len(ids) > 1_000_000 or any(type(i) is not int or not 0 <= i <= 0xFFFFFFFF for i in ids):
        raise ValueError("invalid token prefix")
    return struct.pack(f"<{len(ids)}I", *ids)


@dataclass(frozen=True)
class TensorSpec:
    layers: int
    kv_heads: int
    head_dim: int
    dtype: str
    layout: str
    position_contract: str

    def validate(self) -> None:
        if (any(type(x) is not int for x in (self.layers, self.kv_heads, self.head_dim))
                or not (1 <= self.layers * self.kv_heads <= MAX_COMPONENTS
                        and self.layers > 0 and self.kv_heads > 0 and 1 <= self.head_dim <= 0xFFFFFFFF)):
            raise ValueError("tensor shape out of bounds")
        if (self.dtype not in ("f32", "f16", "bf16") or not isinstance(self.layout, str) or not self.layout
                or not isinstance(self.position_contract, str) or not self.position_contract):
            raise ValueError("unsupported or unidentified tensor format")

    @property
    def bytes_per_value(self) -> int:
        self.validate()
        return 4 if self.dtype == "f32" else 2

    @property
    def layer_specs(self) -> tuple["AttentionLayerSpec", ...]:
        self.validate()
        layer = AttentionLayerSpec(self.kv_heads, self.head_dim, self.dtype, self.layout, self.position_contract)
        return (layer,) * self.layers


@dataclass(frozen=True)
class AttentionLayerSpec:
    kv_heads: int
    head_dim: int
    dtype: str
    layout: str
    position_contract: str

    def validate(self) -> None:
        TensorSpec(1, self.kv_heads, self.head_dim, self.dtype, self.layout, self.position_contract).validate()

    @property
    def bytes_per_value(self) -> int:
        self.validate()
        return 4 if self.dtype == "f32" else 2


@dataclass(frozen=True)
class LayeredTensorSpec:
    layer_specs: tuple[AttentionLayerSpec, ...]

    @property
    def layers(self) -> int:
        return len(self.layer_specs)

    def validate(self) -> None:
        if type(self.layer_specs) is not tuple or not 1 <= len(self.layer_specs) <= MAX_COMPONENTS:
            raise ValueError("invalid per-layer cache specification")
        for layer in self.layer_specs:
            if not isinstance(layer, AttentionLayerSpec):
                raise ValueError("invalid attention layer descriptor")
            layer.validate()
        if sum(layer.kv_heads for layer in self.layer_specs) > MAX_COMPONENTS:
            raise ValueError("cache component budget exceeded")


@dataclass(frozen=True)
class StateTensorSpec:
    """One runtime-owned buffer; 'tokens' is the sole dynamic shape axis."""

    layer: int
    name: str
    shape: tuple[int | str, ...]
    dtype: str
    layout: str
    contract: str
    token_limit: int | None = None

    def validate(self) -> None:
        if type(self.layer) is not int or self.layer < 0 or not isinstance(self.name, str) or not self.name or len(self.name) > 128:
            raise ValueError("invalid state tensor identity")
        if (type(self.shape) is not tuple or len(self.shape) > 16
                or any(not (type(x) is int and 1 <= x <= 0xFFFFFFFF) and x != "tokens" for x in self.shape)
                or self.shape.count("tokens") > 1):
            raise ValueError("invalid state tensor shape")
        if self.dtype not in ("f32", "f16", "bf16", "i32", "i64", "bytes") or not isinstance(self.layout, str) or not self.layout or not _sha(self.contract):
            raise ValueError("unidentified state encoding or contract")
        if self.token_limit is not None and (type(self.token_limit) is not int or self.token_limit < 1 or "tokens" not in self.shape):
            raise ValueError("invalid token window")

    def byte_length(self, tokens: int) -> int:
        self.validate()
        extent = min(tokens, self.token_limit) if self.token_limit is not None else tokens
        width = {"f32": 4, "f16": 2, "bf16": 2, "i32": 4, "i64": 8, "bytes": 1}[self.dtype]
        return math.prod(extent if x == "tokens" else x for x in self.shape) * width


@dataclass(frozen=True)
class StateSpec:
    layers: int
    tensors: tuple[StateTensorSpec, ...]
    state_contract: str

    def validate(self) -> None:
        if type(self.layers) is not int or not 1 <= self.layers <= MAX_COMPONENTS or not _sha(self.state_contract):
            raise ValueError("invalid runtime state specification")
        if type(self.tensors) is not tuple or not 1 <= len(self.tensors) <= MAX_COMPONENTS:
            raise ValueError("invalid state tensor roster")
        previous = (-1, "")
        for tensor in self.tensors:
            if not isinstance(tensor, StateTensorSpec):
                raise ValueError("invalid state tensor descriptor")
            tensor.validate()
            current = (tensor.layer, tensor.name)
            if tensor.layer >= self.layers or current <= previous:
                raise ValueError("state roster is duplicated, unordered or outside model layers")
            previous = current


@dataclass(frozen=True)
class RuntimeIdentity:
    model_digest: str
    tokenizer_digest: str
    adapter_digest: str
    spec: TensorSpec | LayeredTensorSpec | StateSpec

    def validate(self) -> None:
        if not isinstance(self.spec, (TensorSpec, LayeredTensorSpec, StateSpec)):
            raise ValueError("runtime needs a typed state specification")
        self.spec.validate()
        for digest in (self.model_digest, self.tokenizer_digest, self.adapter_digest):
            if not _sha(digest):
                raise ValueError("runtime identity needs exact SHA-256 digests")


@dataclass(frozen=True)
class HeadPage:
    layer: int
    head: int
    keys: bytes
    values: bytes


@dataclass(frozen=True)
class ExactPageSet:
    identity: RuntimeIdentity
    prefix_ids: tuple[int, ...]
    prefix_digest: str
    pages: tuple[HeadPage, ...]
    roster_digest: str

    @classmethod
    def seal(cls, identity: RuntimeIdentity, prefix_ids: tuple[int, ...],
             pages: tuple[HeadPage, ...]) -> "ExactPageSet":
        identity.validate()
        prefix = _token_bytes(prefix_ids)
        roster = cls._roster(identity, prefix_ids, pages)
        return cls(identity, prefix_ids, _digest(prefix), pages, _digest(roster))

    @staticmethod
    def _roster(identity: RuntimeIdentity, prefix_ids: tuple[int, ...],
                pages: tuple[HeadPage, ...]) -> bytes:
        spec = identity.spec
        if not isinstance(spec, (TensorSpec, LayeredTensorSpec)):
            raise ValueError("named runtime state cannot be represented as K/V head pages")
        layers = spec.layer_specs
        expected = sum(layer.kv_heads for layer in layers)
        if type(pages) is not tuple or len(pages) != expected:
            raise ValueError("incomplete layer/head page roster")
        if sum(len(prefix_ids) * s.kv_heads * s.head_dim * s.bytes_per_value * 2 for s in layers) > MAX_CACHE_BYTES:
            raise ValueError("complete cache exceeds byte budget")
        roster = bytearray(b"ple-exact-pages-v1\0")
        slots = ((layer, head, s) for layer, s in enumerate(layers) for head in range(s.kv_heads))
        for page, (layer, head, s) in zip(pages, slots):
            if not isinstance(page, HeadPage) or type(page.layer) is not int or type(page.head) is not int or (page.layer, page.head) != (layer, head):
                raise ValueError("page roster order or identity mismatch")
            each = len(prefix_ids) * s.head_dim * s.bytes_per_value
            for payload in (page.keys, page.values):
                if type(payload) is not bytes:
                    raise ValueError("tensor payload must be immutable bytes")
                if len(payload) != each:
                    raise ValueError("page tensor shape mismatch")
                _finite(payload, s.dtype)
            roster.extend(struct.pack("<2I", page.layer, page.head))
            roster.extend(_digest(page.keys).encode())
            roster.extend(_digest(page.values).encode())
        return bytes(roster)

    def admit(self, live: RuntimeIdentity, intended_prefix: tuple[int, ...]) -> None:
        live.validate()
        if live != self.identity:
            raise ValueError("local runtime identity or tensor contract changed")
        if intended_prefix != self.prefix_ids or _digest(_token_bytes(intended_prefix)) != self.prefix_digest:
            raise ValueError("causal prefix changed")
        if _digest(self._roster(self.identity, self.prefix_ids, self.pages)) != self.roster_digest:
            raise ValueError("tensor page digest changed")


@dataclass(frozen=True)
class StateTensor:
    layer: int
    name: str
    payload: bytes


@dataclass(frozen=True)
class ExactStateSet:
    identity: RuntimeIdentity
    prefix_ids: tuple[int, ...]
    prefix_digest: str
    tensors: tuple[StateTensor, ...]
    roster_digest: str

    @staticmethod
    def _roster(identity, prefix_ids, tensors) -> bytes:
        spec = identity.spec
        if not isinstance(spec, StateSpec):
            raise ValueError("named state requires a StateSpec")
        if type(tensors) is not tuple or len(tensors) != len(spec.tensors):
            raise ValueError("incomplete runtime state roster")
        sizes = tuple(s.byte_length(len(prefix_ids)) for s in spec.tensors)
        if sum(sizes) > MAX_CACHE_BYTES:
            raise ValueError("complete state exceeds byte budget")
        roster = bytearray(b"ple-exact-state-v1\0")
        for tensor, descriptor, size in zip(tensors, spec.tensors, sizes):
            if (not isinstance(tensor, StateTensor) or type(tensor.layer) is not int
                    or (tensor.layer, tensor.name) != (descriptor.layer, descriptor.name)):
                raise ValueError("runtime state tensor identity mismatch")
            if type(tensor.payload) is not bytes or len(tensor.payload) != size:
                raise ValueError("runtime state tensor shape mismatch")
            _finite(tensor.payload, descriptor.dtype)
            roster.extend(json.dumps([tensor.layer, tensor.name, _digest(tensor.payload)],
                                     ensure_ascii=True, separators=(",", ":")).encode())
        return bytes(roster)

    @classmethod
    def seal(cls, identity, prefix_ids, tensors) -> "ExactStateSet":
        identity.validate()
        prefix = _token_bytes(prefix_ids)
        roster = cls._roster(identity, prefix_ids, tensors)
        return cls(identity, prefix_ids, _digest(prefix), tensors, _digest(roster))

    def admit(self, live: RuntimeIdentity, intended_prefix: tuple[int, ...]) -> None:
        live.validate()
        if live != self.identity:
            raise ValueError("local runtime identity or state contract changed")
        if intended_prefix != self.prefix_ids or _digest(_token_bytes(intended_prefix)) != self.prefix_digest:
            raise ValueError("causal prefix changed")
        if _digest(self._roster(live, self.prefix_ids, self.tensors)) != self.roster_digest:
            raise ValueError("runtime state digest changed")


ExactCheckpoint = ExactPageSet | ExactStateSet


class LocalPageRuntime(Protocol):
    """Opt-in adapter for a local model; hosted drivers need no implementation."""

    def identity(self) -> RuntimeIdentity: ...

    def continue_from_pages(self, pages: ExactCheckpoint, continuation_ids: tuple[int, ...],
                            max_tokens: int) -> tuple[int, ...]: ...


def continue_exact(runtime: LocalPageRuntime, pages: ExactCheckpoint,
                   prefix_ids: tuple[int, ...], continuation_ids: tuple[int, ...],
                   max_tokens: int) -> tuple[int, ...]:
    """Admit an exact prefix before calling a model-specific local adapter."""
    if not isinstance(pages, (ExactPageSet, ExactStateSet)):
        raise ValueError("runtime continuation requires a typed checkpoint")
    pages.admit(runtime.identity(), prefix_ids)
    _token_bytes(continuation_ids)
    if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
        raise ValueError("generation limit out of bounds")
    output = runtime.continue_from_pages(pages, continuation_ids, max_tokens)
    if runtime.identity() != pages.identity:
        raise ValueError("local runtime identity changed during continuation")
    if not isinstance(output, tuple) or len(output) > max_tokens:
        raise ValueError("runtime returned an invalid token sequence or exceeded its limit")
    _token_bytes(output)
    return output
