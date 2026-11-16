"""Assemble an explicitly pinned native nested-KV roster before any forward call."""

from collections.abc import Mapping, Sequence
import hashlib
import inspect
import json

from frontier.kv_contract import (ExactCheckpoint, ExactPageSet, ExactStateSet, HeadPage,
                                  RuntimeIdentity, StateSpec, StateTensor, StateTensorSpec,
                                  TensorSpec, LayeredTensorSpec, MAX_CACHE_BYTES,
                                  _token_bytes, continue_exact)



SESSION_FIELDS = ("workspace", "session", "model_id", "model_revision", "model_digest",
                  "request_digest", "prefix_digest", "tppe_version", "tppe_configuration_digest",
                  "address_contract_digest", "tensor_contract_digest", "geometry_space", "dtype",
                  "layer_count", "kv_head_count", "head_dim", "layout")


def native_position_contract(witness: Mapping) -> str:
    """Runtime-pinned digest of the native transform version and configuration."""
    encoded = json.dumps([witness["tppe_version"], witness["tppe_configuration_digest"]],
                         separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(b"ple-native-position-v1\0" + encoded).hexdigest()


def validate_roster(identity: RuntimeIdentity, prefix_ids: tuple[int, ...],
                    witnesses: Sequence[Mapping], recalled_addresses: set[str],
                    max_bytes: int = 256 * 1024 * 1024) -> tuple[dict, ...]:
    """Validate a frozen snapshot of all metadata before any payload lookup."""
    identity.validate()
    spec = identity.spec
    prefix_digest = hashlib.sha256(_token_bytes(prefix_ids)).hexdigest()
    if not isinstance(spec, (TensorSpec, LayeredTensorSpec)):
        raise ValueError("nested K/V schema cannot persist named or recurrent runtime state")
    layers = spec.layer_specs
    if any(s.layout != "token_major" for s in layers):
        raise ValueError("nested assembly requires a token-major runtime layout")
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_CACHE_BYTES:
        raise ValueError("invalid complete-cache byte budget")
    expected_bytes = sum(len(prefix_ids) * s.kv_heads * s.head_dim * s.bytes_per_value * 2 for s in layers)
    if expected_bytes > max_bytes:
        raise ValueError("complete cache exceeds byte budget")
    if not witnesses or len(witnesses) > 65536:
        raise ValueError("empty or oversized native page roster")
    roster = tuple(dict(q) for q in witnesses)
    first = roster[0]
    common = {name: first.get(name) for name in SESSION_FIELDS}
    if any(value is None for value in common.values()):
        raise ValueError("native session identity is incomplete")
    if first["model_digest"] != identity.model_digest or first["prefix_digest"] != prefix_digest:
        raise ValueError("native cache differs from pinned runtime or prefix")
    # Uniform shorthand retains its single-session guarantee. Explicit layers
    # may use separate native sessions, whose immutable fields are still pinned.
    shared = SESSION_FIELDS if isinstance(spec, TensorSpec) else (
        "workspace", "model_id", "model_revision", "model_digest", "prefix_digest",
        "address_contract_digest", "geometry_space", "layer_count")
    sessions = {}
    layer_sessions = {}
    layer, head, token = 0, 0, 0
    seen = set()
    for q in roster:
        if any(q.get(name) != common[name] for name in shared):
            raise ValueError("native roster mixes sessions or contracts")
        if any(q.get(name) is None for name in SESSION_FIELDS):
            raise ValueError("native session identity is incomplete")
        if any(type(q.get(name)) is not int for name in ("layer_count", "kv_head_count", "head_dim")):
            raise ValueError("native session dimensions must be integers")
        numeric = ("layer", "kv_head", "token_start", "token_end", "revision")
        if any(type(q.get(name)) is not int for name in numeric):
            raise ValueError("native page index must be an integer")
        if (q["layer"], q["kv_head"], q["token_start"]) != (layer, head, token):
            raise ValueError("native roster has missing, duplicate, overlapping or unordered spans")
        if layer >= len(layers):
            raise ValueError("native roster has extra layers")
        s = layers[layer]
        if (native_position_contract(q) != s.position_contract or q["dtype"] != s.dtype
                or q["layout"] != s.layout or (q["layer_count"], q["kv_head_count"], q["head_dim"])
                != (spec.layers, s.kv_heads, s.head_dim)):
            raise ValueError("native cache differs from pinned runtime layer contract")
        session = q["session"]
        if not isinstance(session, str) or not session:
            raise ValueError("native session ID is invalid")
        snapshot = tuple(q[name] for name in SESSION_FIELDS)
        if sessions.setdefault(session, snapshot) != snapshot or layer_sessions.setdefault(layer, session) != session:
            raise ValueError("native roster mixes sessions or contracts within a layer")
        end = q["token_end"]
        if not token < end <= len(prefix_ids) or q.get("tensor_shape") != [end - token, s.head_dim]:
            raise ValueError("native page interval or tensor shape mismatch")
        if q.get("address") not in recalled_addresses:
            raise ValueError("native page address is absent from selected recall")
        page = q.get("page")
        if not isinstance(page, str) or not page or page in seen:
            raise ValueError("native page ID is absent or duplicated")
        seen.add(page)
        for name in ("key_digest", "value_digest", "phase_state_digest"):
            digest = q.get(name)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("native tensor digest is invalid")
        if q["revision"] < 1:
            raise ValueError("native page revision is invalid")
        token = end
        if token == len(prefix_ids):
            token = 0
            head += 1
            if head == s.kv_heads:
                head = 0
                layer += 1
    if (layer, head, token) != (spec.layers, 0, 0):
        raise ValueError("native roster is missing a layer, head or token suffix")
    return roster


def _resolve_tensor_witness(t: StateTensorSpec, idx: int, witnesses: Sequence[Mapping] | Mapping, identity: RuntimeIdentity) -> dict:
    if isinstance(witnesses, Mapping):
        if (t.layer, t.name) in witnesses:
            return dict(witnesses[(t.layer, t.name)])
        if t.name in witnesses:
            return dict(witnesses[t.name])
        w = dict(witnesses)
        w.setdefault("layer", t.layer)
        w.setdefault("name", t.name)
        return w
    if isinstance(witnesses, Sequence):
        for item in witnesses:
            if isinstance(item, Mapping) and item.get("layer") == t.layer and item.get("name") == t.name:
                return dict(item)
        if idx < len(witnesses) and isinstance(witnesses[idx], Mapping):
            return dict(witnesses[idx])
    return {
        "model_digest": identity.model_digest,
        "tokenizer_digest": identity.tokenizer_digest,
        "adapter_digest": identity.adapter_digest,
        "state_contract_digest": identity.spec.state_contract,
        "layer": t.layer,
        "name": t.name,
    }


def _fetch_named_state(mind, witness: dict, identity: RuntimeIdentity, executable: str, recalled_addresses: set[str]) -> bytes:
    fn = mind.named_state_exact
    try:
        sig = inspect.signature(fn)
        pos_params = [
            p for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        kw_only = {
            p.name for p in sig.parameters.values()
            if p.kind == inspect.Parameter.KEYWORD_ONLY
        }
        if len(pos_params) == 1:
            return fn(witness)
        if len(pos_params) == 2 and "recalled_addresses" in kw_only:
            return fn(witness, executable, recalled_addresses=recalled_addresses)
        if len(pos_params) == 3:
            return fn(witness, identity, recalled_addresses)
    except (ValueError, TypeError):
        pass
    try:
        return fn(witness, identity, recalled_addresses)
    except TypeError:
        try:
            return fn(witness)
        except TypeError:
            return fn(witness, executable=executable, recalled_addresses=recalled_addresses)


def assemble_exact(mind, identity: RuntimeIdentity, prefix_ids: tuple[int, ...], witnesses: Sequence[Mapping],
                   executable: str, *, recalled_addresses: set[str],
                   max_bytes: int = 256 * 1024 * 1024) -> ExactCheckpoint:
    identity.validate()
    if isinstance(identity.spec, StateSpec):
        if not hasattr(mind, "named_state_exact"):
            raise ValueError("cannot persist named runtime state: schema extension required")
        if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_CACHE_BYTES:
            raise ValueError("invalid complete-cache byte budget")
        expected_bytes = sum(s.byte_length(len(prefix_ids)) for s in identity.spec.tensors)
        if expected_bytes > max_bytes:
            raise ValueError("complete state exceeds byte budget")
        tensors = []
        for idx, t in enumerate(identity.spec.tensors):
            w = _resolve_tensor_witness(t, idx, witnesses, identity)
            payload = _fetch_named_state(mind, w, identity, executable, recalled_addresses)
            if not isinstance(payload, bytes):
                raise ValueError("named state payload must be bytes")
            if "payload_digest" in w and hashlib.sha256(payload).hexdigest() != w["payload_digest"]:
                raise ValueError("native named state payload digest mismatch")
            tensors.append(StateTensor(t.layer, t.name, payload))
        return ExactStateSet.seal(identity, prefix_ids, tuple(tensors))

    roster = validate_roster(identity, prefix_ids, witnesses, recalled_addresses, max_bytes)
    layers = identity.spec.layer_specs
    pages = []
    keys, values = bytearray(), bytearray()
    for q in roster:
        key, value = mind.tensor_exact(q, executable, recalled_addresses=recalled_addresses)
        layer = layers[q["layer"]]
        each = (q["token_end"] - q["token_start"]) * layer.head_dim * layer.bytes_per_value
        if len(key) != each or len(value) != each:
            raise ValueError("native tensor payload size disagrees with its interval")
        if (hashlib.sha256(key).hexdigest() != q["key_digest"]
                or hashlib.sha256(value).hexdigest() != q["value_digest"]):
            raise ValueError("native tensor payload digest mismatch")
        keys.extend(key)
        values.extend(value)
        if q["token_end"] == len(prefix_ids):
            pages.append(HeadPage(q["layer"], q["kv_head"], bytes(keys), bytes(values)))
            keys.clear()
            values.clear()
    return ExactPageSet.seal(identity, prefix_ids, tuple(pages))


def continue_nested(runtime, mind, prefix_ids, witnesses, continuation_ids, max_tokens,
                    executable, *, recalled_addresses):
    _token_bytes(continuation_ids)
    if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
        raise ValueError("generation limit out of bounds")
    identity = runtime.identity()
    pages = assemble_exact(mind, identity, prefix_ids, witnesses, executable,
                           recalled_addresses=recalled_addresses)
    # Re-read live identity after fetching; a changed model cannot inherit it.
    return continue_exact(runtime, pages, prefix_ids, continuation_ids, max_tokens)
