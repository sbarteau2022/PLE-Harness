"""Deterministic native checkpoint addressing; no tensor side store or history scan."""

from dataclasses import asdict, dataclass
import hashlib
import json
import re

from frontier.kv_contract import (ExactPageSet, TensorSpec, LayeredTensorSpec,
                                  MAX_COMPONENTS, MAX_CACHE_BYTES, _sha, _token_bytes)
from frontier.kv_assembly import native_position_contract, validate_roster

FRAGMENT_TOKENS = 256
FRAGMENT_BYTES = 16 * 1024 * 1024


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def record_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value):
        raise ValueError("invalid native record ID")
    return value


@dataclass(frozen=True)
class NativePosition:
    version: str
    configuration_digest: str


@dataclass(frozen=True)
class NativeCacheContext:
    address: str
    geometry_space: str
    address_contract_digest: str
    phase_state_digest: str
    positions: tuple[NativePosition, ...]

    def validate(self, identity):
        identity.validate()
        record_id(self.address)
        record_id(self.geometry_space)
        if not _sha(self.address_contract_digest) or not _sha(self.phase_state_digest):
            raise ValueError("invalid native geometry/phase contract")
        if not isinstance(identity.spec, (TensorSpec, LayeredTensorSpec)):
            raise ValueError("native checkpoint schema cannot persist named runtime state")
        layers = identity.spec.layer_specs
        if type(self.positions) is not tuple or len(self.positions) not in (1, len(layers)):
            raise ValueError("native context needs one position recipe or one per layer")
        for layer, spec in enumerate(layers):
            p = self.positions[0 if len(self.positions) == 1 else layer]
            if (not isinstance(p, NativePosition) or not isinstance(p.version, str) or not 1 <= len(p.version) <= 256
                    or not _sha(p.configuration_digest)):
                raise ValueError("invalid native position recipe")
            if (spec.layout != "token_major" or native_position_contract({"tppe_version": p.version,
                    "tppe_configuration_digest": p.configuration_digest}) != spec.position_contract):
                raise ValueError("native position/layout differs from runtime contract")


def prefix_anchors(prompt_ids):
    """Hash all strict power-of-two prefixes in one linear pass over token bytes."""
    raw = _token_bytes(prompt_ids)
    result, previous, length = [], 0, 1
    digest = hashlib.sha256()
    while length < len(prompt_ids):
        digest.update(raw[previous * 4:length * 4])
        result.append((length, digest.hexdigest()))
        previous, length = length, length * 2
    return tuple(result)


def eligible_anchors(identity, prompt_ids):
    """Apply byte/component budgets without loading or allocating tensor data."""
    identity.validate()
    if not isinstance(identity.spec, (TensorSpec, LayeredTensorSpec)):
        raise ValueError("native checkpoint schema cannot persist named runtime state")
    layers = identity.spec.layer_specs
    per_token = sum(s.kv_heads * s.head_dim * s.bytes_per_value * 2 for s in layers)
    steps = tuple(min(FRAGMENT_TOKENS, FRAGMENT_BYTES // (s.head_dim * s.bytes_per_value * 2)) for s in layers)
    candidates = prefix_anchors(prompt_ids)
    if any(step == 0 for step in steps):
        return ()
    return tuple((length, digest) for length, digest in candidates
                 if length * per_token <= MAX_CACHE_BYTES and
                 sum(s.kv_heads * ((length + step - 1) // step) for s, step in zip(layers, steps)) <= MAX_COMPONENTS)


def plan(mind, identity, prefix_ids, context):
    if not isinstance(context, NativeCacheContext):
        raise ValueError("checkpoint requires a typed native context")
    context.validate(identity)
    workspace = record_id(mind.workspace.removeprefix("workspace:"))
    prefix_digest = hashlib.sha256(_token_bytes(prefix_ids)).hexdigest()
    contract = _hash(["ple-runtime-state-v1", asdict(identity)])
    bundle = _hash(["ple-checkpoint-v1", workspace, mind.model, contract, prefix_digest, asdict(context), FRAGMENT_TOKENS, FRAGMENT_BYTES])
    layers = identity.spec.layer_specs
    size = sum(len(prefix_ids) * s.kv_heads * s.head_dim * s.bytes_per_value * 2 for s in layers)
    if size > MAX_CACHE_BYTES:
        raise ValueError("complete checkpoint exceeds byte budget")
    roster = []
    for layer, s in enumerate(layers):
        p = context.positions[0 if len(context.positions) == 1 else layer]
        session = "plecp_" + _hash([bundle, "session", 0 if isinstance(identity.spec, TensorSpec) else layer])
        step = min(FRAGMENT_TOKENS, FRAGMENT_BYTES // (2 * s.head_dim * s.bytes_per_value))
        if step < 1:
            raise ValueError("one tensor token exceeds fragment byte budget")
        for head in range(s.kv_heads):
            for start in range(0, len(prefix_ids), step):
                end = min(start + step, len(prefix_ids))
                roster.append(dict(workspace=workspace, session=session,
                    page="plecp_" + _hash([bundle, layer, head, start, end]), model_id=mind.model,
                    model_revision=identity.model_digest, model_digest=identity.model_digest,
                    request_digest=_hash(["ple-session-request-v1", session]), prefix_digest=prefix_digest,
                    tppe_version=p.version, tppe_configuration_digest=p.configuration_digest,
                    address_contract_digest=context.address_contract_digest, tensor_contract_digest=contract,
                    geometry_space=context.geometry_space, dtype=s.dtype, layer_count=len(layers), kv_head_count=s.kv_heads,
                    head_dim=s.head_dim, layer=layer, kv_head=head, token_start=start, token_end=end,
                    tensor_shape=[end - start, s.head_dim], layout=s.layout, address=context.address,
                    phase_state_digest=context.phase_state_digest, revision=1))
                if len(roster) > MAX_COMPONENTS:
                    raise ValueError("checkpoint fragment budget exceeded")
    return tuple(roster)


def page_receipt(q):
    return _hash(["ple-page-receipt-v1", q["page"], q["key_digest"], q["value_digest"]])


def _same(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def discover(mind, identity, prefix_ids, context):
    expected = plan(mind, identity, prefix_ids, context)
    atom = mind.tensor_grounding(context.address, context.geometry_space)
    sessions = mind.tensor_metadata("transformer_kv_session", tuple(dict.fromkeys(q["session"] for q in expected)))
    missing = False
    for q in expected:
        s = sessions.get(q["session"])
        if s is None:
            missing = True
            continue
        fields = {name: q[name] for name in ("model_id", "model_revision", "model_digest", "request_digest", "prefix_digest",
                  "address_contract_digest", "tensor_contract_digest", "tppe_configuration_digest", "dtype", "layer_count", "kv_head_count", "head_dim")}
        fields.update(workspace="workspace:" + q["workspace"], geometry_space="geometry_space:" + q["geometry_space"],
                      toroidal_phase_encoder_version=q["tppe_version"], active=True)
        if any(not _same(s.get(name), value) for name, value in fields.items()):
            raise ValueError("stored checkpoint session differs from runtime/context")
    if missing:
        return None
    pages = mind.tensor_metadata("transformer_kv_page", tuple(q["page"] for q in expected))
    roster = []
    for q in expected:
        stored = pages.get(q["page"])
        if stored is None:
            missing = True
            continue
        fields = {name: q[name] for name in ("layer", "kv_head", "token_start", "token_end", "tensor_shape", "dtype", "layout", "phase_state_digest", "revision")}
        fields.update(workspace="workspace:" + q["workspace"], session="transformer_kv_session:" + q["session"],
                      address="atom_address:" + q["address"], active=True, trajectory_atoms=["atom:" + atom])
        if any(not _same(stored.get(name), value) for name, value in fields.items()):
            raise ValueError("stored checkpoint page differs from runtime/context")
        q = {**q, "key_digest": stored.get("key_digest"), "value_digest": stored.get("value_digest")}
        if not _sha(q["key_digest"]) or not _sha(q["value_digest"]) or stored.get("receipt_digest") != page_receipt(q):
            raise ValueError("stored checkpoint payload/lineage digest is invalid")
        roster.append(q)
    if missing:
        return None
    return validate_roster(identity, prefix_ids, roster, {context.address})


def publish(mind, checkpoint, context, executable):
    if not isinstance(checkpoint, ExactPageSet):
        raise ValueError("native publication requires a complete K/V checkpoint")
    checkpoint.admit(checkpoint.identity, checkpoint.prefix_ids)
    expected = plan(mind, checkpoint.identity, checkpoint.prefix_ids, context)
    atom = mind.tensor_grounding(context.address, context.geometry_space)
    pages = {(p.layer, p.head): p for p in checkpoint.pages}
    layers = checkpoint.identity.spec.layer_specs
    prepared = []
    for q in expected:
        page = pages[q["layer"], q["kv_head"]]
        width = layers[q["layer"]].head_dim * layers[q["layer"]].bytes_per_value
        start, end = q["token_start"] * width, q["token_end"] * width
        key, value = page.keys[start:end], page.values[start:end]
        q = {**q, "key_digest": hashlib.sha256(key).hexdigest(), "value_digest": hashlib.sha256(value).hexdigest(),
             "trajectory_atoms": [atom]}
        q["receipt_digest"] = page_receipt(q)
        prepared.append((q, key, value))
    validate_roster(checkpoint.identity, checkpoint.prefix_ids, [q for q, _, _ in prepared], {context.address})
    for q, key, value in prepared:
        mind.tensor_put(q, key, value, executable, recalled_addresses={context.address})
    return tuple(q for q, _, _ in prepared)
