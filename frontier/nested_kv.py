"""Binary access to the existing SurrealDB nested K/V through its owning Rust runtime."""

import hashlib
import json
import os
import struct
import subprocess


def publish_exact(identity: dict, key: bytes, value: bytes, *, executable: str, env: dict[str, str]) -> None:
    """Append an immutable native page; tensors travel only in the binary body."""
    if type(key) is not bytes or type(value) is not bytes or not key or not value or len(key) + len(value) > 16 * 1024 * 1024:
        raise ValueError("invalid publication tensor bytes or byte budget")
    if (hashlib.sha256(key).hexdigest() != identity.get("key_digest")
            or hashlib.sha256(value).hexdigest() != identity.get("value_digest")):
        raise ValueError("publication tensor digest mismatch")
    metadata = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(metadata) > 65536:
        raise ValueError("publication metadata exceeds bound")
    request = b"PKP1" + struct.pack("<3I", len(metadata), len(key), len(value)) + metadata + key + value
    result = subprocess.run([executable, "put"], input=request, capture_output=True,
                            env={**os.environ, **env}, timeout=60)
    if result.returncode:
        raise RuntimeError("nested KV publication refused: " + result.stderr.decode(errors="replace")[:400])
    if result.stdout != b"PWA1" + hashlib.sha256(request).digest():
        raise ValueError("nested KV publication receipt mismatch")


def fetch_exact(identity: dict, *, executable: str, env: dict[str, str]) -> tuple[bytes, bytes]:
    request = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(request) > 65536:
        raise ValueError("nested KV identity request exceeds bound")
    result = subprocess.run([executable], input=request, capture_output=True,
                            env={**os.environ, **env}, timeout=60)
    if result.returncode:
        raise RuntimeError("nested KV exact lookup refused: " + result.stderr.decode(errors="replace")[:400])
    return decode_exact(request, result.stdout, identity)


def decode_exact(request: bytes, frame: bytes, identity: dict) -> tuple[bytes, bytes]:
    if len(frame) < 44 or frame[:4] != b"PKV1":
        raise ValueError("invalid nested KV binary frame")
    nk, nv = struct.unpack_from("<2I", frame, 4)
    if nk + nv > 16 * 1024 * 1024 or len(frame) != 44 + nk + nv:
        raise ValueError("nested KV frame size mismatch")
    if hashlib.sha256(request + frame[:-32]).digest() != frame[-32:]:
        raise ValueError("nested KV request receipt mismatch")
    key, value = frame[12:12 + nk], frame[12 + nk:-32]
    if (hashlib.sha256(key).hexdigest() != identity["key_digest"]
            or hashlib.sha256(value).hexdigest() != identity["value_digest"]):
        raise ValueError("nested KV tensor digest mismatch")
    return key, value


def publish_named_state_exact(identity: dict, payload: bytes, *, executable: str, env: dict) -> None:
    """Publish an immutable named state page; state travels only in binary body."""
    if type(payload) is not bytes or not payload or len(payload) > 16 * 1024 * 1024:
        raise ValueError("invalid publication payload bytes or byte budget")
    if hashlib.sha256(payload).hexdigest() != identity.get("payload_digest"):
        raise ValueError("publication payload digest mismatch")
    metadata = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(metadata) > 65536:
        raise ValueError("publication metadata exceeds bound")
    request = b"NSP1" + struct.pack("<3I", len(metadata), len(payload), 0) + metadata + payload
    result = subprocess.run([executable, "ns_put"], input=request, capture_output=True,
                            env={**os.environ, **env}, timeout=60)
    if result.returncode:
        raise RuntimeError("nested KV named state publication refused: " + result.stderr.decode(errors="replace")[:400])
    if result.stdout != b"NWA1" + hashlib.sha256(request).digest():
        raise ValueError("nested KV named state publication receipt mismatch")


def fetch_named_state_exact(identity: dict, *, executable: str, env: dict) -> bytes:
    request = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(request) > 65536:
        raise ValueError("nested KV identity request exceeds bound")
    result = subprocess.run([executable, "ns_get"], input=request, capture_output=True,
                            env={**os.environ, **env}, timeout=60)
    if result.returncode:
        raise RuntimeError("nested KV named state lookup refused: " + result.stderr.decode(errors="replace")[:400])
    return decode_named_state_response(request, result.stdout, identity)


def decode_named_state_response(request: bytes, frame: bytes, identity: dict) -> bytes:
    if len(frame) < 40 or frame[:4] != b"NSR1":
        raise ValueError("invalid nested KV named state binary frame")
    (payload_len,) = struct.unpack_from("<I", frame, 4)
    if payload_len > 16 * 1024 * 1024 or len(frame) != 40 + payload_len:
        raise ValueError("nested KV named state frame size mismatch")
    if hashlib.sha256(request + frame[:-32]).digest() != frame[-32:]:
        raise ValueError("nested KV request receipt mismatch")
    payload = frame[8:8 + payload_len]
    if hashlib.sha256(payload).hexdigest() != identity.get("payload_digest"):
        raise ValueError("nested KV payload digest mismatch")
    return payload

