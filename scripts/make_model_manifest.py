#!/usr/bin/env python3
"""Writes a runtime model manifest for any local model folder, so the model brings its own
tokenizer and frozen artifacts. Hashes every artifact (SHA-256) from disk.

  scripts/make_model_manifest.py <model_dir> <name> [--out runtime/globalmind-runtime/manifests/<name>.manifest]

The header (workspace, method, configuration and TPPE digests) is copied from an existing
candidate manifest so every model shares one runtime configuration. The manifest records
identity only; it admits nothing.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parents[1]
MANIFESTS = HERE / "runtime/globalmind-runtime/manifests"
HEADER_KEYS = ("workspace", "method", "configuration_digest", "tppe_version", "tppe_configuration_digest")
ROLES = [("weights", re.compile(r"^model-\d+-of-\d+\.safetensors$")), ("mtp-weights", re.compile(r"^model-mtp\.safetensors$")),
         ("weight-index", re.compile(r"^model\.safetensors\.index\.json$")), ("model-config", re.compile(r"^config\.json$")),
         ("tokenizer", re.compile(r"^tokenizer\.json$")), ("tokenizer-config", re.compile(r"^tokenizer_config\.json$")),
         ("chat-template", re.compile(r"^chat_template\.jinja$"))]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def header() -> list[str]:
    for template in sorted(MANIFESTS.glob("*-candidate*.manifest")):
        pairs = dict(line.split(" = ", 1) for line in template.read_text().splitlines() if " = " in line and not line.startswith("#"))
        if all(k in pairs for k in HEADER_KEYS):
            return [f"{k} = {pairs[k]}" for k in HEADER_KEYS]
    sys.exit("no candidate manifest to copy the runtime header from")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        sys.exit(__doc__)
    model_dir, name = pathlib.Path(argv[0]).expanduser().resolve(), argv[1]
    out = pathlib.Path(argv[argv.index("--out") + 1]) if "--out" in argv else MANIFESTS / f"{name.lower()}-candidate.manifest"
    files = sorted(p for p in model_dir.iterdir() if p.is_file())
    entries = []
    for role, pattern in ROLES:
        matched = [p for p in files if pattern.match(p.name)]
        for index, path in enumerate(matched, 1):
            entries.append((f"{role}-{index:05d}" if role == "weights" else role, path))
    if not any(r == "tokenizer" for r, _ in entries):
        sys.exit(f"{model_dir} has no tokenizer.json: every model must bring its own tokenizer")
    lines = [f"# Model manifest for {name}, written by scripts/make_model_manifest.py on {time.strftime('%Y-%m-%d')}.",
             "# Digests computed from disk. Identity only: no eligibility or judge receipt is implied.", *header(), "",
             "model.0.provider = local", f"model.0.name = {name}", f"model.0.revision = local-frozen-{time.strftime('%Y-%m-%d')}",
             "model.0.modality = text", ""]
    for index, (role, path) in enumerate(entries):
        print(f"  hashing {path.name}", flush=True)
        lines += [f"model.0.artifact.{index}.role = {role}", f"model.0.artifact.{index}.path = {path}",
                  f"model.0.artifact.{index}.sha256 = {sha256(path)}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(entries)} artifacts)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
