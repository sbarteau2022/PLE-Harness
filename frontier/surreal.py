"""SurrealDB, native to the harness: the HyperMind atoms and nested hyperbolic KV.

The harness talks to SurrealDB directly. Models get it two ways: relevant atoms are put into
their context automatically, and tools let them search, walk and mint.

Shared mind (database `hypermind/core`)
    atoms            distilled knowledge: text, domain, H3xT2 address (lorentz4 + torus2)
    transformer_kv   tensor pages addressed by atom address (metadata; payloads on request)
    Reads are SELECTs built here. Writes go only through `Mind`, which follows the schema's
    lineage tables and marks every row with provenance `ple-harness`. Models never send SurrealQL.

Short-term memory defaults to a local SQLite file. The older `frontier/n3`
SurrealDB backend remains available only when explicitly selected; it is not
the shared atom and tensor-page object.

Coordinates
    The harness places an entry in the Poincare 3-ball (direction = topic, radius = depth).
    That is the same manifold as H3, so the point converts exactly to Lorentz coordinates:
    x0 = (1 + r^2) / (1 - r^2), xi = 2 pi / (1 - r^2). The torus phases are a harness
    convention (order in the store, session), not the TPPE positional encoder.

Credentials: HYPERMIND_SURREAL_URL / _USER / _PASS. If the password is not set, it is read
from the arguments of the running local `surreal start` process, so no secret is stored.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from frontier import n3  # noqa: E402

URL = os.environ.get("HYPERMIND_SURREAL_URL", "http://127.0.0.1:8000")
CORE_NS = os.environ.get("HYPERMIND_SURREAL_NS", "hypermind")
CORE_DB = os.environ.get("HYPERMIND_SURREAL_DB", "core")
MEM_NS = os.environ.get("FRONTIER_SURREAL_NS", "frontier")
MEM_DB = os.environ.get("FRONTIER_SURREAL_DB", "n3")
GOLDEN = 0.6180339887498949

SCHEMA = """
DEFINE TABLE IF NOT EXISTS n3_space SCHEMAFULL COMMENT 'Address-space contract for harness memory.';
DEFINE FIELD IF NOT EXISTS manifold ON n3_space TYPE string;
DEFINE FIELD IF NOT EXISTS coordinate_model ON n3_space TYPE string;
DEFINE FIELD IF NOT EXISTS note ON n3_space TYPE string;
UPSERT n3_space:h3_t2 SET manifold = 'H3xT2', coordinate_model = 'lorentz4-plus-torus2',
  note = 'h3_lorentz converted exactly from a Poincare 3-ball placement; t2_phase is a harness convention (store order, session), not TPPE';
DEFINE TABLE IF NOT EXISTS n3_entry SCHEMAFULL COMMENT 'Harness memory nested in the hyperbolic KV address model.';
DEFINE FIELD IF NOT EXISTS scope ON n3_entry TYPE string;
DEFINE FIELD IF NOT EXISTS n ON n3_entry TYPE int;
DEFINE FIELD IF NOT EXISTS session ON n3_entry TYPE string;
DEFINE FIELD IF NOT EXISTS kind ON n3_entry TYPE string;
DEFINE FIELD IF NOT EXISTS key ON n3_entry TYPE string;
DEFINE FIELD IF NOT EXISTS text ON n3_entry TYPE string;
DEFINE FIELD IF NOT EXISTS tags ON n3_entry TYPE string;
DEFINE FIELD IF NOT EXISTS parent ON n3_entry TYPE option<int>;
DEFINE FIELD IF NOT EXISTS depth ON n3_entry TYPE int;
DEFINE FIELD IF NOT EXISTS poincare ON n3_entry TYPE array<float>;
DEFINE FIELD IF NOT EXISTS h3_lorentz ON n3_entry TYPE array<float>;
DEFINE FIELD IF NOT EXISTS t2_phase ON n3_entry TYPE array<float>;
DEFINE FIELD IF NOT EXISTS pinned ON n3_entry TYPE bool DEFAULT false;
DEFINE FIELD IF NOT EXISTS created_at ON n3_entry TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS n3_entry_scope_n ON n3_entry FIELDS scope, n UNIQUE;
"""


class SurrealError(RuntimeError):
    pass


def credentials() -> tuple[str, str]:
    user = os.environ.get("HYPERMIND_SURREAL_USER", "root")
    password = os.environ.get("HYPERMIND_SURREAL_PASS", "")
    if not password:
        args = subprocess.run(["ps", "-axo", "args"], capture_output=True, text=True).stdout
        for line in args.splitlines():
            if "surreal" in line and " start" in line and "--pass" in line:
                found = re.search(r"--pass\s+(\S+)", line)
                user_found = re.search(r"--user\s+(\S+)", line)
                if found:
                    password = found.group(1)
                    user = user_found.group(1) if user_found else user
                    break
    if not password:
        raise SurrealError("no SurrealDB credentials: set HYPERMIND_SURREAL_USER and HYPERMIND_SURREAL_PASS")
    return user, password


def sql(ns: str | None, db: str | None, query: str, variables: dict[str, str] | None = None, timeout: int = 30) -> list:
    """Runs statements built by this module. `variables` become $name string parameters."""
    user, password = credentials()
    headers = {"Accept": "application/json", "Content-Type": "text/plain",
               "Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}
    if ns:
        headers["Surreal-NS"] = ns
    if db:
        headers["Surreal-DB"] = db
    url = URL.rstrip("/") + "/sql" + ("?" + urllib.parse.urlencode(variables) if variables else "")
    request = urllib.request.Request(url, data=query.encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            replies = json.loads(response.read())
    except Exception as exc:
        raise SurrealError(f"SurrealDB request failed: {type(exc).__name__}: {exc}") from None
    for reply in replies:
        if reply.get("status") != "OK":
            raise SurrealError(f"SurrealDB: {str(reply.get('result'))[:300]}")
    return [reply["result"] for reply in replies]


def available() -> bool:
    try:
        sql(None, None, "RETURN 1;", timeout=4)
        return True
    except SurrealError:
        return False


# ---------- geometry ----------

def poincare_to_lorentz(point) -> list[float]:
    r2 = sum(v * v for v in point)
    scale = 1.0 / max(1.0 - r2, 1e-9)
    return [(1.0 + r2) * scale] + [2.0 * v * scale for v in point]


def lorentz_distance(u, v) -> float:
    inner = u[0] * v[0] - sum(a * b for a, b in zip(u[1:], v[1:]))
    return math.acosh(max(inner, 1.0))


def torus_distance(a, b, period: float = 1.0) -> float:
    total = 0.0
    for x, y in zip(a, b):
        d = abs(x - y) % period
        total += min(d, period - d) ** 2
    return math.sqrt(total)


def _terms(query: str, limit: int = 5) -> list[str]:
    return sorted(set(n3._tokens(query)), key=len, reverse=True)[:limit]


def _contains(field: str, terms: list[str]) -> tuple[str, dict[str, str]]:
    clause = " OR ".join(f"string::lowercase({field}) CONTAINS $t{i}" for i in range(len(terms)))
    return clause or "true", {f"t{i}": term for i, term in enumerate(terms)}


# ---------- atoms and tensor pages (read-only) ----------

_ADDRESSES: dict[str, tuple[list[float], list[float]]] = {}


def atom_search(query: str, k: int = 5) -> list[dict]:
    terms = _terms(query)
    clause, variables = _contains("content_text", terms)
    rows = sql(CORE_NS, CORE_DB,
               "SELECT id, domain, salience, string::slice(content_text, 0, 1400) AS text, "
               "current_address.h3_lorentz AS h3 FROM atom WHERE " + clause + " LIMIT 400;", variables)[0]
    def hits(row):
        text = (row.get("text") or "").lower()
        return sum(text.count(term) > 0 for term in terms), sum(text.count(term) for term in terms)
    rows.sort(key=hits, reverse=True)
    return [{"id": str(r["id"]), "domain": r.get("domain"), "text": r.get("text") or ""} for r in rows[:k]]


def atom_get(atom_id: str) -> dict | None:
    if not re.fullmatch(r"atom:[\w\-]+", atom_id):
        raise SurrealError("atom ids look like atom:123")
    rows = sql(CORE_NS, CORE_DB, f"SELECT id, workspace, domain, content_kind, content_text, content_uri, "
               f"content_digest, salience, "
               f"current_address.h3_lorentz AS h3_lorentz, current_address.t2_phase AS t2_phase FROM {atom_id};")[0]
    return ({**rows[0], "id": str(rows[0]["id"])}) if rows else None


def _addresses() -> dict[str, tuple[list[float], list[float]]]:
    if not _ADDRESSES:
        rows = sql(CORE_NS, CORE_DB, "SELECT atom, h3_lorentz, t2_phase FROM atom_address WHERE active = true;")[0]
        for row in rows:
            _ADDRESSES[str(row["atom"])] = (row["h3_lorentz"], row.get("t2_phase") or [0.0, 0.0])
    return _ADDRESSES


def atom_neighbors(atom_id: str, k: int = 5) -> list[dict]:
    """Nearest atoms in H3 (Lorentz distance), with torus phase as a tie-break."""
    table = _addresses()
    if atom_id not in table:
        raise SurrealError(f"no active address for {atom_id}")
    home = table[atom_id]
    # The live map's retrieval cost: hyperbolic distance + 0.25 * phase chord.
    ranked = sorted((lorentz_distance(home[0], addr[0]) + 0.25 * phase_chord(home[1], addr[1]), other)
                    for other, addr in table.items() if other != atom_id)[:k]
    ids = ", ".join(other for _, other in ranked)
    rows = sql(CORE_NS, CORE_DB, f"SELECT id, domain, string::slice(content_text, 0, 900) AS text FROM {ids};")[0]
    by_id = {str(r["id"]): r for r in rows}
    return [{"id": other, "distance": round(distance, 4), "domain": by_id.get(other, {}).get("domain"),
             "text": by_id.get(other, {}).get("text", "")} for distance, other in ranked]


def kv_pages(limit: int = 20) -> list[dict]:
    """Tensor KV page metadata (never the payload): what exists and where it is addressed."""
    rows = sql(CORE_NS, CORE_DB,
               "SELECT id, session, layer, kv_head, token_start, token_end, tensor_shape, dtype, layout, "
               "address.h3_lorentz AS h3_lorentz, address.t2_phase AS t2_phase, trajectory_atoms, hits, active "
               f"FROM transformer_kv_page LIMIT {int(limit)};")[0]
    return [{k: (str(v) if k in ("id", "session") else v) for k, v in row.items()} for row in rows]


# ---------- harness memory nested in SurrealDB ----------

_SCHEMA_READY = False


def ensure_memory_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    sql(None, None, f"DEFINE NAMESPACE IF NOT EXISTS {MEM_NS};")
    sql(MEM_NS, None, f"DEFINE DATABASE IF NOT EXISTS {MEM_DB};")
    sql(MEM_NS, MEM_DB, SCHEMA)
    _SCHEMA_READY = True


class SurrealN3(n3.N3):
    """The N3 memory interface, stored in SurrealDB instead of SQLite.

    `scope` separates one project from another (default: a hash of the target folder), so
    several harnesses and sessions working on the same folder share one memory.
    """

    def __init__(self, scope: str, session: str = "") -> None:  # pylint: disable=super-init-not-called
        ensure_memory_schema()
        self.scope = re.sub(r"[^A-Za-z0-9_\-\.]", "_", scope)[:80]
        self.session = session or time.strftime("%Y%m%d-%H%M%S")
        self.fts = False

    def _rows(self, where: str, variables: dict[str, str] | None = None, limit: int = 200) -> list[dict]:
        merged = {"scope": self.scope, **(variables or {})}
        return sql(MEM_NS, MEM_DB, "SELECT n, kind, key, text, tags, depth, pinned, poincare FROM n3_entry "
                   f"WHERE scope = $scope AND ({where}) ORDER BY n DESC LIMIT {int(limit)};", merged)[0]

    def deposit(self, text: str, key: str = "", tags: str = "", kind: str = "note",
                parent: int | None = None, pin: bool = False) -> int:
        depth, parent_point = 1, None
        if parent is not None:
            found = self._rows(f"n = {int(parent)}", limit=1)
            if found:
                depth, parent_point = found[0]["depth"] + 1, tuple(found[0]["poincare"])
        point = n3.place(f"{key} {tags} {text[:2000]}", depth, parent_point)
        session_phase = int(hashlib.sha1(self.session.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        query = (
            "BEGIN TRANSACTION;"
            "LET $next = ((SELECT VALUE n FROM n3_entry WHERE scope = $scope ORDER BY n DESC LIMIT 1)[0] ?? 0) + 1;"
            "CREATE n3_entry SET scope = $scope, n = $next, session = $p_session, kind = $p_kind, key = $p_key, "
            f"text = $p_text, tags = $p_tags, parent = {int(parent) if parent is not None else 'NONE'}, depth = {depth}, "
            f"poincare = {json.dumps(list(point))}, h3_lorentz = {json.dumps(poincare_to_lorentz(point))}, "
            f"t2_phase = [($next * {GOLDEN}) % 1.0, {session_phase}], pinned = {'true' if pin else 'false'};"
            "RETURN $next;"
            "COMMIT TRANSACTION;")
        # Parameter names are prefixed: $session and a few others are reserved by SurrealDB.
        results = sql(MEM_NS, MEM_DB, query, {"scope": self.scope, "p_session": self.session, "p_kind": kind,
                                              "p_key": key, "p_text": text, "p_tags": tags})
        return int(next(r for r in reversed(results) if isinstance(r, int)))

    def get(self, entry_id: int) -> dict | None:
        rows = self._rows(f"n = {int(entry_id)}", limit=1)
        return self._entry(rows[0]) if rows else None

    def pin(self, entry_id: int, on: bool = True) -> None:
        sql(MEM_NS, MEM_DB, f"UPDATE n3_entry SET pinned = {'true' if on else 'false'} "
            f"WHERE scope = $scope AND n = {int(entry_id)};", {"scope": self.scope})

    def core(self, limit: int | None = None) -> list[dict]:
        return [self._entry(r) for r in reversed(self._rows("pinned = true", limit=limit or 200))]

    def recall(self, query: str, k: int = 5, kinds: tuple[str, ...] | None = None) -> list[dict]:
        terms = _terms(query, 6)
        text_clause, variables = _contains("text", terms)
        key_clause, _ = _contains("key", terms)
        rows = self._rows(f"({text_clause}) OR ({key_clause})", variables) if terms else self._rows("true", limit=40)
        if kinds:
            rows = [r for r in rows if r["kind"] in kinds]
        anchor = n3.place(query, 2)
        def score(row):
            body = f"{row['key']} {row['text']}".lower()
            lexical = sum(term in body for term in terms) / max(len(terms), 1)
            geometric = min(n3.hyperbolic_distance(anchor, row["poincare"]) / 8.0, 1.0)
            return -(0.7 * lexical) + 0.3 * geometric
        return [self._entry(r) for r in sorted(rows, key=score)[:k]]

    @staticmethod
    def _entry(row: dict) -> dict:
        return {"id": row["n"], "kind": row["kind"], "key": row["key"], "text": row["text"],
                "tags": row["tags"], "depth": row["depth"], "pinned": row["pinned"]}


def scope_for(target: pathlib.Path) -> str:
    return f"{target.name}-{hashlib.sha1(str(target.resolve()).encode()).hexdigest()[:10]}"


def open_memory(target: pathlib.Path, session: str = "") -> n3.N3:
    """Short-term memory for a target folder. SQLite is the default.

    FRONTIER_N3 = "sqlite" (default), "surreal" (required), or "auto" (legacy fallback).
    """
    mode = os.environ.get("FRONTIER_N3", "sqlite")
    if mode != "sqlite":
        try:
            return SurrealN3(scope_for(target), session)
        except SurrealError:
            if mode == "surreal":
                raise
    return n3.N3(target / "frontier-loop" / "n3.sqlite", session)


# ---------- canonical H3xT2 addressing (port of hypermind-nested-kv/src/main.rs) ----------

GEOMETRY_MODEL = "hash-idf-rank"
GEOMETRY_VERSION = "H3xT2-Lorentz-canonical-v2"
PROVENANCE = "ple-harness"


def canonical_tokens(text: str) -> list[str]:
    """text::tokenize: lowercase, split on anything that is not alphanumeric."""
    out, current = [], []
    for ch in text.lower():
        if ch.isalnum():
            current.append(ch)
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out


def _fnv1a(token: str) -> int:
    value = 0xCBF29CE484222325
    for byte in token.encode():
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def canonical_address(text: str, salience: float = 0.5) -> tuple[list[float], list[float]]:
    """embed_direction + coord_from_text at rank 0: hashed term counts in 256 dims, folded to a
    3D direction, placed at radius (1 - salience) * 2.5 + 0.1. Returns (h3_lorentz, t2_phase)."""
    import struct
    vec = [0.0] * 256
    for token in canonical_tokens(text):
        h = _fnv1a(token)
        vec[h % 256] += 1.0
        vec[(h >> 16) % 256] += 0.5
    vec = [struct.unpack("f", struct.pack("f", v))[0] for v in vec]   # the Rust path passes through f32
    norm = max(math.sqrt(sum(v * v for v in vec)), 1e-9)
    out = [0.0, 0.0, 0.0]
    for index, value in enumerate(vec):
        out[(index // 32) % 3] += (value / norm) * (1.0 + 0.13 * ((index % 32) / 32.0))
    length = max(math.sqrt(sum(v * v for v in out)), 1e-9)
    radius = (1.0 - min(max(salience, 0.0), 1.0)) * 2.5 + 0.1
    x, y, z = (v / length * radius for v in out)
    return ([math.sqrt(1.0 + x * x + y * y + z * z), x, y, z],
            [math.atan2(y, x), math.atan2(z, math.sqrt(x * x + y * y))])


def phase_chord(a, b) -> float:
    return (1.0 - math.cos(a[0] - b[0])) + (1.0 - math.cos(a[1] - b[1]))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def judge_text(text: str) -> tuple[bool, str]:
    """Structural admission judge (port of judge_text): no model call, reproducible."""
    tokens = canonical_tokens(text)
    if not tokens:
        return False, "empty after tokenize"
    if len(text) < 8:
        return False, "too short (<8 chars)"
    if len(set(tokens)) < 2 and all(len(t) < 4 for t in tokens):
        return False, "below diversity floor"
    return True, "structural gates passed"


# ---------- minting atoms and trajectories into the shared mind ----------

class Mind:
    """Mints atoms and trajectories into the multi-model HyperMind graph, following its lineage
    tables: model_source -> distillation_admission -> atom + atom_address + atom_origin ->
    distilled_trajectory + trajectory_origin, with `next` edges along the walk.

    Atoms are model-agnostic: the content and its address do not depend on which model wrote
    them. The model appears only as audit lineage (model_source), never as a recall key.
    """

    def __init__(self, provider: str, model: str, artifact: str) -> None:
        self.workspace = "workspace:hypermind"
        self.provider, self.model, self.artifact = provider, model, artifact
        self._admission: str | None = None
        self.minted: list[str] = []
        self.minted_trajectories: list[str] = []

    def _core(self, query: str, variables: dict[str, str] | None = None) -> list:
        return sql(CORE_NS, CORE_DB, query, variables)

    def tensor_exact(self, identity: dict, executable: str, *, recalled_addresses: set[str]) -> tuple[bytes, bytes]:
        """Read existing nested K/V using the same Surreal authority as this mind.

        This is a runtime method, never a model text tool. It returns one head's
        exact bytes; the caller must assemble and admit the complete roster.
        """
        from frontier.nested_kv import fetch_exact
        return fetch_exact(identity, executable=executable,
                           env=self._tensor_env(identity, recalled_addresses))

    def tensor_put(self, identity: dict, key: bytes, value: bytes, executable: str, *, recalled_addresses: set[str]) -> None:
        """Publish one immutable native page linked to its existing knowledge atom.

        Runtime-only binary API; complete checkpoint export/discovery is separate.
        """
        from frontier.nested_kv import publish_exact
        publish_exact(identity, key, value, executable=executable,
                      env=self._tensor_env(identity, recalled_addresses))

    def tensor_grounding(self, address: str, geometry_space: str) -> str:
        """Resolve an active selected address and its atom in this workspace."""
        from frontier.checkpoints import record_id
        address, geometry_space = record_id(address), record_id(geometry_space)
        workspace = record_id(self.workspace.removeprefix("workspace:"))
        rows = self._core(f"SELECT atom FROM atom_address:{address} WHERE active = true AND workspace = workspace:{workspace};")[0]
        if len(rows) != 1 or not isinstance(rows[0].get("atom"), str) or not rows[0]["atom"].startswith("atom:"):
            raise ValueError("checkpoint address has no active same-workspace atom")
        atom = record_id(rows[0]["atom"].removeprefix("atom:"))
        rows = self._core(f"SELECT id FROM atom:{atom} WHERE active = true AND workspace = workspace:{workspace};")[0]
        geometry = self._core(f"SELECT id FROM geometry_space:{geometry_space};")[0]
        if len(rows) != 1 or len(geometry) != 1:
            raise ValueError("checkpoint atom or geometry is absent")
        return atom

    def tensor_metadata(self, table: str, ids: tuple[str, ...]) -> dict:
        """Bounded exact record reads; never select tensor bytes or scan history."""
        from frontier.checkpoints import record_id
        from frontier.kv_contract import MAX_COMPONENTS
        fields = {
            "transformer_kv_session": "id, workspace, model_id, model_revision, model_digest, request_digest, prefix_digest, "
                "address_contract_digest, tensor_contract_digest, tppe_configuration_digest, dtype, layer_count, kv_head_count, "
                "head_dim, geometry_space, toroidal_phase_encoder_version, active",
            "transformer_kv_page": "id, workspace, session, layer, kv_head, token_start, token_end, tensor_shape, dtype, layout, "
                "address, phase_state_digest, revision, key_digest, value_digest, receipt_digest, trajectory_atoms, active",
        }
        if table not in fields or type(ids) is not tuple or not 1 <= len(ids) <= MAX_COMPONENTS or len(set(ids)) != len(ids):
            raise ValueError("invalid exact metadata lookup")
        for item in ids:
            record_id(item)
        result = {}
        requested = set(ids)
        for start in range(0, len(ids), 128):
            records = ", ".join(f"{table}:{item}" for item in ids[start:start + 128])
            rows = self._core(f"SELECT {fields[table]} FROM {records};")[0]
            for row in rows:
                key = row.get("id", "").removeprefix(table + ":")
                if key not in requested or key in result:
                    raise ValueError("foreign or duplicate exact metadata record")
                result[key] = row
        return result

    def _tensor_env(self, identity: dict, recalled_addresses: set[str]) -> dict[str, str]:
        workspace = self.workspace.removeprefix("workspace:")
        if identity.get("workspace") != workspace:
            raise ValueError("tensor witness belongs to a different workspace")
        if not identity.get("address") or identity["address"] not in recalled_addresses:
            raise ValueError("tensor address is absent from the selected recall")
        user, password = credentials()
        return {
            "GLOBALMIND_SURREAL_URL": URL, "GLOBALMIND_SURREAL_NS": CORE_NS,
            "GLOBALMIND_SURREAL_DB": CORE_DB, "GLOBALMIND_SURREAL_USER": user,
            "GLOBALMIND_SURREAL_PASS": password,
            "GLOBALMIND_ALLOW_QUARANTINED_CORE": os.environ.get("GLOBALMIND_ALLOW_QUARANTINED_CORE", "0"),
        }

    def tensor_cache(self, runtime_identity, prefix_ids, witnesses, executable: str, *, recalled_addresses):
        """Assemble complete exact pages from this mind before local forward admission."""
        from frontier.kv_assembly import assemble_exact
        return assemble_exact(self, runtime_identity, prefix_ids, witnesses, executable,
                              recalled_addresses=recalled_addresses)

    def continue_tensor_cache(self, runtime, prefix_ids, witnesses, continuation_ids,
                              max_tokens, executable: str, *, recalled_addresses):
        """Continue through an optional local runtime; never sends tensors to text tools."""
        from frontier.kv_assembly import continue_nested
        return continue_nested(runtime, self, prefix_ids, witnesses, continuation_ids, max_tokens,
                               executable, recalled_addresses=recalled_addresses)

    def admission(self) -> str:
        """The lineage record every mint from this run cites. Created once."""
        if self._admission:
            return self._admission
        variables = {"p_provider": self.provider, "p_model": self.model, "p_uri": self.artifact,
                     "p_digest": _sha(self.artifact), "p_model_digest": _sha(f"{self.provider}/{self.model}"),
                     "p_receipt": _sha(json.dumps({"harness": PROVENANCE, "artifact": self.artifact,
                                                   "model": self.model, "judge": "structural+gates"}))}
        source = self._core(
            f"SELECT VALUE id FROM model_source WHERE workspace = {self.workspace} AND provider = $p_provider "
            "AND model = $p_model AND revision = 'api';", variables)[0]
        if not source:
            source = self._core(
                f"CREATE model_source CONTENT {{ workspace: {self.workspace}, provider: $p_provider, model: $p_model, "
                "revision: 'api', modality: 'text', artifact_digest: $p_model_digest } RETURN VALUE id;", variables)[0]
        source_id = str(source[0])
        found = self._core(
            f"SELECT VALUE id FROM distillation_admission WHERE workspace = {self.workspace} "
            f"AND model_source = {source_id} AND source_artifact_digest = $p_digest;", variables)[0]
        if not found:
            found = self._core(
                f"CREATE distillation_admission CONTENT {{ workspace: {self.workspace}, model_source: {source_id}, "
                "source_artifact_digest: $p_digest, source_artifact_uri: $p_uri, eligibility_receipt_digest: $p_receipt, "
                "judge_receipt_digest: $p_receipt, roster_receipt_digest: $p_receipt } RETURN VALUE id;", variables)[0]
        self._admission = str(found[0])
        return self._admission

    def mint_atom(self, text: str, domain: str = "general", salience: float = 0.5, step: str = "") -> str:
        """Mints one atom (or returns the existing atom with the same content). Returns its id."""
        text = text.strip()
        ok, reason = judge_text(text)
        if not ok:
            raise SurrealError(f"atom rejected by the judge: {reason}")
        digest = _sha(text)
        existing = self._core(f"SELECT VALUE id FROM atom WHERE workspace = {self.workspace} "
                              "AND content_digest = $p_digest LIMIT 1;", {"p_digest": digest})[0]
        if existing:
            return str(existing[0])
        admission = self.admission()
        h3, t2 = canonical_address(text, salience)
        domain = re.sub(r"[^A-Za-z0-9_]", "_", domain)[:40] or "general"
        query = (
            "BEGIN TRANSACTION;"
            "LET $next = ((SELECT VALUE math::max(<int> external_id) FROM atom GROUP ALL)[0] ?? 0) + 1;"
            "LET $atom = type::record('atom', $next);"
            "LET $addr = type::record('atom_address', $next);"
            f"CREATE $atom CONTENT {{ workspace: {self.workspace}, external_id: <string> $next, content_kind: 'text', "
            f"content_text: $p_text, content_digest: $p_digest, domain: '{domain}', provenance: '{PROVENANCE}', "
            f"taint: 'none', rank: 0, salience: {float(salience)}, pinned: false }} RETURN NONE;"
            f"CREATE $addr CONTENT {{ workspace: {self.workspace}, atom: $atom, h3_lorentz: {json.dumps(h3)}, "
            f"t2_phase: {json.dumps(t2)}, geometry_model: '{GEOMETRY_MODEL}', geometry_version: '{GEOMETRY_VERSION}', "
            "revision: 1, active: true, space: geometry_space:h3_t2_r12_v1 } RETURN NONE;"
            "UPDATE $atom SET current_address = $addr RETURN NONE;"
            f"CREATE atom_origin CONTENT {{ workspace: {self.workspace}, atom: $atom, admission: {admission}, "
            "mint_receipt_digest: $p_receipt, source_step_digest: $p_step } RETURN NONE;"
            "RETURN $atom;"
            "COMMIT TRANSACTION;")
        results = self._core(query, {"p_text": text, "p_digest": digest, "p_step": _sha(step or text),
                                     "p_receipt": _sha(f"{admission}|{digest}|{GEOMETRY_VERSION}")})
        atom_id = str(next(r for r in reversed(results) if isinstance(r, str) and r.startswith("atom:")))
        _ADDRESSES.clear()
        self.minted.append(atom_id)
        return atom_id

    def relate(self, source: str, target: str, kind: str = "part_of") -> None:
        """Adds a typed edge between two atoms (the structure around a walk). Idempotent."""
        if not all(re.fullmatch(r"atom:\d+", a) for a in (source, target)) or source == target:
            return
        if kind not in ("semantic", "causal", "temporal", "supports", "contradicts", "part_of",
                        "derived_from", "references", "next"):
            raise SurrealError(f"unknown edge kind {kind}")
        exists = self._core(f"SELECT VALUE id FROM atom_edge WHERE in = {source} AND out = {target} "
                            f"AND kind = '{kind}' LIMIT 1;")[0]
        if not exists:
            self._core(f"RELATE {source}->atom_edge->{target} CONTENT {{ workspace: {self.workspace}, kind: '{kind}', "
                       f"weight: 1.0, confidence: 1.0, producer: '{PROVENANCE}', evidence_digest: $p_digest, revision: 1 }} "
                       "RETURN NONE;", {"p_digest": _sha(f"{source}|{kind}|{target}")})

    def mint_trajectory(self, atoms: list[str], judge_receipt: str) -> str | None:
        """Mints an ordered walk over atoms, with `next` edges between consecutive atoms."""
        if len(atoms) > 4096 or any(not isinstance(a, str) or not re.fullmatch(r"atom:\d+", a) for a in atoms):
            raise SurrealError("trajectory requires at most 4096 valid atom IDs")
        if len(atoms) < 2:
            return None
        admission = self.admission()
        digest = _sha("|".join(atoms))
        found = self._core(f"SELECT VALUE id FROM distilled_trajectory WHERE workspace = {self.workspace} "
                           "AND trajectory_digest = $p_digest;", {"p_digest": digest})[0]
        if found:
            return str(found[0])
        # A later walk often shares steps with an earlier one. An existing `next` edge between the
        # same two atoms already records that transition, and a duplicate would break the unique
        # edge index and fail the whole transaction, so only new transitions get an edge.
        pairs = {(a, b) for a, b in zip(atoms, atoms[1:]) if a != b}
        existing = self._core(
            f"SELECT in, out FROM atom_edge WHERE workspace = {self.workspace} AND kind = 'next' "
            f"AND in IN [{', '.join(sorted({a for a, _ in pairs}))}];")[0] if pairs else []
        known = {(str(row["in"]), str(row["out"])) for row in existing}
        edges = "".join(
            f"RELATE {a}->atom_edge->{b} CONTENT {{ workspace: {self.workspace}, kind: 'next', weight: 1.0, confidence: 1.0, "
            f"producer: '{PROVENANCE}', evidence_digest: $p_digest, revision: 1 }} RETURN NONE;"
            for a, b in zip(atoms, atoms[1:]) if a != b and (a, b) not in known)
        query = (
            "BEGIN TRANSACTION;"
            f"LET $t = (CREATE distilled_trajectory CONTENT {{ workspace: {self.workspace}, atoms: [{', '.join(atoms)}], "
            "judge_receipt_digest: $p_judge, trajectory_digest: $p_digest } RETURN VALUE id)[0];"
            f"CREATE trajectory_origin CONTENT {{ workspace: {self.workspace}, trajectory: $t, admission: {admission}, "
            "receipt_digest: $p_judge, source_walk_digest: $p_digest } RETURN NONE;"
            + edges + "RETURN $t; COMMIT TRANSACTION;")
        results = self._core(query, {"p_digest": digest, "p_judge": _sha(judge_receipt)})
        trajectory_id = str(next(r for r in reversed(results) if isinstance(r, str) and r.startswith("distilled_trajectory:")))
        self.minted_trajectories.append(trajectory_id)
        return trajectory_id


def trajectories_through(atom_id: str, limit: int = 3) -> list[dict]:
    if not re.fullmatch(r"atom:\d+", atom_id):
        raise SurrealError("invalid atom ID")
    limit = max(1, min(int(limit), 10))
    rows = sql(CORE_NS, CORE_DB,
               f"SELECT id, atoms FROM distilled_trajectory WHERE atoms CONTAINS {atom_id} LIMIT {limit};")[0]
    return [{"id": str(row["id"]), "atoms": [str(a) for a in row["atoms"]]} for row in rows]


def trajectory_get(trajectory_id: str, start: int = 0, limit: int = 8) -> dict | None:
    """Open a bounded page of one stored walk, verifying every returned text atom."""
    if not isinstance(trajectory_id, str) or not re.fullmatch(r"distilled_trajectory:[A-Za-z0-9_-]+", trajectory_id):
        raise SurrealError("invalid trajectory ID")
    if not isinstance(start, int) or start < 0 or not isinstance(limit, int) or not 1 <= limit <= 16:
        raise SurrealError("trajectory page requires start >= 0 and limit 1..16")
    rows = sql(CORE_NS, CORE_DB,
               f"SELECT id, workspace, atoms, trajectory_digest FROM {trajectory_id};")[0]
    if not rows:
        return None
    row = rows[0]
    if str(row.get("id")) != trajectory_id or str(row.get("workspace")) != "workspace:hypermind":
        raise SurrealError("trajectory identity or workspace mismatch")
    atoms = row.get("atoms")
    if not isinstance(atoms, list) or len(atoms) > 4096:
        raise SurrealError("trajectory atom roster is invalid or oversized")
    atom_ids = [str(a) for a in atoms]
    if any(not re.fullmatch(r"atom:\d+", a) for a in atom_ids):
        raise SurrealError("trajectory contains an invalid atom ID")
    if row.get("trajectory_digest") != _sha("|".join(atom_ids)):
        raise SurrealError("trajectory roster digest mismatch")
    steps = []
    for index in range(start, min(start + limit, len(atom_ids))):
        atom = atom_get(atom_ids[index])
        if atom is None or str(atom.get("workspace")) != "workspace:hypermind":
            raise SurrealError(f"trajectory atom {atom_ids[index]} is missing or foreign")
        body = atom.get("content_text")
        if body is not None and (not isinstance(body, str) or atom.get("content_digest") != _sha(body)):
            raise SurrealError(f"trajectory atom {atom_ids[index]} has a content digest mismatch")
        steps.append({"index": index, "id": atom_ids[index], "domain": atom.get("domain"),
                      "text": body[:1600] if body else None, "content_uri": atom.get("content_uri")})
    next_start = start + len(steps)
    return {"id": trajectory_id, "total": len(atom_ids), "start": start,
            "next_start": next_start if next_start < len(atom_ids) else None, "steps": steps}


def mind_recall(query: str, k: int = 4) -> str:
    """What the mind holds on a topic: best text matches, their hyperbolic neighbours, and any
    trajectory that passes through the best match (the steps a model took before)."""
    hits = atom_search(query, k)
    if not hits:
        return ""
    lines = [f"{a['id']} [{a['domain']}] {a['text'][:700]}" for a in hits]
    try:
        near = [n for n in atom_neighbors(hits[0]["id"], 3) if n["id"] not in {a["id"] for a in hits}]
        lines += [f"{n['id']} (near {hits[0]['id']}) {n['text'][:400]}" for n in near[:2]]
        for walk in trajectories_through(hits[0]["id"], 1):
            lines.append("trajectory " + walk["id"] + " through " + hits[0]["id"] + ": "
                         + " -> ".join(walk["atoms"]) + " (read it with trajectory_get)")
    except SurrealError:
        pass
    return "\n\n".join(lines)


def mind_enabled() -> bool:
    """FRONTIER_MIND = "on", "off", or "auto" (default: on when SurrealDB answers)."""
    mode = os.environ.get("FRONTIER_MIND", "auto")
    return mode == "on" or (mode == "auto" and available())


# ---------- tools for models ----------

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "atom_search",
        "description": "Search the HyperMind knowledge atoms (distilled notes, protocols, designs) by text.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "atom_neighbors",
        "description": "Atoms nearest to a given atom in the hyperbolic map (related knowledge). Give an id like atom:12.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}, "k": {"type": "integer"}},
                       "required": ["id"]}}},
    {"type": "function", "function": {
        "name": "atom_get",
        "description": "Read one atom in full by id.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}},
    {"type": "function", "function": {
        "name": "trajectory_get",
        "description": "Read an ordered page of a shared-mind trajectory by its ID, with verified atom text.",
        "parameters": {"type": "object", "properties": {
            "id": {"type": "string"}, "start": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 16}}, "required": ["id"]}}},
]


async def _atom_search(args):
    import asyncio
    try:
        found = await asyncio.to_thread(atom_search, args["query"], min(int(args.get("k") or 5), 10))
    except SurrealError as exc:
        return f"error: {exc}"
    return "\n\n".join(f"{a['id']} [{a['domain']}]\n{a['text']}" for a in found) or "no atoms matched"


async def _atom_neighbors(args):
    import asyncio
    try:
        found = await asyncio.to_thread(atom_neighbors, args["id"], min(int(args.get("k") or 5), 10))
    except SurrealError as exc:
        return f"error: {exc}"
    return "\n\n".join(f"{a['id']} (distance {a['distance']}) [{a['domain']}]\n{a['text']}" for a in found)


async def _atom_get(args):
    import asyncio
    try:
        atom = await asyncio.to_thread(atom_get, args["id"])
    except SurrealError as exc:
        return f"error: {exc}"
    return json.dumps(atom, default=str)[:12000] if atom else "no such atom"


async def _trajectory_get(args):
    import asyncio
    try:
        page = await asyncio.to_thread(trajectory_get, args["id"], int(args.get("start", 0)),
                                       int(args.get("limit", 8)))
    except (SurrealError, ValueError, TypeError) as exc:
        return f"error: {exc}"
    return json.dumps(page, default=str) if page else "no such trajectory"


MIND_TOOLS: list[dict[str, Any]] = TOOLS + [
    {"type": "function", "function": {
        "name": "mind_mint",
        "description": ("Mint a new atom into the shared hyperbolic mind: one self-contained, verified piece of "
                        "knowledge (a finding, a method that worked, a dead end and why). Any model can read it later."),
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}, "domain": {"type": "string", "description": "for example S_research, S_code, general"},
            "salience": {"type": "number", "description": "0 to 1; higher sits nearer the centre of the map"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "mind_mint_trajectory",
        "description": "Record an ordered walk over 2 to 32 existing shared-mind atoms, with an evidence note.",
        "parameters": {"type": "object", "properties": {
            "atoms": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 32},
            "evidence": {"type": "string", "description": "What each step establishes and how it was checked"}},
            "required": ["atoms", "evidence"]}}},
]


def make_handlers(mind: Mind | None) -> dict[str, Any]:
    import asyncio

    async def mind_mint(args):
        if mind is None:
            return "error: the mind is not connected"
        try:
            atom = await asyncio.to_thread(mind.mint_atom, args["text"], args.get("domain", "general"),
                                           float(args.get("salience", 0.5)), "model-initiated")
        except SurrealError as exc:
            return f"error: {exc}"
        return f"minted {atom}"

    async def mind_mint_trajectory(args):
        if mind is None:
            return "error: the mind is not connected"
        atoms = args.get("atoms")
        evidence = args.get("evidence")
        if (not isinstance(atoms, list) or not 2 <= len(atoms) <= 32
                or any(not isinstance(a, str) or not re.fullmatch(r"atom:\d+", a) for a in atoms)
                or not isinstance(evidence, str) or len(evidence.strip()) < 20):
            return "error: provide 2..32 atom IDs and a substantive evidence note"
        try:
            for atom_id in set(atoms):
                atom = await asyncio.to_thread(atom_get, atom_id)
                if atom is None or str(atom.get("workspace")) != "workspace:hypermind":
                    return f"error: atom {atom_id} is missing or foreign"
                body = atom.get("content_text")
                if body is not None and (not isinstance(body, str) or atom.get("content_digest") != _sha(body)):
                    return f"error: atom {atom_id} has a content digest mismatch"
            trajectory = await asyncio.to_thread(mind.mint_trajectory, atoms, evidence.strip())
        except SurrealError as exc:
            return f"error: {exc}"
        return f"minted {trajectory}" if trajectory else "error: trajectory was not minted"

    return {"atom_search": _atom_search, "atom_neighbors": _atom_neighbors,
            "atom_get": _atom_get, "trajectory_get": _trajectory_get,
            "mind_mint": mind_mint, "mind_mint_trajectory": mind_mint_trajectory}


HANDLERS = {"atom_search": _atom_search, "atom_neighbors": _atom_neighbors,
            "atom_get": _atom_get, "trajectory_get": _trajectory_get}
