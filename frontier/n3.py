"""N3 memory, native to the harness: a short-term store the model deposits to and recalls
from, addressed in a 3D hyperbolic ball, plus Qdrant RAG for research sources.

Short-term store
    One SQLite file. Every entry has a key, text, tags and an address in the Poincare
    3-ball: the direction encodes the topic and the radius encodes depth (general notes sit
    near the centre, details near the boundary, children are placed outward from their
    parent). Recall blends lexical match (FTS5) with hyperbolic distance to the query, so
    related notes at nearby depth rank together.

Context without compaction
    `N3.manage(messages, budget)` archives completed old turns in SQLite and keeps a
    pointer plus recent complete tool-call groups. Nothing is discarded; the model reads
    archived text in pages with memory_recall. A bounded view of pinned entries stays in
    context, while the complete pinned set remains in the store.

Qdrant RAG
    The local Qdrant Edge shard is the canonical research library. `rag_search` uses
    its ranked BM25 text index, so every chat model uses the same retrieval contract.
    Verified sources are ingested as payload-only points into that shard. Existing
    Qwen vectors remain untouched for a separately admitted dense-query client.

Later: at session end the pinned core can be pushed to Qdrant for long-term storage
(`N3.core()` returns exactly that set).
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
import pathlib
import re
import sqlite3
import time
import uuid
from typing import Any

from qdrant_edge import (EdgeShard, FieldCondition, Filter, MatchValue, PayloadSchemaType,
                         Point, QueryRequest, ScrollRequest, TextQuery, UpdateOperation)

EDGE_SHARD = pathlib.Path(os.environ.get(
    "FRONTIER_EDGE_SHARD",
    "/Users/stewartbarteau/Desktop/Ai Research and Preprint Vault/qdrant-edge-research-v1",
))
LIBRARY = os.environ.get("FRONTIER_RAG_LIBRARY", "research_library_v1")
RAG_TEXT_FIELD = os.environ.get("FRONTIER_RAG_TEXT_FIELD", "text")
RAG_CHUNK_CHARS = 3200
RAG_CHUNK_OVERLAP = 200
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-\.]+")

# BM25 recall/cost tradeoff; see HYPERBOLIC-BOUNDS.md for complexity characterisation.
N3_FTS5_CANDIDATE_LIMIT = 40
# Fallback row scan limit when FTS5 is unavailable; see HYPERBOLIC-BOUNDS.md.
N3_FTS5_FALLBACK_LIMIT = 256
STOP = set("the a an and or of to in for on with is are was were be by as at from that this it its "
           "we our their can may using use used via than then which how what when not".split())


# ---------- 3D hyperbolic addressing ----------

def _tokens(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text) if w.lower() not in STOP]


def _hash(token: str, salt: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{salt}:{token}".encode(), digest_size=8).digest(), "big")


def direction(text: str) -> tuple[float, float, float]:
    """Topic direction on the unit sphere: a signed random projection of the token bag to 3D."""
    vec = [0.0, 0.0, 0.0]
    for token in _tokens(text):
        for axis in range(3):
            h = _hash(token, f"axis{axis}")
            vec[axis] += ((h % 2001) / 1000.0 - 1.0)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    if norm == 1.0 and not any(vec):
        return (1.0, 0.0, 0.0)
    return tuple(v / norm for v in vec)  # type: ignore[return-value]


def place(text: str, depth: int, parent: tuple[float, float, float] | None = None) -> tuple[float, float, float]:
    """Address in the open Poincare ball; deeper descendants move outward."""
    if not isinstance(depth, int) or depth < 1:
        raise ValueError("memory depth must be a positive integer")
    own = direction(text)
    if parent is not None:
        if len(parent) != 3 or not all(math.isfinite(v) for v in parent) or sum(v * v for v in parent) >= 1:
            raise ValueError("parent address must be inside the Poincare ball")
        pnorm = math.sqrt(sum(v * v for v in parent)) or 1.0
        mixed = [0.65 * (p / pnorm) + 0.35 * o for p, o in zip(parent, own)]
        norm = math.sqrt(sum(v * v for v in mixed)) or 1.0
        own = tuple(v / norm for v in mixed)  # type: ignore[assignment]
    # Keep legacy shallow addresses stable. A slower radial law after depth 7
    # preserves strict ordering without clipping every deep child to 0.999.
    distance_from_origin = depth if depth <= 7 else 7.0 + math.log1p(depth - 7)
    radius = math.tanh(distance_from_origin / 2.0)
    if radius >= 1.0:
        raise ValueError("memory depth exceeds representable Poincare radius")
    return tuple(radius * v for v in own)  # type: ignore[return-value]


def hyperbolic_distance(u, v) -> float:
    if (len(u) != 3 or len(v) != 3 or not all(math.isfinite(x) for x in (*u, *v))):
        raise ValueError("hyperbolic distance requires finite 3D points")
    du = 1.0 - sum(a * a for a in u)
    dv = 1.0 - sum(a * a for a in v)
    if du <= 0 or dv <= 0:
        raise ValueError("hyperbolic distance requires points inside the open ball")
    diff = sum((a - b) ** 2 for a, b in zip(u, v))
    return 2.0 * math.asinh(math.sqrt(diff / (du * dv)))


# ---------- short-term store ----------

class N3:
    def __init__(self, path: str | os.PathLike, session: str = "") -> None:
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session = session or time.strftime("%Y%m%d-%H%M%S")
        self.db = sqlite3.connect(self.path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS mem (
            id INTEGER PRIMARY KEY, ts TEXT, session TEXT, kind TEXT, key TEXT, text TEXT, tags TEXT,
            parent INTEGER, depth INTEGER, x REAL, y REAL, z REAL, pinned INTEGER DEFAULT 0)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS mem_pinned_recent ON mem(pinned, id DESC)")
        try:
            self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(key, text, tags)")
            self.fts = True
        except sqlite3.OperationalError:
            self.fts = False
        self.db.commit()

    def deposit(self, text: str, key: str = "", tags: str = "", kind: str = "note",
                parent: int | None = None, pin: bool = False) -> int:
        depth, parent_point = 1, None
        if parent is not None:
            row = self.db.execute("SELECT depth, x, y, z FROM mem WHERE id=?", (parent,)).fetchone()
            if not row:
                raise ValueError(f"memory parent m{parent} does not exist")
            depth, parent_point = row[0] + 1, (row[1], row[2], row[3])
        x, y, z = place(f"{key} {tags} {text[:2000]}", depth, parent_point)
        cursor = self.db.execute(
            "INSERT INTO mem (ts, session, kind, key, text, tags, parent, depth, x, y, z, pinned) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.strftime("%Y-%m-%d %H:%M:%S"), self.session, kind, key, text, tags, parent, depth, x, y, z, int(pin)))
        if self.fts:
            self.db.execute("INSERT INTO mem_fts (rowid, key, text, tags) VALUES (?,?,?,?)",
                            (cursor.lastrowid, key, text, tags))
        self.db.commit()
        return cursor.lastrowid

    def get(self, entry_id: int) -> dict | None:
        row = self.db.execute("SELECT id, kind, key, text, tags, depth, pinned FROM mem WHERE id=?",
                              (entry_id,)).fetchone()
        return dict(zip(("id", "kind", "key", "text", "tags", "depth", "pinned"), row)) if row else None

    def pin(self, entry_id: int, on: bool = True) -> None:
        self.db.execute("UPDATE mem SET pinned=? WHERE id=?", (int(on), entry_id))
        self.db.commit()

    def core(self, limit: int | None = None) -> list[dict]:
        if limit is None:
            rows = self.db.execute("SELECT id, kind, key, text FROM mem WHERE pinned=1 ORDER BY id").fetchall()
        else:
            rows = self.db.execute("SELECT id, kind, key, text FROM mem WHERE pinned=1 "
                                   "ORDER BY id DESC LIMIT ?", (max(1, limit),)).fetchall()
            rows.reverse()
        return [dict(zip(("id", "kind", "key", "text"), row)) for row in rows]

    def recall(self, query: str, k: int = 5, kinds: tuple[str, ...] | None = None) -> list[dict]:
        tokens = _tokens(query)[:12]
        rows = []
        if self.fts and tokens:
            match = " OR ".join(f'"{t}"' for t in tokens)
            try:
                rows = self.db.execute(
                    "SELECT m.id, m.kind, m.key, m.text, m.tags, m.x, m.y, m.z, bm25(mem_fts) FROM mem_fts "
                    "JOIN mem m ON m.id = mem_fts.rowid WHERE mem_fts MATCH ? ORDER BY bm25(mem_fts) LIMIT ?",
                    (match, N3_FTS5_CANDIDATE_LIMIT)).fetchall()
            except sqlite3.OperationalError:
                rows = []
        if not rows:
            # A LIKE '%term%' predicate would scan the whole table. The
            # fallback is bounded by recent row IDs when FTS5 is absent.
            recent = self.db.execute("SELECT id, kind, key, text, tags, x, y, z FROM mem "
                                     "ORDER BY id DESC LIMIT ?", (N3_FTS5_FALLBACK_LIMIT,)).fetchall()
            rows = [r + (0.0,) for r in recent if not tokens or any(
                token in f"{r[2]} {r[3]} {r[4]}".lower() for token in tokens[:4])][:N3_FTS5_CANDIDATE_LIMIT]
        if kinds:
            rows = [r for r in rows if r[1] in kinds]
        if not rows:
            return []
        anchor = place(query, 2)
        best = min(r[8] for r in rows)
        worst = max(r[8] for r in rows)
        span = (worst - best) or 1.0
        scored = []
        for r in rows:
            lexical = (r[8] - best) / span                       # 0 is the best lexical match
            geometric = hyperbolic_distance(anchor, (r[5], r[6], r[7])) / 8.0
            scored.append((0.7 * lexical + 0.3 * min(geometric, 1.0), r))
        scored.sort(key=lambda item: item[0])
        return [dict(zip(("id", "kind", "key", "text", "tags"), r[:5])) for _, r in scored[:k]]

    # ----- context management that replaces compaction -----

    def manage(self, messages: list[dict[str, Any]], budget: int, keep_recent: int = 10) -> None:
        """Archive completed old turns without splitting native tool-call groups."""
        self._refresh_core(messages)
        if _estimate(messages) <= budget:
            return
        if len(messages) <= 3:
            return  # the fixed prompt alone exceeds the target
        cutoff = max(3, len(messages) - keep_recent)
        # A tool result cannot survive without the assistant call that named
        # its ID. Move the boundary to the start of that completed group.
        while cutoff > 3 and messages[cutoff].get("role") == "tool":
            cutoff -= 1
        # Keep the latest complete tool-call group, but archive more old groups
        # when the normal recent window still exceeds the target budget.
        latest_call = max((i for i, m in enumerate(messages)
                           if m.get("role") == "assistant" and m.get("tool_calls")), default=3)
        if latest_call > 3:
            cutoff = min(cutoff, latest_call)
        sizes = [len(json.dumps(m, ensure_ascii=False)) for m in messages]
        suffix = [0] * (len(messages) + 1)
        for i in range(len(messages) - 1, -1, -1):
            suffix[i] = suffix[i + 1] + sizes[i]
        fixed = sum(sizes[:3]) + 400  # reserve room for the archive pointer
        while cutoff < latest_call and (fixed + suffix[cutoff]) // 4 > budget:
            cutoff += 1
            while cutoff < latest_call and messages[cutoff].get("role") == "tool":
                cutoff += 1
        if cutoff <= 3:
            return  # fixed prompt or recent group exceeds budget; never break protocol
        archived = messages[3:cutoff]
        if not archived:
            return
        entry = self.deposit(json.dumps(archived, ensure_ascii=False),
                             key=f"conversation turns ({len(archived)} messages)",
                             kind="conversation_archive")
        messages[3:cutoff] = [{"role": "user", "content":
                              f"[memory m{entry}: {len(archived)} earlier messages archived. "
                              f"Read with memory_recall(id={entry}); use start to page.]"}]

    def _refresh_core(self, messages: list[dict[str, Any]]) -> None:
        recent = self.core(limit=20)
        lines = [f"m{e['id']} {e['key'][:80]}: {e['text'][:120]}" for e in recent]
        text = "[core memory]\n" + ("\n".join(lines) or
                "(empty: pin what must not be forgotten with memory_deposit(pin=true))")
        if len(recent) == 20:
            text += "\n(older pinned entries, if any, remain searchable with memory_recall)"
        if len(messages) > 2 and str(messages[2].get("content", "")).startswith("[core memory]"):
            messages[2]["content"] = text
        elif len(messages) >= 2:
            messages.insert(2, {"role": "user", "content": text})


def _estimate(messages: list[dict[str, Any]]) -> int:
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) // 4


# ---------- Qdrant RAG ----------


def research_chunks(text: str) -> list[str]:
    """Split all source text for Qdrant; never truncate a verified source."""
    step = RAG_CHUNK_CHARS - RAG_CHUNK_OVERLAP
    return [text[i:i + RAG_CHUNK_CHARS] for i in range(0, len(text), step)] or [text]


def research_point_id(source: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{index}"))


class Rag:
    def __init__(self, shard_path: str | pathlib.Path | None = None) -> None:
        self.collection = os.environ.get("FRONTIER_RAG_LIBRARY", LIBRARY)
        self.text_field = os.environ.get("FRONTIER_RAG_TEXT_FIELD", RAG_TEXT_FIELD)
        self.shard_path = pathlib.Path(shard_path or EDGE_SHARD)

    @contextmanager
    def _open(self):
        """One process owns an Edge shard at a time; serialize harness callers."""
        if not self.shard_path.is_dir():
            raise FileNotFoundError(f"Qdrant Edge shard is missing: {self.shard_path}")
        with (self.shard_path / ".ple-edge.lock").open("a+b") as lock:
            deadline = time.monotonic() + 30
            while True:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Qdrant Edge shard is busy for more than 30 seconds")
                    time.sleep(0.05)
            shard = EdgeShard.load(str(self.shard_path))
            try:
                field = shard.info().payload_schema.get(self.text_field)
                if (field is None or field.data_type != PayloadSchemaType.Text or
                        getattr(field.params, "scoring", None) is None):
                    raise RuntimeError(f"Qdrant Edge field {self.text_field!r} lacks a scored text index")
                yield shard
            finally:
                shard.close()
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def ingest(self, url: str, title: str, text: str, year: str = "", note: str = "") -> int:
        if not text.strip():
            raise ValueError("cannot ingest blank source text")
        chunks = research_chunks(text)
        points = []
        for index, chunk in enumerate(chunks):
            digest = hashlib.sha256(chunk.encode()).hexdigest()
            points.append(Point(id=research_point_id(url, index), vector={},
                                payload={"url": url, "source_path": url, "title": title,
                                         "document_id": title, "source_type": "verified-web",
                                         "content_hash": digest, "language": "und",
                                         "research_domain": "frontier-verified", "citation_ids": [],
                                         "tenant_id": "local", "silo": "frontier", "year": year,
                                         self.text_field: chunk, "chunk_index": index, "note": note,
                                         "verified": True, "ingested_by": "frontier-loop",
                                         "retrieval_mode": "qdrant-edge-bm25-text-v1",
                                         "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S")}))
        with self._open() as shard:
            ids = [point.id for point in points]
            for existing in shard.retrieve(ids, False, True):
                if existing.vector:
                    raise ValueError(f"refusing to replace dense vectors on point {existing.id}")
            condition = Filter(must=[FieldCondition("source_path", match=MatchValue(url))])
            offset = None
            stale_ids = []
            while True:
                records, offset = shard.scroll(ScrollRequest(
                    limit=256, offset=offset, filter=condition,
                    with_payload=["chunk_index", "source_path"], with_vector=False))
                for record in records:
                    payload = record.payload or {}
                    index = payload.get("chunk_index")
                    if (isinstance(index, int) and index >= len(chunks) and
                            str(record.id) == research_point_id(url, index)):
                        stale_ids.append(str(record.id))
                if offset is None:
                    break
            for start in range(0, len(points), 32):
                shard.update(UpdateOperation.upsert_points(points[start:start + 32]))
            if stale_ids:
                shard.update(UpdateOperation.delete_points(stale_ids))
            shard.flush()
        return len(points)

    def search(self, query: str, k: int = 5) -> list[dict]:
        query = query.strip()
        if not query:
            return []
        if len(query.encode()) > 4096:
            return [{"source": "", "title": "research library unavailable",
                     "text": "query exceeds 4096 UTF-8 bytes", "from": "error"}]
        try:
            with self._open() as shard:
                points = shard.query(QueryRequest(query=TextQuery(self.text_field, query),
                                                  limit=max(1, min(k, 50)), with_payload=True))
            hits = []
            for point in points:
                payload = point.payload or {}
                hits.append({"source": payload.get("url") or payload.get("source_path", ""),
                             "title": payload.get("title") or payload.get("document_id", ""),
                             "text": payload.get(self.text_field, ""), "score": point.score,
                             "from": self.collection})
            return hits
        except Exception as exc:
            return [{"source": "", "title": "research library unavailable",
                     "text": f"{type(exc).__name__}: {exc}", "from": "error"}]

    def open_document(self, source: str, start_chunk: int = 0, limit_chunks: int = 8) -> list[dict]:
        """Read indexed passages by exact source path without touching source files."""
        if not source or len(source.encode()) > 4096:
            raise ValueError("source path must be 1 to 4096 UTF-8 bytes")
        start_chunk = max(0, start_chunk)
        limit_chunks = max(1, min(limit_chunks, 12))
        condition = Filter(must=[FieldCondition("source_path", match=MatchValue(source))])
        passages: list[dict] = []
        with self._open() as shard:
            offset = None
            while True:
                records, offset = shard.scroll(ScrollRequest(
                    limit=256, offset=offset, filter=condition,
                    with_payload=["source_path", "document_id", "title", "chunk_index", self.text_field],
                    with_vector=False))
                for record in records:
                    payload = record.payload or {}
                    if payload.get("source_path") == source and isinstance(payload.get(self.text_field), str):
                        passages.append({"index": payload.get("chunk_index", 0),
                                         "title": payload.get("title") or payload.get("document_id", ""),
                                         "text": payload[self.text_field]})
                if offset is None:
                    break
        passages.sort(key=lambda item: item["index"])
        return passages[start_chunk:start_chunk + limit_chunks]


# ---------- tools ----------

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "memory_deposit",
        "description": ("Save a note to short-term memory (decisions, facts, results, dead ends). Set pin=true "
                        "for things that must stay in view all run. parent links a detail under an earlier note."),
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}, "key": {"type": "string", "description": "short title"},
            "tags": {"type": "string"}, "pin": {"type": "boolean"}, "parent": {"type": "integer"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "memory_recall",
        "description": "Recall by query, or read an entry by ID in bounded character pages.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "id": {"type": "integer"}, "k": {"type": "integer"},
            "start": {"type": "integer", "minimum": 0},
            "chars": {"type": "integer", "minimum": 1, "maximum": 8000}}}}},
    {"type": "function", "function": {
        "name": "rag_search",
        "description": ("Search the local Qdrant Edge research library. Returned source paths are archive "
                        "keys; use rag_open to read more passages because filesystem originals may be absent."),
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "rag_ingest",
        "description": ("Ingest a source you opened and verified into the research library. Give the URL, title, "
                        "year and the passages you relied on (omit text to ingest the page's readable text)."),
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}, "title": {"type": "string"}, "year": {"type": "string"},
            "text": {"type": "string"}, "note": {"type": "string", "description": "what you verified"}},
            "required": ["url", "title"]}}},
    {"type": "function", "function": {
        "name": "rag_open",
        "description": "Read ordered passages of one research source returned by rag_search, entirely from Qdrant Edge.",
        "parameters": {"type": "object", "properties": {
            "source": {"type": "string", "description": "Exact source path or URL from rag_search"},
            "start_chunk": {"type": "integer"}, "limit_chunks": {"type": "integer"}},
            "required": ["source"]}}},
]
MEMORY_TOOLS = TOOLS[:2]
RAG_TOOLS = TOOLS[2:]


def make_handlers(memory: N3, rag: Rag | None, ingested: list[dict] | None = None, fetch=None) -> dict[str, Any]:
    """Tool handlers bound to one memory store. `ingested` collects what rag_ingest stored."""

    async def memory_deposit(args):
        entry = memory.deposit(args["text"], key=args.get("key", ""), tags=args.get("tags", ""),
                               parent=args.get("parent"), pin=bool(args.get("pin")))
        return f"saved as m{entry}" + (" (pinned)" if args.get("pin") else "")

    async def memory_recall(args):
        if args.get("id") is not None:
            entry = memory.get(int(args["id"]))
            if not entry:
                return "no such memory id"
            start = max(0, int(args.get("start") or 0))
            chars = max(1, min(int(args.get("chars") or 4000), 8000))
            body = entry["text"]
            next_start = start + chars if start + chars < len(body) else None
            return (f"m{entry['id']} [{entry['kind']}] {entry['key']} "
                    f"(chars {start}:{min(start + chars, len(body))} of {len(body)}; next_start={next_start})\n"
                    + body[start:start + chars])
        found = memory.recall(args.get("query", ""), k=min(int(args.get("k") or 5), 10))
        if not found:
            return "nothing relevant in memory"
        return "\n\n".join(f"m{e['id']} [{e['kind']}] {e['key']}\n{e['text'][:1500]}" for e in found)

    async def rag_search(args):
        if rag is None:
            return "error: the research library is not configured"
        hits = await asyncio.to_thread(rag.search, args["query"], min(int(args.get("k") or 5), 8))
        body = "\n\n".join(f"[{h['from']}] {h['title']}\nsource key: {h['source']}\n{h['text'][:900]}"
                           for h in hits)
        return ("Use rag_open with a source key to read more passages.\n\n" + body) if body else "no results"

    async def rag_ingest(args):
        if rag is None:
            return "error: the research library is not configured"
        text = args.get("text") or ""
        if not text and fetch is not None:
            text = await fetch({"url": args["url"]})
            if text.startswith("error") or "%PDF" in text[:20]:
                return "error: could not read that URL as text; pass the passages in `text` (for arXiv use the /abs or /html page)"
        if len(text.strip()) < 200:
            return "error: give at least a few sentences of the source text you verified"
        try:
            count = await asyncio.to_thread(rag.ingest, args["url"], args["title"], text,
                                            str(args.get("year", "")), args.get("note", ""))
        except Exception as exc:
            return f"error: ingest failed: {type(exc).__name__}: {exc}"
        if ingested is not None:
            ingested.append({"url": args["url"], "title": args["title"], "chunks": count})
        memory.deposit(f"{args['title']} ({args.get('year', '')}) {args['url']}\n{args.get('note', '')}",
                       key=args["title"][:80], tags="source verified", kind="source")
        return f"ingested {count} chunk(s) into {rag.collection}"

    async def rag_open(args):
        if rag is None:
            return "error: the research library is not configured"
        try:
            chunks = await asyncio.to_thread(rag.open_document, args["source"],
                                             int(args.get("start_chunk") or 0),
                                             int(args.get("limit_chunks") or 8))
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"
        return "\n\n".join(f"chunk {item['index']} — {item['title']}\n{item['text']}"
                           for item in chunks) or "no indexed passages for that source"

    return {"memory_deposit": memory_deposit, "memory_recall": memory_recall,
            "rag_search": rag_search, "rag_ingest": rag_ingest, "rag_open": rag_open}
