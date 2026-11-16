"""Portable, model-neutral artifact renderer for frontier agents and Zed."""

from __future__ import annotations

import hashlib
import html
import io
import json
import math
import os
import re
import tempfile
import textwrap
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

TOOL = {"type": "function", "function": {
    "name": "artifact_render",
    "description": "Render a structured report or graph to portable workspace files with SHA-256 receipts.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "Relative output stem inside the active workspace."},
        "title": {"type": "string"},
        "kind": {"type": "string", "enum": ["document", "mindmap", "dag", "relation", "graph3d", "graph4d"]},
        "formats": {"type": "array", "items": {"type": "string", "enum": ["txt", "md", "pdf", "docx", "svg", "html"]}},
        "sections": {"type": "array", "items": {"type": "object", "properties": {
            "heading": {"type": "string"}, "body": {"type": "string"}}}},
        "nodes": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "label": {"type": "string"},
            "x": {"type": "number"}, "y": {"type": "number"},
            "z": {"type": "number"}, "w": {"type": "number"}}}},
        "edges": {"type": "array", "items": {"type": "object", "properties": {
            "source": {"type": "string"}, "target": {"type": "string"}, "label": {"type": "string"}}}},
        "note": {"type": "string", "description": "Evidence, coordinate meanings, and limitations."},
    }, "required": ["path", "title", "kind"]}}}

ALLOWED = {"document": {"txt", "md", "pdf", "docx"},
           "mindmap": {"svg", "txt", "md"}, "dag": {"svg", "txt", "md"},
           "relation": {"svg", "txt", "md"},
           "graph3d": {"html", "txt", "md"}, "graph4d": {"html", "txt", "md"}}
MAX_NODES, MAX_EDGES, MAX_SECTIONS = 80, 160, 40


def _validate(args: dict) -> tuple[str, str, list[dict], list[dict], list[dict], list[str], str]:
    kind = str(args.get("kind", ""))
    if kind not in ALLOWED:
        raise ValueError("unsupported artifact kind")
    title = str(args.get("title", "")).strip()
    if not title or len(title) > 160:
        raise ValueError("title must have 1–160 characters")
    stem = str(args.get("path", ""))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,180}", stem) or Path(stem).suffix:
        raise ValueError("path must be a relative extension-free output stem")
    if any(p in (".", "..") for p in Path(stem).parts):
        raise ValueError("path traversal is forbidden")
    formats = args.get("formats") or (["md"] if kind == "document" else ["svg"] if kind in {"mindmap", "dag", "relation"} else ["html"])
    if not isinstance(formats, list) or not formats or len(formats) > 4 or len(set(formats)) != len(formats) or any(f not in ALLOWED[kind] for f in formats):
        raise ValueError("invalid or duplicate formats for kind")
    if kind != "document" and not any(f in formats for f in ("txt", "md")):
        formats = [*formats, "txt"]
    sections, nodes, edges = args.get("sections") or [], args.get("nodes") or [], args.get("edges") or []
    if not all(isinstance(x, list) for x in (sections, nodes, edges)) or len(sections) > MAX_SECTIONS or len(nodes) > MAX_NODES or len(edges) > MAX_EDGES:
        raise ValueError("section, node, or edge limit exceeded")
    clean_sections = []
    for s in sections:
        if not isinstance(s, dict):
            raise ValueError("section must be an object")
        heading, body = str(s.get("heading", "")), str(s.get("body", ""))
        if len(heading) > 160 or len(body) > 20000:
            raise ValueError("section too long")
        clean_sections.append({"heading": heading, "body": body})
    clean_nodes, ids = [], set()
    for n in nodes:
        if not isinstance(n, dict) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", str(n.get("id", ""))):
            raise ValueError("invalid node ID")
        nid = n["id"]
        if nid in ids or len(str(n.get("label", nid))) > 120:
            raise ValueError("duplicate node or long label")
        ids.add(nid)
        clean = {"id": nid, "label": str(n.get("label", nid))}
        for axis in ("x", "y", "z", "w"):
            value = n.get(axis, 0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e6:
                raise ValueError("coordinates must be finite bounded numbers")
            clean[axis] = float(value)
        clean_nodes.append(clean)
    clean_edges = []
    for e in edges:
        if not isinstance(e, dict) or e.get("source") not in ids or e.get("target") not in ids or len(str(e.get("label", ""))) > 120:
            raise ValueError("edge endpoint or label invalid")
        clean_edges.append({"source": e["source"], "target": e["target"], "label": str(e.get("label", ""))})
    if kind != "document" and not clean_nodes:
        raise ValueError("graph needs nodes")
    if kind == "dag":
        graph = {nid: [] for nid in ids}
        for e in clean_edges:
            graph[e["source"]].append(e["target"])
        state = {}
        def walk(nid):
            if state.get(nid) == 1:
                raise ValueError("DAG contains a cycle")
            if state.get(nid) == 2:
                return
            state[nid] = 1
            for child in graph[nid]:
                walk(child)
            state[nid] = 2
        for nid in ids:
            walk(nid)
    note = str(args.get("note", ""))
    if len(note) > 4000:
        raise ValueError("note too long")
    return stem, title, clean_sections, clean_nodes, clean_edges, formats, note


def _lines(title, sections, nodes, edges, note):
    lines = [title, "=" * len(title), ""]
    for section in sections:
        if section["heading"]:
            lines += [section["heading"], "-" * len(section["heading"]), ""]
        lines += [section["body"], ""]
    if nodes:
        lines += ["Nodes", "-----"]
        lines += [f"{n['id']}: {n['label']} (x={n['x']:g}, y={n['y']:g}, z={n['z']:g}, w={n['w']:g})" for n in nodes]
        lines += ["", "Links", "-----"]
        lines += [f"{e['source']} -> {e['target']}: {e['label']}" for e in edges]
    if note:
        lines += ["", "Evidence and limits", "-------------------", note]
    return "\n".join(lines).strip() + "\n"


def _markdown(title, sections, nodes, edges, note):
    lines = [f"# {title}", ""]
    for s in sections:
        if s["heading"]:
            lines += [f"## {s['heading']}", ""]
        lines += [s["body"], ""]
    if nodes:
        lines += ["## Graph legend", ""]
        lines += [f"- `{n['id']}`: {n['label']} (x={n['x']:g}, y={n['y']:g}, z={n['z']:g}, w={n['w']:g})" for n in nodes]
        lines += ["", "## Links", ""]
        lines += [f"- `{e['source']}` → `{e['target']}`: {e['label']}" for e in edges]
    if note:
        lines += ["", "## Evidence and limits", "", note]
    return "\n".join(lines).strip() + "\n"


def _pdf(title, sections, nodes, edges, note):
    text = _lines(title, sections, nodes, edges, note)
    lines = []
    for paragraph in text.splitlines():
        lines.extend(textwrap.wrap(paragraph, width=93, replace_whitespace=False) or [""])
    pages = [lines[i:i+52] for i in range(0, len(lines), 52)] or [[]]
    objects = []
    def obj(data):
        objects.append(data)
        return len(objects)
    catalog = obj(b"")
    page_tree = obj(b"")
    font = obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    for page in pages:
        commands = [b"BT /F1 11 Tf 50 790 Td 14 TL"]
        for line in page:
            encoded = line.encode("latin-1", "replace").replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
            commands.append(b"(" + encoded + b") Tj T*")
        commands.append(b"ET")
        stream = b"\n".join(commands)
        content = obj(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
        page_ids.append(obj(f"<< /Type /Page /Parent {page_tree} 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>".encode()))
    objects[page_tree-1] = f"<< /Type /Pages /Kids [{' '.join(f'{n} 0 R' for n in page_ids)}] /Count {len(page_ids)} >>".encode()
    objects[catalog-1] = f"<< /Type /Catalog /Pages {page_tree} 0 R >>".encode()
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, data in enumerate(objects, 1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode() + data + b"\nendobj\n")
    start = len(out)
    out.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode())
    out.extend(f"trailer\n<< /Size {len(objects)+1} /Root {catalog} 0 R >>\nstartxref\n{start}\n%%EOF\n".encode())
    return bytes(out)


def _docx(title, sections, nodes, edges, note):
    paragraphs = [("Title", title)]
    for s in sections:
        if s["heading"]:
            paragraphs.append(("Heading1", s["heading"]))
        paragraphs.extend(("Normal", line) for line in s["body"].splitlines() or [""])
    if nodes:
        paragraphs += [("Heading1", "Graph legend")]
        paragraphs.extend(("Normal", line) for line in _lines("", [], nodes, edges, note).splitlines() if line)
    elif note:
        paragraphs += [("Heading1", "Evidence and limits"), ("Normal", note)]
    body = []
    for style, value in paragraphs:
        body.append(f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr><w:r><w:t xml:space="preserve">{xml_escape(value)}</w:t></w:r></w:p>')
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           '<w:body>' + ''.join(body) + '<w:sectPr/></w:body></w:document>')
    styles = ('<?xml version="1.0" encoding="UTF-8"?>'
              '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
              '<w:style w:type="paragraph" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
              '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>'
              '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style></w:styles>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>')
        z.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        z.writestr("word/document.xml", doc)
        z.writestr("word/styles.xml", styles)
    return buf.getvalue()


def _svg(title, kind, nodes, edges, note):
    count = len(nodes)
    width, height = 1000, max(300, 130 + count * 82)
    by_id = {n["id"]: i for i, n in enumerate(nodes)}
    if kind == "mindmap":
        coordinates = {n["id"]: (180 if i == 0 else 650, 110 + i * 72) for i, n in enumerate(nodes)}
    else:
        coordinates = {n["id"]: (190 + (i % 3) * 300, 110 + (i // 3) * 135) for i, n in enumerate(nodes)}
        height = max(300, 240 + ((count - 1) // 3) * 135)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title, quote=True)}">',
             '<rect width="100%" height="100%" fill="#fff"/>',
             f'<text x="30" y="35" font-family="Arial" font-size="23">{html.escape(title)}</text>',
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#51627b"/></marker></defs>']
    for e in edges:
        x1, y1 = coordinates[e["source"]]
        x2, y2 = coordinates[e["target"]]
        parts += [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#51627b" stroke-width="2" marker-end="url(#arrow)"/>',
                  f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2-6}" font-family="Arial" font-size="12">{html.escape(e["label"])}</text>']
    for n in nodes:
        x, y = coordinates[n["id"]]
        parts += [f'<rect x="{x-105}" y="{y-20}" width="210" height="42" rx="8" fill="#e8f2ff" stroke="#2170b5"/>',
                  f'<text x="{x}" y="{y+5}" text-anchor="middle" font-family="Arial" font-size="14">{html.escape(n["label"][:28])}</text>']
    if note:
        parts.append(f'<desc>{html.escape(note)}</desc>')
    return ('\n'.join(parts) + '\n</svg>\n').encode()


def _spatial_html(title, kind, nodes, edges, note):
    data = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=True).replace("<", "\\u003c")
    low, high = min(n["w"] for n in nodes), max(n["w"] for n in nodes)
    midpoint = (low + high) / 2
    slider = (f'<label for="w">Fourth coordinate (w): <output id="wv">{midpoint:g}</output></label>'
              f'<input id="w" type="range" min="{low:g}" max="{high:g}" step="any" value="{midpoint:g}"/>') if kind == "graph4d" else ''
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:16px system-ui;margin:2rem;color:#182230}}canvas{{border:1px solid #ccd;width:min(100%,900px);height:560px}}label{{display:block;margin:1rem 0}}#legend{{white-space:pre-wrap}}</style>
<h1>{html.escape(title)}</h1><p>{html.escape(note)}</p><p>Axes: x, y, z. Drag horizontally to rotate; drag vertically to tilt. {"The slider filters the fourth coordinate w." if kind == "graph4d" else ""}</p>
{slider}<canvas id="view" width="900" height="560" aria-label="Interactive {kind} graph"></canvas><h2>Accessible graph data</h2><pre id="legend"></pre>
<script>const graph={data};const canvas=document.getElementById('view'),c=canvas.getContext('2d');
const slider=document.getElementById('w');let yaw=0.5,pitch=0.25,drag=false,last=[0,0];
document.getElementById('legend').textContent=graph.nodes.map(n=>`${{n.id}}: ${{n.label}} (${{n.x}}, ${{n.y}}, ${{n.z}}, ${{n.w}})`).join('\n')+'\n'+graph.edges.map(e=>`${{e.source}} -> ${{e.target}}: ${{e.label}}`).join('\n');
function render(){{c.clearRect(0,0,900,560);c.fillStyle='#fff';c.fillRect(0,0,900,560);const w=slider?Number(slider.value):0;if(slider)document.getElementById('wv').textContent=w;
const band=Math.max(0.001,(Number(slider?.max??0)-Number(slider?.min??0))/4);const visible=graph.nodes.filter(n=>!slider||Math.abs(n.w-w)<=band);const span=Math.max(1,...graph.nodes.flatMap(n=>[Math.abs(n.x),Math.abs(n.y),Math.abs(n.z)]));
const points={{}};for(const n of visible){{let x=n.x*Math.cos(yaw)-n.z*Math.sin(yaw),z=n.x*Math.sin(yaw)+n.z*Math.cos(yaw),y=n.y*Math.cos(pitch)-z*Math.sin(pitch);points[n.id]=[450+x*190/span,280-y*190/span];}}
c.strokeStyle='#7c90ad';c.lineWidth=2;for(const e of graph.edges){{if(!points[e.source]||!points[e.target])continue;c.beginPath();c.moveTo(...points[e.source]);c.lineTo(...points[e.target]);c.stroke();}}
for(const n of visible){{const [x,y]=points[n.id];c.fillStyle='#176bb5';c.beginPath();c.arc(x,y,8,0,Math.PI*2);c.fill();c.fillStyle='#182230';c.font='14px system-ui';c.fillText(n.label,x+12,y-10);}}}}
canvas.onpointerdown=e=>{{drag=true;last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)}};
canvas.onpointerup=()=>drag=false;canvas.onpointermove=e=>{{if(!drag)return;yaw+=(e.clientX-last[0])*0.01;pitch+=(e.clientY-last[1])*0.01;last=[e.clientX,e.clientY];render()}};
if(slider)slider.oninput=render;render();</script></html>'''.encode()


def render(session, args: dict) -> list[dict]:
    stem, title, sections, nodes, edges, formats, note = _validate(args)
    pending = []
    for fmt in formats:
        rel = f"{stem}.{fmt}"
        path = session._path(rel)
        if fmt == "txt": payload = _lines(title, sections, nodes, edges, note).encode()
        elif fmt == "md": payload = _markdown(title, sections, nodes, edges, note).encode()
        elif fmt == "pdf": payload = _pdf(title, sections, nodes, edges, note)
        elif fmt == "docx": payload = _docx(title, sections, nodes, edges, note)
        elif fmt == "svg": payload = _svg(title, args["kind"], nodes, edges, note)
        else: payload = _spatial_html(title, args["kind"], nodes, edges, note)
        pending.append((path, rel, payload))
    outputs = []
    for path, rel, payload in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".ple-artifact-", delete=False) as tmp:
            temp_name = tmp.name
            try:
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
            except BaseException:
                os.unlink(temp_name)
                raise
        os.replace(temp_name, path)
        outputs.append({"path": rel, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    return outputs


def make_handler(session):
    async def handler(args):
        return json.dumps({"artifacts": render(session, args)}, separators=(",", ":"))
    return handler
