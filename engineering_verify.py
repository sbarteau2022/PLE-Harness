"""Bounded, deterministic engineering-manifest checks for every model seat."""

from __future__ import annotations

import hashlib
import json
import math


TOOL = {"type": "function", "function": {
    "name": "engineering_verify",
    "description": "Check declared 3D boxes, connection routes, clearances and time-series flow balance. Returns findings, not CAD or physics certification.",
    "parameters": {"type": "object", "properties": {
        "manifest": {"type": "object", "description": "Inline engineering manifest v1; see docs/ENGINEERING-VERIFIER-DESIGN.md."},
    }, "required": ["manifest"]},
}}

UNITS = {"mm", "m", "in", "ft"}
MAX_ITEMS = 200
BOUND = 1e9
TOL = 1e-8


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > BOUND:
        raise ValueError(f"{name} must be a finite bounded number")
    return float(value)


def _items(value, name):
    if not isinstance(value, list) or len(value) > MAX_ITEMS:
        raise ValueError(f"{name} must be a list of at most {MAX_ITEMS}")
    if not all(isinstance(x, dict) for x in value):
        raise ValueError(f"{name} entries must be objects")
    return value


def _id(value, name):
    if not isinstance(value, str) or not value or len(value) > 80:
        raise ValueError(f"{name} must be a nonempty ID of at most 80 characters")
    return value


def verify(manifest: dict) -> dict:
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > 250_000:
        raise ValueError("manifest exceeds 250000 bytes")
    if manifest.get("units") not in UNITS:
        raise ValueError("units must be mm, m, in, or ft")
    findings = []

    def add(code, item, detail):
        findings.append({"code": code, "id": item, "detail": detail})

    components = {}
    component_items = _items(manifest.get("components", []), "components")
    if not component_items:
        raise ValueError("manifest requires at least one component")
    for c in component_items:
        cid = _id(c.get("id"), "component id")
        if cid in components:
            raise ValueError(f"duplicate component: {cid}")
        box = c.get("box")
        if not isinstance(box, dict):
            raise ValueError(f"component {cid} needs box")
        vals = {k: _number(box.get(k), f"{cid}.{k}") for k in ("x", "y", "z", "dx", "dy", "dz")}
        if any(vals[k] <= 0 for k in ("dx", "dy", "dz")):
            add("nonpositive_dimension", cid, "box dimensions must be positive")
        if not isinstance(c.get("source"), str) or not c.get("source", "").strip():
            add("missing_source", cid, "drawing or specification reference required")
        components[cid] = vals
    ids = list(components)
    for i, left in enumerate(ids):
        a = components[left]
        if any(a[k] <= 0 for k in ("dx", "dy", "dz")):
            continue
        for right in ids[i + 1:]:
            b = components[right]
            if all(b[k] > 0 for k in ("dx", "dy", "dz")) and all(
                max(a[k], b[k]) < min(a[k] + a["d" + k], b[k] + b["d" + k]) - TOL
                for k in ("x", "y", "z")
            ):
                add("box_overlap", f"{left}|{right}", "axis-aligned boxes overlap; inspect whether this is intentional")

    connections = {}
    graph = {cid: set() for cid in components}
    for e in _items(manifest.get("connections", []), "connections"):
        eid = _id(e.get("id"), "connection id")
        if eid in connections:
            raise ValueError(f"duplicate connection: {eid}")
        src, dst = _id(e.get("from"), "connection from"), _id(e.get("to"), "connection to")
        if src not in components or dst not in components:
            add("unknown_endpoint", eid, f"{src} -> {dst}")
        else:
            graph[src].add(dst)
        connections[eid] = (src, dst)
    for route in _items(manifest.get("required_routes", []), "required_routes"):
        src, dst = _id(route.get("from"), "route from"), _id(route.get("to"), "route to")
        if src not in graph or dst not in graph:
            add("unknown_route_endpoint", f"{src}|{dst}", "route endpoint is not a component")
            continue
        seen, pending = {src}, [src]
        while pending:
            for nxt in graph[pending.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    pending.append(nxt)
        if dst not in seen:
            add("disconnected_route", f"{src}|{dst}", "no directed path")

    for gap in _items(manifest.get("clearances", []), "clearances"):
        left, right = _id(gap.get("a"), "clearance a"), _id(gap.get("b"), "clearance b")
        axis = gap.get("axis")
        if axis not in ("x", "y", "z"):
            raise ValueError("clearance axis must be x, y, or z")
        minimum = _number(gap.get("min"), "clearance min")
        if minimum < 0:
            raise ValueError("clearance min must be nonnegative")
        if left not in components or right not in components:
            add("unknown_clearance_component", f"{left}|{right}", "component missing")
            continue
        a, b = components[left], components[right]
        actual = max(b[axis] - (a[axis] + a["d" + axis]), a[axis] - (b[axis] + b["d" + axis]), 0)
        if actual + TOL < minimum:
            add("insufficient_clearance", f"{left}|{right}", f"{axis} gap {actual:g} < {minimum:g} {manifest['units']}")

    states = _items(manifest.get("states", []), "states")
    if states and (not manifest.get("time_unit") or not manifest.get("flow_unit")):
        raise ValueError("states require time_unit and flow_unit")
    junctions = manifest.get("junctions", [])
    if not isinstance(junctions, list) or len(junctions) > MAX_ITEMS:
        raise ValueError("junctions must be a bounded list")
    for node in junctions:
        _id(node, "junction")
        if node not in components:
            add("unknown_junction", node, "junction is not a component")
    previous = None
    for index, state in enumerate(states):
        t = _number(state.get("time"), f"state {index} time")
        if previous is not None and t <= previous:
            add("nonmonotonic_time", str(index), "timestamps must strictly increase")
        previous = t
        flows = state.get("flows")
        if not isinstance(flows, dict):
            raise ValueError(f"state {index} flows must be an object")
        if set(flows) != set(connections):
            add("flow_ids_mismatch", str(index), "flows must contain exactly the declared connection IDs")
        numeric = {eid: _number(value, f"state {index} flow {eid}") for eid, value in flows.items()}
        for node in junctions:
            if node not in components or set(flows) != set(connections):
                continue
            net = sum(value * ((1 if dst == node else 0) - (1 if src == node else 0))
                      for eid, value in numeric.items() for src, dst in [connections[eid]])
            scale = max(1.0, sum(abs(value) for value in numeric.values()))
            if abs(net) > TOL * scale:
                add("flow_imbalance", f"{index}|{node}", f"net {net:g} {manifest['flow_unit']}")
    return {"passed": not findings, "checks": {"components": len(components), "connections": len(connections),
            "states": len(states), "junctions": len(junctions)}, "findings": findings,
            "input_sha256": hashlib.sha256(encoded).hexdigest(),
            "scope": "declared manifest invariants only; not CAD, IFC, CFD, or field certification"}


def run(args: dict) -> str:
    return json.dumps(verify(args["manifest"]), sort_keys=True, separators=(",", ":"))


def main() -> int:
    """Local CI gate: python -m engineering_verify path/to/manifest.json."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        result = verify(json.loads(args.manifest.read_text()))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
