# PLE engineering and 2D-to-4D modeling

This is a capability map for the PLE harness, grounded in the [PLE kit protocol](https://github.com/google-research/ple/blob/8320f0402fdca9f42476607f72bcfd5271cd50c9/docs/protocol.md), [kit checks](https://github.com/google-research/ple/blob/8320f0402fdca9f42476607f72bcfd5271cd50c9/kit/README.md), and public task briefs. It is not a claim that a CAD or physics solver is present in every task.

## Tools and availability

| Layer | Tools | Boundary |
| --- | --- | --- |
| Every PLE task | Harness `bash`, `read_file`, `write_file`, `engineering_verify`; kit `exec`, `read_file`, `write_file` | Inspect sources, run headless scripts, write deliverables, check a declared engineering manifest. |
| Task-local applications | `mcp_list`, `mcp_schema`, `mcp_call` over kit `http` | Discover the actual MCP catalog. CAD, PDF, spreadsheet, and document tools exist only when the task exposes them. Never hardcode a presumed CAD endpoint. |
| Task environment | Installed Python/CLI packages and input files | Inspect imports and binaries first. The MEP task pins IfcOpenShell 0.8.4 and NumPy; another task may supply no BIM library. |
| Frontier workspace | `artifact_render` | Portable reports and explanatory 3D/4D graphs. This is neither a solid-modeling kernel nor a hydraulic solver. |

The kit permits up to eight concurrent requests, subject to task limits. The task container owns source files and final deliverables; the harness must reread its outputs. [Kit artifact guidance](https://github.com/google-research/ple/blob/8320f0402fdca9f42476607f72bcfd5271cd50c9/docs/artifacts.md) and task instructions decide accepted formats.

## 2D drawings to a defensible 3D model

1. Inventory plans, sections, elevations, schedules, notes, and units. Record a source page or file for each extracted dimension. For raster sheets, calibrate scale from known dimensions and cross-check a second span; mark occluded or missing depth as an assumption.
2. Choose one world coordinate frame and consistent units. Register each sheet against shared grid lines or control points. Reconcile plan positions with section heights and equipment schedules before extrusion.
3. Build parametric footprints and solids, then equipment and connected MEP routes. Give each object a stable ID, type, source reference, material/system metadata, and spatial parent. Use IFC4 when the task requires BIM. A raw IFC4 STEP writer is possible when no BIM library exists, but demands additional syntax and geometry checks.
4. Verify source dimensions, unit transforms, topology, route continuity, elevations, clearances, accessibility, collisions, and required property sets. Validate IFC schema and EXPRESS rules where available. Render plan, section, and isometric views and compare them with the original drawings. Keep a drawing-note reconciliation register.

The local public dev-set brief `ple-dev-set/tasks/data-center-site-model/instruction.md` requires an IFC4 campus model and hard feasibility gates. The local `ple-dev-set/tasks/data-hall-mep-routing/instruction.md` brief requires drawing-faithful routing, exact IFC classes and properties, headroom/clearance checks, deterministic headless generation, and `express_rules=True` validation. The dev-set inputs are distributed separately from this repository; these are task-specific contracts, not defaults for all PLE tasks.

## 4D cooling-system model

Here 4D means the same identified 3D components plus time-indexed state. Maintain a pipe/pump/valve/tank graph linked by stable IDs to the CAD objects. Store initial conditions, boundary conditions, pump curves, valve states, fluid properties, event schedule, and timestep. Compute flow, pressure/head, temperature, and chemistry only to the fidelity supported by inputs and solver. Check mass balance at every node and timestep; check energy balance if thermal behavior is claimed. Track supply, return, bypass, discharge, and accumulated volumes. Animate computed states on the geometry with a clock, legend, units, and pause/scrub controls. Color or particles alone do not prove water movement.

The local public dev-set brief `ple-dev-set/tasks/cooling-water-flush-plan/instruction.md` calls for flow, velocity, Reynolds number, head, duration, water/discharge quantities, chemistry, resource limits, and a predecessor schedule. Those CSV and procedure outputs are authoritative for that task. A 4D rendering is a review aid unless explicitly requested as a deliverable. A hydraulic network model is not CFD; resolve the required fidelity before claiming local turbulence or transient behavior.

## Verification loops and engineering practice

| Loop | Pass condition |
| --- | --- |
| Input | Every geometry and parameter has a source, unit, and confidence/assumption status. |
| Geometry | Independent dimension/area checks; connected routes; no prohibited clashes; measurable clearances. |
| BIM | Reopen exported IFC; verify hierarchy, classes, coordinates, properties, and schema rules. |
| Physics | Recompute balances and constraints from source inputs; test closed valves, missing branches, zero flow, reversals, and extreme schedules. |
| Time | Monotonic timestamps, valid predecessors, resource capacity, event consistency, and reproducible state at each frame. |
| Deliverable | Headless rebuild from script, reopen files, inspect views, hash outputs, record failures and fixes. |

Apply the task's adopted codes, drawing notes, and project specifications first. Record standard title, edition, clause, and applicability before claiming compliance; IFC4 is an exchange schema, not a substitute for engineering review. Use hard feasibility gates before optimizing length, cost, or visual quality, as [PLE rules](https://github.com/google-research/ple/blob/8320f0402fdca9f42476607f72bcfd5271cd50c9/docs/rules.md) require for constrained design tasks. Do not inspect hidden graders or turn a generated animation into measured field evidence. Run kit checks and the permitted public evaluation after the artifact-level loops.

## Implementation gaps

The harness now has task-tool discovery, explanatory graph rendering, and a bounded [engineering verifier](ENGINEERING-VERIFIER-DESIGN.md) for declared boxes, routes, clearances and flow states. A drawing-registration service, typed IFC geometry builder, full constraint checker, hydraulic/thermal solver adapter, and CAD-linked time-series viewer remain model-agnostic gaps. Define their contracts and preregister geometry/physics validation before implementation. Keep all new implementation inside `harness/`.
