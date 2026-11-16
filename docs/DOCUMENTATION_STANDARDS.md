# PLE documentation standards

Date: 2026-11-16. Applies to the PLE harness and artifacts its frontier agents
write. This is the canonical local standard; the referenced Desktop
`OPS_BOARD/DOCUMENTATION STANDARDS.MD` was not present when this file was made.

## Content first

Start with the reader's decision or question. State current status, inputs,
method, evidence, result, limitations, and next action in that order when they
apply. Use one claim per paragraph. Explain the first use of specialized terms.
Put paths, units, model identity, run date, and source receipts next to measured
claims. Mark proposal, component test, live measurement, and inference clearly.

## Document outputs

- `.txt`: plain UTF-8, stable headings, no hidden formatting.
- `.md`: one `#` title, ordered heading levels, relative links, fenced code
  with language, alt text for images, and source paths under each figure.
- `.pdf`: selectable text, readable page margins, title and section hierarchy.
- `.docx`: real Word paragraphs and headings, no screenshot-only text.

Render only formats the task needs. A Markdown source may accompany exported
formats for revision. Check that every declared output exists and has a hash.

## Visual outputs

For CAD-derived geometry or physical time simulations, follow
[`PLE-ENGINEERING-MODELING.md`](PLE-ENGINEERING-MODELING.md) and retain source,
unit, solver, and verification receipts for each claimed state.

Use a mind map for topic hierarchy, a DAG for ordered work and dependencies,
and a relational graph for typed links. Label every node and edge. A DAG must
be acyclic. Provide a text legend/adjacency description alongside each visual
so the information remains accessible without the graphic. Use SVG for stable
2D visuals. Use self-contained HTML for interactive 3D and 4D projections,
with axis/slider labels and a stated meaning for each dimension. Never imply
that a projection is a measured physical geometry.

## Evidence and review

Distinguish source facts from derived values and hypotheses. For charts, name
the source receipt, sample count, axes, units, exclusions, and uncertainty.
Review legibility, links, structure, hash receipts, and whether an independent
reader can reproduce the conclusion. Preserve negative results and failures.

## PLE tool contract

The model loads `skill("ple-artifacts")` for authoring rules, then calls
`artifact_render` with structured text and graph data. The renderer writes
inside the active workspace and returns file hashes. It does not replace task
MCP document applications; use those when the task requires native edits in an
existing application document.
