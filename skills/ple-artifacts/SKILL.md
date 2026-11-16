---
name: ple-artifacts
description: Create or revise a PLE report, handoff, document, PDF, DOCX, mind map, workflow DAG, relational diagram, or interactive 3D/4D graph. Use this skill whenever a frontier agent needs a portable artifact or a visual explanation, including work launched from Zed.
---

# PLE artifacts

Read `docs/DOCUMENTATION_STANDARDS.md` in the harness before publishing a
document. Write the claim and evidence first, then choose the output formats.
For drawing-derived CAD or time-varying physical systems, also read
`docs/PLE-ENGINEERING-MODELING.md` before selecting tools or claiming a result.

Call `artifact_render` with structured sections and, for visuals, nodes and
edges. Use `kind: document` for text formats, `mindmap`, `dag`, or `relation`
for SVG, and `graph3d` or `graph4d` for interactive HTML. Output paths are
relative to the active workspace. Include a caption or note that explains
the dimensions and data source. The tool returns path, size and SHA-256 for
each generated file; check that receipt before `finish`.

For a DAG, verify the workflow really has no cycles. For 3D/4D, provide real
numeric coordinates and axis meanings; the fourth coordinate is controlled by
a slider. If no coordinates exist, use a 2D relation diagram instead.

Do not turn a simulation into a live measurement. Do not hide failed cases or
claim physical energy or geometry from a visualization alone.
