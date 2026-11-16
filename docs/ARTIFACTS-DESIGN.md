# PLE artifact authoring design

Date: 2026-11-16. Status: implementation contract.

## Goal

Give every OpenAI-compatible driver in the PLE frontier loop one typed tool to
produce documents and explanatory visuals. Zed uses the same frontier entry
point and therefore receives the same tool. The tool takes structured content,
not model-specific commands or layer counts.

## Formats and authority

`artifact_render` accepts a document title, sections, optional graph nodes and
edges, a kind, and requested formats. Text and Markdown use UTF-8. PDF and
DOCX use deterministic local writers. Workflow DAG and relational/mind-map
views use SVG. Spatial graphs use a self-contained HTML canvas: 3D coordinates
are rotated and projected; 4D graphs expose the fourth coordinate as a slider.
Every visual also emits a text adjacency file unless Markdown is requested.
Each output is written inside the selected workspace through `LocalSession._path`
and receives a path, byte count and SHA-256 receipt. No network, third-party
model service, or provider-specific artifact API is required. Browser Artifacts
may view the portable output; they are not the storage authority.

## Boundaries

The renderer accepts no executable HTML or JavaScript from the model. It
escapes labels, caps sections, nodes and edges, rejects path escapes, and checks
for cycles when kind is `dag`. It does not read task graders or artifact
directories. A 4D view is a time/parameter projection, not a physical fourth
dimension proof. PDF and DOCX preserve text and section structure; rich graph
views are separate linked artifacts.

## Verification

Exercise the tool through `ToolRunner`, not just its internal renderer. Verify
every requested format, hashes and signatures, path containment, DAG cycle
rejection, HTML escaping, and the Zed frontier tool list. Update README and
the handoff with the capability and its limits.
