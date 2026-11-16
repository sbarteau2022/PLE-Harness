# Claude project description — PLE Harness

**Repository:** https://github.com/sbarteau2022/PLE-Harness

## Paste into “Describe your project, goals, subject, etc.”

I’m building PLE Harness, a model-agnostic agent runtime and experimental platform for Physical-Engineering’s Last Exam. Its purpose is to find out whether an AI system can do more than produce plausible engineering prose: read unfamiliar specifications and drawings, use the tools provided by a task, construct real deliverables, check its calculations and constraints, and leave an auditable trail of what it tried and learned.

The working system is primarily Python, with Rust for the local runtime and nested K/V bridge, plus shell tooling for reproducible evaluation. It speaks the PLE WebSocket protocol and connects OpenAI-compatible hosted or local models to bounded shell and file tools, task-specific MCP applications, research retrieval, artifact generation, and engineering checks. The driver seat can change models without changing the harness contract or assuming a fixed transformer layer count.

One research direction is a multimodel mind: verified knowledge atoms and trajectories in nested SurrealDB, addressed and retrieved in a hyperbolic space so one model can build on work another model established. The nested store also owns tensor-page identities and bytes; short-term SQLite session memory is separate and does not hold tensors. The storage and readback contracts are implemented, but reusable cross-context tensor attention and a production local forward-pass adapter remain open qualification gates.

I’m designing a recall-first forward policy: let a small, bounded scout pass decide whether to retrieve relevant atoms before the main prefill, then measure the *total* cost of scouting, retrieval, prefill, decoding, and tools. A separate local-runtime experiment will test sparse K/V page selection. The proposed optimizer draws on active inference and the Free Energy Principle to balance task success, uncertainty reduction, latency, and compute cost. That controller, any physical-energy savings, and any true reduction in attention work still require implementation and matched live evidence.

The engineering method is deliberately empirical. The harness records model responses, tool-call trajectories, token usage, hashes, memory receipts, failures, and judged deliverables. It is being compared with other harnesses in preregistered ablations across API and local models, including mind-on versus mind-off runs. Early live probes and a token-carving replay demonstrate functioning tool loops and reduced prompt tokens in one matched replay; they do not yet prove better engineering quality or lower end-to-end cost. The full cross-harness campaign is incomplete.

My goal is to turn this into a trustworthy product-development and systems-engineering platform: one that can move from 2D source drawings to checked 3D CAD/IFC models, connect those models to time-based 4D simulations, and use explicit verification loops to catch geometry, routing, physics, schedule, and provenance errors. I want every improvement to survive adversarial tests and real physical-engineering tasks, with honest negative results and clear limits on what the evidence proves.
