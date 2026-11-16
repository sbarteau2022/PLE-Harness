# Live multi-turn call probe amendment

Date: 2026-11-16. Status: PREREGISTERED AMENDMENT.

[PROPOSED] The expensive PLE-dev smoke is replaced by one controlled live task
for harness plumbing and behavior qualification. It requests 27 tool calls:
20 distinct operations, six identical checksum verifications, and one finish.
The same instruction, files, allowed commands, model settings, and 30-step cap
are used for every arm.

[PROPOSED] The probe measures live response IDs, calls/steps, latency, tokens,
tool names, successes/errors, repeated-call behavior, finish sequencing,
deliverable hashes, and the complete observer event trajectory. It does not
measure PLE engineering-task quality and cannot support a task-quality claim.

[PROPOSED] A run is closed only when it has a live response ID, 20–30 actual
tool calls, 5–8 calls with the prescribed repeated fingerprint, all three
deliverables, and a finish call by itself after prior results. Runs outside
these bounds are logged but excluded. No task-specific grader or Harbor is
used.
