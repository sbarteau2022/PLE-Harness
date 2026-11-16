# Campaign failure ledger

Entries are append-only and excluded from the final admitted-results table.

## 2026-11-16 live multi-turn probe exclusions

[VERIFIED] The first generic Mistral probe launch stopped before a model call
because the standalone script lacked the harness root on `sys.path`. The import
was fixed and the rerun closed; the empty `20261116-061933` directory is not
evidence.

[VERIFIED] PLE + Step 5 (`20261116-062626`) completed all deliverables but made
only three of the six prescribed identical checksum calls. The PLE repeated-
state guard then prompted a different approach; Step 5 attempted one combined
verification command, which the probe allowlist rejected, and finished. This
is useful observed stop/replan behavior, but the preregistered 5–8-repeat gate
was unmet, so the run is excluded and no atoms were minted for it.

## 2026-11-16 Generic baseline packaging gate

[VERIFIED] The first generic-baseline kit check built and routed the model but
the listener exited because `serve_generic.py` imported the PLE normalizer only
for its marker constant while the minimal entry omitted the normalizer's
receipt dependency. No benchmark task ran. The generic server now owns the
literal observation marker and imports no PLE normalizer. The repeated check
passed build, live model routing, and listener; smoke remained intentionally
skipped. Evidence: `ablations/work/generic-mistral/`.

## 2026-11-16 Mis-scoped PLE-only pilot

[VERIFIED] A live Mistral `wind-diaphragm-shear` PLE run was started after
misreading "generic harness" as "generic PLE entry." That would collapse the
baseline-versus-PLE comparison and is not the requested ablation. The trial
remained running for about seven minutes without token, judge, deliverable, or
completion telemetry and was cancelled cleanly. Its key file was removed and
its containers were stopped. The run at
`ablations/runs/ple/mistral/20261116-054318/` is permanently excluded.

## 2026-11-16 Autonomous API campaign orchestration stop

[VERIFIED] `ablations/api/` was launched with the autonomous command, but the
worker exhausted its Codex usage limit while packaging the DeepSeek Harness
workspace. It made no paid model call and produced no live task attempt.
`ablations/api/results.json` therefore contains only explicit `incomplete` /
`UNMET: no attempt` cells and is excluded from all comparisons. The local mind
A/B command was not started.

[VERIFIED] A cached keyless retry of `scripts/run_ablation.sh harness2 check`
again passed submission/file validation but stalled at the same frozen Bun
dependency resolution. It was interrupted after confirming the identical
failure; no model access was exercised.

## 2026-11-15 Harness2 source baseline

[VERIFIED] Attempt 1 failed during collection because the pinned source package
was not on Python's import path. No model or task ran. Re-approach: run with the
pinned checkout in `PYTHONPATH`.

[VERIFIED] Attempts 2 and 3 collected and ran the suite, then reported 4 failed,
60 passed, and 6 skipped because the JobBench runner invokes GNU `timeout`,
which was absent on macOS. No model or task ran. Re-approach: an upstream-neutral
PLE-local compatibility command is tested under `tests/test_ablation_timeout.py`
and is prepended to `PATH` only for campaign processes.

## 2026-11-15 DeepSeek Harness source baseline

[VERIFIED] The frozen dependency installation passed the upstream supply-chain
policy check. A subsequent standalone `build:check` failed because generated
`lib/types` artifacts were absent (TS6305), so it is not an admitted build.
Re-approach: execute the repository-declared `build:artifacts` dependency before
repeating `build:check`.

## 2026-11-15 Harness2 PLE packaging gate

[VERIFIED] The first no-smoke kit check stopped before build because the
alternate submission contained unsupported metadata key `description`. No
model or task ran. The key was removed; the generated entry is rebuilt from the
corrected source before retry.

# 2026-11-15 Harness2 image gate interrupted

- Cell: `harness2:mistral`
- Action: keyless `kit check --no-smoke`; no model access exercised.
- Observed: submission and 4,648-file validation passed; Python dependencies
  installed; OpenCode frozen Bun install remained at `Resolving dependencies`.
- Disposition: manually interrupted for operator handoff. Incomplete and
  excluded from admitted results. Rerun after diagnosing image installation.
- Evidence: `ablations/work/harness2-mistral/build.log` and `builder.log`.


# 2026-11-16 Harness2 pilot run 20261116-002016 (wind-diaphragm-shear)

- Cell: `harness2:mistral`, task `wind-diaphragm-shear`.
- Action: `scripts/run_ablation.sh harness2 run wind-diaphragm-shear`.
- Observed: score 0.0, `measured: false`, `status: evaluator: RuntimeError`.
  Harbor failed before the agent started: every `docker compose --project-name ...`
  invocation exited 125 with `unknown flag: --project-name`. No model call ran
  inside the pilot; the serve container never started the task.
- Disposition: failed, excluded from admitted results. Rerun after the compose
  fault is fixed; a later manual `docker compose --project-name test123 ls`
  succeeded, so the fault looks environmental/transient, not a harness defect.
- Evidence: `ablations/runs/harness2/mistral/20261116-002016/` (results.json,
  export.json, attempt1/jobs/dev/job.log).

# 2026-11-16 Harness2 pilot rerun 20261116-005454 (wind-diaphragm-shear)

- Cell: `harness2:mistral`, task `wind-diaphragm-shear`.
- Action: `scripts/run_ablation.sh harness2 run wind-diaphragm-shear` on the
  rebuilt entry (telemetry markers, MCP parity, per-seat builder).
- Observed: score 0.0, `measured: false`, `status: evaluator: RuntimeError`.
  The earlier `unknown flag: --project-name` fault did not recur. Harbor now
  failed building the task environment: `FROM python:3.12-slim` ->
  `NotFound: content digest sha256:c61cffc3...: not found`. No model call ran
  inside the pilot. Re-approach: `docker pull python:3.12-slim` succeeded
  (digest `sha256:5e2bdb1b...`), so the cache was stale; rerun the pilot.
- Disposition: failed, excluded from admitted results.
- Evidence: `ablations/runs/harness2/mistral/20261116-005454/` (results.json,
  attempt1/jobs/dev/job.log).

# 2026-11-16 Harness2 pilot rerun 20261116-010150 (wind-diaphragm-shear)

- Cell: `harness2:mistral`, task `wind-diaphragm-shear`.
- Observed: score 0.0, `measured: false`, `status: evaluator: RuntimeError`.
  Task environment image built for arm64 but the kit pins `linux/amd64`:
  `image ... was found but does not provide the specified platform
  (linux/amd64)`. Root causes found: (1) every `~/.docker/cli-plugins/*`
  symlink was stale (AppTranslocation temp path), so `docker buildx` was
  missing and compose fell back to the classic builder; fixed by repointing
  the links at `/Applications/Docker.app/...` (buildx v0.38.1 restored).
  (2) The colima builder exposes only `linux/arm64, linux/386`;
  `qemu-x86_64` binfmt is disabled in the VM. Re-approach: restart colima
  with `--vm-type vz --vz-rosetta` per the README, verify `linux/amd64` in
  `docker buildx ls`, then rerun the pilot.
- Disposition: failed, excluded from admitted results.
- Evidence: `ablations/runs/harness2/mistral/20261116-010150/`.
