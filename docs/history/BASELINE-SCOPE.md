# Research harness baseline — scope

Subject: `research-grade-harness/` (crate `rgah` v0.1.0), as copied into this folder.
Source: `MetaSuperIntelligenceLabs-GlobalHack-A-thon-SoloDevTrack/research-grade-harness` (origin: sbarteau2022/MetaSuperIntelligenceLabs-GlobalHack-A-thon-SoloDevTrack). The copy has no `.git`, so record the source commit when the baseline runs. The last recorded commit at copy time is `8723065c`.

Purpose: a reproducible, dated record of what the harness does and does not prove today. This is a measurement, not a feature pass. Nothing here changes harness code.

## 1. Current state (from existing receipts)

From `GRADING-2026-11-05.md` and `RG-N3-WIRE-RECEIPT-2026-11-05.txt`:

| Check | Last result |
|---|---|
| `cargo build --release --offline` | built |
| `cargo test --offline` | 0 failed. lib 45; `a14_run01` 1; `bounded_resource_policy` 3; `build_345` 10; `engine_n3` 5 (3 ignored); `globalmind_bridge` 1; `implementation` 8; `pressure` 11; `receipt_adversarial` 3 |
| `cargo clippy --offline --all-targets -- -D warnings` | clean |
| Live N3 gate (`--features live-n3`, `HYPERMIND_MCP_BIN` set) | 8 passed |
| `bridge_reads_one_real_atom_as_binary_recall` (`--ignored`) | passed against a disposable SurrealKV |
| `rg_n3_wire` (isolated mode) | PASS, 39 map tools |

Explicitly **not** evidence (per `NO-SYNTHETIC-BENCHMARKS.md` and the grading doc): synthetic benches, `longtask_bench`, `benches/smoke`, and any real-model run. No quality, energy, or throughput claim is supported.

## 2. Baseline scope — in

Run in this order. Use `--offline` throughout, as the earlier runs did.

1. **Toolchain record.** Capture `rustc --version` (1.99.0 at copy time) and `cargo --version`. Edition 2024 needs a recent toolchain.
2. **Static gates.** `cargo fmt --check`, `cargo build --release --offline`, `cargo clippy --offline --all-targets -- -D warnings`.
3. **Default test suite.** `cargo test --offline --all-targets`. Expect the per-target counts in section 1.
4. **Live N3 gate.** Rebuild `hypermind-mcp` from `hypermind-nested-kv` (the earlier binary was stale). Then run `HYPERMIND_MCP_BIN=<abs path> cargo test --offline --features live-n3 --test engine_n3 -- --test-threads=1`. Use `HYPERMIND_SURREAL=off` for isolated mode, as the Nov 5 rerun did.
5. **GlobalMind bridge live test.** `cargo test --offline --test globalmind_bridge_live -- --ignored --nocapture`. Needs `surreal` (installed at `/opt/homebrew/bin/surreal`) and a disposable instance on a random loopback port. The sandbox may block loopback binds, so run this step unsandboxed.
6. **Not yet run, decide in or out.** `map_plugin_live` (feature `live-map-plugin`), never run per the grading doc. Include it only if you want that coverage.
7. **Receipt.** Write `BASELINE-RESULTS-<date>.md` in this folder with: source commit, toolchain versions, each command, pass/fail counts, and any environment fixes. Label every claim with the design spec's evidence levels (§0).

## 3. Blockers to resolve before step 1 succeeds

- **Path dependency.** `Cargo.toml` pins `globalmind-runtime = { path = "../globalmind-runtime" }`. That resolves to `PhyEngLastExam/globalmind-runtime`, which does not exist here. Options:
  - (a) copy `globalmind-runtime/` next to the harness (same relative layout), or
  - (b) point the path at the GitHub folder's copy.

  Option (a) keeps the folder self-contained. **Decision needed.** Option (a) is my default.
- **Missing `hypermind-mcp` binary.** Required for steps 4 and 5. It lives in `hypermind-nested-kv` in the GitHub folder. Build it there or copy the source in.
- **Offline cache.** Earlier runs used `--offline` because crates.io was unreachable from the sandbox. Confirm `~/.cargo` has the crates, or the build will fail.

## 4. Out of scope for the baseline

- Any real-model run (none exists; the doctrine requires one before citing a benchmark number).
- Any PLE dev-set run, or any PLE claim.
- Energy, throughput, or quality numbers. The grading doc labels them PROPOSED.
- Continuous/daemon wiring (proposed in N3 wiring doc §8, not built).
- Changing harness code. If a step fails, record the failure as the baseline result. Fix it in a separate pass.

## 5. Gap to PLE (planning only, not baseline)

Entering PLE requires building things the harness does not have yet:

1. A WebSocket server (`serve.py`) following `ple/docs/protocol.md`, driving task-container commands and file I/O.
2. A model client that reads `MODEL_BASE_URL` / `*_BASE_URL` and calls one `google-cloud/<id>` model.
3. An agent loop that produces `artifacts/` through `eval/run.py`.
4. A `submission.yaml` and a repo layout that passes `kit check`.

Items 1–4 are a separate scope. Start it only after the baseline receipt exists.

## 6. Done when

- Each step in section 2 has a recorded command, exit status, and pass/fail count.
- Blockers in section 3 are resolved or explicitly deferred.
- The receipt lists what is proven (code contracts, live N3 round-trip) and what is not (any real-model or PLE result).
