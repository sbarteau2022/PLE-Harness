# DeepSeek Harness PLE adapter (UNMET scaffold)

Status 2026-11-16: no DeepSeek campaign cell is runnable. All
`deepseek-harness` cells in `ablations/api/results.json` are `incomplete`
(UNMET). No paid DeepSeek call has been made.

Pinned source: `ablations/vendor/deepseek-harness`
(`39163dd30f703a79079e2d2446545cff3c453738`, MIT).

## What the adapter must do

1. **Provider profile.** Declare a pi-ai OpenAI-compatible gateway pointing at
   `https://integrate.api.nvidia.com/v1` with `MODEL_API_KEY`, one profile per
   seat (mistral-large-4, step-5-preview, nemotron-3-ultra-550b).
2. **Task driver.** A small headless driver (Node, run from the kit serve
   container) that takes the PLE instruction + relay URL, opens a DSH session
   with the pinned harness, and closes with a chosen deliverable.
3. **Tool bridge.** Expose the relay (`exec`, `read_file`, `write_file`,
   `mcp/...`) as DSH tools, reusing `ablations/bridge/ple_tool.py` semantics.
4. **Telemetry marker.** Print `PLE-ABLATION-RUN {...}` JSON (see
   `ablations/normalize.py`) with live turn IDs, tool counts, and timing so
   host-side normalization can build receipts.
5. **Entry + launcher.** `scripts/prepare_deepseek_entry.sh` building
   `ablations/entries/deepseek-<seat>/`, wired into `scripts/run_ablation.sh`.

## Gates before any paid call

- Keyless `kit check --no-smoke` passes (submission, files, build, model, listen).
- MCP parity test passes against a task with MCP servers.
- One no-MCP pilot closes; receipt validates; mint readback verifies.
