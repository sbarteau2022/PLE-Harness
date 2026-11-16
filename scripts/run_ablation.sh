#!/usr/bin/env bash
# Live cross-harness campaign launcher. Secrets exist only in a mode-600 temp file.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${PLE_ROOT:-/Users/stewartbarteau/Desktop/PhyEngLastExam}"
KIT="$ROOT/ple"
TASKS="$ROOT/ple-dev-set/tasks"

if [ -z "${NVIDIA_API_KEY:-}" ]; then
  NVIDIA_API_KEY="$(launchctl getenv NVIDIA_API_KEY 2>/dev/null || true)"
fi
if [ -z "${NVIDIA_API_KEY:-}" ]; then
  NVIDIA_API_KEY="$(security find-internet-password -s https://integrate.api.nvidia.com/v1 -w 2>/dev/null || true)"
fi
: "${NVIDIA_API_KEY:?NVIDIA_API_KEY is absent from environment, launchd, and keychain}"

ENV_FILE="$(mktemp -t ple-ablation-env)"
trap 'rm -f "$ENV_FILE"' EXIT
chmod 600 "$ENV_FILE"
{
  echo "MODEL_API_KEY=$NVIDIA_API_KEY"
  echo "JUDGE_BASE_URL=https://integrate.api.nvidia.com/v1"
  echo "JUDGE_API_KEY=$NVIDIA_API_KEY"
  echo "JUDGE_MODEL=${JUDGE_MODEL:-mistralai/mistral-large-4-1t-a52b}"
} > "$ENV_FILE"

HARNESS="${1:-generic}"
ACTION="${2:-check}"
MODEL="${MODEL:-mistral}"
case "$MODEL" in
  mistral) MODEL_ID="mistralai/mistral-large-4-1t-a52b" ;;
  step5) MODEL_ID="stepfun-ai/step-5-preview" ;;
  nemotron-ultra) MODEL_ID="nvidia/nemotron-3-ultra-550b-a55b" ;;
  *) echo "unsupported campaign seat: $MODEL" >&2; exit 2 ;;
esac
case "$HARNESS" in
  generic) SERVE="serve_generic.py"; PREPARE="prepare_generic_entry.sh" ;;
  harness2) SERVE="serve_harness2.py"; PREPARE="prepare_harness2_entry.sh" ;;
  ple) SERVE="serve.py"; PREPARE="prepare_ple_entry.sh" ;;
  *) echo "unsupported campaign harness: $HARNESS" >&2; exit 2 ;;
esac

serve_local() {
  # Light path: the harness serves from this Mac (no 5 GB serve container);
  # only the small task/verifier environments stay in colima. Same task
  # packet, tools, budgets, and judges as the docker path.
  TASK="${1:?usage: run_ablation.sh HARNESS run-local TASK}"
  OUT="$HERE/ablations/runs/$HARNESS/$MODEL/$(date +%Y%m%d-%H%M%S)-local"
  SERVER_LOG="$HERE/ablations/work/serve-$HARNESS-$MODEL.log"
  mkdir -p "$HERE/ablations/work"
  PORT="${PORT:-8765}"
  MODEL_NAME="$MODEL_ID" MODEL_BASE_URL="https://integrate.api.nvidia.com/v1" \
    MODEL_API_KEY="$NVIDIA_API_KEY" MAX_OUTPUT_TOKENS="16000" PORT="$PORT" \
    PATH="$HERE/ablations/bin:$PATH" \
    "$HERE/.venv/bin/python" "$HERE/$SERVE" > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  trap 'kill $SERVER_PID 2>/dev/null; rm -f "$ENV_FILE"' EXIT
  for _ in $(seq 1 60); do
    if PORT="$PORT" "$HERE/.venv/bin/python" -c \
      "import os,socket;socket.create_connection(('127.0.0.1',int(os.environ['PORT'])),2).close()" 2>/dev/null; then
      break
    fi
    sleep 2
  done
  cd "$KIT"
  caffeinate -dimsu uv run python eval/run.py --url "ws://127.0.0.1:$PORT" \
    --name "ple-$HARNESS-ablation" --model "openai-compatible/$MODEL_ID" \
    --env-file "$ENV_FILE" --tasks-dir "$TASKS" --tasks "$TASK" \
    --trials 1 --attempts 1 --no-export --out "$OUT"
  kill $SERVER_PID 2>/dev/null || true
  cp "$SERVER_LOG" "$OUT/server.log" 2>/dev/null || true
  trap 'rm -f "$ENV_FILE"' EXIT
}

case "$ACTION" in
  check)
    ENTRY="$("$HERE/scripts/$PREPARE" "$MODEL")"
    cd "$KIT"
    uv run python -m kit check "$ENTRY" --env-file "$ENV_FILE" --no-smoke --work "$HERE/ablations/work/$HARNESS-$MODEL"
    ;;
  run)
    ENTRY="$("$HERE/scripts/$PREPARE" "$MODEL")"
    TASK="${3:?usage: run_ablation.sh HARNESS run TASK}"
    OUT="$HERE/ablations/runs/$HARNESS/$MODEL/$(date +%Y%m%d-%H%M%S)"
    cd "$KIT"
    caffeinate -dimsu uv run python eval/run.py "$ENTRY" --env-file "$ENV_FILE" \
      --tasks-dir "$TASKS" --tasks "$TASK" --trials 1 --attempts 1 --no-export --out "$OUT"
    ;;
  run-local)
    serve_local "${3:?usage: run_ablation.sh HARNESS run-local TASK}"
    ;;
  *) echo "action must be check, run, or run-local" >&2; exit 2 ;;
esac

