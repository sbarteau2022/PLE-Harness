#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${PLE_ROOT:-/Users/stewartbarteau/Desktop/PhyEngLastExam}"
SEAT="${MODEL_SEAT:-mistral}"; ACTION="${1:-check}"; TASK="${2:-}"
ENTRY="$("$HERE/scripts/prepare_generic_entry.sh" "$SEAT")"
KEY="${NVIDIA_API_KEY:-}"; [ -n "$KEY" ] || KEY="$(launchctl getenv NVIDIA_API_KEY 2>/dev/null || true)"
[ -n "$KEY" ] || KEY="$(security find-internet-password -s https://integrate.api.nvidia.com/v1 -w 2>/dev/null || true)"
: "${KEY:?NVIDIA_API_KEY is absent}"
ENV_FILE="$(mktemp -t ple-generic-baseline-env)"; trap 'rm -f "$ENV_FILE"' EXIT; chmod 600 "$ENV_FILE"
printf 'MODEL_API_KEY=%s\nJUDGE_BASE_URL=https://integrate.api.nvidia.com/v1\nJUDGE_API_KEY=%s\n' "$KEY" "$KEY" > "$ENV_FILE"
case "$ACTION" in
  check) (cd "$ROOT/ple" && uv run python -m kit check "$ENTRY" --env-file "$ENV_FILE" --no-smoke --work "$HERE/ablations/work/generic-$SEAT") ;;
  run)
    [ -n "$TASK" ] || { echo "usage: $0 run TASK" >&2; exit 2; }
    OUT="$HERE/ablations/runs/generic/$SEAT/$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
    (cd "$ROOT/ple" && caffeinate -dimsu uv run python eval/run.py "$ENTRY" --env-file "$ENV_FILE" \
      --tasks-dir "$ROOT/ple-dev-set/tasks" --tasks "$TASK" --trials 1 --attempts 1 --no-export --out "$OUT") ;;
  *) echo "action must be check or run" >&2; exit 2 ;;
esac
