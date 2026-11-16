#!/usr/bin/env bash
# Local validation against NVIDIA's hosted API. Never prints the key.
#   scripts/run_local.sh check                 # kit check (build, model, smoke, MCP)
#   scripts/run_local.sh dev [task,task,...]   # dev-set run (all tasks if none given)
# Key: NVIDIA_API_KEY from the environment, harness/.env (gitignored), or the macOS keychain.
# Optional: MODEL (default mistralai/mistral-large-4-1t-a52b), JUDGE_MODEL (default same), TRIALS.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${PLE_ROOT:-/Users/stewartbarteau/Desktop/PhyEngLastExam}"
KIT="$ROOT/ple"
TASKS_DIR="$ROOT/ple-dev-set/tasks"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }
# Fallback: the NVIDIA key saved in the macOS keychain (Zed stores it under the API URL).
if [ -z "${NVIDIA_API_KEY:-}" ]; then
  NVIDIA_API_KEY="$(security find-internet-password -s https://integrate.api.nvidia.com/v1 -w 2>/dev/null || true)"
fi
: "${NVIDIA_API_KEY:?set NVIDIA_API_KEY in the environment or $HERE/.env, or save it in the keychain}"
BASE="https://integrate.api.nvidia.com/v1"
MODEL="${MODEL:-mistralai/mistral-large-4-1t-a52b}"

# Run-time env file lives outside the entry so the key never enters the bundle.
ENVF="$(mktemp -t ple-env)"; trap 'rm -f "$ENVF"' EXIT
chmod 600 "$ENVF"
{
  echo "MODEL_API_KEY=$NVIDIA_API_KEY"
  echo "JUDGE_BASE_URL=${JUDGE_BASE_URL:-$BASE}"
  echo "JUDGE_API_KEY=${JUDGE_API_KEY:-$NVIDIA_API_KEY}"
  echo "JUDGE_MODEL=${JUDGE_MODEL:-$MODEL}"
} > "$ENVF"

# Keep submission.yaml's model_api in step with MODEL.
sed -i '' "s#^model_api: .*#model_api: openai-compatible/$MODEL#" "$HERE/submission.yaml"

# Task and sidecar images run as linux/amd64; on Apple Silicon build them that way too.
export DOCKER_DEFAULT_PLATFORM="${DOCKER_DEFAULT_PLATFORM:-linux/amd64}"

cd "$KIT"
case "${1:-check}" in
  check) uv run python -m kit check "$HERE" --env-file "$ENVF" ;;
  dev)
    args=("$HERE" --env-file "$ENVF" --tasks-dir "$TASKS_DIR" --no-export --out "$ROOT/runs/ple-harness-$(date +%Y%m%d-%H%M%S)")
    [ -n "${2:-}" ] && args+=(--tasks "$2")
    [ -n "${TRIALS:-}" ] && args+=(--trials "$TRIALS")
    # caffeinate keeps the Mac (and the Colima VM) awake for the whole run.
    caffeinate -dimsu uv run python eval/run.py "${args[@]}" ;;
  *) echo "usage: $0 check|dev [tasks]" >&2; exit 2 ;;
esac
