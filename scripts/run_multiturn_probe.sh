#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; HARNESS="${1:?generic or ple}"; SEAT="${2:?seat}"
case "$SEAT" in
  mistral) MODEL="mistralai/mistral-large-4-1t-a52b" ;;
  step5) MODEL="stepfun-ai/step-5-preview" ;;
  nemotron-ultra) MODEL="nvidia/nemotron-3-ultra-550b-a55b" ;;
  *) echo "unknown seat $SEAT" >&2; exit 2 ;;
esac
KEY="${NVIDIA_API_KEY:-}"; [ -n "$KEY" ] || KEY="$(launchctl getenv NVIDIA_API_KEY 2>/dev/null || true)"
[ -n "$KEY" ] || KEY="$(security find-internet-password -s https://integrate.api.nvidia.com/v1 -w 2>/dev/null || true)"
: "${KEY:?NVIDIA_API_KEY is absent}"
NVIDIA_API_KEY="$KEY" "$HERE/.venv/bin/python" "$HERE/scripts/run_multiturn_probe.py" \
  --harness "$HARNESS" --seat "$SEAT" --model "$MODEL"
