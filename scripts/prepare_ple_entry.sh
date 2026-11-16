#!/usr/bin/env bash
# Build the minimal PLE kit source tree for the PLE-harness campaign cell.
# Usage: prepare_ple_entry.sh [mistral|step5|nemotron-ultra]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SEAT="${1:-mistral}"
case "$SEAT" in
  mistral) MODEL_ID="mistralai/mistral-large-4-1t-a52b" ;;
  step5) MODEL_ID="stepfun-ai/step-5-preview" ;;
  nemotron-ultra) MODEL_ID="nvidia/nemotron-3-ultra-550b-a55b" ;;
  *) echo "unknown seat: $SEAT" >&2; exit 2 ;;
esac
ENTRY="$HERE/ablations/entries/ple-$SEAT"
STAGE="$(mktemp -d "$HERE/ablations/.entry-ple.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/ablations" "$STAGE/frontier"
sed "s|openai-compatible/mistralai/mistral-large-4-1t-a52b|openai-compatible/$MODEL_ID|" \
  "$HERE/submission.yaml" > "$STAGE/submission.yaml"
cp -p "$HERE/serve.py" "$HERE/agent.py" "$HERE/taskbridge.py" "$HERE/requirements.txt" "$STAGE/"
cp -p "$HERE/ablations/__init__.py" "$HERE/ablations/normalize.py" "$HERE/ablations/receipts.py" \
  "$STAGE/ablations/"
cp -p "$HERE"/frontier/*.py "$STAGE/frontier/"

mkdir -p "$ENTRY"
rsync -a --delete "$STAGE/" "$ENTRY/"
git -C "$ENTRY" init -q
git -C "$ENTRY" add -A
git -C "$ENTRY" -c user.name='PLE Ablation Builder' -c user.email='local@invalid' \
  commit -q --no-gpg-sign -m 'generated pinned PLE entry'
echo "$ENTRY"
