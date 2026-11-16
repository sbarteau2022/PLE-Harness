#!/usr/bin/env bash
# Build the minimal PLE kit source tree for the pinned Harness2 ablation.
# Usage: prepare_harness2_entry.sh [mistral|step5|nemotron-ultra]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SEAT="${1:-mistral}"
case "$SEAT" in
  mistral) MODEL_ID="mistralai/mistral-large-4-1t-a52b" ;;
  step5) MODEL_ID="stepfun-ai/step-5-preview" ;;
  nemotron-ultra) MODEL_ID="nvidia/nemotron-3-ultra-550b-a55b" ;;
  *) echo "unknown seat: $SEAT" >&2; exit 2 ;;
esac
ENTRY="$HERE/ablations/entries/harness2-$SEAT"
STAGE="$(mktemp -d "$HERE/ablations/.entry-harness2.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/ablations/bridge" \
  "$STAGE/ablations/vendor/harness2/dependencies" \
  "$STAGE/ablations/vendor/harness2"
sed "s|openai-compatible/mistralai/mistral-large-4-1t-a52b|openai-compatible/$MODEL_ID|" \
  "$HERE/submission.harness2.yaml" > "$STAGE/submission.yaml"
cp -p "$HERE/serve_harness2.py" "$HERE/taskbridge.py" "$HERE/requirements.txt" "$STAGE/"
cp -p "$HERE/ablations/__init__.py" "$HERE/ablations/harness2_ple.py" \
  "$HERE/ablations/normalize.py" "$HERE/ablations/receipts.py" "$HERE/ablations/mint.py" \
  "$STAGE/ablations/"
cp -p "$HERE/ablations/bridge/__init__.py" "$HERE/ablations/bridge/ple_tool.py" "$STAGE/ablations/bridge/"
cp -p "$HERE/ablations/vendor/harness2/pyproject.toml" \
  "$HERE/ablations/vendor/harness2/LICENSE" "$STAGE/ablations/vendor/harness2/"
rsync -a --exclude '__pycache__' \
  "$HERE/ablations/vendor/harness2/harness2/" "$STAGE/ablations/vendor/harness2/harness2/"
rsync -a --exclude '.git' --exclude 'node_modules' --exclude '.turbo' \
  "$HERE/ablations/vendor/harness2/dependencies/opencode/" \
  "$STAGE/ablations/vendor/harness2/dependencies/opencode/"

mkdir -p "$ENTRY"
rsync -a --delete "$STAGE/" "$ENTRY/"
git -C "$ENTRY" init -q
git -C "$ENTRY" add -A
git -C "$ENTRY" -c user.name='PLE Ablation Builder' -c user.email='local@invalid' \
  commit -q --no-gpg-sign -m 'generated pinned Harness2 entry'
echo "$ENTRY"

