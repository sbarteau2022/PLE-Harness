#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SEAT="${1:-mistral}"
case "$SEAT" in
  mistral) MODEL_ID="mistralai/mistral-large-4-1t-a52b" ;;
  step5) MODEL_ID="stepfun-ai/step-5-preview" ;;
  nemotron-ultra) MODEL_ID="nvidia/nemotron-3-ultra-550b-a55b" ;;
  *) echo "unknown seat: $SEAT" >&2; exit 2 ;;
esac
ENTRY="$HERE/ablations/entries/generic-$SEAT"
STAGE="$(mktemp -d "$HERE/ablations/.entry-generic.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/ablations/generic"
sed "s|openai-compatible/mistralai/mistral-large-4-1t-a52b|openai-compatible/$MODEL_ID|" \
  "$HERE/submission.generic.yaml" > "$STAGE/submission.yaml"
cp -p "$HERE/serve_generic.py" "$HERE/taskbridge.py" "$HERE/requirements.txt" "$STAGE/"
cp -p "$HERE/ablations/__init__.py" "$STAGE/ablations/"
cp -p "$HERE/ablations/generic/__init__.py" "$HERE/ablations/generic/agent.py" "$STAGE/ablations/generic/"
mkdir -p "$ENTRY"; rsync -a --delete "$STAGE/" "$ENTRY/"
git -C "$ENTRY" init -q; git -C "$ENTRY" add -A
git -C "$ENTRY" -c user.name='PLE Ablation Builder' -c user.email='local@invalid' \
  commit -q --no-gpg-sign -m 'generated generic baseline entry'
echo "$ENTRY"
