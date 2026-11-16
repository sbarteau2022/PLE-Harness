#!/usr/bin/env bash
# Prints "score=<mean reward>" for a dev-set subset (official graders; needs Docker). A frontier-loop gate.
#   scripts/score.sh                 optimisation set  (FRONTIER_TASKS)
#   scripts/score.sh --holdout       held-out set      (FRONTIER_HOLDOUT_TASKS), for a "holdout": true gate
#   scripts/score.sh task,task,...   explicit tasks
# It also writes failure evidence for the loop: the agent's own step log for the lowest-scoring
# task goes to frontier-loop/evidence/score.txt (never grader files). Use it in a gate with
#   "evidence": "frontier-loop/evidence/*.txt"
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
case "${1:-}" in
  --holdout) TASKS="${FRONTIER_HOLDOUT_TASKS:-seismic-diaphragm-shear,lv-cable-sizing}"; NAME=holdout ;;
  "")        TASKS="${FRONTIER_TASKS:-wind-diaphragm-shear,cooling-water-flush-plan}"; NAME=score ;;
  *)         TASKS="$1"; NAME=score ;;
esac
LOG="$(mktemp -t ple-score)"
"$HERE/scripts/run_local.sh" dev "$TASKS" > "$LOG" 2>&1 || true
RESULTS="$(grep -o 'results: .*results.json' "$LOG" | tail -1 | cut -d' ' -f2)"
[ -f "$RESULTS" ] || { tail -20 "$LOG"; echo "score=0"; exit 1; }
mkdir -p "$HERE/frontier-loop/evidence"
python3 - "$RESULTS" "$HERE/frontier-loop/evidence/$NAME.txt" "$NAME" <<'PY'
import json, pathlib, sys
results, out, name = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
data = json.loads(results.read_text())
for task in data["tasks"]:
    print(task["id"], task["score"], task["status"])
print("score=%.4f" % data["score"])
if name == "holdout":          # holdout evidence is never written: builders must not learn from it
    out.unlink(missing_ok=True)
    sys.exit(0)
worst = min(data["tasks"], key=lambda t: t["score"])
lines = [f"Lowest-scoring task: {worst['id']} score={worst['score']} status={worst['status']} "
         f"seconds={worst.get('seconds')} tokens={worst.get('tokens')}", "Agent step log (tail):"]
trial = worst.get("trial")
logs = sorted((results.parent / trial / "agent").glob("*.txt")) if trial else []
lines.append(logs[0].read_text(errors="replace")[-6000:] if logs else "(no agent log was captured)")
out.write_text("\n".join(lines))
PY
