# Goal: raise the PLE harness's dev-set score without breaking it

Target folder: `/Users/stewartbarteau/Desktop/PhyEngLastExam/harness` (agent loop in `agent.py`).

Improve how the agent works on engineering tasks: its system prompt, tool design, context
handling, time and step budgeting, and deliverable discipline. Success: the mean official
reward on the optimisation tasks reaches the target set in the preregistration, with the
offline tests still green, and no drop on the held-out tasks.

Constraints:
- Keep the serve protocol and `taskbridge.py` unchanged.
- No task-specific branching and no hard-coded answers. Never read task `tests/` or `artifacts/`.
- `tests/`, `scripts/` and `frontier/` are out of scope.

Gates that always apply (the preregistration adds the score target):

```gates
[{"name": "unit", "cmd": ".venv/bin/python -m pytest tests -q"},
 {"name": "wire", "cmd": ".venv/bin/python tests/protocol_smoke.py"}]
```

Suggested preregistration gates for the score (they use the official graders, so they need
Docker and take hours per run):

- `{"name": "dev-score", "cmd": "scripts/score.sh", "metric": "score=([0-9.]+)", "min": 0.30,
  "repeats": 2, "evidence": "frontier-loop/evidence/score.txt", "timeout_sec": 50000}`
- `{"name": "holdout-score", "cmd": "scripts/score.sh --holdout", "metric": "score=([0-9.]+)",
  "min": 0.0, "holdout": true, "timeout_sec": 50000}`

`FRONTIER_TASKS` and `FRONTIER_HOLDOUT_TASKS` choose the two task sets.
