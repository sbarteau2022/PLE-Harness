# Google – Agent Harness on Physical-Engineering's Last Exam (PLE)

Competition page: https://www.kaggle.com/competitions/physical-engineerings-last-exam
Benchmark site: https://ple-bench.com · Paper: https://arxiv.org/abs/2610.30537 · Dataset: https://huggingface.co/datasets/google/physical-engineerings-last-exam

## At a glance

| Item | Detail |
|---|---|
| Host | Google (Featured Hackathon) |
| Tags | Engineering, Model Comparison, Agentic, NLP |
| Prize pool | $400,000 (Main Track): 1st $250,000 · 2nd $100,000 · 3rd $50,000 |
| Start | November 9, 2026 |
| Final submission deadline | January 5, 2027 |
| Stage 1 judging complete | January 19, 2027 |
| Stage 2 results / winners | January 26, 2027 |
| Deadlines | 11:59 PM UTC on the stated day |
| Participation (at capture) | 1,442 entrants · 23 teams · 23 submissions |
| Partner | Mercor (benchmark partner) |
| Judges | Yufan Zhuang, Rujun Han, Arijit Ray, Chen-Yu Lee (Google Research) |

## What the competition asks for

Build an **open-source autonomous agent harness**: a server that drives one of eight Google-sponsored models through real engineering tasks. The harness must solve tasks that produce project deliverables (calculation packages, design reviews, drawings, technical memos) in an isolated task container. Each task is graded 0–1 by automated checks, an LLM rubric judge, or both.

The stated goal is practical: take months of tedious engineering planning overhead off the plates of human engineers.

## The benchmark (PLE-Bench)

85 tasks across four engineering disciplines. The competition splits them into two sets.

| Discipline | Dev set (public, 29) | Test set (hidden, 56) |
|---|---|---|
| Structural | 4 | 6 |
| Civil | 9 | 9 |
| Electrical | 9 | 18 |
| Mechanical | 7 | 23 |

- **Dev set:** 29 tasks, graders included, published in the gated Hugging Face dataset. Run locally with `eval/run.py`. Entrants commit the resulting `artifacts/` with their entry.
- **Test set:** 56 hidden tasks used for Stage 2 scoring.
- Each task runs for up to **6 hours** in a container with **2 CPUs and 4 GB RAM**, with no network. The harness talks to the task container over a WebSocket JSON protocol: run shell commands, read/write files, call task-local MCP servers (documents, spreadsheets, email, CAD).

## Scoring

**Per task:** 0–1, the grader's reward divided by the task maximum.

- Constraint/design-objective tasks score each item as `s = 0.4·p + 0.6·G·O′`, where `p` = fraction of constraints met, `G` = 1 only if all hard (safety/feasibility) constraints hold, and `O′` = normalized design objective against an unoptimized baseline. Breaking a hard constraint caps the item at 0.4.
- Harness failures count against the task: crashes, protocol violations, the 6-hour limit, or a model `400` response. Deliverables already written before a failure are still graded. A harness that errors before the model produces any output is retried, up to three attempts in all.

**Dev score:** grade of the committed `artifacts/`.

**Stage 1 — writeup gate (100 points, judged):**

1. Insight & failure analysis — non-obvious lessons; real dev-set failures.
2. Engineering reasoning — a concrete dev-set trajectory compared against a licensed engineer.
3. Cross-model generalizability — the 29 dev tasks, at least two models (including any local open-weight or persona run).
4. Efficiency & scaling — dev score vs. compute (calls, tokens, runtime, or parallel attempts).
5. Clarity & visuals — written for an ML researcher without a civil, mechanical or electrical background.

Per-criterion point splits were not captured in extraction; the total is 100.

**Stage 2 — hidden test (top 100 finalists):**

- Each finalist runs all 56 hidden tasks, **3 independent times**, and is scored on the mean.
- Results are plotted as a **performance–cost Pareto frontier** (mean score vs. average cost).
- Prizes go to the top 3 by mean score among finalists on that frontier. The page also mentions a threshold of "at least 0.138"; exact wording was not captured, so verify before relying on it.
- Stage 1 is a qualification gate only. Ties break on Stage 1 writeup score, then earlier Kaggle submission time.
- Organizers publish all systems' scores and costs.

## Sponsored models (the only models an entry may call)

| Identifier | Model | Family |
|---|---|---|
| `google-cloud/gemini-4-argon` | Gemini 4 Argon | Gemini |
| `google-cloud/gemini-3.8-flash` | Gemini 3.8 Flash | Gemini |
| `google-cloud/gemini-3.1-pro-preview` | Gemini 3.1 Pro (Preview) | Gemini |
| `google-cloud/gemma-4-26b` | Gemma 4 26B A4B | Gemma (open-weight) |
| `google-cloud/claude-opus-5-5` | Claude Opus 5.5 | Claude |
| `google-cloud/gpt-oss-120b` | gpt-oss 120B | Open-weight |
| `google-cloud/glm-5.2` | GLM 5.2 | GLM (open-weight) |
| `google-cloud/llama-4-maverick` | Llama 4 Maverick 17B 128E | Llama (open-weight) |

The harness reads the model URL and key from environment variables (`MODEL_BASE_URL`, `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GOOGLE_GEMINI_BASE_URL`). Entries contain no API keys. Local runs use `kit/model_proxy`. Judge tasks default to Claude Opus 5.5.

## Submission requirements

A valid entry has two parts.

1. **Kaggle Writeup** (≤ 2,000 words). Covers all five rubric criteria, a Media Gallery cover image, and three required figures (including an architecture figure). Must link the GitHub repo and a 40-character commit SHA under **Project Links**. Writeups can be un-submitted, edited and resubmitted until the deadline.
2. **GitHub repository** at that commit. The root must contain:
   - `submission.yaml` (sets `model_api` to exactly one `google-cloud/<id>`)
   - `serve.py` and `harness.py` (the WebSocket server entry points)
   - a license file (Apache-2.0 or MIT)
   - `artifacts/` exported by `eval/run.py` from this harness and model
   - the directory must pass `uv run python -m kit check` and `uv run python -m kit artifacts`. Limits: 512 MiB and 10,000 files, unpacked.

The GitHub App `ple-submission` must be installed on the repo. Direct registration on ple-bench.com is closed.

Starter entries: `google-research/ple` (`sample_submissions/pi-opus`, `hermes-opus`, `agy-gemini`) and `physical-engineerings-last-exam/submission-template` (Pi coding agent on Claude Opus 5.5).

## Rules that constrain design

- Do not hard-code task answers or branch on specific tasks, dev tasks included.
- Never read grader scripts, reference files, or your own `artifacts/` at build or run time.
- `artifacts/` must be produced by `eval/run.py` from the submitted harness and declared model. No hand-edited or other-harness artifacts.
- No web search (including provider search tools) and no external APIs. Only the declared model endpoint is reachable through the egress proxy.
- Do not bypass the proxy or tamper with evaluation containers.

## What is in the git dev set (`google-research/ple`)

Cloned to `PhyEngLastExam/ple/`. The repo holds the kit and evaluator, not the task data.

- `kit/`: container contract, `model_proxy`, local `check` and `artifacts` commands
- `eval/`: `run.py` (dev-set runner and offline regrader), `dev_set.yaml` (29-task manifest)
- `docs/`: `rules.md`, `protocol.md` (WebSocket protocol), `artifacts.md`, `models.md`, `local-models.md`
- `sample_submissions/`: starter harnesses
- Runtime: Docker, `uv`, Harbor 0.20.0 (`uv tool install harbor==0.20.0 --with websockets`), and `HF_TOKEN` for the gated dataset

## Research harness in this folder

`research-grade-harness/` is a copy of `MetaSuperIntelligenceLabs-GlobalHack-A-thon-SoloDevTrack/research-grade-harness` (excluding `target/`, 977 MB of build output, and `.DS_Store`). It is a Rust crate, `rgah` v0.1.0, described as a plan-driven, judge-gated, receipt-based agentic harness. It is **not yet a PLE entry**: it has no WebSocket server, no PLE protocol adapter, and no `submission.yaml`. Details are in [BASELINE-SCOPE.md](BASELINE-SCOPE.md).

## Key dates to watch

- Nov 9, 2026: start
- Jan 5, 2027: final deadline (writeup and GitHub commit)
- Jan 19, 2027: Stage 1 judging complete
- Jan 26, 2027: Stage 2 results and winners
