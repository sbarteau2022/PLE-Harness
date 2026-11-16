# Active inference, FEP, and forward-pass requirements

Date: 2026-11-16. Status: design audit, no runtime qualification.

## Current implementation finding

The PLE harness contains `frontier/local_chat.py`, Surreal recall, a runtime
attention seam, and exact-prefix KV contracts. Its source tree does not contain
an executable active-inference or Free Energy Principle (FEP) controller. The
GlobalMind README uses active-inference terminology, but that is not a PLE
forward-pass implementation. `frontier/HYPERBOLIC-BOUNDS.md` explicitly leaves
variational free energy and physical energy claims unverified. No FEP-derived
selection, posterior, or precision measurement is admitted as current evidence.

## Missing contract

1. **State and observations.** Define typed latent state, observation, action,
   preference, and belief-update boundaries. Identify what is actually
   observable before prefill: task bytes, bounded metadata, Surreal candidate
   summaries, and runtime capabilities. Do not label a prompt or KV tensor a
   posterior without a specified generative model and inference procedure.
2. **Policy set.** Enumerate finite admissible actions: skip recall, retrieve
   atom IDs, retrieve trajectories, use clean prefix branch, or re-prefill.
   Sparse-page selection is a separate runtime action after qualification.
3. **Objective.** Specify a calibrated expected-free-energy or decision-cost
   equation, with units, sign, horizon, posterior, prior preferences, epistemic
   value, task risk, and compute/latency price. A heuristic score must be named
   a heuristic. The controller must show each term in a receipt.
4. **Belief update.** Define how failed searches, conflicting atoms, tool
   results, and judge feedback update beliefs. Version updates and prevent
   held-out evaluation labels from entering the online policy.
5. **Precision and uncertainty.** Calibrate uncertainty against held-out
   outcomes per model/runtime; separate retrieval confidence from model
   confidence. Cap precision to avoid a single noisy atom dominating action.
6. **Safety and termination.** Enforce workspace, provenance, taint, digest,
   and tool authority before policy optimization. Bound scout calls, retrieval,
   retries, wall time, and resource use; permit a `none_needed` action. Stop
   on no expected benefit or repeated unchanged observations.
7. **Runtime capability.** Discover tokenizer, template, position scheme,
   architecture, layers, KV layout, and non-KV state from the runtime. Exact
   clean-prefix branching requires isolated complete state and parity against
   re-prefill. A model without that capability still uses the text path.
8. **Memory authority.** Keep atom and tensor pages in nested SurrealDB with
   immutable IDs, identity/digest bindings, snapshot receipts, and restart
   readback. Session SQLite may hold conversation text and references only.
9. **Attention mechanism.** Recall before main prefill changes prompt
   selection; it does not alter dense attention. Selective page attention
   requires a runtime hook, page index, position semantics, causal masking,
   complete hybrid state, selection overhead, and dense-equivalence quality
   gates. Reusable cross-prefix KV is unproved by TPPE relocation alone.
10. **Accounting.** Measure scout, Surreal read, policy computation, fork or
    re-prefill, main prefill, decode, and tool phases. Record tokens, key reads,
    wall time, memory, cost, and physical energy only with a calibrated meter.

## Implications for the forward pass

- A recall decision always follows at least one bounded model or deterministic
  controller computation. It cannot precede all inference.
- The scout's speculative cache must never contaminate the main branch. If
  isolation or identity fails, re-prefill and record the fallback.
- The policy may save main-prompt tokens while increasing total tokens and
  latency. The A/B comparison must count every scout and retry.
- FEP terminology does not itself yield an attention kernel, complexity
  improvement, or physical energy reduction. Those need separate proofs and
  measurements.
- A policy trained or tuned on the same task roster can overfit. Freeze its
  version and calibration set before the held-out ablation.

## Required qualification gates

Add an F0 arm with the same recall candidate surface as A1 but a deterministic
cost-only controller; add F1 with the specified active-inference objective.
Freeze both policy versions, priors, calibrations, and task roster before live
evaluation. Compare F1 to F0 and A0/A1 per model on quality, full-run resource
cost, calibration, unsafe-atom rejection, loop termination, and no-recall
controls. Missing objective terms, posterior update, or calibration leave F1
unqualified; they cannot be paper evidence. A sparse attention arm remains
separate from F0/F1.
