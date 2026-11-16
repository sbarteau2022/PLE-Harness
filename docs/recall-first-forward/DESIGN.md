# Recall-first forward policy and attention economics

Date: 2026-11-16. Status: DESIGN, not implemented. Owner: PLE harness.

## Question and decision

[PROPOSED] Can a bounded scout inference select verified SurrealDB atoms and
trajectories before the main prefill, improving task quality or reducing total
attention work after the scout and retrieval costs are included? The result
decides whether PLE enables recall-first dispatch by default and what claims
the attention-economics paper may make.

## Current evidence boundary

[VERIFIED] `frontier/local_chat.py` prepares the full chat before it selects an
exact native prefix; `frontier/checkpoints.py` handles exact-prefix discovery.
`frontier/TPPE-LOCAL-FORWARD-DESIGN.md` permits only prefix-identical K/V reuse
without new model qualification. There is no production local adapter proving
forked KV state, scout continuation, or selective in-forward page attention.

[VERIFIED] `frontier/surreal.py` exposes bounded atom and trajectory readers in
the existing nested Surreal authority. SQLite/N3 is short-term conversation
memory, not tensor storage. Hosted APIs cannot load external K/V pages. The
existing PLE 27-turn live probe measures tool-loop behavior only, not this
policy or engineering quality (`ablations/probes/results.json`).

## Policy boundary

[PROPOSED] A model must execute a small forward pass to choose recall. “Before
main prefill” means after this scout, never before all inference. The policy
has a single typed orchestration boundary in PLE, selected by capability rather
than model name or layer count:

1. Encode a bounded task-only prefix with the runtime's tokenizer and template.
2. Run a bounded scout from that prefix. It emits `RecallPlanV1`, a structured
   set of 0–4 queries, optional atom/trajectory hints, and a `none_needed`
   decision. The scout may not issue shell, network, or database statements.
3. Retrieve candidates through PLE's existing `Mind` using bounded exact
   readers and search. Admit by workspace, active status, provenance, taint,
   digest, relevance threshold, deduplication, and token budget. Record rejected
   IDs and reasons. If retrieval fails, continue with the task-only prefix and
   record the failure; an identity or digest mismatch fails closed.
4. Assemble the main prompt from the task prefix and admitted, cited text
   atoms/trajectory excerpts. The main runtime verifies its own identity and
   tokenization. Main generation and tool behavior stay at the current PLE
   agent boundary.

[PROPOSED] Local runtimes may add `fork_prefix` and `continue_branch` to a
capability interface. The scout branch is discarded after planning; the main
branch starts from the clean task-prefix state. A branch receipt binds model
weights, tokenizer, adapter, position contract, causal token IDs, and every
declared cache/state buffer. The runtime owns layer count and buffer shapes.
KV and named recurrent state must be copied or isolated atomically. If the
runtime cannot prove this, it re-prefills the main prompt; it must not pretend
to reuse a cache. Exact persisted prefix pages remain in nested SurrealDB.

[PROPOSED] Hosted endpoints use two ordinary calls: scout then main. Only text
atoms enter the main prompt; any provider prompt caching is measured from the
provider receipt. The same `RecallPlanV1` and admission rules apply. No tensor
bytes enter SQLite or an API prompt.

## Attention economics boundary

[PROPOSED] The first implementation changes the number of prompt tokens and
the placement of recall. It does not make dense attention sparse. A separate
local runtime experiment may select K/V pages plus a recent window through
`runtime/core_lm/src/attention_backend.rs` or another runtime-owned equivalent.
This requires a query-dependent page index, measured selection cost, complete
state handling for hybrid models, and accuracy comparison with exact dense
attention. TPPE key relocation alone is insufficient for cross-prefix content
reuse. It is not silently enabled by recall-first policy.

[PROPOSED] For every call, account for task prefill, scout decode, retrieval,
admission, branch/fork or re-prefill, main prefill, decode, tool latency, and
all extra calls. Record input, output, cached and uncached tokens; context
length; GPU/CPU time; peak memory; page bytes read; page hit rate; and provider
cost when reported. Distinguish useful-token reduction from actual attention
kernel work, wall time, money, and physical energy. The paper may claim only
quantities measured on the matched runs. A theoretical dense-attention
quadratic term is background, not a measured end-to-end complexity claim.

## Implementation packets

| Packet | Deliverable | Depends on | Deterministic gate |
|---|---|---|---|
| P0 | Freeze preregistration and source roster | none | Scope and criteria present |
| P1 | `RecallPlanV1` parser and bounded admission in `frontier/` | P0 | Malformed/foreign/tampered/oversize cases fail |
| P2 | Optional scout orchestration at `frontier/local_chat.py` and API path | P1 | Main prompt shows only admitted IDs; none-needed path works |
| P3 | Runtime-owned clean-branch capability with full state receipts | P2 | Fork vs clean re-prefill logits/token IDs match after restart |
| P4 | Phase and resource instrumentation, receipt writer | P2 | Totals reconcile; missing counters explicit null |
| P5 | Matched live A/B, independent judges | P3/P4 | Preregistered quality and economics gates |
| P6 | Attention-economics manuscript in `harness/reports/` | P5 | Every result cites immutable receipt; limitations present |

[PROPOSED] P3 is optional for an API-only result but mandatory for a claim of
local KV branch reuse. P6 cannot be promoted from synthetic tests or the prior
tool-loop probe.

## Risks and falsifiers

[PROPOSED] Scout calls can add latency/cost, retrieve misleading knowledge, or
break provider prefix caching. A fixed memory budget can omit necessary facts.
Forking incomplete hybrid state can change logits despite matching K/V. The
policy fails promotion if quality regresses beyond the preregistered margin,
total cost rises without a prespecified quality gain, provenance checks fail,
or any claimed exact branch disagrees with clean re-prefill. Do not tune gates
after seeing held-out outcomes; amendments must be dated and run as new arms.

## Source register for this design

| ID | URL or local origin | Version | License | Relied-on claim |
|---|---|---|---|---|
| L1 | `harness/frontier/TPPE-LOCAL-FORWARD-DESIGN.md` | workspace snapshot 2026-11-16 | local project | [VERIFIED] Exact causal prefix is the admitted native reuse boundary. |
| L2 | `harness/frontier/local_chat.py` | workspace snapshot 2026-11-16 | local project | [VERIFIED] Chat preparation and exact-prefix selection are current seams. |
| L3 | `harness/frontier/surreal.py` | workspace snapshot 2026-11-16 | local project | [VERIFIED] Atom and trajectory reads use the nested Surreal authority. |
| L4 | `harness/runtime/core_lm/src/attention_backend.rs` | workspace snapshot 2026-11-16 | local project | [VERIFIED] A runtime-owned attention backend interface exists. |

[PROPOSED] Before manuscript drafting, search the local Edge research library,
then open primary papers and add URL, version, license, and exact relied-on
claim to the paper's source register. Do not infer physical energy savings from
token counts or cite the local design as external empirical evidence.
