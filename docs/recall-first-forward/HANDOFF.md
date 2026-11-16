# Recall-first forward implementation handoff

Date: 2026-11-16. Status: design and preregistration complete; implementation
and paper evidence open.

Read `DESIGN.md` and `PREREGISTRATION.md`, then
`FEP-INTEGRATION.md`, then
`../../frontier/TPPE-LOCAL-FORWARD-DESIGN.md`, `../../frontier/local_chat.py`,
`../../frontier/kv_contract.py`, `../../frontier/surreal.py`, and
`../../runtime/core_lm/src/attention_backend.rs`.

1. Freeze the first task roster and snapshot digests before coding or live
   evaluation. Record a baseline build/test receipt.
2. Implement typed `RecallPlanV1`, a bounded scout controller, and Surreal
   admission in PLE `frontier/`. Keep atom IDs, rejects, phase times, and
   prompt token IDs in a reproducible per-run receipt.
3. Wire both local and hosted dispatch, with a task-only fallback. No model
   name or fixed layer count in the policy. Update `README.md` with flags and
   limitations for every implementation.
4. Qualify clean-branch runtime capability on at least two independent local
   architectures before claiming model-agnostic local branch support. Test
   complete attention/recurrent state and restart parity. Keep all tensor bytes
   in nested SurrealDB; SQLite remains session memory only.
5. Instrument every phase, run adversarial gates, then execute paired A0/A1
   and eligible A2/A3 arms. Preserve failures and failed hypotheses.
6. Draft the attention-economics paper only from admitted live receipts and
   opened primary-source literature. Search the local Edge library first;
   record URL/version/license/claim for every external source. Regenerate all
   plots and tables from immutable receipts.

Do not call the earlier KV relocation algebra, synthetic restart test, or
27-call tool probe proof of selective attention or economic benefit. The
current handoff has no live policy result and no paper conclusion.
