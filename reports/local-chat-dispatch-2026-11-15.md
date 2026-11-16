# PLE local chat dispatch verification — 2026-11-15

The previous state-contract change left the agent on HTTP for every request.
This change connects optional driver-owned local codecs and nested cache
admission to the actual agent call site. Implementations are in the PLE
harness; the Meta Hackathon source and production Surreal database were not
modified.

## Connected path

`drivers.resolve` copies the selected profile's optional `local_runtime` into
`agent.Config`. `ModelClient` uses that configured runtime rather than creating
an HTTP client. `pipeline.run_agent` passes its existing `Mind` into `agent.run`.
The runtime supplies its tokenizer/chat template via `prepare_chat`, cold
generation via `generate_tokens`, and assistant-message decoding via
`decode_chat`. PLE retains native tools and its existing text tool protocol.

If preparation supplies a typed `NestedPrefix`, PLE validates the pinned
runtime, strict causal prefix and explicit native roster, then calls that same
`Mind.continue_tensor_cache`. Payloads pass through the existing Rust bridge
and nested Surreal K/V tables before the admitted suffix reaches the runtime.
Without a prefix candidate, local cold generation is used. Invalid candidates
are errors. No model-specific codec or implicit model fallback is bundled.

The local work runs in a worker thread with a per-client lock. Binding the
shared mind is also serialized. Calls check the selected runtime identity
before and after plugin stages. Plugins must additionally enforce atomic model
identity within their own forward operation.

Usage includes total prompt tokens, generated tokens and the admitted cached
prefix count. The frontier persists the latter as `budget.tokens_cached`;
`tokens_in` continues to include the entire prompt. These are accounting
observations, not measured latency, provider billing, or heat savings.

## Verification and adversarial review

- Full Python suite with the native bridge enabled: **167 passed**. The
  disposable Surreal integration ran rather than skipping.
- PLE protocol smoke: **passed**. Rust bridge: **2 tests passed** offline.
- Driver -> agent -> codec -> same `Mind` -> Rust -> disposable Surreal ->
  admitted local callback -> native `finish` completed end to end.
- A separate two-turn synthetic run took a cold path, read a real tool result,
  then finished through the admitted warm path. Native and text tool protocols
  were exercised. The pipeline test checks shared-object identity and persisted
  cached-token accounting.
- Mismatched runtime identity, changed/full prefixes, missing addresses/heads,
  missing bridge/mind, malformed preparation, invalid decoder messages,
  invalid/oversized output token sequences and incomplete plugin capabilities
  fail locally. Tests prohibit HTTP fallback.
- Identity changes inside each codec/generation stage are rejected; concurrent
  requests on one client do not interleave plugin stages.

## Remaining work against the full goal

No production model plugin is qualified. The codecs and generation callbacks
in these tests are synthetic and do not prove trained-model logits, restart
parity, or speed. Automatic trusted checkpoint publication and indexed prefix
discovery remain unimplemented; plugins must currently supply explicit native
receipts. Named recurrent/packed-state persistence remains unsupported by the
owning K/V schema and is rejected before lookup.

This wiring does not establish indexed hyperbolic neighbor bounds, the
probability/radial attention policy, FEP/energy measurements, or TPPE quality in
a trained model. It also cannot alter a hosted provider's internal attention;
the shared text/trajectory and prompt-archiving path remains that transport's
available mechanism. The overall goal remains active.
