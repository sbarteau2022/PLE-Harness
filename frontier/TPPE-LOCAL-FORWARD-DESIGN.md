# Model-neutral tensor pages and optional TPPE

Status: PLE harness admission contract and position-math reference implemented;
model-specific runtime adapters and live injection remain open.

## Shared mind storage

Tensor state must remain in the existing nested SurrealDB substrate, under the
shared atom/address object. SQLite is exclusively short-term memory. Do not
introduce a parallel SQL tensor-bundle table or a separate tensor store.
The PLE admission contract is implemented. `Mind.tensor_exact` now reaches the
owning nested store through the Rust binary bridge and checks membership in
the selected recall. Complete-model cache assembly and forward admission are
implemented for supported explicit K/V rosters; production runtime qualification
remains open. Exact lookup complexity must be distinguished from the
unavoidable O(B) cost of reading and verifying B tensor bytes. General
hyperbolic neighbor search still lacks the requested indexed bound.

The owning PLE-local source is `../runtime/globalmind-runtime/src/store.rs`:
`create_tensor_session`, `put_tensor_page`, and
`fetch_tensor_payload_exact` in `kv_identity.rs`. These already bind tensors
to an atom address in the nested SurrealDB schema. The existing implementation
uses SurrealQL internally, with native SurrealDB byte fields; it is not a
SQLite tensor store. SurrealQL is permitted for this access; the user's storage
constraint applies to the separate SQLite session store. The parallel tensor
bundle candidate was removed without writes to the live database.

## Every driver seat

The PLE frontier driver accepts any OpenAI-compatible model. Shared atoms,
ordered trajectories, and SQLite archives are text/data tools available to
every seat. A hosted provider owns its attention runtime, so it cannot accept
our tensor pages through that API. The harness must keep that path fully
functional when no local tensor capability is present.

`kv_contract.py` defines a model-neutral opt-in interface for local runtimes.
An adapter reports exact model, tokenizer, adapter, shape, layout, dtype and
position-transform identities. Before invoking the adapter, the harness checks
the intended causal token prefix, a complete ordered layer/head K/V roster,
tensor byte lengths, and page digests. No model name, vendor, or wire format is
hard-coded in the gate. Every adapter must additionally validate its numerical
tensor format and independently attest the live model identity.

The first admissible use is exact continuation from a byte-identical prefix.
Changing the prefix changes deeper-layer hidden states, K, and V. Relocating
the positional rotation of a K vector cannot repair those hidden states.
Retrieved knowledge pages from unrelated prefixes therefore require a
separately trained architecture and quality qualification. Native RoPE also
supports algebraic key re-addressing, so TPPE relocation alone does not solve
cross-context reuse.

## Optional TPPE reference

`tppe_pages.py` accepts a runtime-supplied rotary pair count, radial gain,
multiscale base, phases and radius. It proves only that a projected key's
positional transform can be moved by a rotation difference and scale ratio.
Values are untouched. The two-angle v2 arm is the current experiment;
multiscale v3 is an unqualified alias-reduction candidate. Neither arm is
automatically applied to a driver seat or substituted for a model's trained
position transform.

## Local forward gate

`runtime/core_lm` provides one example: Glimmer exports
exact pages through `GPF2`/`GPO2` and imports them through `GCX2`, with a
component restart-parity test. That wire belongs in a Glimmer adapter, outside
the model-neutral gate. The server verifies shape and finite values; its import
frame does not itself attest the causal prefix, so an adapter must validate the
export receipt and exact model/prefix identity before sending it. No Glimmer
listener is active in this session. The PLE harness has no admitted local
adapter yet.

To qualify any adapter, compare uninterrupted and imported-prefix logits and
generated IDs for the same token tape, including restart, identity mismatch,
missing head, corruption, and a changed prefix. Benchmark matched prefill time
and memory. A separate real-model quality bakeoff is required before using a
TPPE arm in a trained model's forward pass.

## Complete cache assembly

The PLE assembler accepts a caller-pinned runtime identity, intended token
prefix, and the complete native page identity roster. It validates the entire
metadata roster before fetching bytes: immutable contracts within each session,
the same effective model digest, per-layer dtype and dimensions, ordered contiguous token
intervals starting at zero and ending at the prefix length for every head,
unique page IDs, and membership in selected recall. Native prefix digest must
equal SHA-256 of little-endian u32 token IDs. Unsupported layouts are rejected.
The current assembler accepts token-major unquantized f32/f16/bf16 pages.

Only after every page is retrieved and rehashed does it expose a complete
`ExactPageSet` to the runtime gate. It never injects partially fetched heads.
Native metadata does not independently attest the tokenizer or adapter: the
caller must obtain the pinned runtime identity from its trusted export/admission
receipt. A self-supplied roster is not proof of how a page was computed.
No session discovery or global table scan is needed for this explicit roster.
For P page fragments, lookup costs P exact reads plus O(B) payload work;
this is not an O(log N) total-forward claim.

## Driver-owned runtime registration

`drivers.resolve_local_runtime(seat)` loads only the runtime factory explicitly
named by that driver profile. `runtime_registry.py` has no built-in model,
layer count, or serving-wire default. The profile pins a model digest; the
factory receives the selected model and returns `LocalPageRuntime`. The wrapper
pins its reported tokenizer, adapter, tensor shape and positional identity as
well, and rechecks them at export and continuation boundaries.

The Glimmer-specific page-wire implementation was removed from the harness.
Its single-model experiment reports are historical component evidence only;
they do not qualify this model-agnostic interface or another model. Every
runtime plugin still requires real uninterrupted/imported-prefix qualification.
Generic boundary tests vary layer count, head count and head dimension.

## Architecture-neutral state boundary

The runtime owns the number of layers and the schema of every state buffer.
`TensorSpec` remains a uniform K/V shorthand; `LayeredTensorSpec` describes
attention layers with independently specified head counts, dimensions, dtypes,
layouts and position contracts. Nested K/V assembly must check every layer
against its descriptor. Heterogeneous layers use explicit native session IDs
per layer, retaining the existing nested session/page tables and raw bytes.
Session fields stay immutable within each session; shared model,
prefix, workspace and address contracts must agree across the full snapshot.
No session discovery or schema migration is implied by this read path.

`StateSpec` describes named tensors rather than requiring attention at every
layer. Fixed-size recurrent state, token-axis attention buffers (optionally
window-limited), auxiliary counters, and opaque packed state can be represented
with their own shape, encoding/layout and contract. `ExactStateSet` binds the
complete declared roster and bytes to the exact causal prefix. A plugin must
declare ALL state required for continuation and validate its own packing and
numerical semantics. These descriptors do not translate caches between models.

Driver registration and exact continuation accept either checkpoint type.
Unsupported state must fail before a database lookup or forward call. The
existing native Surreal schema stores rank-two K/V pages, so named recurrent
state persistence is explicitly unimplemented: do not encode it as fake K/V,
drop it, or spill it into SQLite. Extending that owning schema and qualifying
independent real runtimes remain required before universal local reuse claims.
Shape bounds are aggregate resource limits, not one model's layer count.
Identity checks at harness boundaries do not replace a runtime's atomic check
of the active model and checkpoint inside its forward/import operation.

## Agent chat dispatch

An optional `Config.local_runtime` carries the selected driver's trusted plugin
configuration into `agent.ModelClient`; absent that configuration, HTTP remains
the existing transport. The pipeline passes its existing `Mind` object to the
agent, so cache payloads and knowledge use the same authority and workspace.
There is no built-in chat template, tokenizer, decoder, model or server wire.

A chat-capable plugin adds `prepare_chat(messages, tools, mind) -> PreparedChat`,
`generate_tokens(prompt_ids, max_tokens) -> tuple[int, ...]`, and
`decode_chat(output_ids) -> dict` (an assistant message). Preparation owns the
model's tokenizer/template and may supply a `NestedPrefix` from its trusted
export receipt: exact runtime identity, causal prefix IDs, explicit native page
witnesses, and selected recall addresses. PLE requires a strict token prefix
of the prepared prompt, retrieves/admits ALL pages through the same `Mind`,
then invokes `continue_from_pages` with the remaining token suffix. A missing
candidate takes the cold local path; a malformed candidate is an error, never
a silent transport switch. The strict prefix leaves at least one token to
evaluate, because existing K/V checkpoints do not include next-token logits.

The output decoder must produce a valid assistant message; PLE wraps it in the
normal completion envelope and accounts for input, generated and reused tokens.
Plugin calls run outside the async event loop and are serialized per client.
All plugin stages recheck the pinned identity. Providers must still enforce
identity atomically inside their own runtime. Trustworthy runtime export,
named-state persistence and independent real-model qualification remain
separate qualification work; this dispatch does not fabricate
checkpoints or let a model write tensor payloads through text tools.

## Automatic native checkpoint publication and discovery

The coordinator is implemented in `checkpoints.py` and connected to local chat.
Its persistent restart test uses the actual binary bridge and copied production
tensor schema, with a synthetic runtime. Production adapters remain unqualified.

`PreparedChat` may carry a typed `NativeCacheContext`: a selected existing atom
address, geometry/address/phase contracts, and native position version/config
per layer (or one shared recipe). This opts the trusted runtime into automatic
publication through `export_prefix`. Named recurrent state remains rejected by
the current owning K/V schema; no alternate tensor table is introduced.

PLE probes power-of-two causal prefix lengths strictly below the prepared
prompt length, filtered by aggregate byte and fragment budgets. Prefix hashes
are accumulated in one token pass, so computing
all candidates is O(L) for L prompt tokens. Record IDs commit to workspace,
full runtime identity (including tokenizer/adapter), model label, prefix hash,
native context and fragmentation version. All metadata reads are exact record
lookups, never table-wide history scans. There are O(log L) candidate probes;
payload work remains O(B). This is not O(log N) total forward computation.

The longest complete matching candidate is admitted through the existing Mind
and Rust exact reader. After a successful completion, the runtime exports the
largest eligible anchor; PLE admits it, fragments it and publishes raw binary
pages through the owning native store. All metadata is checked before writes.
The production schema's unique workspace/model/request index requires distinct
request digests for distinct per-layer sessions. Uniform shorthand uses one
session; explicitly layered state uses independent session request identities.

Publication is append-only and idempotent for identical bytes. Existing
immutable session/page conflicts are refused rather than overwritten. A
partially uploaded checkpoint is a miss until every expected page exists;
the complete roster is always validated before any forward admission. A later
export can resume identical partial publication. No cache eviction is added.
The context's address must resolve to an active atom in the same workspace.
Publication does not claim a change to radial reuse policy or trained TPPE
quality, and production adapters still require independent forward qualification.
