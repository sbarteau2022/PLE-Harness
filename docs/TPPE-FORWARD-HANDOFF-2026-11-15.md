# TPPE and local forward pass: handoff (2026-11-15)

Status: findings measured, one candidate module added, nothing qualified or promoted.
`src/tppe.rs`, the Layer-3 trace producer and the `tppe-layer3-candidate-v2` contract are
unchanged (tppe.rs SHA-256 still `2af606fe…c00af298`).

## What was added

`src/tppe_page.rs` (registered as `pub mod tppe_page` in `src/lib.rs`), 6 tests, all pass with
`cargo test --offline --lib tppe` (12 TPPE tests total):

| Function | What it does |
|---|---|
| `encode_at(vec, arm, state)` | Rotates the 32 rotary pairs by the arm's angles and applies the radial scale. For `TwoAngleV2` it is the same operation as the trace producer's TPPE arm. |
| `relocate_key(vec, arm, from, to)` | Re-addresses a key encoded at state `from` to state `to` without recomputing the projection: rotate each pair by the angle difference, multiply by the scale ratio. Tested equal to fresh encoding (max error < 2e-5) and score-preserving for a 16-key page moved from positions 100..116 to 900..916. |
| `relocate_rope_key(vec, rope_config, from, to)` | The same for the trained RoPE transform (signed position difference, f64, no table limit). Tested equal to `RopeFrequencies::apply_in_place` at the target. NoPE layers need no relocation. |
| `positional_kernel(arm, a, b)` | Mean cosine of pair-angle differences; 1.0 means two positions are indistinguishable through position. |
| `max_encodable_positions(config, limit)` | How many positions a configuration steps through before the state fails closed. |
| `PageArm::MultiscaleSpiralV3` | Candidate arm (see finding 3). Not qualified. |

## Findings (measured, with the contract's numbers)

1. **`linear_log_v1` cannot encode more than 624 positions.** The hyperbolic radius passes
   about 35.6 by position 623; `tanh(r/2)` then equals 1.0 in f64, so the strict-expansion check
   in `TppeState::step` fails closed at step 624. At position 255 the represented radius is
   already 0.9974, so the radial scale is flat (1.04987 to 1.05). Test:
   `linear_log_stops_at_624_positions_and_log_does_not`.
   `log_v1` (span matched at one revolution, as `default_tppe_config` does) still runs at two
   million positions (radius 3.2, represented 0.92). This is a length limit, not a quality
   verdict; the ruling's bakeoff is still required. It does mean any trace or KV page longer
   than 624 tokens cannot be produced under `linear_log_v1` as configured.

2. **The v2 TPPE arms alias far-apart positions.** They rotate every pair by one of two wrapped
   angles. Over 512 positions, 615 pairs at least 8 apart have a positional kernel above 0.99;
   positions 346 and 411 (65 apart) have 0.99998. Over 4,096 positions: 26,625 such pairs.
   The model's RoPE over the same 32 pairs has none above 0.83. The fast phase increment passes
   a full turn near position 357 and the slow one near 689, which is where the collisions start.
   Test: `two_angle_arm_aliases_far_positions_and_multiscale_does_not`.

3. **Candidate fix, `multiscale_spiral_v3`.** Keep the frozen phase recurrence, drives and
   derived revolutions; give pair `2j` the angle `theta_fast_unwrapped * 1e4^(-j/16)` and pair
   `2j+1` the same for the slow phase. Worst far-pair kernel drops from 0.99998 to 0.898 at 512
   positions and stays 0.898 at 4,096 (zero pairs above 0.90). Adjacent positions remain
   distinguishable (adjacent kernel 0.35 to 0.999). The radial term is unchanged. The base 1e4
   and 16 steps are untuned choices.

4. **Relocation is exact for position, not for content.** Moving a stored K page to a new
   position fixes its positional encoding exactly. It does not make the page equal to what the
   model would compute at the new position, because a key at layer L depends on every earlier
   token in its original context. Reuse is exact only when the page's own prefix is the same
   (prefix caching), which `ExactKvCache` already proves bit-exact over 52 layers. Reusing a
   page under a different prefix is an approximation whose error must be measured per layer.

## On the O(n²) cost

Exact attention reads all n cached keys per token. What can bring a local forward pass below
that, with what exists here:

- **Sparse retrieval of pages.** Score page summaries (for example the mean key of each page)
  against the query, attend only over the top k pages plus a recent window. Cost per token is
  O(pages + k·page_len) instead of O(n). The hyperbolic address of each page (`transformer_kv_page.address`)
  can serve as the coarse index. This is approximate; quality must be measured.
- **Prefix reuse.** Already exact and implemented: a shared prefix is computed once.
- **Relocation** (this module) makes retrieved pages usable at any position.

None of these is implemented in the forward pass yet.

## Next steps, in order

1. Register `multiscale_spiral_v3` as its own contract identity (new contract file, new source
   digests) before any trace uses it. Do not change `tppe-layer3-candidate-v2`.
2. Decide the radial law for long contexts: `linear_log_v1` as configured cannot pass 624
   positions. Options: `log_v1`, or a smaller `linear_log_v1` span with its own identity.
3. Run the existing Layer-3 bakeoff (`globalmind_tppe_qualify`, `qualify_position_encoders`)
   with `multiscale_spiral_v3` as an extra arm against `NativeRope`, on the real Qwen3.6 trace.
   Blocked here: the Qwen3.6 tokenizer and checkpoint at
   `/Users/stewartbarteau/models/fused/Qwen3.6-MoE-Combined/` are absent on this Mac (as the
   2026-11-10 receipt also found).
4. Glimmer (`~/models/Muse-Glimmer-q4km-gs128-MLX`, 17 GB) is present and `glimmer_exact_smoke`
   exists. Its trained transform is RoPE on sliding layers and NoPE on full layers, so TPPE
   cannot replace it without retraining; `relocate_rope_key` is the piece that applies to it.
   Next measurable experiment: store K/V pages from one context, relocate them into a second
   context, and measure logit error against exact recomputation per layer. Free memory was about
   12 GB when checked, below the model size, so close other models first.
5. Only after 3 and 4: sparse page retrieval in `AttentionBackend`, measured against exact.

## Not done

No forward pass was run this session. No bakeoff was run. Nothing in the live SurrealDB tensor
tables was changed (one 2×4 test page remains).
