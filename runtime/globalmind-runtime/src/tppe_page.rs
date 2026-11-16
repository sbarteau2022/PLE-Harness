//! TPPE page operations at the Q/K boundary. CANDIDATE code, added 2026-11-15.
//!
//! Nothing here changes `tppe.rs`, the Layer-3 trace producer, or the
//! `tppe-layer3-candidate-v2` contract. It adds three things that contract does
//! not have, each as a pure function with tests:
//!
//! 1. **Key relocation.** Every TPPE arm encodes a key as a rotation of rotary
//!    pairs by angles that depend only on the positional state, times a positive
//!    radial scale. Rotations and positive scales form a group, so a key encoded
//!    at state `a` is re-addressed to state `b` exactly by rotating each pair by
//!    the angle difference and multiplying by the scale ratio. This is what a
//!    stored K page needs in order to be reused at a different position without
//!    recomputing the projection. Values carry no position and need no change.
//!    It is exact for the positional part only: a hidden state at depth still
//!    depends on the context it was computed in.
//!
//! 2. **Encodable length.** `max_encodable_positions` reports how many positions
//!    a configuration can step through before the state fails closed. With the
//!    contract's numbers, `linear_log_v1` stops at 624 positions (the represented
//!    radius stops growing in f64); `log_v1` passes two million.
//!
//! 3. **`multiscale_spiral_v3` candidate arm.** The v2 TPPE arms rotate every
//!    pair by one of two wrapped angles, so far-apart positions can receive
//!    nearly the same code (positions 346 and 411 have a positional kernel of
//!    0.99998). The candidate keeps the frozen phase recurrence and derived
//!    revolutions, and spreads the two *unwrapped* phases over a geometric scale
//!    ladder, as RoPE does with raw position. It is NOT qualified: it needs its
//!    own contract identity and a real-model bakeoff before any model use.

use crate::TppeState;

/// Rotary pairs at the Layer-3 Q/K boundary (64-wide rotary domain, half split).
pub const ROTARY_PAIRS: usize = 32;
/// Radial gain of the TPPE arms: `1 + RADIAL_GAIN * poincare_radius`.
pub const RADIAL_GAIN: f64 = 0.05;
/// Scale ladder base for `multiscale_spiral_v3` (candidate numeric, not frozen).
pub const MULTISCALE_BASE: f64 = 1.0e4;
/// Scales per phase in `multiscale_spiral_v3`.
pub const MULTISCALE_STEPS: usize = ROTARY_PAIRS / 2;

/// Which boundary operation turns a positional state into pair angles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PageArm {
    /// The v2 TPPE arms: wrapped fast phase on even pairs, slow on odd pairs.
    TwoAngleV2,
    /// Candidate: unwrapped fast/slow phase times `MULTISCALE_BASE^(-j/16)`.
    MultiscaleSpiralV3,
}

impl PageArm {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TwoAngleV2 => "two_angle_v2",
            Self::MultiscaleSpiralV3 => "multiscale_spiral_v3",
        }
    }

    /// Rotation angle of one rotary pair at a positional state.
    pub fn pair_angle(self, state: &TppeState, pair: usize) -> f64 {
        match self {
            Self::TwoAngleV2 => state.phase_t2()[pair % 2],
            Self::MultiscaleSpiralV3 => {
                let phase = if pair % 2 == 0 {
                    state.theta_fast_unwrapped
                } else {
                    state.theta_slow_unwrapped
                };
                let step = (pair / 2) as f64 / MULTISCALE_STEPS as f64;
                phase * MULTISCALE_BASE.powf(-step)
            }
        }
    }
}

pub fn radial_scale(state: &TppeState) -> f64 {
    1.0 + RADIAL_GAIN * state.poincare_radius
}

fn rotate_pairs(vec: &mut [f32], angle: impl Fn(usize) -> f64, scale: f64) -> Result<(), String> {
    if vec.len() < 2 * ROTARY_PAIRS {
        return Err("vector is shorter than the rotary domain".into());
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err("radial scale must be finite and positive".into());
    }
    for pair in 0..ROTARY_PAIRS {
        let (sin_a, cos_a) = angle(pair).sin_cos();
        let (x0, x1) = (vec[pair] as f64, vec[pair + ROTARY_PAIRS] as f64);
        vec[pair] = (scale * (x0 * cos_a - x1 * sin_a)) as f32;
        vec[pair + ROTARY_PAIRS] = (scale * (x1 * cos_a + x0 * sin_a)) as f32;
    }
    Ok(())
}

/// Encodes a raw (normalized, unrotated) key or query at a positional state.
/// For `TwoAngleV2` this is the same operation as the trace producer's TPPE arm.
pub fn encode_at(vec: &mut [f32], arm: PageArm, state: &TppeState) -> Result<(), String> {
    state.validate()?;
    rotate_pairs(vec, |pair| arm.pair_angle(state, pair), radial_scale(state))
}

/// Re-addresses a key that was encoded at `from` so that it equals the same raw
/// key encoded at `to`. No projection or normalization is recomputed.
pub fn relocate_key(
    vec: &mut [f32],
    arm: PageArm,
    from: &TppeState,
    to: &TppeState,
) -> Result<(), String> {
    from.validate()?;
    to.validate()?;
    rotate_pairs(
        vec,
        |pair| arm.pair_angle(to, pair) - arm.pair_angle(from, pair),
        radial_scale(to) / radial_scale(from),
    )
}

/// Positional kernel between two states: the mean over pairs of the cosine of the
/// angle difference. 1.0 means the two positions are indistinguishable to
/// attention through position alone.
pub fn positional_kernel(arm: PageArm, a: &TppeState, b: &TppeState) -> f64 {
    (0..ROTARY_PAIRS)
        .map(|pair| (arm.pair_angle(a, pair) - arm.pair_angle(b, pair)).cos())
        .sum::<f64>()
        / ROTARY_PAIRS as f64
}

/// Number of positions the configuration can step through, up to `limit`, using
/// the same positional drives as `trace::tppe_states_for_positions`.
pub fn max_encodable_positions(config: &crate::TppeConfig, limit: usize) -> usize {
    let Ok(mut state) = TppeState::origin(*config) else {
        return 0;
    };
    for step in 0..limit {
        let drive_fast = (step as f64 + 1.0) * 0.017;
        let drive_slow = (step as f64 + 1.0) * 0.009;
        match state.step(*config, drive_fast, drive_slow) {
            Ok(next) => state = next,
            Err(_) => return step,
        }
    }
    limit
}

/// Re-addresses a RoPE-encoded key from token position `from` to `to`, for the
/// native (trained) position transform: half-split pairs over the first
/// `head_dim * partial_rotary_factor` dimensions, angle `position * theta^(-2i/head_dim)`.
/// Layers without a position transform (NoPE) need no relocation at all.
/// Computed in f64 from the signed position difference; no table, no range limit.
pub fn relocate_rope_key(
    vec: &mut [f32],
    config: &crate::position_transform::RopeConfig,
    from: usize,
    to: usize,
) -> Result<(), String> {
    let rotary_dim = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
    let half = rotary_dim / 2;
    if half == 0 || vec.len() < rotary_dim {
        return Err("vector is shorter than the RoPE rotary domain".into());
    }
    let delta = to as f64 - from as f64;
    for i in 0..half {
        let freq = config.theta.powf(-((2 * i) as f64) / config.head_dim as f64);
        let (sin_a, cos_a) = (delta * freq).sin_cos();
        let (x0, x1) = (vec[i] as f64, vec[i + half] as f64);
        vec[i] = (x0 * cos_a - x1 * sin_a) as f32;
        vec[i + half] = (x1 * cos_a + x0 * sin_a) as f32;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{default_tppe_config, tppe_states_for_positions};
    use crate::RadialLaw;

    fn raw_key(seed: u32) -> Vec<f32> {
        // Deterministic non-trivial vector; no randomness crate needed.
        (0..2 * ROTARY_PAIRS)
            .map(|i| (((i as u32).wrapping_mul(2654435761u32) ^ seed) % 2001) as f32 / 1000.0 - 1.0)
            .collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn relocation_equals_fresh_encoding_for_both_arms() {
        let config = default_tppe_config(RadialLaw::LogV1);
        let states = tppe_states_for_positions(&config, 4096).unwrap();
        for arm in [PageArm::TwoAngleV2, PageArm::MultiscaleSpiralV3] {
            for (from, to) in [(0usize, 1usize), (7, 300), (300, 7), (511, 4095), (4095, 0)] {
                let raw = raw_key(from as u32 * 31 + to as u32);
                let mut moved = raw.clone();
                encode_at(&mut moved, arm, &states[from]).unwrap();
                relocate_key(&mut moved, arm, &states[from], &states[to]).unwrap();
                let mut fresh = raw.clone();
                encode_at(&mut fresh, arm, &states[to]).unwrap();
                assert!(
                    max_abs_diff(&moved, &fresh) < 2e-5,
                    "{} {from}->{to}: {}",
                    arm.as_str(),
                    max_abs_diff(&moved, &fresh)
                );
            }
        }
    }

    #[test]
    fn attention_scores_survive_relocating_a_whole_page() {
        // A page of keys computed at positions 100..116 is reused at 900..916.
        // Scores against a query must equal those of keys computed there directly.
        let config = default_tppe_config(RadialLaw::LogV1);
        let states = tppe_states_for_positions(&config, 1024).unwrap();
        let arm = PageArm::MultiscaleSpiralV3;
        let mut query = raw_key(99);
        encode_at(&mut query, arm, &states[1000]).unwrap();
        for offset in 0..16usize {
            let raw = raw_key(offset as u32);
            let mut page_key = raw.clone();
            encode_at(&mut page_key, arm, &states[100 + offset]).unwrap();
            relocate_key(&mut page_key, arm, &states[100 + offset], &states[900 + offset]).unwrap();
            let mut direct = raw.clone();
            encode_at(&mut direct, arm, &states[900 + offset]).unwrap();
            let dot = |k: &[f32]| query.iter().zip(k).map(|(q, x)| q * x).sum::<f32>();
            assert!((dot(&page_key) - dot(&direct)).abs() < 1e-3);
        }
    }

    #[test]
    fn linear_log_stops_at_624_positions_and_log_does_not() {
        let linear = default_tppe_config(RadialLaw::LinearLogV1);
        let log = default_tppe_config(RadialLaw::LogV1);
        assert_eq!(max_encodable_positions(&linear, 100_000), 624);
        assert!(tppe_states_for_positions(&linear, 625).is_err());
        assert_eq!(max_encodable_positions(&log, 200_000), 200_000);
    }

    #[test]
    fn two_angle_arm_aliases_far_positions_and_multiscale_does_not() {
        let config = default_tppe_config(RadialLaw::LogV1);
        let states = tppe_states_for_positions(&config, 2048).unwrap();
        // Measured alias of the v2 arm: positions 346 and 411, 65 apart.
        assert!(positional_kernel(PageArm::TwoAngleV2, &states[346], &states[411]) > 0.9999);
        let mut worst_v2 = 0.0f64;
        let mut worst_v3 = 0.0f64;
        for m in 0..states.len() {
            for n in (m + 8)..states.len() {
                worst_v2 = worst_v2.max(positional_kernel(PageArm::TwoAngleV2, &states[m], &states[n]));
                worst_v3 = worst_v3.max(positional_kernel(PageArm::MultiscaleSpiralV3, &states[m], &states[n]));
            }
        }
        assert!(worst_v2 > 0.9999, "v2 worst far kernel {worst_v2}");
        assert!(worst_v3 < 0.92, "v3 worst far kernel {worst_v3}");
    }

    #[test]
    fn rope_relocation_equals_fresh_rope_encoding() {
        use crate::position_transform::{RopeConfig, RopeFrequencies};
        let config = RopeConfig::default();
        let table = RopeFrequencies::new(&config, 4096);
        for (from, to) in [(0usize, 1usize), (12, 900), (900, 12), (2047, 4095), (4095, 3)] {
            let raw: Vec<f32> = (0..config.head_dim).map(|i| ((i * 37 + from) % 101) as f32 / 50.0 - 1.0).collect();
            let mut moved = raw.clone();
            table.apply_in_place(&mut moved, from);
            relocate_rope_key(&mut moved, &config, from, to).unwrap();
            let mut fresh = raw.clone();
            table.apply_in_place(&mut fresh, to);
            assert!(max_abs_diff(&moved, &fresh) < 1e-4, "{from}->{to}: {}", max_abs_diff(&moved, &fresh));
        }
    }

    #[test]
    fn short_vectors_and_bad_states_fail_closed() {
        let config = default_tppe_config(RadialLaw::LogV1);
        let states = tppe_states_for_positions(&config, 4).unwrap();
        let mut short = vec![0.0f32; 10];
        assert!(encode_at(&mut short, PageArm::TwoAngleV2, &states[0]).is_err());
        let mut forged = states[1];
        forged.poincare_radius = 1.5;
        let mut key = raw_key(1);
        assert!(relocate_key(&mut key, PageArm::TwoAngleV2, &states[0], &forged).is_err());
    }
}
