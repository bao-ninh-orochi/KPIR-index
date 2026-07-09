//! SimplePIR LWE parameters and the plaintext-modulus correctness bound.
//!
//! # Purpose
//!
//! [`SimpleParams`] carries the runtime LWE knobs consumed by the matvec
//! routines (`lwe_dim`, `plaintext_bits`, `sigma`, public seed).
//! [`SimpleConfig`] is the small, stable user-facing surface
//! (`lwe_dim`, `sigma`) a deployment commits to. [`noise_bound_satisfied`]
//! is the SimplePIR decode-correctness predicate that both the backend
//! guard and the `kpir-index` operating-point selector consult.
//!
//! # Rationale
//!
//! `q = 2^32` is implicit (native `u32` wraparound). The secret is
//! sampled uniformly over `Z_q`; the error is a discrete Gaussian of
//! width `sigma`, defaulting to the SimplePIR §4.2 value `σ = 6.4`. The
//! default LWE dimension is `N = 1275` (128-bit security under the ADPS16
//! core-SVP model, matching the RisePIR SimplePIR backend; the original
//! `mpc4j` build uses `N = 1024`). Unlike `mpc4j`'s fixed `p = 256`, the
//! plaintext width is chosen **adaptively** per database geometry — the
//! largest `plaintext_bits` that still satisfies [`noise_bound_satisfied`]
//! — matching RisePIR so the two schemes are benchmarked at the same
//! operating point.
//!
//! # Correctness bound: the non-square correction
//!
//! SimplePIR's published decode bound (Henzinger et al., USENIX Sec'23,
//! Thm C.1) reads `⌊q/p⌋ ≥ √2·σ·p·N^(1/4)·√(ln(2/δ))`. The `N^(1/4)`
//! there is `√(√N)` **only because the paper assumes a square `√N×√N`
//! database** — each output noise coordinate then sums over exactly `√N`
//! cells. KPIR^index matrices are **not** square: the encoded DB is
//! `R × C` with `C = ⌈√(n·partition)⌉` (query/upload dim) and
//! `R = rows·partition` (response/download dim). The online answer
//! `ans = D·qu` makes each output coordinate `Σ_{c<C} D[r,c]·e[c]` — a
//! sum over the **`C` columns**, since the query error `e` has length `C`
//! — so the noise-summation dimension is `C`, and the paper's `N^(1/4)`
//! must be replaced by `√C`. A second adjustment (`2√2` vs the paper's
//! `√2`) covers cells living in `[0, p)` rather than centered
//! `[−p/2, p/2)`. Both are baked into [`noise_bound_satisfied`], and the
//! backend guard ([`crate::SimplePirServer::from_transposed_db`])
//! re-checks the same predicate.
//!
//! With these corrections the adaptive selector (`kpir-index`'s
//! `MatrixShape::choose`) picks `plaintext_bits = 9 / 9 / 8` for
//! `ℓ = 32 / 256 / 1024 B` at `m = 10⁶` (`σ = 6.4`), matching RisePIR-S.

/// SimplePIR decode-failure budget `δ = 2⁻⁴⁰` (the value SimplePIR §4.2
/// instantiates), used by the [`noise_bound_satisfied`] correctness bound.
const LOG2_DELTA: f64 = -40.0;

/// Largest plaintext width the adaptive selector considers. Matches the
/// RisePIR scan range (`ikpir-common::pir_params`) so the two schemes pick
/// identical operating points wherever their geometries coincide.
pub const MAX_PLAINTEXT_BITS: u32 = 14;

/// Whether plaintext width `plaintext_bits` decodes correctly at `q = 2³²`
/// over a matvec that sums `summation_dim` cells, under the SimplePIR
/// Gaussian correctness bound (Henzinger et al., USENIX Security 2023,
/// Theorem C.1):
///
/// ```text
/// Δ = q/p ≥ 2√2 · σ · √(ln(2/δ)) · p · √summation_dim ,   δ = 2⁻⁴⁰ ,
/// ```
///
/// with `Δ = 2^(32 − plaintext_bits)` and `p = 2^plaintext_bits`. The
/// leading `2√2` (vs the paper's `√2`) accounts for **uncentered** cells
/// in `[0, p)`: the worst-case database-row norm is `p·√dim`, not
/// `(p/2)·√dim`. `summation_dim` is the number of cells the online answer
/// accumulates — for Row-KOPIR the query dimension `C` (the answer
/// `ans = D·qu` sums over the `C` columns). Using `√summation_dim` here,
/// rather than the paper's `N^(1/4) = √(√N)`, is the **non-square**
/// correction: `N^(1/4)` is only the summed-cell count for a square
/// `√N×√N` matrix, which KPIR^index's `R×C` geometry is not (see the
/// module docs).
///
/// This is the single source of truth shared by the backend guard
/// ([`crate::SimplePirServer::from_transposed_db`]) and the `kpir-index`
/// selector, so the two can never drift. Twin of RisePIR's
/// `ikpir-common::pir_params::simple_max_plaintext_bits`.
#[inline]
pub fn noise_bound_satisfied(plaintext_bits: u32, summation_dim: u32, sigma: f64) -> bool {
    // 2√2 · σ · √(ln(2/δ)):  ln(2/δ) = (1 − log₂δ)·ln 2 = 41·ln 2.
    let factor = 2.0 * std::f64::consts::SQRT_2 * sigma * ((1.0 - LOG2_DELTA) * 2f64.ln()).sqrt();
    let delta = 2f64.powi((u32::BITS - plaintext_bits) as i32);
    let p = 2f64.powi(plaintext_bits as i32);
    delta >= factor * p * f64::from(summation_dim).sqrt()
}

/// Runtime LWE parameters for a Row-KOPIR (SimplePIR) instance.
#[derive(Clone, Debug)]
pub struct SimpleParams {
    /// LWE dimension `N`.
    pub lwe_dim: u32,
    /// log₂ of the plaintext modulus `p`. Required: `1 ≤ plaintext_bits ≤ 31`.
    /// Chosen adaptively per database geometry (see [`noise_bound_satisfied`]).
    pub plaintext_bits: u32,
    /// Standard deviation of the discrete-Gaussian error (SimplePIR: `6.4`).
    pub sigma: f64,
    /// 128-bit public seed used to expand the LWE matrix `A`.
    pub seed: [u8; 16],
}

impl SimpleParams {
    /// Default LWE dimension for 128-bit security (ADPS16, matches RisePIR-S).
    pub const DEFAULT_LWE_DIM: u32 = 1275;

    /// Default error standard deviation (SimplePIR §4.2).
    pub const DEFAULT_SIGMA: f64 = 6.4;

    /// Construct explicit LWE parameters.
    ///
    /// # Constraints
    ///
    /// Panics if `plaintext_bits` is outside `1..=31`, if `lwe_dim == 0`,
    /// or if `sigma` is not positive and finite. The upper bound `31`
    /// keeps `Δ = q/p = 2^(32 − plaintext_bits) ≥ 2` and the recover
    /// rounding shift in range; decode *correctness* for a given geometry
    /// is enforced separately by [`noise_bound_satisfied`].
    pub fn new(lwe_dim: u32, plaintext_bits: u32, sigma: f64, seed: [u8; 16]) -> Self {
        assert!(
            (1..=31).contains(&plaintext_bits),
            "plaintext_bits must be in 1..=31, got {plaintext_bits}"
        );
        assert!(lwe_dim > 0, "lwe_dim must be positive");
        assert!(
            sigma > 0.0 && sigma.is_finite(),
            "sigma must be positive and finite, got {sigma}"
        );
        Self {
            lwe_dim,
            plaintext_bits,
            sigma,
            seed,
        }
    }

    /// The plaintext-to-ciphertext scaling `Δ = q/p = 2^(32 − plaintext_bits)`.
    #[inline]
    pub const fn delta(&self) -> u32 {
        1u32 << (u32::BITS - self.plaintext_bits)
    }

    /// The plaintext modulus `p = 2^plaintext_bits`.
    #[inline]
    pub const fn plaintext_modulus(&self) -> u32 {
        1u32 << self.plaintext_bits
    }
}

/// User-facing tunable knobs for a Row-KOPIR (SimplePIR) instance.
#[derive(Clone, Debug, PartialEq)]
pub struct SimpleConfig {
    /// LWE dimension `N`. Must be `> 0`.
    pub lwe_dim: u32,
    /// Discrete-Gaussian error `σ`. Must be `> 0` and finite.
    pub sigma: f64,
}

impl SimpleConfig {
    /// New config with both knobs explicit.
    ///
    /// # Constraints
    ///
    /// Panics if `lwe_dim == 0` or `sigma` is not positive and finite.
    pub fn new(lwe_dim: u32, sigma: f64) -> Self {
        assert!(lwe_dim > 0, "lwe_dim must be positive");
        assert!(
            sigma > 0.0 && sigma.is_finite(),
            "sigma must be positive and finite, got {sigma}"
        );
        Self { lwe_dim, sigma }
    }

    /// New config overriding only `lwe_dim`; `sigma` stays at `6.4`.
    pub fn with_lwe_dim(lwe_dim: u32) -> Self {
        Self::new(lwe_dim, SimpleParams::DEFAULT_SIGMA)
    }

    /// Materialise runtime [`SimpleParams`] at the given (adaptively
    /// chosen) plaintext width with the given public seed.
    pub fn to_params(&self, seed: [u8; 16], plaintext_bits: u32) -> SimpleParams {
        SimpleParams::new(self.lwe_dim, plaintext_bits, self.sigma, seed)
    }
}

impl Default for SimpleConfig {
    /// 128-bit security default (`lwe_dim = 1275`, `sigma = 6.4`).
    fn default() -> Self {
        Self {
            lwe_dim: SimpleParams::DEFAULT_LWE_DIM,
            sigma: SimpleParams::DEFAULT_SIGMA,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bound is monotone: if `pb` decodes at a given dimension, so does
    /// every smaller width; and a wider dimension can only tighten it.
    #[test]
    fn bound_is_monotone() {
        let dim = 15_000;
        // Find the max pb that passes at this dimension.
        let max_pb = (1..=MAX_PLAINTEXT_BITS)
            .rev()
            .find(|&pb| noise_bound_satisfied(pb, dim, 6.4))
            .unwrap();
        for pb in 1..=max_pb {
            assert!(noise_bound_satisfied(pb, dim, 6.4), "pb={pb} must hold");
        }
        assert!(
            !noise_bound_satisfied(max_pb + 1, dim, 6.4),
            "pb={} must fail",
            max_pb + 1
        );
        // A larger summation dimension never admits a larger pb.
        assert!(noise_bound_satisfied(max_pb, dim, 6.4));
        assert!(!noise_bound_satisfied(max_pb + 1, dim * 4, 6.4));
    }

    /// Pinned operating points at the CANS2026 head-to-head column counts
    /// (`m = 10⁶`, `σ = 6.4`): the max pb from the bound is 9 / 9 / 8 for
    /// the ℓ = 32 / 256 / 1024 B geometries, matching RisePIR-S.
    #[test]
    fn pinned_max_pb_matches_risepir() {
        // (columns C at the chosen pb, expected max pb).
        let cases = [(6000u32, 9u32), (15330, 9), (32125, 8)];
        for (dim, want) in cases {
            let got = (1..=MAX_PLAINTEXT_BITS)
                .rev()
                .find(|&pb| noise_bound_satisfied(pb, dim, 6.4))
                .unwrap();
            assert_eq!(got, want, "dim={dim}");
        }
    }

    /// Wider error (larger σ) can only shrink the operating point.
    #[test]
    fn monotone_in_sigma() {
        let dim = 15_000;
        let pb_wide = (1..=MAX_PLAINTEXT_BITS)
            .rev()
            .find(|&pb| noise_bound_satisfied(pb, dim, 12.8))
            .unwrap();
        let pb_narrow = (1..=MAX_PLAINTEXT_BITS)
            .rev()
            .find(|&pb| noise_bound_satisfied(pb, dim, 6.4))
            .unwrap();
        assert!(pb_wide <= pb_narrow);
    }
}
