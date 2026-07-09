//! SimplePIR LWE parameters.
//!
//! # Purpose
//!
//! [`SimpleParams`] carries the runtime LWE knobs consumed by the matvec
//! routines (`lwe_dim`, `plaintext_bits`, `sigma`, public seed).
//! [`SimpleConfig`] is the small, stable user-facing surface
//! (`lwe_dim`, `sigma`) a deployment commits to.
//!
//! # Rationale
//!
//! `q = 2^32` is implicit (native `u32` wraparound). The secret is
//! sampled uniformly over `Z_q`; the error is a discrete Gaussian of
//! width `sigma`, defaulting to the SimplePIR §4.2 value `σ = 6.4`. The
//! default LWE dimension is `N = 1275` (128-bit security under the ADPS16
//! core-SVP model, matching the RisePIR SimplePIR backend; the original
//! `mpc4j` build uses `N = 1024`). KPIR^index fixes `plaintext_bits = 8`
//! (`p = 256`, one byte per `Z_p` cell), the `mpc4j` setting.

/// Runtime LWE parameters for a Row-KOPIR (SimplePIR) instance.
#[derive(Clone, Debug)]
pub struct SimpleParams {
    /// LWE dimension `N`.
    pub lwe_dim: u32,
    /// log₂ of the plaintext modulus `p`. Required: `1 ≤ plaintext_bits ≤ 24`
    /// (KPIR^index uses `8`, i.e. `p = 256`).
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

    /// KPIR^index plaintext width: `p = 2^8 = 256` (one byte per cell).
    pub const KPIR_PLAINTEXT_BITS: u32 = 8;

    /// Construct explicit LWE parameters.
    ///
    /// # Constraints
    ///
    /// Panics if `plaintext_bits` is outside `1..=24`, if `lwe_dim == 0`,
    /// or if `sigma` is not positive and finite. The upper bound `24`
    /// keeps `Δ = q/p = 2^(32-plaintext_bits) ≥ 2^8`, so the recover
    /// rounding has a byte of headroom.
    pub fn new(lwe_dim: u32, plaintext_bits: u32, sigma: f64, seed: [u8; 16]) -> Self {
        assert!(
            (1..=24).contains(&plaintext_bits),
            "plaintext_bits must be in 1..=24, got {plaintext_bits}"
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

    /// Materialise runtime [`SimpleParams`] at the KPIR^index plaintext
    /// width (`p = 256`) with the given public seed.
    pub fn to_params(&self, seed: [u8; 16]) -> SimpleParams {
        SimpleParams::new(
            self.lwe_dim,
            SimpleParams::KPIR_PLAINTEXT_BITS,
            self.sigma,
            seed,
        )
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
