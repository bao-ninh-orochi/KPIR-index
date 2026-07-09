//! KPIR^index matrix geometry and fixed parameters.
//!
//! # Purpose
//!
//! [`MatrixShape`] computes the encoded-database dimensions from the key
//! count `n`, the value length `ℓ`, the approximation error `ε`, and the
//! plaintext width `plaintext_bits`, using the `mpc4j`
//! `SimplePgmCpKsPirDesc.getMatrixSize` formula generalised to a variable
//! plaintext modulus. Reproducing this formula is what makes the
//! head-to-head communication numbers match the reference (CANS2026
//! Table 3, Hao Table 5) — at `plaintext_bits = 8` it is the exact `mpc4j`
//! byte geometry; wider `p` packs the same payload into fewer cells.
//!
//! # Formula
//!
//! Each stored entry is a `fingerprint(64 bits) ‖ value(8ℓ bits)`
//! bitstring of `payload_bits = 8·(ℓ + FINGERPRINT_BYTES)` bits, packed
//! into `pt`-bit `Z_p` cells:
//!
//! ```text
//! partition  = ceil( payload_bits / plaintext_bits )   // Z_p cells per entry
//! columns    = ceil( sqrt(n · partition) )              // query upload dim  C
//! data_rows  = max( ceil(n / columns), ε + 2 )
//! rows       = data_rows + 2ε + 3                       // ε+1 top, ε+2 bottom
//! ```
//!
//! The Row-KOPIR matrix is `R × C` with `R = rows · partition` (the
//! response download dimension) and `C = columns`. At `plaintext_bits = 8`,
//! `partition = ℓ + FINGERPRINT_BYTES` — one byte per cell, the `mpc4j`
//! byte-plane layout.
//!
//! # Operating point
//!
//! [`MatrixShape::choose`] picks the largest `plaintext_bits ≤
//! `[`simplepir::MAX_PLAINTEXT_BITS`] whose geometry still decodes under
//! the SimplePIR bound ([`simplepir::noise_bound_satisfied`]) — the same
//! adaptive selection RisePIR makes, so the two are compared at matched
//! operating points.

/// 8-byte fingerprint prepended to every stored value (`mpc4j`
/// `DIGEST_BYTE_L`). Used to disambiguate the `≤ 2ε+3` candidate entries
/// in a retrieved column. A *hash* width, independent of the plaintext
/// modulus.
pub const FINGERPRINT_BYTES: usize = 8;

/// Default approximation error `ε` (`mpc4j` `EPSILON`).
pub const DEFAULT_EPSILON: u32 = 4;

/// Encoded-database geometry for one KPIR^index instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixShape {
    /// Number of key-value pairs `n`.
    pub n: usize,
    /// Value length `ℓ` in bytes.
    pub value_bytes: usize,
    /// Plaintext width `log₂ p` (adaptively chosen; `8` = `mpc4j`).
    pub plaintext_bits: u32,
    /// `Z_p` cells per stored entry: `ceil(payload_bits / plaintext_bits)`,
    /// where `payload_bits = 8·(ℓ + FINGERPRINT_BYTES)`. At
    /// `plaintext_bits = 8` this is `ℓ + FINGERPRINT_BYTES` (one byte per
    /// cell, the `mpc4j` byte-plane count).
    pub partition: usize,
    /// Query upload dimension `C = columns`.
    pub columns: usize,
    /// Data entries per column (before the boundary padding).
    pub data_rows: usize,
    /// Matrix rows per plane `= data_rows + 2ε + 3`.
    pub rows: usize,
    /// Approximation error `ε`.
    pub epsilon: u32,
}

impl MatrixShape {
    /// Payload bits per entry: `8·(value_bytes + FINGERPRINT_BYTES)`
    /// (fingerprint ‖ value).
    #[inline]
    pub fn payload_bits(value_bytes: usize) -> usize {
        8 * (value_bytes + FINGERPRINT_BYTES)
    }

    /// Compute the geometry for `n` pairs of `value_bytes`-byte values at
    /// error `epsilon` and an explicit plaintext width `plaintext_bits`.
    ///
    /// Use [`MatrixShape::choose`] to pick the width adaptively; this
    /// constructor is for the mpc4j-exact (`plaintext_bits = 8`) case and
    /// as the inner step of `choose`.
    ///
    /// # Constraints
    ///
    /// Panics if `n == 0`, `value_bytes == 0`, or `plaintext_bits == 0`.
    pub fn new(n: usize, value_bytes: usize, epsilon: u32, plaintext_bits: u32) -> Self {
        assert!(n > 0, "n must be positive");
        assert!(value_bytes > 0, "value_bytes must be positive");
        assert!(plaintext_bits > 0, "plaintext_bits must be positive");
        let partition = Self::payload_bits(value_bytes).div_ceil(plaintext_bits as usize);
        let cells = (n as u64)
            .checked_mul(partition as u64)
            .expect("n·partition overflow");
        let columns = ceil_sqrt(cells) as usize;
        let data_rows = n.div_ceil(columns).max(epsilon as usize + 2);
        let rows = data_rows + 2 * epsilon as usize + 3;
        Self {
            n,
            value_bytes,
            plaintext_bits,
            partition,
            columns,
            data_rows,
            rows,
            epsilon,
        }
    }

    /// Compute the geometry at the adaptively chosen operating point: the
    /// largest `plaintext_bits ≤ `[`simplepir::MAX_PLAINTEXT_BITS`] whose
    /// query dimension `C` still decodes under
    /// [`simplepir::noise_bound_satisfied`] at error width `sigma`. The
    /// answer matvec sums over the `C` columns, so `C` is the noise
    /// dimension. Structural twin of RisePIR's `simple_max_plaintext_bits`.
    ///
    /// # Constraints
    ///
    /// Panics if `n == 0` or `value_bytes == 0` (via [`MatrixShape::new`]).
    pub fn choose(n: usize, value_bytes: usize, epsilon: u32, sigma: f64) -> Self {
        (1..=simplepir::MAX_PLAINTEXT_BITS)
            .rev()
            .map(|pb| Self::new(n, value_bytes, epsilon, pb))
            .find(|s| simplepir::noise_bound_satisfied(s.plaintext_bits, s.columns as u32, sigma))
            // plaintext_bits = 1 satisfies the bound for any u32 geometry.
            .unwrap_or_else(|| Self::new(n, value_bytes, epsilon, 1))
    }

    /// Row-KOPIR query dimension `C` (query upload = `C` `u32` cells).
    #[inline]
    pub fn query_dim(&self) -> usize {
        self.columns
    }

    /// Row-KOPIR response dimension `R = rows · partition` (response
    /// download = `R` `u32` cells).
    #[inline]
    pub fn response_dim(&self) -> usize {
        self.rows * self.partition
    }

    /// Query upload size in bytes (fixed-width LE `u32` per cell).
    #[inline]
    pub fn query_bytes(&self) -> usize {
        self.query_dim() * core::mem::size_of::<u32>()
    }

    /// Response download size in bytes (fixed-width LE `u32` per cell).
    #[inline]
    pub fn response_bytes(&self) -> usize {
        self.response_dim() * core::mem::size_of::<u32>()
    }

    /// Database expansion rate `N/m`: encoded cells (`columns · rows ·
    /// partition`) carrying the `n · partition` payload cells, i.e.
    /// `columns · rows / n`. Reproduces CANS2026 Table 6 / Hao Table 5.
    #[inline]
    pub fn expansion_rate(&self) -> f64 {
        (self.columns * self.rows) as f64 / self.n as f64
    }
}

/// `ceil(sqrt(x))` — smallest `s` with `s·s ≥ x`.
#[inline]
fn ceil_sqrt(x: u64) -> u64 {
    if x == 0 {
        return 0;
    }
    let s = x.isqrt();
    if s * s == x {
        s
    } else {
        s + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At `plaintext_bits = 8` the geometry is the exact `mpc4j` byte
    /// layout: reproduces the original CANS2026 Table 3 numbers (m = 10⁶)
    /// to the byte, and the Table 6 expansion rate. Pins the base formula.
    #[test]
    fn matches_mpc4j_byte_geometry() {
        // (value_bytes, columns, rows, query kB, response kB, expansion N/m)
        let cases = [
            (32usize, 6325usize, 170usize, 25.3f64, 27.2f64, 1.075f64),
            (256, 16249, 73, 65.0, 77.1, 1.186),
            (1024, 32125, 43, 128.5, 177.5, 1.381),
        ];
        for (vb, cols, rows, qkb, rkb, exp) in cases {
            let s = MatrixShape::new(1_000_000, vb, 4, 8);
            assert_eq!(s.columns, cols, "columns @ {vb}B");
            assert_eq!(s.rows, rows, "rows @ {vb}B");
            assert_eq!(s.partition, vb + FINGERPRINT_BYTES, "partition @ {vb}B");
            assert!(
                (s.query_bytes() as f64 / 1000.0 - qkb).abs() < 0.05,
                "query kB @ {vb}B: {}",
                s.query_bytes() as f64 / 1000.0
            );
            assert!(
                (s.response_bytes() as f64 / 1000.0 - rkb).abs() < 0.05,
                "response kB @ {vb}B: {}",
                s.response_bytes() as f64 / 1000.0
            );
            assert!(
                (s.expansion_rate() - exp).abs() < 0.001,
                "expansion @ {vb}B: {} (want {exp})",
                s.expansion_rate()
            );
        }
    }

    /// The adaptive selector picks `plaintext_bits = 9 / 9 / 8` for
    /// ℓ = 32 / 256 / 1024 B at m = 10⁶ (σ = 6.4) — matching RisePIR-S —
    /// and the resulting geometry is smaller than the mpc4j pt=8 layout for
    /// the two narrower value sizes.
    #[test]
    fn adaptive_operating_point() {
        // (value_bytes, expected pt, expected columns, expected partition).
        let cases = [
            (32usize, 9u32, 6000usize, 36usize),
            (256, 9, 15330, 235),
            (1024, 8, 32125, 1032),
        ];
        for (vb, pt, cols, part) in cases {
            let s = MatrixShape::choose(1_000_000, vb, 4, 6.4);
            assert_eq!(s.plaintext_bits, pt, "pt @ {vb}B");
            assert_eq!(s.partition, part, "partition @ {vb}B");
            assert_eq!(s.columns, cols, "columns @ {vb}B");
            // The chosen point decodes; one bit wider would not.
            assert!(simplepir::noise_bound_satisfied(pt, cols as u32, 6.4));
            let wider = MatrixShape::new(1_000_000, vb, 4, pt + 1);
            assert!(!simplepir::noise_bound_satisfied(
                pt + 1,
                wider.columns as u32,
                6.4
            ));
        }
    }

    #[test]
    fn ceil_sqrt_exact() {
        assert_eq!(ceil_sqrt(0), 0);
        assert_eq!(ceil_sqrt(1), 1);
        assert_eq!(ceil_sqrt(4), 2);
        assert_eq!(ceil_sqrt(5), 3);
        assert_eq!(ceil_sqrt(40_000_000), 6325);
        assert_eq!(ceil_sqrt(39_992_976), 6324); // 6324^2 exactly
    }
}
