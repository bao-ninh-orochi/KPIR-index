//! KPIR^index matrix geometry and fixed parameters.
//!
//! # Purpose
//!
//! [`MatrixShape`] computes the encoded-database dimensions from the key
//! count `n`, the value length `ℓ`, and the approximation error `ε`,
//! using the exact `mpc4j` `SimplePgmCpKsPirDesc.getMatrixSize` formula.
//! Reproducing this formula is what makes the head-to-head communication
//! numbers match the reference to the byte (CANS2026 Table 3, Hao Table 5).
//!
//! # Formula
//!
//! ```text
//! partition  = ℓ + FINGERPRINT_BYTES          // byte-planes: 8-byte fp ‖ value
//! columns    = ceil( sqrt(n · partition) )     // query upload dimension  C
//! data_rows  = max( ceil(n / columns), ε + 2 )
//! rows       = data_rows + 2ε + 3              // mpc4j padding (ε+1 top, ε+2 bottom)
//! ```
//! The Row-KOPIR matrix is `R × C` with `R = rows · partition` (the
//! response download dimension) and `C = columns`.

/// 8-byte fingerprint prepended to every stored value (`mpc4j`
/// `DIGEST_BYTE_L`). Used to disambiguate the `≤ 2ε+3` candidate entries
/// in a retrieved column.
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
    /// Bytes per stored entry: `ℓ + FINGERPRINT_BYTES` (= `Z_p` cells,
    /// one byte each). Also the number of `mpc4j` byte-planes.
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
    /// Compute the geometry for `n` pairs of `value_bytes`-byte values at
    /// error `epsilon`.
    ///
    /// # Constraints
    ///
    /// Panics if `n == 0` or `value_bytes == 0`.
    pub fn new(n: usize, value_bytes: usize, epsilon: u32) -> Self {
        assert!(n > 0, "n must be positive");
        assert!(value_bytes > 0, "value_bytes must be positive");
        let partition = value_bytes + FINGERPRINT_BYTES;
        let cells = (n as u64)
            .checked_mul(partition as u64)
            .expect("n·partition overflow");
        let columns = ceil_sqrt(cells) as usize;
        let data_rows = div_ceil(n, columns).max(epsilon as usize + 2);
        let rows = data_rows + 2 * epsilon as usize + 3;
        Self {
            n,
            value_bytes,
            partition,
            columns,
            data_rows,
            rows,
            epsilon,
        }
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

    /// Database expansion rate `N/m`: encoded slots (`columns · rows`,
    /// each `partition` bytes) per key. Equals
    /// `columns · rows · partition / (n · partition) = columns · rows / n`.
    /// Reproduces CANS2026 Table 6 and Hao Table 5 exactly (e.g. at
    /// `m = 10^6`: 1.075 / 1.186 / 1.381 for ℓ = 32 / 256 / 1024 B).
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

/// `ceil(a / b)` for positive `b`.
#[inline]
fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The geometry reproduces the CANS2026 Table 3 head-to-head numbers
    /// (m = 10^6 keys) to the byte, and the Table 6 expansion rate.
    #[test]
    fn matches_cans2026_table3() {
        // (value_bytes, columns, rows, query kB, response kB, expansion N/m)
        let cases = [
            (32usize, 6325usize, 170usize, 25.3f64, 27.2f64, 1.075f64),
            (256, 16249, 73, 65.0, 77.1, 1.186),
            (1024, 32125, 43, 128.5, 177.5, 1.381),
        ];
        for (vb, cols, rows, qkb, rkb, exp) in cases {
            let s = MatrixShape::new(1_000_000, vb, 4);
            assert_eq!(s.columns, cols, "columns @ {vb}B");
            assert_eq!(s.rows, rows, "rows @ {vb}B");
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
