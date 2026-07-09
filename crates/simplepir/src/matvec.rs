//! Width-adaptive register-blocked accumulate matvec — the shared
//! `acc += qᵀ·D mod 2³²` kernel behind the LWE hot loops.
//!
//! # Purpose
//!
//! Every online/offline LWE operation in this crate is a vector times a
//! row-major matrix folded into a row-wide accumulator: `answer`
//! (`qu × D`), `setup` (`A_col × D`), and the client precompute
//! (`s × A`, `s × hint`). The naive loop
//!
//! ```text
//! for i in rows { for j in width { acc[j] += q[i] · d[i][j] } }
//! ```
//!
//! stores to the accumulator once per cell; register-blocking `R` rows
//! per pass amortises that store and gives the autovectorizer `R`
//! independent products to pipeline. Ported from the RisePIR reference
//! kernel so the two schemes are benchmarked under the same model.
//!
//! # Design / architecture
//!
//! - **All `u32`.** Database, query, `A`, and `hint` are every one `u32`
//!   in `Z_q` (`q = 2³²`); the database cells additionally lie in `[0, p)`
//!   (high bits zero), which keeps the `u32` accumulator faithful to the
//!   mod-`q` sum. Matches RisePIR's plain-`u32` kernel.
//! - **Register blocking.** `R` rows per pass; `R` is the largest power
//!   of two `≤ 16` whose block footprint `R·width` stays within 2048
//!   cells.
//! - **Bit-exact.** `u32` wrapping add is associative and commutative, so
//!   regrouping the row sum by blocks is bit-identical to the naive loop
//!   for any `R`. Pinned by the unit tests.
//! - **Constant-time schedule.** No data-dependent branches or indices —
//!   the loop shape depends only on the public `(rows, width)`.

/// Fold `qᵀ · D` into `acc` (all arithmetic mod `2³²`).
///
/// `D` is `d`, a row-major `q.len() × acc.len()` matrix: cell `(i, j)`
/// lives at `d[i * acc.len() + j]`. `acc[j] += Σ_i q[i] · d[i][j]`.
///
/// # Constraints
///
/// Panics (debug) if `d.len() != q.len() * acc.len()`. An empty `acc`
/// or `q` is a no-op.
///
/// # Complexity
///
/// `Θ(q.len() · acc.len())` wrapping multiply-adds.
pub(crate) fn matvec_accumulate(acc: &mut [u32], d: &[u32], q: &[u32]) {
    debug_assert_eq!(d.len(), q.len() * acc.len(), "matvec shape mismatch");
    match acc.len() {
        0 => (),
        1..=128 => block_pass::<16>(acc, d, q),
        129..=256 => block_pass::<8>(acc, d, q),
        257..=512 => block_pass::<4>(acc, d, q),
        513..=1024 => block_pass::<2>(acc, d, q),
        _ => block_pass::<1>(acc, d, q),
    }
}

/// One monomorphised blocking level: fold `R` rows per pass over `acc`,
/// then the `q.len() mod R` tail rows one at a time.
#[inline]
fn block_pass<const R: usize>(acc: &mut [u32], d: &[u32], q: &[u32]) {
    let width = acc.len();
    let n = q.len();
    if width == 0 || n == 0 {
        return;
    }
    let full = n - n % R;

    let blocks = d[..full * width].chunks_exact(R * width);
    for (block, qs) in blocks.zip(q[..full].chunks_exact(R)) {
        for j in 0..width {
            let mut x = acc[j];
            for r in 0..R {
                x = x.wrapping_add(qs[r].wrapping_mul(block[r * width + j]));
            }
            acc[j] = x;
        }
    }

    // Tail rows: identical arithmetic → bit-for-bit match with the naive loop.
    for (row, &qi) in d[full * width..].chunks_exact(width).zip(&q[full..]) {
        for (x, &cell) in acc.iter_mut().zip(row) {
            *x = x.wrapping_add(qi.wrapping_mul(cell));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill_pseudorandom(buf: &mut [u32], mut state: u32) {
        for cell in buf {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *cell = state;
        }
    }

    fn naive(acc: &mut [u32], d: &[u32], q: &[u32]) {
        let width = acc.len();
        for (i, &qi) in q.iter().enumerate() {
            for j in 0..width {
                acc[j] = acc[j].wrapping_add(qi.wrapping_mul(d[i * width + j]));
            }
        }
    }

    #[test]
    fn matches_naive_across_shapes_u32() {
        let shapes = [
            (1usize, 1usize),
            (7, 3),
            (16, 29),
            (33, 128),
            (129, 129),
            (100, 517),
            (65, 1024),
            (5, 2049),
            (3, 208),
        ];
        for (n, width) in shapes {
            let mut d = vec![0u32; n * width];
            let mut q = vec![0u32; n];
            fill_pseudorandom(&mut d, 0xC0FF_EE00 ^ ((n as u32) << 12) ^ width as u32);
            fill_pseudorandom(&mut q, 0xBEEF_0000 | width as u32);

            let mut expected = vec![0u32; width];
            naive(&mut expected, &d, &q);
            let mut got = vec![0u32; width];
            matvec_accumulate(&mut got, &d, &q);
            assert_eq!(got, expected, "u32 mismatch at n={n} width={width}");
        }
    }

    /// The database path uses `Z_p`-bounded cells (high bits zero); it
    /// still matches the naive loop at the DB-shaped geometries.
    #[test]
    fn matches_naive_bounded_cells() {
        let shapes = [(16usize, 40usize), (129, 129), (73, 264), (43, 1032)];
        for (n, width) in shapes {
            let mut d = vec![0u32; n * width];
            let mut q = vec![0u32; n];
            fill_pseudorandom(&mut d, 0x51A7_0000 ^ width as u32);
            fill_pseudorandom(&mut q, 0x0DD0_0000 | n as u32);
            // Bound the DB cells to [0, 512) as the p = 2^9 path would.
            for cell in &mut d {
                *cell &= 0x1ff;
            }

            let mut expected = vec![0u32; width];
            naive(&mut expected, &d, &q);
            let mut got = vec![0u32; width];
            matvec_accumulate(&mut got, &d, &q);
            assert_eq!(
                got, expected,
                "bounded-cell mismatch at n={n} width={width}"
            );
        }
    }

    #[test]
    fn accumulates_in_place() {
        let (n, width) = (10usize, 40usize);
        let mut d = vec![0u32; n * width];
        let mut q = vec![0u32; n];
        fill_pseudorandom(&mut d, 0x1234_5678);
        fill_pseudorandom(&mut q, 0x9ABC_DEF0);

        let mut base = vec![0u32; width];
        fill_pseudorandom(&mut base, 0x0F0F_0F0F);

        let mut expected = base.clone();
        naive(&mut expected, &d, &q);
        let mut got = base;
        matvec_accumulate(&mut got, &d, &q);
        assert_eq!(got, expected);
    }

    #[test]
    fn empty_inputs_are_noops() {
        let mut acc: [u32; 0] = [];
        matvec_accumulate(&mut acc, &[] as &[u32], &[1, 2, 3]);

        let mut acc = [7u32, 8];
        matvec_accumulate(&mut acc, &[] as &[u32], &[]);
        assert_eq!(acc, [7, 8]);
    }
}
