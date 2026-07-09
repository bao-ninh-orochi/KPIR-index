//! Deterministic LWE sampling primitives for the SimplePIR backend.
//!
//! # Purpose
//!
//! Three samplers shared by setup, key-gen, and query:
//! [`sample_a_transposed`] expands the public LWE matrix `A` (in its
//! transposed `N × C` layout) from a 16-byte seed;
//! [`sample_uniform_zq_into`] fills a slice with uniform `Z_q` samples
//! (the SimplePIR secret distribution); and
//! [`sample_discrete_gaussian_into`] fills a slice with discrete-Gaussian
//! samples of standard deviation `σ` (the SimplePIR error distribution).
//!
//! # Design
//!
//! All three use ChaCha20 as the PRG. `sample_a_transposed` is keyed by
//! the public seed, so server (`setup`) and client (`key-gen`) expand the
//! same `A` without shipping it. The Gaussian uses Box–Muller with a
//! saturating `f64 → i32` cast; at `σ = 6.4` the saturation branch is
//! statistically never taken. Ported from the RisePIR reference so the
//! two schemes share an identical error/secret model.

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// Build a ChaCha20Rng from a 16-byte seed (zero-padded to 32 bytes).
fn rng_from_seed(seed: &[u8; 16]) -> ChaCha20Rng {
    let mut padded = [0u8; 32];
    padded[..16].copy_from_slice(seed);
    ChaCha20Rng::from_seed(padded)
}

/// Sample `Aᵀ`, the transpose of the public matrix `A ∈ Z_q^{C×N}`, in
/// row-major shape `(N × C)`.
///
/// # Rationale
///
/// The backend never needs `A` in `C×N` form: the client precompute
/// `A·s` and the server `D·A` both consume `A` column-by-column, i.e.
/// row-by-row of `Aᵀ`. Storing `Aᵀ` (row `k` = column `k` of `A`) keeps
/// every access sequential. `A` is uniform, so `Aᵀ` is too — we just
/// draw `N·C` successive `u32`s.
///
/// # Returns
///
/// `Vec<u32>` of length `lwe_dim · cols`; `out[k*cols + c] = A[c][k]`.
/// Deterministic in `(seed, cols, lwe_dim)`.
pub fn sample_a_transposed(seed: &[u8; 16], cols: u32, lwe_dim: u32) -> Vec<u32> {
    let len = (cols as usize)
        .checked_mul(lwe_dim as usize)
        .expect("A dimensions overflow usize");
    let mut rng = rng_from_seed(seed);
    let mut out = vec![0u32; len];
    for cell in &mut out {
        *cell = rng.next_u32();
    }
    out
}

/// Fill `dst` with uniform `Z_q` samples (`q = 2³²`) — the SimplePIR
/// secret distribution `s ← Z_q^N`.
pub fn sample_uniform_zq_into<R: RngCore>(rng: &mut R, dst: &mut [u32]) {
    for cell in dst.iter_mut() {
        *cell = rng.next_u32();
    }
}

/// Fill `dst` with discrete-Gaussian samples of standard deviation
/// `sigma`, encoded as two's-complement `u32` (so `dst[i] as i32` is the
/// signed value and wrapping `Z_{2^32}` arithmetic is correct in the
/// matvec). SimplePIR error distribution (§4.2, `σ = 6.4`).
pub fn sample_discrete_gaussian_into<R: RngCore>(rng: &mut R, sigma: f64, dst: &mut [u32]) {
    debug_assert!(sigma > 0.0 && sigma.is_finite());
    let two_pi = std::f64::consts::TAU;
    let mut i = 0;
    while i < dst.len() {
        let u1 = uniform_unit_open(rng);
        let u2 = uniform_unit(rng);
        let radius = sigma * (-2.0 * u1.ln()).sqrt();
        let theta = two_pi * u2;
        let z1 = radius * theta.cos();
        let z2 = radius * theta.sin();

        dst[i] = f64_round_to_i32(z1) as u32;
        i += 1;
        if i < dst.len() {
            dst[i] = f64_round_to_i32(z2) as u32;
            i += 1;
        }
    }
}

/// Uniform `[0, 1)` double from the high 53 bits of one `next_u64`.
#[inline]
fn uniform_unit<R: RngCore>(rng: &mut R) -> f64 {
    let u = rng.next_u64() >> 11;
    (u as f64) * (1.0_f64 / ((1u64 << 53) as f64))
}

/// Uniform `(0, 1)` double — strictly positive, for the `ln(u1)` factor.
#[inline]
fn uniform_unit_open<R: RngCore>(rng: &mut R) -> f64 {
    loop {
        let u = rng.next_u64() >> 11;
        if u != 0 {
            return (u as f64) * (1.0_f64 / ((1u64 << 53) as f64));
        }
    }
}

/// Round a finite `f64` to nearest `i32`, saturating out of range.
#[inline]
fn f64_round_to_i32(x: f64) -> i32 {
    let r = x.round();
    if r >= i32::MAX as f64 {
        i32::MAX
    } else if r <= i32::MIN as f64 {
        i32::MIN
    } else {
        r as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_a_determinism_and_shape() {
        let seed = [0x42u8; 16];
        let a1 = sample_a_transposed(&seed, 7, 1275);
        let a2 = sample_a_transposed(&seed, 7, 1275);
        assert_eq!(a1, a2);
        assert_eq!(a1.len(), 7 * 1275);
    }

    #[test]
    fn gaussian_first_two_moments() {
        let sigma = 6.4_f64;
        let n = 20_000usize;
        let mut rng = rng_from_seed(&[0xCDu8; 16]);
        let mut dst = vec![0u32; n];
        sample_discrete_gaussian_into(&mut rng, sigma, &mut dst);

        let signed: Vec<f64> = dst.iter().map(|&x| (x as i32) as f64).collect();
        let mean: f64 = signed.iter().sum::<f64>() / n as f64;
        let var: f64 = signed.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        assert!(mean.abs() < 5.0 * sigma / (n as f64).sqrt(), "mean {mean}");
        let var_std = sigma * sigma * (2.0 / n as f64).sqrt();
        assert!((var - sigma * sigma).abs() < 5.0 * var_std, "var {var}");
    }

    #[test]
    fn gaussian_saturation_branch_untaken() {
        let mut rng = rng_from_seed(&[0xEFu8; 16]);
        let mut dst = vec![0u32; 50_000];
        sample_discrete_gaussian_into(&mut rng, 6.4, &mut dst);
        for &x in &dst {
            assert!(
                (x as i32).abs() < 200,
                "sample {} out of σ=6.4 range",
                x as i32
            );
        }
    }
}
