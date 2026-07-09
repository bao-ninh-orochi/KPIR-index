//! Recover rounding for the SimplePIR backend.
//!
//! # Purpose
//!
//! After the client forms `d = ans − h_s ∈ Z_q^R`, each entry is
//! `Δ·m + noise (mod q)` for a plaintext `m ∈ [0, p)` and small `|noise| ≪ Δ/2`.
//! [`round_q_to_p`] recovers `m` by rounding to the nearest multiple of
//! `Δ = 2^(32 − plaintext_bits)` and dividing — i.e. `Round_Δ`.
//!
//! # Rationale
//!
//! `round(x) = ((x + Δ/2) >> (32 − plaintext_bits)) mod p`. Adding `Δ/2`
//! then taking the top `plaintext_bits` bits is round-to-nearest; the
//! `wrapping_add` and mask handle the modular boundary (e.g. `m = p−1`
//! with positive noise). This is the exact `mpc4j` recovery
//! (`(element & 0x00800000) > 0 ? (element>>>24)+1 : element>>>24` for
//! `plaintext_bits = 8`), generalised to any width.

use crate::params::SimpleParams;

/// Round a noisy `Z_q` value back to its plaintext `m ∈ [0, p)`.
#[inline]
pub(crate) fn round_q_to_p(x: u32, params: &SimpleParams) -> u32 {
    let shift = u32::BITS - params.plaintext_bits; // = log2(Δ)
    let half = 1u32 << (shift - 1); // Δ/2
    (x.wrapping_add(half) >> shift) & (params.plaintext_modulus() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(plaintext_bits: u32) -> SimpleParams {
        SimpleParams::new(1275, plaintext_bits, 6.4, [0u8; 16])
    }

    /// Exact multiples of Δ round to their plaintext, for every m ∈ [0,p).
    #[test]
    fn exact_multiples_roundtrip() {
        let p = params(8);
        let delta = p.delta();
        for m in 0..p.plaintext_modulus() {
            assert_eq!(round_q_to_p(m.wrapping_mul(delta), &p), m);
        }
    }

    /// Small ± noise (up to just under Δ/2) still decodes correctly,
    /// including the modular wrap at m = p−1.
    #[test]
    fn tolerates_noise_below_half_delta() {
        let p = params(8);
        let delta = p.delta();
        let noises: [i64; 5] = [
            -(delta as i64 / 2 - 1),
            -1000,
            0,
            1000,
            delta as i64 / 2 - 1,
        ];
        for m in [0u32, 1, 127, 200, 255] {
            for &noise in &noises {
                let x = (m.wrapping_mul(delta) as i64 + noise) as u32;
                assert_eq!(round_q_to_p(x, &p), m, "m={m} noise={noise}");
            }
        }
    }

    /// Matches the mpc4j byte recovery for plaintext_bits = 8.
    #[test]
    fn matches_mpc4j_byte_recovery() {
        let p = params(8);
        for raw in [0u32, 0x0080_0000, 0x00FF_FFFF, 0x7F80_0000, 0xFF80_0000] {
            let mpc4j = if raw & 0x0080_0000 > 0 {
                (raw >> 24).wrapping_add(1) & 0xff
            } else {
                (raw >> 24) & 0xff
            };
            assert_eq!(round_q_to_p(raw, &p), mpc4j, "raw={raw:#010x}");
        }
    }
}
