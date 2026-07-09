//! KPIR^index — keyword PIR from an approximate key-to-index mapping
//! (Hao et al., USENIX Security 2025, §6; `mpc4j` `SimplePgmCpKsPir`).
//!
//! # Overview
//!
//! Core-logic-only Rust re-implementation of the `KPIR^index` construction
//! for a fair head-to-head benchmark against ChalametPIR and RisePIR
//! (CANS2026 Table 3). No networking, serialization, or client-server
//! transport — only the cryptographic/algorithmic pipeline.
//!
//! The scheme reduces keyword PIR to standard index PIR via two pieces:
//!
//! - [`pla`] — an approximate key-to-index map (optimal ε-PLA / PGM) that
//!   turns a keyword into a rank within `ε` of the true one.
//! - [`simplepir`] — the Row-KOPIR (SimplePIR) index-PIR backend that
//!   privately retrieves the covering database column.
//!
//! [`scheme`] glues them: sort + hash keys, learn the map, repetition-
//! encode the key-value pairs into an `R × C` `Z_p` matrix (records
//! bit-packed into cells at the adaptively chosen plaintext width), then
//! query / answer / recover with a single Row-KOPIR invocation. See
//! [`params`] for the exact matrix geometry (which, at `plaintext_bits =
//! 8`, reproduces the reference communication numbers to the byte).
//!
//! ```
//! use kpir_index::scheme::{build_synthetic, hash_key, synthetic_value};
//! use simplepir::SimpleConfig;
//! use rand::{rngs::StdRng, SeedableRng};
//!
//! let mut rng = StdRng::seed_from_u64(0);
//! let config = SimpleConfig::with_lwe_dim(256); // small N for a fast doctest
//! let (server, client) = build_synthetic(1000, 32, 4, &config, [0u8; 16], &mut rng);
//!
//! let key = 42u64.to_le_bytes();
//! let qu = client.query(&key, &mut rng);       // keyword → private column query
//! let ans = server.answer(&qu);                // server: ans = D · qu
//! let value = client.recover(&key, &ans).unwrap();
//!
//! let mut want = vec![0u8; 32];
//! synthetic_value(hash_key(&key), &mut want);
//! assert_eq!(value, want);
//! ```
#![warn(missing_docs)]

pub mod params;
pub mod pla;
pub mod scheme;

pub use params::{MatrixShape, DEFAULT_EPSILON, FINGERPRINT_BYTES};
pub use scheme::{
    build_from_pairs, build_synthetic, hash_key, synthetic_value, KpirClient, KpirServer,
};
