//! Row-KOPIR (SimplePIR) — single-server, LWE-based **index** PIR backend.
//!
//! # Purpose
//!
//! This crate implements the row-retrieval PIR abstraction *Row-KOPIR*
//! that Hao et al. (USENIX Security 2025, "Practical Keyword Private
//! Information Retrieval from Key-to-Index Mappings") build their keyword
//! PIR on, instantiated from SimplePIR (Henzinger et al., USENIX Security
//! 2023). It is the standard index-PIR primitive underneath `KPIR^index`
//! (see the [`kpir-index`](../kpir_index/index.html) crate).
//!
//! The database is a matrix `D ∈ Z_p^{C×R}` of `u32` cells, each in
//! `[0, p)` where `p = 2^plaintext_bits` is chosen adaptively per geometry
//! (see [`noise_bound_satisfied`]). Following the FrodoPIR convention (the
//! one RisePIR uses), a query privately selects one **row** `col ∈ [C)`
//! via the left-multiply `ans = qu·D`; the answer returns that row — all
//! `R` cells — which the keyword layer then scans. This matches the
//! `mpc4j` reference (`SimpleCpIdxPir` / `SimplePgmCpKsPir`).
//!
//! # Protocol (all arithmetic mod `q = 2^32`, native `u32` wraparound)
//!
//! Public matrix `A ∈ Z_q^{C×N}` is expanded from a 16-byte seed.
//! `Δ = q/p = 2^(32 − plaintext_bits)`.
//!
//! - **Setup**   `H = Aᵀ · D ∈ Z_q^{N×R}` (client downloads it).
//! - **KeyGen**  secret `s ← Z_q^N`; precompute `a_s = A·s ∈ Z_q^C`,
//!               `h_s = sᵀ·H ∈ Z_q^R`.
//! - **Query**   `qu = a_s + e + Δ·u_col ∈ Z_q^C`, error `e ← χ^C`.
//! - **Answer**  `ans = qu · D ∈ Z_q^R`.
//! - **Recover** `d = ans − h_s`; round each entry
//!               `Δ·m + noise → m ∈ [0,p)`. `d` is exactly row `col`.
//!
//! # Design / architecture
//!
//! - `q = 2^32` is implicit via native `u32` wraparound.
//! - The DB is stored **row-major** over the `C×R` matrix `D` (`db[c*R + r]`,
//!   `u32`, each query-row contiguous) — RisePIR's layout. The answer
//!   `qu·D` reduces over the strided query dimension through the shared
//!   register-blocked `matvec` kernel; measured single-threaded on Apple M1
//!   this beats ChalametPIR's column-major dot product (see `matvec.rs`).
//! - No threads, no explicit SIMD, no external linear-algebra crate — the
//!   math is hand-rolled `u32`, matching the RisePIR / ChalametPIR
//!   reference repos so head-to-head benchmark numbers are measured under
//!   the same model.
//!
//! # Related files
//!
//! - `backend.rs` — [`SimplePirServer`] / [`SimplePirClient`] / [`Hint`].
//! - `matvec.rs`  — the shared register-blocked `acc += qᵀ·D` kernel behind
//!   `answer` (`qu·D`), `setup` (`Aᵀ·D`), and the `A·s` / `sᵀ·H` precompute.
//! - `sampler.rs` — `A` expansion, uniform-`Z_q` secret, discrete Gaussian.
//! - `arith.rs`   — the recover rounding `Round_Δ`.
//! - `params.rs`  — [`SimpleParams`] / [`SimpleConfig`] / [`noise_bound_satisfied`].
#![warn(missing_docs)]

mod arith;
mod backend;
mod matvec;
mod params;
mod sampler;

pub use backend::{Hint, SimplePirClient, SimplePirServer};
pub use params::{noise_bound_satisfied, SimpleConfig, SimpleParams, MAX_PLAINTEXT_BITS};
