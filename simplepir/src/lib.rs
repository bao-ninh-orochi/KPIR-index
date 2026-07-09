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
//! The database is a byte matrix `D ∈ Z_p^{R×C}` (`p = 256`). A query
//! privately selects one **column** `col ∈ [C)`; the answer returns the
//! whole column — all `R` entries — which the keyword layer then scans.
//! This is the transpose of the paper's row-retrieval presentation
//! (Figure 2) and matches the `mpc4j` reference (`SimpleCpIdxPir` /
//! `SimplePgmCpKsPir`, which encrypt a column selector).
//!
//! # Protocol (all arithmetic mod `q = 2^32`, native `u32` wraparound)
//!
//! Public matrix `A ∈ Z_q^{C×N}` is expanded from a 16-byte seed.
//! `Δ = q/p = 2^(32 − plaintext_bits)`.
//!
//! - **Setup**   `hint = D · A ∈ Z_q^{R×N}` (client downloads it).
//! - **KeyGen**  secret `s ← Z_q^N`; precompute `a_s = A·s ∈ Z_q^C`,
//!               `h_s = hint·s ∈ Z_q^R`.
//! - **Query**   `qu = a_s + e + Δ·u_col ∈ Z_q^C`, error `e ← χ^C`.
//! - **Answer**  `ans = D · qu ∈ Z_q^R`.
//! - **Recover** `d = ans − h_s`; round each entry
//!               `Δ·m + noise → m ∈ [0,p)`. `d` is exactly column `col`.
//!
//! # Design / architecture
//!
//! - `q = 2^32` is implicit via native `u32` wraparound.
//! - The DB is stored **transposed** (`C × R`, row-major, `u8`) so that
//!   the hot answer path `D·qu` is a single left-multiply through the
//!   shared [`matvec`] kernel, and `u8` storage keeps the (potentially
//!   multi-GiB) matrix compact.
//! - No threads, no explicit SIMD, no external linear-algebra crate — the
//!   math is hand-rolled `u32`, matching the RisePIR reference repo so
//!   head-to-head benchmark numbers are measured under the same model.
//!
//! # Related files
//!
//! - `backend.rs` — [`SimplePirServer`] / [`SimplePirClient`] / [`Hint`].
//! - `matvec.rs`  — the shared `acc += qᵀ·D` kernel.
//! - `sampler.rs` — `A` expansion, uniform-`Z_q` secret, discrete Gaussian.
//! - `arith.rs`   — the recover rounding `Round_Δ`.
//! - `params.rs`  — [`SimpleParams`] / [`SimpleConfig`].
#![warn(missing_docs)]

mod arith;
mod backend;
mod matvec;
mod params;
mod sampler;

pub use backend::{Hint, SimplePirClient, SimplePirServer};
pub use params::{SimpleConfig, SimpleParams};
