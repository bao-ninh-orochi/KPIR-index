//! Row-KOPIR (SimplePIR) server, client, and hint.
//!
//! # Purpose
//!
//! The three public types of the crate. [`SimplePirServer`] owns the
//! column-major database and produces the [`Hint`] and per-query answers;
//! [`SimplePirClient`] holds the LWE secret material and builds queries /
//! recovers columns.
//!
//! # Layout
//!
//! The logical database `D` is `R × C` (`R` = response dimension,
//! `C` = query dimension). It is stored **transposed** as `tdb`, a
//! `C × R` row-major `u32` buffer (`tdb[c*R + r] = D[r][c]`, each cell in
//! `[0, p)`), so the hot answer path `ans = D·qu` is one left-multiply
//! `matvec(ans, tdb, qu)`.
//! The public matrix `A ∈ Z_q^{C×N}` is expanded on demand from the seed
//! in its transposed `N × C` form (`a_nc`), and never stored on the
//! server or shipped to the client.
//!
//! # Related files
//!
//! - `matvec.rs`  — the `acc += qᵀ·D` kernel used by every method here.
//! - `sampler.rs` — `A` expansion, secret, and error samplers.
//! - `arith.rs`   — the recover rounding `Round_Δ`.

use rand::RngCore;

use crate::matvec::matvec_accumulate;
use crate::params::SimpleParams;
use crate::sampler::{sample_a_transposed, sample_discrete_gaussian_into, sample_uniform_zq_into};

/// The setup hint `hint = D · A ∈ Z_q^{R×N}`, downloaded once by the
/// client. Stored transposed as `hint_t` (`N × R`, row-major) so the
/// client precompute `hint·s` is a left-multiply.
#[derive(Clone, Debug)]
pub struct Hint {
    /// `hint_t[k*rows + r] = hint[r][k] = (D·A)[r][k]`. Length `lwe_dim·rows`.
    hint_t: Vec<u32>,
    /// Response dimension `R`.
    rows: usize,
    /// Query dimension `C`.
    cols: usize,
    /// LWE dimension `N`.
    lwe_dim: usize,
    /// Public seed used to expand `A` (the client re-expands, not shipped).
    seed: [u8; 16],
}

impl Hint {
    /// Response dimension `R`.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Query dimension `C`.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// On-wire size in bytes: `R·N` fixed-width `u32` cells. The public
    /// matrix `A` is regenerated from the seed by the client, so it is
    /// excluded (matching the SimplePIR/RisePIR accounting).
    #[inline]
    pub fn wire_byte_size(&self) -> usize {
        self.hint_t.len() * core::mem::size_of::<u32>()
    }
}

/// Row-KOPIR (SimplePIR) server: owns the transposed database.
pub struct SimplePirServer {
    /// Transposed database `tdb[c*R + r] = D[r][c]`, length `C·R`, each
    /// cell a `Z_p` value in `[0, p)` (high `32 − plaintext_bits` bits zero).
    tdb: Vec<u32>,
    rows: usize,
    cols: usize,
    params: SimpleParams,
}

impl SimplePirServer {
    /// Build a server from a database already stored **transposed**
    /// (`tdb`, `C × R` row-major, `tdb[c*rows + r] = D[r][c]`).
    ///
    /// The keyword layer builds `tdb` directly (one `u32` `Z_p` cell per
    /// entry), avoiding a second full-matrix copy.
    ///
    /// # Constraints
    ///
    /// Panics if `tdb.len() != rows * cols`. **Correctness guard:** panics
    /// if the plaintext width is unsafe for this geometry — i.e. the answer
    /// matvec sums `cols` cells and the chosen `p = 2^plaintext_bits`
    /// violates the SimplePIR decode bound ([`crate::noise_bound_satisfied`]).
    /// This backstops the adaptive operating-point selection: a bench (or
    /// any caller) that hands in a too-wide `p` fails loudly rather than
    /// silently mis-decoding. In debug builds also checks every cell lies in
    /// `[0, p)`.
    pub fn from_transposed_db(
        tdb: Vec<u32>,
        rows: usize,
        cols: usize,
        params: SimpleParams,
    ) -> Self {
        assert_eq!(tdb.len(), rows * cols, "tdb shape mismatch");
        assert!(
            crate::params::noise_bound_satisfied(params.plaintext_bits, cols as u32, params.sigma),
            "unsafe SimplePIR parameters: p = 2^{} over C = {cols} summed cells \
             violates the δ = 2⁻⁴⁰ decode bound (σ = {}); choose a smaller plaintext_bits",
            params.plaintext_bits,
            params.sigma,
        );
        debug_assert!(
            tdb.iter().all(|&c| c < params.plaintext_modulus()),
            "database cell exceeds plaintext modulus p = 2^{}",
            params.plaintext_bits,
        );
        Self {
            tdb,
            rows,
            cols,
            params,
        }
    }

    /// Response dimension `R`.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Query dimension `C`.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// The LWE parameters this server was built with.
    #[inline]
    pub fn params(&self) -> &SimpleParams {
        &self.params
    }

    /// Compute the setup hint `hint = D·A ∈ Z_q^{R×N}` (returned
    /// transposed). One-time preprocessing: `Θ(R·C·N)` multiply-adds.
    pub fn setup(&self) -> Hint {
        let n = self.params.lwe_dim as usize;
        // Aᵀ in (N × C) row-major: row k = column k of A.
        let a_nc = sample_a_transposed(&self.params.seed, self.cols as u32, self.params.lwe_dim);
        let mut hint_t = vec![0u32; n * self.rows];
        for k in 0..n {
            let a_col = &a_nc[k * self.cols..(k + 1) * self.cols];
            let hint_row = &mut hint_t[k * self.rows..(k + 1) * self.rows];
            // hint_t[k][r] = Σ_c A[c][k]·D[r][c] = (D·A)[r][k].
            matvec_accumulate(hint_row, &self.tdb, a_col);
        }
        Hint {
            hint_t,
            rows: self.rows,
            cols: self.cols,
            lwe_dim: n,
            seed: self.params.seed,
        }
    }

    /// Answer a query: `ans = D · qu ∈ Z_q^R` (the selected column across
    /// all `R` response cells). The measured online server cost:
    /// `Θ(R·C)` multiply-adds.
    ///
    /// # Constraints
    ///
    /// Panics if `qu.len() != C`.
    pub fn answer(&self, qu: &[u32]) -> Vec<u32> {
        assert_eq!(qu.len(), self.cols, "query length must equal C");
        let mut ans = vec![0u32; self.rows];
        matvec_accumulate(&mut ans, &self.tdb, qu);
        ans
    }
}

/// Row-KOPIR (SimplePIR) client: holds the per-secret precompute
/// `a_s = A·s` and `h_s = hint·s`, enough to build queries and recover.
pub struct SimplePirClient {
    /// `a_s = A·s ∈ Z_q^C` — the query base.
    a_s: Vec<u32>,
    /// `h_s = hint·s ∈ Z_q^R` — the recover offset.
    h_s: Vec<u32>,
    rows: usize,
    cols: usize,
    params: SimpleParams,
}

impl SimplePirClient {
    /// Set up a client for a fresh LWE secret `s ← Z_q^N`.
    ///
    /// Re-expands `A` from the hint's seed and precomputes
    /// `a_s = A·s` (`Θ(C·N)`) and `h_s = hint·s` (`Θ(R·N)`). Security
    /// note: one secret must back one query (fresh `s` per query); call
    /// this per query when measuring the full online client cost.
    pub fn new<R: RngCore>(hint: &Hint, params: SimpleParams, rng: &mut R) -> Self {
        assert_eq!(hint.lwe_dim, params.lwe_dim as usize, "lwe_dim mismatch");
        let n = params.lwe_dim as usize;
        let cols = hint.cols;
        let rows = hint.rows;

        let mut s = vec![0u32; n];
        sample_uniform_zq_into(rng, &mut s);

        // a_s = A·s ; a_nc = Aᵀ in (N × C) row-major (row k = column k of A).
        let a_nc = sample_a_transposed(&hint.seed, cols as u32, params.lwe_dim);
        let mut a_s = vec![0u32; cols];
        matvec_accumulate(&mut a_s, &a_nc, &s);

        // h_s = hint·s ; hint_t is (N × R) row-major.
        let mut h_s = vec![0u32; rows];
        matvec_accumulate(&mut h_s, &hint.hint_t, &s);

        Self {
            a_s,
            h_s,
            rows,
            cols,
            params,
        }
    }

    /// Fast client set-up for a **co-located** server (benchmarks and
    /// correctness checks), avoiding the full `Θ(R·C·N)` hint.
    ///
    /// Computes `a_s = A·s` (`Θ(C·N)`) and then the recover offset
    /// `h_s = hint·s = (D·A)·s = D·(A·s) = server.answer(a_s)` in one
    /// `Θ(R·C)` matvec through the local server, instead of materialising
    /// the whole `hint = D·A`. The result is bit-identical to
    /// [`SimplePirClient::new`] fed the real hint (`h_s = hint·s`); the
    /// hint's on-wire size is still `R·N·4` and is reported analytically.
    /// Only valid when the client and server share a process.
    pub fn with_local_server<R: RngCore>(server: &SimplePirServer, rng: &mut R) -> Self {
        let params = server.params.clone();
        let n = params.lwe_dim as usize;
        let (rows, cols) = (server.rows, server.cols);

        let mut s = vec![0u32; n];
        sample_uniform_zq_into(rng, &mut s);

        let a_nc = sample_a_transposed(&params.seed, cols as u32, params.lwe_dim);
        let mut a_s = vec![0u32; cols];
        matvec_accumulate(&mut a_s, &a_nc, &s);

        // h_s = hint·s = D·(A·s) = D·a_s, one matvec via the local server.
        let h_s = server.answer(&a_s);

        Self {
            a_s,
            h_s,
            rows,
            cols,
            params,
        }
    }

    /// Query dimension `C` (query wire length in `u32`).
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// Response dimension `R`.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Build a query selecting column `col`: `qu = a_s + e + Δ·u_col`.
    ///
    /// # Constraints
    ///
    /// Panics if `col >= C`.
    pub fn query<R: RngCore>(&self, col: usize, rng: &mut R) -> Vec<u32> {
        assert!(col < self.cols, "column {col} out of range {}", self.cols);
        let mut e = vec![0u32; self.cols];
        sample_discrete_gaussian_into(rng, self.params.sigma, &mut e);
        let mut qu = self.a_s.clone();
        for (q, ei) in qu.iter_mut().zip(&e) {
            *q = q.wrapping_add(*ei);
        }
        qu[col] = qu[col].wrapping_add(self.params.delta());
        qu
    }

    /// Recover the selected column: `d = ans − h_s`, rounded cell-wise to
    /// its plaintext `Z_p` value. Returns the `R` cells of the column (each
    /// in `[0, p)`); the keyword layer unpacks these into records.
    ///
    /// # Constraints
    ///
    /// Panics if `ans.len() != R`.
    pub fn recover(&self, ans: &[u32]) -> Vec<u32> {
        assert_eq!(ans.len(), self.rows, "answer length must equal R");
        ans.iter()
            .zip(&self.h_s)
            .map(|(&a, &h)| crate::arith::round_q_to_p(a.wrapping_sub(h), &self.params))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// A random column is recovered exactly through the full
    /// setup → keygen → query → answer → recover pipeline.
    #[test]
    fn recovers_selected_column() {
        let mut rng = StdRng::seed_from_u64(0xA11CE);
        let (rows, cols) = (170usize, 211usize);
        // Random Z_p DB (p = 256) in transposed layout, each cell in [0, p).
        let mut tdb = vec![0u32; rows * cols];
        for c in tdb.iter_mut() {
            *c = rng.next_u32() & 0xff;
        }
        // Reference D[r][c] from tdb[c*rows + r].
        let d = |r: usize, c: usize| tdb[c * rows + r];

        let params = SimpleParams::new(512, 8, 6.4, [7u8; 16]);
        let server = SimplePirServer::from_transposed_db(tdb.clone(), rows, cols, params.clone());
        let hint = server.setup();
        let client = SimplePirClient::new(&hint, params, &mut rng);

        for &col in &[0usize, 1, 57, 128, cols - 1] {
            let qu = client.query(col, &mut rng);
            let ans = server.answer(&qu);
            let got = client.recover(&ans);
            let want: Vec<u32> = (0..rows).map(|r| d(r, col)).collect();
            assert_eq!(got, want, "column {col} mismatch");
        }
    }

    /// Recovers a wider plaintext (`p = 2^10`) exactly — exercises the
    /// adaptive-width path, not just the `mpc4j` byte case.
    #[test]
    fn recovers_wide_plaintext() {
        let mut rng = StdRng::seed_from_u64(0xD00D);
        let (rows, cols) = (64usize, 96usize);
        let pb = 10u32;
        let p = 1u32 << pb;
        let mut tdb = vec![0u32; rows * cols];
        for c in tdb.iter_mut() {
            *c = rng.next_u32() % p;
        }
        let d = |r: usize, c: usize| tdb[c * rows + r];
        let params = SimpleParams::new(512, pb, 6.4, [2u8; 16]);
        let server = SimplePirServer::from_transposed_db(tdb.clone(), rows, cols, params.clone());
        let hint = server.setup();
        let client = SimplePirClient::new(&hint, params, &mut rng);
        for &col in &[0usize, 5, cols - 1] {
            let qu = client.query(col, &mut rng);
            let ans = server.answer(&qu);
            let got = client.recover(&ans);
            let want: Vec<u32> = (0..rows).map(|r| d(r, col)).collect();
            assert_eq!(got, want, "wide column {col} mismatch");
        }
    }

    /// The correctness guard fires when the plaintext width is too large
    /// for the summed dimension.
    #[test]
    #[should_panic(expected = "unsafe SimplePIR parameters")]
    fn guard_rejects_unsafe_plaintext_bits() {
        let (rows, cols) = (16usize, 20000usize);
        let tdb = vec![0u32; rows * cols];
        // p = 2^20 over 20k summed cells at σ = 6.4 blows past Δ/2.
        let params = SimpleParams::new(1275, 20, 6.4, [0u8; 16]);
        let _ = SimplePirServer::from_transposed_db(tdb, rows, cols, params);
    }

    /// The fast co-located client (`with_local_server`) recovers exactly
    /// what the real-hint client (`new`) does — `h_s = hint·s = D·(A·s)`.
    #[test]
    fn local_server_client_matches_hint_client() {
        let mut rng = StdRng::seed_from_u64(0xBEE5);
        let (rows, cols) = (48usize, 71usize);
        let mut tdb = vec![0u32; rows * cols];
        for c in tdb.iter_mut() {
            *c = rng.next_u32() & 0xff;
        }
        let params = SimpleParams::new(384, 8, 6.4, [5u8; 16]);
        let server = SimplePirServer::from_transposed_db(tdb, rows, cols, params.clone());

        // Same secret stream for both clients → identical a_s, h_s.
        let mut r1 = StdRng::seed_from_u64(0x1234);
        let mut r2 = StdRng::seed_from_u64(0x1234);
        let hint = server.setup();
        let full = SimplePirClient::new(&hint, params, &mut r1);
        let fast = SimplePirClient::with_local_server(&server, &mut r2);
        assert_eq!(full.a_s, fast.a_s, "a_s differs");
        assert_eq!(full.h_s, fast.h_s, "h_s differs");
    }

    /// The hint excludes `A`: its wire size is exactly `R·N·4`.
    #[test]
    fn hint_wire_size() {
        let (rows, cols, n) = (10usize, 12usize, 64u32);
        let tdb = vec![0u32; rows * cols];
        let params = SimpleParams::new(n, 8, 6.4, [0u8; 16]);
        let server = SimplePirServer::from_transposed_db(tdb, rows, cols, params);
        let hint = server.setup();
        assert_eq!(hint.wire_byte_size(), rows * n as usize * 4);
    }
}
