//! The KPIR^index keyword-PIR construction (Hao et al. §6.2, Figure 8;
//! `mpc4j` `SimplePgmCpKsPir`).
//!
//! # Purpose
//!
//! Ties the approximate key-to-index map ([`crate::pla`]) to the
//! Row-KOPIR backend ([`simplepir`]). Setup sorts the hashed keys, learns
//! the PLA map, and repetition-encodes the key-value pairs into the
//! `R × C` byte matrix; a query extracts the approximate rank, selects the
//! covering column, and privately retrieves it; recover scans the `≤ 2ε+3`
//! candidate entries of that column for the fingerprint match.
//!
//! # Encoding (closed form of `mpc4j`'s column-major placement)
//!
//! Sort the `n` pairs by hashed key. The matrix cell at column `col`,
//! plane-row `j` holds the sorted entry
//! `fingerprint(k_g) ‖ value_g` where
//! ```text
//! g = col · data_rows + (j − ε − 1),   0 ≤ g < n   (else ⊥ = 0xFF)
//! ```
//! Each entry spans `partition = ε_bytes + ℓ` bytes across the response
//! dimension `R = rows · partition`. The `ε+1` top / `ε+2` bottom
//! boundary rows thus automatically hold the neighbouring columns' edge
//! entries, so the true entry — within `ε+1` of the PLA estimate — always
//! lands inside the single retrieved column.
//!
//! # Query / Recover
//!
//! - `Query(k)`: `pos = map.extract(hash(k))`; `col = pos / data_rows`;
//!   Row-KOPIR query for column `col`.
//! - `Recover(k)`: Row-KOPIR recover → the `rows` candidate entries;
//!   return the value whose 8-byte fingerprint equals `hash(k)`, else `⊥`.

use rand::RngCore;
use xxhash_rust::xxh3::xxh3_64;

use simplepir::{Hint, SimpleConfig, SimplePirClient, SimplePirServer};

use crate::params::{MatrixShape, FINGERPRINT_BYTES};
use crate::pla::KeyIndexMap;

/// Hash an arbitrary keyword to the uniform 64-bit value used both as the
/// PLA index key and as the stored 8-byte fingerprint.
///
/// `mpc4j` uses `SHA-256` truncated to 8 bytes; we use `xxh3` (a fast
/// uniform 64-bit hash) — the choice is immaterial to the scheme, which
/// only needs a near-uniform, near-injective map so the PLA learns few
/// segments and fingerprints rarely collide.
#[inline]
pub fn hash_key(key: &[u8]) -> u64 {
    xxh3_64(key)
}

/// Deterministic synthetic value used by the benchmark database: a
/// splitmix64 byte stream seeded by the entry's fingerprint. Recomputable
/// from the queried key (`hash_key(key)`), so recovered values can be
/// verified without storing the plaintext database.
pub fn synthetic_value(seed: u64, out: &mut [u8]) {
    let mut x = seed;
    let mut i = 0;
    while i < out.len() {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let bytes = z.to_le_bytes();
        let take = (out.len() - i).min(8);
        out[i..i + take].copy_from_slice(&bytes[..take]);
        i += take;
    }
}

/// KPIR^index server: owns the encoded Row-KOPIR database.
pub struct KpirServer {
    inner: SimplePirServer,
    shape: MatrixShape,
}

impl KpirServer {
    /// The encoded-database geometry.
    #[inline]
    pub fn shape(&self) -> &MatrixShape {
        &self.shape
    }

    /// Answer a query: the online server operation, `ans = D·qu`.
    #[inline]
    pub fn answer(&self, qu: &[u32]) -> Vec<u32> {
        self.inner.answer(qu)
    }

    /// Compute the one-time setup hint `hint = D·A` for the client.
    #[inline]
    pub fn setup_hint(&self) -> Hint {
        self.inner.setup()
    }
}

/// KPIR^index client: the downloaded PLA map plus one Row-KOPIR secret.
pub struct KpirClient {
    inner: SimplePirClient,
    map: KeyIndexMap,
    shape: MatrixShape,
    hint_wire_bytes: usize,
}

impl KpirClient {
    /// Build a query for `key`: extract its approximate rank, map it to a
    /// column, and privately retrieve that column.
    pub fn query<R: RngCore>(&self, key: &[u8], rng: &mut R) -> Vec<u32> {
        let pos = self.map.extract(hash_key(key));
        let col = (pos / self.shape.data_rows).min(self.shape.columns - 1);
        self.inner.query(col, rng)
    }

    /// Recover the value for `key` from a server answer, or `None` (`⊥`)
    /// if `key` is absent. Scans the retrieved column's candidate entries
    /// for the fingerprint match.
    pub fn recover(&self, key: &[u8], ans: &[u32]) -> Option<Vec<u8>> {
        let want = hash_key(key);
        let bytes = self.inner.recover(ans);
        let partition = self.shape.partition;
        for j in 0..self.shape.rows {
            let base = j * partition;
            let fp = u64::from_le_bytes(bytes[base..base + FINGERPRINT_BYTES].try_into().unwrap());
            if fp == want {
                return Some(bytes[base + FINGERPRINT_BYTES..base + partition].to_vec());
            }
        }
        None
    }

    /// The encoded-database geometry.
    #[inline]
    pub fn shape(&self) -> &MatrixShape {
        &self.shape
    }

    /// Client key-to-index map size in bytes (`num_segments · 24`).
    #[inline]
    pub fn map_wire_bytes(&self) -> usize {
        self.map.wire_byte_size()
    }

    /// Number of PLA segments in the client map.
    #[inline]
    pub fn num_segments(&self) -> usize {
        self.map.num_segments()
    }

    /// One-time setup hint download size in bytes (`R · N · 4`).
    #[inline]
    pub fn hint_wire_bytes(&self) -> usize {
        self.hint_wire_bytes
    }
}

/// Repetition-encode sorted entries into the transposed Row-KOPIR matrix
/// `tdb` (`C × R`, `tdb[col*R + r] = D[r][col]`). `fill_value(g, buf)`
/// writes the `value_bytes`-byte value of the `g`-th sorted entry.
fn encode_tdb(
    sorted_hashes: &[u64],
    shape: &MatrixShape,
    mut fill_value: impl FnMut(usize, &mut [u8]),
) -> Vec<u8> {
    let n = shape.n;
    let r = shape.response_dim();
    let partition = shape.partition;
    let data_rows = shape.data_rows as i64;
    let eps = shape.epsilon as i64;

    // ⊥ default; boundary cells (g out of range) stay 0xFF.
    let mut tdb = vec![0xFFu8; shape.columns * r];
    let mut valbuf = vec![0u8; shape.value_bytes];

    for col in 0..shape.columns {
        let col_base = col * r;
        for j in 0..shape.rows {
            let g = col as i64 * data_rows + (j as i64 - eps - 1);
            if g < 0 || g as usize >= n {
                continue;
            }
            let g = g as usize;
            let base = col_base + j * partition;
            tdb[base..base + FINGERPRINT_BYTES].copy_from_slice(&sorted_hashes[g].to_le_bytes());
            fill_value(g, &mut valbuf);
            tdb[base + FINGERPRINT_BYTES..base + partition].copy_from_slice(&valbuf);
        }
    }
    tdb
}

/// Finish a setup: assemble the server and a client with a fresh secret.
///
/// The client is built via the fast co-located path
/// ([`SimplePirClient::with_local_server`]): it computes the recover
/// offset `hint·s` directly as `D·(A·s)` rather than materialising the
/// whole `hint = D·A` (a `Θ(R·C·N)` cost that would dominate a benchmark
/// run). The hint's on-wire size — the offline setup download the
/// head-to-head reports — is `R·N·4` and is recorded analytically.
fn assemble<R: RngCore>(
    tdb: Vec<u8>,
    sorted_hashes: &[u64],
    shape: MatrixShape,
    config: &SimpleConfig,
    seed: [u8; 16],
    rng: &mut R,
) -> (KpirServer, KpirClient) {
    let params = config.to_params(seed);
    let inner =
        SimplePirServer::from_transposed_db(tdb, shape.response_dim(), shape.columns, params);
    let hint_wire_bytes =
        shape.response_dim() * config.lwe_dim as usize * core::mem::size_of::<u32>();
    let map = KeyIndexMap::build(sorted_hashes, shape.epsilon);
    let client_inner = SimplePirClient::with_local_server(&inner, rng);
    let server = KpirServer { inner, shape };
    let client = KpirClient {
        inner: client_inner,
        map,
        shape,
        hint_wire_bytes,
    };
    (server, client)
}

/// Set up KPIR^index over a **synthetic** database of `n` key-value pairs
/// (original keys `0..n`, values from [`synthetic_value`]). Low memory —
/// no plaintext database is materialised. For benchmarks.
///
/// # Constraints
///
/// Panics if `n == 0` or `value_bytes == 0`.
pub fn build_synthetic<R: RngCore>(
    n: usize,
    value_bytes: usize,
    epsilon: u32,
    config: &SimpleConfig,
    seed: [u8; 16],
    rng: &mut R,
) -> (KpirServer, KpirClient) {
    let mut hashes: Vec<u64> = (0..n as u64).map(|i| hash_key(&i.to_le_bytes())).collect();
    hashes.sort_unstable();
    hashes.dedup(); // hash collisions among distinct keys are ~impossible at n ≤ 2^30
    let shape = MatrixShape::new(hashes.len(), value_bytes, epsilon);
    let tdb = encode_tdb(&hashes, &shape, |g, buf| synthetic_value(hashes[g], buf));
    assemble(tdb, &hashes, shape, config, seed, rng)
}

/// Set up KPIR^index over an explicit set of key-value pairs. Values are
/// truncated / zero-padded to `value_bytes`. For correctness tests.
///
/// # Constraints
///
/// Panics if `pairs` is empty, if `value_bytes == 0`, or if two keys
/// collide under [`hash_key`] (astronomically unlikely).
pub fn build_from_pairs<R: RngCore>(
    pairs: &[(Vec<u8>, Vec<u8>)],
    value_bytes: usize,
    epsilon: u32,
    config: &SimpleConfig,
    seed: [u8; 16],
    rng: &mut R,
) -> (KpirServer, KpirClient) {
    let mut items: Vec<(u64, &[u8])> = pairs
        .iter()
        .map(|(k, v)| (hash_key(k), v.as_slice()))
        .collect();
    items.sort_by_key(|(h, _)| *h);
    for w in items.windows(2) {
        assert!(w[0].0 != w[1].0, "keyword fingerprint collision");
    }
    let hashes: Vec<u64> = items.iter().map(|(h, _)| *h).collect();
    let shape = MatrixShape::new(hashes.len(), value_bytes, epsilon);
    let tdb = encode_tdb(&hashes, &shape, |g, buf| {
        let v = items[g].1;
        let take = v.len().min(buf.len());
        buf[..take].copy_from_slice(&v[..take]);
        buf[take..].fill(0);
    });
    assemble(tdb, &hashes, shape, config, seed, rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// End-to-end: every present key retrieves its exact value; an absent
    /// key yields ⊥. This is the definitive correctness check — it
    /// exercises hashing, the PLA, the column encoding/padding, and the
    /// full LWE query/answer/recover pipeline together.
    #[test]
    fn end_to_end_present_and_absent() {
        let mut rng = StdRng::seed_from_u64(0x5EED);
        let value_bytes = 24;
        let n = 4000usize;
        let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
            .map(|i| {
                let key = format!("user-{i:06}").into_bytes();
                let mut val = vec![0u8; value_bytes];
                val[..8].copy_from_slice(&(i as u64).to_le_bytes());
                val[8] = (i % 251) as u8;
                (key, val)
            })
            .collect();

        let config = SimpleConfig::with_lwe_dim(512); // small N for a fast test
        let (server, client) =
            build_from_pairs(&pairs, value_bytes, 4, &config, [9u8; 16], &mut rng);

        // Every present key round-trips to its exact value.
        for (key, want) in &pairs {
            let qu = client.query(key, &mut rng);
            let ans = server.answer(&qu);
            let got = client.recover(key, &ans);
            assert_eq!(
                got.as_ref(),
                Some(want),
                "present key {:?}",
                String::from_utf8_lossy(key)
            );
        }

        // Absent keys return ⊥.
        for i in n..n + 200 {
            let key = format!("user-{i:06}").into_bytes();
            let qu = client.query(&key, &mut rng);
            let ans = server.answer(&qu);
            assert_eq!(client.recover(&key, &ans), None, "absent key {i}");
        }
    }

    /// The synthetic builder used by benches is self-consistent: a sampled
    /// present key recovers `synthetic_value(hash(key))`.
    #[test]
    fn synthetic_roundtrip() {
        let mut rng = StdRng::seed_from_u64(0xCAFE);
        let n = 3000usize;
        let value_bytes = 32usize;
        let config = SimpleConfig::with_lwe_dim(512);
        let (server, client) = build_synthetic(n, value_bytes, 4, &config, [3u8; 16], &mut rng);

        for i in [0u64, 1, 7, 100, 1500, (n - 1) as u64] {
            let key = i.to_le_bytes();
            let qu = client.query(&key, &mut rng);
            let ans = server.answer(&qu);
            let got = client.recover(&key, &ans).expect("present");
            let mut want = vec![0u8; value_bytes];
            synthetic_value(hash_key(&key), &mut want);
            assert_eq!(got, want, "synthetic key {i}");
        }
    }

    /// Geometry accessors line up with `MatrixShape` (guards the wiring
    /// between the scheme and the head-to-head metrics).
    #[test]
    fn reports_geometry() {
        let mut rng = StdRng::seed_from_u64(1);
        let config = SimpleConfig::with_lwe_dim(256);
        let (server, client) = build_synthetic(2000, 32, 4, &config, [0u8; 16], &mut rng);
        let s = *server.shape();
        assert_eq!(client.shape(), &s);
        assert_eq!(client.hint_wire_bytes(), s.response_dim() * 256 * 4);
        assert!(client.num_segments() >= 1);
        assert_eq!(client.map_wire_bytes(), client.num_segments() * 24);
    }
}
