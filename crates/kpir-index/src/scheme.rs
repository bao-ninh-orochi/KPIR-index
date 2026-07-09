//! The KPIR^index keyword-PIR construction (Hao et al. §6.2, Figure 8;
//! `mpc4j` `SimplePgmCpKsPir`).
//!
//! # Purpose
//!
//! Ties the approximate key-to-index map ([`crate::pla`]) to the
//! Row-KOPIR backend ([`simplepir`]). Setup sorts the hashed keys, learns
//! the PLA map, and repetition-encodes the key-value pairs into the
//! `R × C` `Z_p` matrix; a query extracts the approximate rank, selects the
//! covering column, and privately retrieves it; recover scans the `≤ 2ε+3`
//! candidate entries of that column for the fingerprint match.
//!
//! # Encoding (closed form of `mpc4j`'s column-major placement)
//!
//! Sort the `n` pairs by hashed key. The slot at column `col`,
//! plane-row `j` holds the sorted entry `fingerprint(k_g) ‖ value_g` where
//! ```text
//! g = col · data_rows + (j − ε − 1),   0 ≤ g < n   (else ⊥)
//! ```
//! Each entry's `fingerprint(64 bits) ‖ value(8ℓ bits)` bitstring is
//! packed LSB-first into `partition = ceil(payload_bits / plaintext_bits)`
//! `Z_p` cells across the response dimension `R = rows · partition` (at
//! `plaintext_bits = 8` this is one byte per cell — the `mpc4j` layout).
//! Empty slots default to `⊥` (every cell `p − 1`, which reads back as
//! fingerprint `u64::MAX`). The `ε+1` top / `ε+2` bottom boundary rows
//! automatically hold the neighbouring columns' edge entries, so the true
//! entry — within `ε+1` of the PLA estimate — always lands inside the
//! single retrieved column.
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
    /// if `key` is absent.
    ///
    /// # Rationale
    ///
    /// **Side-channel hardening.** Every one of the `rows` candidate slots in
    /// the retrieved column is unpacked and merged into a fixed-size
    /// accumulator via a branchless OR-masked select (`ct_eq_u64_mask`): the
    /// scan runs the full `0..rows` regardless of where — or whether — the
    /// fingerprint matches, and no value byte is read conditionally. The probe
    /// path is therefore independent of the matched row, so a co-located timing
    /// observer learns nothing beyond the public geometry `(rows, partition)`.
    /// This is best-effort hand-rolled masking, not a formally verified
    /// constant-time guarantee; the underlying SimplePIR LWE decode
    /// ([`SimplePirClient::recover`](simplepir::SimplePirClient::recover)) is a
    /// separate, un-audited concern.
    ///
    /// **Fingerprint-collision behaviour.** The returned value is the bitwise
    /// OR of every matching slot's bytes. Stored fingerprints are globally
    /// distinct (enforced at build time) and boundary-replicated copies of an
    /// entry are byte-identical, so at most one *distinct* value is ever OR'd
    /// in; empty slots read back as `u64::MAX` and never match a real key hash.
    pub fn recover(&self, key: &[u8], ans: &[u32]) -> Option<Vec<u8>> {
        let want = hash_key(key);
        let cells = self.inner.recover(ans);
        let partition = self.shape.partition;
        let pt = self.shape.plaintext_bits;

        // Accumulator allocated up front so the memory path is data-independent.
        let mut acc = vec![0u8; self.shape.value_bytes];
        let mut found: u64 = 0;
        for j in 0..self.shape.rows {
            let slot = &cells[j * partition..(j + 1) * partition];
            let fp = read_bits(slot, pt, 0, FINGERPRINT_BITS);
            let mask = ct_eq_u64_mask(fp, want); // all-ones if equal, else 0
            let mask8 = (mask & 0xFF) as u8;
            for (k, a) in acc.iter_mut().enumerate() {
                let byte = read_bits(slot, pt, FINGERPRINT_BITS + 8 * k, 8) as u8;
                *a |= mask8 & byte;
            }
            found |= mask;
        }
        if found != 0 {
            Some(acc)
        } else {
            None
        }
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

/// Fingerprint width in bits (`FINGERPRINT_BYTES · 8`).
const FINGERPRINT_BITS: usize = FINGERPRINT_BYTES * 8;

/// Branchless `u64` equality mask: `u64::MAX` if `a == b`, else `0`.
///
/// Constant-time trick: `a ^ b == 0` iff `a == b`; squeeze zero/non-zero into
/// the sign bit via `x | -x`, shift it down, then subtract 1 to flip the
/// meaning. Mirrors `ct_eq_u32_mask` in the RisePIR reference decode, widened
/// to the 64-bit fingerprint. Used by [`KpirClient::recover`] to select a
/// matching slot's value without a data-dependent branch.
#[inline]
const fn ct_eq_u64_mask(a: u64, b: u64) -> u64 {
    let x = a ^ b;
    ((x | x.wrapping_neg()) >> 63).wrapping_sub(1)
}

/// Write the low `n_bits` of `value` into `cells` starting at global bit
/// `bit_offset`, LSB-first, where each cell holds `pt` bits (cell `i` =
/// bits `[i·pt, (i+1)·pt)`). Straddles cell boundaries; leaves untouched
/// bits (and the ragged high bits of the final cell) as they were.
/// `n_bits ≤ 64`; `pt ≤ 31`, so every per-cell slice is `≤ 31 < 64` bits.
#[inline]
fn write_bits(cells: &mut [u32], pt: u32, bit_offset: usize, n_bits: usize, value: u64) {
    let pt = pt as usize;
    let mut written = 0usize;
    while written < n_bits {
        let global = bit_offset + written;
        let cell = global / pt;
        let intra = global % pt;
        let take = (pt - intra).min(n_bits - written);
        let mask = (1u64 << take) - 1;
        let chunk = (value >> written) & mask;
        let cur = cells[cell] as u64;
        cells[cell] = ((cur & !(mask << intra)) | (chunk << intra)) as u32;
        written += take;
    }
}

/// Inverse of [`write_bits`]: read `n_bits` from `cells` at global bit
/// `bit_offset` into the low bits of a `u64`.
#[inline]
fn read_bits(cells: &[u32], pt: u32, bit_offset: usize, n_bits: usize) -> u64 {
    let pt = pt as usize;
    let mut acc = 0u64;
    let mut read = 0usize;
    while read < n_bits {
        let global = bit_offset + read;
        let cell = global / pt;
        let intra = global % pt;
        let take = (pt - intra).min(n_bits - read);
        let mask = (1u64 << take) - 1;
        let chunk = (cells[cell] as u64 >> intra) & mask;
        acc |= chunk << read;
        read += take;
    }
    acc
}

/// Repetition-encode sorted entries into the transposed Row-KOPIR matrix
/// `tdb` (`C × R`, `tdb[col*R + r] = D[r][col]`). Each slot holds
/// `fingerprint(64 bits) ‖ value(8ℓ bits)` bit-packed into `partition`
/// `pt`-bit `Z_p` cells; empty slots are `⊥` (every cell `p − 1`).
/// `fill_value(g, buf)` writes the `value_bytes`-byte value of the `g`-th
/// sorted entry.
fn encode_tdb(
    sorted_hashes: &[u64],
    shape: &MatrixShape,
    mut fill_value: impl FnMut(usize, &mut [u8]),
) -> Vec<u32> {
    let n = shape.n;
    let r = shape.response_dim();
    let partition = shape.partition;
    let pt = shape.plaintext_bits;
    let data_rows = shape.data_rows as i64;
    let eps = shape.epsilon as i64;

    // ⊥ default: every cell p − 1, so an untouched slot reads back as
    // fingerprint u64::MAX (never matches a real key hash).
    let sentinel = (1u32 << pt) - 1;
    let mut tdb = vec![sentinel; shape.columns * r];
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
            let slot = &mut tdb[base..base + partition];
            slot.fill(0); // clean valid slot (unused high bits stay 0)
            write_bits(slot, pt, 0, FINGERPRINT_BITS, sorted_hashes[g]);
            fill_value(g, &mut valbuf);
            for (k, &b) in valbuf.iter().enumerate() {
                write_bits(slot, pt, FINGERPRINT_BITS + 8 * k, 8, b as u64);
            }
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
    tdb: Vec<u32>,
    sorted_hashes: &[u64],
    shape: MatrixShape,
    config: &SimpleConfig,
    seed: [u8; 16],
    rng: &mut R,
) -> (KpirServer, KpirClient) {
    let params = config.to_params(seed, shape.plaintext_bits);
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
    let shape = MatrixShape::choose(hashes.len(), value_bytes, epsilon, config.sigma);
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
    let shape = MatrixShape::choose(hashes.len(), value_bytes, epsilon, config.sigma);
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

    /// The branchless equality mask returns all-ones exactly on equality and
    /// all-zeros otherwise — the invariant the constant-time `recover` scan
    /// relies on (including the `u64::MAX` empty-slot sentinel as a non-match).
    #[test]
    fn ct_eq_u64_mask_is_branchless_equality() {
        let vals = [
            0u64,
            1,
            2,
            42,
            u64::MAX,
            u64::MAX - 1,
            1 << 63,
            (1 << 63) + 1,
        ];
        for &x in &vals {
            assert_eq!(ct_eq_u64_mask(x, x), u64::MAX, "equal case x={x:#x}");
            for &y in &vals {
                if x != y {
                    assert_eq!(ct_eq_u64_mask(x, y), 0, "unequal case x={x:#x} y={y:#x}");
                }
            }
        }
        // The empty-slot sentinel never masks in a real key hash.
        assert_eq!(ct_eq_u64_mask(u64::MAX, 0x1234_5678_9abc_def0), 0);
    }

    /// Bit-packing round-trips a fingerprint + value across cell boundaries
    /// for several plaintext widths, and every cell stays in `[0, p)`.
    #[test]
    fn bitpack_roundtrip() {
        let mut rng = StdRng::seed_from_u64(0xB17);
        for pt in [8u32, 9, 10, 12, 14] {
            for value_bytes in [1usize, 7, 16, 32, 100] {
                let payload_bits = FINGERPRINT_BITS + 8 * value_bytes;
                let partition = payload_bits.div_ceil(pt as usize);
                let mut cells = vec![0u32; partition];
                let fp = rng.next_u64();
                let val: Vec<u8> = (0..value_bytes)
                    .map(|_| (rng.next_u32() & 0xff) as u8)
                    .collect();

                write_bits(&mut cells, pt, 0, FINGERPRINT_BITS, fp);
                for (k, &b) in val.iter().enumerate() {
                    write_bits(&mut cells, pt, FINGERPRINT_BITS + 8 * k, 8, b as u64);
                }
                assert!(
                    cells.iter().all(|&c| c < (1u32 << pt)),
                    "cell overflow pt={pt}"
                );
                assert_eq!(read_bits(&cells, pt, 0, FINGERPRINT_BITS), fp, "fp pt={pt}");
                for (k, &b) in val.iter().enumerate() {
                    assert_eq!(
                        read_bits(&cells, pt, FINGERPRINT_BITS + 8 * k, 8) as u8,
                        b,
                        "byte {k} pt={pt} vb={value_bytes}"
                    );
                }
            }
        }
    }

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
