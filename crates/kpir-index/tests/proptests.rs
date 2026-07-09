//! Property tests over the public API (mirrors the RisePIR `tests/`
//! proptest convention).
//!
//! Three layers, each pinned as an invariant rather than an example:
//!
//! - **End-to-end**: for arbitrary key-value pairs, every present key
//!   recovers exactly its (padded/truncated) value and an absent key
//!   yields ⊥ — across random value widths, ε, and database sizes.
//! - **PLA**: the Extract guarantee `|extract(k) − rank(k)| ≤ ε + 1`
//!   holds for arbitrary strictly-increasing key sets (the `+1` is the
//!   mpc4j-faithful intercept truncation the column padding covers).
//! - **Geometry**: `MatrixShape` satisfies the defining inequalities of
//!   the mpc4j `getMatrixSize` formula at every `(n, ℓ, ε, pt)`, and
//!   `choose` really picks the largest decodable plaintext width.

use proptest::collection::{btree_map, btree_set};
use proptest::prelude::*;

use kpir_index::pla::KeyIndexMap;
use kpir_index::{build_from_pairs, MatrixShape};
use rand::rngs::StdRng;
use rand::SeedableRng;
use simplepir::{noise_bound_satisfied, SimpleConfig, MAX_PLAINTEXT_BITS};

proptest! {
    // Each case builds a full (small) PIR instance; keep the count modest.
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Every present key recovers exactly its value (zero-padded or
    /// truncated to `value_bytes`); a key outside the set recovers ⊥.
    #[test]
    fn end_to_end_roundtrip(
        pairs in btree_map(
            proptest::collection::vec(any::<u8>(), 1..24),
            proptest::collection::vec(any::<u8>(), 0..40),
            2..60,
        ),
        value_bytes in 1usize..32,
        epsilon in 1u32..6,
        seed in any::<u64>(),
    ) {
        let mut rng = StdRng::seed_from_u64(seed);
        let pairs: Vec<(Vec<u8>, Vec<u8>)> = pairs.into_iter().collect();
        let config = SimpleConfig::with_lwe_dim(128); // small N: fast, correctness only
        let (server, client) =
            build_from_pairs(&pairs, value_bytes, epsilon, &config, [1u8; 16], &mut rng);

        for (key, val) in &pairs {
            let qu = client.query(key, &mut rng);
            let ans = server.answer(&qu);
            let got = client.recover(key, &ans);
            // Expected: `val` truncated / zero-padded to value_bytes.
            let mut want = vec![0u8; value_bytes];
            let take = val.len().min(value_bytes);
            want[..take].copy_from_slice(&val[..take]);
            prop_assert_eq!(got, Some(want), "present key {:?}", key);
        }

        // 25 bytes long — longer than every stored key, hence absent.
        let absent = vec![0xA5u8; 25];
        let qu = client.query(&absent, &mut rng);
        let ans = server.answer(&qu);
        prop_assert_eq!(client.recover(&absent, &ans), None, "absent key");
    }
}

proptest! {
    /// The Extract guarantee: within `ε + 1` of the true rank for every
    /// training key, for arbitrary strictly-increasing key sets and ε.
    #[test]
    fn pla_extract_guarantee(
        keys in btree_set(any::<u64>(), 1..400),
        epsilon in 0u32..9,
    ) {
        let keys: Vec<u64> = keys.into_iter().collect();
        let map = KeyIndexMap::build(&keys, epsilon);
        let bound = epsilon as i64 + 1;
        for (rank, &k) in keys.iter().enumerate() {
            let pos = map.extract(k) as i64;
            prop_assert!(
                (pos - rank as i64).abs() <= bound,
                "key #{rank}={k}: extract={pos}, bound={bound}, segments={}",
                map.num_segments()
            );
        }
    }

    /// `MatrixShape::new` satisfies the defining (in)equalities of the
    /// mpc4j `getMatrixSize` formula at every operating point.
    #[test]
    fn matrix_shape_invariants(
        n in 1usize..5_000_000,
        value_bytes in 1usize..2048,
        epsilon in 0u32..9,
        pt in 1u32..=14,
    ) {
        let s = MatrixShape::new(n, value_bytes, epsilon, pt);
        let payload_bits = MatrixShape::payload_bits(value_bytes);
        let cells = n as u64 * s.partition as u64;
        let (c, pt_us) = (s.columns as u64, pt as usize);

        // partition = ceil(payload_bits / pt): minimal cell count.
        prop_assert!(s.partition * pt_us >= payload_bits);
        prop_assert!((s.partition - 1) * pt_us < payload_bits);
        // columns = ceil(sqrt(n · partition)).
        prop_assert!(c * c >= cells);
        prop_assert!((c - 1) * (c - 1) < cells);
        // Every entry gets a data slot; padding rows are exactly 2ε + 3.
        prop_assert!(s.columns * s.data_rows >= n);
        prop_assert!(s.data_rows >= epsilon as usize + 2);
        prop_assert_eq!(s.rows, s.data_rows + 2 * epsilon as usize + 3);
        prop_assert_eq!(s.response_dim(), s.rows * s.partition);
        // pt = 8 is the exact mpc4j byte-plane layout.
        if pt == 8 {
            prop_assert_eq!(s.partition, value_bytes + 8);
        }
    }

    /// `choose` picks the largest decodable width: the chosen geometry
    /// satisfies the SimplePIR bound, and (unless already at the cap) one
    /// bit wider — at its own, re-derived geometry — does not.
    #[test]
    fn choose_is_maximal(
        n in 1usize..5_000_000,
        value_bytes in 1usize..2048,
    ) {
        let sigma = 6.4;
        let s = MatrixShape::choose(n, value_bytes, 4, sigma);
        prop_assert!(noise_bound_satisfied(s.plaintext_bits, s.columns as u32, sigma));
        if s.plaintext_bits < MAX_PLAINTEXT_BITS {
            let wider = MatrixShape::new(n, value_bytes, 4, s.plaintext_bits + 1);
            prop_assert!(!noise_bound_satisfied(
                wider.plaintext_bits,
                wider.columns as u32,
                sigma
            ));
        }
    }
}
