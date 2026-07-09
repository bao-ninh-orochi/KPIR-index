# kpir-index

The **KPIR^index** keyword-PIR construction of Hao et al. (USENIX
Security 2025, §6; `mpc4j` `SimplePgmCpKsPir`): an approximate
key-to-index mapping reduces keyword PIR to one invocation of the
[`simplepir`](../simplepir/README.md) index-PIR backend. Core logic plus
the head-to-head benchmarks for the CANS2026 Table 3 comparison against
ChalametPIR and RisePIR.

## Role in the workspace

```
kpir-index ──► simplepir
```

This crate owns everything keyword-specific: hashing, the PLA map, the
matrix geometry, the bit-packed encoding, and the query/answer/recover
glue. The LWE math lives entirely in `simplepir`.

## What's here

| Module | Items |
|---|---|
| `pla` | `KeyIndexMap` — optimal ε-PLA (PGM-index, Ferragina & Vinciguerra), a direct port of the `mpc4j` `PlaModel` (`f64` key coordinate, `i64` index coordinate). `build` over sorted hashed keys, `extract` within `ε + 1` of the true rank |
| `params` | `MatrixShape` — the `mpc4j` `getMatrixSize` formula generalised to a variable plaintext width. `new(n, ℓ, ε, pt)` is explicit; `choose(n, ℓ, ε, σ)` picks the largest decodable width (the adaptive operating point). `FINGERPRINT_BYTES = 8`, `DEFAULT_EPSILON = 4` |
| `scheme` | `hash_key` (xxh3), `build_synthetic` / `build_from_pairs`, `KpirServer` / `KpirClient`, plus the private bit-packing (`write_bits`/`read_bits`) and the closed-form column-major encoder (`encode_db`) |
| `benches/` | `helpers.rs` (shared CLI/sampler/CSV/verify harness, included via `#[path]`), `headtohead`, `kpir_answer`, `kpir_query`, `kpir_decode` |
| `tests/` | `proptests.rs` — property tests: end-to-end roundtrip on arbitrary pairs, the PLA extract guarantee, and the `MatrixShape` defining inequalities |

## How a query works

1. **Setup** — hash every key to a uniform `u64` (fingerprint = PLA
   coordinate), sort, learn the ε-PLA, and place entry `g` (the `g`-th
   smallest hash) at column `g / data_rows`, row `(g mod data_rows) + ε + 1`
   of the matrix; each entry is `fingerprint(64 b) ‖ value(8ℓ b)`
   bit-packed LSB-first into `partition` `pt`-bit `Z_p` cells. The
   `ε+1`/`ε+2` boundary rows replicate neighbouring columns' edge
   entries, covering the PLA's worst-case `ε + 1` error.
2. **Query** — `extract` the approximate rank, map it to the covering
   column, issue one Row-KOPIR query (`C` `u32` cells on the wire).
3. **Recover** — round & unpack the returned column (`R` cells), scan
   its candidate entries for the matching 64-bit fingerprint, return the
   value — or `⊥` if the key is absent.

The geometry (`partition`, `columns`, `data_rows`, `rows`) reproduces the
reference communication numbers **to the byte** at `plaintext_bits = 8`
(pinned by `params::tests::matches_mpc4j_byte_geometry`); the adaptive
width picks `pt = 9/9/8` at `m = 10⁶`, `ℓ = 32/256/1024 B`, matching
RisePIR-S (pinned by `params::tests::adaptive_operating_point`).

## Benchmarks

Every bench is a standalone `main` (`harness = false`) with a `clap` CLI,
gates on a built-in correctness verifier before reporting, and appends
one CSV row per config under `${KPIR_RESULTS_DIR:-results}/<bench>.csv`
— the same harness conventions as the RisePIR repo.

```bash
../../scripts/smoke.sh                                  # all benches, tiny config
../../scripts/headtohead.sh                             # CANS2026 Table 3 sweep
../../scripts/bench.sh kpir_answer --m 1000000 --value-bytes 256
```

## Status

Research-grade core logic for benchmarking; see the
[workspace README](../../README.md#security) for the security notice.
The public API surface (`build_*`, `KpirServer`, `KpirClient`,
`MatrixShape`, `KeyIndexMap`) is small and stable; serialization and
transport are intentionally out of scope.
