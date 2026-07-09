# KPIR^index — engineering guide

Core-logic Rust re-implementation of the **KPIR^index** keyword-PIR scheme
(Hao et al., USENIX Sec'25) for a fair head-to-head benchmark against
ChalametPIR and RisePIR (CANS2026 Table 3). No networking/serialization — only
the cryptographic pipeline. Mirrors the RisePIR (`../Incremental-Keyword-PIR`)
repo conventions so numbers are measured under the same model.

## Workspace

Two crates, dependency direction `kpir-index → simplepir`:

- **`simplepir`** — Row-KOPIR (SimplePIR) index-PIR backend. Owns all LWE math.
  - `params.rs` — `SimpleParams` (runtime) / `SimpleConfig` (user knobs). Split
    mirrors RisePIR. `q = 2³²` implicit; `p = 256`; `N = 1275` default; `σ = 6.4`.
  - `matvec.rs` — the shared register-blocked `acc += qᵀ·D` kernel, generic over
    the cell type (`u8` database, `u32` for `A`/`hint`). Bit-exact, single-thread,
    no explicit SIMD. Ported from RisePIR.
  - `sampler.rs` — ChaCha-seeded `A` expansion (transposed `N×C` layout),
    uniform-`Z_q` secret, Box–Muller discrete Gaussian.
  - `arith.rs` — `round_q_to_p` (`Round_Δ`), matches `mpc4j`'s byte recovery.
  - `backend.rs` — `SimplePirServer` (transposed `C×R` `u8` DB, `setup`/`answer`),
    `SimplePirClient` (`new` from real hint, or `with_local_server` fast path),
    `Hint`.
- **`kpir-index`** — the keyword scheme.
  - `params.rs` — `MatrixShape`: the `mpc4j getMatrixSize` formula. **This is
    what makes the head-to-head comms match to the byte** — do not "simplify" it.
  - `pla.rs` — optimal ε-PLA (PGM). Direct port of `mpc4j` `PlaModel`
    (`f64` key coord, `i64` index coord). `build` + `extract`.
  - `scheme.rs` — `hash_key` (xxh3), `synthetic_value`, `encode_tdb` (the closed
    -form column-major placement), `build_synthetic` / `build_from_pairs`,
    `KpirServer` / `KpirClient`.
  - `benches/` — `helpers.rs` (shared via `#[path]`), `headtohead`, `kpir_answer`,
    `kpir_query`, `kpir_decode`.

## Load-bearing facts (do not regress)

- **Matrix geometry** (`MatrixShape`): `partition = ℓ + 8`,
  `columns = ceil(sqrt(n·partition))` (= query dim `C`),
  `data_rows = max(ceil(n/columns), ε+2)`, `rows = data_rows + 2ε + 3`,
  response dim `R = rows·partition`. Reproduces CANS2026 Table 3 / Table 6 and
  Hao Table 5 exactly — pinned by `params::tests::matches_cans2026_table3`.
- **Column encoding** (`encode_tdb`): cell `(col, j)` holds sorted entry `g =
  col·data_rows + (j − ε − 1)` (⊥ = `0xFF` if out of range). The boundary rows
  automatically carry neighbouring columns' edges; the PLA's `ε+1` effective
  error is covered by the `ε+1`/`ε+2` padding — pinned by the end-to-end test.
- **Row-KOPIR is column-selection** (transpose of the paper's Figure 2), matching
  `mpc4j`: query length = `C`, response length = `R`. `hint = D·A`.
- **`with_local_server` == `new(hint)`**: the fast client computes `hint·s` as
  `D·(A·s)`; pinned by `backend::tests::local_server_client_matches_hint_client`.
- **N choice**: default 1275 (128-bit, matches RisePIR-S). `--lwe-dim 1024` gives
  the original `mpc4j` setting. `p = 256` is fixed (`mpc4j`).

## Commands

```bash
cargo test --workspace                          # unit + doctests
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
./scripts/smoke.sh                              # correctness-gated smoke of every bench
./scripts/headtohead.sh                         # CANS2026 Table 3 sweep
./scripts/bench.sh kpir_answer --m 1000000 --value-bytes 256
```

## Conventions

- Toolchain pinned `1.85.0`, edition 2021. `unsafe_op_in_unsafe_fn = deny`;
  `missing_docs` per-library. Single-thread, no rayon/ndarray/external crypto —
  match RisePIR so head-to-head numbers are comparable.
- Benches: `harness = false`, standalone `main`, clap `Cli` (filters the
  `--bench` cargo injects), manual throughput sampler → mean/min/max/stddev
  ops/s, one config = one appended CSV row under
  `${KPIR_RESULTS_DIR}/<bench>.csv`. `autobenches = false` (helpers is a shared
  module, not a target). Communication = fixed-width LE wire bytes; `A` excluded
  from the hint (regenerated from seed).
- Every bench gates on `helpers::verify` before reporting numbers.
