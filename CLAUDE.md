# KPIR^index — engineering guide

Core-logic Rust re-implementation of the **KPIR^index** keyword-PIR scheme
(Hao et al., USENIX Sec'25) for a fair head-to-head benchmark against
ChalametPIR and RisePIR (CANS2026 Table 3). No networking/serialization — only
the cryptographic pipeline. Mirrors the RisePIR (`../Incremental-Keyword-PIR`)
repo conventions so numbers are measured under the same model.

## Workspace

Two crates, dependency direction `kpir-index → simplepir`. Each crate has a
`README.md` (referenced from its `Cargo.toml`); the root `README.md` carries
the measured head-to-head row, the fairness/measurement model, and the
security notice.

- **`simplepir`** — Row-KOPIR (SimplePIR) index-PIR backend. Owns all LWE math.
  - `params.rs` — `SimpleParams` (runtime) / `SimpleConfig` (user knobs). Split
    mirrors RisePIR. `q = 2³²` implicit; `N = 1275` default; `σ = 6.4`.
    `plaintext_bits` (`p = 2^plaintext_bits`) is **adaptive**, not fixed:
    `noise_bound_satisfied` is the SimplePIR Gaussian decode bound
    (`Δ ≥ 2√2·σ·√(ln(2/δ))·p·√C`, `δ = 2⁻⁴⁰`, `MAX_PLAINTEXT_BITS = 14`).
  - `matvec.rs` — the shared register-blocked `acc += qᵀ·D` kernel; all `u32`
    (DB cells in `[0, p)`). Bit-exact, single-thread, no explicit SIMD.
    **Bit-identical port of RisePIR's kernel** (`ikpir-common` commit
    `dc2dd04`): width-adaptive dispatch (largest power-of-two `R ≤ 16` with
    `R·width ≤ 2048`). Used by every op — `answer` (`qu·D`), `setup` (`Aᵀ·D`),
    the `A·s` / `sᵀ·H` precompute. **Do not switch the DB answer to a
    ChalametPIR-style column-major dot product**: measured single-thread on
    Apple M1 this blocked kernel is 28–37% *faster* than column-major when the
    `R`-cell accumulator fits L1 (equal when RAM-bound); ChalametPIR's edge was
    rayon parallelism, which this crate forgoes. The module doc records the
    numbers.
  - `sampler.rs` — ChaCha-seeded `A` expansion (transposed `N×C` layout = `Aᵀ`),
    uniform-`Z_q` secret, Box–Muller discrete Gaussian.
  - `arith.rs` — `round_q_to_p` (`Round_Δ`), width-generic (matches `mpc4j`'s
    byte recovery at `plaintext_bits = 8`).
  - `backend.rs` — `SimplePirServer` (row-major `C×R` `u32` DB `db[c*R+r]` —
    RisePIR's layout, each query-column contiguous — `setup`/`answer`, with a
    decode-bound **guard** in `from_row_major_db`; `recover → Vec<u32>` of
    `Z_p` cells), `SimplePirClient` (`new` from real hint, or
    `with_local_server` fast path), `Hint`.
- **`kpir-index`** — the keyword scheme.
  - `params.rs` — `MatrixShape`: the `mpc4j getMatrixSize` formula generalised
    to variable `plaintext_bits`. **This is what makes the head-to-head comms
    match the reference** — do not "simplify" it. `new(n, ℓ, ε, pt)` is explicit;
    `choose(n, ℓ, ε, σ)` picks the largest safe `pt` (adaptive operating point).
  - `pla.rs` — optimal ε-PLA (PGM). Direct port of `mpc4j` `PlaModel`
    (`f64` key coord, `i64` index coord). `build` + `extract`.
  - `scheme.rs` — `hash_key` (xxh3), `synthetic_value`, `write_bits`/`read_bits`
    (LSB-first bit-packing into `pt`-bit cells), `encode_db` (closed-form
    column-major placement → `Vec<u32>`), `build_synthetic` / `build_from_pairs`,
    `KpirServer` / `KpirClient`.
  - `benches/` — `helpers.rs` (shared via `#[path]`), `headtohead`, `kpir_answer`,
    `kpir_query`, `kpir_decode`.
  - `tests/proptests.rs` — property tests over the public API: end-to-end
    roundtrip on arbitrary pairs, PLA extract guarantee, `MatrixShape`
    defining inequalities, `choose` maximality.

## Load-bearing facts (do not regress)

- **Matrix geometry** (`MatrixShape`): `partition = ceil(payload_bits / pt)`
  where `payload_bits = 8·(ℓ + FINGERPRINT_BYTES)` (= `Z_p` cells per entry;
  `= ℓ + 8` at `pt = 8`), `columns = ceil(sqrt(n·partition))` (= query dim `C`),
  `data_rows = max(ceil(n/columns), ε+2)`, `rows = data_rows + 2ε + 3`,
  response dim `R = rows·partition`. At `pt = 8` this is the exact `mpc4j` byte
  geometry (CANS2026 Table 3 / Table 6, Hao Table 5) — pinned by
  `params::tests::matches_mpc4j_byte_geometry`. The **adaptive** operating point
  (`choose`) picks `pt = 9/9/8` for `ℓ = 32/256/1024 B` at `m = 10⁶`, matching
  RisePIR-S — pinned by `params::tests::adaptive_operating_point`.
- **Adaptive plaintext width**: `pt = min(MAX_PLAINTEXT_BITS, largest pt whose
  query dim C still decodes)` via `simplepir::noise_bound_satisfied` (the noise
  dim is `C`, since `answer` sums over the `C` query positions). Same choice
  RisePIR-S makes. The backend `from_row_major_db` **guards** on this bound.
- **Column encoding** (`encode_db → Vec<u32>`, row-major `db[col*R + r]`):
  column `col`, plane-row `j` holds sorted entry `g = col·data_rows + (j − ε − 1)`,
  its `fingerprint(64b) ‖ value(8ℓ b)` bit-packed LSB-first into `partition`
  `pt`-bit cells at contiguous response rows `r = j·partition + p`
  (`db[col*R + r]`), so a recovered column's `cells[r]` are in plane order for
  `recover` (⊥ = every cell `p − 1`, reads back as fingerprint `u64::MAX`).
  Boundary rows carry neighbouring columns' edges; the PLA's `ε+1` effective
  error is covered by the `ε+1`/`ε+2` padding — pinned by the end-to-end test
  (which now runs at `pt = 10`) and the `tests/proptests.rs` roundtrip.
- **Constant-time decode** (`KpirClient::recover`): a **branchless full-column
  scan** — it visits all `rows` slots and OR-masks the value into a fixed
  accumulator via `ct_eq_u64_mask` (port of RisePIR's `ct_eq_u32_mask`), so the
  probe path leaks no matched-row timing. **Do not reintroduce an early return
  / `if fp == want`.** Best-effort (not verified); the underlying SimplePIR LWE
  decode is a separate concern. Mirrors `ikpir-client`'s `decode`.
- **FrodoPIR convention** `ans = qu·D` (matches RisePIR): query `qu` (length `C`)
  left-selects one column of `D ∈ Z_p^{C×R}` stored **row-major** (`db[c*R+r]`,
  each column contiguous — RisePIR's exact layout); response length `R`.
  `hint = Aᵀ·D`. Structurally identical to RisePIR (storage + blocked kernel);
  the change from the old code was the naming (was written `D·qu`), not the
  bytes. Column-major (ChalametPIR) was measured and rejected — see `matvec.rs`.
- **`with_local_server` == `new(hint)`**: the fast client computes `sᵀ·H` as
  `(A·s)·D`; pinned by `backend::tests::local_server_client_matches_hint_client`.
- **N choice**: default 1275 (128-bit, matches RisePIR-S). `--lwe-dim 1024` gives
  the original `mpc4j` setting. `σ = 6.4` + uniform-`Z_q` secret = SimplePIR (kept
  in sync with RisePIR-S).

## Commands

```bash
cargo test --workspace                          # unit + doc + property tests
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
cargo doc --workspace --no-deps                 # rustdoc must stay warning-free
./scripts/smoke.sh                              # correctness-gated smoke of every bench
./scripts/headtohead.sh                         # CANS2026 Table 3 sweep
./scripts/bench.sh kpir_answer --m 1000000 --value-bytes 256
```

## Benching gotchas

- The `ℓ = 1024 B, m = 10⁶` config allocates a **5.7 GiB** `u32` database.
  On a 16 GiB machine the answer measurement is memory-compression-bound
  (~1.8 s instead of ~170 ms of compute) — treat that row as
  hardware-limited, don't "fix" the kernel. When the DB fits in RAM the
  kernel streams at ~34 GB/s on Apple M1 (verified at `--m 300000`).
- Never run builds or other memory-heavy work concurrently with a sweep;
  it visibly degrades the measured bandwidth.
- Bench CSVs append; the schema gained a `plaintext_bits` column with the
  adaptive-width change. Archive (don't mix) old-schema CSVs — see
  `results/kpir-index/archive-pre-adaptive/` (local only, gitignored),
  whose rows also predate the `u32`-cell model (4× less memory traffic, so
  their latencies are not comparable).

## Conventions

- Toolchain pinned `1.85.0`, edition 2021. `unsafe_op_in_unsafe_fn = deny`;
  `missing_docs` per-library. Single-thread, no rayon/ndarray/external crypto —
  match RisePIR so head-to-head numbers are comparable.
- Benches: `harness = false`, standalone `main`, clap `Cli` (filters the
  `--bench` cargo injects), criterion `iter_custom` throughput measurement
  (`helpers::measure`) → mean/min/max/stddev ops/s, one config = one appended
  CSV row under `${KPIR_RESULTS_DIR}/<bench>.csv`. The criterion contract is
  clap-tunable (`--sample-size`/`--warmup-secs`/`--measurement-secs`, defaults
  100 / 3 s / 5 s — the same Table 3 contract RisePIR and ChalametPIR pin), and
  criterion's native report lands under `target/criterion/`. `autobenches =
  false` (helpers is a shared module, not a target). Communication = fixed-width
  LE wire bytes; `A` excluded from the hint (regenerated from seed).
- Every bench gates on `helpers::verify` before reporting numbers.
- `smoke.sh` writes to `results/.smoke/` (its own scratch base, cleared at
  start and end) — it must never touch the real `results/kpir-index/` CSVs.
- CI mirrors RisePIR: three jobs (fmt / clippy / test) on
  `dtolnay/rust-toolchain@1.85.0` + `Swatinem/rust-cache@v2`, plus a bench
  build and `smoke.sh` in the test job.
- Dual-licensed `MIT OR Apache-2.0`: `LICENSE-APACHE` + `LICENSE-MIT` both
  exist and the `license` field must stay in sync with them.
