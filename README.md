# KPIR^index — a Rust re-implementation for head-to-head keyword-PIR benchmarks

A faithful, **core-logic-only** Rust re-implementation of the **KPIR^index**
keyword-PIR construction of Hao et al., *"Practical Keyword Private Information
Retrieval from Key-to-Index Mappings"* (USENIX Security 2025), built to be
benchmarked **head-to-head** against ChalametPIR and RisePIR under a common
measurement model (CANS2026 Table 3).

> **Status.** Research artifact accompanying the CANS2026 measurements.
> The cryptographic pipeline is complete and correctness-gated end to end;
> networking, serialization, and persistence are intentionally out of scope.
> See [Security](#security) before considering any other use.

## Why this repo

The reference implementation of KPIR^index lives in
[`mpc4j`](https://github.com/alibaba-edu/mpc4j). Comparing it directly against
ChalametPIR and RisePIR conflates the algorithm with the runtime, in two ways
(both called out in the CANS2026 paper itself):

1. **Runtime.** `mpc4j` runs on the **JVM** with a real client-server
   transport (serialization, sockets, RTT); RisePIR and ChalametPIR are
   native Rust libraries measuring only core cryptographic logic. In
   CANS2026 Table 3, KPIR^index answers a 1 kB-value query in **2439 ms**,
   much of which is JVM + transport overhead, not the algorithm.
2. **Operating point.** `mpc4j` fixes the plaintext modulus at `p = 2⁸`,
   while ChalametPIR and RisePIR pick the **largest `p` that still
   decodes** at the given database size — a strictly better trade-off the
   reference build never exercises.

This repo removes both distortions: the KPIR^index core logic re-implemented
in Rust — no networking, no serialization — with the plaintext width chosen
adaptively by the **same rule RisePIR uses**, over the **same `u32` cell
model**, driven through a **bit-identical** register-blocked matvec kernel.
What remains in the comparison is the algorithm.

### Measured head-to-head row (m = 10⁶ keys, N = 1275, ε = 4)

| ℓ (value) | pt | Query upload | Response download | Expansion N/m | Answer — this repo, native | Answer — JVM `mpc4j` (Table 3) |
|-----------|---:|-------------:|------------------:|--------------:|---------------------------:|-------------------------------:|
| 32 B      | 9  | 24.00 kB     | 25.63 kB          | 1.068         | 3.09 ms                    | 68.5 ms                        |
| 256 B     | 9  | 61.32 kB     | 72.38 kB          | 1.180         | 30.87 ms                   | 328 ms                         |
| 1024 B    | 8  | 128.50 kB    | 177.50 kB         | 1.381         | 1.82 s †                   | 2439 ms                        |

Measured on an Apple M1 (16 GiB), single-threaded, via
`./scripts/headtohead.sh`. Query/response sizes and the expansion rate are
implementation-independent; answer latency is hardware-dependent.

† The ℓ = 1024 B config is an outlier on this machine, not in the algorithm:
its transposed `u32` database is **5.7 GiB**, which a 16 GiB laptop cannot
keep resident, so the measurement is dominated by macOS memory compression.
The same kernel streams at ~34 GB/s whenever the database fits in RAM — at
`--m 300000` the 1024 B answer takes **53.6 ms** over a 1.8 GiB database —
implying roughly 170 ms of actual compute at m = 10⁶ on a machine with
sufficient memory. Re-measure on the paper's reference hardware for
publishable numbers.

At the reference width **`pt = 8`** the geometry is byte-identical to Hao
Table 5 and `mpc4j`'s `getMatrixSize` (32 B → 25.30/27.20 kB, 256 B →
65.00/77.09 kB, 1024 B → 128.50/177.50 kB) — pinned by
`params::tests::matches_mpc4j_byte_geometry`. The adaptive selector picks
`pt = 9/9/8`, matching RisePIR-S — pinned by
`params::tests::adaptive_operating_point`.

## The scheme

KPIR^index reduces keyword PIR to standard index PIR with two building blocks:

1. **Approximate key-to-index mapping (PLA / PGM)** —
   [`kpir-index/src/pla.rs`](crates/kpir-index/src/pla.rs). Hash each keyword to a
   uniform 64-bit value, sort, and learn an optimal piece-wise linear
   approximation (Ferragina & Vinciguerra). For any trained key it returns a
   rank within `ε + 1` of the true one.
2. **Row-KOPIR (SimplePIR)** — the [`simplepir`](crates/simplepir/README.md) crate.
   The database is a `Z_p` matrix of `u32` cells (`p = 2^pt`, chosen
   adaptively); one query privately retrieves one whole **column**.

**Setup** sorts the `n` pairs by hashed key, learns the PLA map, and encodes
each `fingerprint(k) ‖ value` entry — bit-packed LSB-first into `pt`-bit
cells — into the matrix with an `ε+1` top / `ε+2` bottom boundary so the true
entry always lands in the single retrieved column. **Query** extracts the
approximate rank, maps it to a column, and issues one Row-KOPIR query.
**Recover** rounds & unpacks the returned column and returns the value whose
64-bit fingerprint matches the queried key, else `⊥`.

The full derivation — the exact matrix-geometry formula, the closed-form
column-major placement, and the decode bound — lives in the rustdoc:

```bash
cargo doc --workspace --no-deps --open
```

## Repository tour

| Path | What it is |
|---|---|
| [`crates/simplepir/`](crates/simplepir/README.md) | Row-KOPIR (SimplePIR) index-PIR backend: all LWE math, the shared blocked matvec kernel, samplers, decode bound |
| [`crates/kpir-index/`](crates/kpir-index/README.md) | The keyword scheme: PLA, matrix geometry, bit-packed encoding, query/answer/recover glue, benches, property tests |
| [`scripts/`](scripts/) | `bench.sh` (one bench, one config) · `headtohead.sh` (Table 3 sweep) · `smoke.sh` (correctness-gated tiny run of every bench) · `lib.sh` (shared) |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | fmt · clippy `-D warnings` · tests · bench build · smoke, on the pinned 1.85.0 toolchain |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Toolchain pin, local gates, bench workflow, PR conventions |
| [`SECURITY.md`](SECURITY.md) | Prototype threat-model caveats and private vulnerability reporting |
| `results/` | Bench CSVs (gitignored; append-per-run) |

## Parameters

| Parameter | Value | Note |
|-----------|-------|------|
| LWE modulus `q` | `2³²` | native `u32` wraparound |
| Plaintext modulus `p` | `2^pt`, **adaptive** (`pt = 9/9/8` at m = 10⁶, ℓ = 32/256/1024 B) | largest width satisfying the SimplePIR decode bound (`noise_bound_satisfied`, `δ = 2⁻⁴⁰`, cap `pt ≤ 14`); same rule as RisePIR. `pt = 8` = `mpc4j` |
| Error `σ` | `6.4` | discrete Gaussian, uniform `Z_q` secret (SimplePIR §4.2) |
| LWE dim `N` | **1275** (default) | 128-bit security (ADPS16), matches RisePIR-S; `mpc4j` uses 1024 (`--lwe-dim 1024`) |
| Approx error `ε` | `4` | `mpc4j` `EPSILON` |
| Fingerprint | 8 bytes | disambiguates the `≤ 2ε+3` candidate entries per column |

## Measurement model — what makes the head-to-head fair

Every convention below matches the RisePIR repo, so a latency difference
between the schemes is attributable to the algorithms, not the harnesses:

- **Same inner loop.** The `acc += qᵀ·D` register-blocked kernel
  (`crates/simplepir/src/matvec.rs`) is a bit-identical port of RisePIR's — the same
  width-adaptive blocking rule (largest power-of-two `R ≤ 16` with
  `R·width ≤ 2048` cells), pinned bit-exact against the naive loop by tests.
  RisePIR adopted this kernel to close a harness-induced gap against
  ChalametPIR; both repos now measure on it.
- **Same operating point.** `plaintext_bits` is chosen by the same decode
  bound RisePIR evaluates, over the same `u32` cell model.
- **Same execution model.** Single thread; no rayon, no explicit SIMD, no
  external crypto/linear-algebra crates; pinned toolchain `1.85.0`; `lto =
  "fat"`, `codegen-units = 1`.
- **Same wire accounting.** Communication = fixed-width little-endian `u32`
  cells: query `C·4` B, response `R·4` B, one-time hint `R·N·4` B. `A` is
  regenerated from its 16-byte seed, never shipped.
- **Honest hint accounting.** Benches skip materialising `hint = D·A`
  (a `Θ(R·C·N)` offline cost) by computing the client offset `hint·s` as
  `D·(A·s)` through the co-located server — **bit-identical** to using the
  real hint (pinned by `backend::tests::local_server_client_matches_hint_client`);
  the hint's wire size is still reported analytically.
- **Correctness-gated numbers.** Every bench first verifies the pipeline
  (present keys recover exact values, absent keys return `⊥`) and refuses to
  report numbers otherwise.

## Build · test · bench

```bash
cargo test --workspace                 # unit + doc + property tests (incl. end-to-end)
cargo clippy --workspace --all-targets -- -D warnings
./scripts/smoke.sh                     # fast, correctness-gated run of every bench

# Reproduce the CANS2026 Table 3 KPIR^index row (m = 10⁶, ℓ ∈ {32,256,1024} B):
./scripts/headtohead.sh                # appends to results/kpir-index/kpir_headtohead.csv

# One bench at one config:
./scripts/bench.sh kpir_answer --m 1000000 --value-bytes 256 --lwe-dim 1275
cargo bench -p kpir-index --bench headtohead -- --m 1000000 --value-bytes 32
```

Benches append one CSV row per config to
`${KPIR_RESULTS_BASE:-results}/kpir-index/<bench>.csv`.

**Note on scale.** The matrix grows with `ℓ`: at `ℓ = 1024 B`, m = 10⁶ the
transposed `u32` database is ~5.7 GiB (4× a byte-packed database — the price
of the RisePIR-comparable cell model). Budget RAM accordingly; see the †
footnote above for what happens when it doesn't fit.

## Paper ↔ code

| Paper (Hao et al. / `mpc4j`) | Code |
|------------------------------|------|
| Row-KOPIR `Setup/Query/Answer/Recover` | `SimplePirServer::{setup,answer}`, `SimplePirClient::{query,recover}` |
| `hint = A·D` | `SimplePirServer::setup` (`hint = D·A`, transposed layout) |
| `AKIM.Map_ε` / `Extract_ε` | `pla::KeyIndexMap::{build,extract}` |
| matrix `r × (c+2ε)`, `mpc4j getMatrixSize` | `params::MatrixShape::{new, choose}` |
| `ε`, fingerprint `DIGEST_BYTE_L` | `epsilon`, `FINGERPRINT_BYTES = 8` |
| `p = 2^pt` (adaptive), `q = 2³²`, `Δ = q/p` | `noise_bound_satisfied` / `MatrixShape::choose`, native `u32`, `SimpleParams::delta` |

## Security

This is **research benchmarking code, not a hardened PIR deployment**:

- The parameters (`N = 1275`, `σ = 6.4`, uniform `Z_q` secret) target
  128-bit security under the ADPS16 core-SVP model, but the code has had
  **no independent security audit**.
- The matvec kernel's schedule is data-independent (shape-only), and the
  `recover` decode is a branchless full-column scan (an OR-masked select that
  visits every candidate slot, leaking no matched-row timing) — but this is
  best-effort, not verified: no constant-time samplers or zeroization of
  secrets, and the underlying LWE decode is unaudited.
- `hash_key` is xxh3 (the reference uses truncated SHA-256): near-uniform
  and near-injective, which is all the scheme requires of it — it is a
  keyword→coordinate map, **not** a cryptographic commitment. Swap in a
  keyed/cryptographic hash before any adversarial-keyword setting.

## License

Dual-licensed under either of [Apache License 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.
