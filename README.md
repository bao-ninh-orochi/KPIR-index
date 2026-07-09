# KPIR^index — a Rust re-implementation for head-to-head keyword-PIR benchmarks

A faithful, **core-logic-only** Rust re-implementation of the **KPIR^index**
keyword-PIR construction of Hao et al., *"Practical Keyword Private Information
Retrieval from Key-to-Index Mappings"* (USENIX Security 2025), built to be
benchmarked **head-to-head** against ChalametPIR and RisePIR under a common
measurement model (CANS2026 Table 3).

## Why this repo

The reference implementation of KPIR^index lives in
[`mpc4j`](https://github.com/alibaba-edu/mpc4j) — it runs on the **JVM** and
uses a real **client-server transport** (serialization, sockets, RTT). RisePIR
and ChalametPIR are native Rust libraries that implement only the **core
cryptographic logic**. Comparing a networked JVM service against a native
in-process library conflates the algorithm with the runtime: in CANS2026
Table 3, KPIR^index answers a 1 kB-value query in **2439 ms**, most of which is
JVM + transport overhead, not computation.

This repo re-implements *only* the KPIR^index core logic in Rust — no
networking, no serialization — mirroring the RisePIR/ChalametPIR measurement
model, so the head-to-head comparison reflects the **algorithm**. On the same
1 kB config the native answer is **~315 ms** (≈8× faster than the JVM number),
over `Z_p` cells packed as `u32` exactly as RisePIR does.

### CANS2026 Table 3 (m = 10⁶ keys, N = 1275, ε = 4)

KPIR^index selects the plaintext width `pt` **adaptively** — the largest that
still decodes under the SimplePIR bound — exactly as RisePIR does, so the
head-to-head is measured at matched operating points:

| ℓ (value) | pt | Query upload | Response download | Expansion N/m | Answer (native Rust) | Answer (JVM, Table 3) |
|-----------|---:|-------------:|------------------:|--------------:|---------------------:|----------------------:|
| 32 B      | 9  | 24.00 kB     | 25.63 kB          | 1.068         | ~3.1 ms              | 68.5 ms               |
| 256 B     | 9  | 61.32 kB     | 72.38 kB          | 1.180         | ~31 ms               | 328 ms                |
| 1024 B    | 8  | 128.50 kB    | 177.50 kB         | 1.381         | ~315 ms              | 2439 ms               |

Query/response sizes and the expansion rate are implementation-independent;
answer latencies are hardware-dependent (Apple M-series here, `u32` cells as in
RisePIR — 4× the bytes of a byte-packed database, so the `ℓ = 1024 B` answer is
bandwidth-bound). At the reference width **`pt = 8`** the `columns`/`rows`
geometry is byte-identical to Hao Table 5 and `mpc4j`'s
`SimplePgmCpKsPirDesc.getMatrixSize` (32 B → 25.30/27.20 kB, 256 B →
65.00/77.09 kB, 1024 B → 128.50/177.50 kB) — pinned by
`params::tests::matches_mpc4j_byte_geometry`.

## The scheme

KPIR^index reduces keyword PIR to standard index PIR with two building blocks:

1. **Approximate key-to-index mapping (PLA / PGM)** — `pla`.
   Hash each keyword to a uniform 64-bit value, sort, and learn an optimal
   piece-wise linear approximation (Ferragina & Vinciguerra). For any key it
   returns a rank within `ε` (`+1` for the truncated intercept) of the true one.
2. **Row-KOPIR (SimplePIR)** — the [`simplepir`](simplepir/) crate.
   The database is a `Z_p` matrix `D ∈ Z_p^{R×C}` of `u32` cells
   (`p = 2^pt`, chosen adaptively); a query privately retrieves one whole
   **column**, which the keyword layer scans.

**Setup** sorts the `n` pairs by hashed key, learns the PLA map, and
repetition-encodes each `fingerprint(k) ‖ value` entry — bit-packed LSB-first
into `pt`-bit cells — into the matrix with an `ε+1` top / `ε+2` bottom boundary
so the true entry always lands in the single retrieved column. **Query**
extracts the approximate rank, maps it to a column, and issues one Row-KOPIR
query. **Recover** rounds & unpacks the returned column and returns the value
whose 64-bit fingerprint matches the queried key, else `⊥`.

See [`PLAN.md`](PLAN.md) for the full derivation and the exact encoding formula.

## Layout

```
Cargo.toml            workspace (2 crates, pinned toolchain 1.85.0)
simplepir/            Row-KOPIR (SimplePIR) index-PIR backend
  src/{lib,params,sampler,arith,matvec,backend}.rs
kpir-index/           keyword-PIR scheme + PLA + benchmarks
  src/{lib,params,pla,scheme}.rs
  benches/{helpers,headtohead,kpir_answer,kpir_query,kpir_decode}.rs
scripts/{bench,headtohead,smoke,lib}.sh
```

- **`simplepir`** — LWE math (all hand-rolled `u32`, `q = 2³²` via native
  wraparound): the shared `matvec` kernel, ChaCha-seeded `A` expansion,
  uniform-`Z_q` secret, discrete-Gaussian error (`σ = 6.4`), `Round_Δ`
  recovery, and the `SimplePirServer` / `SimplePirClient` protocol.
- **`kpir-index`** — the PLA (`pla`), the encoding + query/answer/recover glue
  (`scheme`), the matrix geometry (`params`), and the head-to-head benches.

## Parameters

| Parameter | Value | Note |
|-----------|-------|------|
| LWE modulus `q` | `2³²` | native `u32` wraparound |
| Plaintext modulus `p` | `2^pt`, **adaptive** (`pt = 9/9/8` for `ℓ = 32/256/1024 B`) | largest width that decodes (`noise_bound_satisfied`, `MAX_PLAINTEXT_BITS = 14`); matches RisePIR-S. `pt = 8` = `mpc4j` |
| Error `σ` | `6.4` | discrete Gaussian, uniform `Z_q` secret |
| LWE dim `N` | **1275** (default) | 128-bit security (ADPS16), matches RisePIR-S; `mpc4j` default is 1024 (`--lwe-dim 1024`) |
| Approx error `ε` | `4` | `mpc4j` `EPSILON` |
| Fingerprint | 8 bytes | disambiguates candidate entries |

## Build · test · bench

```bash
cargo test --workspace                 # unit + doctests (incl. end-to-end correctness)
./scripts/smoke.sh                     # fast, correctness-gated run of every bench

# Reproduce the CANS2026 Table 3 KPIR^index row (m = 10^6, ℓ ∈ {32,256,1024} B):
./scripts/headtohead.sh                # appends to results/kpir-index/kpir_headtohead.csv

# One bench at one config:
./scripts/bench.sh kpir_answer --m 1000000 --value-bytes 256 --lwe-dim 1275
cargo bench -p kpir-index --bench headtohead -- --m 1000000 --value-bytes 32
```

Benches append one CSV row per config to
`${KPIR_RESULTS_BASE:-results}/kpir-index/<bench>.csv`. Each bench first runs a
built-in correctness gate (present keys recover their exact value; absent keys
return `⊥`) and refuses to report numbers for a broken pipeline.

**Note on scale.** The matrix and setup grow with `ℓ`: at `ℓ = 1024 B` the
transposed `u32` database is ~5.7 GiB (4× a byte-packed DB — the price of the
RisePIR-comparable cell model, and the reason that answer is bandwidth-bound).
The benches skip materialising the full `hint = D·A` (a `Θ(R·C·N)` offline
cost) — the client's recover offset `hint·s` is computed as `D·(A·s)` in one
matvec via the co-located server, which is bit-identical to using the real
hint — so a full head-to-head sweep runs in a few minutes. The hint's on-wire
size (`R·N·4`, the offline setup download) is reported analytically.

## Paper ↔ code

| Paper (Hao et al. / `mpc4j`) | Code |
|------------------------------|------|
| Row-KOPIR `Setup/Query/Answer/Recover` | `SimplePirServer::{setup,answer}`, `SimplePirClient::{query,recover}` |
| `hint = A·D` | `SimplePirServer::setup` (`hint = D·A`, transposed layout) |
| `AKIM.Map_ε` / `Extract_ε` | `pla::KeyIndexMap::{build,extract}` |
| matrix `r × (c+2ε)`, `mpc4j getMatrixSize` | `params::MatrixShape::{new, choose}` |
| `ε`, fingerprint `DIGEST_BYTE_L` | `epsilon`, `FINGERPRINT_BYTES = 8` |
| `p = 2^pt` (adaptive), `q = 2³²`, `Δ = q/p` | `noise_bound_satisfied` / `MatrixShape::choose`, native `u32`, `SimpleParams::delta` |

## License

MIT OR Apache-2.0.
