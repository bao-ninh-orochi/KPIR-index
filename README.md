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
1 kB config the native answer is **~124 ms** (≈20× faster than the JVM number),
while every communication figure matches the reference **to the byte**.

### Reproduction of CANS2026 Table 3 (m = 10⁶ keys, N = 1275, ε = 4)

| ℓ (value) | Query upload | Response download | Expansion N/m | Answer (native Rust) | Answer (JVM, Table 3) |
|-----------|-------------:|------------------:|--------------:|---------------------:|----------------------:|
| 32 B      | 25.30 kB     | 27.20 kB          | 1.075         | ~2.5 ms              | 68.5 ms               |
| 256 B     | 65.00 kB     | 77.09 kB          | 1.186         | ~19 ms               | 328 ms                |
| 1024 B    | 128.50 kB    | 177.50 kB         | 1.381         | ~124 ms              | 2439 ms               |

Query/response sizes and the expansion rate are implementation-independent and
match the reference exactly; answer latencies are hardware-dependent (measured
on an Apple M-series laptop here). The `columns`/`rows` geometry equals Hao
Table 5 and `mpc4j`'s `SimplePgmCpKsPirDesc.getMatrixSize`.

## The scheme

KPIR^index reduces keyword PIR to standard index PIR with two building blocks:

1. **Approximate key-to-index mapping (PLA / PGM)** — `pla`.
   Hash each keyword to a uniform 64-bit value, sort, and learn an optimal
   piece-wise linear approximation (Ferragina & Vinciguerra). For any key it
   returns a rank within `ε` (`+1` for the truncated intercept) of the true one.
2. **Row-KOPIR (SimplePIR)** — the [`simplepir`](simplepir/) crate.
   The database is a byte matrix `D ∈ Z_p^{R×C}` (`p = 256`); a query privately
   retrieves one whole **column**, which the keyword layer scans.

**Setup** sorts the `n` pairs by hashed key, learns the PLA map, and
repetition-encodes each `fingerprint(k) ‖ value` entry into the matrix with an
`ε+1` top / `ε+2` bottom boundary so the true entry always lands in the single
retrieved column. **Query** extracts the approximate rank, maps it to a column,
and issues one Row-KOPIR query. **Recover** rounds the returned column and
returns the value whose 8-byte fingerprint matches the queried key, else `⊥`.

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
| Plaintext modulus `p` | `2⁸ = 256` | one byte per `Z_p` cell; `mpc4j` fixes this |
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
transposed database is ~1.4 GiB. The benches skip materialising the full
`hint = D·A` (a `Θ(R·C·N)` offline cost) — the client's recover offset `hint·s`
is computed as `D·(A·s)` in one matvec via the co-located server, which is
bit-identical to using the real hint — so a full head-to-head sweep runs in
under a minute. The hint's on-wire size (`R·N·4`, the offline setup download)
is reported analytically.

## Paper ↔ code

| Paper (Hao et al. / `mpc4j`) | Code |
|------------------------------|------|
| Row-KOPIR `Setup/Query/Answer/Recover` | `SimplePirServer::{setup,answer}`, `SimplePirClient::{query,recover}` |
| `hint = A·D` | `SimplePirServer::setup` (`hint = D·A`, transposed layout) |
| `AKIM.Map_ε` / `Extract_ε` | `pla::KeyIndexMap::{build,extract}` |
| matrix `r × (c+2ε)`, `mpc4j getMatrixSize` | `params::MatrixShape` |
| `ε`, fingerprint `DIGEST_BYTE_L` | `epsilon`, `FINGERPRINT_BYTES = 8` |
| `p = 2⁸`, `q = 2³²`, `Δ = q/p` | `plaintext_bits = 8`, native `u32`, `SimpleParams::delta` |

## License

MIT OR Apache-2.0.
