# KPIR^index (Hao et al., USENIX Sec'25) — Rust re-implementation plan

Faithful, **core-logic-only** (no network / serialization) Rust port of the
`KPIR^index` keyword-PIR construction, built for a **head-to-head** benchmark
against ChalametPIR and RisePIR (CANS2026 Table 3). Mirrors the RisePIR
(`../Incremental-Keyword-PIR`) repo conventions.

## The scheme (Hao et al. §6, Fig. 8; mpc4j `SimplePgmCpKsPir`)

KPIR^index = **Row-KOPIR (SimplePIR)** + **approximate key-to-index map (PLA)**.

### Fixed parameters (mpc4j `SimplePgmCpKsPirDesc` + CANS2026 §6.3)
- LWE modulus `q = 2^32` (native u32 wraparound).
- Plaintext modulus `p = 2^pt` **adaptive** (unified with RisePIR-S): `pt` is the
  largest width `≤ MAX_PLAINTEXT_BITS=14` whose geometry still decodes under the
  SimplePIR Gaussian bound `Δ = q/p ≥ 2√2·σ·√(ln(2/δ))·p·√C`, `δ=2⁻⁴⁰`
  (`simplepir::noise_bound_satisfied`; noise dim `C = columns`). Selects
  `pt = 9/9/8` for `ℓ = 32/256/1024 B` at `m=10⁶`. `pt = 8` (one byte per cell)
  is the mpc4j special case.
- Gaussian error `σ = 6.4` (discrete), uniform secret over Z_q.
- LWE dim `N = 1275` (default; 128-bit sec, matches RisePIR-S; mpc4j default is 1024).
- Approx error `ε = 4`. Fingerprint length `f = 8` bytes (64 bits, independent of `pt`).

### Matrix layout (mpc4j formula generalised to variable `pt`)
For `n` key-value pairs, value length `ℓ` bytes, payload `payload_bits = 8·(ℓ+8)`:
```
partition = ceil( payload_bits / pt )   # Z_p cells per entry (= ℓ+8 at pt=8)
columns   = ceil( sqrt(n · partition) ) # = query upload dimension
dataRows  = max( ceil(n / columns), ε+2 )
rows      = dataRows + 2ε + 3           # = dataRows + 11  (response = rows · partition)
```
Column-major placement, entry `i` (i-th smallest hashed key) → `col = i/dataRows`,
`row = (i%dataRows) + ε + 1`. Boundary rows duplicate neighbouring columns' edge
entries so the ±ε window always stays inside the retrieved column. Equivalent
closed form used here:
```
slot(col, row) = sorted_entry[ g ]   where g = col*dataRows + (row - ε - 1)
                 or ⊥ if g ∉ [0, n)
```
where `sorted_entry[g] = fingerprint64(key_g) ‖ value_g` is bit-packed LSB-first
into `partition` `pt`-bit `Z_p` cells (`write_bits`/`read_bits`); ⊥ = every cell
`p−1` (reads back as fingerprint `u64::MAX`).

Verified: m=10^6 → mpc4j pt=8 32B=25.3/27.2kB, 256B=65.0/77.1kB, 1024B=128.5/177.5kB;
adaptive 32B(pt9)=24.0/25.6kB, 256B(pt9)=61.3/72.4kB, 1024B(pt8)=128.5/177.5kB.

### Row-KOPIR (SimplePIR, column-selection form)
Database `D` is a single `R × C` `u32` `Z_p` matrix (cells in `[0, p)`),
`R = rows·partition`, `C = columns`. Public `A ∈ Z_q^{C×N}` sampled from a seed.
- **Setup**: `hint = D · A ∈ Z_q^{R×N}` (client downloads it).
- **Client keygen** (per secret `s ← Z_q^N`): `a_s = A·s ∈ Z_q^C`, `h_s = hint·s ∈ Z_q^R`.
- **Query(col)**: `qu = a_s + e + Δ·u_col ∈ Z_q^C`, `e ← χ^C`.
- **Answer(qu)**: `ans = D · qu ∈ Z_q^R`.
- **Recover**: `d = ans − h_s`; each cell rounds `Δ·plaintext + noise → plaintext`
  via `round(x) = (x + Δ/2) >> (32−pt)  (mod p)` (`Δ = 2^(32−pt)`). Yields column
  `col` = `rows` slots × `partition` cells, which the keyword layer bit-unpacks.

### Approximate key-to-index map (PLA / O'Rourke greedy, ε=4)
1. Hash each key to a uniform 8-byte value (u64); sort ascending (distinct).
2. Build optimal ε-PLA over points `(hashed_key, index)`: greedy convex-hull
   segment learner; each segment `(first_key, slope, intercept)`, `d ≈ 0.013n`.
3. **Extract(k)**: hash k → find segment (binary search) → `pos = slope·(k−first)+intercept`,
   clamp to `[0,n)`. Guarantee `|pos − true_index| ≤ ε`.

### Query/Answer/Recover glue
- `Query(k)`: `pos = Extract(k)`; `col = pos / dataRows`; Row-KOPIR.Query(col).
- `Answer`: Row-KOPIR.Answer.
- `Recover(k)`: Row-KOPIR.Recover → `rows` entries; scan for the entry whose
  8-byte fingerprint == `fingerprint8(k)`; return its value, else ⊥.

## Repo structure (mirrors RisePIR)
```
Cargo.toml (workspace)  rust-toolchain.toml (1.85.0)  rustfmt.toml  .gitignore  LICENSE
simplepir/   src/{lib,params,sampler,arith,matvec,backend}.rs   # Row-KOPIR backend
kpir-index/  src/{lib,pla,scheme,params}.rs                     # keyword PIR + PLA
             benches/{helpers,kpir_answer,kpir_query,kpir_decode,headtohead}.rs
             tests/e2e.rs
scripts/     {bench.sh, smoke.sh, lib.sh}
```

## Methodology (matches RisePIR for fair head-to-head)
- Single-threaded, no SIMD intrinsics, no rayon. `q=2^32` via u32 wraparound.
- Communication measured as fixed-width LE wire bytes: query = `C·4`,
  response = `R·4`, hint = `R·N·4` (A regenerated from seed, not shipped).
- Timing via criterion `iter_custom`; report answers/sec + latency.
- Head-to-head configs: `m ∈ {10^6}`, `ℓ ∈ {32, 256, 1024} B`, N=1275, ε=4, `pt`
  adaptive (9/9/8).

## Correctness checks
- End-to-end: build DB, query each present key → recovers correct value; absent key → ⊥.
- Noise margin: recovered `d` decodes with `|noise| ≪ Δ/2 = 2^(31−pt)` at all configs;
  the adaptive `pt` is the largest width that keeps this margin (backend guard).
</content>
