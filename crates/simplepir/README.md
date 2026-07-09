# simplepir

Row-KOPIR — the single-server, LWE-based **index**-PIR backend underneath
[KPIR^index](../../README.md), instantiated from SimplePIR (Henzinger et al.,
USENIX Security 2023). Core logic only: no networking, no serialization,
no external crypto/linear-algebra crates — all math is hand-rolled `u32`
so head-to-head numbers against RisePIR and ChalametPIR are measured
under the same model.

## Role in the workspace

```
kpir-index ──► simplepir
```

`kpir-index` (the keyword layer) encodes its key-value matrix into this
crate's row-major database, then drives `setup → query → answer →
recover`. Nothing here knows about keywords, fingerprints, or the PLA.

## Protocol

The database is `D ∈ Z_p^{C×R}` (`p = 2^plaintext_bits`, each cell a
`u32` in `[0, p)`). Following the FrodoPIR convention (as RisePIR does), a
query privately selects one **row** via the left-multiply `ans = qu·D`;
the answer returns all `R` cells of it. All arithmetic is mod `q = 2³²`
via native `u32` wraparound; `Δ = q/p`. The public matrix `A ∈ Z_q^{C×N}`
is expanded from a 16-byte seed and never shipped.

| Step | Computation | Where |
|---|---|---|
| Setup | `H = Aᵀ·D ∈ Z_q^{N×R}` (client downloads once) | `SimplePirServer::setup` |
| KeyGen | `s ← Z_q^N`; `a_s = A·s`, `h_s = sᵀ·H` | `SimplePirClient::new` |
| Query | `qu = a_s + e + Δ·u_col ∈ Z_q^C`, `e ← χ^C` | `SimplePirClient::query` |
| Answer | `ans = qu·D ∈ Z_q^R` | `SimplePirServer::answer` |
| Recover | `Round_Δ(ans − h_s)` → row `col` | `SimplePirClient::recover` |

This `ans = qu·D` form is the FrodoPIR convention and matches the `mpc4j`
reference (`SimpleCpIdxPir`).

## What's here

| Module | Items |
|---|---|
| `backend` | `SimplePirServer` (row-major `C×R` `u32` DB; `setup`/`answer`; decode-bound guard in `from_row_major_db`), `SimplePirClient` (`new` from a real hint, or `with_local_server` — the bit-identical fast path for co-located benchmarks), `Hint` |
| `params` | `SimpleParams` (runtime knobs) / `SimpleConfig` (user-facing knobs), `noise_bound_satisfied` — the SimplePIR Gaussian decode bound shared by the backend guard and the `kpir-index` operating-point selector, `MAX_PLAINTEXT_BITS = 14` |
| `matvec` | the shared register-blocked `acc += qᵀ·D` kernel behind every op — `answer` (`qu·D`), `setup` (`Aᵀ·D`), the `A·s` / `sᵀ·H` precompute — **bit-identical** to the RisePIR kernel (width-adaptive blocking, `R` rows per pass with `R·width ≤ 2048`). Measured single-threaded on Apple M1 this beats ChalametPIR's column-major dot product (28–37% when the accumulator fits L1; ChalametPIR's edge was rayon parallelism, which this crate forgoes) |
| `sampler` | ChaCha20-seeded `A` expansion (transposed `N×C` layout = `Aᵀ`), uniform-`Z_q` secret, Box–Muller discrete Gaussian |
| `arith` | `round_q_to_p` (`Round_Δ`), width-generic; equals `mpc4j`'s byte recovery at `plaintext_bits = 8` |

## Parameters

- `q = 2³²` (implicit), secret uniform over `Z_q`, error discrete
  Gaussian `σ = 6.4` — the SimplePIR §4.2 instantiation.
- `N = 1275` default (128-bit security under the ADPS16 core-SVP model,
  matching RisePIR-S; the original `mpc4j` build uses `N = 1024`).
- `plaintext_bits` is **adaptive**, not fixed: the largest width `≤ 14`
  that satisfies `noise_bound_satisfied` for the geometry's summation
  dimension (`δ = 2⁻⁴⁰`). `from_row_major_db` refuses geometries that
  violate the bound, so a mis-chosen width fails loudly at build time.

## Status

Research-grade core logic for benchmarking; see the
[workspace README](../../README.md#security) for the security notice.
Everything is single-threaded with no explicit SIMD — the answer's perf
comes from the register-blocked kernel (RisePIR's approach) and the
autovectorizer.
