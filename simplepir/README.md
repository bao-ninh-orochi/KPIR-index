# simplepir

Row-KOPIR — the single-server, LWE-based **index**-PIR backend underneath
[KPIR^index](../README.md), instantiated from SimplePIR (Henzinger et al.,
USENIX Security 2023). Core logic only: no networking, no serialization,
no external crypto/linear-algebra crates — all math is hand-rolled `u32`
so head-to-head numbers against RisePIR and ChalametPIR are measured
under the same model.

## Role in the workspace

```
kpir-index ──► simplepir
```

`kpir-index` (the keyword layer) encodes its key-value matrix into this
crate's transposed database, then drives `setup → query → answer →
recover`. Nothing here knows about keywords, fingerprints, or the PLA.

## Protocol

The database is `D ∈ Z_p^{R×C}` (`p = 2^plaintext_bits`, each cell a
`u32` in `[0, p)`). A query privately selects one **column**; the answer
returns all `R` cells of it. All arithmetic is mod `q = 2³²` via native
`u32` wraparound; `Δ = q/p`. The public matrix `A ∈ Z_q^{C×N}` is
expanded from a 16-byte seed and never shipped.

| Step | Computation | Where |
|---|---|---|
| Setup | `hint = D·A ∈ Z_q^{R×N}` (client downloads once) | `SimplePirServer::setup` |
| KeyGen | `s ← Z_q^N`; `a_s = A·s`, `h_s = hint·s` | `SimplePirClient::new` |
| Query | `qu = a_s + e + Δ·u_col ∈ Z_q^C`, `e ← χ^C` | `SimplePirClient::query` |
| Answer | `ans = D·qu ∈ Z_q^R` | `SimplePirServer::answer` |
| Recover | `Round_Δ(ans − h_s)` → column `col` | `SimplePirClient::recover` |

This column-selection form is the transpose of the paper's Figure 2
presentation and matches the `mpc4j` reference (`SimpleCpIdxPir`).

## What's here

| Module | Items |
|---|---|
| `backend` | `SimplePirServer` (transposed `C×R` `u32` DB; `setup`/`answer`; decode-bound guard in `from_transposed_db`), `SimplePirClient` (`new` from a real hint, or `with_local_server` — the bit-identical fast path for co-located benchmarks), `Hint` |
| `params` | `SimpleParams` (runtime knobs) / `SimpleConfig` (user-facing knobs), `noise_bound_satisfied` — the SimplePIR Gaussian decode bound shared by the backend guard and the `kpir-index` operating-point selector, `MAX_PLAINTEXT_BITS = 14` |
| `matvec` | the shared register-blocked `acc += qᵀ·D` kernel behind every hot loop — **bit-identical** to the RisePIR kernel (same width-adaptive blocking rule, `R` rows per pass with `R·width ≤ 2048` cells), so the two schemes are benchmarked on the same inner loop |
| `sampler` | ChaCha20-seeded `A` expansion (transposed `N×C` layout), uniform-`Z_q` secret, Box–Muller discrete Gaussian |
| `arith` | `round_q_to_p` (`Round_Δ`), width-generic; equals `mpc4j`'s byte recovery at `plaintext_bits = 8` |

## Parameters

- `q = 2³²` (implicit), secret uniform over `Z_q`, error discrete
  Gaussian `σ = 6.4` — the SimplePIR §4.2 instantiation.
- `N = 1275` default (128-bit security under the ADPS16 core-SVP model,
  matching RisePIR-S; the original `mpc4j` build uses `N = 1024`).
- `plaintext_bits` is **adaptive**, not fixed: the largest width `≤ 14`
  that satisfies `noise_bound_satisfied` for the geometry's summation
  dimension (`δ = 2⁻⁴⁰`). `from_transposed_db` refuses geometries that
  violate the bound, so a mis-chosen width fails loudly at build time.

## Status

Research-grade core logic for benchmarking; see the
[workspace README](../README.md#security) for the security notice.
Everything is single-threaded with no explicit SIMD — perf comes from
the register-blocked kernel and the autovectorizer, exactly as in the
RisePIR reference.
