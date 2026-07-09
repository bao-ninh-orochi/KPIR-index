# Contributing

Thanks for your interest in KPIR^index! This document covers the mechanics of
getting a change accepted; the architecture itself is documented in the root
[`CLAUDE.md`](CLAUDE.md), the [README](README.md), and the per-crate
`README.md` files.

## Toolchain

The workspace pins Rust **1.85.0** via [`rust-toolchain.toml`](rust-toolchain.toml)
(rustup selects it automatically). 1.85 is also the declared MSRV
(`rust-version` in the root manifest). Bump the pin deliberately and in its own
commit — the paper's benchmark numbers are measured on the pinned channel.

## Local gates

Every change must pass, from the workspace root:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo bench --workspace --no-run --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps   # rustdoc stays warning-free
./scripts/smoke.sh                                           # correctness-gated bench smoke
```

The first four mirror CI (`.github/workflows/ci.yml`); the rustdoc check is a
project convention (see [`CLAUDE.md`](CLAUDE.md)), and `smoke.sh` runs every
bench once at a tiny config, gated on its built-in correctness verifier (present
keys recover exactly, absent keys yield ⊥) — a broken pipeline fails, not just
the numbers.

## Benches

- One bench at one config: `./scripts/bench.sh <name> [flags]` (routes CSV
  output under `results/kpir-index/`).
- Reproduce the CANS2026 Table 3 sweep: `./scripts/headtohead.sh`.
- Fast correctness pass over every bench: `./scripts/smoke.sh`.
- Never commit `results/` — it is git-ignored and regenerated on demand. The
  bench CSV schema is append-only; archive old-schema CSVs, don't mix them.

## Workspace conventions

- Crates live under `crates/`; version, edition, MSRV, license, lints, and
  `publish = false` are inherited from the root manifest — a new crate opts in
  with `.workspace = true` keys and `[lints] workspace = true`.
- Add new dependencies to `[workspace.dependencies]` in the root manifest
  first, then reference them with `{ workspace = true }`. Keep the external
  dependency set lean.
- **Fairness invariants** — these keep the head-to-head numbers comparable with
  RisePIR; do not regress them: single-threaded, no rayon/ndarray/explicit
  SIMD/external crypto; the `simplepir` matvec kernel is a bit-identical port of
  RisePIR's; `plaintext_bits` is chosen adaptively by the same decode bound over
  the same `u32` cell model. See [`CLAUDE.md`](CLAUDE.md) "Load-bearing facts".
- Invariant violations use `assert!`/`panic!` with a message; the core logic has
  no error enums or `Result` public API by design (no `thiserror`/`anyhow`).

## Branches, commits, PRs

- Branch names: `feature/…`, `perf/…`, `fix/…`, `chore/…`, `refactor/…`,
  `docs/…`.
- Commit style: `type(scope): summary` (lowercase summary), e.g.
  `perf(simplepir): tighten the blocked matvec inner loop`.
- Keep each commit building green; the local gates above must pass.

## Licensing

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as in [README § License](README.md#license), without any
additional terms or conditions.
