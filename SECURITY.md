# Security Policy

## Scope and status

This repository is the research-prototype artifact accompanying the CANS 2026
head-to-head PIR measurements — a **core-logic-only** Rust re-implementation of
the KPIR^index keyword-PIR construction (Hao et al., USENIX Security 2025),
built to be benchmarked against ChalametPIR and RisePIR under a common model.
It has **not** been independently audited and is **not** intended for production
deployments handling real user data.

Known, deliberate limitations of the prototype:

- **No serialization or transport.** Only the cryptographic pipeline exists;
  there is no networking, serialization, or persistence layer. Any deployment
  must add — and independently harden — its own encoding and transport.
- **Side-channel posture is best-effort, not audited.** The LWE matrix-vector
  kernel's schedule is data-independent — it depends only on the public matrix
  shape, never on secret query or database values. The `recover` decode scans
  *all* candidate slots of the retrieved column and merges the value via a
  branchless OR-masked select (`ct_eq_u64_mask`), so its probe path is
  independent of which slot (if any) matches — a co-located timing observer
  learns nothing beyond the public geometry. This is best-effort hand-rolled
  masking, **not** a formally verified constant-time guarantee: the samplers are
  not constant-time, secrets are not zeroized, and the underlying SimplePIR LWE
  decode is a separate, un-audited concern.
- **`hash_key` is xxh3, not a cryptographic hash.** It is a near-uniform,
  near-injective keyword→coordinate map — all the scheme requires — **not** a
  cryptographic commitment. Swap in a keyed/cryptographic hash before any
  adversarial-keyword setting.

See the [README security notice](README.md#security) for the parameter-level
security target (128-bit under ADPS16) with these caveats in context.

## Reporting a vulnerability

Please **do not open a public issue** for security-relevant findings. Instead,
use one of:

- GitHub's private vulnerability reporting on
  [`orochi-network/KPIR-index`](https://github.com/orochi-network/KPIR-index/security/advisories)
  ("Report a vulnerability"), or
- email the maintainer: <bao.ninh@orochi.network>.

## Supported versions

Pre-1.0: only the `main` branch receives fixes. Response and remediation are
best-effort; there is no SLA while the project is a paper artifact.
