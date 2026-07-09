#!/usr/bin/env bash
# Shared helpers for the KPIR^index bench scripts. Sourced, never executed.
#
# Mirrors the RisePIR (`ikpir`) scripts: colored logging, a bench registry,
# and the head-to-head config matrix from CANS2026 Table 3.

# Repo root = parent of scripts/.
KPIR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log()  { printf '\033[0;34m[kpir]\033[0m %s\n' "$*"; }
ok()   { printf '\033[0;32m[ ok ]\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[0;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# All benches live in the kpir-index crate.
BENCHES=(headtohead kpir_answer kpir_query kpir_decode)

# CANS2026 Table 3 head-to-head configuration.
HEADTOHEAD_M=1000000               # m = 10^6 keys
HEADTOHEAD_VALUE_BYTES=(32 256 1024)
HEADTOHEAD_LWE_DIM=1275            # 128-bit security; matches RisePIR-S's SimplePIR backend
DEFAULT_EPSILON=4                  # mpc4j EPSILON

# Is $1 a known bench name?
is_known_bench() {
    local b
    for b in "${BENCHES[@]}"; do
        [[ "$b" == "$1" ]] && return 0
    done
    return 1
}
