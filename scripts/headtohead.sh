#!/usr/bin/env bash
# Reproduce the KPIR^index row of CANS2026 Table 3: the head-to-head
# online cost at m = 10^6 keys across value sizes {32, 256, 1024} B.
#
# Usage:  ./scripts/headtohead.sh [m] [extra bench flags...]
#   m defaults to 1000000. Extra flags (e.g. --samples 30) are forwarded.
#
# Appends one row per value size to
#   ${KPIR_RESULTS_BASE:-<repo>/results}/kpir-index/kpir_headtohead.csv
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

M="${1:-$HEADTOHEAD_M}"
shift || true

RESULTS_BASE="${KPIR_RESULTS_BASE:-$KPIR_ROOT/results}"
export KPIR_RESULTS_DIR="$RESULTS_BASE/kpir-index"
mkdir -p "$KPIR_RESULTS_DIR"

log "KPIR^index head-to-head: m=$M, l in {${HEADTOHEAD_VALUE_BYTES[*]}}B, N=$HEADTOHEAD_LWE_DIM, eps=$DEFAULT_EPSILON"
for vb in "${HEADTOHEAD_VALUE_BYTES[@]}"; do
    log "  value_bytes=$vb"
    cargo bench -p kpir-index --bench headtohead -- \
        --m "$M" --value-bytes "$vb" --lwe-dim "$HEADTOHEAD_LWE_DIM" \
        --epsilon "$DEFAULT_EPSILON" "$@"
done
ok "sweep done -> $KPIR_RESULTS_DIR/kpir_headtohead.csv"
