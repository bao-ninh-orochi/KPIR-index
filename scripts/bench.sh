#!/usr/bin/env bash
# Run ONE bench at ONE config.
#
# Usage:
#   ./scripts/bench.sh <bench> [--m N] [--value-bytes N] [--lwe-dim N]
#                              [--epsilon N] [--batch N] [--seed N]
#                              [--sample-size N] [--warmup-secs S] [--measurement-secs S]
#
#   <bench> ∈ {headtohead, kpir_answer, kpir_query, kpir_decode}
#
# Any flags after the bench name are forwarded to the bench binary
# (see `Cli` in kpir-index/benches/helpers.rs for defaults). Results are
# appended to ${KPIR_RESULTS_BASE:-<repo>/results}/kpir-index/<bench>.csv.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

BENCH="${1:-}"
[[ -n "$BENCH" ]] || die "usage: bench.sh <bench> [flags...]  (benches: ${BENCHES[*]})"
shift
is_known_bench "$BENCH" || die "unknown bench '$BENCH' (one of: ${BENCHES[*]})"

RESULTS_BASE="${KPIR_RESULTS_BASE:-$KPIR_ROOT/results}"
export KPIR_RESULTS_DIR="$RESULTS_BASE/kpir-index"
mkdir -p "$KPIR_RESULTS_DIR"

log "running $BENCH  ->  $KPIR_RESULTS_DIR/$BENCH.csv"
cargo bench -p kpir-index --bench "$BENCH" -- "$@"
ok "done"
