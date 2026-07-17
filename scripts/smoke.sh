#!/usr/bin/env bash
# Fast correctness/smoke check: run every bench at a tiny config. Each bench
# gates on the built-in `verify()` (present keys recover their exact value,
# absent keys return ⊥), so a non-zero exit means a real correctness break.
#
# Usage:  ./scripts/smoke.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

# Write to a throwaway base so smoke's tiny rows never touch the real
# results/kpir-index/ CSVs (benches append). Cleared at start and end.
SMOKE_BASE="$KPIR_ROOT/results/.smoke"
export KPIR_RESULTS_DIR="$SMOKE_BASE/kpir-index"
rm -rf "$SMOKE_BASE"
mkdir -p "$KPIR_RESULTS_DIR"

# Tiny criterion knobs (minimum sample count, sub-second warm-up/measurement)
# keep the smoke pass to a couple of minutes while still gating on verify().
ARGS=(--m 20000 --value-bytes 32 --lwe-dim 512 --epsilon 4 --batch 4 \
      --sample-size 10 --warmup-secs 0.5 --measurement-secs 0.5)
log "smoke: all benches at ${ARGS[*]}"
for b in "${BENCHES[@]}"; do
    if out=$(cargo bench -p kpir-index --bench "$b" -- "${ARGS[@]}" 2>&1); then
        ok "  $b"
    else
        printf '%s\n' "$out" >&2
        die "$b failed"
    fi
done
rm -rf "$SMOKE_BASE"
ok "smoke passed"
