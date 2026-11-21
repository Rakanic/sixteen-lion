#!/usr/bin/env bash
# Run tools/shading_rounding.py on every checkpoint of a sweep, as soon as
# each one finishes training. Start several workers; each claims a run with
# an atomic mkdir, so no run is done twice.
#   tools/rounding_worker.sh out/rounding-base 48
set -u
sweep=$1; total=$2; shift 2
while :; do
  claimed=0
  for m in $(find "$sweep" -name model.pt | sort); do
    run=$(dirname "$m")
    [ -f "$run/metrics.json" ] || continue
    mkdir "$run/rounding.lock" 2>/dev/null || continue
    claimed=1
    echo "[$(date +%T)] $run"
    python tools/shading_rounding.py "$run" "$@" > "$run/rounding.log" 2>&1 || echo "  FAILED (see $run/rounding.log)"
    grep -E "^(nearest|texel|shading) " "$run/rounding.log"
  done
  done_n=$(find "$sweep" -name rounding.lock | wc -l)
  [ "$claimed" = 0 ] && [ "$done_n" -ge "$total" ] && break
  [ "$claimed" = 0 ] && sleep 30
done
