#!/usr/bin/env bash
# concurrency-sweep.sh -- measure how many PARALLEL streaming requests the
# system holds (N = 1..10 default levels) and where the "knee" sits.
#
# Part C (round 87) of the MAX_NUM_SEQS=10 change. The script drives an
# ALREADY RUNNING engine -- it never restarts the engine and writes nothing
# to the configs or the LMCache (see the warning printed at startup).
#
# Usage: bash scripts/concurrency-sweep.sh [--levels 1,2,4,6,8,10] [options]
#        (see --help of the Python tool for every knob)
#
# All the logic lives in concurrency_sweep.py (stdlib only -- no pip installs,
# works offline once the engine is up).
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# r90 C5: ONE warning before the start when the engine attaches LMCache
# (KV_PERSIST=1 -> --kv-transfer-config contains LMCacheMPConnector) -- the
# KNOWN upstream race (LMCache #4185 / vLLM #49089) killed the engine at
# N>=2 concurrency on this path. It is fixed by the custom LMCache build
# (downstream PR #4136, see kv-persistence/vendor/patches/lmcache/); verify
# the fix on hardware before trusting high-N measurements.
CONTAINER="${CONTAINER:-}"
if [ -z "$CONTAINER" ] && [ -f "$HERE/../../live/engine_container" ]; then
  CONTAINER="$(head -n1 "$HERE/../../live/engine_container" | tr -d '[:space:]')"
fi
CONTAINER="${CONTAINER:-mimo-pro-tp4-hotsplit}"
if command -v docker >/dev/null 2>&1 && docker inspect "$CONTAINER" >/dev/null 2>&1; then
  if docker inspect -f '{{json .Config.Cmd}}' "$CONTAINER" 2>/dev/null | grep -qi 'LMCache'; then
    echo "WARNING: $CONTAINER runs WITH LMCache (--kv-transfer-config)." >&2
    echo "  Known upstream issue: at N>=2 a second parallel request used to kill" >&2
    echo "  the engine (LMCache #4185 / vLLM #49089). FIXED by the custom LMCache" >&2
    echo "  build (downstream PR #4136) -- verify on hardware before trusting N>=2." >&2
  fi
fi
exec python3 "$HERE/concurrency_sweep.py" "$@"