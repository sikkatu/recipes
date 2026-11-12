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
exec python3 "$HERE/concurrency_sweep.py" "$@"