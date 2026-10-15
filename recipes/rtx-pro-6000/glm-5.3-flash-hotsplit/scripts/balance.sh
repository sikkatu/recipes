#!/usr/bin/env bash
# Compute the expert balance (hot/cold split) for this model. No GPU needed.
# usage: balance.sh [--counts traffic.json] [--weights decode=1,prefill=0] [--gib N] [--json]
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIB="${HOTSPLIT_HOT_GIB:-31}"   # per rank; default = the mid rung of this recipe's ladder
exec python3 "$HERE/../patches/hotsplit_balance.py" "$HERE/../balance.json" --gib "$GIB" "$@"
