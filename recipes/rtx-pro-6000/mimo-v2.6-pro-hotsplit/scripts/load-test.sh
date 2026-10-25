#!/bin/bash
# load-test.sh — mixed-traffic load run for the hot/cold split: decode tok/s + TTFT per
# topic and the hot-bank hit rate over time. Run from
# recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit/ while the server is up.
#
# Usage: bash scripts/load-test.sh [rounds]        (default 5 rounds x 10 topics = 50 requests)
#
# Per request (streaming): gen_tok, TTFT (time to first token = prompt processing + queue),
# decode tok/s (pure generation speed, separated from TTFT) and wall time. The summary
# prints per-topic and OVERALL decode speed + TTFT. Results also land in
# live/bench_results.json so runs can be compared later (e.g. dynamic vs static).
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

ROUNDS="${1:-5}"
CONTAINER="${CONTAINER:-mimo-pro-tp4-hotsplit}"

python3 scripts/bench_load.py "$ROUNDS" http://127.0.0.1:30007 live/api_key

echo
echo "=== dynamic hit-rate evolution (from container log) ==="
docker logs "$CONTAINER" 2>&1 | grep 'hit-rate' | tail -10
echo
echo "=== dynamic updates so far ==="
docker logs "$CONTAINER" 2>&1 | grep 'hotsplit dynamic update' | tail -6