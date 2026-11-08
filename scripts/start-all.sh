#!/usr/bin/env bash
# start-all.sh -- ONE entry point for the whole stack (delegator).
#
# The production entry point lives in kv-persistence:
#   observability -> persistent cache -> vLLM engine with dynamic hotsplit.
# This script just delegates, so `bash scripts/start-all.sh` works from this
# repository too (and from any working directory). The recipe's own
# scripts/run-tp4-hotsplit.sh remains the INDEPENDENT recipe-only launcher.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Round 59: tell the KV launcher WHERE the recipe is -- the delegate must not
# rely on ~/sikkatu-recipes or ~/recipes guesses. The recipe always lives next
# to this delegator (../recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit).
export RECIPE_DIR="$HERE/../recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit"
KVP_HOME="${KVP_REPO:-${KVP_REPO_ROOT:-$HOME/kv-persistence}}"   # both spellings accepted (round 63)
if [ ! -x "$KVP_HOME/scripts/start-all.sh" ] && [ ! -f "$KVP_HOME/scripts/start-all.sh" ]; then
  echo "ERROR: kv-persistence not found at $KVP_HOME" >&2
  echo "Clone it first: git clone https://github.com/sikkatu/kv-persistence ~/kv-persistence" >&2
  echo "Or set KVP_REPO=/path/to/kv-persistence" >&2
  exit 1
fi
echo "delegating to $KVP_HOME/scripts/start-all.sh (one stack: monitoring + cache + engine)"
exec bash "$KVP_HOME/scripts/start-all.sh" "$@"
