#!/usr/bin/env bash
# Rollback: stop and remove the hotsplit container, then optionally start the plain UVA baseline.
# HONEST SCOPE: the launchers write to three places OUTSIDE the container, and rollback does NOT
# undo them:
#   - the Hugging Face cache ($HF_CACHE, default ~/.cache/huggingface) -- downloaded model
#     weights and tokenizer files persist;
#   - $LIVE_DIR (default ./live) -- routing counters, timestamps and snapshots persist;
#   - the recipe checkout itself (git tree) is never modified at runtime.
# So "rollback" here means: the server container and any in-container state are gone. Your model
# cache and diagnostic counters are preserved deliberately. Use manage.sh clean-cache to remove
# cached model data after showing its size and asking for confirmation.
# usage: rollback.sh [--start-baseline]
set -Eeuo pipefail
die(){ echo "ERROR: $*" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOTSPLIT_CONTAINER="${HOTSPLIT_CONTAINER:-mimo-pro-tp4-hotsplit}"
BASELINE_CONTAINER="${BASELINE_CONTAINER:-mimo-pro-tp4-uva}"
START_BASELINE=0
[ "${1:-}" = "--start-baseline" ] && START_BASELINE=1
command -v docker >/dev/null || die "Docker is missing."
docker info >/dev/null 2>&1 || die "Docker daemon unavailable or permission denied."
if docker container inspect "$HOTSPLIT_CONTAINER" >/dev/null 2>&1; then
  echo "Stopping and removing $HOTSPLIT_CONTAINER ..."
  docker stop "$HOTSPLIT_CONTAINER" >/dev/null 2>&1 || true
  docker rm "$HOTSPLIT_CONTAINER" >/dev/null 2>&1 || true
  echo "Removed $HOTSPLIT_CONTAINER"
else
  echo "No hotsplit container ($HOTSPLIT_CONTAINER) present - nothing to roll back."
fi
if [ "$START_BASELINE" = "1" ]; then
  if docker container inspect "$BASELINE_CONTAINER" >/dev/null 2>&1; then
    echo "Baseline container $BASELINE_CONTAINER already exists; start it with: docker start $BASELINE_CONTAINER"
  else
    echo "Starting the plain UVA baseline (no hot/cold split) ..."
    bash "$HERE/run-tp4-uva.sh"
  fi
else
  echo "Next step (optional): bash $HERE/run-tp4-uva.sh   # stock UVA baseline"
fi
echo "Rollback done. Container removed. NOT touched (preserved on purpose): the Hugging Face model cache and $LIVE_DIR counters -- see manage.sh clean-cache."
