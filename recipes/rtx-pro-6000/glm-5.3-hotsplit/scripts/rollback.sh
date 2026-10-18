#!/usr/bin/env bash
# Rollback: stop and remove the hotsplit container, then optionally start the plain UVA baseline.
# HONEST SCOPE: the launchers write OUTSIDE the container too, and rollback does NOT undo it:
#   - the Hugging Face cache (HF_CACHE, default ~/.cache/huggingface) keeps model weights;
#   - LIVE_DIR (default ./live) keeps routing counters and snapshots.
# Rollback removes the server container only. Model cache and counters are preserved on purpose;
# use manage.sh clean-cache to drop the cache after a size preview and explicit confirmation.
# usage: rollback.sh [--start-baseline]
set -Eeuo pipefail
die(){ echo "ERROR: $*" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOTSPLIT_CONTAINER="${HOTSPLIT_CONTAINER:-glm53-tp4-hotsplit}"
BASELINE_CONTAINER="${BASELINE_CONTAINER:-glm53-tp4-uva}"
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
