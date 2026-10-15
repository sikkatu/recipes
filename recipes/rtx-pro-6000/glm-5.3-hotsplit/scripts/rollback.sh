#!/usr/bin/env bash
# Rollback: stop and remove the hotsplit container, then optionally start the plain UVA baseline.
# Nothing outside the container is ever modified by the launchers, so rollback is container-level.
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
echo "Rollback done. No host state was modified by hotsplit."
