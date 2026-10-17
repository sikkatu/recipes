#!/usr/bin/env bash
# Day-to-day management for the MiMo-V2.6-Pro TP4 hotsplit recipe.
# Thin wrapper over the existing launchers -- no second source of truth for env or flags.
#
# usage: manage.sh <command>
#   status      show container state, GPU state and API reachability
#   logs        follow the hotsplit container log (Ctrl-C to stop following)
#   start       start the hotsplit server (HOTSPLIT_DYNAMIC from env, default 1)
#   start-uva   start the stock UVA baseline instead
#   stop        stop and remove the running server container
#   restart     stop + start (picks up current env and tp4.env if present)
#   update      git pull, then restart with the updated tree
#   rollback    delegate to scripts/rollback.sh (stop hotsplit, optional baseline)
#   check       run the hotsplit verification (static/dynamic from HOTSPLIT_DYNAMIC)
#
# Settings come from the environment, optionally seeded from ./tp4.env if it exists
# (copy tp4.env.example and edit). Values already in the environment win.
set -Eeuo pipefail
die(){ echo "ERROR: $*" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

CONTAINER="${CONTAINER:-deepseek-flash-tp4-hotsplit}"
BASELINE_CONTAINER="${BASELINE_CONTAINER:-deepseek-flash-tp4-uva}"
BASE="${BASE:-http://127.0.0.1:30007}"

# Seed settings from ./tp4.env (values already exported win over the file).
if [ -f "$ROOT/tp4.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/tp4.env"
  set +a
fi

cmd="${1:-}"

# usage must work without docker; the runtime commands need it
need_docker(){
  command -v docker >/dev/null || die "docker is not installed (see the wiki Quick Start)."
  docker info >/dev/null 2>&1 || die "docker daemon unavailable or permission denied."
}
case "$cmd" in
  status)
    need_docker
    echo "== containers =="
    for c in "$CONTAINER" "$BASELINE_CONTAINER"; do
      state="$(docker inspect -f '{{.State.Status}} (started {{.State.StartedAt}})' "$c" 2>/dev/null || echo 'not created')"
      printf '  %-28s %s\n' "$c" "$state"
    done
    echo
    echo "== GPUs =="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null \
      || echo "  nvidia-smi unavailable"
    echo
    echo "== API ($BASE) =="
    if curl -fsS -m 5 -o /dev/null "$BASE/v1/models" 2>/dev/null; then
      echo "  reachable"
    else
      echo "  not reachable (server down, still starting, or API_KEY/BASE mismatch)"
    fi
    ;;
  logs)
    need_docker
    docker logs -f --tail 200 "$CONTAINER"
    ;;
  start)
    need_docker
    exec bash "$HERE/run-tp4-hotsplit.sh"
    ;;
  start-uva)
    need_docker
    exec bash "$HERE/run-tp4-uva.sh"
    ;;
  stop)
    need_docker
    for c in "$CONTAINER" "$BASELINE_CONTAINER"; do
      if docker container inspect "$c" >/dev/null 2>&1; then
        echo "Stopping and removing $c ..."
        docker stop "$c" >/dev/null 2>&1 || true
        docker rm "$c" >/dev/null 2>&1 || true
      fi
    done
    echo "Stopped."
    ;;
  restart)
    bash "${BASH_SOURCE[0]}" stop
    exec bash "${BASH_SOURCE[0]}" start
    ;;
  update)
    need_docker
    echo "Updating $ROOT from git ..."
    GITROOT="$(git -C "$ROOT" rev-parse --show-toplevel 2>/dev/null)" || die "not a git checkout: $ROOT"
    git -C "$GITROOT" pull --ff-only || die "git pull failed (local changes?). Resolve, then re-run."
    bash "${BASH_SOURCE[0]}" restart
    ;;
  rollback)
    need_docker
    exec bash "$HERE/rollback.sh" "${2:-}"
    ;;
  check)
    need_docker
    mode=static
    [ "${HOTSPLIT_DYNAMIC:-1}" = "1" ] && mode=dynamic
    exec bash "$HERE/check-hotsplit-tp4.sh" "$CONTAINER" "$mode"
    ;;
  *)
    sed -n '4,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac