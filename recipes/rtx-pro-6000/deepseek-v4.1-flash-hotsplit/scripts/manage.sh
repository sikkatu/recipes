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
#   clean-cache show the HF cache size, then delete it only after explicit confirmation
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

# Seed settings from ./tp4.env. Values already in the environment WIN over the file:
# the file is applied only for names that are not already set (parameter expansion).
if [ -f "$ROOT/tp4.env" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|'#'*) continue ;;                 # skip blanks and comments
    esac
    name="${line%%=*}"
    [ "$name" = "$line" ] && continue       # not an assignment
    name="$(printf '%s' "$name" | tr -d ' ')"
    [ -z "$name" ] && continue
    if [ -z "${!name:-}" ]; then            # only fill values not already exported
      val="${line#*=}"
      # strip inline comment: everything after " # " or trailing " #..."
      val="${val%% #*}"
      val="${val%"${val##*[![:space:]]}"}"   # strip trailing whitespace
      export "$name=$val"
    fi
  done < "$ROOT/tp4.env"
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
    # send the key when we have one; without a key the request is only valid for
    # ALLOW_NO_AUTH loopback servers (a protected API answers 401/403 without it)
    CURL_AUTH=()
    if [ -n "${API_KEY:-}" ]; then
      CURL_AUTH=(-H "Authorization: Bearer $API_KEY")
    fi
    code="$(curl -s -m 5 -o /dev/null -w '%{http_code}' "${CURL_AUTH[@]}" "$BASE/v1/models" 2>/dev/null || echo 000)"
    case "$code" in
      200) echo "  reachable (authenticated)" ;;
      401|403)
        if [ -n "${API_KEY:-}" ]; then
          echo "  reachable but REJECTED our key (HTTP $code) -- API_KEY mismatch"
        else
          echo "  reachable but REQUIRES a key (HTTP $code) -- set API_KEY in this shell or tp4.env"
        fi ;;
      000) echo "  not reachable (server down, still starting, or wrong BASE)" ;;
      *)   echo "  unexpected HTTP $code from $BASE/v1/models" ;;
    esac
    ;;
  logs)
    need_docker
    docker logs -f --tail 200 "$CONTAINER"
    ;;
  start)
    need_docker
    if [ -f "$HERE/run-tp4-hotsplit.sh" ]; then
      exec bash "$HERE/run-tp4-hotsplit.sh"
    else
      echo "NOTE: run-tp4-hotsplit.sh is not shipped in this recipe (degrade gracefully)."
      echo "      Starting the UVA baseline instead: run-tp4-uva.sh"
      exec bash "$HERE/run-tp4-uva.sh"
    fi
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
  clean-cache)
    # shows the cache size first, then asks before deleting -- never deletes silently
    CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
    # REFUSE anything that is not the huggingface cache dir specifically. A mistyped HF_CACHE
    # (e.g. $HOME/.cache) must not take unrelated caches or the whole home with it.
    CACHE_REAL="$(readlink -f "$CACHE" 2>/dev/null || printf '%s' "$CACHE")"
    HOME_REAL="$(readlink -f "$HOME" 2>/dev/null || printf '%s' "$HOME")"
    # STRICT allow-list: only the canonical HF cache root and its hub/ subdir. Arbitrary
    # paths that merely contain "huggingface" in their name are NOT accepted (review P2a):
    # /tmp/huggingface/important-data must be refused.
    HF_ROOT="${HF_HOME:-$HOME_REAL/.cache/huggingface}"
    HF_ROOT_REAL="$(readlink -f "$HF_ROOT" 2>/dev/null || printf '%s' "$HF_ROOT")"
    case "$CACHE_REAL" in
      "$HF_ROOT_REAL"|"$HF_ROOT_REAL/hub")
        : ;;   # the HF cache root or its hub/ subdir -- allowed
      *)
        echo "REFUSING to delete '$CACHE_REAL': this is not a huggingface cache directory."
        echo "clean-cache only deletes the Hugging Face model cache (default \$HOME/.cache/huggingface)."
        echo "To wipe something else, do it manually -- this guard exists so a typo cannot"
        echo "remove ~/.cache, the home directory or anything unrelated."
        exit 1 ;;
    esac
    if [ -d "$CACHE" ]; then
      echo "Hugging Face cache: $CACHE"
      du -sh "$CACHE" 2>/dev/null || true
    else
      echo "No cache directory at $CACHE"
      exit 0
    fi
    printf "Delete the model cache at %s? Type 'yes' to confirm: " "$CACHE"
    read -r answer
    if [ "$answer" = "yes" ]; then
      rm -rf "$CACHE"
      echo "Cache removed. Next start will re-download model files."
    else
      echo "Aborted; cache untouched."
    fi
    ;;
  check)
    need_docker
    mode=static
    [ "${HOTSPLIT_DYNAMIC:-1}" = "1" ] && mode=dynamic
    if [ -f "$HERE/check-hotsplit-tp4.sh" ]; then
      exec bash "$HERE/check-hotsplit-tp4.sh" "$CONTAINER" "$mode"
    else
      echo "NOTE: check-hotsplit-tp4.sh is not shipped in this recipe (degrade gracefully)."
      echo "      Checking API health instead:"
      exec bash "$HERE/check-tp4-api.sh" 2>/dev/null || {
        echo "      check-tp4-api.sh also missing; basic curl check:"
        curl -s -m 5 -H "Authorization: Bearer ${API_KEY:-}" "${BASE:-http://127.0.0.1:30007}/v1/models" || true
      }
    fi
    ;;
  *)
    sed -n '4,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac