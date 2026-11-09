#!/usr/bin/env bash
# Cross-repo compatibility pin (round 63/64) -- must match kv-persistence.
KVP_COMPAT_VERSION=1
# hotsplit_env.sh -- ONE place that builds the HotSplit container env + docker args.
#
# Sourced by BOTH launchers (review round 59): the recipe's run-tp4-hotsplit.sh
# and kv-persistence's run-vllm-with-persist.sh. Until now each launcher listed
# the HOTSPLIT_* variables by hand and they drifted (the KV launcher was missing
# HOTSPLIT_NUMA_MAP / LIVE_* / DYN_* / MAX_TOKENS, --cap-add IPC_LOCK/SYS_NICE
# and --compilation-config FULL_DECODE_ONLY -- the hardware-verified recipe
# config). Keep the list HERE only:
#   * new variables added to tp4.env.example reach BOTH launchers automatically;
#   * the cross-repo contract test asserts both launchers source this module.
#
# Caller contract: source this file AFTER the tp4.env seeding (so ${VAR:-}
# picks up file values) and BEFORE building the docker command. The caller
# decides profile defaults first (HOTSPLIT_HOT_GIB etc.); this module only
# turns the CURRENT values into docker arguments.
#
# Sets three arrays/strings:
#   HOTSPLIT_ENVS           -e arguments for the container env
#   HOTSPLIT_DOCKER_CAPS    --cap-add IPC_LOCK --cap-add SYS_NICE (NUMA binding)
#   HOTSPLIT_COMPILE_ARG    --compilation-config FULL_DECODE_ONLY (verified
#                             CUDA-graph profile of the hardware-tested recipe)
#
# HONESTY NOTE: the FULL_DECODE_ONLY profile is what the recipe measured on the
# 4x RTX PRO 6000 host. Keeping it here means KV-persistence runs get the SAME
# engine compilation config -- a deviation would be an untested configuration.

# docker capabilities: SYS_NICE for numa_bind.py scheduling, IPC_LOCK for
# locked memory (ulimit memlock is set separately by the launcher).
HOTSPLIT_DOCKER_CAPS=(--cap-add IPC_LOCK --cap-add SYS_NICE)
# CUDA graph profile -- exactly the recipe's verified flag.
HOTSPLIT_COMPILE_ARG='--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}'

# The env list. Fixed variables first (with the recipe's own defaults), then
# the optional knobs pass through only when set -- identical semantics to the
# recipe's ENVS block.
HOTSPLIT_ENVS=(
  -e PYTHONPATH=/w
  -e "HOTSPLIT_DYNAMIC=${HOTSPLIT_DYNAMIC:-1}"
  -e "HOTSPLIT_DRYRUN=${HOTSPLIT_DRYRUN:-0}"
  -e "HOTSPLIT_COUNTS=${HOTSPLIT_COUNTS:-/w/expert_hist_mix.json}"
  -e "HOTSPLIT_WEIGHTS=${HOTSPLIT_WEIGHTS:-decode=1,prefill=0}"
  -e "HOTSPLIT_ALLOW_STATIC_FALLBACK=${HOTSPLIT_ALLOW_STATIC_FALLBACK:-0}"
  -e "HOTSPLIT_HOST_FLOOR_GIB=${HOTSPLIT_HOST_FLOOR_GIB:-128}"
  -e "HOTSPLIT_NUMA=${HOTSPLIT_NUMA:-1}"
  -e "HOTSPLIT_NUMA_MAP=${HOTSPLIT_NUMA_MAP:-}"
  -e "HOTSPLIT_NUMA_POLICY=${HOTSPLIT_NUMA_POLICY:-preferred}"
  -e "HOTSPLIT_BOOT_HOOK=${HOTSPLIT_BOOT_HOOK:-1}"
  -e "HOTSPLIT_LIVE_COUNTS=${HOTSPLIT_LIVE_COUNTS:-/live/counts.json}"
  -e "HOTSPLIT_LIVE_SECS=${HOTSPLIT_LIVE_SECS:-600}"
)
# HOT_GIB: REQUIRED for a real split (empty plan = empty hot set). The recipe
# leaves it to tp4.env (empty = stock placement); the KV launcher sets 30
# before sourcing. Pass through whenever non-empty.
[ -n "${HOTSPLIT_HOT_GIB:-}" ] && HOTSPLIT_ENVS+=(-e "HOTSPLIT_HOT_GIB=$HOTSPLIT_HOT_GIB")

# Optional knobs -- pass through ONLY when set (same list as the recipe).
for _v in HOTSPLIT_DYN_STEPS HOTSPLIT_DYN_MAX_SWAPS HOTSPLIT_DYN_DECAY HOTSPLIT_DYN_MARGIN \
          HOTSPLIT_DYN_PRIOR_TOKENS HOTSPLIT_DYN_LOG HOTSPLIT_TOPK HOTSPLIT_LIVE_DECODE_MAX \
          HOTSPLIT_DYN_MAX_PLAN_ERRORS HOTSPLIT_STRICT_HOOK HOTSPLIT_VRAM_RESERVE_GIB \
          HOTSPLIT_MAX_TOKENS HOTSPLIT_NUMA_STRICT HOTSPLIT_DYN_MAX_COMMIT_ERRORS HOTSPLIT_COUNTS_MODE; do
  [ -n "${!_v:-}" ] && HOTSPLIT_ENVS+=(-e "$_v=${!_v}")
done
unset _v


# ---------------------------------------------------------------------------
# _hotsplit_validate_config [TP_SIZE] -- validate the HotSplit knobs on the
# HOST, before the 13-16 minute model load (review rounds 59-62, 76). ONE
# copy: both launchers call this, so switching the engine path cannot lose
# the checks. Returns 0 when the config is valid, 1 otherwise (errors are
# printed to stderr; nothing is started by the caller in that case).
#
# Round 76 (review): the block used to live only in the kv launcher -- after
# KVP_ENGINE=recipe became the planned default the recipe path would lose
# every check. It now lives HERE and additionally verifies that
# HOTSPLIT_NUMA_MAP covers every GPU rank 0..TP-1 (a map "0:0,1:0" on TP4
# silently leaves GPU 2 and 3 unbound while numa_bind reports ACTIVE).
_hotsplit_validate_config(){
  _tp="${1:-${TP_SIZE:-4}}"
  _hs_ok=1
  if [ -n "${HOTSPLIT_HOT_GIB:-}" ]; then
    case "$HOTSPLIT_HOT_GIB" in
      *[!0-9]*|'') echo "ERROR: HOTSPLIT_HOT_GIB must be a whole number of GiB (e.g. 30), got '$HOTSPLIT_HOT_GIB'" >&2; _hs_ok=0 ;;
    esac
  fi
  for _kv in HOTSPLIT_HOST_FLOOR_GIB HOTSPLIT_VRAM_RESERVE_GIB; do
    _val="${!_kv:-}"
    if [ -n "$_val" ] && ! printf '%s' "$_val" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
      echo "ERROR: $_kv must be a non-negative number of GiB (e.g. 128), got '$_val'" >&2
      _hs_ok=0
    fi
  done
  for _kv in HOTSPLIT_DYN_STEPS HOTSPLIT_DYN_MAX_SWAPS HOTSPLIT_TOPK \
             HOTSPLIT_LIVE_DECODE_MAX HOTSPLIT_DYN_MAX_PLAN_ERRORS \
             HOTSPLIT_DYN_MAX_COMMIT_ERRORS HOTSPLIT_LIVE_SECS \
             HOTSPLIT_MAX_TOKENS HOTSPLIT_DYN_PRIOR_TOKENS; do
    _val="${!_kv:-}"
    if [ -n "$_val" ]; then
      case "$_val" in
        *[!0-9]*|'') echo "ERROR: $_kv must be a non-negative integer, got '$_val'" >&2; _hs_ok=0 ;;
        0)
          case "$_kv" in
            HOTSPLIT_MAX_TOKENS|HOTSPLIT_DYN_PRIOR_TOKENS) : ;;
            *) echo "ERROR: $_kv must be >= 1, got 0" >&2; _hs_ok=0 ;;
          esac ;;
      esac
    fi
  done
  if [ -n "${HOTSPLIT_DYN_MAX_SWAPS:-}" ] && [ "${HOTSPLIT_DYN_MAX_SWAPS}" -gt 1024 ] 2>/dev/null; then
    echo "ERROR: HOTSPLIT_DYN_MAX_SWAPS must be 1..1024, got '$HOTSPLIT_DYN_MAX_SWAPS'" >&2
    _hs_ok=0
  fi
  _val="${HOTSPLIT_DYN_DECAY:-}"
  if [ -n "$_val" ]; then
    if ! printf '%s' "$_val" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
      echo "ERROR: HOTSPLIT_DYN_DECAY must be a number in 0..1 (e.g. 0.5), got '$_val'" >&2; _hs_ok=0
    else
      _ip="${_val%%.*}"
      _fp=""; case "$_val" in *.*) _fp="${_val#*.}" ;; esac
      _ipn=$((10#${_ip:-0}))
      if [ "$_ipn" -gt 1 ] || { [ "$_ipn" -eq 1 ] && [ -n "$_fp" ] && [ -n "${_fp//0/}" ]; }; then
        echo "ERROR: HOTSPLIT_DYN_DECAY must be in 0..1 (hotsplit.py range), got '$_val'" >&2; _hs_ok=0
      fi
    fi
  fi
  _val="${HOTSPLIT_DYN_MARGIN:-}"
  if [ -n "$_val" ]; then
    if ! printf '%s' "$_val" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
      echo "ERROR: HOTSPLIT_DYN_MARGIN must be a number >= 1 (e.g. 1.25), got '$_val'" >&2; _hs_ok=0
    else
      _ip="${_val%%.*}"
      _ipn=$((10#${_ip:-0}))
      if [ "$_ipn" -lt 1 ]; then
        echo "ERROR: HOTSPLIT_DYN_MARGIN must be >= 1 (hotsplit.py range), got '$_val'" >&2; _hs_ok=0
      fi
    fi
  fi
  if [ "${HOTSPLIT_NUMA:-1}" = "1" ] && [ -n "${HOTSPLIT_NUMA_MAP:-}" ]; then
    if ! printf '%s' "$HOTSPLIT_NUMA_MAP" | grep -Eq '^[0-9]+:[0-9]+(,[0-9]+:[0-9]+)*$'; then
      echo "ERROR: HOTSPLIT_NUMA_MAP must look like '0:0,1:0,2:1,3:1' (gpu:node pairs, comma-separated), got '$HOTSPLIT_NUMA_MAP'" >&2
      echo "       (see tp4.env.example -- HOTSPLIT_NUMA_MAP; confirm the map with: numactl --hardware)" >&2
      _hs_ok=0
    else
      # Round 76 (review): the map must COVER every GPU rank 0..TP-1.
      # numa_bind.py map_from_env silently returns -1 for a missing GPU and
      # the numa hook still reports ACTIVE -- a partial map looks healthy.
      _covered=""
      _pairs="${HOTSPLIT_NUMA_MAP//,/$'\n'}"
      while IFS=: read -r _g _n; do
        [ -n "$_g" ] && _covered="$_covered $_g "
      done <<<"$_pairs"
      for _i in $(seq 0 $((_tp - 1))); do
        case "$_covered" in
          *" $_i "*) : ;;
          *)
            echo "ERROR: HOTSPLIT_NUMA_MAP='$HOTSPLIT_NUMA_MAP' has no entry for GPU $_i (TP_SIZE=$_tp)." >&2
            echo "       numa_bind would silently leave that rank unbound while reporting ACTIVE." >&2
            _hs_ok=0 ;;
        esac
      done
    fi
  fi
  if [ "$_hs_ok" = "1" ]; then
    echo "hotsplit config OK: HOT_GIB=${HOTSPLIT_HOT_GIB:-stock} DYN_STEPS=${HOTSPLIT_DYN_STEPS:-default} NUMA=${HOTSPLIT_NUMA:-1}${HOTSPLIT_NUMA_MAP:+ map=$HOTSPLIT_NUMA_MAP}"
    return 0
  fi
  echo "ERROR: HotSplit configuration is invalid -- fix the values above and rerun (nothing was started)." >&2
  return 1
}
