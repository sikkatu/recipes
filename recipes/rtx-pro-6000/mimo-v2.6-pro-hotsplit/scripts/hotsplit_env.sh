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
