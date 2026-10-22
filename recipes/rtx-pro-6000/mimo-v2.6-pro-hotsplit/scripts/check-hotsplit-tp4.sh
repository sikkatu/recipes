#!/usr/bin/env bash
# Reads the container logs and reports which hotsplit stages actually happened.
# usage: check-hotsplit-tp4.sh [container] [MODE: dynamic|static] [expect-progress]
#        (defaults: mimo-pro-tp4-hotsplit, dynamic; expect-progress = traffic already ran)
#
# Diagnostic only — it does not prove that the dynamic counter INCREASED during generation.
# For that, generate text while this script watches, then re-run it: the "dynamic update"
# lines and the live-counter file must show growth. See ../RUNNING.md.
#
# Every requested hook must be ACTIVE or the run is flagged. HOOK STATUS lines are emitted
# by patches/sitecustomize.py for numa_bind and boot_hook; the hotsplit dynamic hook status
# is logged by hotsplit.py itself (dynamic_hook=ACTIVE / dynamic_hook=INACTIVE).
set -u
C="${1:-mimo-pro-tp4-hotsplit}"
# MODE: dynamic (default) or static. The check is mode-aware: in dynamic mode a missing or
# inactive hook on any rank is a FAIL (exit 1), not just a note.
MODE="${2:-dynamic}"
# EXPECT_PROGRESS=1 (or 3rd arg "expect-progress"): the caller asserts real traffic already ran,
# so a dynamic run with zero committed swaps is a FAIL, not a WARN. Without it the verdict is
# WARN when it is simply too early to tell (no traffic seen yet).
EXPECT_PROGRESS="${EXPECT_PROGRESS:-0}"
[ "${3:-}" = "expect-progress" ] && EXPECT_PROGRESS=1
L="$(docker logs "$C" 2>&1)" || { echo "cannot read logs of $C" >&2; exit 2; }
fail=0
chk(){ # label, regex, expected count (default >=1)
  local n; n="$(grep -Ec "$2" <<<"$L")"
  if [ "$n" -ge "${3:-1}" ]; then echo "PASS  $1 ($n)"; else echo "MISS  $1 ($n, expected >= ${3:-1})"; fail=1; fi
}
echo "== engine =="
chk "Marlin MXFP4 backend selected"           "MARLIN"
chk "UVA offload active"                      "Total CPU offloaded parameters"
echo
echo "== hook status (every requested hook must be ACTIVE; FAILED is a hard miss) =="
chk "HOOK STATUS numa_bind"                   "HOOK STATUS numa_bind=(ACTIVE|INACTIVE|FAILED)"
if grep -q "HOOK STATUS numa_bind=FAILED" <<<"$L"; then
  echo "FAIL  numa_bind FAILED — see log lines above"; fail=1
fi
chk "HOOK STATUS boot_hook"                   "HOOK STATUS boot_hook=(ACTIVE|INACTIVE|FAILED)"
if grep -q "HOOK STATUS boot_hook=FAILED" <<<"$L"; then
  echo "FAIL  boot_hook FAILED — see log lines above"; fail=1
fi
echo
echo "== hotsplit stages =="
chk "hotsplit plan pass logged on every rank"      "hotsplit dynamic plan pass #.*\[rank [0-9]/4\]" 4
chk "hotsplit finished on every rank"         "hotsplit done \[rank [0-9]/4\]" 4
echo
echo "== dynamic mode (MODE=$MODE) =="
# Per-rank check: every rank must report dynamic_hook=ACTIVE in dynamic mode.
# In static mode every rank must report dynamic_hook=INACTIVE (or absent if the run is static
# by design). A missing status line means the rank never reached the decision point -- a hard
# miss, not a warning. Exit code is non-zero if the required mode is not active on all ranks.
for r in 0 1 2 3; do
  st="$(grep -Eo "\[rank $r/4\].*dynamic_hook=[A-Z]+" <<<"$L" | grep -Eo 'dynamic_hook=[A-Z]+' | tail -1)"
  case "$MODE" in
    dynamic)
      if [ "$st" = "dynamic_hook=ACTIVE" ]; then
        echo "PASS  rank $r: $st"
      else
        echo "FAIL  rank $r: ${st:-NO STATUS} (dynamic mode requires ACTIVE)"; fail=1
      fi
      ;;
    static)
      if [ "$st" = "dynamic_hook=INACTIVE" ] || [ -z "$st" ]; then
        echo "PASS  rank $r: ${st:-INACTIVE (no dynamic request)}"
      else
        echo "FAIL  rank $r: $st (static mode expects INACTIVE)"; fail=1
      fi
      ;;
    *)
      echo "FAIL  unknown MODE '$MODE' (use dynamic or static)"; fail=1
      ;;
  esac
done
if grep -q 'dynamic_hook=FAILED' <<<"$L"; then
  echo "FAIL  dynamic_hook=FAILED on at least one rank"; fail=1
fi
if [ "$MODE" = "dynamic" ]; then
  # in dynamic mode the step hook must be installed on every rank
  chk "dynamic step hook installed on every rank" "hotsplit dynamic: step hook installed" 4
else
  # in static mode the hook may be absent; note it, do not fail
  chk "dynamic step hook (static mode, optional)" "hotsplit dynamic: step hook installed" 0
fi
echo
echo "== dynamic metrics (per rank from logs) =="
# The numbers that prove the mode is doing real work -- not just installed:
# hook status, planner passes, planner errors, committed swaps, last update time, hit rate.
for r in 0 1 2 3; do
  hook="$(grep -Eo "\\[rank $r/4\\].*dynamic_hook=[A-Z]+" <<<"$L" | grep -Eo 'dynamic_hook=[A-Z]+' | tail -1)"
  passes="$(grep -c "\\[rank $r/4\\].*hotsplit dynamic plan pass #" <<<"$L" || true)"
  commits="$(grep -Eo "hotsplit dynamic update #[0-9]+" <<<"$L" | wc -l)"
  perr="$(grep -Ec "planning failed|commit failed" <<<"$L" || true)"
  last="$(grep -Eo "hotsplit dynamic update #[0-9]+ at step [0-9]+" <<<"$L" | tail -1)"
  hr="$(grep -Eo "hot [0-9]+ / cold [0-9]+ of [0-9]+ routed rows \\([0-9.]+% hot" <<<"$L" | tail -1 | grep -Eo '[0-9.]+% hot')"
  printf "  rank %s: hook=%s plan_passes=%s commits=%s errors=%s last=[%s] hit=%s\n" \
    "$r" "${hook:-NONE}" "$passes" "$commits" "$perr" "${last:-none}" "${hr:-n/a}"
done
echo "  NOTE: commits is the global update count (shared across ranks)."
echo
echo "== live counter =="
chk "live counter registered"                 "hotsplit live counter"
echo
echo "== per-layer memory telemetry =="
chk "per-layer VRAM peak logged"              "hotsplit mem \[rank" 1
chk "per-layer release check logged"          "hotsplit release \[rank" 1
echo
echo "---- warnings worth reading ----"
grep -E "hotsplit.*(warning|WARNING|host floor|not importable|failed|skipped)|HOOK STATUS.*FAILED|HOTSPLIT_NUMA.*(NOT set|not binding|skipped|failed)|expert parallelism is active" <<<"$L" | tail -20
echo
echo "---- last dynamic updates (growth here = the dynamic counter really moves) ----"
grep "hotsplit dynamic update" <<<"$L" | tail -5
n_updates="$(grep -c "hotsplit dynamic update" <<<"$L")"
echo "total dynamic updates logged: $n_updates (0 is only OK if no generation has happened yet)"
echo
echo "---- GPU memory ----"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo
echo "Locality (should show the pinned expert memory on the node of each GPU, not split across nodes):"
echo "  for p in \$(pgrep -f 'VLLM|vllm' | head -8); do echo PID \$p; numastat -p \$p | tail -3; done"
echo
echo "NOTE: this script reads logs only. To prove the dynamic counter moves during generation,"
echo "      run a few requests and re-run this script: the dynamic update count above should grow"
echo "      and live/counts.rank*.json timestamps should advance. In dynamic mode (MODE=$MODE)"
echo "      a zero update count after real traffic means the counter never fired -- treat as a FAIL."
if [ "$MODE" = "dynamic" ]; then
  plans_seen="$(grep -Ec "hotsplit dynamic plan pass #" <<<"$L" || true)"
  if [ "${n_updates:-0}" -eq 0 ]; then
    if [ "${plans_seen:-0}" -ge 2 ]; then
      # planner ran several times but never committed a swap: either the workload matches the
      # offline plan exactly (unlikely after several windows) or the swap path is dead
      if [ "$EXPECT_PROGRESS" = "1" ]; then
        echo "FAIL  dynamic mode: planner ran ${plans_seen}x but 0 swaps committed, and the caller"
        echo "      asserts real traffic already ran (expect-progress). Dynamic adaptation is not working."
        fail=1
      else
        echo "WARN  dynamic mode: planner ran ${plans_seen}x but 0 swaps committed."
        echo "      If you have already sent real traffic this is a problem; before any generation"
        echo "      it is simply too early to tell. Re-run with 'expect-progress' (or EXPECT_PROGRESS=1)"
        echo "      after traffic to turn this into a FAIL."
      fi
    else
      echo "WARN  dynamic mode: only ${plans_seen:-0} planning passes seen -- too early to judge."
      echo "      Generate some text and re-run this script for a verdict."
    fi
  fi
fi
exit "$fail"