#!/usr/bin/env bash
# Reads the container logs and reports which hotsplit stages actually happened.
# usage: check-hotsplit-tp4.sh [container]   (default mimo-pro-tp4-hotsplit)
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
chk "hotsplit plan logged on every rank"      "hotsplit plan \[rank [0-9]/4\]" 4
chk "hotsplit finished on every rank"         "hotsplit done \[rank [0-9]/4\]" 4
echo
echo "== dynamic mode =="
# The dynamic hook status is per rank: ACTIVE or INACTIVE (both deliberate) or absent (= not reached)
dyn_active="$(grep -Ec 'dynamic_hook=ACTIVE' <<<"$L")"
dyn_inactive="$(grep -Ec 'dynamic_hook=INACTIVE' <<<"$L")"
if [ "$dyn_active" -ge 1 ]; then
  echo "PASS  dynamic_hook=ACTIVE ($dyn_active ranks)"
else
  echo "MISS  dynamic_hook=ACTIVE (0; inactive=$dyn_inactive)"
fi
if grep -q 'dynamic_hook=FAILED' <<<"$L"; then
  echo "FAIL  dynamic_hook=FAILED on at least one rank"; fail=1
fi
chk "dynamic step hook installed (if requested)" "hotsplit dynamic: step hook installed" 0
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
echo "      run a few requests and re-run this script: $n_updates should grow and"
echo "      live/counts.rank*.json timestamps should advance."
exit "$fail"