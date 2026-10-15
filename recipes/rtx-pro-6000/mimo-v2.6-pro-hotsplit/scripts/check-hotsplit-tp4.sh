#!/usr/bin/env bash
# Reads the container logs and reports which hotsplit stages actually happened. Diagnostic only.
# usage: check-hotsplit-tp4.sh [container]   (default mimo-pro-tp4-hotsplit)
set -u
C="${1:-mimo-pro-tp4-hotsplit}"
L="$(docker logs "$C" 2>&1)" || { echo "cannot read logs of $C" >&2; exit 2; }
fail=0
chk(){ # label, regex, expected count (default >=1)
  local n; n="$(grep -Ec "$2" <<<"$L")"
  if [ "$n" -ge "${3:-1}" ]; then echo "PASS  $1 ($n)"; else echo "MISS  $1 ($n, expected >= ${3:-1})"; fail=1; fi
}
chk "Marlin MXFP4 backend selected"           "MARLIN"
chk "UVA offload active"                      "Total CPU offloaded parameters"
chk "hotsplit plan logged on every rank"      "hotsplit plan \[rank [0-9]/4\]" 4
chk "hotsplit finished on every rank"         "hotsplit done \[rank [0-9]/4\]" 4
chk "NUMA binding logged for every rank"      "HOTSPLIT_NUMA cuda:[0-9] .* -> node" 4
chk "dynamic step hook installed on every rank" "hotsplit dynamic: step hook installed" 4
chk "live counter registered"                 "hotsplit live counter"
echo
echo "---- warnings worth reading ----"
grep -E "hotsplit.*(warning|WARNING|host floor|not importable|failed|skipped)|HOTSPLIT_NUMA.*(NOT set|not binding|skipped|failed)|expert parallelism is active" <<<"$L" | tail -20
echo
echo "---- last dynamic updates ----"
grep "hotsplit dynamic update" <<<"$L" | tail -5
echo
echo "---- GPU memory ----"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo
echo "Locality (should show the pinned expert memory on the node of each GPU, not split across nodes):"
echo "  for p in \$(pgrep -f 'VLLM|vllm' | head -8); do echo PID \$p; numastat -p \$p | tail -3; done"
exit "$fail"
