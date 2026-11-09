#!/usr/bin/env bash
# Collect the ACTUAL NUMA placement of pinned host memory per TP worker.
# Logging a mempolicy line is not proof of locality (preferred is advisory) -- this script is.
#
# usage: numastat-report.sh [container]   (default mimo-pro-tp4-hotsplit)
# Run while the server is up. For each vLLM worker PID it prints the per-node page counts
# and, if numastat is available on the host, its per-node summary.
#
# Interpreting the output:
#   - with HOTSPLIT_NUMA_POLICY=bind, the pinned expert memory must be on the worker's GPU node;
#   - with HOTSPLIT_NUMA_POLICY=preferred (default), pages MAY spill to other nodes under
#     pressure -- that is a documented tradeoff, not a bug;
#   - the two-host comparison (preferred vs bind vs none) on one workload is the way to
#     measure the bandwidth effect; see ../RUNNING.md.
set -u
C="${1:-mimo-pro-tp4-hotsplit}"
command -v numactl >/dev/null || echo "NOTE: numactl not found on the host; per-node output below uses /proc/<pid>/numa_maps only"
PIDS=$(docker exec "$C" sh -c 'ls /proc | grep -E "^[0-9]+$"' 2>/dev/null | sort -n)
found=0
for pid in $PIDS; do
  cmd=$(docker exec "$C" sh -c "tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null" 2>/dev/null || true)
  case "$cmd" in
    *VLLM*|*vllm*|*python*)
      found=1
      echo "=== worker PID $pid ==="
      echo "cmdline: $cmd"
      docker exec "$C" sh -c "grep -E 'N[0-9]+=' /proc/$pid/numa_maps 2>/dev/null | awk '{for(i=2;i<=NF;i++){if(\$i ~ /^N[0-9]+=[0-9]+\$/){split(\$i,a,\"=\"); n[a[1]]+=a[2]}}} END {for(k in n) print k\"=\"n[k]}' | sort" 2>/dev/null || echo "  (numa_maps not readable from container)"
      if command -v numastat >/dev/null; then
        echo "--- host numastat -p $pid ---"
        numastat -p "$pid" 2>/dev/null | tail -n +2 || echo "  (numastat needs root or the worker to be visible from the host)"
      fi
      echo
      ;;
  esac
done
if [ "$found" = 0 ]; then
  echo "no vLLM worker PIDs found in $C (is the server running?)" >&2
  exit 2
fi
echo "Reminder: HOTSPLIT_NUMA_POLICY=preferred is advisory; use 'bind' and re-run this"
echo "script to compare. H2D bandwidth measurement on the same workload is the final word."