#!/bin/bash
# collect_system_snapshot.sh — write results/<run-id>/system.json from what the machine actually reports.
# Usage: bash scripts/collect_system_snapshot.sh > system.json
# Runs on the host (not inside the container). Needs nvidia-smi; everything else is optional.
# Lists EVERY GPU (workstations often have a display GPU next to the compute GPU) and marks the
# largest-memory one as `compute_gpu` — that is the one recipes pin against.
# Since 2026-09-19 also records each GPU's power cap / draw / clocks / throttle reasons and the
# coherent-memory mode and power caps are recorded because a moved cap looks exactly like
# reference drift between two runs of the same recipe.
set -u
gpus=$(nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null)
power=$(nvidia-smi --query-gpu=index,power.limit,power.max_limit,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active --format=csv,noheader,nounits 2>/dev/null)
cuda=$(nvidia-smi 2>/dev/null | grep -o "CUDA Version: [0-9.]*" | awk '{print $3}')
os=$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")
kernel=$(uname -r)
arch=$(uname -m)
host_mem_kib=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
cpu=$(lscpu 2>/dev/null | grep "Model name" | head -1 | sed 's/.*: *//')
docker_v=$(docker --version 2>/dev/null | sed 's/Docker version //; s/,.*//')
coherent_mode=$(grep -o 'CoherentGPUMemoryMode: "[a-z]*"' /proc/driver/nvidia/params 2>/dev/null | sed 's/.*: "//; s/"//')
vsloshd=$(systemctl is-active vsloshd 2>/dev/null || true)
python3 - "$gpus" "$cuda" "$os" "$kernel" "$arch" "$host_mem_kib" "$cpu" "$docker_v" "$power" "$coherent_mode" "$vsloshd" <<'PY'
import json, sys, datetime, socket
gpus_csv, cuda, os_, k, arch, hmem, cpu, dk, power_csv, coherent, vsloshd = sys.argv[1:]
gpus = []
for line in gpus_csv.strip().splitlines():
    idx, name, drv, mem = [x.strip() for x in line.split(",")]
    gpus.append({"index": int(idx), "name": name, "driver": drv, "memory_visible_gib": round(float(mem) / 1024, 1)})
def fnum(x):
    try: return float(x)
    except Exception: return None
for line in power_csv.strip().splitlines():
    f = [x.strip() for x in line.split(",")]
    if len(f) != 8: continue
    idx = int(f[0])
    for g in gpus:
        if g["index"] == idx:
            g["power"] = {"limit_w": fnum(f[1]), "max_limit_w": fnum(f[2]), "draw_w": fnum(f[3]),
                          "sm_mhz": fnum(f[4]), "mem_mhz": fnum(f[5]), "temp_c": fnum(f[6]), "throttle_reasons": f[7]}
compute = max(gpus, key=lambda g: g["memory_visible_gib"]) if gpus else None
def num(x, div):
    try: return round(float(x) / div, 1)
    except Exception: return None
print(json.dumps({
    "collected_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "hostname": socket.gethostname(),
    "gpus": gpus,
    "compute_gpu": compute,
    "cuda": cuda,
    "coherent_gpu_memory_mode": coherent or None,
    "vsloshd": vsloshd or None,
    "host_memory_gib": num(hmem, 1024 * 1024),
    "cpu": cpu, "arch": arch, "os": os_, "kernel": k, "docker": dk,
}, indent=2))
PY
