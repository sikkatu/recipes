#!/usr/bin/env bash
set -u
echo '== Host =='
uname -a || true
echo
echo '== CPU / NUMA =='
lscpu || true
if command -v numactl >/dev/null 2>&1; then numactl --hardware || true; fi
echo
echo '== Memory =='
grep -E 'MemTotal|MemAvailable|SwapTotal|HugePages' /proc/meminfo || true
echo
echo '== GPUs / PCIe =='
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,pci.bus_id,driver_version --format=csv || true
  nvidia-smi topo -m || true
else
  echo 'ERROR: nvidia-smi not found' >&2
fi
echo
echo '== Python / PyTorch / CUDA =='
if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY'
try:
    import torch
    print('torch:', torch.__version__)
    print('torch CUDA build:', torch.version.cuda)
    print('CUDA available:', torch.cuda.is_available())
    print('GPU count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {p.name}; VRAM={p.total_memory / (1024**3):.1f} GiB')
except Exception as e:
    print('PyTorch/CUDA check failed:', repr(e))
try:
    import vllm
    print('vLLM:', getattr(vllm, '__version__', 'version not exposed'))
except Exception as e:
    print('vLLM import unavailable:', repr(e))
PY
else
  echo 'python3 not found'
fi
echo
echo '== PCIe device summary =='
if command -v lspci >/dev/null 2>&1; then lspci -nn | grep -Ei 'NVIDIA|VGA|3D controller|PCI bridge' || true; fi
echo
echo '== GPU NUMA locality (PCI address -> numa_node; -1 = host reports one node) =='
for d in /sys/bus/pci/devices/*; do
  [ "$(cat "$d/vendor" 2>/dev/null)" = 0x10de ] || continue
  case "$(cat "$d/class" 2>/dev/null)" in
    0x03*) echo "$(basename "$d")  numa_node=$(cat "$d/numa_node" 2>/dev/null)";;
  esac
done
if command -v numactl >/dev/null 2>&1; then numactl --show || true; fi
echo '(hotsplit pins each TP worker to its GPU node when HOTSPLIT_NUMA=1; NUMA topology settings in BIOS (NPS, SNC, cluster mode) change the node count)'
echo
echo '== Checks =='
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  if [ "$GPU_COUNT" = 4 ]; then echo 'PASS: detected 4 NVIDIA GPUs'; else echo "WARN: expected 4 NVIDIA GPUs, detected $GPU_COUNT"; fi
fi
echo 'This script is diagnostic only. It does not install packages, patch vLLM, or start a server.'
