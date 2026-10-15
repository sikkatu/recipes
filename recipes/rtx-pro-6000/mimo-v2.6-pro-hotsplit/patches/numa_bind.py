"""Best-effort NUMA pinning of each vLLM TP worker to the socket its GPU hangs off (Linux only).

Why: UVA expert offload reads pinned host memory straight from the GPU. On a dual-socket host the pinned
buffers land on whatever node the allocating thread runs on; a GPU on socket 1 reading memory on socket 0
crosses the inter-socket link on every cold-expert read. This module restricts the worker thread's CPU
affinity to the GPU's NUMA node and sets a memory policy for it, before the weights are loaded.

Activated by sitecustomize.py when HOTSPLIT_NUMA=1. HOTSPLIT_NUMA_POLICY = preferred (default; falls back
to the other node instead of failing when the node is full) or bind (strict). Needs --cap-add SYS_NICE in
Docker for set_mempolicy. Verify afterwards with `numastat -p <worker pid>`; do not assume it worked.
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys

MPOL_PREFERRED = 1
MPOL_BIND = 2
_SYSCALL_SET_MEMPOLICY = {"x86_64": 238, "aarch64": 237}
_BOUND: set[int] = set()


def _log(msg: str) -> None:
    sys.stderr.write(f"HOTSPLIT_NUMA {msg}\n")


def parse_cpulist(text: str) -> list[int]:
    """'0-3,8,10-11' -> [0, 1, 2, 3, 8, 10, 11]"""
    cpus: list[int] = []
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return cpus


def pci_addr(domain: int, bus: int, device: int) -> str:
    return f"{domain:04x}:{bus:02x}:{device:02x}.0"


def gpu_numa_node(addr: str, sysfs: str = "/sys") -> int:
    """NUMA node of a PCI device, -1 if unknown."""
    try:
        with open(f"{sysfs}/bus/pci/devices/{addr}/numa_node") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return -1


def node_cpus(node: int, sysfs: str = "/sys") -> list[int]:
    try:
        with open(f"{sysfs}/devices/system/node/node{node}/cpulist") as f:
            return parse_cpulist(f.read())
    except (OSError, ValueError):
        return []


def set_mempolicy(node: int, mode: int = MPOL_PREFERRED) -> bool:
    nr = _SYSCALL_SET_MEMPOLICY.get(platform.machine())
    if nr is None:
        return False
    words = node // 64 + 1
    mask = (ctypes.c_ulong * words)()
    mask[node // 64] = 1 << (node % 64)
    libc = ctypes.CDLL(None, use_errno=True)
    # libnuma passes size + 1: the kernel decrements maxnode before reading the mask.
    rc = libc.syscall(nr, mode, ctypes.byref(mask), words * 64 + 1)
    if rc != 0:
        _log(f"set_mempolicy failed errno={ctypes.get_errno()} (Docker needs --cap-add SYS_NICE)")
    return rc == 0


def bind_for_device(index: int):
    """Returns (ok, detail). ok=True means a NUMA policy was actually applied for this device."""
    if index in _BOUND:
        return True, "already bound"
    _BOUND.add(index)
    import torch  # already imported by the time a worker selects its device
    props = torch.cuda.get_device_properties(index)
    bus = getattr(props, "pci_bus_id", None)
    if bus is None:
        _log(f"cuda:{index}: torch does not expose pci_bus_id; not binding")
        return False, "torch does not expose pci_bus_id"
    addr = pci_addr(getattr(props, "pci_domain_id", 0), bus, getattr(props, "pci_device_id", 0))
    node = gpu_numa_node(addr)
    if node < 0:
        _log(f"cuda:{index} {addr}: numa_node unknown (single-node host?); not binding")
        return False, "numa_node unknown"
    cpus = node_cpus(node)
    if cpus:
        os.sched_setaffinity(0, set(cpus))
    policy = os.environ.get("HOTSPLIT_NUMA_POLICY", "preferred")
    mode = MPOL_BIND if policy == "bind" else MPOL_PREFERRED
    ok = set_mempolicy(node, mode)
    if policy == "preferred" and ok:
        # honest caveat: preferred is advisory -- the kernel may still place pages on another
        # node under pressure. Only 'bind' (HOTSPLIT_NUMA_POLICY=bind) hard-guarantees locality.
        _log(f"cuda:{index} {addr} -> node {node}, {len(cpus)} cpus, mempolicy=preferred (advisory: "
             "pages may still land on another node under pressure; use HOTSPLIT_NUMA_POLICY=bind "
             "for a hard guarantee)")
    else:
        _log(f"cuda:{index} {addr} -> node {node}, {len(cpus)} cpus, mempolicy {'set' if ok else 'NOT set'} "
             f"({'bind' if mode == MPOL_BIND else 'preferred'})")
    if not ok and os.environ.get("HOTSPLIT_NUMA_STRICT", "0") == "1":
        raise RuntimeError(f"numa_bind: strict mode and set_mempolicy failed on cuda:{index} -> node {node}")
    return ok, f"node {node} {'bind' if mode == MPOL_BIND else 'preferred'} {'set' if ok else 'NOT set'}"


def _device_index(device) -> int | None:
    import torch

    if isinstance(device, int):
        return device
    try:
        d = torch.device(device)
    except Exception:
        return None
    if d.type != "cuda":
        return None
    return d.index if d.index is not None else torch.cuda.current_device()


def install() -> None:
    """Wrap torch.cuda.set_device so the binding happens when a worker picks its GPU.

    Raises RuntimeError under HOTSPLIT_NUMA_STRICT=1 if binding fails.
    Note: this is best-effort localisation. `preferred` (the default) is advisory; only
    `HOTSPLIT_NUMA_POLICY=bind` hard-guarantees pinned memory lands on the GPU's node.
    Verify real placement with `scripts/numastat-report.sh` -- logging is not proof of locality.
    """
    import torch
    if getattr(torch.cuda, "_hotsplit_numa_wrapped", False):
        return
    orig = torch.cuda.set_device
    def set_device(device):
        orig(device)
        try:
            idx = _device_index(device)
            if idx is not None:
                ok, detail = bind_for_device(idx)
                if not ok and os.environ.get("HOTSPLIT_NUMA_STRICT", "0") == "1":
                    raise RuntimeError(f"numa_bind strict: cuda:{idx} binding failed: {detail}")
        except Exception as e:
            if os.environ.get("HOTSPLIT_NUMA_STRICT", "0") == "1":
                raise
            _log(f"binding skipped: {e!r}")
    torch.cuda.set_device = set_device
    torch.cuda._hotsplit_numa_wrapped = True
