"""hotsplit for tensor-parallel workstations: per-expert GPU-residency for UVA-offloaded MXFP4/Marlin MoE.

AUTHORED, NOT RUN ON TARGET HARDWARE. Derived from an earlier single-GPU hotsplit (TP=1, static). Drop-in
replacement: the patched mimo_v2.py hook does `from hotsplit import apply_hotsplit`, so mounting this file as
/w/hotsplit.py is all the model side needs.

What it does: after process_weights_after_loading every routed-expert layer
is re-partitioned into a HOT bank (compact GPU-memory tensors) and a COLD bank (pinned host memory read through UVA).
apply() runs Marlin twice (one launch per bank, expert_map = global id -> bank row, -1 = skip) and sums the
partial outputs. No kernel changes.

What is new here:
  * TP>1: every rank partitions its own slice of every expert; the hot list is identical on all ranks
    (router ids are replicated), the byte budget is per rank. Live-counter files are per rank. The host-memory
    floor accounts for all ranks loading at the same time.
  * HOTSPLIT_DYNAMIC=1: the hot set follows live routing. The cold bank keeps ALL experts (the original UVA
    tensors, no gather), the hot bank is a fixed-address slot buffer, and two persistent int32 maps
    ([layers, experts]) say where each expert lives. Every HOTSPLIT_DYN_STEPS engine steps, between steps and
    on the compute stream, the weakest hot experts of each layer are replaced by the strongest cold ones
    (rows copied host->GPU, then both maps flipped in place). Stream order makes the flip atomic with respect
    to forward passes, nothing is re-captured (CUDA graphs only read the fixed-address maps), and all ranks
    take the same decision from the same counters. No new launches per token versus static hotsplit.

Env (see tp4.env.example for the full list):
  HOTSPLIT_COUNTS, HOTSPLIT_WEIGHTS, HOTSPLIT_HOT_GIB (per rank), HOTSPLIT_DRYRUN, HOTSPLIT_HOST_FLOOR_GIB
  HOTSPLIT_LIVE_COUNTS, HOTSPLIT_LIVE_SECS, HOTSPLIT_LIVE_DECODE_MAX, HOTSPLIT_TOPK, HOTSPLIT_MAX_TOKENS
  HOTSPLIT_DYNAMIC, HOTSPLIT_DYN_STEPS, HOTSPLIT_DYN_MAX_SWAPS, HOTSPLIT_DYN_DECAY, HOTSPLIT_DYN_MARGIN,
  HOTSPLIT_DYN_PRIOR_TOKENS, HOTSPLIT_DYN_LOG
"""
from __future__ import annotations

import functools
import json
import os
import re
import sys
import threading
import time
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hotsplit_plan import (  # noqa: E402
    decay_scores,
    load_counts,
    plan_hot_cells,
    plan_swaps,
    rank_path,
    swap_index_updates,
)

try:
    from vllm.logger import init_logger

    logger = init_logger("vllm.hotsplit")
except Exception:  # pragma: no cover - lets the module import in unit tests
    import logging

    logger = logging.getLogger("vllm.hotsplit")

_LAYER_RE = re.compile(r"layers\.(\d+)\.")
_LIVE: "_LiveCounter | None" = None


def _dev() -> torch.device:
    # HOTSPLIT_DEVICE=cpu exists only so the swap mechanics can be unit-tested without a GPU.
    return torch.device(os.environ.get("HOTSPLIT_DEVICE", "cuda"))


def _sync() -> None:
    if _dev().type == "cuda":
        torch.cuda.synchronize()


def _tp_info() -> tuple[int, int]:
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        return int(get_tensor_model_parallel_rank()), int(get_tensor_model_parallel_world_size())
    except Exception:
        return 0, 1


def _is_mtp_module(name: str) -> bool:
    r"""True when a module path belongs to an MTP/speculative-draft subtree.

    A model can carry several `layers.<n>` subtrees (main decoder plus drafter blocks such as
    `model.mtp.layers.0`). The routing histogram describes the main decoder only, so drafter
    layers must never be folded into the same plan by a bare `layers\.(\d+)` match.
    """
    parts = name.split(".")
    return "mtp" in parts[:-1] or parts[0] == "mtp"


def _is_uva(p: torch.Tensor) -> bool:
    return bool(getattr(p, "_vllm_is_uva_offloaded", False))


def _gather_rows(w: torch.Tensor, ids: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """GPU-memory copy of w[ids] (w may be a UVA view; reads stream over PCIe)."""
    out = torch.empty((ids.numel(),) + tuple(w.shape[1:]), dtype=w.dtype, device=_dev())
    for s in range(0, ids.numel(), chunk):
        out[s:s + chunk].copy_(w.index_select(0, ids[s:s + chunk]))
    return out


def _gather_rows_to_host(w: torch.Tensor, ids: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """Exact-size pinned host buffer holding w[ids], returned as a UVA view the GPU can read."""
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    host = torch.empty((ids.numel(),) + tuple(w.shape[1:]), dtype=w.dtype, device="cpu")
    view = get_accelerator_view_from_cpu_tensor(host)
    del host
    for s in range(0, ids.numel(), chunk):
        view[s:s + chunk].copy_(w.index_select(0, ids[s:s + chunk]))
    _sync()
    view._vllm_is_uva_offloaded = True
    return view


def _host_avail_gib() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return -1.0


# --------------------------------------------------------------------------------------------------
# apply(): shared by the static and the dynamic split
# --------------------------------------------------------------------------------------------------
def _hotsplit_apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
    assert shared_experts is None, "hotsplit: shared experts path not supported"
    slot = getattr(self, "_hotsplit_live_slot", None)
    if slot is not None and _LIVE is not None and _LIVE.buf is not None \
            and topk_ids.shape[0] <= _LIVE.decode_max:  # decode-sized steps only (static shape under capture)
        _LIVE.count(slot, topk_ids)
    out = None
    for kern, pw13, pw2, m in self._hotsplit_parts:
        o = kern.apply(hidden_states=x, w1=pw13, w2=pw2, topk_weights=topk_weights, topk_ids=topk_ids,
                       activation=layer.activation, global_num_experts=layer.global_num_experts,
                       apply_router_weight_on_input=layer.apply_router_weight_on_input, expert_map=m,
                       shared_experts=None, shared_experts_input=None)
        out = o if out is None else out.add_(o)
    return out



# --------------------------------------------------------------------------------------------------
# VRAM accounting helpers (pre-check and per-layer telemetry)
# --------------------------------------------------------------------------------------------------
def _row_bytes(tens, rows: int) -> int:
    """Bytes of `rows` leading-dim slices of `tens`, or 0 if absent/empty.

    Scales are small but not free, and their layout differs per quant, so the real tensor is
    measured (duck-typed: anything with numel/shape/element_size works, which keeps this
    testable without a GPU).
    """
    if tens is None:
        return 0
    numel = getattr(tens, "numel", None)
    shape = getattr(tens, "shape", None)
    esize = getattr(tens, "element_size", None)
    if not (callable(numel) and callable(esize) and shape is not None and len(shape) > 0):
        return 0
    n = int(numel())
    if n == 0 or int(shape[0]) == 0:
        return 0
    per_row = n // int(shape[0]) * int(esize())
    return int(rows * per_row)


def _vram_precheck(hot_bytes: int, free_now: int, reserve_gib: float, floor_gib: float = 2.0):
    """Decide whether the hot-bank copy of ONE layer fits in free VRAM.

    The hot copy is made BEFORE the originals are freed, so the transient peak is
    (hot weights + hot scales) on top of whatever is already resident. `reserve_gib` must also
    cover what comes AFTER the split phase: CUDA-graph capture buffers, the MoE kernel workspace
    (Marlin), and the KV cache vLLM allocates after model load. This function only guards the
    split-phase transient; it is not a promise that the KV cache will fit later -- size
    HOTSPLIT_VRAM_RESERVE_GIB accordingly (see tp4.env.example).

    Returns (ok, reserve_bytes). A refusal must happen BEFORE any weight of this layer changes.
    """
    import math
    if hot_bytes < 0:
        raise ValueError(f"_vram_precheck: hot_bytes must be >= 0, got {hot_bytes}")
    if not math.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError(f"_vram_precheck: reserve_gib must be finite and >= 0, got {reserve_gib}")
    reserve = max(int(reserve_gib * 2**30), int(floor_gib * 2**30))
    ok = (free_now - hot_bytes) >= reserve
    return ok, reserve


def _weights_release_check(tag: str, li: int, free_before: int, expected_release: int) -> None:
    """Honest accounting of the original weight storage, measured across the WHOLE layer split.

    `del` alone proves nothing: the old kernel or its quant config can still hold a reference.
    What is measured here: free VRAM right before the split vs free VRAM after the split and
    `empty_cache()`, i.e. the net effect of (originals dropped) - (new hot/scale banks allocated).
    `expected_release` is that net difference computed from the real tensor shapes. The two are
    compared and a shortfall is reported as a WARNING naming the likely cause (lingering refs
    from the old kernel/quant config) -- this is a measurement, not a proof of release.
    """
    if _dev().type != "cuda":
        return
    try:
        torch.cuda.empty_cache()
        free_after, _ = torch.cuda.mem_get_info()
        got = free_after - free_before
        exp = int(expected_release)
        gap = exp - got
        logger.info("hotsplit release %s: layer %d free-VRAM delta %+.2f GiB (expected %+.2f GiB from "
                    "shapes); %s", tag, li, got / 2**30, exp / 2**30,
                    "as expected" if abs(gap) < 2**30 else f"short by {gap / 2**30:.2f} GiB")
        if gap >= 2**30:
            logger.warning("hotsplit release %s: layer %d did not give back what the shapes predict "
                           "(short %.2f GiB). Likely cause: the old kernel or its quant config still "
                           "references the original tensors. Check qm.moe_kernel / quant config before "
                           "trusting the freed amount.", tag, li, gap / 2**30)
    except Exception as e:
        logger.warning("hotsplit release %s: layer %d check failed: %r", tag, li, e)


def _mem_peak_start() -> dict:
    """Snapshot the allocator BASELINE and reset the peak counters before one layer is split.

    The baseline (allocated bytes now) is what the per-layer increment is measured against:
    `max_memory_allocated()` after `reset_peak_memory_stats()` is an absolute peak since the
    reset, not the cost of this layer -- subtracting the baseline turns it into the increment."""
    try:
        if _dev().type == "cuda":
            base_alloc = torch.cuda.memory_allocated()
            base_reserved = torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            return {"ok": True, "base_alloc": base_alloc, "base_reserved": base_reserved}
    except Exception:
        pass
    return {"ok": False}


def _mem_peak_report(tag: str, li: int, state: dict) -> None:
    """Log the per-layer peak so a first controlled boot can see the real transient, and verify
    the originals were actually released (not just `del`ed). `max_memory_allocated` covers only
    PyTorch's allocator -- nvidia-smi sees more (CUDA context, driver) -- so both are printed."""
    if not state.get("ok"):
        return
    try:
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        base_alloc = state.get("base_alloc", 0)
        base_reserved = state.get("base_reserved", 0)
        # the increment is what THIS layer added on top of the baseline, not the absolute peak
        incr_alloc = (peak_alloc - base_alloc) / 2**30
        incr_reserved = (peak_reserved - base_reserved) / 2**30
        free, total = torch.cuda.mem_get_info()
        logger.info("hotsplit mem %s: layer %d peak +%.2f GiB alloc / +%.2f GiB reserved above baseline "
                    "(peak alloc %.2f GiB of which baseline %.2f GiB; reserved %.2f GiB); "
                    "free %.2f/%.2f GiB, host avail %.1f GiB",
                    tag, li, incr_alloc, incr_reserved,
                    peak_alloc / 2**30, base_alloc / 2**30, peak_reserved / 2**30,
                    free / 2**30, total / 2**30, _host_avail_gib())
    except Exception as e:  # telemetry must never take the lane down
        logger.warning("hotsplit mem %s: per-layer report failed: %r", tag, e)


def _free_full_params(layer, w13, w2, free_scales: bool) -> None:
    dev = _dev()
    layer.w13_weight = torch.nn.Parameter(torch.empty(0, dtype=w13.dtype, device=dev), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.empty(0, dtype=w2.dtype, device=dev), requires_grad=False)
    if free_scales:
        # the old kernel's quant config still points at these Parameters; they are unused from here on
        for sname in ("w13_weight_scale", "w2_weight_scale"):
            sp = getattr(layer, sname, None)
            if isinstance(sp, torch.Tensor):
                sp.data = torch.empty(0, dtype=sp.dtype, device=sp.device)
    _sync()


def _patch_humming_bank(kern, n_rows: int, s13, s2) -> None:
    """Make a humming experts object accept a partial (bank) weight set.

    Two things are wrong for a bank of n_rows experts instead of the full 384:
      1. humming's LayerConfig carries num_experts=384 and the C++ launcher checks
         expected_shape[0] == weight.size(0) (observed 2026-10-23: '97 != 384').
         LayerConfig is a dataclass -> rebuild per bank with dataclasses.replace.
      2. humming_forward reads weight_scale from quant_config (property w1_scale over
         _w1.scale), which get_humming_moe_quant_config filled from the FULL layer
         scales. Point _w1.scale/_w2.scale at the bank's own sliced scales (s13/s2).
    moe_problem_size additionally derives E from the actual weight tensor.
    Marlin needs none of this (it derives E from w1.size(0) and takes scales as args).
    """
    import dataclasses as _dc
    fe = getattr(kern, "fused_experts", None)
    if fe is None or not hasattr(fe, "humming_configs"):
        return   # not a humming experts object -- Marlin etc. need no patch
    fe.num_experts = n_rows
    # 1) per-bank LayerConfig (serialized to C++; expected_shape comes from num_experts)
    for sub in ("w13", "w2"):
        cfg = fe.humming_configs[sub]
        fe.humming_configs[sub] = _dc.replace(cfg, num_experts=n_rows)
    # 2) per-bank scales: w1_scale/w2_scale are properties over _w1.scale/_w2.scale
    qc = getattr(fe, "quant_config", None)
    if qc is not None:
        for desc_name, tens in (("_w1", s13), ("_w2", s2)):
            desc = getattr(qc, desc_name, None)
            if desc is not None and hasattr(desc, "scale"):
                try:
                    desc.scale = tens
                except Exception:
                    pass
    # 3) derive E from the actual bank tensor in moe_problem_size
    def _bank_problem_size(a1, w1, w2, topk_ids):
        meta1 = fe.humming_configs["w13"]
        meta2 = fe.humming_configs["w2"]
        e = w1.size(0)                 # derive from the ACTUAL bank tensor, not the config
        assert w2.size(0) == e, f"bank w2 rows {w2.size(0)} != w1 rows {e}"
        top_k = topk_ids.size(1)
        if not fe.is_batched():
            num_tokens = a1.size(0)
            assert topk_ids.size(0) == num_tokens
        else:
            assert a1.dim() == 3 and a1.size(0) == e
            num_tokens = a1.size(1)
        intermediate_dim = meta2.shape_k - meta2.pad_shape_k
        hidden_dim = meta1.shape_k - meta1.pad_shape_k
        return e, num_tokens, intermediate_dim, hidden_dim, top_k

    fe.moe_problem_size = _bank_problem_size


def _make_bank(layer, qm, make_kernel, make_qcfg, w13, w2, s13, s2, emap):
    qcfg = make_qcfg(mxfp4_backend=qm.mxfp4_backend, w1_scale=s13, w2_scale=s2,
                     swiglu_limit=getattr(layer, "swiglu_limit", None), layer=layer)
    kern = make_kernel(moe_quant_config=qcfg, moe_config=qm.moe, experts_cls=qm.experts_cls,
                       mxfp4_backend=qm.mxfp4_backend, routing_tables=layer._expert_routing_tables())
    _patch_humming_bank(kern, w13.size(0), s13, s2)
    return kern, w13, w2, emap


# --------------------------------------------------------------------------------------------------
# static split
# --------------------------------------------------------------------------------------------------
def _split_layer_static(layer, hot_ids: list[int], E: int, make_kernel, make_qcfg, li: int) -> None:
    qm = layer.quant_method
    dev = _dev()
    _free_before = torch.cuda.mem_get_info()[0] if _dev().type == "cuda" else 0
    hot_set = set(hot_ids)
    h = torch.tensor(sorted(hot_set), dtype=torch.long, device=dev)
    c = torch.tensor([e for e in range(E) if e not in hot_set], dtype=torch.long, device=dev)

    def emap(ids: torch.Tensor) -> torch.Tensor:
        m = torch.full((E,), -1, dtype=torch.int32, device=dev)
        if ids.numel():
            m[ids] = torch.arange(ids.numel(), dtype=torch.int32, device=dev)
        return m

    w13, w2 = layer.w13_weight, layer.w2_weight
    s13, s2 = layer.w13_weight_scale, layer.w2_weight_scale
    _orig = (w13.numel() * w13.element_size() + w2.numel() * w2.element_size()
             + s13.numel() * s13.element_size() + s2.numel() * s2.element_size())
    parts = []
    _newdev = 0   # device bytes kept by the new banks (hot + bank scales stay on device)
    for ids, where in ((h, "hbm"), (c, "host")):
        if ids.numel() == 0:
            continue
        if where == "hbm":
            pw13, pw2 = _gather_rows(w13, ids), _gather_rows(w2, ids)
        else:
            pw13, pw2 = _gather_rows_to_host(w13, ids), _gather_rows_to_host(w2, ids)
        ps13, ps2 = s13.index_select(0, ids).contiguous(), s2.index_select(0, ids).contiguous()
        if where == "hbm":
            _newdev += sum(x.numel() * x.element_size() for x in (pw13, pw2, ps13, ps2))
        else:
            # the cold bank is host-resident, but its scale copies stay on the device
            _newdev += sum(x.numel() * x.element_size() for x in (ps13, ps2))
        parts.append(_make_bank(layer, qm, make_kernel, make_qcfg, pw13, pw2, ps13, ps2, emap(ids)))
    _free_full_params(layer, w13, w2, free_scales=True)
    del w13, w2
    _weights_release_check(tag="static", li=li, free_before=_free_before,
                           expected_release=_orig - _newdev)
    qm._hotsplit_parts = parts
    qm._hotsplit_nhot = h.numel()
    qm.apply = types.MethodType(_hotsplit_apply, qm)
    if _LIVE is not None:
        _LIVE.register(qm, li)


# --------------------------------------------------------------------------------------------------
# dynamic split: cold bank = every expert (host), hot bank = fixed GDDR7 slots, two persistent maps
# --------------------------------------------------------------------------------------------------
def _dyn_int(name: str, default: str, lo: int, hi: int | None = None) -> int:
    """Read an integer env var with a hard range check. Invalid values raise at startup --
    a bad value must not survive into a modulo/ZeroDivisionError deep inside the step hook,
    where the wrapper would swallow it and leave a "dynamic" run that never updates."""
    raw = os.environ.get(name, default)
    try:
        v = int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"hotsplit: {name}={raw!r} is not an integer") from e
    if v < lo or (hi is not None and v > hi):
        raise ValueError(f"hotsplit: {name}={v} out of range [{lo}, {hi if hi is not None else 'inf'}]")
    return v


def _dyn_float(name: str, default: str, lo: float, hi: float | None = None) -> float:
    """Same contract as _dyn_int for float parameters."""
    raw = os.environ.get(name, default)
    try:
        v = float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"hotsplit: {name}={raw!r} is not a number") from e
    if not (v >= lo) or (hi is not None and not (v <= hi)):
        raise ValueError(f"hotsplit: {name}={v} out of range [{lo}, {hi if hi is not None else 'inf'}]")
    return v


class _null_ctx:
    """No-op context manager so the swap code has one shape for CPU and GPU."""
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


class _DynLayer:
    def __init__(self, li, hot, hw13, hw2, hs13, hs2, c13, c2, s13, s2):
        self.li, self.hot = li, list(hot)  # hot[slot] = expert id
        self.hw13, self.hw2, self.hs13, self.hs2 = hw13, hw2, hs13, hs2
        self.c13, self.c2, self.s13, self.s2 = c13, c2, s13, s2


class _Dyn:
    def __init__(self, layer_ids: list[int], E: int):
        dev = _dev()
        self.rows = {li: i for i, li in enumerate(layer_ids)}
        self.E = E
        n = len(layer_ids)
        self.emap_hot = torch.full((n, E), -1, dtype=torch.int32, device=dev)
        self.emap_cold = torch.arange(E, dtype=torch.int32, device=dev).unsqueeze(0).repeat(n, 1).contiguous()
        self.layers: dict[int, _DynLayer] = {}
        # All scheduler parameters are validated at startup. `every` in particular must be > 0:
        # `self.steps % self.every` with every=0 is a ZeroDivisionError that the step-hook
        # wrapper used to swallow, leaving a server that answers while never updating.
        self.every = _dyn_int("HOTSPLIT_DYN_STEPS", "2000", lo=1)
        self.max_swaps = _dyn_int("HOTSPLIT_DYN_MAX_SWAPS", "2", lo=1, hi=1024)
        self.decay = _dyn_float("HOTSPLIT_DYN_DECAY", "0.5", lo=0.0, hi=1.0)
        self.margin = _dyn_float("HOTSPLIT_DYN_MARGIN", "1.25", lo=1.0)
        self.prior_tokens = _dyn_float("HOTSPLIT_DYN_PRIOR_TOKENS", "2000", lo=0.0)
        self.topk = _dyn_int("HOTSPLIT_TOPK", "8", lo=1)
        self.log_updates = os.environ.get("HOTSPLIT_DYN_LOG", "1") == "1"
        # Success/error accounting for the planner. `plan_ok` counts real planning passes,
        # `updates` counts committed swaps, `plan_errors` counts swallowed failures. The hook
        # makes repeated errors fatal (see _install_step_hook), so this is auditable afterwards.
        self.plan_ok = 0
        self.plan_errors = 0
        self.max_plan_errors = _dyn_int("HOTSPLIT_DYN_MAX_PLAN_ERRORS", "3", lo=1)
        self.steps = 0
        self.updates = 0
        self.stream = None  # dedicated CUDA stream for weight swaps (P0-2)
        # commit transaction state (P1): an update either fully lands or is rolled back;
        # after commit_errors >= max_commit_errors the run aborts instead of serving
        # from a bank that may be half-swapped.
        self.update_state = "IDLE"
        self.commit_errors = 0
        self.max_commit_errors = _dyn_int("HOTSPLIT_DYN_MAX_COMMIT_ERRORS", "3", lo=1)
        self.last_update_at = None
        self.prev: np.ndarray | None = None
        self.crow: dict[int, int] = {}
        self.score: dict[int, np.ndarray] = {}
        self.prior: dict[int, np.ndarray] = {}

    def set_prior(self, shares: dict[int, np.ndarray]) -> None:
        # offline shares seed the score so a short, unrepresentative live sample cannot evict the plan
        self.prior = {li: s / max(float(s.sum()), 1e-12) * self.prior_tokens * self.topk
                      for li, s in shares.items()}

    def plan(self):
        """Called between engine steps. Returns [(layer, swaps)] when an update is due, else None."""
        live = _LIVE
        if live is None or live.buf is None:
            return None
        self.steps += 1
        if self.prev is None:
            # first real step: baseline after CUDA-graph capture/warm-up so dummy runs are not counted
            self.prev = live.buf.to("cpu").numpy().copy()
            self.crow = {li: s for s, li in enumerate(live.layer_idx)}
            return None
        if self.steps % self.every:
            return None
        snap = live.buf.to("cpu").numpy().copy()
        delta = (snap - self.prev).astype(np.float64)
        self.prev = snap
        plans = []
        for li, dl in self.layers.items():
            c = self.crow.get(li)
            if c is None:
                continue
            base = self.score.get(li)
            if base is None:
                base = self.prior.get(li, np.zeros(self.E, dtype=np.float64))
            sc = decay_scores(base, delta[c], self.decay)
            self.score[li] = sc
            sw = plan_swaps(sc, dl.hot, self.max_swaps, self.margin)
            if sw:
                plans.append((dl, sw))
        # Only a pass that got through the whole loop counts as good. Resetting this BEFORE the loop
        # made `plan_errors` stick at 1 forever, so HOTSPLIT_DYN_MAX_PLAN_ERRORS could never fire.
        self.plan_ok += 1
        self.plan_errors = 0
        logger.info("hotsplit dynamic plan pass #%d at step %d: %d layer(s) staged for swap %s",
                    self.plan_ok, self.steps, len(plans), getattr(self, "tag", ""))
        return plans

    def commit(self, plans) -> None:
        """Copy the incoming rows into the weakest slots, then flip both maps. Runs on the current stream
        between steps, so no forward pass can observe a half-applied swap. Everything that can fail
        cheaply (index tensors) is built BEFORE the first byte is overwritten."""
        if not plans:
            return
        t0 = time.time()
        dev = _dev()
        # P0-2: run the swap copies on a dedicated stream and publish them to the compute
        # stream through a CUDA event. Relying on "the hook runs before execute_model on the
        # same thread" is not a guarantee across vLLM versions: if execute_model is entered on
        # a different stream than the one that replays the decode graph, the graph could read
        # the hot/cold banks while they are being rewritten. `stream.wait_stream(current)`
        # orders the copies after everything enqueued so far; the event + `current.wait_event`
        # orders every reader after the copies. Both are no-ops on CPU (unit tests).
        use_stream = dev.type == "cuda"
        if use_stream:
            if self.stream is None:
                self.stream = torch.cuda.Stream(device=dev)
            # Capture the COMPUTE stream now: we are still on it here, and the event at the
            # end must be published to THIS stream, not to the swap stream we enter below.
            compute_stream = torch.cuda.current_stream(device=dev)
            stream = self.stream
            swap_ctx = torch.cuda.stream(stream)
        else:
            compute_stream = None
            stream = None
            swap_ctx = None
        staged = []
        hr, hc, hv, cr, cc, cv = [], [], [], [], [], []
        for dl, sw in plans:
            slots = torch.tensor([s for s, _, _ in sw], dtype=torch.long, device=dev)
            new = torch.tensor([n for _, _, n in sw], dtype=torch.long, device=dev)
            staged.append((dl, slots, new))
            a, b, c, d, e, f = swap_index_updates(self.rows[dl.li], sw)
            hr += a; hc += b; hv += c; cr += d; cc += e; cv += f  # noqa: E702
        t_hr = torch.tensor(hr, dtype=torch.long, device=dev)
        t_hc = torch.tensor(hc, dtype=torch.long, device=dev)
        t_hv = torch.tensor(hv, dtype=torch.int32, device=dev)
        t_cr = torch.tensor(cr, dtype=torch.long, device=dev)
        t_cc = torch.tensor(cc, dtype=torch.long, device=dev)
        t_cv = torch.tensor(cv, dtype=torch.int32, device=dev)
        # ---- transactional update (P1) ----
        # Snapshot everything that will be overwritten BEFORE the first write: the affected
        # weight rows and the current map entries. If any CUDA op raises, or a post-copy sync
        # surfaces an async error, the snapshots are written back and the update is aborted.
        self.update_state = "SNAPSHOT"
        old_rows = []  # (dl, slots, hw13_rows, hw2_rows, hs13_rows, hs2_rows)
        for dl, slots, new in staged:
            old_rows.append((dl, slots,
                             dl.hw13.index_select(0, slots),   # index_select/advanced indexing already copy
                             dl.hw2.index_select(0, slots),
                             dl.hs13.index_select(0, slots),
                             dl.hs2.index_select(0, slots)))
        old_hv = self.emap_hot[t_hr, t_hc]
        old_cv = self.emap_cold[t_cr, t_cc]
        # Order the swap stream only NOW: every snapshot above is enqueued on the compute
        # stream, so stream.wait_stream(compute_stream) makes the copies wait for them.
        # (Calling it earlier would let a copy start while snapshots are still being made.)
        if use_stream:
            stream.wait_stream(compute_stream)
        try:
            self.update_state = "COPYING"
            with (swap_ctx if swap_ctx is not None else _null_ctx()):
                for dl, slots, new in staged:
                    dl.hw13.index_copy_(0, slots, dl.c13.index_select(0, new))
                    dl.hw2.index_copy_(0, slots, dl.c2.index_select(0, new))
                    dl.hs13.index_copy_(0, slots, dl.s13.index_select(0, new))
                    dl.hs2.index_copy_(0, slots, dl.s2.index_select(0, new))
                if use_stream:
                    ev = torch.cuda.Event()
                    ev.record(stream)
            # Publish outside the swap-stream context: compute_stream must wait for the swap
            # event. Inside the context current_stream() would be the swap stream itself.
            if use_stream:
                compute_stream.wait_event(ev)
            # VERIFY: force completion so an async CUDA error surfaces here, while the
            # snapshots are still intact and the maps have NOT been flipped yet.
            self.update_state = "VERIFYING"
            _sync()
            self.update_state = "MAPS"
            self.emap_hot[t_hr, t_hc] = t_hv
            self.emap_cold[t_cr, t_cc] = t_cv
            _sync()
        except Exception as e:
            # best-effort rollback: restore the snapshotted rows and map entries, then either
            # abort the run (repeat offender) or skip this update loudly.
            self.update_state = "ROLLBACK"
            try:
                with (swap_ctx if swap_ctx is not None else _null_ctx()):
                    for dl, slots, w13, w2, s13, s2 in old_rows:
                        dl.hw13.index_copy_(0, slots, w13)
                        dl.hw2.index_copy_(0, slots, w2)
                        dl.hs13.index_copy_(0, slots, s13)
                        dl.hs2.index_copy_(0, slots, s2)
                    self.emap_hot[t_hr, t_hc] = old_hv
                    self.emap_cold[t_cr, t_cc] = old_cv
                    if use_stream:
                        rev = torch.cuda.Event()
                        rev.record(stream)
                    else:
                        rev = None
                if rev is not None:
                    compute_stream.wait_event(rev)
                _sync()
                self.update_state = "IDLE"
            except Exception as e2:
                self.update_state = "FAILED"
                logger.error("hotsplit dynamic: rollback ALSO failed (%r); bank state is "
                             "uncertain -- aborting before serving corrupted weights", e2)
                raise RuntimeError(
                    f"hotsplit dynamic: commit failed ({e!r}) and rollback failed ({e2!r}); "
                    f"refusing to serve from an uncertain hot bank") from e
            self.commit_errors += 1
            logger.error("hotsplit dynamic: commit failed (%r), rolled back cleanly; "
                         "commit error %d of %d before abort",
                         e, self.commit_errors, self.max_commit_errors)
            if self.commit_errors >= self.max_commit_errors:
                raise RuntimeError(
                    f"hotsplit dynamic: {self.commit_errors} commit failures in a row; "
                    f"refusing to keep swapping on this hardware") from e
            return  # skip this update; maps and dl.hot are unchanged
        self.update_state = "COMMITTED"
        self.commit_errors = 0  # a clean commit resets the consecutive-error count
        n = 0
        for dl, sw in plans:
            for slot, _, e_in in sw:
                dl.hot[slot] = e_in
                n += 1
        self.updates += 1
        self.last_update_at = time.time()
        if self.log_updates:
            _sync()
            logger.info("hotsplit dynamic update #%d at step %d: %d swaps in %d layers, %.0f ms",
                        self.updates, self.steps, n, len(plans), 1000 * (time.time() - t0))
            self.log_hit_rates(n, time.time() - t0)

    def log_hit_rates(self, swapped_rows: int, secs: float) -> None:
        """P1 effectiveness telemetry: how the router traffic is split between the GPU hot
        bank and the host cold bank right now, plus the cost of this update. This is the number
        that must move for the dynamic mode to be worth anything: hot-bank hit ratio up,
        cold (PCIe) share down. Bytes are counted at the routed-row level (topk dispatches),
        which is what the banks actually transfer on a cold hit."""
        live = _LIVE
        if live is None or live.buf is None or not self.layers:
            return
        try:
            snap = live.buf.to("cpu", non_blocking=False).numpy()
            # Map layer -> buffer row the same way plan() does: the live buffer is indexed in
            # registration (split) order (live.layer_idx), NOT in sorted-layer order (self.rows).
            # Using self.rows here reads the wrong row whenever UVA and GPU layers are split in
            # a different order than sorted(layers) -- the hit-rate would be silently wrong.
            crow = {li: s for s, li in enumerate(live.layer_idx)}
            hot_hits = cold_hits = 0
            for li, dl in self.layers.items():
                r = crow.get(li)
                if r is None:
                    continue
                counts = snap[r]
                hot_set = set(dl.hot)
                for e, c in enumerate(counts.tolist()):
                    if not c:
                        continue
                    if e in hot_set:
                        hot_hits += c
                    else:
                        cold_hits += c
            total = hot_hits + cold_hits
            if total:
                # row sizes are model constants; report the routed-row split and the PCIe
                # traffic this split implies on the host link
                logger.info(
                    "hotsplit dynamic hit-rate: hot %d / cold %d of %d routed rows (%.1f%% hot, "
                    "%.1f%% cold -> PCIe); %d rows swapped in %.0f ms",
                    hot_hits, cold_hits, total,
                    100.0 * hot_hits / total, 100.0 * cold_hits / total,
                    swapped_rows, 1000 * secs)
        except Exception as e:  # telemetry must never take the lane down
            logger.warning("hotsplit dynamic hit-rate telemetry failed: %r", e)


def _split_layer_dynamic(layer, hot_ids: list[int], E: int, make_kernel, make_qcfg, li: int, dyn: _Dyn) -> None:
    qm = layer.quant_method
    dev = _dev()
    _free_before = torch.cuda.mem_get_info()[0] if _dev().type == "cuda" else 0
    hot_sorted = sorted(set(hot_ids))
    s = len(hot_sorted)
    w13, w2 = layer.w13_weight, layer.w2_weight
    s13, s2 = layer.w13_weight_scale, layer.w2_weight_scale
    row = dyn.rows[li]
    # cold bank: ALL experts, host-resident. Offloaded layers are reused in place (no copy at all).
    if _is_uva(w13) and _is_uva(w2):
        c13, c2 = w13, w2
    else:
        allids = torch.arange(E, dtype=torch.long, device=dev)
        c13, c2 = _gather_rows_to_host(w13, allids), _gather_rows_to_host(w2, allids)
    emap_cold = dyn.emap_cold[row]
    parts = []
    hw13 = hw2 = hs13 = hs2 = None
    if s:
        h = torch.tensor(hot_sorted, dtype=torch.long, device=dev)
        hw13, hw2 = _gather_rows(w13, h), _gather_rows(w2, h)
        hs13, hs2 = s13.index_select(0, h).contiguous(), s2.index_select(0, h).contiguous()
        emap_hot = dyn.emap_hot[row]
        emap_hot[h] = torch.arange(s, dtype=torch.int32, device=dev)
        emap_cold[h] = -1
        parts.append(_make_bank(layer, qm, make_kernel, make_qcfg, hw13, hw2, hs13, hs2, emap_hot))
    parts.append(_make_bank(layer, qm, make_kernel, make_qcfg, c13, c2, s13, s2, emap_cold))
    # net release expected on this rank: the full original weights are dropped, the hot bank is a
    # new device copy, the cold bank is host-resident (UVA or a host copy), and the FULL scales
    # stay on device because the cold bank needs every expert's scales.
    _orig_w = w13.numel() * w13.element_size() + w2.numel() * w2.element_size()
    _newdev_w = 0
    if s and hw13 is not None:
        _newdev_w += hw13.numel() * hw13.element_size() + hw2.numel() * hw2.element_size()
    if s and hs13 is not None:
        _newdev_w += hs13.numel() * hs13.element_size() + hs2.numel() * hs2.element_size()
    # keep the full scales (the cold bank needs every expert's scales); drop only the full weight params
    _free_full_params(layer, w13, w2, free_scales=False)
    del w13, w2
    _weights_release_check(tag="dynamic", li=li, free_before=_free_before,
                           expected_release=_orig_w - _newdev_w)
    dyn.layers[li] = _DynLayer(li, hot_sorted, hw13, hw2, hs13, hs2, c13, c2, s13, s2)
    qm._hotsplit_parts = parts
    qm._hotsplit_nhot = s
    qm.apply = types.MethodType(_hotsplit_apply, qm)
    if _LIVE is not None:
        _LIVE.register(qm, li)


def _dynamic_hook_decision(hook_installed: bool, dynamic_requested: bool, allow_static_fallback: bool):
    """Decide what to do when the dynamic step hook could not be installed.

    Returns (mode, error) where mode is "dynamic" or "static" and error is a message to raise
    (or None). A silent fallback to static makes a dynamic-vs-static comparison unreliable: the
    server answers but the hot set never adapts. So with HOTSPLIT_DYNAMIC=1 the run REFUSES
    unless HOTSPLIT_ALLOW_STATIC_FALLBACK=1 says otherwise, and every path logs an explicit
    dynamic_hook=ACTIVE / dynamic_hook=INACTIVE status per rank."""
    if not dynamic_requested:
        return "static", None
    if hook_installed:
        return "dynamic", None
    if allow_static_fallback:
        return "static", None
    return "static", (
        "dynamic mode was requested (HOTSPLIT_DYNAMIC=1) but the step hook could not be installed: "
        "the run would silently degrade to static hotsplit, which makes a dynamic-vs-static "
        "comparison unreliable. Set HOTSPLIT_ALLOW_STATIC_FALLBACK=1 to accept static, or fix the "
        "vLLM layout mismatch."
    )


def _install_step_hook(dyn: _Dyn) -> bool:
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:
        logger.warning("hotsplit dynamic: GPUModelRunner not importable (%r); running as static hotsplit", e)
        return False
    if getattr(GPUModelRunner, "_hotsplit_wrapped", False):
        return True
    orig = GPUModelRunner.execute_model
    @functools.wraps(orig)
    def wrapped(self, *args, **kwargs):
        plans = None
        try:
            plans = dyn.plan()
        except Exception as e:  # planning is read-only; skip this update
            dyn.plan_errors += 1
            # P0-1: a planner that keeps failing is worse than a stopped server -- the run
            # keeps claiming dynamic_hook=ACTIVE while the hot set never adapts. Repeat
            # offenders (HOTSPLIT_DYN_MAX_PLAN_ERRORS, default 3) are fatal.
            logger.error("hotsplit dynamic: planning failed (%r); error %d of %d before abort",
                         e, dyn.plan_errors, dyn.max_plan_errors)
            if dyn.plan_errors >= dyn.max_plan_errors:
                raise RuntimeError(
                    f"hotsplit dynamic: planner failed {dyn.plan_errors} times in a row "
                    f"({e!r}); refusing to keep serving with a dead dynamic hook") from e
        if plans:
            dyn.commit(plans)  # deliberately not wrapped: a failure after the copy must be loud
        return orig(self, *args, **kwargs)

    GPUModelRunner.execute_model = wrapped
    GPUModelRunner._hotsplit_wrapped = True
    logger.info("hotsplit dynamic: step hook installed (update every %d steps, <=%d swaps/layer, margin %.2f)",
                dyn.every, dyn.max_swaps, dyn.margin)
    return True


# --------------------------------------------------------------------------------------------------
# live routing counter (graph-safe, per rank)
# --------------------------------------------------------------------------------------------------
class _LiveCounter:
    """index_add_ into a preallocated int64 device tensor (no host sync, no allocation), so it is captured
    into the decode CUDA graph and replays with it. If `path` is set a daemon thread snapshots it as
    {"live": {"decode": {layer: [E]}}, "meta": {...}} every `secs` seconds (the format hotsplit_rerank.py
    reads). Counts incl. padded slots of a captured batch."""

    def __init__(self, path: str | None, secs: int):
        self.path, self.secs = path, secs
        self.decode_max = int(os.environ.get("HOTSPLIT_LIVE_DECODE_MAX", "16"))
        self.buf = None
        self.layer_idx: list[int] = []
        self.ones = None

    def register(self, qm, li: int):
        qm._hotsplit_live_slot = len(self.layer_idx)
        self.layer_idx.append(li)

    def finalize(self, E: int, max_tokens: int, topk: int):
        dev = _dev()
        self.buf = torch.zeros((len(self.layer_idx), E), dtype=torch.int64, device=dev)
        self.ones = torch.ones(max_tokens * topk, dtype=torch.int64, device=dev)
        if self.path:
            threading.Thread(target=self._loop, daemon=True, name="hotsplit-live").start()
        logger.info("hotsplit live counter: %d layers -> %s", len(self.layer_idx), self.path or "(in-memory only)")

    def count(self, slot: int, topk_ids: torch.Tensor):
        ids = topk_ids.reshape(-1).to(torch.int64)
        self.buf[slot].index_add_(0, ids, self.ones[: ids.numel()])

    def _loop(self):
        t0 = time.time()
        while True:
            time.sleep(self.secs)
            try:
                snap = self.buf.to("cpu", non_blocking=False)
                out = {"live": {"decode": {str(li): snap[s].tolist() for s, li in enumerate(self.layer_idx)}},
                       "meta": {"since": t0, "at": time.time(), "note": "live lane counts incl. padded slots"}}
                tmp = self.path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(out, f)
                os.replace(tmp, self.path)
            except Exception as e:  # never take the lane down for telemetry
                logger.warning("hotsplit live snapshot failed: %s", e)


# --------------------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------------------
def apply_hotsplit(model: torch.nn.Module) -> None:
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        make_mxfp4_moe_kernel,
        make_mxfp4_moe_quant_config,
    )
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

    global _LIVE
    if getattr(model, "_hotsplit_applied", False):
        return
    model._hotsplit_applied = True
    rank, tp = _tp_info()
    tag = f"[rank {rank}/{tp}]"
    dynamic = os.environ.get("HOTSPLIT_DYNAMIC") == "1"
    t0 = time.time()
    shares = load_counts(os.environ["HOTSPLIT_COUNTS"], os.environ.get("HOTSPLIT_WEIGHTS", ""))
    # Collect by FULL module path, not by bare layer index. A model can carry more than one
    # `layers.<n>` subtree (e.g. `model.layers.3` and `model.mtp.layers.3` for an MTP drafter),
    # and keying by the numeric index alone lets one silently overwrite the other. MTP subtrees
    # are excluded unless the histogram explicitly contains their index space (see below).
    layers: dict[int, torch.nn.Module] = {}
    _seen_paths: dict[int, str] = {}
    _mtp_seen = 0
    for name, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if not (isinstance(qm, Mxfp4MoEMethod) and hasattr(mod, "w13_weight") and qm.moe_kernel is not None):
            continue
        m = _LAYER_RE.search(name)
        if not m:
            continue
        li = int(m.group(1))
        if _is_mtp_module(name):
            # MTP drafter layers live in their own subtree. The routing histogram describes the
            # main decoder; do not fold drafter layers into the same plan by accident.
            _mtp_seen += 1
            logger.warning("hotsplit %s: skipping MTP layer module %r (index %d) -- the routing "
                           "histogram covers the main decoder only; drafter layers are left stock",
                           tag, name, li)
            continue
        if li in layers:
            # Two main-decoder modules claim the same index: refuse rather than guess which wins.
            raise RuntimeError(
                f"hotsplit {tag}: duplicate layer index {li} at {_seen_paths[li]!r} and {name!r}; "
                "the plan is keyed by layer index and must be unambiguous. Not touching any weights.")
        _seen_paths[li] = name
        layers[li] = mod
    if not layers:
        logger.warning("hotsplit %s: no eligible MXFP4 MoE layers found; nothing to do", tag)
        return
    if _mtp_seen:
        logger.info("hotsplit %s: %d MTP drafter module(s) left stock", tag, _mtp_seen)
    # Strict consistency gate BEFORE any weight is modified: the histogram's layer-index set and
    # the model's MoE-layer index set must match EXACTLY. A mismatch is an error, not a warning:
    #   * histogram entry with no model layer  -> stale/truncated histogram or a load that dropped
    #     layers (--hf-overrides num_hidden_layers); planning on it would silently under-split.
    #   * model layer with no histogram entry -> was skipped above and would silently stay stock.
    _hist_set = set(shares)
    _model_set = set(_seen_paths)
    _hist_only = sorted(_hist_set - _model_set)
    _model_only = sorted(_model_set - _hist_set)
    if _hist_only or _model_only:
        raise RuntimeError(
            f"hotsplit {tag}: histogram/model layer-index mismatch -- refusing before touching any "
            f"weight. In histogram but not in model: {_hist_only or '[]'}; in model but not in "
            f"histogram: {_model_only or '[]'}. Regenerate the histogram for this exact checkpoint "
            "and engine build (see patches/hotsplit_rerank.py), or fix the load.")
    logger.info("hotsplit %s: collected %d MoE layer(s) from %d module(s); histogram covers the same "
                "%d layer index(es) -- exact match", tag, len(layers), len(_seen_paths), len(shares))
    first = next(iter(layers.values()))
    E = first.w13_weight.shape[0]
    gne = getattr(first, "global_num_experts", E)
    if gne != E:
        logger.warning("hotsplit %s: expert parallelism is active (%d local of %d experts); the expert maps here "
                       "assume global ids == local rows. Use TP without --enable-expert-parallel. Not splitting.",
                       tag, E, gne)
        return
    cell_bytes = {li: (l.w13_weight[0].numel() * l.w13_weight.element_size()
                       + l.w2_weight[0].numel() * l.w2_weight.element_size()) for li, l in layers.items()}
    hbm_now = sum(cell_bytes[li] * E for li, l in layers.items() if not _is_uva(l.w13_weight))
    budget = float(os.environ["HOTSPLIT_HOT_GIB"]) * 2**30 if os.environ.get("HOTSPLIT_HOT_GIB") else hbm_now
    hot, _ = plan_hot_cells({li: shares[li] for li in layers}, cell_bytes, budget, E)
    cov = sum(float(shares[li][hot[li]].sum()) for li in layers) / sum(float(shares[li].sum()) for li in layers)
    logger.info("hotsplit plan %s: %d layers, budget %.1f GiB/rank (stock GPU-resident experts %.1f GiB), hot cells %d/%d, "
                "train-weighted coverage %.1f%%, mode %s", tag, len(layers), budget / 2**30, hbm_now / 2**30,
                sum(len(v) for v in hot.values()), E * len(layers), 100 * cov, "dynamic" if dynamic else "static")
    logger.info("hotsplit per-layer hot counts %s: %s", tag, {li: len(hot[li]) for li in sorted(hot)})
    if os.environ.get("HOTSPLIT_DRYRUN") == "1":
        return

    live_path = rank_path(os.environ.get("HOTSPLIT_LIVE_COUNTS"), rank, tp)
    if live_path or dynamic:
        _LIVE = _LiveCounter(live_path, int(os.environ.get("HOTSPLIT_LIVE_SECS", "600")))
    dyn = None
    if dynamic:
        dyn = _Dyn(sorted(layers), E)
        dyn.tag = tag
        dyn.set_prior({li: shares[li] for li in layers})

    # order: UVA layers (consume GPU memory, free host) while GPU memory allows, else a GPU-resident layer (frees GPU memory, consumes host)
    uva_q = [li for li in sorted(layers) if _is_uva(layers[li].w13_weight)]
    hbm_q = [li for li in sorted(layers) if not _is_uva(layers[li].w13_weight)]
    margin = 6 * 2**30
    floor = float(os.environ.get("HOTSPLIT_HOST_FLOOR_GIB", "24"))
    done = 0
    while uva_q or hbm_q:
        free, _ = torch.cuda.mem_get_info()
        if uva_q and (free - len(hot[uva_q[0]]) * cell_bytes[uva_q[0]] > margin or not hbm_q):
            li = uva_q.pop(0)
        else:
            li = hbm_q.pop(0)
        # --- VRAM pre-check for the hot-bank copy of THIS layer (byte-accurate, per rank) ---
        # The hot copy is made before the originals are freed, so the transient peak is
        # (hot weights + hot scales). Refuse loudly before allocating if it cannot fit; a
        # mid-copy OOM would leave the layer half-repartitioned.
        _nhot = len(hot[li])
        _hot_bytes = (_nhot * cell_bytes[li]
                      + _row_bytes(getattr(layers[li], "w13_weight_scale", None), _nhot)
                      + _row_bytes(getattr(layers[li], "w2_weight_scale", None), _nhot))
        _free_now, _ = torch.cuda.mem_get_info()
        _reserve_gib = float(os.environ.get("HOTSPLIT_VRAM_RESERVE_GIB", "2"))
        _pre_ok, _reserve = _vram_precheck(_hot_bytes, _free_now, _reserve_gib)
        if not _pre_ok:
            # Refuse the WHOLE split, not just this layer. A silent `break` would leave a
            # half-hotsplit model: already-split layers banked, the rest stock -- correct but
            # surprising, and impossible to reason about from the outside. Abort loudly.
            # This runs BEFORE any weight of this layer is touched; if it fires on the FIRST
            # layer, no weight in the model has been modified yet.
            raise RuntimeError(
                f"hotsplit {tag}: VRAM pre-check FAILED for layer {li} of {len(layers)}: hot copy "
                f"needs {_hot_bytes / 2**30:.2f} GiB (weights+scales, {_nhot} rows) but only "
                f"{_free_now / 2**30:.2f} GiB free with a {_reserve / 2**30:.2f} GiB reserve. "
                f"{done} layer(s) already split, {len(uva_q) + len(hbm_q)} left stock. "
                "Lower HOTSPLIT_HOT_GIB, raise HOTSPLIT_VRAM_RESERVE_GIB headroom accounting, or free "
                "VRAM and relaunch; nothing outside the container was modified.")
        _peak0 = _mem_peak_start()
        if dynamic:  # offloaded layers are reused in place; resident layers get a full host copy
            host_need = 0.0 if _is_uva(layers[li].w13_weight) else E * cell_bytes[li] / 2**30
        else:
            host_need = (E - len(hot[li])) * cell_bytes[li] / 2**30
        # all TP ranks load at the same time and share MemAvailable: assume each needs the same
        if _host_avail_gib() - host_need * tp < floor:
            logger.warning("hotsplit %s: host floor hit (avail %.1f GiB, need %.1f x %d ranks). "
                           "PARTIAL SPLIT BY DESIGN: %d layer(s) hotsplit, %d left stock (layer %d and the "
                           "rest). The model stays correct -- stock layers simply fall back to whole-layer "
                           "residency -- but hit rate is lower than planned.",
                           tag, _host_avail_gib(), host_need, tp, done, len(uva_q) + len(hbm_q), li)
            break
        if dyn is not None:
            _split_layer_dynamic(layers[li], hot[li], E, make_mxfp4_moe_kernel, make_mxfp4_moe_quant_config, li, dyn)
        else:
            _split_layer_static(layers[li], hot[li], E, make_mxfp4_moe_kernel, make_mxfp4_moe_quant_config, li)
        done += 1
        _mem_peak_report(tag, li, _peak0)   # per-layer peak + original-weight release check
        if done % 10 == 0:
            torch.cuda.empty_cache()
            logger.info("hotsplit %s: %d/%d layers; GPU memory free %.1f GiB; host avail %.1f GiB",
                        tag, done, len(layers), torch.cuda.mem_get_info()[0] / 2**30, _host_avail_gib())
    torch.cuda.empty_cache()
    if _LIVE is not None and _LIVE.layer_idx:
        _LIVE.finalize(E, int(os.environ.get("HOTSPLIT_MAX_TOKENS", "8192")), int(os.environ.get("HOTSPLIT_TOPK", "8")))
    if dyn is not None and dyn.layers:
        _hook_ok = _install_step_hook(dyn)
        _mode, _err = _dynamic_hook_decision(
            _hook_ok, dynamic,
            os.environ.get("HOTSPLIT_ALLOW_STATIC_FALLBACK", "0") == "1")
        logger.info("hotsplit %s: dynamic_hook=%s (mode=%s)",
                    tag, "ACTIVE" if _hook_ok else "INACTIVE", _mode)
        if _err:
            raise RuntimeError(f"hotsplit {tag}: {_err}")
        if dynamic and not _hook_ok:
            logger.warning("hotsplit %s: falling back to STATIC hotsplit by explicit "
                           "HOTSPLIT_ALLOW_STATIC_FALLBACK=1; do not use this run to compare "
                           "dynamic vs static", tag)
    elif dynamic:
        logger.info("hotsplit %s: dynamic_hook=INACTIVE (mode=static, no dynamic layers)", tag)
    logger.info("hotsplit done %s in %.0fs; GPU memory free %.1f GiB; host avail %.1f GiB",
                tag, time.time() - t0, torch.cuda.mem_get_info()[0] / 2**30, _host_avail_gib())
