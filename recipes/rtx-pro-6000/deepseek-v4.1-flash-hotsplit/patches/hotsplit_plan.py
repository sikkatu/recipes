"""Pure planning helpers for hotsplit on tensor-parallel workstations (numpy only, unit-tested).

Nothing here imports torch or vLLM, so the logic can be tested without a GPU:
  * rank_path          per-rank file names so TP workers never write the same counts file
  * load_counts        read a routing histogram JSON (two normalisation modes, see below)
  * plan_hot_cells     initial hot set: greedy by routing share per byte under a byte budget
  * plan_hot_cells_oracle  exact brute-force optimum for SMALL problems only (test oracle)
  * decay_scores       exponential moving score of live routing counts
  * plan_swaps         which experts enter / leave one layer's hot bank at a dynamic update
  * swap_index_updates the (row, expert, value) triples that flip the two device expert maps

Validation policy: invalid statistics (NaN/inf, negative counts, expert-count mismatches,
missing layers, zero/negative budgets where disallowed) abort with a diagnostic error instead
of producing a plausible-but-wrong plan.

Normalisation modes (set HOTSPLIT_COUNTS_MODE):
  * "phase-balanced"  (default) — each phase is normalised to sum 1 independently, then scaled by
    its weight. Phases therefore have equal total influence regardless of how many observations
    each contains. This is the historical behaviour.
  * "traffic-weighted" — each phase contributes proportionally to its observed volume (raw sums),
    so a phase with 100x more tokens dominates. Use when the histogram is a faithful traffic mix.

Heuristic caveat: plan_hot_cells is a GREEDY by share-per-byte — it is not guaranteed optimal
(see plan_hot_cells_oracle for comparison on small inputs).
"""
from __future__ import annotations

import json
import os
import numpy as np


def rank_path(path: str | None, rank: int, tp_size: int) -> str | None:
    """counts.json -> counts.rank2.json when tp_size > 1; '{rank}' in the path is substituted instead."""
    if not path or tp_size <= 1:
        return path
    if "{rank}" in path:
        return path.replace("{rank}", str(rank))
    root, ext = os.path.splitext(path)
    return f"{root}.rank{rank}{ext}"


def _validate_counts(arr: np.ndarray, layer: int, experts: int, ctx: str) -> np.ndarray:
    """Reject invalid routing statistics before they can seed a wrong plan."""
    if arr.ndim != 1:
        raise ValueError(f"{ctx}: layer {layer}: expected 1-D count vector, got shape {arr.shape}")
    if arr.shape[0] != experts:
        raise ValueError(f"{ctx}: layer {layer}: {arr.shape[0]} experts in histogram, geometry says {experts}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{ctx}: layer {layer}: non-finite values (NaN/inf) in counts")
    if (arr < 0).any():
        raise ValueError(f"{ctx}: layer {layer}: negative counts are not valid statistics")
    return arr


def load_counts(path: str, weights_env: str = "", mode: str = "") -> dict[int, np.ndarray]:
    """Per-layer routing share vectors from a histogram JSON.

    Formats: {"train": {"decode"|"prefill": {layer: [E counts]}}} (phases mixed by weight, default
    decode=1, prefill=0.25, override with 'decode=1,prefill=0') or {layer: {"counts": [E counts]}}.

    Normalisation mode (HOTSPLIT_COUNTS_MODE or `mode`):
      phase-balanced  - each phase normalised to sum 1 then scaled by weight (equal total influence)
      traffic-weighted - phases contribute proportionally to raw observation volume
    """
    if not mode:
        mode = os.environ.get("HOTSPLIT_COUNTS_MODE", "phase-balanced")
    if mode not in ("phase-balanced", "traffic-weighted"):
        raise ValueError(f"unknown normalisation mode {mode!r}; use phase-balanced or traffic-weighted")
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: empty or invalid histogram JSON")
    weights = {"decode": 1.0, "prefill": 0.25}
    for kv in weights_env.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                weights[k.strip()] = float(v)
            except ValueError as e:
                raise ValueError(f"invalid phase weight {kv!r}: {e}") from e
    out: dict[int, np.ndarray] = {}
    if "train" in raw:
        phases = raw["train"]
        if not isinstance(phases, dict) or not phases:
            raise ValueError(f"{path}: 'train' block is empty or invalid")
        # for traffic-weighted mode we need the raw phase totals first
        raw_sums = {}
        for phase, layers in phases.items():
            total = 0.0
            for li, c in layers.items():
                arr = np.asarray(c, dtype=np.float64)
                experts = arr.shape[0] if arr.ndim == 1 else 0
                arr = _validate_counts(arr, int(li), experts, path)
                total += float(arr.sum())
            raw_sums[phase] = max(total, 1e-12)
        for phase, layers in phases.items():
            w = weights.get(phase, 1.0)
            for li, c in layers.items():
                arr = np.asarray(c, dtype=np.float64)
                experts = arr.shape[0] if arr.ndim == 1 else 0
                arr = _validate_counts(arr, int(li), experts, path)
                if mode == "phase-balanced":
                    t = arr / max(float(arr.sum()), 1.0) * w
                else:
                    # traffic-weighted: contribution is proportional to the phase's raw volume
                    # times its weight -- divide by the total ONCE. (The old code also multiplied
                    # w by raw_sums[phase]/total here, which squared the volume influence:
                    # a phase 10x larger with weight 0.25 contributed 400:1 instead of 40:1.)
                    t = arr / sum(raw_sums.values()) * w
                out[int(li)] = out[int(li)] + t if int(li) in out else t
    else:
        for li, v in raw.items():
            if not isinstance(v, dict) or "counts" not in v:
                raise ValueError(f"{path}: layer {li} entry must be {{'counts': [...]}}")
            arr = np.asarray(v["counts"], dtype=np.float64)
            experts = arr.shape[0] if arr.ndim == 1 else 0
            arr = _validate_counts(arr, int(li), experts, path)
            out[int(li)] = arr / max(float(arr.sum()), 1.0)
    return out


def plan_hot_cells(shares: dict[int, np.ndarray], cell_bytes: dict[int, int], budget: float,
                   num_experts: int) -> tuple[dict[int, list[int]], int]:
    """GREEDY hot set (heuristic, not guaranteed optimal).

    A cell is (layer, expert); rank by routing share per byte, take while it fits. Cells that do
    not fit are skipped, smaller ones may still fit later. Returns ({layer: sorted expert ids},
    bytes used). For an exact optimum on small inputs use plan_hot_cells_oracle.

    Validation: non-finite shares, negative budgets, or zero cell sizes abort with an error.
    """
    if budget < 0 or not np.isfinite(budget):
        raise ValueError(f"plan_hot_cells: budget must be finite and >= 0, got {budget}")
    for li, c in shares.items():
        if not np.isfinite(c).all():
            raise ValueError(f"plan_hot_cells: layer {li}: non-finite routing shares")
        if (c < 0).any():
            raise ValueError(f"plan_hot_cells: layer {li}: negative routing shares")
        if c.shape[0] != num_experts:
            raise ValueError(f"plan_hot_cells: layer {li}: {c.shape[0]} experts, expected {num_experts}")
        cb = cell_bytes[li]
        if cb <= 0:
            raise ValueError(f"plan_hot_cells: layer {li}: cell size must be > 0, got {cb}")
    cells = []
    for li, c in shares.items():
        cb = cell_bytes[li]
        for e in range(num_experts):
            cells.append((float(c[e]) / cb, li, e))
    cells.sort(reverse=True)
    hot: dict[int, list[int]] = {li: [] for li in shares}
    used = 0
    for _, li, e in cells:
        cb = cell_bytes[li]
        if used + cb > budget:
            continue
        hot[li].append(e)
        used += cb
    for li in hot:
        hot[li].sort()
    return hot, used


def plan_hot_cells_oracle(shares: dict[int, np.ndarray], cell_bytes: dict[int, int], budget: float,
                          num_experts: int, max_total_cells: int = 20) -> tuple[dict[int, list[int]], int]:
    """EXACT brute-force optimum for SMALL inputs only (unit-test oracle).

    Enumerates subsets when the total number of (layer, expert) cells is <= max_total_cells.
    Refuses larger inputs: the search is O(2^n) and must never run on a real model's expert bank.
    Use this to measure the greedy planner's gap on synthetic inputs.
    """
    cells = []
    for li, c in shares.items():
        cb = cell_bytes[li]
        for e in range(num_experts):
            cells.append((float(c[e]), li, e, cb))
    n = len(cells)
    if n > max_total_cells:
        raise ValueError(
            f"plan_hot_cells_oracle: {n} cells exceeds max_total_cells={max_total_cells}; "
            "brute force is a test oracle only, never run it on a full model")
    if budget < 0:
        raise ValueError(f"plan_hot_cells_oracle: budget must be >= 0, got {budget}")
    best_val = -1.0
    best_used = 0
    best_set: tuple[tuple[int, int], ...] = ()
    for mask in range(1 << n):
        used = 0
        val = 0.0
        ok = True
        chosen = []
        for i in range(n):
            if mask >> i & 1:
                used += cells[i][3]
                if used > budget:
                    ok = False
                    break
                val += cells[i][0]
                chosen.append((cells[i][1], cells[i][2]))
        if ok and val > best_val:
            best_val, best_used, best_set = val, used, tuple(chosen)
    hot: dict[int, list[int]] = {li: [] for li in shares}
    for li, e in best_set:
        hot[li].append(e)
    for li in hot:
        hot[li].sort()
    return hot, best_used


def decay_scores(score: np.ndarray, delta: np.ndarray, decay: float) -> np.ndarray:
    """score <- decay * score + delta (delta = routed-token count since the last update)."""
    if not (0.0 <= decay <= 1.0):
        raise ValueError(f"decay_scores: decay must be in [0, 1], got {decay}")
    if score.shape != delta.shape:
        raise ValueError(f"decay_scores: shape mismatch {score.shape} vs {delta.shape}")
    return decay * score + delta


def plan_swaps(score: np.ndarray, hot: list[int], max_swaps: int, margin: float) -> list[tuple[int, int, int]]:
    """Swaps for one layer: [(slot, expert_out, expert_in)].

    The weakest hot slots are paired with the strongest cold experts; a pair is taken only while
    score_in > margin * score_out (and score_in > 0). Because victims are sorted ascending and
    candidates descending, the first failing pair ends the plan. At most max_swaps pairs.

    Invariants (checked by tests): no expert appears in both hot and cold; hot size is unchanged;
    indices are in [0, len(score)); equal/zero scores never swap (deterministic, no churn).
    """
    s = len(hot)
    if s == 0 or max_swaps <= 0:
        return []
    hot_arr = np.asarray(hot, dtype=np.int64)
    if hot_arr.min() < 0 or hot_arr.max() >= len(score):
        raise ValueError(f"plan_swaps: hot indices out of range [0, {len(score)})")
    if len(set(hot)) != s:
        raise ValueError("plan_swaps: duplicate entries in hot list")
    if margin < 1.0:
        raise ValueError(f"plan_swaps: margin must be >= 1.0, got {margin}")
    victims = np.argsort(score[hot_arr], kind="stable")  # slot indices, weakest first
    cand = np.array(score, dtype=np.float64, copy=True)
    cand[hot_arr] = -np.inf
    n = min(max_swaps, s, len(score) - s)
    if n <= 0:
        return []
    order = np.argsort(-cand, kind="stable")[:n]
    out: list[tuple[int, int, int]] = []
    for k in range(n):
        slot = int(victims[k])
        e_out = int(hot_arr[slot])
        e_in = int(order[k])
        s_in, s_out = float(score[e_in]), float(score[e_out])
        if s_in <= 0.0 or s_in <= margin * s_out:
            break
        out.append((slot, e_out, e_in))
    return out


def swap_index_updates(row: int, swaps: list[tuple[int, int, int]]):
    """Index/value triples that flip the two [layers, experts] int32 expert maps for one layer.

    emap_hot[row, e]  = slot in the hot bank, or -1      emap_cold[row, e] = row in the cold bank, or -1
    After a swap (slot, out, in): hot[in] = slot, hot[out] = -1, cold[in] = -1, cold[out] = out.
    Returns (hot_rows, hot_cols, hot_vals, cold_rows, cold_cols, cold_vals) as python lists.

    Invariant (checked by tests): after applying the updates to fresh maps, every hot slot maps to
    exactly one expert and no expert sits in both banks. Applying several updates in sequence keeps
    the maps consistent.
    """
    hr, hc, hv, cr, cc, cv = [], [], [], [], [], []
    for slot, e_out, e_in in swaps:
        if slot < 0 or e_out < 0 or e_in < 0:
            raise ValueError(f"swap_index_updates: negative index in {(slot, e_out, e_in)}")
        hr += [row, row]
        hc += [e_in, e_out]
        hv += [slot, -1]
        cr += [row, row]
        cc += [e_in, e_out]
        cv += [-1, e_out]
    return hr, hc, hv, cr, cc, cv