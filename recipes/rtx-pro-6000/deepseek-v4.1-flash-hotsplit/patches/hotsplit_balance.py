#!/usr/bin/env python3
"""hotsplit_balance.py -- expert-balance calculator for TP4 workstations (numpy only, no GPU).

This is the *balance* half of hotsplit: given a model's expert geometry and a routing
histogram, it decides which routed experts live in GPU memory (the "hot" bank) and which stay in
pinned host memory (the "cold" bank), under a per-rank byte budget -- the same greedy
share-per-byte rule the mimo-v2.6-pro-hotsplit recipe uses (hotsplit_plan.plan_hot_cells).

It is deliberately independent of vLLM/SGLang and of any model file: you feed it geometry
(slab `balance.json`) and a histogram JSON (`HOTSPLIT_COUNTS`), it prints the plan. Use it to
size `HOTSPLIT_HOT_GIB` before you ever launch a server.

    python3 hotsplit_balance.py balance.json --gib 52
    python3 hotsplit_balance.py balance.json --counts my_traffic.json --gib 52 --dump-hot /tmp/hot.json

Geometry file (balance.json):
    {"layers": 40, "experts": 384, "topk": 6, "cell_mib_full": 17.9, "tp": 4}

Histogram JSON: same formats load_counts() accepts, i.e. either
    {"train": {"decode"|"prefill": {layer: [E counts]}}}  or  {layer: {"counts": [E counts]}}
If --counts is omitted a uniform prior is used (every expert equal), which yields a
deterministic id-order hot set -- useful only for sizing, not for real placement.

Per-rank math: a full expert is `cell_mib_full` MiB; at TP=tp each rank holds a 1/tp shard,
so one expert slice is `cell_mib_full / tp` MiB. All byte budgets below are PER RANK.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hotsplit_plan import load_counts, plan_hot_cells  # noqa: E402

MiB = 1024 ** 2
GiB = 1024 ** 3


def geometry(path: str) -> dict:
    with open(path) as f:
        g = json.load(f)
    for k in ("layers", "experts", "topk", "cell_mib_full", "tp"):
        if k not in g:
            raise SystemExit(f"balance.json missing '{k}'")
    return g


def uniform_shares(layers: int, experts: int) -> dict[int, np.ndarray]:
    return {li: np.full(experts, 1.0 / experts, dtype=np.float64) for li in range(layers)}


def load_or_uniform(counts: str | None, layers: int, experts: int, weights: str) -> dict[int, np.ndarray]:
    """Load a histogram, rejecting partial coverage: a histogram with fewer layers than the
    model config produces a misleadingly confident plan (e.g. 37/75 layers reported 94.8%
    coverage when the real coverage is ~47%). The runtime rejects such histograms; this
    utility must too (review bug G)."""
    if not counts:
        return uniform_shares(layers, experts)
    import json
    with open(counts) as f:
        raw = json.load(f)
    hist_layers = set()
    if "train" in raw:
        for phase in raw["train"].values():
            hist_layers.update(int(k) for k in phase.keys())
    else:
        hist_layers.update(int(k) for k in raw.keys())
    if hist_layers and len(hist_layers) != layers:
        raise ValueError(
            f"partial histogram: {len(hist_layers)} layers found but the model has {layers}. "
            f"Coverage numbers from a partial histogram are misleading; "
            f"collect counts for all {layers} layers first.")
    return uniform_shares(layers, experts) if not counts else _load_real(counts, layers, experts, weights)

def _load_real(counts: str, layers: int, experts: int, weights: str) -> dict[int, np.ndarray]:
    if not counts:
        return uniform_shares(layers, experts)
    shares = load_counts(counts, weights)
    for li, v in shares.items():
        if v.shape[0] != experts:
            raise SystemExit(f"histogram layer {li} has {v.shape[0]} experts, geometry says {experts}")
    return shares


def plan(g: dict, shares: dict[int, np.ndarray], budget_gib: float):
    layers, experts, tp = g["layers"], g["experts"], g["tp"]
    cell_rank = g["cell_mib_full"] / tp * MiB            # bytes of one expert shard, per rank
    cell_bytes = {li: int(cell_rank) for li in range(layers)}
    budget = budget_gib * GiB
    hot, used = plan_hot_cells(shares, cell_bytes, budget, experts)
    return hot, used, cell_rank


def report(g: dict, shares: dict, hot, used: int, cell_rank: float, budget_gib: float) -> dict:
    layers, experts, tp = g["layers"], g["experts"], g["tp"]
    bank_bytes = layers * experts * cell_rank
    hot_cells = sum(len(v) for v in hot.values())
    total_cells = layers * experts
    # routing coverage: share of routed tokens served from the hot bank
    tot = sum(float(s.sum()) for s in shares.values())
    cov = sum(float(shares[li][hot[li]].sum()) for li in hot if hot[li]) / max(tot, 1e-12)
    out = {
        "layers": layers, "experts": experts, "topk": g["topk"], "tp": tp,
        "cell_mib_full": g["cell_mib_full"], "cell_mib_rank": round(cell_rank / MiB, 3),
        "bank_gib_rank": round(bank_bytes / GiB, 2),
        "budget_gib_rank": round(budget_gib, 2),
        "hot_gib_rank": round(used / GiB, 2),
        "cold_gib_rank": round((bank_bytes - used) / GiB, 2),
        "host_gib_all_ranks": round((bank_bytes - used) * tp / GiB, 1),
        "hot_cells": hot_cells, "total_cells": total_cells,
        "hot_fraction": round(hot_cells / total_cells, 4),
        "routing_coverage": round(cov, 4),
        "decode_bytes_per_tok_mib_rank": round(g["topk"] * cell_rank / MiB, 1),
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("balance_json", help="geometry file")
    ap.add_argument("--counts", help="routing histogram JSON (omit for a uniform prior)")
    ap.add_argument("--weights", default="", help="phase weights, e.g. 'decode=1,prefill=0'")
    ap.add_argument("--gib", type=float, required=True, help="hot budget, GiB PER RANK")
    ap.add_argument("--dump-hot", help="write the {layer: [expert ids]} hot map to this JSON file")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    a = ap.parse_args(argv)

    g = geometry(a.balance_json)
    shares = load_or_uniform(a.counts, g["layers"], g["experts"], a.weights)
    hot, used, cell_rank = plan(g, shares, a.gib)
    out = report(g, shares, hot, used, cell_rank, a.gib)

    if a.dump_hot:
        with open(a.dump_hot, "w") as f:
            json.dump({str(li): v for li, v in sorted(hot.items())}, f)
        out["hot_map"] = a.dump_hot

    if a.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"model geometry: {out['layers']} layers x {out['experts']} experts, top-{out['topk']}, TP{out['tp']}")
        print(f"one expert slice: {out['cell_mib_rank']} MiB/rank (full {out['cell_mib_full']} MiB)")
        print(f"routed bank:      {out['bank_gib_rank']} GiB/rank")
        print(f"budget:           {out['budget_gib_rank']} GiB/rank")
        print(f"HOT  (GPU):       {out['hot_gib_rank']} GiB/rank  = {out['hot_cells']}/{out['total_cells']} cells "
              f"({out['hot_fraction']*100:.1f}%)")
        print(f"COLD (host):      {out['cold_gib_rank']} GiB/rank, {out['host_gib_all_ranks']} GiB across {out['tp']} ranks")
        print(f"routing coverage: {out['routing_coverage']*100:.2f}% of routed tokens served from GPU memory")
        print(f"decode traffic:   ~{out['decode_bytes_per_tok_mib_rank']} MiB/token/rank (top-{out['topk']} x slice)")
    return 0


if __name__ == "__main__":
    sys.exit(main())