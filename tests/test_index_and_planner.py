"""Index generator tests (audit block 4) and planner invariants (block 8).

Index: escaping keeps the table structure intact for every adversarial value; Model stays
short; Headline shows at most one number; the main metric follows the explicit `primary`
flag; dates normalise to YYYY-MM-DD.

Planner: input validation rejects invalid statistics instead of producing a plausible plan;
the two normalisation modes really differ; the greedy heuristic is checked against an exact
brute-force oracle on small inputs; expert-map swaps keep their invariants over many updates.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import render_index as ri  # noqa: E402

PLAN = ROOT / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit" / "patches" / "hotsplit_plan.py"
sys.path.insert(0, str(PLAN.parent))
import hotsplit_plan as hp  # noqa: E402


# ================================================================== index escaping
def test_md_escape_pipes_and_backslashes():
    assert ri.md_escape("mean |dlogp|") == r"mean \|dlogp\|"
    assert ri.md_escape("a\\b") == "a\\\\b"
    assert ri.md_escape("line1\nline2") == "line1 line2"
    assert ri.md_escape(None) == ""


def test_table_columns_stay_constant_for_adversarial_values():
    """A value like 'mean |dlogp|' must not create extra columns."""
    r = {
        "title": "T",
        "status": "draft",
        "updated": "2026-01-01",
        "model": {"name": "Model | with | pipes", "quantization": "x|y"},
        "metrics": [{"value": 1.5, "unit": "|", "workload": "w", "label": "a|b"}],
    }
    row = f"| hw | [t](u/) | {ri.model_cell(r)} | {ri.STATUS_ICON['draft']} | {ri.headline(r)} | {ri.norm_date(r['updated'])} |"
    header_cols = len(ri.render().splitlines()[2].split("|"))
    # count unescaped pipes
    import re
    unescaped = re.split(r"(?<!\\)\|", row)
    assert len(unescaped) == header_cols, f"column count changed: {len(unescaped)} vs {header_cols}"


def test_headline_single_number_no_protocol():
    r = {"metrics": [{"value": 42.5, "unit": "tok/s", "workload": "T=0, ignore_eos, 192 tok, warm, prose, knee.sh r2, 3 reps", "label": None}]}
    h = ri.headline(r)
    assert "42.5" in h
    assert "knee.sh" not in h, "test protocol must not leak into the index"
    assert h.count("|") == 0


def test_primary_metric_flag_wins():
    r = {"metrics": [
        {"value": 1, "unit": "a", "workload": "w"},
        {"value": 2, "unit": "b", "workload": "w", "primary": True},
    ]}
    assert "2" in ri.headline(r)


def test_no_metrics_shows_dash():
    assert ri.headline({}) == "—"
    assert ri.headline({"metrics": []}) == "—"


def test_date_normalisation():
    import datetime
    assert ri.norm_date("2026-10-15") == "2026-10-15"
    assert ri.norm_date("2026-10-15T12:00:00") == "2026-10-15"
    assert ri.norm_date(datetime.date(2026, 1, 2)) == "2026-01-02"
    assert ri.norm_date(None) == "—"
    assert ri.norm_date("garbage") == "—"


def test_model_cell_short():
    r = {"model": {"name": "MiMo-V2.6-Pro", "quantization": "MXFP4 (I8 + E8M0 block scales), routed experts 494.9 GiB; long text"}}
    c = ri.model_cell(r)
    assert "MiMo-V2.6-Pro" in c
    assert "494.9" not in c, "file sizes must not leak into the index"
    assert len(c) < 80


def test_unicode_long_titles_survive():
    r = {"title": "Очень длинный заголовок " * 20, "model": {"name": "модель"}, "status": "draft", "updated": None}
    row = ri.model_cell(r)
    assert "|" not in row.replace("\\|", "")


# ================================================================== planner validation
def test_load_counts_rejects_nan_and_negative(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"0": {"counts": [1, float("nan"), 3]}}))
    with pytest.raises(ValueError, match="non-finite"):
        hp.load_counts(str(p))
    p.write_text(json.dumps({"0": {"counts": [1, -2, 3]}}))
    with pytest.raises(ValueError, match="negative"):
        hp.load_counts(str(p))


def test_load_counts_rejects_empty_and_bad_json(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="empty"):
        hp.load_counts(str(p))
    p.write_text("")
    with pytest.raises(Exception):
        hp.load_counts(str(p))


def test_plan_hot_cells_rejects_bad_inputs():
    shares = {0: np.array([0.5, 0.5])}
    with pytest.raises(ValueError, match="budget"):
        hp.plan_hot_cells(shares, {0: 10}, -1.0, 2)
    with pytest.raises(ValueError, match="cell size"):
        hp.plan_hot_cells(shares, {0: 0}, 10, 2)
    with pytest.raises(ValueError, match="experts"):
        hp.plan_hot_cells(shares, {0: 10}, 10, 5)
    with pytest.raises(ValueError, match="non-finite"):
        hp.plan_hot_cells({0: np.array([0.5, np.nan])}, {0: 10}, 10, 2)


def test_normalisation_modes_differ(tmp_path):
    """phase-balanced: equal influence; traffic-weighted: proportional to volume."""
    p = tmp_path / "c.json"
    # decode has 10x the observations of prefill
    p.write_text(json.dumps({"train": {
        "decode": {"0": [100, 0]},
        "prefill": {"0": [0, 10]},
    }}))
    pb = hp.load_counts(str(p), "", mode="phase-balanced")
    tw = hp.load_counts(str(p), "", mode="traffic-weighted")
    # phase-balanced: decode share .5*1 + prefill .5*0.25 -> expert0 dominates
    assert pb[0][0] > pb[0][1]
    # traffic-weighted: decode (100 obs) dominates even more
    assert tw[0][0] / max(tw[0][1], 1e-12) > pb[0][0] / max(pb[0][1], 1e-12)
    # and the two really differ
    assert not np.allclose(pb[0], tw[0])


def test_mode_is_deterministic(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"train": {"decode": {"0": [1, 2]}, "prefill": {"0": [3, 4]}}}))
    a = hp.load_counts(str(p), "", mode="traffic-weighted")
    b = hp.load_counts(str(p), "", mode="traffic-weighted")
    assert np.allclose(a[0], b[0])


def test_greedy_vs_oracle_gap_measured():
    """The greedy planner is a heuristic: compare with the exact oracle on a small input."""
    rng = np.random.default_rng(42)
    shares = {0: rng.random(4), 1: rng.random(4)}
    cell = {0: 10, 1: 30}
    budget = 50
    g_hot, g_used = hp.plan_hot_cells(shares, cell, budget, 4)
    o_hot, o_used = hp.plan_hot_cells_oracle(shares, cell, budget, 4, max_total_cells=8)
    # oracle value >= greedy value always
    g_val = sum(float(shares[li][e]) for li in g_hot for e in g_hot[li])
    o_val = sum(float(shares[li][e]) for li in o_hot for e in o_hot[li])
    assert o_val >= g_val - 1e-9, "oracle must never be worse than greedy"
    assert g_used <= budget and o_used <= budget


def test_oracle_refuses_large_inputs():
    shares = {0: np.ones(30)}
    with pytest.raises(ValueError, match="oracle"):
        hp.plan_hot_cells_oracle(shares, {0: 1}, 10, 30, max_total_cells=20)


# ================================================================== expert-map invariants
def test_swap_invariants_over_many_updates():
    """After each swap: no expert in both banks, hot size constant, indices valid."""
    rng = np.random.default_rng(7)
    E = 10
    hot = list(range(5))
    cold = set(range(E)) - set(hot)
    for _ in range(20):
        score = rng.random(E) * 10
        sw = hp.plan_swaps(score, hot, max_swaps=2, margin=1.01)
        for slot, e_out, e_in in sw:
            assert e_in in cold, "expert entering hot must come from cold"
            assert e_out in hot, "expert leaving hot must be in hot"
            assert 0 <= slot < len(hot)
            hot[slot] = e_in
            cold.discard(e_in)
            cold.add(e_out)
        assert len(hot) == 5, "hot size must stay constant"
        assert len(set(hot)) == 5, "no duplicates in hot"
        assert set(hot).isdisjoint(cold), "no expert in both banks"


def test_swap_index_updates_map_consistency():
    """Apply updates to real maps; every hot slot maps to exactly one expert, no overlaps."""
    E = 6
    emap_hot = np.full(E, -1, dtype=np.int64)
    emap_cold = np.arange(E, dtype=np.int64)
    hot = [0, 1, 2]
    for i, e in enumerate(hot):
        emap_hot[e] = i
        emap_cold[e] = -1
    swaps = [(0, 0, 3), (1, 1, 4)]
    hr, hc, hv, cr, cc, cv = hp.swap_index_updates(0, swaps)
    for c, v in zip(hc, hv):
        emap_hot[c] = v
    for c, v in zip(cc, cv):
        emap_cold[c] = v
    # hot map: exactly 3 valid slots
    slots = emap_hot[emap_hot >= 0]
    assert sorted(slots.tolist()) == [0, 1, 2]
    # no expert in both banks
    in_hot = set(np.where(emap_hot >= 0)[0].tolist())
    in_cold = set(np.where(emap_cold >= 0)[0].tolist())
    assert in_hot.isdisjoint(in_cold)
    # all experts accounted for
    assert in_hot | in_cold == set(range(E))


def test_swap_index_updates_reject_negative():
    with pytest.raises(ValueError, match="negative"):
        hp.swap_index_updates(0, [(-1, 0, 1)])


def test_plan_swaps_zero_and_equal_scores_no_churn():
    score = np.zeros(6)
    assert hp.plan_swaps(score, [0, 1, 2], 2, 1.25) == []
    score = np.ones(6) * 5.0
    # equal scores never beat margin>1
    assert hp.plan_swaps(score, [0, 1], 2, 1.25) == []


def test_plan_swaps_deterministic():
    score = np.array([9.0, 1.0, 0.5, 8.0, 7.0, 0.0])
    a = hp.plan_swaps(score, [1, 2, 5], 2, 1.25)
    b = hp.plan_swaps(score, [1, 2, 5], 2, 1.25)
    assert a == b


def test_plan_swaps_rejects_bad_hot_list():
    score = np.ones(5)
    with pytest.raises(ValueError, match="out of range"):
        hp.plan_swaps(score, [0, 99], 1, 1.0)
    with pytest.raises(ValueError, match="duplicate"):
        hp.plan_swaps(score, [1, 1], 1, 1.0)
    with pytest.raises(ValueError, match="margin"):
        hp.plan_swaps(score, [0], 1, 0.5)


def test_decay_scores_validation():
    with pytest.raises(ValueError, match="decay"):
        hp.decay_scores(np.ones(3), np.ones(3), 1.5)
    with pytest.raises(ValueError, match="shape"):
        hp.decay_scores(np.ones(3), np.ones(4), 0.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))