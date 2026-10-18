"""Tests for the TP4 hotsplit recipe: the numpy-only planning logic, NUMA helpers, launcher syntax.

No torch/vLLM/GPU needed except `test_dynamic_commit_cpu`, which skips without torch. No fixtures are used
so the functions can also be called directly (see the __main__ block).
"""
from __future__ import annotations

import json
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

RECIPE = Path(__file__).resolve().parent.parent / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit"
PATCHES = RECIPE / "patches"
sys.path.insert(0, str(PATCHES))

import hotsplit_plan as hp  # noqa: E402
import numa_bind  # noqa: E402


def _skip(msg):
    try:
        import pytest

        pytest.skip(msg)
    except ImportError:
        print("SKIP:", msg)


# ---------------------------------------------------------------- rank_path / load_counts
def test_rank_path():
    assert hp.rank_path("/live/counts.json", 2, 4) == "/live/counts.rank2.json"
    assert hp.rank_path("/live/counts", 0, 4) == "/live/counts.rank0"
    assert hp.rank_path("/live/c_{rank}.json", 3, 4) == "/live/c_3.json"
    assert hp.rank_path("/live/counts.json", 0, 1) == "/live/counts.json"  # TP1: unchanged
    assert hp.rank_path(None, 1, 4) is None


def test_load_counts_train_format():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"
        p.write_text(json.dumps({"train": {"decode": {"3": [1, 3, 0, 0]}, "prefill": {"3": [0, 0, 5, 5]}}}))
        out = hp.load_counts(str(p), "")
        assert set(out) == {3}
        # decode weight 1 (0.25, 0.75, 0, 0) + prefill weight 0.25 (0, 0, 0.125, 0.125)
        assert np.allclose(out[3], [0.25, 0.75, 0.125, 0.125])
        out0 = hp.load_counts(str(p), "decode=1,prefill=0")
        assert np.allclose(out0[3], [0.25, 0.75, 0, 0])


def test_load_counts_flat_format():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"
        p.write_text(json.dumps({"7": {"counts": [2, 2, 0, 0]}}))
        assert np.allclose(hp.load_counts(str(p))[7], [0.5, 0.5, 0, 0])


# ---------------------------------------------------------------- plan_hot_cells
def test_plan_hot_cells_respects_budget_and_prefers_share_per_byte():
    shares = {0: np.array([0.7, 0.2, 0.05, 0.05]), 1: np.array([0.4, 0.4, 0.1, 0.1])}
    cell_bytes = {0: 10, 1: 10}
    hot, used = hp.plan_hot_cells(shares, cell_bytes, budget=30, num_experts=4)
    assert used == 30 and sum(len(v) for v in hot.values()) == 3
    assert hot[0] == [0] and hot[1] == [0, 1]  # shares .7, .4, .4 are the three largest


def test_plan_hot_cells_cheaper_layer_wins_per_byte():
    shares = {0: np.array([0.5, 0.5]), 1: np.array([0.5, 0.5])}
    hot, used = hp.plan_hot_cells(shares, {0: 10, 1: 5}, budget=10, num_experts=2)
    assert used == 10 and hot[0] == [] and hot[1] == [0, 1]


def test_plan_hot_cells_zero_budget_and_deterministic():
    shares = {0: np.array([0.5, 0.5])}
    hot, used = hp.plan_hot_cells(shares, {0: 10}, budget=0, num_experts=2)
    assert used == 0 and hot == {0: []}
    a = hp.plan_hot_cells(shares, {0: 10}, budget=10, num_experts=2)
    b = hp.plan_hot_cells(shares, {0: 10}, budget=10, num_experts=2)
    assert a == b  # every TP rank must derive the same list


# ---------------------------------------------------------------- scores and swaps
def test_decay_scores():
    assert np.allclose(hp.decay_scores(np.array([4.0, 2.0]), np.array([1.0, 0.0]), 0.5), [3.0, 1.0])


def test_plan_swaps_pairs_weakest_hot_with_strongest_cold():
    score = np.array([9.0, 1.0, 0.5, 8.0, 7.0, 0.0])
    hot = [1, 2, 5]  # slot 0 -> expert 1 (1.0), slot 1 -> expert 2 (0.5), slot 2 -> expert 5 (0.0)
    sw = hp.plan_swaps(score, hot, max_swaps=2, margin=1.25)
    assert sw == [(2, 5, 0), (1, 2, 3)]  # weakest (expert 5) out for strongest (expert 0), then 2 for 3
    assert len(hp.plan_swaps(score, hot, max_swaps=1, margin=1.25)) == 1


def test_plan_swaps_margin_and_no_gain():
    score = np.array([1.0, 1.1, 1.0, 1.0])
    assert hp.plan_swaps(score, [0, 2], 2, 1.25) == []  # 1.1 <= 1.25 * 1.0
    assert hp.plan_swaps(score, [0, 2], 2, 1.0) == [(0, 0, 1)]
    assert hp.plan_swaps(np.zeros(4), [0, 1], 2, 1.0) == []  # no signal: never swap


def test_plan_swaps_edge_sizes():
    score = np.array([3.0, 2.0, 1.0])
    assert hp.plan_swaps(score, [], 2, 1.0) == []
    assert hp.plan_swaps(score, [0, 1, 2], 2, 1.0) == []  # everything already hot
    assert hp.plan_swaps(score, [2], 0, 1.0) == []


def test_plan_swaps_cold_with_positive_score_replaces_zero_score_hot():
    score = np.array([0.0, 0.0, 5.0])
    assert hp.plan_swaps(score, [0], 1, 1.25) == [(0, 0, 2)]


def test_swap_updates_keep_every_expert_in_exactly_one_bank():
    """numpy emulation of the two device maps: after any sequence of swap plans each expert is hot XOR cold,
    the hot slot -> expert table matches emap_hot, and the cold map is the identity for cold experts."""
    rng = np.random.default_rng(0)
    E, S, row = 16, 5, 1
    hot = sorted(rng.choice(E, S, replace=False).tolist())
    emap_hot = np.full((3, E), -1, dtype=np.int32)
    emap_cold = np.tile(np.arange(E, dtype=np.int32), (3, 1))
    for slot, e in enumerate(hot):
        emap_hot[row, e] = slot
        emap_cold[row, e] = -1
    for _ in range(200):
        score = rng.random(E) * rng.integers(0, 3)
        sw = hp.plan_swaps(score, hot, int(rng.integers(0, 4)), float(rng.choice([1.0, 1.25])))
        hr, hc, hv, cr, cc, cv = hp.swap_index_updates(row, sw)
        for r, c, v in zip(hr, hc, hv):
            emap_hot[r, c] = v
        for r, c, v in zip(cr, cc, cv):
            emap_cold[r, c] = v
        for slot, _, e_in in sw:
            hot[slot] = e_in
        in_hot = emap_hot[row] >= 0
        in_cold = emap_cold[row] >= 0
        assert np.all(in_hot ^ in_cold)
        assert sorted(hot) == sorted(np.nonzero(in_hot)[0].tolist()) and len(set(hot)) == S
        for slot, e in enumerate(hot):
            assert emap_hot[row, e] == slot
        for e in np.nonzero(in_cold)[0]:
            assert emap_cold[row, e] == e
        assert np.all(emap_hot[0] == -1) and np.all(emap_cold[2] == np.arange(E))  # other layers untouched


# ---------------------------------------------------------------- NUMA helpers
def test_parse_cpulist():
    assert numa_bind.parse_cpulist("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
    assert numa_bind.parse_cpulist("") == []


def test_pci_addr_and_sysfs_lookup():
    assert numa_bind.pci_addr(0, 0x41, 0) == "0000:41:00.0"
    with tempfile.TemporaryDirectory() as d:
        dev = Path(d) / "bus/pci/devices/0000:41:00.0"
        dev.mkdir(parents=True)
        (dev / "numa_node").write_text("1\n")
        node = Path(d) / "devices/system/node/node1"
        node.mkdir(parents=True)
        (node / "cpulist").write_text("32-63,96-127\n")
        assert numa_bind.gpu_numa_node("0000:41:00.0", d) == 1
        assert numa_bind.gpu_numa_node("0000:99:00.0", d) == -1
        assert numa_bind.node_cpus(1, d)[:2] == [32, 33] and len(numa_bind.node_cpus(1, d)) == 64
        assert numa_bind.node_cpus(7, d) == []


def test_sitecustomize_is_inert_without_env():
    code = (
        "import os, sys; sys.path.insert(0, %r)\n"
        "for k in ('HOTSPLIT_NUMA', 'HOTSPLIT_BOOT_HOOK'): os.environ.pop(k, None)\n"
        "before = list(sys.meta_path)\n"
        "import runpy; runpy.run_path(%r)\n"
        "assert 'numa_bind' not in sys.modules and 'torch' not in sys.modules\n"
        "assert sys.meta_path == before\n"
    ) % (str(PATCHES), str(PATCHES / "sitecustomize.py"))
    subprocess.run([sys.executable, "-c", code], check=True)


# ---------------------------------------------------------------- syntax of everything we ship
def test_python_files_compile():
    with tempfile.TemporaryDirectory() as d:
        for f in sorted(PATCHES.glob("*.py")):
            py_compile.compile(str(f), cfile=os.path.join(d, f.name + "c"), doraise=True)


def test_shell_scripts_parse():
    for f in sorted((RECIPE / "scripts").glob("*.sh")):
        subprocess.run(["bash", "-n", str(f)], check=True)


def test_launcher_has_no_duplicate_flags():
    text = (RECIPE / "scripts" / "run-tp4-hotsplit.sh").read_text()
    for flag in ("--offload-backend", "--cpu-offload-gb", "--tensor-parallel-size", "--moe-backend"):
        assert text.count(flag + " ") == 1, flag
    assert "SYS_NICE" in text and "PYTHONPATH=/w" in text


# ---------------------------------------------------------------- swap mechanics on CPU tensors (needs torch)
def test_dynamic_commit_cpu():
    try:
        import torch
    except ImportError:
        return _skip("torch not installed")
    os.environ["HOTSPLIT_DEVICE"] = "cpu"
    os.environ["HOTSPLIT_DYN_LOG"] = "0"
    import hotsplit as hs

    E, S = 8, 3
    c13 = torch.arange(E * 4, dtype=torch.float32).reshape(E, 4)
    c2 = c13 * 10
    s13 = torch.arange(E, dtype=torch.float32).reshape(E, 1)
    s2 = s13 * 100
    hot = [1, 4, 6]
    h = torch.tensor(hot)
    dyn = hs._Dyn([5], E)
    dyn.emap_hot[0, h] = torch.arange(S, dtype=torch.int32)
    dyn.emap_cold[0, h] = -1
    dl = hs._DynLayer(5, hot, c13[h].clone(), c2[h].clone(), s13[h].clone(), s2[h].clone(), c13, c2, s13, s2)
    dyn.layers[5] = dl
    dyn.commit([(dl, [(1, 4, 7), (2, 6, 0)])])
    assert dl.hot == [1, 7, 0]
    assert torch.equal(dl.hw13[1], c13[7]) and torch.equal(dl.hw2[2], c2[0])
    assert torch.equal(dl.hs13[1], s13[7]) and torch.equal(dl.hs2[2], s2[0])
    assert dyn.emap_hot[0].tolist() == [2, 0, -1, -1, -1, -1, -1, 1]
    assert dyn.emap_cold[0].tolist() == [-1, -1, 2, 3, 4, 5, 6, -1]


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if failed else 0)


# ---------------------------------------------------------------- commit rollback on injected error
def test_dynamic_commit_rollback_on_error():
    """Inject a failure mid-copy and verify the original weights AND both maps are restored
    (review item 1: the update must be all-or-nothing, not half-applied)."""
    try:
        import torch
    except ImportError:
        return _skip("torch not installed")
    os.environ["HOTSPLIT_DEVICE"] = "cpu"
    os.environ["HOTSPLIT_DYN_LOG"] = "0"
    os.environ["HOTSPLIT_DYN_MAX_COMMIT_ERRORS"] = "3"
    import hotsplit as hs

    E, S = 8, 3
    c13 = torch.arange(E * 4, dtype=torch.float32).reshape(E, 4)
    c2 = c13 * 10
    s13 = torch.arange(E, dtype=torch.float32).reshape(E, 1)
    s2 = s13 * 100
    hot = [1, 4, 6]
    h = torch.tensor(hot)
    dyn = hs._Dyn([5], E)
    dyn.emap_hot[0, h] = torch.arange(S, dtype=torch.int32)
    dyn.emap_cold[0, h] = -1
    dl = hs._DynLayer(5, hot, c13[h].clone(), c2[h].clone(), s13[h].clone(), s2[h].clone(), c13, c2, s13, s2)
    dyn.layers[5] = dl

    # snapshots of the pre-commit state
    hw13_before = dl.hw13.clone()
    hw2_before = dl.hw2.clone()
    hs13_before = dl.hs13.clone()
    hs2_before = dl.hs2.clone()
    emap_hot_before = dyn.emap_hot.clone()
    emap_cold_before = dyn.emap_cold.clone()
    hot_list_before = list(dl.hot)

    # inject: the second index_copy_ raises (simulating a CUDA error mid-copy)
    orig_index_copy = torch.Tensor.index_copy_
    calls = {"n": 0}

    def flaky(self, dim, index, source):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected CUDA error")
        return orig_index_copy(self, dim, index, source)

    torch.Tensor.index_copy_ = flaky
    try:
        dyn.commit([(dl, [(1, 4, 7), (2, 6, 0)])])   # must NOT raise: error is rolled back
    finally:
        torch.Tensor.index_copy_ = orig_index_copy

    # everything must be back to the pre-commit state
    assert torch.equal(dl.hw13, hw13_before), "hw13 not restored after rollback"
    assert torch.equal(dl.hw2, hw2_before), "hw2 not restored after rollback"
    assert torch.equal(dl.hs13, hs13_before), "hs13 not restored after rollback"
    assert torch.equal(dl.hs2, hs2_before), "hs2 not restored after rollback"
    assert torch.equal(dyn.emap_hot, emap_hot_before), "emap_hot not restored"
    assert torch.equal(dyn.emap_cold, emap_cold_before), "emap_cold not restored"
    assert dl.hot == hot_list_before, "dl.hot list must be unchanged after rollback"
    assert dyn.updates == 0, "a rolled-back commit must not count as an update"
    assert dyn.commit_errors == 1, "rollback must be counted as a commit error"
    assert dyn.update_state == "IDLE", f"state should return to IDLE after clean rollback, got {dyn.update_state}"


def test_dynamic_commit_fatal_after_repeated_errors():
    """HOTSPLIT_DYN_MAX_COMMIT_ERRORS consecutive failures must abort the run."""
    try:
        import torch
    except ImportError:
        return _skip("torch not installed")
    os.environ["HOTSPLIT_DEVICE"] = "cpu"
    os.environ["HOTSPLIT_DYN_LOG"] = "0"
    os.environ["HOTSPLIT_DYN_MAX_COMMIT_ERRORS"] = "2"
    import hotsplit as hs

    E, S = 8, 3
    c13 = torch.arange(E * 4, dtype=torch.float32).reshape(E, 4)
    c2 = c13 * 10
    s13 = torch.arange(E, dtype=torch.float32).reshape(E, 1)
    s2 = s13 * 100
    hot = [1, 4, 6]
    h = torch.tensor(hot)
    dyn = hs._Dyn([5], E)
    dyn.emap_hot[0, h] = torch.arange(S, dtype=torch.int32)
    dyn.emap_cold[0, h] = -1
    dl = hs._DynLayer(5, hot, c13[h].clone(), c2[h].clone(), s13[h].clone(), s2[h].clone(), c13, c2, s13, s2)
    dyn.layers[5] = dl

    orig_index_copy = torch.Tensor.index_copy_
    torch.Tensor.index_copy_ = lambda self, dim, index, source: (_ for _ in ()).throw(RuntimeError("injected"))
    try:
        dyn.commit([(dl, [(1, 4, 7)])])   # error 1: rolled back, no raise
        try:
            dyn.commit([(dl, [(1, 4, 7)])])   # error 2 = threshold: must RAISE
            raised = False
        except RuntimeError as e:
            raised = "commit failures" in str(e)
    finally:
        torch.Tensor.index_copy_ = orig_index_copy
    assert raised, "repeated commit failures must abort with a RuntimeError"
