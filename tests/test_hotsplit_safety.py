"""Safety-gate tests for hotsplit.py: layer collection and VRAM pre-check.

These lock the two real-defect fixes reviewed against the code:

1. Layer-key collision (MTP vs main decoder). `apply_hotsplit()` used to key layers by the
   bare numeric index from `layers\\.(\\d+)\\.`, so `model.layers.3` and `model.mtp.layers.3`
   both mapped to key 3 and one silently overwrote the other. The fix classifies MTP
   drafter subtrees (`_is_mtp_module`) and keeps them out of the main-decoder plan, and it
   refuses on duplicate indices instead of guessing.

2. VRAM pre-check before the hot-bank copy. The hot copy is made before the originals are
   freed, so the transient peak is hot weights + hot scales; the code must compute that
   byte-accurately (`_row_bytes` on the real tensors) and refuse BEFORE allocating.

Both hotspots are pure Python (no torch, no GPU), so they are exercised here directly by
extracting them from the source file. The wiring inside `apply_hotsplit` is asserted from
the AST, because importing the module needs torch.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
RECIPE = Path(__file__).resolve().parent.parent / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit"
BASE = Path(__file__).resolve().parent.parent / "recipes" / "rtx-pro-6000"
PATCHES = RECIPE / "patches"
SRC = RECIPE / "patches" / "hotsplit.py"
SOURCE = SRC.read_text()


# ------------------------------------------------------------------ helpers extracted from source
def _extract(name: str):
    """Compile one top-level function out of hotsplit.py and return it (no torch needed)."""
    tree = ast.parse(SOURCE)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            mod = ast.Module(body=[node], type_ignores=[])
            code = compile(mod, filename=str(SRC), mode="exec")
            ns: dict = {}
            exec(code, ns)   # noqa: S102 - test-local extraction of repo source
            return ns[name]
    raise AssertionError(f"function {name!r} not found in hotsplit.py")


class _FakeTensor:
    """Minimal tensor stand-in: the helpers are duck-typed on purpose so they test without torch."""

    def __init__(self, shape, itemsize):
        self.shape = tuple(shape)
        self._itemsize = itemsize
        n = 1
        for s in self.shape:
            n *= s
        self._numel = n

    def numel(self):
        return self._numel

    def element_size(self):
        return self._itemsize


# ------------------------------------------------------------------ fix 1: MTP classification
def test_is_mtp_module_detects_drafter_subtrees():
    is_mtp = _extract("_is_mtp_module")
    assert is_mtp("model.mtp.layers.0") is True
    assert is_mtp("model.mtp.layers.3.mlp.experts") is True
    assert is_mtp("mtp.layers.0") is True
    assert is_mtp("draft.mtp.layers.2") is True


def test_is_mtp_module_keeps_main_decoder():
    is_mtp = _extract("_is_mtp_module")
    assert is_mtp("model.layers.0") is False
    assert is_mtp("model.layers.37.mlp.experts") is False
    assert is_mtp("layers.3") is False
    # a leaf named mtp is not a subtree
    assert is_mtp("model.layers.0.mtp") is False


def test_main_and_mtp_layer_do_not_collide():
    """The two paths must be classified differently: the bare index is the same, the plan is not."""
    is_mtp = _extract("_is_mtp_module")
    main = "model.layers.3"
    mtp = "model.mtp.layers.3"
    assert is_mtp(main) != is_mtp(mtp)


def test_collection_uses_mtp_guard_and_refuses_duplicates():
    """The collection loop must classify by `_is_mtp_module`, track paths, and raise on
    duplicate indices rather than overwrite silently."""
    tree = ast.parse(SOURCE)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_hotsplit")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert "_is_mtp_module" in names, "collection must classify with _is_mtp_module"
    # duplicate-index refusal: a raise carrying both paths
    raises = [n for n in ast.walk(fn) if isinstance(n, ast.Raise)]
    assert any(isinstance(r.exc, ast.Call) for r in raises), "duplicate indices must raise"
    src_seg = ast.get_source_segment(SOURCE, fn) or ""
    assert "duplicate layer index" in src_seg
    # and the histogram/model consistency gate before weights move
    assert "histogram/model layer-index mismatch" in src_seg, "mismatch must raise, not warn"
    assert "_hist_only" in src_seg and "_model_only" in src_seg
    assert src_seg.find("histogram/model layer-index mismatch") < src_seg.find("_split_layer_static(")
    # the plan must be keyed while the full path is remembered
    assert "_seen_paths" in src_seg


# ------------------------------------------------------------------ fix 2: VRAM pre-check
def test_row_bytes_measures_real_tensors():
    row_bytes = _extract("_row_bytes")
    # 4 experts x 3 x fp4-ish 2-byte cells, 3 rows requested
    t = _FakeTensor((4, 3), itemsize=2)
    assert row_bytes(t, 3) == 3 * 3 * 2
    assert row_bytes(t, 0) == 0
    assert row_bytes(None, 5) == 0
    empty = _FakeTensor((0, 3), itemsize=2)
    assert row_bytes(empty, 5) == 0
    # scales: 1 per row
    s = _FakeTensor((4, 1), itemsize=4)
    assert row_bytes(s, 2) == 2 * 4


def test_row_bytes_is_duck_typed():
    row_bytes = _extract("_row_bytes")
    assert row_bytes("not a tensor", 3) == 0
    assert row_bytes(object(), 3) == 0


def test_apply_hotsplit_prechecks_vram_before_split():
    """The hot-copy size (weights + both scale tensors) must be computed and compared against
    free VRAM BEFORE any `_split_layer_*` call for that layer, and a failure must refuse."""
    tree = ast.parse(SOURCE)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_hotsplit")
    src_seg = ast.get_source_segment(SOURCE, fn) or ""
    i_pre = src_seg.find("VRAM pre-check")
    i_static = src_seg.find("_split_layer_static(")
    i_dynamic = src_seg.find("_split_layer_dynamic(")
    i_split = min(x for x in (i_static, i_dynamic) if x != -1)
    assert i_pre != -1 and i_pre < i_split, "pre-check must run before the layer split"
    # both scale tensors enter the byte estimate
    assert "_row_bytes" in src_seg
    assert "w13_weight_scale" in src_seg and "w2_weight_scale" in src_seg
    # a failed pre-check must abort the whole run (RuntimeError)
    assert "VRAM pre-check FAILED" in src_seg
    assert src_seg.find("VRAM pre-check FAILED") < i_split
    raise_stmts = [n for n in ast.walk(fn) if isinstance(n, ast.Raise)]
    assert any("VRAM pre-check FAILED" in (ast.get_source_segment(SOURCE, r) or "") for r in raise_stmts), "pre-check failure must raise"


def test_per_layer_telemetry_and_release_check_wired():
    tree = ast.parse(SOURCE)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_hotsplit")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert "_mem_peak_start" in names and "_mem_peak_report" in names
    for split in ("_split_layer_static", "_split_layer_dynamic"):
        sfn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == split)
        called = {n.func.id for n in ast.walk(sfn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "_weights_release_check" in called, f"{split} must verify the originals were freed"


def test_release_check_compares_shapes_to_measured_delta():
    tree = ast.parse(SOURCE)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_weights_release_check")
    seg = ast.get_source_segment(SOURCE, fn) or ""
    assert "expected_release" in seg
    assert "free_after - free_before" in seg
    assert "short" in seg


def test_mem_peak_includes_baseline_increment():
    start = _extract("_mem_peak_start")
    state = start()
    assert ("base_alloc" in state) or (state.get("ok") is False)
    tree = ast.parse(SOURCE)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_mem_peak_report")
    seg = ast.get_source_segment(SOURCE, fn) or ""
    assert "base_alloc" in seg
    assert "peak_alloc - base_alloc" in seg


def test_vram_precheck_refuses_insufficient_reserve():
    """The pre-check must refuse BEFORE any weight is touched when the transient peak
    would exceed free VRAM minus the reserve (CUDA graphs + workspace + KV headroom)."""
    precheck = _extract("_vram_precheck")
    # 5 GiB hot copy, 10 GiB free, 2 GiB reserve -> fits (10-5=5 >= 2)
    ok, reserve = precheck(5 * 2**30, 10 * 2**30, 2.0)
    assert ok and reserve == 2 * 2**30
    # 5 GiB hot copy, 6 GiB free, 2 GiB reserve -> does NOT fit (6-5=1 < 2)
    ok, reserve = precheck(5 * 2**30, 6 * 2**30, 2.0)
    assert not ok
    # floor applies even if reserve_gib is 0
    ok, reserve = precheck(1 * 2**30, 2 * 2**30, 0.0)
    assert not ok and reserve == 2 * 2**30
    # validation
    with pytest.raises(ValueError, match="hot_bytes"):
        precheck(-1, 10, 1.0)
    with pytest.raises(ValueError, match="reserve_gib"):
        precheck(1, 10, -1.0)


def test_vram_precheck_refusal_happens_before_any_split():
    """AST check: the pre-check call site must come before any _split_layer_* call."""
    tree = ast.parse(SOURCE)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_hotsplit")
    seg = ast.get_source_segment(SOURCE, fn) or ""
    i_pre = seg.find("_vram_precheck")
    i_split = min(x for x in (seg.find("_split_layer_static("), seg.find("_split_layer_dynamic(")) if x != -1)
    assert i_pre != -1 and i_pre < i_split, "VRAM pre-check must run before any layer split"
    # the reserve knob must be read from the environment
    assert "HOTSPLIT_VRAM_RESERVE_GIB" in seg


def test_sitecustomize_hook_statuses():
    """sitecustomize.py must emit HOOK STATUS ACTIVE/INACTIVE/FAILED and support strict mode."""
    sc = (PATCHES / "sitecustomize.py").read_text()
    assert "HOOK STATUS" in sc
    assert "ACTIVE" in sc and "INACTIVE" in sc and "FAILED" in sc
    assert "HOTSPLIT_STRICT_HOOK" in sc
    # strict mode must raise, not just warn
    assert "SystemExit" in sc or "raise" in sc


def test_numa_bind_strict_and_honest_caveat():
    """numa_bind.py must expose strict mode and document that preferred is advisory."""
    nb = (PATCHES / "numa_bind.py").read_text()
    assert "HOTSPLIT_NUMA_STRICT" in nb
    assert "advisory" in nb.lower()
    assert "HOTSPLIT_NUMA_POLICY" in nb
    # bind_for_device returns (ok, detail) for the caller to enforce strict mode
    assert "return ok," in nb or "return True," in nb or "return False," in nb


def test_launcher_autodetects_site_and_probes_api():
    """run-tp4-hotsplit.sh must auto-detect SITE inside the image and probe patch compatibility."""
    launcher = (BASE / "mimo-v2.6-pro-hotsplit" / "scripts" / "run-tp4-hotsplit.sh").read_text()
    assert "SITE" in launcher
    assert "find_spec" in launcher, "must probe the image's own model files before mounting patches"
    assert "load_weights" in launcher or "process_weights_after_loading" in launcher
    assert "SHA-256" in launcher or "sha256" in launcher.lower()


def test_dynamic_hook_decision_refuses_silent_fallback():
    """HOTSPLIT_DYNAMIC=1 must NOT silently degrade to static: the run refuses unless
    HOTSPLIT_ALLOW_STATIC_FALLBACK=1, and every path logs dynamic_hook=ACTIVE/INACTIVE."""
    decision = _extract("_dynamic_hook_decision")
    # dynamic requested, hook failed, no explicit fallback -> error (refuse)
    mode, err = decision(False, True, False)
    assert mode == "static"
    assert err is not None and "ALLOW_STATIC_FALLBACK" in err
    # dynamic requested, hook failed, explicit fallback -> static, no error
    mode, err = decision(False, True, True)
    assert mode == "static" and err is None
    # dynamic requested, hook ok -> dynamic
    mode, err = decision(True, True, False)
    assert mode == "dynamic" and err is None
    # static requested -> static regardless
    mode, err = decision(False, False, False)
    assert mode == "static" and err is None
    # the call site must log the per-rank status and use the decision
    tree = ast.parse(SOURCE)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_hotsplit")
    seg = ast.get_source_segment(SOURCE, fn) or ""
    assert "dynamic_hook=" in seg
    assert "_dynamic_hook_decision" in seg
    assert "HOTSPLIT_ALLOW_STATIC_FALLBACK" in seg


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))