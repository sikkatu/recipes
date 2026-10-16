"""Tests for the P0/P1 dynamic-hotsplit review fixes.

P0-1: scheduler parameter validation + fatal repeated planner errors + counters
P0-2: explicit CUDA-stream/event ordering of the weight swaps
P1:   hit-rate telemetry hook
B:    _shard_fp8_qkv_proj row/scale-grid integrity (pure-Python index math)
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent / "recipes" / "rtx-pro-6000"
PATCHES = BASE / "mimo-v2.6-pro-hotsplit" / "patches"
SRC = PATCHES / "hotsplit.py"
SOURCE = SRC.read_text()
sys.path.insert(0, str(PATCHES))


def _extract(name: str):
    tree = ast.parse(SOURCE)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            ns: dict = {"os": os}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRC), "exec"), ns)  # noqa: S102
            return ns[name]
    raise AssertionError(f"{name!r} not found")


def _fn_text(name: str) -> str:
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"function {name!r} not found")


# ============================================= P0-1: parameter validation
@pytest.mark.parametrize("name,value,ok", [
    ("HOTSPLIT_DYN_STEPS", "2000", True),
    ("HOTSPLIT_DYN_STEPS", "1", True),
    ("HOTSPLIT_DYN_STEPS", "0", False),  # the ZeroDivisionError case
    ("HOTSPLIT_DYN_STEPS", "-5", False),
    ("HOTSPLIT_DYN_STEPS", "abc", False),
    ("HOTSPLIT_DYN_STEPS", "1.5", False),
    ("HOTSPLIT_DYN_MAX_PLAN_ERRORS", "3", True),
    ("HOTSPLIT_DYN_MAX_PLAN_ERRORS", "0", False),
    ("HOTSPLIT_DYN_MAX_PLAN_ERRORS", "-1", False),
])
def test_dyn_int_validation(name, value, ok):
    _dyn_int = _extract("_dyn_int")
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        if ok:
            v = _dyn_int(name, "2000", lo=1)
            assert isinstance(v, int) and v >= 1
        else:
            with pytest.raises(ValueError):
                _dyn_int(name, "2000", lo=1)
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


@pytest.mark.parametrize("name,value,ok", [
    ("HOTSPLIT_DYN_DECAY", "0.5", True),
    ("HOTSPLIT_DYN_DECAY", "0.0", True),
    ("HOTSPLIT_DYN_DECAY", "1.0", True),
    ("HOTSPLIT_DYN_DECAY", "1.5", False),
    ("HOTSPLIT_DYN_DECAY", "-0.1", False),
    ("HOTSPLIT_DYN_MARGIN", "1.25", True),
    ("HOTSPLIT_DYN_MARGIN", "0.5", False),  # margin must be >= 1
    ("HOTSPLIT_DYN_MARGIN", "nan", False),
])
def test_dyn_float_validation(name, value, ok):
    _dyn_float = _extract("_dyn_float")
    old = os.environ.get(name)
    os.environ[name] = value
    lo = 1.0 if name == "HOTSPLIT_DYN_MARGIN" else 0.0
    hi = 1.0 if name == "HOTSPLIT_DYN_DECAY" else None
    try:
        if ok:
            _dyn_float(name, "0.5", lo=lo, hi=hi)
        else:
            with pytest.raises(ValueError):
                _dyn_float(name, "0.5", lo=lo, hi=hi)
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def test_every_zero_is_rejected_at_startup():
    """HOTSPLIT_DYN_STEPS=0 must raise at _Dyn construction, never reach `steps % every`."""
    pytest.importorskip("torch")
    import hotsplit as hs  # noqa: E402
    old = os.environ.get("HOTSPLIT_DYN_STEPS")
    os.environ["HOTSPLIT_DYN_STEPS"] = "0"
    os.environ["HOTSPLIT_DEVICE"] = "cpu"
    try:
        with pytest.raises(ValueError):
            hs._Dyn([0], 8)
    finally:
        if old is None:
            os.environ.pop("HOTSPLIT_DYN_STEPS", None)
        else:
            os.environ["HOTSPLIT_DYN_STEPS"] = old


# ============================================= P0-1: fatal repeated planner errors
def test_hook_source_makes_repeated_errors_fatal():
    text = _fn_text("wrapped")
    assert "plan_errors" in text, "planner errors must be counted"
    assert "RuntimeError" in text, "repeat offenders must be fatal"
    assert "max_plan_errors" in text, "threshold must come from the validated knob"
    assert ">=" in text or ">" in text, "fatal raise must be gated on a threshold"


def test_plan_counters_exist():
    assert "self.plan_ok" in SOURCE and "self.plan_errors" in SOURCE
    assert "self.max_plan_errors" in SOURCE
    assert "self.plan_errors = 0" in SOURCE  # good pass resets consecutive errors


# ============================================= P0-2: explicit stream ordering
def test_commit_uses_explicit_stream_and_event():
    text = _fn_text("commit")
    assert "torch.cuda.Stream" in text, "swaps need a dedicated stream"
    assert "wait_stream" in text, "swap stream must wait on the current stream first"
    assert "Event" in text and "wait_event" in text, \
        "readers must be ordered after the copies through a CUDA event"
    assert "P0-2" in SOURCE, "the fix must be marked so reviewers can find the reasoning"


def test_null_ctx_is_a_noop_context_manager():
    ctx = _extract("_null_ctx")()
    with ctx as c:
        assert c is ctx
    assert ctx.__exit__(None, None, None) in (False, None)


# ============================================= P1: hit-rate telemetry
def test_hit_rate_telemetry_present():
    text = _fn_text("log_hit_rates")
    assert "hot" in text and "cold" in text
    assert "logger.info" in text
    assert "except" in text, "telemetry must never take the lane down"
    assert "log_hit_rates" in _fn_text("commit"), "commit must call the telemetry"


# ============================================= copies are identical
def test_hotsplit_py_identical_across_recipes():
    ref = (PATCHES / "hotsplit.py").read_bytes()
    for d in sorted(BASE.iterdir()):
        p = d / "patches" / "hotsplit.py"
        if p.is_file():
            assert p.read_bytes() == ref, f"{d.name}/patches/hotsplit.py drifted from mimo's"


# ============================================= B: fp8 qkv shard integrity
def _qkv_row_index(num_heads, num_kv_heads, head_dim, v_head_dim, tp_rank, tp_size, ckpt_tp):
    """Reimplementation of the row-selection math in _shard_fp8_qkv_proj (pure Python).
    If the patch's indexing is correct, this must produce exactly the same rows -- and the
    two invariants (exactly-once coverage, valid scale-grid mapping) hold."""
    q_per_chunk = (num_heads // ckpt_tp) * head_dim
    k_per_chunk = (num_kv_heads // ckpt_tp) * head_dim
    v_per_chunk = (num_kv_heads // ckpt_tp) * v_head_dim
    rows_per_chunk = q_per_chunk + k_per_chunk + v_per_chunk

    if tp_size <= num_kv_heads:
        kv_head_ids = list(range(tp_rank * (num_kv_heads // tp_size),
                                (tp_rank + 1) * (num_kv_heads // tp_size)))
    else:
        kv_head_ids = [tp_rank // (tp_size // num_kv_heads)]
    q_head_ids = list(range(tp_rank * (num_heads // tp_size),
                            (tp_rank + 1) * (num_heads // tp_size)))

    q_heads_per_chunk = num_heads // ckpt_tp
    kv_heads_per_chunk = num_kv_heads // ckpt_tp
    index = []
    for head in q_head_ids:
        chunk = head // q_heads_per_chunk
        index += [chunk * rows_per_chunk + (head % q_heads_per_chunk) * head_dim + r
                  for r in range(head_dim)]
    for head in kv_head_ids:
        chunk = head // kv_heads_per_chunk
        index += [chunk * rows_per_chunk + q_per_chunk
                  + (head % kv_heads_per_chunk) * head_dim + r for r in range(head_dim)]
    for head in kv_head_ids:
        chunk = head // kv_heads_per_chunk
        index += [chunk * rows_per_chunk + q_per_chunk + k_per_chunk
                  + (head % kv_heads_per_chunk) * v_head_dim + r for r in range(v_head_dim)]
    return index, rows_per_chunk


@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim,v_head_dim,tp_size,ckpt_tp", [
    (16, 4, 128, 128, 4, 4),  # tp == ckpt (fast path geometry)
    (16, 4, 128, 128, 4, 2),  # gather path
    (16, 4, 128, 128, 2, 4),  # gather path, fewer ranks than chunks
    (16, 8, 128, 128, 4, 8),  # KV replication
    (8, 2, 64, 64, 2, 2),
])
def test_qkv_row_selection_is_exact_permutation(num_heads, num_kv_heads, head_dim,
                                                v_head_dim, tp_size, ckpt_tp):
    """Every rank's row selection must be an exactly-once, in-range set of source rows."""
    all_selected = []
    for tp_rank in range(tp_size):
        index, rows_per_chunk = _qkv_row_index(
            num_heads, num_kv_heads, head_dim, v_head_dim, tp_rank, tp_size, ckpt_tp)
        total_rows = rows_per_chunk * ckpt_tp
        expected = (num_heads // tp_size) * head_dim \
            + (num_kv_heads // tp_size) * head_dim \
            + (num_kv_heads // tp_size) * v_head_dim
        assert len(index) == expected, f"rank {tp_rank}: {len(index)} rows, expected {expected}"
        assert all(0 <= i < total_rows for i in index), f"rank {tp_rank}: index out of range"
        assert len(set(index)) == len(index), f"rank {tp_rank}: duplicate rows"
        all_selected.append(index)

    if tp_size == ckpt_tp and num_heads % tp_size == 0 and num_kv_heads % tp_size == 0:
        union = sorted(i for idx in all_selected for i in idx)
        total = max(union) + 1
        assert len(union) == len(set(union)) == total, \
            "rank selections must tile the checkpoint exactly when tp == ckpt_tp"


def test_qkv_scale_grid_mapping_is_in_range():
    """The scale_index must map every selected weight row onto a valid scale row -- for
    both the per-chunk and the continuous-grid scale layouts."""
    num_heads, num_kv_heads, head_dim, v_head_dim = 16, 4, 128, 128
    tp_size, ckpt_tp, block = 4, 2, 128
    index, rows_per_chunk = _qkv_row_index(
        num_heads, num_kv_heads, head_dim, v_head_dim, 0, tp_size, ckpt_tp)
    total_rows = rows_per_chunk * ckpt_tp
    chunk_scale_rows = -(-rows_per_chunk // block)  # cdiv

    def per_chunk(rows):
        return [(r // rows_per_chunk) * chunk_scale_rows + (r % rows_per_chunk) // block
                for r in rows]

    def continuous(rows):
        return [r // block for r in rows]

    for mapper, n_scale_rows in (
        (per_chunk, ckpt_tp * chunk_scale_rows),
        (continuous, -(-total_rows // block)),
    ):
        si = mapper(index)
        assert len(si) == len(index)
        assert all(0 <= s < n_scale_rows for s in si), \
            f"scale index out of range for {n_scale_rows} scale rows"


def _extract_from(path, names):
    tree = ast.parse(path.read_text())
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<x>", "exec"), ns)  # noqa: S102
    return ns


def test_shard_fp8_qkv_rows_with_torch():
    """Run the real _shard_fp8_qkv_proj against a stub quantizer and check the rows it
    gathers equal the pure-Python selection exactly."""
    torch = pytest.importorskip("torch")
    ns = _extract_from(PATCHES / "mimo_v2_hotsplit.py",
                       {"_shard_fp8_qkv_proj", "_requantize_fp8"})
    ns["torch"] = torch
    ns["cdiv"] = lambda a, b: -(-a // b)
    ns["GroupShape"] = lambda a, b: (a, b)
    seen = {}
    block = 128

    def fake_quantize(grouped, gs, dtype, compute_dtype=None):
        seen["rows"] = grouped.shape[0]
        seen["vals"] = grouped.clone()
        return (torch.zeros(grouped.shape[0], grouped.shape[1]),
                torch.zeros(-(-grouped.shape[0] // block), grouped.shape[1]))

    ns["scaled_quantize"] = fake_quantize

    num_heads, num_kv_heads, head_dim, v_head_dim = 16, 4, 128, 128
    tp_size, ckpt_tp = 4, 2
    q_per_chunk = (num_heads // ckpt_tp) * head_dim
    k_per_chunk = (num_kv_heads // ckpt_tp) * head_dim
    v_per_chunk = (num_kv_heads // ckpt_tp) * v_head_dim
    rows_per_chunk = q_per_chunk + k_per_chunk + v_per_chunk
    total_rows = rows_per_chunk * ckpt_tp
    chunk_scale_rows = -(-rows_per_chunk // block)

    w_full = torch.arange(total_rows, dtype=torch.float32).unsqueeze(1) * 3.0
    s_full = torch.ones(ckpt_tp * chunk_scale_rows, 1, dtype=torch.float32)

    for tp_rank in range(tp_size):
        seen.clear()
        ns["_shard_fp8_qkv_proj"](w_full, s_full, num_heads, num_kv_heads,
                                  head_dim, v_head_dim, tp_rank, tp_size, ckpt_tp, block)
        want_index, _ = _qkv_row_index(num_heads, num_kv_heads, head_dim, v_head_dim,
                                       tp_rank, tp_size, ckpt_tp)
        expect = torch.tensor([[w_full[i, 0].item() * 1.0] for i in want_index])
        got = seen["vals"][:, :1]
        assert torch.allclose(got, expect), f"rank {tp_rank}: gathered rows differ from selection"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
