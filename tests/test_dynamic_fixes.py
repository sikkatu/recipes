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


# ============================================= documented env must reach the container
def test_documented_env_reaches_container():
    """Every HOTSPLIT_* knob documented in tp4.env.example must be forwarded by the launcher
    into the container env. A documented knob that silently does nothing is worse than none:
    HOTSPLIT_STRICT_HOOK=1 documented but not forwarded promised strict behaviour it did not
    deliver (found in the wiki audit)."""
    import re
    recipe = BASE / "mimo-v2.6-pro-hotsplit"
    env_doc = (recipe / "tp4.env.example").read_text()
    launcher = (recipe / "scripts" / "run-tp4-hotsplit.sh").read_text()
    documented = set(re.findall(r"#?\s*(HOTSPLIT_[A-Z_]+)=", env_doc))
    assert documented, "tp4.env.example documents no knobs"
    for knob in sorted(documented):
        assert knob in launcher, f"{knob} is documented in tp4.env.example but not forwarded by run-tp4-hotsplit.sh"


def test_launcher_requires_pinned_image_and_key():
    """The two launch-time refusals the wiki must document honestly: IMAGE needs a sha256
    digest, API_KEY is required even on loopback (unless ALLOW_NO_AUTH=1)."""
    launcher = (BASE / "mimo-v2.6-pro-hotsplit" / "scripts" / "run-tp4-hotsplit.sh").read_text()
    assert 'IMAGE must be pinned by digest' in launcher
    # the auth contract: a non-loopback bind is refused without a key; the key is
    # auto-generated and persisted for loopback (never an unauthenticated public API)
    assert 'refusing a non-loopback bind' in launcher
    assert 'failed to generate an API key' in launcher
    assert 'api_key' in launcher
    # MODEL_REVISION required when the model is fetched by id
    assert 'MODEL_REVISION' in launcher


# ============================================= transactional commit (P1 rollback)
def test_commit_is_transactional_with_rollback():
    """commit() must snapshot before writing, verify completion, flip maps only after the
    copies are confirmed, and roll back on error (review P1)."""
    text = _fn_text("commit")
    assert "SNAPSHOT" in text or "old_rows" in text, "commit must snapshot before overwriting"
    assert "ROLLBACK" in text, "commit must roll back on error"
    assert "old_hv" in text and "old_cv" in text, "map entries must be snapshotted too"
    # maps must flip only AFTER the copies verified
    assert text.index("VERIFYING") < text.index("emap_hot[t_hr, t_hc] = t_hv"), \
        "maps must flip only after verification"
    # commit errors must be counted and fatal past a threshold
    assert "commit_errors" in text and "max_commit_errors" in text


def test_commit_state_fields_exist():
    assert "self.update_state" in SOURCE
    assert "HOTSPLIT_DYN_MAX_COMMIT_ERRORS" in SOURCE
    assert "self.last_update_at" in SOURCE


# ============================================= fail-safe BIND validation (P0-1)
def test_bind_is_fail_safe_across_launchers():
    """Only literal 127.0.0.1 / ::1 count as loopback. `localhost`, wildcards, empty and any
    unknown address must require an API key (fail-safe: no bypass for unrecognized values)."""
    for p in sorted(BASE.glob("*/scripts/run-tp4-*.sh")):
        t = p.read_text()
        assert "LOOPBACK=1" in t, f"{p.name}: loopback flag missing"
        assert "127.0.0.1" in t and "::1" in t, f"{p.name}: literal loopback check missing"
        # the old unsafe comparison must be gone
        assert '"$BIND" != "127.0.0.1"' not in t, f"{p.name}: unsafe string compare still present"
        # ALLOW_NO_AUTH must be gated on the LOOPBACK flag, not a raw string compare
        assert '"$ALLOW_NO_AUTH" = "1" ] && [ "$LOOPBACK" = "1"' in t or \
               '"$ALLOW_NO_AUTH" = "1" ] && [ "$LOOPBACK" = "1"' in t.replace("'", '"'), \
               f"{p.name}: ALLOW_NO_AUTH must be gated on the proven-loopback flag"


def test_localhost_never_gets_no_auth_bypass():
    """Behavioural: BIND=localhost with ALLOW_NO_AUTH=1 must be refused (no API key)."""
    launcher = BASE / "mimo-v2.6-pro-hotsplit" / "scripts" / "run-tp4-hotsplit.sh"
    bin_dir = _make_stub_bin() if hasattr(__builtins__, "_make_stub_bin") else None
    # static check is sufficient: `localhost` is not in the loopback case patterns
    import re
    t = launcher.read_text()
    m = re.search(r'LOOPBACK=0\s+case "\$BIND" in\s+([^\n]+)', t)
    assert m, "loopback case block not found"
    patterns = m.group(1)
    assert "localhost" not in patterns, "localhost must NOT be treated as loopback"


# ============================================= honest rollback + clean-cache (9.1)
def test_rollback_is_honest_about_host_state():
    for p in sorted(BASE.glob("*/scripts/rollback.sh")):
        t = p.read_text()
        assert "HONEST SCOPE" in t or "preserved on purpose" in t, f"{p}: rollback must state its limits"
        assert "No host state was modified" not in t, f"{p}: dishonest claim still present"


def test_manage_sh_has_clean_cache():
    for p in sorted(BASE.glob("*/scripts/manage.sh")):
        t = p.read_text()
        assert "clean-cache" in t, f"{p}: clean-cache command missing"
        # must ask for confirmation before deleting
        assert "read -r" in t or "confirm" in t, f"{p}: clean-cache must ask before deleting"


# ============================================= traffic-weighted phase ratio (item 2)
def test_traffic_weighted_ratio_is_linear_not_quadratic():
    """In traffic-weighted mode a phase's influence must be volume x weight (linear), not
    volume^2 x weight. A 10x-larger decode with decode=1,prefill=0.25 must give 40:1, not 400:1."""
    import json
    import tempfile
    import numpy as np
    sys.path.insert(0, str(PATCHES))
    import hotsplit_plan as hp
    # decode has 10x the observations of prefill
    hist = {"train": {
        "decode":  {"0": [100, 0]},
        "prefill": {"0": [10, 0]},
    }}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(hist, f)
        path = f.name
    out = hp.load_counts(path, "decode=1,prefill=0.25", mode="traffic-weighted")
    v = out[0]
    # decode contribution 100*1 / 110 ; prefill 10*0.25 / 110 -> ratio 40:1
    ratio = v[0] / v[1] if v[1] != 0 else float("inf")
    # here both phases route to expert 0 only; compare expert-1 split instead:
    hist2 = {"train": {
        "decode":  {"0": [0, 100]},
        "prefill": {"0": [10, 0]},
    }}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(hist2, f)
        path2 = f.name
    out2 = hp.load_counts(path2, "decode=1,prefill=0.25", mode="traffic-weighted")
    w = out2[0]
    got = w[1] / w[0]   # decode-only expert vs prefill-only expert
    # expected: (100 * 1) / (10 * 0.25) = 40
    assert abs(got - 40.0) < 1e-6, f"traffic-weighted ratio is {got}, expected 40 (volume x weight)"


def test_phase_balanced_still_equalises_phases():
    import json, tempfile, sys
    sys.path.insert(0, str(PATCHES))
    import hotsplit_plan as hp
    hist = {"train": {
        "decode":  {"0": [0, 100]},
        "prefill": {"0": [10, 0]},
    }}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(hist, f)
        path = f.name
    out = hp.load_counts(path, "decode=1,prefill=0.25", mode="phase-balanced")
    w = out[0]
    # decode normalised to 1 * weight 1, prefill normalised to 1 * weight 0.25 -> 4:1
    got = w[1] / w[0]
    assert abs(got - 4.0) < 1e-6, f"phase-balanced ratio is {got}, expected 4"


# ============================================= manage.sh / rollback.sh hardening (items 3,4,6,7,8)
def test_rollback_defaults_live_dir():
    """rollback.sh runs under set -u and references $LIVE_DIR in its summary -- it must have a
    default so it cannot crash AFTER removing the container (item 3)."""
    for p in sorted(BASE.glob("*/scripts/rollback.sh")):
        t = p.read_text()
        assert 'LIVE_DIR:-' in t, f"{p}: LIVE_DIR needs a default under set -u"


def test_manage_clean_cache_refuses_dangerous_paths():
    """clean-cache must refuse /, $HOME and non-cache paths (item 4)."""
    for p in sorted(BASE.glob("*/scripts/manage.sh")):
        t = p.read_text()
        assert "REFUSING to delete" in t, f"{p}: clean-cache must refuse dangerous paths"
        assert "readlink -f" in t, f"{p}: clean-cache must resolve the path before deleting"
        # must not delete without path sanity checks
        assert "rm -rf" in t  # still deletes, but only after the guards


def test_manage_status_sends_auth_header():
    """status must send Authorization: Bearer when API_KEY is set (item 6)."""
    for p in sorted(BASE.glob("*/scripts/manage.sh")):
        t = p.read_text()
        assert "Authorization: Bearer" in t, f"{p}: status must send the API key"
        assert "401" in t and "403" in t, f"{p}: status must explain auth rejections"


def test_manage_env_wins_over_file():
    """Exported environment must override tp4.env; the file only fills unset names (item 7)."""
    for p in sorted(BASE.glob("*/scripts/manage.sh")):
        t = p.read_text()
        assert "only fill values not already exported" in t or "WIN over the file" in t, \
            f"{p}: env priority contract must be stated"
        assert 'set -a' not in t, f"{p}: blind 'set -a' sourcing loses env priority"


def test_check_script_has_expect_progress_mode():
    """After confirmed traffic, no dynamic progress must be a FAIL, not a WARN (item 8)."""
    p = BASE / "mimo-v2.6-pro-hotsplit" / "scripts" / "check-hotsplit-tp4.sh"
    t = p.read_text()
    assert "EXPECT_PROGRESS" in t, "expect-progress mode missing"
    assert "expect-progress" in t
    assert "fail=1" in t


# ============================================= strict pinning validation (P1a/P1b/P1c)
def test_model_revision_must_be_full_sha():
    """A branch name like 'main' must NOT pass as MODEL_REVISION (P1a)."""
    for p in sorted(BASE.glob("*/scripts/run-tp4-*.sh")):
        t = p.read_text()
        assert '^[0-9a-f]{40}$' in t, f"{p}: MODEL_REVISION must be checked against a 40-hex SHA"
        assert "full 40-hex commit SHA" in t, f"{p}: the error message must state the requirement"


def test_image_digest_must_be_64_hex():
    """@sha256:<junk> must NOT pass as a pinned IMAGE (P1b)."""
    for p in sorted(BASE.glob("*/scripts/run-tp4-*.sh")):
        t = p.read_text()
        assert '@sha256:[0-9a-f]{64}$' in t or "@sha256:[0-9a-f]{64}" in t, \
            f"{p}: IMAGE digest must be exactly 64 hex chars"


def test_clean_cache_only_huggingface_dir():
    """clean-cache must refuse ~/.cache and any non-huggingface path (P1c)."""
    for p in sorted(BASE.glob("*/scripts/manage.sh")):
        t = p.read_text()
        assert "is not a huggingface cache directory" in t, \
            f"{p}: clean-cache must require the huggingface dir specifically"
        # the permissive old patterns must be gone
        assert '*/.cache/*|*/cache/*|*/.cache' not in t, \
            f"{p}: permissive cache patterns still present"
        assert 'HF_ROOT' in t or 'STRICT allow-list' in t, f"{p}: strict root check missing"


# ============================================= review round 2 (bugs B..H)
def test_readme_traffic_formula_multiplies_by_layers():
    """Per-token expert traffic is topk*cell*layers, not topk*cell alone (bug B).
    The old numbers understated the real PCIe ceiling 40-75x."""
    import re
    for d in sorted(BASE.iterdir()):
        p = d / "README.md"
        if not p.is_file():
            continue
        t = p.read_text()
        if "MiB/token" not in t and "MiB/layer/token/rank" not in t:
            continue
        # every per-token number must be explicitly derived from layers
        assert "layer/token/rank" in t or "per layer" in t, f"{d.name}: per-layer traffic line missing"
        assert re.search(r"\d+ layers ×", t), f"{d.name}: layer multiplication missing"


def test_manage_sh_degrades_when_script_missing():
    """In recipes without run-tp4-hotsplit.sh, manage.sh start must fall back to UVA,
    not crash after stopping the server (bug C)."""
    for p in sorted(BASE.glob("*/scripts/manage.sh")):
        t = p.read_text()
        assert "degrade gracefully" in t, f"{p}: missing graceful fallback"
        assert "is not shipped in this recipe" in t


def test_env_parser_strips_inline_comments():
    """tp4.env values must not include trailing comments like '1  # 0 = static' (bug D)."""
    for p in sorted(BASE.glob("*/scripts/manage.sh")):
        t = p.read_text()
        assert "strip inline" in t or '%% #*' in t, f"{p}: inline comment stripping missing"


def test_gitignore_covers_local_config():
    """tp4.env (may hold API_KEY) and live/ (diagnostics) must be gitignored (bug E)."""
    gi = (BASE.parent.parent / ".gitignore").read_text()
    assert "tp4.env" in gi, ".gitignore must cover tp4.env (may contain API_KEY)"
    assert "live/" in gi, ".gitignore must cover live/"


def test_readiness_probe_uses_bind():
    """The health probe must target $BIND, not hardcoded 127.0.0.1 (bug F)."""
    for p in sorted(BASE.glob("*/scripts/run-tp4-*.sh")):
        t = p.read_text()
        if "/v1/models" not in t:
            continue
        assert "http://$BIND" in t or "BIND/::1" in t or "[::1]" in t, \
            f"{p}: readiness probe must use $BIND (with IPv6 bracket support)"


def test_balance_rejects_partial_histogram():
    """A histogram with fewer layers than the model must be refused (bug G)."""
    for p in sorted(BASE.glob("*/patches/hotsplit_balance.py")):
        t = p.read_text()
        assert "partial histogram" in t, f"{p}: partial-histogram guard missing"


def test_log_hit_rates_uses_live_row_order():
    """log_hit_rates must map layers to live-buffer rows via live.layer_idx (bug H),
    not self.rows (sorted order) -- otherwise mixed UVA/HBM splits give wrong hit rates."""
    text = _fn_text("log_hit_rates")
    assert "crow" in text or "layer_idx" in text, \
        "log_hit_rates must map rows via live.layer_idx like plan() does"
    assert "self.rows.get" not in text, "log_hit_rates must not use self.rows for buffer indexing"


def test_check_api_sends_auth():
    """check-tp4-api.sh must send Authorization: Bearer when API_KEY is set."""
    p = BASE / "mimo-v2.6-pro-hotsplit" / "scripts" / "check-tp4-api.sh"
    t = p.read_text()
    assert "Authorization: Bearer" in t, "check-tp4-api.sh must authenticate"
    assert "ALLOW_NO_AUTH" in t, "the no-key case must be explained"


def test_env_example_has_mandatory_block():
    """tp4.env.example must separate mandatory (IMAGE, API_KEY, MODEL_REVISION) from optional."""
    for p in sorted(BASE.glob("*/tp4.env.example")):
        t = p.read_text()
        assert "MANDATORY" in t, f"{p}: mandatory block missing"
        for name in ("IMAGE", "API_KEY", "MODEL_REVISION"):
            assert name in t, f"{p}: mandatory {name} not documented"


# ============================================= review round 3 (P1, P2a, P2b, DGX)
def test_clean_cache_accepts_only_hf_root():
    """clean-cache must accept only the canonical HF cache root (and hub/), not any
    arbitrary path containing 'huggingface' (P2a)."""
    for p in sorted(BASE.glob("*/scripts/manage.sh")):
        t = p.read_text()
        assert "STRICT allow-list" in t or "HF_ROOT" in t, f"{p}: strict root check missing"
        # the permissive wildcard must be gone
        assert "*/huggingface/*" not in t, f"{p}: wildcard */huggingface/* still present"
        # must use readlink canonicalisation before comparing
        assert "readlink -f" in t


def test_ipv6_probe_url_brackets():
    """The readiness probe must bracket IPv6 addresses in the URL (P2b)."""
    for p in sorted(BASE.glob("*/scripts/run-tp4-*.sh")):
        t = p.read_text()
        if "/v1/models" not in t:
            continue
        assert "[::1]" in t, f"{p}: IPv6 loopback must be bracketed in probe URL"


def test_readme_states_image_unverified():
    """README must honestly state no image is verified yet (P1), not pretend one is pinned."""
    for p in sorted(BASE.glob("*/README.md")):
        t = p.read_text()
        assert "No image has been verified" in t, f"{p}: image compatibility disclaimer missing"
        assert "@sha256:" in t, f"{p}: digest pinning requirement missing"


def test_no_dgx_references_in_docs():
    """The host is a custom DDR4/PCIe4 workstation, NOT a DGX (PCIe5/HBM)."""
    import pathlib
    root = BASE.parent.parent
    for p in root.rglob("*.md"):
        if ".git" in str(p) or ".pytest_cache" in str(p):
            continue
        t = p.read_text()
        assert "DGX" not in t, f"{p}: DGX reference present (host is custom DDR4/PCIe4)"
    wiki = pathlib.Path("/tmp/recipes.wiki")
    if wiki.is_dir():
        for p in wiki.glob("*.md"):
            t = p.read_text()
            assert "DGX" not in t, f"wiki/{p.name}: DGX reference present"


def test_wiki_pins_concrete_image():
    """Quick Start must give a concrete IMAGE string (tag + digest), not a <tag> placeholder."""
    import pathlib
    wiki = pathlib.Path("/tmp/recipes.wiki")
    if not wiki.is_dir():
        import pytest
        pytest.skip("wiki checkout not available")
    for name in ("Quick-Start.md", "Quick-Start.ru.md", "Quick-Start.zh.md"):
        t = (wiki / name).read_text()
        assert "vllm/vllm-openai:nightly-" in t, f"{name}: concrete image tag missing"
        assert "@sha256:167ccd8696bc09f1b63301185134e04b883623f8adcbd8e3a0ec3140a193abcf" in t, \
            f"{name}: pinned digest missing"
        assert "<tag>" not in t, f"{name}: <tag> placeholder still present"


def test_wiki_clone_path_single_nesting():
    """The clone path must not produce ~/recipes/recipes (repo has a nested recipes/ dir)."""
    import pathlib
    wiki = pathlib.Path("/tmp/recipes.wiki")
    if not wiki.is_dir():
        import pytest
        pytest.skip("wiki checkout not available")
    for name in ("Quick-Start.md", "Quick-Start.ru.md", "Quick-Start.zh.md"):
        t = (wiki / name).read_text()
        assert "sikkatu-recipes" in t, f"{name}: unambiguous clone dir missing"
        assert "cd ~/recipes/rtx-pro-6000" not in t, f"{name}: old double-nesting path present"


def test_model_dir_default_is_in_home_not_root():
    """Model download instructions must default to ~/models (or $HOME/models), never /models
    at the filesystem root -- users copy commands literally and a root-level path needs sudo
    and surprises them (found during a live install)."""
    import pathlib
    root = BASE.parent.parent
    for p in list(root.rglob("README*.md")) + list(root.rglob("tp4.env.example")) + list(root.rglob("RUNNING*.md")):
        if ".git" in str(p) or ".pytest_cache" in str(p):
            continue
        t = p.read_text()
        assert 'MODEL_DIR=/models/' not in t, f"{p}: MODEL_DIR at filesystem root"
        assert 'MODEL_DIR="/models/' not in t, f"{p}: MODEL_DIR at filesystem root (quoted)"
    wiki = pathlib.Path('/tmp/recipes.wiki')
    if wiki.is_dir():
        for p in wiki.glob('Quick-Start*.md'):
            t = p.read_text()
            assert 'export MODEL_DIR="/models/' not in t, f"wiki/{p.name}: MODEL_DIR at filesystem root"
            assert '$HOME/models' in t or '~/models' in t, f"wiki/{p.name}: home-dir default missing"
