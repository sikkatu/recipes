"""Guard against the return of fork traces and foreign-hardware references (audit block 1).

Scans the CURRENT TREE only (not git history). Forbidden: DGX Station / GB300 references and
personal/foreign handles. Not a false-positive trap: plain technical words (GPU, CUDA, MoE,
Grace-independent terms) are fine; only the specific foreign-hardware product names and the
old fork's identity are rejected. A case-insensitive word-boundary match on the forbidden
terms keeps "grace period" or generic text from triggering.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# exact foreign-hardware product names and the old fork's identity
FORBIDDEN = re.compile(
    r"(?i)(?<![A-Za-z0-9_-])(DGX\s*Station|GB300|J-M-Recipes|James Meadlock)(?![A-Za-z0-9_-])")

SKIP_SUFFIX = {".png", ".jpg", ".jpeg", ".svg", ".pt", ".safetensors", ".gz", ".zip", ".pyc", ".bundle"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache"}
# this guard file itself contains the forbidden terms as test vectors — skip it
SKIP_FILES = {"test_fork_traces.py"}


def _iter_files():
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.name in SKIP_FILES:
            continue
        if p.suffix.lower() in SKIP_SUFFIX:
            continue
        if p.stat().st_size > 5_000_000:
            continue
        yield p


def test_no_fork_traces_in_current_tree():
    hits = []
    for p in _iter_files():
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            m = FORBIDDEN.search(line)
            if m:
                hits.append(f"{p.relative_to(ROOT)}:{i}: {m.group(0)!r}")
    assert not hits, "forbidden fork/hardware traces returned to the tree:\n" + "\n".join(hits[:20])


def test_guard_is_not_a_false_positive_machine():
    """Sanity-check the guard itself: generic technical prose must NOT trigger."""
    assert not FORBIDDEN.search("GPU memory and CUDA kernels, MoE routing, graceful fallback")
    assert not FORBIDDEN.search("the model is fine, the laptop is fine")
    assert FORBIDDEN.search("on the DGX Station box")
    assert FORBIDDEN.search("gb300 machine")
    assert FORBIDDEN.search("J-M-Recipes fork")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))