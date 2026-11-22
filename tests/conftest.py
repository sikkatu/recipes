# SPDX-License-Identifier: Apache-2.0
"""r25 G9 shared fixtures: the test suite must never write into the
OPERATOR's kv-persistence tree.

Contract (r25 G9):
1. _contract.kv_repo() returns a per-session TEMP COPY of the kv checkout
   (scripts/, configs/, implementation/, vendor/, observability/, live/) --
   every start-all.sh / launcher run writes its state records there.
2. KVP_OBS_DIR points at a tmp COPY of the extras sources -- extras-up.sh
   renders targets/*.json and gpu/dcgm-counters.live.csv there, never into
   the repo's observability/extras/ (same contract as kv-persistence's
   conftest, r24 E1.3).
3. A session guard fails the run if the real tree's observability/extras
   generated files (targets/*.json, gpu/dcgm-counters.live.csv) changed.
"""
from __future__ import annotations
import hashlib
import os
import pathlib
import shutil
import tempfile

import pytest

RECIPES = pathlib.Path(__file__).resolve().parent.parent
EXTRAS = RECIPES.parent / "kv-persistence" / "observability" / "extras"
GUARDED = (
    "targets/gpu.json",
    "targets/vllm.json",
    "gpu/dcgm-counters.live.csv",
)


def _guarded_state() -> dict:
    snap = {}
    for rel in GUARDED:
        p = EXTRAS / rel
        if p.is_file():
            snap[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return snap


@pytest.fixture(scope="session", autouse=True)
def _extras_isolation():
    """Point KVP_OBS_DIR at a tmp copy of the extras sources (E1.3/G9) and
    guard the REAL generated files against modification."""
    before = _guarded_state()
    tmp_obs = pathlib.Path(tempfile.mkdtemp(prefix="kvp-recipes-obs-"))
    if EXTRAS.is_dir():
        shutil.copytree(EXTRAS, tmp_obs / "extras")
    os.environ["KVP_OBS_DIR"] = str(tmp_obs)
    yield
    os.environ.pop("KVP_OBS_DIR", None)
    shutil.rmtree(tmp_obs, ignore_errors=True)
    after = _guarded_state()
    problems = []
    for rel, h in after.items():
        if rel not in before:
            problems.append(f"ADDED {rel}")
        elif before[rel] != h:
            problems.append(f"CHANGED {rel}")
    for rel in before:
        if rel not in after:
            problems.append(f"REMOVED {rel}")
    if problems:
        pytest.fail(
            "r25 G9: the run modified the REAL observability/extras generated "
            f"files (tests must use KVP_OBS_DIR / the kv_repo tmp copy): "
            + ", ".join(problems))