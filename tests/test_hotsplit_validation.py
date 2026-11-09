"""Round 76: BEHAVIOURAL tests for the shared HotSplit config validation
(hotsplit_env.sh::_hotsplit_validate_config, review: the checks used to live
only in the kv launcher -- switching the engine path would have deleted them).

Runs the REAL shell function on the host (no docker):
  * the NUMA_MAP must COVER every GPU rank 0..TP-1 (review: "0:0,1:0" on TP4
    silently binds 2 of 4 GPUs while numa_bind reports ACTIVE) -- this check
    is a CONDITION for switching the default to the recipe launcher;
  * the documented map "0:0,1:0,2:1,3:1" passes at TP4;
  * the plan-code ranges still hold (DYN_DECAY 0..1, DYN_MARGIN >= 1,
    DYN_MAX_SWAPS 1..1024) in the SHARED module -- both engine paths.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
import pytest

RECIPES = Path(__file__).resolve().parent.parent
MODULE = (RECIPES / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit" /
          "scripts" / "hotsplit_env.sh")
SKIP_REASON = "shared hotsplit_env.sh not found"


def _validate(env: dict, tp: str = "4") -> subprocess.CompletedProcess:
    e = {"PATH": "/usr/bin:/bin", "HOTSPLIT_NUMA": "1", **env}
    return subprocess.run(
        ["bash", "-c", f'. "{MODULE}"; _hotsplit_validate_config {tp}'],
        env=e, capture_output=True, text=True, timeout=30)


@pytest.fixture(autouse=True)
def _require_module():
    if not MODULE.is_file():
        pytest.skip(SKIP_REASON)


def test_full_map_passes_at_tp4():
    """The documented map covers all four GPUs -- must pass."""
    r = _validate({"HOTSPLIT_NUMA_MAP": "0:0,1:0,2:1,3:1"})
    assert r.returncode == 0, f"the full map must pass:\n{r.stdout}{r.stderr}"


def test_partial_map_is_rejected_at_tp4():
    """Review: "0:0,1:0" at TP4 leaves GPU 2 and 3 unbound while numa_bind
    reports ACTIVE -- the validator must refuse it BEFORE the launch."""
    r = _validate({"HOTSPLIT_NUMA_MAP": "0:0,1:0"})
    assert r.returncode == 1, (
        f"a partial map must fail at TP4:\n{r.stdout}{r.stderr}")
    assert "no entry for GPU 2" in r.stderr, r.stderr
    assert "no entry for GPU 3" in r.stderr, r.stderr


def test_map_matching_tp_passes():
    """The same partial map is legal at TP=2 -- coverage is relative to TP."""
    r = _validate({"HOTSPLIT_NUMA_MAP": "0:0,1:0"}, tp="2")
    assert r.returncode == 0, f"0:0,1:0 must pass at TP=2:\n{r.stdout}{r.stderr}"


def test_malformed_map_is_rejected():
    """The old node:cpu format and garbage must fail with a clear error."""
    for bad in ("0:0,1,2,3;1:4,5,6,7", "a:b", "0:0,"):
        r = _validate({"HOTSPLIT_NUMA_MAP": bad})
        assert r.returncode == 1, f"{bad!r} must fail:\n{r.stdout}{r.stderr}"
        assert "HOTSPLIT_NUMA_MAP" in r.stderr, r.stderr


def test_plan_code_ranges_still_enforced():
    """The plan-code ranges (rounds 60-62) must still be enforced in the
    shared module: DYN_DECAY 0..1, DYN_MARGIN >= 1, DYN_MAX_SWAPS 1..1024."""
    for env, needle in (
        ({"HOTSPLIT_DYN_DECAY": "2"}, "DYN_DECAY"),
        ({"HOTSPLIT_DYN_DECAY": "-0.5"}, "DYN_DECAY"),
        ({"HOTSPLIT_DYN_MARGIN": "0.5"}, "DYN_MARGIN"),
        ({"HOTSPLIT_DYN_MAX_SWAPS": "2000"}, "DYN_MAX_SWAPS"),
    ):
        r = _validate(env)
        assert r.returncode == 1, (
            f"{env} must fail the plan-code range:\n{r.stdout}{r.stderr}")
        assert needle in r.stderr, (needle, r.stderr)


def test_valid_config_passes():
    """A fully valid config (documented values) must pass cleanly."""
    r = _validate({"HOTSPLIT_HOT_GIB": "30", "HOTSPLIT_NUMA_MAP": "0:0,1:0,2:1,3:1",
                   "HOTSPLIT_DYN_DECAY": "0.5", "HOTSPLIT_DYN_MARGIN": "1.25",
                   "HOTSPLIT_DYN_MAX_SWAPS": "2"})
    assert r.returncode == 0, f"the documented config must pass:\n{r.stdout}{r.stderr}"
    assert "hotsplit config OK" in r.stdout, r.stdout