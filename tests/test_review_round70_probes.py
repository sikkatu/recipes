import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import _contract as _T
"""Review round 70: three delegation defects as FAILING regression tests.
Drop into recipes/tests/ (reuses the fixtures of test_engine_delegation.py).
Each test fails on commit 09a2654 (kv) / 8553763 (recipes) and must pass after the fix.
"""
import re
import subprocess
from test_engine_delegation import (stub_env, _start_all, _docker_run_line,
                                    KV_REPO, RECIPE, GOOD_MARKERS, Path)

X = {"RECIPE_DIR": str(RECIPE), "IMAGE": "x@sha256:" + "a" * 64}
REC = KV_REPO / "live" / "engine_container"


def _name(line):
    m = re.search(r"--name (\S+)", line or "")
    return m.group(1) if m else None


def _hotsplit_env(line):
    return dict(re.findall(r"-e (HOTSPLIT_[A-Z_]+)=(\S*)", line or ""))


def test_A_delegated_launch_has_the_same_hot_bank_as_the_kv_path(stub_env):
    """No tp4.env, clean env: KVP_ENGINE=recipe must not silently turn the
    hot bank into a no-op (kv path: HOTSPLIT_HOT_GIB=30)."""
    env, log, _ = stub_env
    REC.unlink(missing_ok=True)
    r = _start_all(env, log, {**X, "KVP_ENGINE": "recipe"}, GOOD_MARKERS)
    recipe_env = _hotsplit_env(_docker_run_line(log))
    out = r.stdout + r.stderr
    REC.unlink(missing_ok=True)
    _start_all(env, log, {**X, "KVP_ENGINE": "recipe"}, GOOD_MARKERS)  # Round 80b: one engine path
    kv_env = _hotsplit_env(_docker_run_line(log))
    assert "split is a no-op" not in out, "recipe launcher warned: hot bank would be EMPTY"
    assert recipe_env.get("HOTSPLIT_HOT_GIB") == kv_env.get("HOTSPLIT_HOT_GIB") == "30", (recipe_env, kv_env)


def test_B_stop_engine_stops_what_the_default_path_started(stub_env):
    """recipe run, then default (kv) run, then stop-engine.sh: the record in
    live/engine_container must follow the LAST launch."""
    env, log, _ = stub_env
    REC.unlink(missing_ok=True)
    _start_all(env, log, {**X, "KVP_ENGINE": "recipe"}, GOOD_MARKERS)
    _start_all(env, log, {**X, "KVP_ENGINE": "recipe"}, GOOD_MARKERS)  # Round 80b: one engine path
    started = _name(_docker_run_line(log))
    log.unlink(missing_ok=True)
    subprocess.run(["bash", str(KV_REPO / "scripts" / "stop-engine.sh")], env=env,
                   capture_output=True, text=True, timeout=60)
    stops = [l for l in log.read_text().splitlines()
             if "docker" in l and re.search(r" (stop|rm|kill) ", l)]
    REC.unlink(missing_ok=True)
    assert started and any(started in l for l in stops), (started, stops)


def test_C_recipe_path_on_a_fresh_machine_generates_the_key_like_the_kv_path(stub_env):
    env, log, _ = stub_env
    key = Path(env["KVP_API_KEY_FILE"])
    key.unlink()
    (KV_REPO / "live" / "api_key").unlink(missing_ok=True)
    r = _start_all(env, log, {**X, "KVP_ENGINE": "recipe"}, GOOD_MARKERS)
    REC.unlink(missing_ok=True)
    (KV_REPO / "live" / "api_key").unlink(missing_ok=True)
    assert r.returncode == 0, (r.stdout + r.stderr)[-300:]
    assert key.exists() and key.read_text().strip(), "no shared key was created"
