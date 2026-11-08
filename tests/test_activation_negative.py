"""Round 69: step-2 prep tests -- activation negative + docker-run contract snapshot.

Checklist items prepared BEFORE KVP_ENGINE=recipe exists:
  1. "Activation hotsplit without the needed log marker must give exit 1" --
     the recipe launcher now runs the same marker check as the kv launcher
     (round 69a); this test proves a MISSING marker fails the launch and a
     complete set passes it, with the code visible to start-all / kvp up.
  2. "Without the flag nothing changes -- a snapshot of the args before and
     after is enough" -- test_snapshot_contract compares the live docker run
     line against tests/fixtures/launcher-docker-run.txt (regenerate with
     KVP_UPDATE_SNAPSHOT=1).
"""
from __future__ import annotations
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

RECIPES = Path(__file__).resolve().parent.parent
RECIPE = RECIPES / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit"
LAUNCHER = RECIPE / "scripts" / "run-tp4-hotsplit.sh"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "launcher-docker-run.txt"

GOOD_LOG = """HOOK STATUS boot_hook=INACTIVE
hotsplit dynamic: step hook installed on GPUModelRunnerImpl (update every 2000 steps, <=2 swaps/layer, margin 1.25)
hotsplit live counter: 69 layers -> /live/counts.json
HOOK STATUS numa_bind=ACTIVE
"""
BAD_LOG = "some engine output without any hotsplit markers\n"

STUB_DOCKER = '''#!/usr/bin/env bash
echo "$0 $*" >> "$STUB_LOG"
case "$(basename "$0")" in
  docker)
    case "$1" in
      info) exit 0 ;;
      container) exit 1 ;;
      inspect) echo "running exit=0 oom=false"; exit 0 ;;
      logs) printf '%s' "$STUB_DOCKER_LOGS" ;; 
      run)
        case "$*" in
          *--rm*) echo "/usr/local/lib/python3.12/dist-packages"; exit 0 ;;
        esac
        echo "STUB_DOCKER_RUN" >> "$STUB_LOG"
        exit 0 ;;
    esac
    exit 0 ;;
  nvidia-smi) [ "$1" = "-L" ] && for i in 1 2 3 4; do echo "GPU $i: STUB"; done; exit 0 ;;
  curl)
    # answer the readiness poll and the smoke test like a healthy server
    case "$*" in
      *"/v1/models"*) echo '{"data":[{"id":"mimo-pro"}]}' ;;
      *"/v1/chat/completions"*) echo '{"choices":[{"message":{"content":"TP4_OK"},"finish_reason":"stop"}]}' ;;
      *) echo '{}' ;;
    esac
    exit 0 ;;
  sha256sum) exec /usr/bin/sha256sum "$@" ;;
  *) exit 0 ;;
esac
'''


@pytest.fixture()
def stub_env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "stub.log"
    for name in ("docker", "nvidia-smi", "curl", "sha256sum"):
        p = bin_dir / name
        p.write_text(STUB_DOCKER)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    for d in ("model", "home", "live"):
        (tmp_path / d).mkdir()
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["STUB_LOG"] = str(log)
    env["MODEL_DIR"] = str(tmp_path / "model")
    env["HF_CACHE"] = str(tmp_path / "hf")
    env["LIVE_DIR"] = str(tmp_path / "live")
    env["HOME"] = str(tmp_path / "home")
    env["KVP_API_KEY_FILE"] = str(tmp_path / "home" / ".config" / "kvp" / "api_key")
    env["WAIT_TRIES"] = "1"
    env["WAIT_SLEEP"] = "0"
    return env, log


def _launch(env, log, docker_logs: str):
    if log.exists():
        log.unlink()
    e = {**env,
         "STUB_DOCKER_LOGS": docker_logs,
         "API_KEY": "secret123", "IMAGE": "x@sha256:" + "a" * 64, "ALLOW_RUN": "1"}
    return subprocess.run(["bash", str(LAUNCHER)], env=e,
                          capture_output=True, text=True, timeout=120)


def test_activation_missing_marker_fails_the_launch(stub_env):
    """Checklist (step 2): the activation check WITHOUT the needed log marker
    must give exit 1 -- the launch is not a success just because /v1/models
    answered and the smoke test returned TP4_OK."""
    env, log = stub_env
    r = _launch(env, log, BAD_LOG)
    out = r.stdout + r.stderr
    assert r.returncode == 1, f"missing markers must fail the launch, got rc={r.returncode}\n{out[-800:]}"
    assert "NOT VERIFIED" in out, f"the failure must name the activation check:\n{out[-800:]}"
    assert "[MISSING]" in out, "each missing marker must be listed"


def test_activation_complete_markers_pass(stub_env):
    """The same launch with the FULL marker set must succeed (exit 0)."""
    env, log = stub_env
    r = _launch(env, log, GOOD_LOG)
    out = r.stdout + r.stderr
    assert r.returncode == 0, f"complete markers must pass, got rc={r.returncode}\n{out[-800:]}"
    assert "VERIFIED" in out


def test_activation_skip_switch_works(stub_env):
    """KVP_SKIP_HOTSPLIT_CHECK=1 is the documented opt-out (e.g. for
    PATCH_MODEL_FILES=0 runs without the machinery)."""
    env, log = stub_env
    if log.exists():
        log.unlink()
    e = {**env, "STUB_DOCKER_LOGS": BAD_LOG,
         "API_KEY": "secret123", "IMAGE": "x@sha256:" + "a" * 64, "ALLOW_RUN": "1",
         "KVP_SKIP_HOTSPLIT_CHECK": "1"}
    r = subprocess.run(["bash", str(LAUNCHER)], env=e,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"the skip switch must bypass the check:\n{(r.stdout + r.stderr)[-800:]}"


def _norm(line: str) -> str:
    """Strip the tmp_path tokens (they differ per run) from the docker run."""
    return re.sub(r"/tmp/pytest-of-[^/]+/pytest-\d+/[a-z0-9_]+", "<TMP>", line)


def test_snapshot_contract(stub_env):
    """Checklist (step 2): WITHOUT the KVP_ENGINE flag the final docker run
    must stay byte-identical to the recorded contract. Regenerate with
    KVP_UPDATE_SNAPSHOT=1 (do that only on an intentional change)."""
    env, log = stub_env
    r = _launch(env, log, GOOD_LOG)
    assert log.exists(), "the launcher did not reach docker run"
    run_line = None
    for line in log.read_text().splitlines():
        if "docker run " in line and "--rm" not in line:
            run_line = _norm(line)
    assert run_line, "no docker run line in the stub log"

    if os.environ.get("KVP_UPDATE_SNAPSHOT") == "1":
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(
            "# Round 69: the EXPECTED docker run (normalized). Regenerate with\n"
            "# KVP_UPDATE_SNAPSHOT=1 python3 -m pytest tests/test_activation_negative.py::test_snapshot_contract\n"
            + run_line + "\n")
        pytest.skip("snapshot regenerated")

    assert FIXTURE.is_file(), (
        "the contract snapshot is missing -- generate it once with KVP_UPDATE_SNAPSHOT=1")
    expected = [ln for ln in FIXTURE.read_text().splitlines()
                if ln and not ln.startswith("#")]
    assert len(expected) == 1, "the snapshot must hold exactly one docker run line"
    assert run_line == expected[0], (
        "the docker run CHANGED vs the recorded contract:\n"
        f"  expected: {expected[0][:300]}\n  got:      {run_line[:300]}")