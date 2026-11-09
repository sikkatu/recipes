"""Round 76: BIG-LOG regression -- the activation check must not false-negative
on a large engine log (review, reproducible).

The bug: `printf '%s' "$_hs_log" | grep -q ...` under `set -o pipefail`.
grep -q exits on the FIRST match and closes the pipe; printf then gets
SIGPIPE (141) and the pipeline is "failed", so the `if` takes the
NOT-found branch even though the marker IS in the log. On a model-load log
(17 min, MBs) ALL four markers read [MISSING] while the engine is fine --
and the SAME pipes guard the ERROR checks, so a real error is missed too.

These tests feed a 100 KiB log WITH the markers at the top and assert the
launch is VERIFIED (the old pipe code reads [MISSING] for every marker).
"""
from __future__ import annotations
import os
import stat
import subprocess
from pathlib import Path
import pytest

RECIPES = Path(__file__).resolve().parent.parent
RECIPE = RECIPES / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit"
LAUNCHER = RECIPE / "scripts" / "run-tp4-hotsplit.sh"

GOOD_MARKERS = """HOOK STATUS boot_hook=INACTIVE
hotsplit dynamic: step hook installed on GPUModelRunnerImpl (update every 2000 steps, <=2 swaps/layer, margin 1.25)
hotsplit live counter: 69 layers -> /live/counts.json
HOOK STATUS numa_bind=ACTIVE
"""
# 100 KiB of plausible load-log filler AFTER the markers -- big enough to
# overflow the 64 KiB pipe buffer and trigger SIGPIPE in the old code.
BIG_LOG = GOOD_MARKERS + ("(Worker_TP0 pid=361) INFO loading layer weights\n" * 2000)

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
    e = {**env, "STUB_DOCKER_LOGS": docker_logs,
         "API_KEY": "secret123", "IMAGE": "x@sha256:" + "a" * 64, "ALLOW_RUN": "1"}
    return subprocess.run(["bash", str(LAUNCHER)], env=e,
                          capture_output=True, text=True, timeout=120)


def test_big_log_with_markers_is_verified(stub_env):
    """100 KiB engine log with the markers at the top -> VERIFIED, exit 0.
    The old `printf | grep -q` code reported [MISSING] for all four markers
    (SIGPIPE under pipefail) and exited 1."""
    env, log = stub_env
    assert len(BIG_LOG) > 65536, "the fixture must exceed the 64 KiB pipe buffer"
    r = _launch(env, log, BIG_LOG)
    out = r.stdout + r.stderr
    assert r.returncode == 0, (
        f"a big log with all markers must PASS the activation check, "
        f"got rc={r.returncode}\n{out[-1200:]}")
    assert "VERIFIED" in out, f"activation must be VERIFIED:\n{out[-1200:]}"
    assert "[MISSING]" not in out, (
        f"no marker may read MISSING when it is in the log:\n{out[-1200:]}")


def test_big_log_without_markers_still_fails(stub_env):
    """The fix must not weaken the check: a big log WITHOUT the markers must
    still fail the launch (exit 1) -- both directions of the pipe bug."""
    env, log = stub_env
    filler = "some engine output line without any marker\n" * 2000
    assert len(filler) > 65536
    r = _launch(env, log, filler)
    out = r.stdout + r.stderr
    assert r.returncode == 1, (
        f"missing markers must fail the launch even on a big log, "
        f"got rc={r.returncode}\n{out[-800:]}")
    assert "NOT VERIFIED" in out, out[-800:]


def test_big_log_error_marker_is_caught(stub_env):
    """The ERROR checks had the same pipe bug: a real error on a big log must
    be reported, not silently missed."""
    env, log = stub_env
    bad = GOOD_MARKERS + ("loading line\n" * 2000) + \
        "(Worker_TP0 pid=1) ERROR hotsplit dynamic: commit failed ('x'); rolled back cleanly\n"
    r = _launch(env, log, bad)
    out = r.stdout + r.stderr
    assert r.returncode == 1, (
        f"a transaction error on a big log must fail the launch, "
        f"got rc={r.returncode}\n{out[-800:]}")
    assert "transaction errors" in out, out[-800:]