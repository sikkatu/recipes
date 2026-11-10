"""Round 79 regression: recipe same-UID default + cache-server fail-fast.

The delegated recipe path (KVP_ENGINE=recipe) lost the kv launcher's same-UID
contract: docker run had no --user, so the engine ran as ROOT while the cache
server ran as the invoking user -- the EXACT uid mismatch behind the recurring
register_kv_caches 300 s timeout (CUDA_ERROR_MAP_FAILED / "Failed to
open/create interprocess memory file / remote IPC semaphore pool", EACCES on
root-owned cuda.shm.* in /dev/shm). Round 79:
* run-tp4-hotsplit.sh runs the engine as $(id -u):$(id -g) by DEFAULT
  (KVP_SAME_UID=1, opt out with KVP_SAME_UID=0) with HOME/HF/XDG/TRITON
  caches on /tmp -- exactly like run-vllm-with-persist.sh;
* the recipe API-wait loop fail-fasts on a fresh 'Error handling operation' in
  the cache-server log instead of waiting out the engine's 300 s timeout.
These tests are behavioural: they run the launcher against stub docker/curl
and inspect the real docker run line + the fail-fast exit.
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

GOOD_LOG = (
    "HOOK STATUS boot_hook=INACTIVE\n"
    "hotsplit dynamic: step hook installed on GPUModelRunnerImpl (update every 2000 steps, <=2 swaps/layer, margin 1.25)\n"
    "hotsplit live counter: 69 layers -> /live/counts.json\n"
    "HOOK STATUS numa_bind=ACTIVE\n"
)

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
        # simulate the cache server failing DURING the load (after the mark)
        [ -n "${STUB_SERVER_LOG_APPEND:-}" ] && printf '%s
' "$STUB_SERVER_LOG_APPEND" >> "${KVP_CACHE_ROOT:-/tmp}/lmcache-server.log"
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
    for d in ("model", "home", "live", "cache"):
        (tmp_path / d).mkdir()
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["STUB_LOG"] = str(log)
    env["MODEL_DIR"] = str(tmp_path / "model")
    env["HF_CACHE"] = str(tmp_path / "hf")
    env["LIVE_DIR"] = str(tmp_path / "live")
    env["HOME"] = str(tmp_path / "home")
    env["KVP_API_KEY_FILE"] = str(tmp_path / "home" / ".config" / "kvp" / "api_key")
    env["KVP_CACHE_ROOT"] = str(tmp_path / "cache")
    env["WAIT_TRIES"] = "1"
    env["WAIT_SLEEP"] = "0"
    env["KVP_SKIP_IPC_TEST"] = "1"
    return env, log


def _launch(env, log, extra):
    if log.exists():
        log.unlink()
    e = {**env, "STUB_DOCKER_LOGS": GOOD_LOG,
         "API_KEY": "secret123", "IMAGE": "x@sha256:" + "a" * 64,
         "ALLOW_RUN": "1", **extra}
    return subprocess.run(["bash", str(LAUNCHER)], env=e,
                          capture_output=True, text=True, timeout=120)


def _docker_run_line(log: Path):
    for line in log.read_text().splitlines():
        if "docker run " in line and "--rm" not in line:
            return line
    return ""


def test_same_uid_is_the_default_recipe_engine_user(stub_env):
    """The recipe engine must run as $(id -u):$(id -g) -- the SAME user as the
    cache server -- with the caches on /tmp (a non-root user cannot write
    /root). This is the kv launcher's round-45h/48 contract; losing it in the
    recipe path caused the recurring CUDA_ERROR_MAP_FAILED."""
    env, log = stub_env
    r = _launch(env, log, {"KVP_SAME_UID": "1"})
    assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
    line = _docker_run_line(log)
    assert line, "no docker run line in the stub log"
    uid = os.getuid()
    gid = os.getgid()
    assert f"--user {uid}:{gid}" in line, (
        f"same-UID mode must pass --user uid:gid (got {uid}:{gid}): {line}")
    assert "-e HF_HOME=/tmp/hf-cache" in line, (
        "same-UID mode must use /tmp caches (non-root cannot write /root): " + line)
    assert "HF_HOME=/root/.cache/huggingface" not in line, (
        "same-UID mode must not use the root HF cache: " + line)


def test_root_fallback_opt_out_keeps_the_old_behaviour(stub_env):
    """KVP_SAME_UID=0 is the documented root fallback: no --user, root caches."""
    env, log = stub_env
    r = _launch(env, log, {"KVP_SAME_UID": "0"})
    assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
    line = _docker_run_line(log)
    assert "--user " not in line, (
        f"root fallback must not pass --user: {line}")
    assert "-e HF_HOME=/root/.cache/huggingface" in line, (
        "root fallback keeps the root HF cache: " + line)


def test_fail_fast_on_server_handler_error(stub_env):
    """A fresh 'Error handling operation' in the cache-server log (e.g.
    CUDA_ERROR_MAP_FAILED) must abort the API wait immediately with the
    root cause -- waiting out the engine's 300 s register_kv_caches timeout
    cannot fix it (the review's fail-fast requirement)."""
    env, log = stub_env
    # The error must appear DURING the launch (the fail-fast mark is taken
    # before docker run; only FRESH server-log bytes are scanned -- same as
    # the kv launcher). The stub appends it at the docker run moment.
    err = (
        "[2026-11-10 13:22:12,063] LMCache ERROR: Error handling operation "
        "register_kv_cache (mq.py:798)\n"
        "torch.AcceleratorError: CUDA error: mapping of buffer object failed\n"
        "Returning 205 (CUDA_ERROR_MAP_FAILED) from cuIpcOpenEventHandle")
    r = _launch(env, log, {"STUB_SERVER_LOG_APPEND": err})
    out = r.stdout + r.stderr
    assert r.returncode == 1, f"the launch must fail fast, got rc={r.returncode}\n{out[-800:]}"
    assert "fail-fast" in out, f"the failure must be labelled fail-fast:\n{out[-800:]}"
    assert "Error handling operation" in out, "the root cause must be printed"
    assert "MAP_FAILED" in out or "mapping of buffer object failed" in out, (
        "the driver-level cause must be printed")

STUB_UNREGISTER_NOISE = (
    "[2026-11-10 05:11:19,000] LMCache ERROR: Error handling operation "
    "unregister_kv_cache (mq.py:798)\\n"
    "zmq.error.ZMQError: Host unreachable")


def test_benign_unregister_noise_does_not_trigger_fail_fast(stub_env):
    """Round 79 review: the fail-fast trigger must match ONLY
    'Error handling operation register_kv_cache'. The broad pattern also
    matches the benign 'unregister_kv_cache' / ZMQError: Host unreachable
    noise the server writes when an old engine dies -- a false abort after
    the log mark."""
    env, log = stub_env
    r = _launch(env, log, {"STUB_SERVER_LOG_APPEND": STUB_UNREGISTER_NOISE})
    out = r.stdout + r.stderr
    assert "fail-fast" not in out, (
        f"benign unregister noise must NOT trigger fail-fast: {out[-600:]}")
