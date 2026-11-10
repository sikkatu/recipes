"""Round 80b: the engine docker-run CONTRACT (was launcher parity).

Round 66/72 compared the kv launcher's docker run against the recipe's to
catch drift between the two engine paths. The kv launcher is DELETED (round
80b) -- there is ONE engine path now (run-tp4-hotsplit.sh), so "parity" is
meaningless. What survives is the CONTRACT the parity test protected: the
hardware-verified docker profile must not regress. These tests assert it on
the recipe directly:
  * the capabilities + CUDA-graph profile of the verified recipe;
  * the /w hotsplit machinery and /live counters mounts;
  * the key travels via --env-file, NEVER in argv;
  * the shared hotsplit env reaches the container.
"""
from __future__ import annotations
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import _contract as _T
import os
import stat
import subprocess
from pathlib import Path

import pytest

RECIPES = Path(__file__).resolve().parent.parent
RECIPE = RECIPES / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit"
LAUNCHER = RECIPE / "scripts" / "run-tp4-hotsplit.sh"

STUB = '''#!/usr/bin/env bash
echo "$0 $*" >> "$STUB_LOG"
case "$(basename "$0")" in
  docker)
    case "$1" in
      info) exit 0 ;;
      container) exit 1 ;;
      inspect) echo "running exit=0 oom=false"; exit 0 ;;
      logs) printf '%s' "${STUB_DOCKER_LOGS:-}" ;;
      ps) exit 0 ;;
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

PYTHON_STUB = '''#!/usr/bin/env bash
case "$*" in
  *kvpersist.cli*model-name*) echo "mimo-pro/testhash" ;;
esac
exec /usr/bin/python3 "$@"
'''

GOOD_MARKERS = (
    "HOOK STATUS boot_hook=INACTIVE\n"
    "hotsplit dynamic: step hook installed on GPUModelRunnerImpl (update every 2000 steps, <=2 swaps/layer, margin 1.25)\n"
    "hotsplit live counter: 69 layers -> /live/counts.json\n"
    "HOOK STATUS numa_bind=ACTIVE\n"
)


@pytest.fixture()
def stub_env(tmp_path):
    import atexit
    import socket
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "stub.log"
    for name in ("docker", "nvidia-smi", "curl", "sha256sum"):
        p = bin_dir / name
        p.write_text(STUB)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    py = bin_dir / "python3"
    py.write_text(PYTHON_STUB)
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    for d in ("model", "home", "live", "cache"):
        (tmp_path / d).mkdir()
    home = tmp_path / "home"
    (home / ".config" / "kvp").mkdir(parents=True)
    key_file = home / ".config" / "kvp" / "api_key"
    key_file.write_text("sharedkey123\n")
    key_file.chmod(0o600)
    (tmp_path / "cache" / "lmcache-server.pid").write_text(str(os.getpid()))
    # a real listener so the preserved TCP pre-flight passes
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    atexit.register(srv.close)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["STUB_LOG"] = str(log)
    env["STUB_DOCKER_LOGS"] = GOOD_MARKERS
    env["MODEL_DIR"] = str(tmp_path / "model")
    env["HF_CACHE"] = str(tmp_path / "hf")
    env["LIVE_DIR"] = str(tmp_path / "live")
    env["HOME"] = str(home)
    env["KVP_API_KEY_FILE"] = str(key_file)
    env["KVP_CACHE_ROOT"] = str(tmp_path / "cache")
    env["KVP_SKIP_GPU_CHECK"] = "1"
    env["KVP_SKIP_IPC_TEST"] = "1"
    env["KVP_SKIP_HOTSPLIT_CHECK"] = "1"
    env["WAIT_TRIES"] = "1"
    env["WAIT_SLEEP"] = "0"
    env["LMCACHE_HOST"] = "127.0.0.1"
    env["LMCACHE_PORT"] = str(srv.getsockname()[1])
    env["KV_PERSIST_HOST"] = "127.0.0.1"
    env["KV_PERSIST_PORT"] = str(srv.getsockname()[1])
    return env, log


def _launch(env, log, extra):
    if log.exists():
        log.unlink()
    e = {**env, "API_KEY": "secret123", "IMAGE": "x@sha256:" + "a" * 64,
         "ALLOW_RUN": "1", "KV_PERSIST": "0", **extra}
    return subprocess.run(["bash", str(LAUNCHER)], env=e,
                          capture_output=True, text=True, timeout=180)


def _docker_run_line(log: Path) -> str:
    for line in log.read_text().splitlines():
        if "docker run " in line and "--rm" not in line:
            return line
    return ""


def test_docker_run_keeps_the_verified_profile(stub_env):
    """The hardware-verified docker profile must not regress: the capability
    adds, the FULL_DECODE_ONLY CUDA-graph profile and the /w hotsplit
    machinery are the recipe's identity."""
    env, log = stub_env
    r = _launch(env, log, {})
    line = _docker_run_line(log)
    assert line, f"the launcher did not reach docker run:\n{(r.stdout + r.stderr)[-600:]}"
    for flag in ("--cap-add IPC_LOCK", "--cap-add SYS_NICE",
                 "--compilation-config", "/w/hotsplit.py", "/w/sitecustomize.py"):
        assert flag in line, f"the docker run is missing {flag}: {line}"


def test_key_travels_via_env_file_never_in_argv(stub_env):
    """The API key must travel in an --env-file, never as a command-line
    value (ps output would leak it)."""
    env, log = stub_env
    r = _launch(env, log, {})
    line = _docker_run_line(log)
    assert line, (r.stdout + r.stderr)[-600:]
    assert "--env-file" in line, "the key must travel via --env-file"
    assert "VLLM_API_KEY=" not in line.split("--env-file")[0], (
        "the key must never appear in argv before --env-file")
    assert "VLLM_API_KEY=" not in line, "the key must never be an argv value"


def test_shared_hotsplit_env_reaches_the_container(stub_env):
    """The shared hotsplit module's env must reach the docker run (the
    /w machinery without its env is a silent no-op)."""
    env, log = stub_env
    r = _launch(env, log, {})
    line = _docker_run_line(log)
    assert line, (r.stdout + r.stderr)[-600:]
    for var in ("HOTSPLIT_DYNAMIC=", "HOTSPLIT_NUMA=", "HOTSPLIT_BOOT_HOOK=",
                "HOTSPLIT_COUNTS=", "HOTSPLIT_HOST_FLOOR_GIB="):
        assert var in line, f"docker run is missing {var} (the shared module must provide it)"
    assert "/live" in line, "the /live counters mount must exist"