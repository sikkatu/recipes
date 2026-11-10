"""Round 80 review (LIVE_DIR drift): start-all.sh must pin ONE canonical LIVE_DIR.

The recipe launcher defaulted LIVE_DIR to $PWD/live, so launching
start-all.sh from ANY directory other than the repo root wrote the state
(live/image, the api_key mirror, live/engine_container) to the caller's
$PWD/live -- the state silently drifted. start-all.sh now exports
LIVE_DIR=$ROOT/live. This test launches start-all.sh by ABSOLUTE path from a
foreign cwd and asserts every state file lands in the repo's live/.
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
KV_REPO = _T.kv_repo()
SKIP = "kv-persistence checkout not found (the drift test needs start-all.sh)"

GOOD_MARKERS = (
    "HOOK STATUS boot_hook=INACTIVE\n"
    "hotsplit dynamic: step hook installed on GPUModelRunnerImpl (update every 2000 steps, <=2 swaps/layer, margin 1.25)\n"
    "hotsplit live counter: 69 layers -> /live/counts.json\n"
    "HOOK STATUS numa_bind=ACTIVE\n"
)

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
    # a FOREIGN cwd -- the whole point of the test
    foreign = tmp_path / "somewhere-else"
    foreign.mkdir()
    for d in ("model", "home"):
        (tmp_path / d).mkdir()
    home = tmp_path / "home"
    (home / ".config" / "kvp").mkdir(parents=True)
    key_file = home / ".config" / "kvp" / "api_key"
    key_file.write_text("sharedkey123\n")
    key_file.chmod(0o600)
    # a live listener so the preserved TCP pre-flight passes
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
    env["HOME"] = str(home)
    env["KVP_API_KEY_FILE"] = str(key_file)
    env["KVP_SKIP_GPU_CHECK"] = "1"
    env["KVP_SKIP_IPC_TEST"] = "1"
    env["KVP_SKIP_HOTSPLIT_CHECK"] = "1"
    env["WAIT_TRIES"] = "1"
    env["WAIT_SLEEP"] = "0"
    # a RUNNING cache-server pid stub makes start-all's cache step skip
    # (no real lmcache CLI in the sandbox) -- same device as the delegation tests
    cache_root = tmp_path / "cache"
    cache_root.mkdir(exist_ok=True)
    (cache_root / "lmcache-server.pid").write_text(str(os.getpid()))
    env["KVP_CACHE_ROOT"] = str(cache_root)
    env["LMCACHE_HOST"] = "127.0.0.1"
    env["LMCACHE_PORT"] = str(srv.getsockname()[1])
    env["KV_PERSIST_HOST"] = "127.0.0.1"
    env["KV_PERSIST_PORT"] = str(srv.getsockname()[1])
    # scrub any inherited LIVE_DIR so the drift test is honest
    env.pop("LIVE_DIR", None)
    return env, log, foreign


def test_state_files_land_in_the_repo_live_dir(stub_env):
    """Launching start-all.sh by absolute path from a foreign cwd must write
    ALL state to the repo's live/ -- never to the caller's $PWD/live."""
    if KV_REPO is None:
        pytest.skip(SKIP)
    env, log, foreign = stub_env
    start_all = KV_REPO / "scripts" / "start-all.sh"
    r = subprocess.run(
        ["bash", str(start_all)],
        env={**env, "RECIPE_DIR": str(RECIPE), "KVP_ENGINE": "recipe",
             "IMAGE": "x@sha256:" + "a" * 64, "ALLOW_RUN": "1",
             "KVP_PERSIST_ENABLED": "false"},
        capture_output=True, text=True, timeout=180,
        cwd=str(foreign))
    # the launch must at least reach the engine step
    assert "KVP_ENGINE=recipe" in (r.stdout + r.stderr), (r.stdout + r.stderr)[-800:]
    # NO state in the foreign cwd
    foreign_live = foreign / "live"
    assert not foreign_live.exists() or not any(foreign_live.iterdir()), (
        f"state drifted into the caller's PWD: {[p.name for p in foreign_live.iterdir()]}")
    # the canonical live/ holds the records
    repo_live = KV_REPO / "live"
    assert (repo_live / "engine_container").exists(), (
        "live/engine_container must be written to the REPO live dir")
    assert (repo_live / "image").exists(), (
        "live/image must be written to the REPO live dir")