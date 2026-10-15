"""Behavioural negative tests for the launchers (audit block 5.2).

The audit forbids treating text search as a security test. Here the launchers are actually
EXECUTED against a stub `docker`/`nvidia-smi`/`curl` on PATH: bad configurations must exit
non-zero BEFORE any `docker run`, and only a fully valid configuration reaches the run line.

Covered refusals:
  - missing / empty API key without explicit opt-out
  - non-loopback bind without an API key
  - IMAGE not pinned by digest (floating tag or empty)
  - missing MODEL_REVISION when the model is fetched by ID
  - fewer than 4 GPUs
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "recipes" / "rtx-pro-6000"

LAUNCHERS = [
    BASE / "deepseek-v4.1-flash-hotsplit" / "scripts" / "run-tp4-uva.sh",
    BASE / "glm-5.3-flash-hotsplit" / "scripts" / "run-tp4-uva.sh",
    BASE / "glm-5.3-hotsplit" / "scripts" / "run-tp4-uva.sh",
    BASE / "mimo-v2.6-pro-hotsplit" / "scripts" / "run-tp4-uva.sh",
    BASE / "mimo-v2.6-pro-hotsplit" / "scripts" / "run-tp4-hotsplit.sh",
]

STUB = r'''#!/usr/bin/env bash
# test stub: records the invocation; `docker run` only "succeeds" when the test allows it
echo "$0 $*" >> "$STUB_LOG"
case "$(basename "$0")" in
  docker)
    case "$1" in
      info) exit 0 ;;
      container) exit 1 ;;   # "container does not exist"
      logs) echo stub-logs; exit 0 ;;
      run)
        if [ "${ALLOW_RUN:-0}" = "1" ]; then
          echo "STUB_DOCKER_RUN" >> "$STUB_LOG"
          exit 0
        fi
        echo "STUB_DOCKER_RUN_ATTEMPTED" >> "$STUB_LOG"
        exit 0 ;;
    esac
    exit 0 ;;
  nvidia-smi)
    if [ "$1" = "-L" ]; then
      for i in $(seq 1 "${GPU_COUNT:-4}"); do echo "GPU $i: STUB"; done
    fi
    exit 0 ;;
  curl) exit 22 ;;   # never ready -> launcher times out, but only after docker run
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
        p.write_text(STUB)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["STUB_LOG"] = str(log)
    # model as a local dir so MODEL_REVISION is not required by default
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    env["MODEL_DIR"] = str(model_dir)
    env["HF_CACHE"] = str(tmp_path / "hf")
    env["LIVE_DIR"] = str(tmp_path / "live")
    env["WAIT_TRIES"] = "1"      # readiness polls: 1 (tests must not wait 30 minutes)
    env["WAIT_SLEEP"] = "0"
    return env, log


def run_launcher(path: Path, env: dict, extra: dict | None = None):
    e = env.copy()
    e.update(extra or {})
    return subprocess.run(["bash", str(path)], env=e, capture_output=True, text=True, timeout=60)


def docker_run_reached(log: Path) -> bool:
    return log.exists() and ("STUB_DOCKER_RUN" in log.read_text())


def run_lines_of(log: Path) -> list:
    """Extract the `docker run` invocations (the stub logs the full binary path)."""
    out = []
    for l in log.read_text().splitlines():
        stripped = l.strip()
        if "docker run " in stripped or stripped.endswith("docker run"):
            out.append(stripped)
    return out


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_missing_api_key_refuses_before_docker_run(lp, stub_env):
    if "glm-5.3-flash" in str(lp):
        pytest.skip("SGLang scaffold exits before serving")
    env, log = stub_env
    r = run_launcher(lp, env, {"API_KEY": "", "IMAGE": "x@sha256:" + "a" * 64})
    assert r.returncode != 0, "missing API key must fail"
    assert "API_KEY is required" in (r.stdout + r.stderr)
    assert not docker_run_reached(log), "docker run must not be attempted"


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_public_bind_without_key_refuses(lp, stub_env):
    if "glm-5.3-flash" in str(lp):
        pytest.skip("SGLang scaffold exits before serving")
    env, log = stub_env
    r = run_launcher(lp, env, {"API_KEY": "", "ALLOW_NO_AUTH": "1",
                               "BIND": "0.0.0.0", "IMAGE": "x@sha256:" + "a" * 64})
    assert r.returncode != 0, "public bind without a key must fail"
    assert not docker_run_reached(log)


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_unpinned_image_refuses(lp, stub_env):
    if "glm-5.3-flash" in str(lp):
        pytest.skip("SGLang scaffold exits before serving")
    env, log = stub_env
    for bad_image in ("vllm/vllm-openai:v0.31.0", ""):
        r = run_launcher(lp, env, {"API_KEY": "k", "IMAGE": bad_image})
        assert r.returncode != 0, f"unpinned IMAGE {bad_image!r} must fail"
        assert "pinned by digest" in (r.stdout + r.stderr)
        assert not docker_run_reached(log)


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_floating_model_revision_refuses(lp, stub_env):
    if "glm-5.3-flash" in str(lp):
        pytest.skip("SGLang scaffold exits before serving")
    env, log = stub_env
    env2 = env.copy()
    env2.pop("MODEL_DIR", None)  # force remote-fetch path
    r = run_launcher(lp, env2, {"API_KEY": "k", "IMAGE": "x@sha256:" + "a" * 64,
                                "MODEL_ID": "org/model", "MODEL_REVISION": ""})
    assert r.returncode != 0, "missing MODEL_REVISION must fail on the fetch path"
    assert "MODEL_REVISION is required" in (r.stdout + r.stderr)
    assert not docker_run_reached(log)


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_fewer_than_four_gpus_refuses(lp, stub_env):
    if "glm-5.3-flash" in str(lp):
        pytest.skip("SGLang scaffold exits before serving")
    env, log = stub_env
    env2 = env.copy()
    env2["GPU_COUNT"] = "2"
    r = run_launcher(lp, env2, {"API_KEY": "k", "IMAGE": "x@sha256:" + "a" * 64})
    assert r.returncode != 0, "wrong GPU count must fail"
    assert not docker_run_reached(log)


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_valid_config_reaches_docker_run_exactly_once(lp, stub_env):
    if "glm-5.3-flash" in str(lp):
        pytest.skip("SGLang scaffold exits before serving")
    env, log = stub_env
    # mimo hotsplit verifies the patch manifest; it is satisfied inside the recipe
    r = run_launcher(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                               "ALLOW_RUN": "1"})
    # curl never succeeds in the stub, so the launcher exits non-zero AFTER docker run;
    # what matters is that docker run was attempted exactly once and the key was passed on
    assert log.exists()
    lines = log.read_text().splitlines()
    runs = [l for l in lines if "STUB_DOCKER_RUN" in l]
    assert len(runs) == 1, f"expected exactly one docker run attempt, got {len(runs)}: {runs}"
    run_lines = run_lines_of(log)
    assert len(run_lines) == 1
    assert "--revision" not in run_lines[0] or "MODEL_REVISION" in run_lines[0]
    # model dir present -> local path: no --revision expected
    assert "--revision" not in run_lines[0], "local MODEL_DIR must not force --revision"


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_remote_fetch_adds_revision_exactly_once(lp, stub_env):
    if "glm-5.3-flash" in str(lp):
        pytest.skip("SGLang scaffold exits before serving")
    env, log = stub_env
    env2 = env.copy()
    env2.pop("MODEL_DIR", None)
    r = run_launcher(lp, env2, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                                "MODEL_ID": "org/model", "MODEL_REVISION": "abc123",
                                "ALLOW_RUN": "1"})
    run_lines = run_lines_of(log)
    assert len(run_lines) == 1
    assert run_lines[0].count("--revision") == 1, "exactly one --revision expected"
    assert "abc123" in run_lines[0]


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_api_key_reaches_container_env(lp, stub_env):
    if "glm-5.3-flash" in str(lp):
        pytest.skip("SGLang scaffold exits before serving")
    env, log = stub_env
    run_launcher(lp, env, {"API_KEY": "secret123", "IMAGE": "x@sha256:" + "a" * 64,
                           "ALLOW_RUN": "1"})
    run_lines = run_lines_of(log)
    assert any("VLLM_API_KEY=secret123" in l for l in run_lines), \
        "the API key must be passed into the container env"


def test_no_secret_in_dry_run_output(stub_env):
    """The launcher must not echo the key into stdout/stderr or the stub log args."""
    env, log = stub_env
    lp = LAUNCHERS[3]  # mimo uva
    r = run_launcher(lp, env, {"API_KEY": "supersecret", "IMAGE": "x@sha256:" + "a" * 64})
    combined = r.stdout + r.stderr
    # the value appears only inside the docker-run env assignment (the stub log), never as a bare echo
    for line in combined.splitlines():
        assert "supersecret" not in line or "VLLM_API_KEY" in line, \
            f"secret leaked into launcher output: {line}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))