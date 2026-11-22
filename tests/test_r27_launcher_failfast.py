"""Round 27 review K6: the engine launcher's fail-fast block must show the
WHOLE traceback and stop the engine container.

The hardware log (round 27 review) proved the defects:
* `grep -A20 ... | head -25` cut the LAST line of the exception
  (torch.AcceleratorError ... 205) -- the operator had to read the server log
  by hand. The block must print to the END of the log (all frames + the
  "The CUDA driver logged these messages" block).
* The "Fix (CUDA IPC boundary)" advice printed for EVERY error -- it belongs
  only to CUDA IPC failures (cuIpc / CUDA_ERROR_MAP_FAILED /
  "invalid device context").
* `exit 1` left the engine container ALIVE holding the GPUs; the next
  pre-flight "GPU memory is free" then blocked. The block must run
  `docker stop "$CONTAINER"` before exit (logs survive the stop).

Behavioural (the r90 C3 pattern): the fail-fast block is cut out of the REAL
launcher and executed against a stub docker + a REAL server-log fixture with
a 40-frame traceback. MUTATIONS caught: restoring `head -25`, dropping the
IPC-only condition, dropping `docker stop`.
"""
from __future__ import annotations
import os
import re
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIMO = ROOT / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit"
LAUNCHER = MIMO / "scripts" / "run-tp4-hotsplit.sh"

DOCKER_STUB = '''#!/usr/bin/env bash
echo "$*" >> "$STUB_LOG"
exit 0
'''

# 40-frame traceback in the LMCache server log: the frames, the exception
# line (the one the old code lost) and the CUDA driver block.
def _traceback_log(ipc: bool) -> str:
    # r29 N4: 40 frames (88-line block, OVER the 80-line bound) -- the
    # start+end limiter must keep `Returning 205` and the driver block
    # (a plain head -n 80 would cut exactly them). frame_39 = the last frame.
    frames = "".join(
        f"  File \"/opt/venv/lib/python3.12/site-packages/lmcache/x.py\", "
        f"line {100 + i}, in frame_{i}\n    do_step_{i}()\n"
        for i in range(40))
    if ipc:
        exc = ("torch.AcceleratorError: CUDA error: mapping of buffer object failed\n"
               "Failed to open/create interprocess memory file\n"
               "Returning 205 (CUDA_ERROR_MAP_FAILED) from cuIpcOpenEventHandle\n")
        driver = ("The CUDA driver logged these messages:\n"
                  "> CUDA_ERROR_MAP_FAILED: mapping of buffer object failed\n")
    else:
        exc = ("ValueError: bad adapter config\n")
        driver = ("The CUDA driver logged these messages:\n"
                  "> no driver-level error recorded\n")
    return (
        "2026-11-22 12:00:00 ERROR Error handling operation register_kv_cache\n"
        "Traceback (most recent call last):\n"
        + frames
        + exc
        + driver)


def _failfast_block() -> str:
    src = LAUNCHER.read_text()
    start = src.index('    _kvp_serr="$(tail -c "+$(( _kvp_slog_before + 1 ))"')
    end = src.index("      exit 1", start) + len("      exit 1")
    block = src[start:end]
    assert block, "the fail-fast block must exist in the launcher"
    return block


def _run_failfast(tmp_path, ipc_error=True):
    """Execute the REAL fail-fast block against a stub docker and a real
    server-log fixture. Returns (returncode, output, stub_log)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub_log = tmp_path / "stub.log"
    docker = bin_dir / "docker"
    docker.write_text(DOCKER_STUB)
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    slog = tmp_path / "lmcache-server.log"
    slog.write_text(_traceback_log(ipc_error))
    block = _failfast_block()
    script = (
        "_kvp_slog_before=0\n"
        f"_kvp_slog=\"{slog}\"\n"
        "CONTAINER=mimo-pro-tp4-hotsplit\n"
        "if [ -f \"$_kvp_slog\" ]; then\n"
        + block + "\n"
        "    fi\n"
        "fi\n"
        "exit 0\n"
    )
    env = dict(os.environ,
               PATH=f"{bin_dir}:{os.environ.get('PATH', '')}",
               STUB_LOG=str(stub_log))
    r = subprocess.run(["bash", "-c", script], capture_output=True,
                       text=True, timeout=30, env=env)
    return r, stub_log


def test_t6a_full_traceback_reaches_the_output(tmp_path):
    """> T6 / r29 N4: a 40-frame traceback (88 lines, OVER the bound) -> the
    LAST line of the exception (Returning 205 ... cuIpcOpenEventHandle) AND
    the "The CUDA driver logged these messages" block survive, and the
    `lines skipped` marker shows the cut. MUTATION caught: restoring
    `grep -A20 | head -25` or the plain `head -n 80` (both cut the tail)."""
    r, _ = _run_failfast(tmp_path, ipc_error=True)
    out = r.stdout + r.stderr
    assert r.returncode != 0, f"the fail-fast block must exit 1:\n{out[-800:]}"
    assert "Returning 205 (CUDA_ERROR_MAP_FAILED) from cuIpcOpenEventHandle" in out, (
        f"the LAST line of the exception must reach the output:\n{out[-1500:]}")
    assert "The CUDA driver logged these messages" in out, (
        f"the CUDA driver block must reach the output:\n{out[-1500:]}")
    # the first line of the traceback and the start+end marker
    assert "Error handling operation register_kv_cache" in out
    assert "lines skipped" in out, (
        f"the limiter marker must show the cut:\n{out[-1500:]}")


def test_t6b_ipc_advice_only_for_ipc_errors(tmp_path):
    """> T6: the ONE-UID advice prints for CUDA IPC errors and NOT for other
    errors. MUTATION caught: dropping the condition (advice always prints)."""
    r, _ = _run_failfast(tmp_path, ipc_error=True)
    out_ipc = r.stdout + r.stderr
    assert "must run under ONE UID" in out_ipc, out_ipc[-1200:]
    assert "docker inspect -f '{{.Config.User}}' lmcache-kvp-server" in out_ipc, (
        out_ipc[-1200:])
    r, _ = _run_failfast(tmp_path, ipc_error=False)
    out_other = r.stdout + r.stderr
    assert "must run under ONE UID" not in out_other, (
        f"a non-IPC error must NOT get the UID advice:\n{out_other[-1200:]}")
    assert "bad adapter config" in out_other, out_other[-1200:]


def test_t6c_engine_container_is_stopped(tmp_path):
    """> T6: `docker stop "$CONTAINER"` runs before exit 1 -- the engine must
    not hold the GPUs. MUTATION caught: dropping the stop."""
    r, stub_log = _run_failfast(tmp_path, ipc_error=True)
    log = stub_log.read_text()
    assert "stop mimo-pro-tp4-hotsplit" in log, (
        f"the engine container must be stopped on fail-fast:\n{log}")