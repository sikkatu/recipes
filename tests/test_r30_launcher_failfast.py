"""Round 30 P3: the launcher fail-fast block must survive the REAL strict
header under SIGPIPE pressure.

The r26/r29 test ran the block WITHOUT `set -Eeuo pipefail` and with a log
that ENDED at the driver block -- so an early `exit` in the first awk kept all
tests green, while the real launcher on a 3.8 MB log died with rc 141 (tail
SIGPIPE) BEFORE printing the traceback or calling `docker stop`.

These tests run the EXACT block of the real launcher under the REAL
`set -Eeuo pipefail` header (taken from the launcher itself) with a 40-frame
traceback followed by 5 MB of DEBUG noise -- the fail-fast path must print the
block, keep `frame_39` and the marker, never the noise, stop the container
and exit 1 (not 141, not silent). Also asserts the MARKER NUMBER (a
`NR - max` -> `NR - hd` swap used to pass everything).

The second launcher place (_kvp_errline branch) reads its input as a FILE --
`exit` there is equivalent (no writer to SIGPIPE), documented in the r30
commit; that block is still run for the marker number.
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
def _write_stub(bin_dir: Path, name: str, text: str):
    p = bin_dir / name
    p.write_text(text)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p

def _traceback_block_40() -> str:
    """40-frame traceback (frame_0..frame_39) + exception + driver block."""
    frames = "".join(
        f'  File "/opt/venv/lib/python3.12/site-packages/lmcache/x.py", '
        f"line {100 + i}, in frame_{i}\n    do_step_{i}()\n"
        for i in range(40))
    exc = ("torch.AcceleratorError: CUDA error: mapping of buffer object failed\n"
           "Failed to open/create interprocess memory file\n"
           "Returning 205 (CUDA_ERROR_MAP_FAILED) from cuIpcOpenEventHandle\n")
    driver = ("The CUDA driver logged these messages:\n"
              "> CUDA_ERROR_MAP_FAILED: mapping of buffer object failed\n")
    return ("[2026-11-22 12:00:00] ERROR Error handling operation register_kv_cache\n"
            "Traceback (most recent call last):\n" + frames + exc + driver)

def _noise(mb: int = 5) -> str:
    unit = "[2026-11-22 12:01:00] LMCache DEBUG skipping eviction L2 usage 0.00\n"
    return unit * int(mb * 1024 * 1024 / len(unit))

def _strict_header() -> str:
    """The REAL `set -Eeuo pipefail` header line of the launcher (P3
    requirement -- the block runs exactly like in the launcher)."""
    return next(line for line in LAUNCHER.read_text().splitlines()
                if line.startswith("set -Eeuo"))

def _first_block() -> str:
    """The REAL first fail-fast block of the launcher (the `_kvp_serr` one --
    the SIGPIPE-critical place: its selector reads a PIPE from tail)."""
    src = LAUNCHER.read_text()
    start = src.index('    _kvp_serr="$(tail -c "+$(( _kvp_slog_before + 1 ))"')
    end = src.index("      exit 1", start) + len("      exit 1")
    block = src[start:end]
    assert "max=80" in block, "stale launcher (no r29 N4 bounded limiter)"
    return block

def _second_block() -> str:
    """The REAL second fail-fast block of the launcher (the `_kvp_errline`
    branch, the dead-engine diagnostics). Reads its input as an awk FILE --
    `exit` there is equivalent (no writer to SIGPIPE), so this block is
    exercised for the content + the MARKER NUMBER only (documented in the
    r30 commit)."""
    src = LAUNCHER.read_text()
    # the WHOLE dead-engine server-log branch (balanced if/fi), which contains
    # the 2nd awk selection (`_kvp_errline`) and ends at `exit 1 ;;`
    start = src.index('      if [ -f "$_kvp_log" ]; then')
    end = src.index("exit 1 ;;", start)
    block = src[start:end] + "exit 1\n"   # the trailing ` ;;` is the case's
                                          # terminator -- drop it (no case here)
    assert "max=80" in block, "stale launcher (no r29 N4 bounded limiter)"
    return block

def _run_first_block(tmp_path: Path, log_text: str):
    """Run the REAL first block under the REAL strict header: the fixture log
    (40-frame block + 5 MB DEBUG noise), stub docker. Returns CompletedProcess
    and the stub log path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub_log = tmp_path / "stub.log"
    _write_stub(bin_dir, "docker", DOCKER_STUB)
    slog = tmp_path / "lmcache-server.log"
    with open(slog, "w") as fh:
        fh.write(log_text)
    script = (
        _strict_header() + "\n"
        "_kvp_slog_before=0\n"
        f'_kvp_slog="{slog}"\n'
        "CONTAINER=mimo-pro-tp4-hotsplit\n"
        "if [ -f \"$_kvp_slog\" ]; then\n"
        + _first_block() + "\n"
        "    fi\n"
        "fi\n"
        "exit 0\n")
    env = dict(os.environ,
               PATH=f"{bin_dir}:{os.environ.get('PATH', '')}",
               STUB_LOG=str(stub_log))
    r = subprocess.run(["bash", "-c", script], capture_output=True,
                       text=True, timeout=120, env=env)
    return r, stub_log

# ---------------------------------------------------------------------------
# P3: the first launcher block under the REAL strict header + 5 MB of noise
# ---------------------------------------------------------------------------
def test_p3_first_block_sigpipe_full_traceback_stops_engine(tmp_path):
    """> P3: 40-frame traceback (87-line block, OVER the 80 bound) + 5 MB
    DEBUG noise, the REAL block of run-tp4-hotsplit.sh under its REAL
    `set -Eeuo pipefail` -> `Returning 205 ... cuIpcOpenEventHandle` and
    `frame_39` print, the `lines skipped` marker shows, ZERO noise, at most
    81 lines from the block, `docker stop` was called, rc=1 (NOT 141, NOT
    silent). MUTATIONS: the early `exit` in the first awk (tail SIGPIPE ->
    rc 141, empty output); `NR - max` -> `NR - hd` in the inlined limiter
    (marker says 20 not 8)."""
    log = _traceback_block_40() + _noise(5)
    r, stub_log = _run_first_block(tmp_path, log)
    out = r.stdout + r.stderr
    assert r.returncode == 1, (
        f"fail-fast must exit 1 (rc={r.returncode} means SIGPIPE death or a leak):\n{out[-1500:]}")
    assert "Returning 205 (CUDA_ERROR_MAP_FAILED) from cuIpcOpenEventHandle" in out, (
        f"the LAST exception line must print:\n{out[-1500:]}")
    assert "frame_39" in out, (
        f"the LAST frame must survive the bound:\n{out[-1500:]}")
    assert "The CUDA driver logged these messages" in out, out[-1500:]
    assert "... (7 lines skipped) ..." in out, (
        f"the marker must count the skipped lines (block 87 lines, 7 dropped):\n{out[-1500:]}")
    assert "skipping eviction" not in out, f"DEBUG noise must be cut:\n{out[-1500:]}"
    # the printed trace = head 20 + marker + tail 60 = exactly 81 lines
    t_start = out.index("Error handling operation register_kv_cache")
    t_end = out.index("This is the root cause")
    trace_lines = out[t_start:t_end].strip().splitlines()
    assert len(trace_lines) == 81, (
        f"the bounded trace must be exactly 81 lines, got {len(trace_lines)}:\n{trace_lines[:3]}\n...\n{trace_lines[-3:]}")
    assert stub_log.exists(), "docker stop must be reached"
    assert "stop mimo-pro-tp4-hotsplit" in stub_log.read_text(), stub_log.read_text()

def test_p3_second_block_marker_number(tmp_path):
    """> P3: the second launcher place (_kvp_errline branch) prints the same
    bounded trace with the CORRECT marker number. It reads its input as an
    awk FILE (no pipe writer), so an `exit` there is equivalent -- the block
    is exercised for the marker arithmetic: `NR - max` -> `NR - hd` must be
    red here (marker says 20, not 7)."""
    block = _second_block()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub_log = tmp_path / "stub.log"
    _write_stub(bin_dir, "docker", DOCKER_STUB)
    slog = tmp_path / "lmcache-server.log"
    slog.write_text(_traceback_block_40())
    logf = str(slog)
    script = (
        _strict_header() + "\n"
        f'_kvp_log="{logf}"\n'
        "_kvp_pidf=/nonexistent.pid\n"
        "CONTAINER=mimo-pro-tp4-hotsplit\n"
        "if [ -n \"$_kvp_log\" ]; then\n"
        + block + "\n"
        "fi\n"
        "exit 0\n")
    env = dict(os.environ,
               PATH=f"{bin_dir}:{os.environ.get('PATH', '')}",
               STUB_LOG=str(stub_log))
    r = subprocess.run(["bash", "-c", script], capture_output=True,
                       text=True, timeout=60, env=env)
    out = r.stdout + r.stderr
    assert r.returncode == 1, (
        f"the _kvp_errline branch must exit 1:\n{out[-1500:]}")
    assert "... (7 lines skipped) ..." in out, (
        f"the marker must count 87-80=7 skipped lines:\n{out[-1500:]}")
    assert "Returning 205 (CUDA_ERROR_MAP_FAILED) from cuIpcOpenEventHandle" in out, (
        f"the LAST exception line must print:\n{out[-1500:]}")
    assert "frame_39" in out, out[-1500:]

def test_p3_first_selector_no_early_exit():
    """> P3: the first launcher selector must use the done FLAG (the early
    `exit` SIGPIPE'd tail under `set -Eeuo pipefail` and killed the script
    with rc 141 before docker stop). MUTATION: `exit` in the first awk."""
    block = _first_block()
    assert "done=1" in block, "the first selector must use the done flag:\n" + block
    assert "f && /^\\[20[0-9][0-9]-/ {exit}" not in block.replace(" ", " ") , (
        "the early-exit selector must be gone:\n" + block)