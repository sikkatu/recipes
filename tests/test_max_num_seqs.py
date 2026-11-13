"""Part A (round 87): MAX_NUM_SEQS default 2 -> 10 for the MiMo recipe.

BEHAVIOURAL tests (review rule: a grep-only test is not a test). The launcher
scripts are actually EXECUTED against stub `docker`/`nvidia-smi`/`curl`
(the pattern of tests/test_launcher_behaviour.py) and the recorded
`docker run` line is compared. Every test names the MUTATION it must catch.

Part 1 (this file): launcher behaviour + env examples + other-recipes guard.
"""
from __future__ import annotations
import hashlib
import os
import re
import stat
import subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "recipes" / "rtx-pro-6000"
MIMO = BASE / "mimo-v2.6-pro-hotsplit"
LAUNCHERS = [MIMO / "scripts" / "run-tp4-hotsplit.sh",
             MIMO / "scripts" / "run-tp4-uva.sh"]
OTHER_L = [BASE / "glm-5.3-hotsplit" / "scripts" / "run-tp4-uva.sh",
           BASE / "glm-5.3-flash-hotsplit" / "scripts" / "run-tp4-uva.sh",
           BASE / "deepseek-v4.1-flash-hotsplit" / "scripts" / "run-tp4-uva.sh"]

STUB = r'''#!/usr/bin/env bash
echo "$0 $*" >> "$STUB_LOG"
case "$(basename "$0")" in
  docker)
    case "$1" in
      info) exit 0 ;;
      container) exit 1 ;;
      inspect) echo "running exit=0 oom=false"; exit 0 ;;
      logs) echo stub-logs; exit 0 ;;
      run)
        case "$*" in
          *--rm*) echo "STUB_DOCKER_RUN_RM" >> "$STUB_LOG"; echo "/usr/local/lib/python3.12/dist-packages"; exit 0 ;;
        esac
        echo "STUB_DOCKER_RUN" >> "$STUB_LOG"; exit 0 ;;
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
    b = tmp_path / "bin"; b.mkdir()
    log = tmp_path / "stub.log"
    for n in ("docker", "nvidia-smi", "curl", "sha256sum"):
        p = b / n
        p.write_text(STUB); p.chmod(p.stat().st_mode | stat.S_IEXEC)
    home = tmp_path / "home"; home.mkdir()
    model = tmp_path / "model"; model.mkdir()
    env = os.environ.copy()
    env["PATH"] = f"{b}:{env['PATH']}"
    env["STUB_LOG"] = str(log)
    env["HOME"] = str(home)
    env["KVP_API_KEY_FILE"] = str(home / ".config" / "kvp" / "api_key")
    env["MODEL_DIR"] = str(model)
    env["HF_CACHE"] = str(tmp_path / "hf")
    env["LIVE_DIR"] = str(tmp_path / "live")
    env["WAIT_TRIES"] = "1"
    env["WAIT_SLEEP"] = "0"
    env["KVP_SKIP_HOTSPLIT_CHECK"] = "1"
    env["API_KEY"] = "secret123"
    env["IMAGE"] = "x@sha256:" + "a" * 64
    return env, log, tmp_path

def _run(lp: Path, env, log, extra: dict | None = None):
    if log.exists():
        log.unlink()
    e = env.copy(); e.update(extra or {})
    return subprocess.run(["bash", str(lp)], env=e,
                          capture_output=True, text=True, timeout=90)

def _run_line(log: Path) -> str:
    for l in log.read_text().splitlines():
        if "docker run " in l and "--rm" not in l:
            return l.strip()
    return ""

def _seqs(line: str) -> str | None:
    m = re.search(r"--max-num-seqs (\S+)", line)
    return m.group(1) if m else None


# ---------------- default-consistency (mutation: 2 stays in a launcher) ----
@pytest.mark.parametrize("lp", LAUNCHERS)
def test_default_is_10_without_variable(lp, stub_env):
    """No env var -> --max-num-seqs 10. MUTATION caught: 2 left in either
    launcher (or KVP_DEFAULT_MAX_NUM_SEQS not consumed)."""
    env, log, _ = stub_env
    r = _run(lp, env, log)
    line = _run_line(log)
    assert _seqs(line) == "10", f"default must be 10, got {_seqs(line)}\n{r.stdout[-400:]}"

@pytest.mark.parametrize("lp", LAUNCHERS)
def test_explicit_value_wins(lp, stub_env):
    """MAX_NUM_SEQS=3 -> 3. MUTATION caught: hardcoding 10 over an explicit
    value."""
    env, log, _ = stub_env
    r = _run(lp, env, log, {"MAX_NUM_SEQS": "3"})
    assert _seqs(_run_line(log)) == "3", f"explicit 3 must win:\n{(r.stdout+r.stderr)[-400:]}"

@pytest.mark.parametrize("bad", ["0", "-1", "abc", ""])
@pytest.mark.parametrize("lp", LAUNCHERS)
def test_garbage_is_refused_before_docker_run(lp, bad, stub_env):
    """MAX_NUM_SEQS=<garbage> must die with ERROR before any docker run.
    MUTATION caught: removing _kvp_validate_posint (garbage reaches the flag)."""
    env, log, _ = stub_env
    r = _run(lp, env, log, {"MAX_NUM_SEQS": bad})
    assert r.returncode != 0, f"{bad!r} must be refused (rc={r.returncode})"
    assert "ERROR" in (r.stdout + r.stderr), f"expected an ERROR message:\n{(r.stdout+r.stderr)[-400:]}"
    assert not log.exists() or "docker run" not in log.read_text(), "no docker run may happen"

# ---------------- seeded-env (mutation: leftover 2 in tp4.env.example) ----
@pytest.mark.parametrize("lp", LAUNCHERS)
def test_seeded_env_gives_10(lp, stub_env):
    """tp4.env seeded from tp4.env.example -> the run uses 10.
    MUTATION caught: MAX_NUM_SEQS=2 still in tp4.env.example."""
    env, log, tmp = stub_env
    import shutil
    shutil.copy(MIMO / "tp4.env.example", tmp / "src")
    # seed like the launcher does: copy into the recipe dir as tp4.env
    tp4 = tmp / "recipe"; tp4.mkdir()
    shutil.copy(MIMO / "tp4.env.example", tp4 / "tp4.env")
    r = _run(lp, env, log, {"TP4_ENV": str(tp4 / "tp4.env")})
    assert _seqs(_run_line(log)) == "10", (
        f"seeded tp4.env must give 10:\n{(r.stdout+r.stderr)[-400:]}")

# ---------------- old-pin-note (mutation: file edited / NOTE twice) -------
# L1 (review 88): BOTH MiMo launchers are exercised -- a mutation that drops
# the NOTE call from run-tp4-uva.sh alone must ALSO be caught.
@pytest.mark.parametrize("launcher_name",
                         ["run-tp4-hotsplit.sh", "run-tp4-uva.sh"])
def test_old_pin_note_keeps_file_and_value(stub_env, launcher_name):
    """Copy the recipe dir into the sandbox, write tp4.env with the old pin
    MAX_NUM_SEQS=2, then run the launcher: one NOTE + the file is
    byte-unchanged + the run still uses 2 (user choice preserved).
    MUTATION caught: launcher rewrites tp4.env, edits it, or prints NOTE
    more than once per run; L1 mutation: __Call removed in one launcher__."""
    env, log, tmp = stub_env
    import shutil
    sandbox = tmp / ("recipe_" + launcher_name.replace(".sh", "").replace("-", "_"))
    shutil.copytree(MIMO, sandbox / "mimo-v2.6-pro-hotsplit")
    tp4 = sandbox / "mimo-v2.6-pro-hotsplit" / "tp4.env"
    tp4.write_text("MAX_NUM_SEQS=2   # old pin\n")
    h_before = hashlib.sha256(tp4.read_bytes()).hexdigest()
    r = _run(sandbox / "mimo-v2.6-pro-hotsplit" / "scripts" / launcher_name,
             env, log, {})
    out = r.stdout + r.stderr
    assert hashlib.sha256(tp4.read_bytes()).hexdigest() == h_before, \
        f"{launcher_name}: tp4.env must NEVER be edited"
    assert out.count("tp4.env pins MAX_NUM_SEQS=2") == 1, \
        f"{launcher_name}: exactly one NOTE expected:\n{out[-500:]}"
    assert _seqs(_run_line(log)) == "2", \
        f"{launcher_name}: user's 2 must stay 2, got {_seqs(_run_line(log))}"

def test_note_helper_speaks_once(stub_env):
    """Direct execution of _kvp_note_old_default against a file that pins the
    old default: ONE NOTE per call. MUTATION caught: printed twice per call
    or the helper edits the input file."""
    env, log, tmp = stub_env
    common = MIMO / "scripts" / "kvp-common.sh"
    tp4 = tmp / "tp4.env"
    tp4.write_text("MAX_NUM_SEQS=2  # old default\n")
    h_before = hashlib.sha256(tp4.read_bytes()).hexdigest()
    script = ("set -e\n. %s\n_execute() { _kvp_note_old_default MAX_NUM_SEQS 2 10 %s\n }\n_execute\n_execute\n%s"
              % (str(common), str(tp4), 'echo DONE'))
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=15)
    assert r.returncode == 0, r.stdout + r.stderr
    # spec: "one NOTE per run", implemented as per-call; the launcher calls
    # the helper exactly ONCE per run -> assert exactly 2 here (2 calls).
    assert r.stderr.count("tp4.env pins MAX_NUM_SEQS=2") == 2, (
        f"exactly one NOTE per helper call expected; got:\n{r.stderr}")
    assert hashlib.sha256(tp4.read_bytes()).hexdigest() == h_before, \
        "the helper must never edit the input file"

def test_note_helper_silent_for_new_default(stub_env):
    """A tp4.env with the NEW default (10) must NOT trigger a NOTE.
    MUTATION caught: NOTE printed regardless of value."""
    env, log, tmp = stub_env
    common = MIMO / "scripts" / "kvp-common.sh"
    tp4 = tmp / "tp4.env"
    tp4.write_text("MAX_NUM_SEQS=10\n")
    script = "set -e\n. %s\n_kvp_note_old_default MAX_NUM_SEQS 2 10 %s\n" % (str(common), str(tp4))
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=15)
    assert r.returncode == 0
    assert r.stderr.count("tp4.env pins MAX_NUM_SEQS=2") == 0, f"no NOTE expected for value=10:\n{r.stderr}"

# ---------------- three-env-examples (mutation: 2 in one locale) ----------
def test_three_env_examples_agree_on_10():
    """All three tp4.env.example say MAX_NUM_SEQS=10 and agree with
    KVP_DEFAULT_MAX_NUM_SEQS. MUTATION caught: 2 left in one locale."""
    common = (MIMO / "scripts" / "kvp-common.sh").read_text()
    m = re.search(r"KVP_DEFAULT_MAX_NUM_SEQS=(\S+)", common)
    assert m and m.group(1) == "10", "kvp-common.sh must pin KVP_DEFAULT_MAX_NUM_SEQS=10"
    for ext in ("", ".ru", ".zh"):
        txt = (MIMO / f"tp4.env.example{ext}").read_text()
        m2 = re.search(r"^MAX_NUM_SEQS=(\S+)", txt, re.M)
        assert m2, f"MAX_NUM_SEQS assignment missing in tp4.env.example{ext}"
        assert m2.group(1) == "10", f"tp4.env.example{ext}: expected 10, got {m2.group(1)}"

# ---------------- other-recipes-guard (mutation: touching glm/deepseek) ---
@pytest.mark.parametrize("lp", OTHER_L)
def test_other_recipes_keep_1(lp, stub_env):
    """glm / deepseek recipes keep --max-num-seqs 1. MUTATION caught:
    changing one of them."""
    env, log, _ = stub_env
    r = _run(lp, env, log)
    line = _run_line(log)
    assert _seqs(line) == "1", f"non-MiMo recipe must keep 1, got {_seqs(line)}"


# ---------------- golden-fixture (mutation: stale 2 in the fixture) -------
def test_golden_fixture_matches_shared_default():
    """tests/fixtures/launcher-docker-run.txt must carry the SAME value as
    KVP_DEFAULT_MAX_NUM_SEQS in kvp-common.sh (one source of truth).
    MUTATION caught: fixture left at --max-num-seqs 2 after the change
    (or the fixture hardcodes a number that drifted from kvp-common.sh)."""
    common = (MIMO / "scripts" / "kvp-common.sh").read_text()
    m = re.search(r"KVP_DEFAULT_MAX_NUM_SEQS=(\S+)", common)
    assert m, "kvp-common.sh must define KVP_DEFAULT_MAX_NUM_SEQS"
    want = m.group(1)
    fixture = ROOT / "tests" / "fixtures" / "launcher-docker-run.txt"
    expect = [ln for ln in fixture.read_text().splitlines()
              if ln and not ln.startswith("#")]
    assert len(expect) == 1, "the fixture must hold exactly one docker run line"
    assert _seqs(expect[0]) == want, (
        f"fixture says --max-num-seqs {_seqs(expect[0])} but kvp-common.sh pins {want}")

# ---------------- unchanged-flags (mutation: a neighbour flag moves) -----
def test_only_max_num_seqs_flag_changed_against_fixture(stub_env):
    """Versus the recorded contract the ONLY change is --max-num-seqs.
    MUTATION caught: an accidental change of --max-num-batched-tokens,
    --gpu-memory-utilization, --block-size or the cudagraph flags alongside."""
    env, log, tmp = stub_env
    r = _run(LAUNCHERS[0], env, log)
    line = _run_line(log)
    fixture = ROOT / "tests" / "fixtures" / "launcher-docker-run.txt"
    expect = [ln for ln in fixture.read_text().splitlines()
              if ln and not ln.startswith("#")][0]
    # normalize the same way the fixture test does
    for tok in ("--max-num-batched-tokens", "--gpu-memory-utilization",
                "--block-size", "cudagraph_mode"):
        m1 = re.search(re.escape(tok) + r"[ =]\S+", line)
        m2 = re.search(re.escape(tok) + r"[ =]\S+", expect)
        if m2:
            assert m1 and m1.group(0) == m2.group(0), (
                f"{tok} changed: {m2.group(0)} -> {m1.group(0) if m1 else 'gone'}")
