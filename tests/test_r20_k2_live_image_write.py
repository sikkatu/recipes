# SPDX-License-Identifier: MIT
"""r20 K2: live/image must be written ONLY after a successful custom-image
verify, and never on stock / KV_PERSIST=0 / KVP_SKIP_IMAGE_CHECK=1 runs.

Hardware defect (operator log 20 Nov 2026): run-tp4-hotsplit.sh wrote the
accepted IMAGE to $LIVE_DIR/image BEFORE verify-image-custom.sh. A run with
a stock digest + KV_PERSIST=1 correctly refused to launch, but live/image
(carrying the CUSTOM id) was overwritten with the stock digest -- the custom
image was "forgotten"; every following start-all.sh took the stock image
until the image was rebuilt. The comment next to the write promised "an
invalid IMAGE never lands in live/image", but a VALID stock digest passes the
format check and landed anyway.

Rule (r20 K2): only two places write live/image -- build-image.sh (after its
verify-gate) and the recipe launcher (after the successful deep check of the
custom image). The launcher write runs ONLY with KV_PERSIST=1 and
KVP_SKIP_IMAGE_CHECK!=1.

Tests drive the REAL launchers (run-tp4-hotsplit.sh and run-tp4-uva.sh) with
a docker stub (labels/identity of a real build) and a sentinel live/image:
  - stock refusal: live/image byte-identical (the hardware scenario);
  - custom success: exactly the verified id is written;
  - KV_PERSIST=0: file not created/changed;
  - KVP_SKIP_IMAGE_CHECK=1: file not changed;
  - all of the above for BOTH launchers.
Mutation: moving the printf back BEFORE the verify block -> the stock-refusal
test is RED.
"""
from __future__ import annotations
import hashlib
import json
import os
import pathlib
import shutil
import subprocess

import pytest

RECIPES = pathlib.Path(__file__).resolve().parent.parent
RECIPE = RECIPES / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit"
SCRIPTS = RECIPE / "scripts"

STOCK = ("vllm/vllm-openai:nightly-8e2beaa3bff595577a069d49d96bde9074294ec2"
         "@sha256:167ccd8696bc09f1b63301185134e04b883623f8adcbd8e3a0ec3140a193abcf")
CUSTOM = "sha256:" + "c12a95bf" + "0" * 56
SENTINEL = "sha256:" + "deadbeef" + "0" * 56   # what live/image holds beforehand


def _kv_repo() -> pathlib.Path | None:
    for c in (RECIPES / "kv-persistence",
              pathlib.Path.home() / "kv-persistence",
              RECIPES.parent / "kv-persistence"):
        if c.is_dir():
            return c
    return None


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _identity_json(kv_repo: pathlib.Path) -> str:
    """The baked identity of a real build-image.sh run (patch SHAs = the
    kv-persistence repo's real patch files, as the verify gate compares)."""
    pd = kv_repo / "vendor" / "patches" / "lmcache"
    return json.dumps({
        "module": "kvp-custom-lmcache-identity",
        "lmcache_source_commit": "07f1c6bb642f4772606a36eab730ae009ddce08f",
        "pr4136_patch_sha256": _sha(pd / "4136-mp-retrieve-abort.patch"),
        "pr5972_patch_sha256": _sha(pd / "5972-incomplete-kv-objects.patch"),
        "wheel": "lmcache-0.5.8rc1.post48+kvp.60dce759ea93-cp312-cp312-linux_x86_64.whl",
        "wheel_sha256": "a" * 64,
        "wheel_version": "0.5.8rc1.post48+kvp.60dce759ea93",
    })


def _docker_stub(bin_dir: pathlib.Path, kv_repo: pathlib.Path,
                 labels: dict) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    ident = bin_dir / "identity.json"
    ident.write_text(_identity_json(kv_repo))
    (bin_dir / "docker").write_text(f"""#!/usr/bin/env bash
cmd="${{1:-}}"; shift || true
case "$cmd" in
  image)
    sub="${{1:-}}"; shift || true
    case "$sub" in
      inspect) echo '{json.dumps(labels)}' ;;
      *) exit 0 ;;
    esac ;;
  run)
    _entry=""
    while [ "$#" -gt 0 ]; do case "$1" in
      --entrypoint) _entry="$2"; shift 2 ;;
      *) shift ;;
    esac; done
    case "$_entry" in
      cat) cat "{ident}" ;;
      python3) echo "0.5.8rc1.post48+kvp.60dce759ea93" ;;
      *) exit 0 ;;
    esac ;;
  inspect) echo '{CUSTOM}' ;;
  ps) echo "" ;;
  *) exit 0 ;;
esac
""")
    (bin_dir / "docker").chmod(0o755)
    # the launch chain probes nvidia-smi (GPU count) -- a test stand without
    # GPUs answers with a stub; the GPU behaviour is not the subject here
    (bin_dir / "nvidia-smi").write_text(
        '#!/usr/bin/env bash\ncase "${1:-}" in\n'
        '  -L) echo "GPU 0: NVIDIA RTX PRO 6000"; echo "GPU 1: NVIDIA RTX PRO 6000"; '
        'echo "GPU 2: NVIDIA RTX PRO 6000"; echo "GPU 3: NVIDIA RTX PRO 6000" ;;\n'
        '  *) echo "fake nvidia-smi" ;;\n'
        'esac\nexit 0\n')
    (bin_dir / "nvidia-smi").chmod(0o755)


def _run_launcher(tmp_path: pathlib.Path, launcher: str,
                  env_image: str | None, kv_persist: str = "1",
                  skip_check: str | None = None,
                  sentinel: str | None = SENTINEL,
                  script_override: pathlib.Path | None = None):
    """REAL launcher (or a mutant) with the docker stub; LIVE_DIR carries the
    sentinel live/image. Returns (proc, live_image_bytes_after)."""
    kv_repo = _kv_repo()
    if kv_repo is None:
        pytest.skip("SKIP (honest reason): kv-persistence checkout not found "
                    "(the verify gate lives there)")
    if not (SCRIPTS / launcher).is_file():
        pytest.skip(f"SKIP (honest reason): {launcher} not present")

    recipe_dir = tmp_path / "recipe"
    # a FULL recipe checkout (the launchers verify $RECIPE/patches/ and read
    # tp4.env) -- exactly what an operator has
    shutil.copytree(RECIPE, recipe_dir, dirs_exist_ok=True)
    if script_override is not None:
        shutil.copy(script_override, recipe_dir / "scripts" / launcher)

    live = tmp_path / "live"
    live.mkdir()
    if sentinel is not None:
        (live / "image").write_text(sentinel + "\n")

    bin_dir = tmp_path / "bin"
    pd = kv_repo / "vendor" / "patches" / "lmcache"
    labels = {
        "org.kvp.lmc-base": STOCK,
        "org.kvp.lmc-patch-4136": _sha(pd / "4136-mp-retrieve-abort.patch"),
        "org.kvp.lmc-patch-5972": _sha(pd / "5972-incomplete-kv-objects.patch"),
    }
    _docker_stub(bin_dir, kv_repo, labels)

    env = dict(os.environ,
               PATH=f"{bin_dir}:{os.environ.get('PATH','')}",
               KVP_REPO=str(kv_repo),
               KVP_LIVE_DIR=str(live),
               LIVE_DIR=str(live),
               KV_PERSIST=kv_persist,
               KVP_SKIP_IPC_TEST="1",
               KVP_PREFLIGHTS_DONE="1",
               # uva requires MODEL_REVISION (or MODEL_DIR) before the image
               # block -- a pinned SHA is what the recipe demands
               MODEL_REVISION="0" * 40,
               # the in-image site-packages probe is not the subject here --
               # the launcher's own supported escape hatch
               SITE="/usr/local/lib/python3.12/dist-packages")
    for var in ("IMAGE", "KVP_MODEL_CONFIG", "MODEL_DIR",
                "KVP_SKIP_LMC_IDENTITY"):
        env.pop(var, None)
    if env_image is not None:
        env["IMAGE"] = env_image
    if skip_check is not None:
        env["KVP_SKIP_IMAGE_CHECK"] = skip_check

    r = subprocess.run(["bash", str(recipe_dir / "scripts" / launcher)],
                       capture_output=True, text=True, timeout=120,
                       env=env, cwd=str(tmp_path))
    after = (live / "image").read_bytes() if (live / "image").exists() else None
    return r, after


LAUNCHERS = ("run-tp4-hotsplit.sh", "run-tp4-uva.sh")


def test_stock_refusal_leaves_live_image_untouched(tmp_path):
    """THE hardware scenario: stock digest + KV_PERSIST=1 -> the launch is
    refused, but live/image must stay byte-identical (it used to be
    overwritten with the stock digest)."""
    for launcher in LAUNCHERS:
        sub = tmp_path / launcher.replace(".sh", "")
        sub.mkdir()
        before = (SENTINEL + "\n").encode()
        r, after = _run_launcher(sub, launcher, env_image=STOCK)
        assert after == before, (
            f"{launcher}: live/image must stay byte-identical on a stock "
            f"refusal (was {before!r}, now {after!r})\n"
            f"out:\n{(r.stdout + r.stderr)[-800:]}")


def test_custom_success_writes_verified_id(tmp_path):
    """Custom image + KV_PERSIST=1 -> exactly the verified id is written."""
    for launcher in LAUNCHERS:
        sub = tmp_path / launcher.replace(".sh", "")
        sub.mkdir()
        r, after = _run_launcher(sub, launcher, env_image=CUSTOM,
                                 sentinel=SENTINEL)
        assert after == (CUSTOM + "\n").encode(), (
            f"{launcher}: live/image must record the verified custom id "
            f"(got {after!r})\nout:\n{(r.stdout + r.stderr)[-800:]}")


def test_kv_persist_0_never_writes(tmp_path):
    """KV_PERSIST=0 (the README's stock runs) -> the file is not written at
    all (no confirmation exists to record)."""
    for launcher in LAUNCHERS:
        sub = tmp_path / launcher.replace(".sh", "")
        sub.mkdir()
        before = (SENTINEL + "\n").encode()
        r, after = _run_launcher(sub, launcher, env_image=STOCK,
                                 kv_persist="0")
        assert after == before, (
            f"{launcher}: KV_PERSIST=0 must not touch live/image "
            f"(was {before!r}, now {after!r})")
        # and with no file at all it must not CREATE one
        sub2 = tmp_path / (launcher.replace(".sh", "") + "_nofile")
        sub2.mkdir()
        r2, after2 = _run_launcher(sub2, launcher, env_image=STOCK,
                                   kv_persist="0", sentinel=None)
        assert after2 is None, (
            f"{launcher}: KV_PERSIST=0 must not create live/image "
            f"(got {after2!r})")


def test_skip_image_check_never_writes(tmp_path):
    """KVP_SKIP_IMAGE_CHECK=1 (test harness) -> the file is not changed:
    the check that CONFIRMS the id was skipped, so there is nothing to
    record."""
    for launcher in LAUNCHERS:
        sub = tmp_path / launcher.replace(".sh", "")
        sub.mkdir()
        before = (SENTINEL + "\n").encode()
        r, after = _run_launcher(sub, launcher, env_image=CUSTOM,
                                 skip_check="1")
        assert after == before, (
            f"{launcher}: KVP_SKIP_IMAGE_CHECK=1 must not touch live/image "
            f"(was {before!r}, now {after!r})")


def test_mutation_write_before_verify_is_red(tmp_path):
    """Mutation: restoring the OLD order (the pre-r20 image resolve + the
    write BEFORE the verify block) must flip the stock-refusal scenario: the
    sentinel gets overwritten with the STOCK digest even though the launch
    is refused. NOTE the r20 K1 resolver runs first and turns the stock pin
    into the custom id -- so the mutant must carry BOTH regressions (no K1
    resolve + the early write) to reproduce the exact hardware defect."""
    kv_repo = _kv_repo()
    if kv_repo is None:
        pytest.skip("SKIP (honest reason): kv-persistence checkout not found")
    src = (SCRIPTS / "run-tp4-hotsplit.sh").read_text()
    # (1) drop the K1 resolve call -- the stock IMAGE stays stock
    k1_call = ('    _kvp_resolve_runtime_image "$_kvp_repo" \\\n'
               '      || die "runtime image resolution failed (see above).\n'
               'Fix: comment out IMAGE= in tp4.env (or unset IMAGE in the '
               'environment), or run\n'
               '  IMAGE=\\"\\$(cat ${KVP_LIVE_DIR:-$_kvp_repo/live}/image)\\" $0"')
    if k1_call in src:
        mutant = src.replace(k1_call, "    :", 1)
    else:
        # fall back: any _kvp_resolve_runtime_image call inside the launcher
        mutant = src.replace('_kvp_resolve_runtime_image "$_kvp_repo"', ':', 1)
    assert mutant != src, "K1 resolve call not found in run-tp4-hotsplit.sh"
    # (2) move the K2 write back BEFORE the verify block (the old order)
    k2_block = ('if [ "${KV_PERSIST:-0}" = "1" ] && '
                '[ "${KVP_SKIP_IMAGE_CHECK:-0}" != "1" ] && [ -n "$IMAGE" ]; then\n'
                '  printf \'%s\\n\' "$IMAGE" > "$LIVE_DIR/image" 2>/dev/null || true\n'
                'fi\n')
    assert k2_block in mutant, "K2 write block not found"
    mutant = mutant.replace(k2_block, '', 1)
    verify_anchor = ('KVP_REPO_FOR_CHECK="${KVP_REPO:-'
                     '${KVP_REPO_ROOT:-$HOME/kv-persistence}}"')
    assert verify_anchor in mutant, "verify anchor not found"
    mutant = mutant.replace(
        verify_anchor,
        'if [ -n "$IMAGE" ]; then\n'
        'printf \'%s\\n\' "$IMAGE" > "$LIVE_DIR/image" 2>/dev/null || true\n'
        'fi\n' + verify_anchor, 1)
    m = tmp_path / "run-tp4-hotsplit-mutant.sh"
    m.write_text(mutant)
    m.chmod(0o755)

    sub = tmp_path / "mutant"
    sub.mkdir()
    before = (SENTINEL + "\n").encode()
    r, after = _run_launcher(sub, "run-tp4-hotsplit.sh", env_image=STOCK,
                             script_override=m)
    assert after != before, (
        "mutation detector: with the old order (stock IMAGE kept + the write "
        f"BEFORE the verify block) the sentinel must be OVERWRITTEN with the "
        f"stock digest on the refusal (got {after!r})\n"
        f"out:\n{(r.stdout + r.stderr)[-600:]}")