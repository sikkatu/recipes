"""Cross-repo contract (review round 58, item 6.2): kv-persistence and the
mimo recipe must not drift apart silently. Runs offline (no GPU, no docker):
it only reads files from both repositories and compares the EFFECTIVE
contract values.

Checked:
  - L2 codec semantics: zstd default / none = explicit opt-out (both sides)
  - the lmcache version pin (0.5.6) is stated on both sides
  - the lmcache MP/ZMQ port default agrees
  - namespace pins (head_dim, num_kv_heads, kv cache dtype) agree with the
    identity config
  - every HotSplit file the kv launcher mounts exists in the recipe
"""
from __future__ import annotations
from pathlib import Path
import re

RECIPES = Path(__file__).resolve().parent.parent
# Round 61 (review): in GitHub Actions the second checkout lands INSIDE the
# recipes workspace ($GITHUB_WORKSPACE/kv-persistence) -- that layout must be
# tried FIRST, or the cross-repo tests skip in CI even though the checkout
# succeeded. Developer layouts (sibling / home) stay as fallbacks.
_KV_CANDIDATES = [
    RECIPES / "kv-persistence",                       # CI: second checkout in-workspace
    Path.home() / "kv-persistence",                   # developer home layout
    RECIPES.parent / "kv-persistence",                # sibling clones
    Path.home() / "sikkatu-kv-persistence",
]
KV_REPO = next((c for c in _KV_CANDIDATES if c.is_dir()), _KV_CANDIDATES[0])
RECIPE = RECIPES / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit"
SKIP_REASON = "kv-persistence clone not available next to recipes (cross-repo check)"

def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def test_codec_semantics_agree():
    """Both sides must fold the codec the same way: zstd default, none opt-out."""
    import pytest
    if not KV_REPO.is_dir():
        pytest.skip(SKIP_REASON)
    env_sh = _read(KV_REPO / "scripts" / "env.sh")
    ident = _read(KV_REPO / "implementation" / "kvpersist" / "identity.py")
    for name, txt in (("env.sh", env_sh), ("identity.py", ident)):
        assert "zstd" in txt
        # the shared semantic table: explicit raw opt-out maps to none
        assert "0|no|false|none" in txt or '"0", "no", "false", "none"' in txt, \
            f"{name} must keep the explicit-raw-opt-out mapping"
    # recipe launchers default to zstd too
    for ln in ("run-tp4-hotsplit.sh", "run-tp4-uva.sh"):
        t = _read(RECIPE / "scripts" / ln)
        assert 'LMCACHE_L2_SERDE=zstd' in t, \
            f"{ln} must default to lossless zstd like kv-persistence"
        assert 'USE_LOSSLESS_SERDE:-}' in t, \
            f"{ln}: the codec default must yield to the documented opt-out switches"

def test_lmcache_version_pin_agrees():
    import pytest
    if not KV_REPO.is_dir():
        pytest.skip(SKIP_REASON)
    pin = "0.5.6"
    kv_text = _read(KV_REPO / "requirements-lmcache.txt") if (KV_REPO / "requirements-lmcache.txt").is_file() else ""
    docs = _read(KV_REPO / "docs" / "deployment.md") if (KV_REPO / "docs" / "deployment.md").is_file() else ""
    rec = _read(RECIPE / "README.md")
    assert pin in kv_text or pin in docs, "kv-persistence must state the lmcache pin"
    assert pin in rec, "the recipe must state the same lmcache pin"

def test_ports_agree():
    import pytest
    if not KV_REPO.is_dir():
        pytest.skip(SKIP_REASON)
    kv = _read(KV_REPO / "scripts" / "run-vllm-with-persist.sh")
    m = re.search(r'LMCACHE_PORT=\$\{LMCACHE_PORT:-([0-9]+)\}', kv) or \
        re.search(r'LMCACHE_PORT:-([0-9]+)', kv)
    kv_port = m.group(1) if m else None
    rec = _read(RECIPE / "scripts" / "run-tp4-hotsplit.sh")
    m2 = re.search(r'LMCACHE_PORT:-([0-9]+)', rec)
    rec_port = m2.group(1) if m2 else None
    if kv_port and rec_port:
        assert kv_port == rec_port, f"lmcache port differs: kv={kv_port} recipe={rec_port}"

def test_namespace_pins_present():
    import pytest
    if not KV_REPO.is_dir():
        pytest.skip(SKIP_REASON)
    ident = _read(KV_REPO / "configs" / "mimo-v2.6-pro" / "identity.env")
    for pin in ("KVP_HEAD_DIM=", "KVP_NUM_KV_HEADS=", "KVP_KV_CACHE_DTYPE="):
        assert pin in ident, f"identity.env must pin {pin}"
    kv = _read(KV_REPO / "scripts" / "run-vllm-with-persist.sh")
    for var in ("MAX_MODEL_LEN", "MAX_NUM_SEQS"):
        assert var in kv, f"engine launcher must honour {var} from tp4.env"

def test_hotsplit_files_mounted_by_kv_launcher_exist():
    import pytest
    if not KV_REPO.is_dir():
        pytest.skip(SKIP_REASON)
    kv = _read(KV_REPO / "scripts" / "run-vllm-with-persist.sh")
    needed = re.findall(r'patches/([A-Za-z0-9_.]+\.py)', kv)
    needed += re.findall(r'patches/(expert_hist_mix\.json)', kv)
    assert needed, "kv launcher must mount the recipe hotsplit machinery"
    for name in sorted(set(needed)):
        assert (RECIPE / "patches" / name).is_file(), \
            f"kv launcher mounts patches/{name} but the recipe has no such file"

def test_hotsplit_module_shared():
    """Round 59 (P1-1): the HotSplit docker config lives in ONE module
    (scripts/hotsplit_env.sh) that BOTH launchers source. No hand-maintained
    duplicate lists that can drift."""
    import pytest
    mod = RECIPE / "scripts" / "hotsplit_env.sh"
    assert mod.is_file(), "the shared hotsplit env module must exist"
    txt = _read(mod)
    for var in ("HOTSPLIT_NUMA_MAP", "HOTSPLIT_LIVE_COUNTS", "HOTSPLIT_LIVE_SECS",
                "HOTSPLIT_DYN_STEPS", "HOTSPLIT_DYN_MAX_SWAPS", "HOTSPLIT_DYN_MARGIN",
                "HOTSPLIT_DYN_DECAY", "HOTSPLIT_DYN_PRIOR_TOKENS", "HOTSPLIT_MAX_TOKENS"):
        assert var in txt, f"the shared module must handle {var}"
    assert "IPC_LOCK" in txt and "SYS_NICE" in txt, "caps must be in the shared module"
    assert "FULL_DECODE_ONLY" in txt, "compilation config must be in the shared module"
    # Both launchers source the module
    for launcher in (RECIPE / "scripts" / "run-tp4-hotsplit.sh",):
        t = _read(launcher)
        assert "hotsplit_env.sh" in t, f"{launcher.name} must source the shared module"
    if not KV_REPO.is_dir():
        pytest.skip(SKIP_REASON)
    kv = _read(KV_REPO / "scripts" / "run-vllm-with-persist.sh")
    assert "hotsplit_env.sh" in kv, "the KV launcher must source the shared module"
    # The caps and compile arg must be wired into the docker run
    assert "HOTSPLIT_CAP_ARGS" in kv or "HOTSPLIT_DOCKER_CAPS" in kv, \
        "the KV launcher must pass the caps from the shared module"
    assert "HOTSPLIT_COMPILE" in kv, \
        "the KV launcher must pass the compilation config from the shared module"

def test_kvp_common_sh_copies_are_byte_identical():
    """Round 67 (review: one shell library): kvp-common.sh is copied per
    recipe (recipes must stay standalone), and the copies must be
    BYTE-IDENTICAL -- otherwise a fix lands in one recipe and not another
    (that is exactly what happened with the 12 kvp_curl copies)."""
    import pytest
    slugs = ("mimo-v2.6-pro-hotsplit", "deepseek-v4.1-flash-hotsplit",
             "glm-5.3-flash-hotsplit", "glm-5.3-hotsplit")
    blobs = {}
    for slug in slugs:
        f = RECIPES / "recipes" / "rtx-pro-6000" / slug / "scripts" / "kvp-common.sh"
        if f.is_file():
            blobs[slug] = f.read_bytes()
    assert len(blobs) >= 2, "kvp-common.sh must exist in the recipes"
    ref_slug, ref = next(iter(blobs.items()))
    for slug, blob in blobs.items():
        assert blob == ref, (
            f"kvp-common.sh in {slug} differs from {ref_slug} -- "
            "the shared helpers must be byte-identical across recipes")
    # and no recipe script may carry an inline copy any more
    for slug in slugs:
        sdir = RECIPES / "recipes" / "rtx-pro-6000" / slug / "scripts"
        if not sdir.is_dir():
            continue
        for sh in sdir.glob("*.sh"):
            if sh.name == "kvp-common.sh":
                continue
            text = sh.read_text()
            assert "kvp_curl(){" not in text, f"{sh}: inline kvp_curl copy (must source kvp-common.sh)"
            assert "_kvp_seed_env_file(){" not in text, f"{sh}: inline parser copy"


def test_env_sh_functions_match_kvp_common():
    """kv-persistence's scripts/env.sh carries the same two functions -- the
    bodies must stay semantically identical (same escape rules, same parser
    contract). Checked by the distinctive lines that encode the semantics."""
    import pytest
    if not KV_REPO.is_dir():
        pytest.skip(SKIP_REASON)
    env_sh = _read(KV_REPO / "scripts" / "env.sh")
    common = _read(RECIPES / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit"
                   / "scripts" / "kvp-common.sh")
    # the key-escaping contract (round 66/67): the _kvp_esc= lines must be
    # IDENTICAL in both files (they encode the \ and " escaping)
    def _esc_lines(text):
        return [ln.strip() for ln in text.splitlines() if ln.strip().startswith("_kvp_esc=")]
    esc_env = _esc_lines(env_sh)
    esc_common = _esc_lines(common)
    assert len(esc_env) == 2 and len(esc_common) == 2, (
        f"escape lines missing: env={esc_env}, common={esc_common}")
    assert esc_env == esc_common, (
        f"escape contract drifted: env={esc_env} vs common={esc_common}")
    # the parser contract: inline comments at " #", $HOME expansion, ident check
    for marker in ('%% #*', "^[A-Za-z_][A-Za-z0-9_]*$", "$HOME"):
        assert marker in env_sh, f"env.sh parser lost: {marker}"
        assert marker in common, f"kvp-common.sh parser lost: {marker}"
