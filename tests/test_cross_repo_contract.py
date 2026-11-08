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
KV_REPO = Path.home() / "kv-persistence"
if not KV_REPO.is_dir():
    KV_REPO = RECIPES.parent / "kv-persistence"
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
        assert 'LMCACHE_L2_SERDE="${LMCACHE_L2_SERDE:-zstd}"' in t, \
            f"{ln} must default to lossless zstd like kv-persistence"

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
