"""Round 80b: resolve kv-persistence contract owners after the legacy kv
launcher (run-vllm-with-persist.sh) was deleted.

The engine launch is owned by the RECIPE launcher now (run-tp4-hotsplit.sh
with KV_PERSIST=1). The cross-repo contracts the old kv launcher held
(ports, identity pins, hotsplit mounts, same-UID, API key) live in the recipe
launcher + kv-persistence's shared modules (env.sh / hotsplit_env.sh). Tests
read that union so a contract can never silently vanish.
"""
from __future__ import annotations
import shutil
import tempfile
from pathlib import Path
RECIPES = Path(__file__).resolve().parent.parent
RECIPE = RECIPES / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit"
_RECIPE_LAUNCHER = RECIPE / "scripts" / "run-tp4-hotsplit.sh"
_HOTSPLIT_ENV = RECIPE / "scripts" / "hotsplit_env.sh"
_KV_CANDIDATES = [
    RECIPES / "kv-persistence",
    Path.home() / "kv-persistence",
    RECIPES.parent / "kv-persistence",
]
# r25 G9: tests must never write into the OPERATOR's kv-persistence tree.
# kv_repo() returns a per-session TEMP COPY of the contract dirs (scripts/,
# configs/, implementation/, vendor/, observability/) -- the real launchers
# hard-code $ROOT/live, and running start-all.sh from the real checkout
# littered the real live/ (crash-*, model_dir, ...) and observability/extras/.
_KV_COPY: Path | None = None
# subdirs the launchers and gates need (the rest -- live/, docs, tests -- is
# not part of the launch chain)
_KV_COPY_DIRS = ("scripts", "configs", "implementation", "vendor", "observability")


def kv_repo() -> Path | None:
    global _KV_COPY
    for c in _KV_CANDIDATES:
        if c.is_dir():
            if _KV_COPY is None:
                tmp = Path(tempfile.mkdtemp(prefix="kvp-kv-copy-"))
                # r25 G9: copy only the launch-chain dirs (not .git -- its
                # pack files fail on FUSE). All writes (live/ state records,
                # extras) stay in this tmp copy.
                for d in ("scripts", "configs", "implementation", "vendor",
                          "observability", "live"):
                    src = c / d
                    if src.is_dir():
                        shutil.copytree(src, tmp / d, dirs_exist_ok=True)
                for f in c.iterdir():
                    if f.is_file() and f.name != ".git":
                        shutil.copy2(f, tmp / f.name)
                (tmp / "live").mkdir(exist_ok=True)
                _KV_COPY = tmp
            return _KV_COPY
    return None


def recipe_launcher() -> Path:
    return _RECIPE_LAUNCHER


def contract_text() -> str:
    """The union of live contract owners: the recipe launcher + the recipe's
    shared hotsplit module + kv-persistence's shared env.sh (when a checkout
    is present). Replaces reads of the deleted run-vllm-with-persist.sh."""
    parts = []
    if _RECIPE_LAUNCHER.is_file():
        parts.append("<<<run-tp4-hotsplit.sh>>>\n" + _RECIPE_LAUNCHER.read_text())
    if _HOTSPLIT_ENV.is_file():
        parts.append("<<<hotsplit_env.sh>>>\n" + _HOTSPLIT_ENV.read_text())
    kv = kv_repo()
    if kv is not None:
        env_sh = kv / "scripts" / "env.sh"
        if env_sh.is_file():
            parts.append("<<<kv env.sh>>>\n" + env_sh.read_text())
    return "\n".join(parts)