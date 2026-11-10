"""Round 80b: resolve kv-persistence contract owners after the legacy kv
launcher (run-vllm-with-persist.sh) was deleted.

The engine launch is owned by the RECIPE launcher now (run-tp4-hotsplit.sh
with KV_PERSIST=1). The cross-repo contracts the old kv launcher held
(ports, identity pins, hotsplit mounts, same-UID, API key) live in the recipe
launcher + kv-persistence's shared modules (env.sh / hotsplit_env.sh). Tests
read that union so a contract can never silently vanish.
"""
from __future__ import annotations

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


def kv_repo() -> Path | None:
    for c in _KV_CANDIDATES:
        if c.is_dir():
            return c
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