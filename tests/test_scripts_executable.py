"""Round 78 follow-up (user finding): every recipe script must be EXECUTABLE
(git mode 100755). check-hotsplit-tp4.sh and run-tp4-hotsplit.sh shipped as
100644 -- `bash scripts/x.sh` worked (bash doesn't need +x) but `./scripts/x.sh`
and any exec()ing caller failed with Permission denied."""
from __future__ import annotations
import os
import subprocess
from pathlib import Path

RECIPES = Path(__file__).resolve().parent.parent
SCRIPTS = RECIPES / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit" / "scripts"


def test_all_recipe_scripts_are_executable():
    missing = []
    for p in sorted(SCRIPTS.iterdir()):
        if p.suffix in (".sh", ".py"):
            if not os.access(p, os.X_OK):
                missing.append(p.name)
    assert not missing, (
        f"recipe scripts without the executable bit: {missing}")


def test_recipe_scripts_committed_with_exec_bit():
    """A fresh `git clone` restores the COMMITTED mode -- check the index."""
    r = subprocess.run(["git", "ls-files", "-s",
                        "recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit/scripts/"],
                       capture_output=True, text=True, cwd=str(RECIPES))
    bad = []
    for line in r.stdout.splitlines():
        mode, _sha, _stage, path = line.split(None, 3)
        if path.endswith((".sh", ".py")):
            if mode != "100755":
                bad.append(f"{mode} {path.split('/')[-1]}")
    assert not bad, (
        f"committed recipe script modes must be 100755, got: {bad}")