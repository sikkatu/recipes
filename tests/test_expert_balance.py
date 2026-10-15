"""Tests for the three non-MiMo TP4 expert-balance recipes.

What is locked here:
  1. the shared planning core (hotsplit_plan.py) is byte-identical across all four
     rtx-pro-6000 hotsplit recipes -- one balance algorithm, not four drifting copies;
  2. hotsplit_balance.py computes the published hot/cold split from geometry alone
     (bank = layers x experts x cell, hot + cold = bank, integer cell counts);
  3. each recipe carries the files it claims (balance.json, calculator, launcher) and
     its README documents the same numbers the calculator prints.

No torch/vLLM/GPU needed.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "recipes" / "rtx-pro-6000"
MIMO = BASE / "mimo-v2.6-pro-hotsplit"

RECIPES = ["deepseek-v4.1-flash-hotsplit", "glm-5.3-flash-hotsplit", "glm-5.3-hotsplit"]

# slug -> (layers, experts, topk, cell_full MiB, expected mid hot GiB/rank)
GEOM = {
    "deepseek-v4.1-flash-hotsplit": (40, 384, 6, 17.9, 52),
    "glm-5.3-flash-hotsplit":       (42, 288, 8, 13.5, 31),
    "glm-5.3-hotsplit":             (75, 256, 8, 20.25, 36),
}

REQUIRED = ["README.md", "balance.json", "tp4.env.example"]


def test_planning_core_identical_everywhere():
    ref = (MIMO / "patches" / "hotsplit_plan.py").read_bytes()
    for slug in RECIPES:
        got = (BASE / slug / "patches" / "hotsplit_plan.py").read_bytes()
        assert got == ref, f"{slug} planning core drifted from mimo's"


@pytest.mark.parametrize("slug", RECIPES)
def test_recipe_files_present(slug):
    for name in REQUIRED:
        p = BASE / slug / name
        assert p.is_file() and p.stat().st_size > 0, f"missing {p}"
    for name in ("hotsplit_balance.py", "hotsplit_plan.py"):
        p = BASE / slug / "patches" / name
        assert p.is_file() and p.stat().st_size > 0, f"missing {p}"
    for name in ("balance.sh", "preflight-tp4.sh", "run-tp4-uva.sh"):
        p = BASE / slug / "scripts" / name
        assert p.is_file() and p.stat().st_size > 0, f"missing {p}"


@pytest.mark.parametrize("slug", RECIPES)
def test_balance_json_matches_geometry(slug):
    layers, experts, topk, cell_full, _ = GEOM[slug]
    g = json.loads((BASE / slug / "balance.json").read_text())
    assert (g["layers"], g["experts"], g["topk"]) == (layers, experts, topk)
    assert g["cell_mib_full"] == cell_full
    assert g["tp"] == 4


@pytest.mark.parametrize("slug", RECIPES)
def test_calculator_arithmetic(slug):
    layers, experts, topk, cell_full, hot_gib = GEOM[slug]
    out = subprocess.run(
        [sys.executable, str(BASE / slug / "patches" / "hotsplit_balance.py"),
         str(BASE / slug / "balance.json"), "--gib", str(hot_gib), "--json"],
        capture_output=True, text=True, check=True)
    r = json.loads(out.stdout)
    cell_rank = cell_full / 4
    # bank follows from geometry
    assert abs(r["bank_gib_rank"] - layers * experts * cell_rank / 1024) < 0.05
    # hot + cold == bank
    assert abs(r["hot_gib_rank"] + r["cold_gib_rank"] - r["bank_gib_rank"]) < 0.05
    # hot is an integer number of cells
    assert abs(r["hot_cells"] * cell_rank / 1024 - r["hot_gib_rank"]) < 0.05
    assert r["hot_cells"] <= layers * experts
    # decode traffic is topk * slice
    assert abs(r["decode_bytes_per_tok_mib_rank"] - topk * cell_rank) < 0.1


@pytest.mark.parametrize("slug", RECIPES)
def test_readme_documents_the_calculator_numbers(slug):
    layers, experts, topk, cell_full, hot_gib = GEOM[slug]
    text = (BASE / slug / "README.md").read_text()
    headings = {m.group(1).strip() for m in re.finditer(r"^##\s+(.+?)\s*$", text, re.M)}
    for s in ("What this runs", "Expert balance (the numbers this recipe is about)",
              "Files", "Launch", "Verify", "Results", "Known limits", "Rollback"):
        assert s in headings, f"{slug} README missing '## {s}'"
    # the published bank/hot numbers must appear
    bank = layers * experts * (cell_full / 4) / 1024
    assert f"{bank:.2f}" in text or f"{bank:.1f}" in text, f"{slug} README bank {bank} not documented"


@pytest.mark.parametrize("slug", RECIPES)
def test_scripts_parse(slug):
    for f in sorted((BASE / slug / "scripts").glob("*.sh")):
        subprocess.run(["bash", "-n", str(f)], check=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))