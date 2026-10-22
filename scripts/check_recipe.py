#!/usr/bin/env python3
"""check_recipe.py — validate one or more recipe.yaml files against the recipe contract.

Usage:
    python3 scripts/check_recipe.py recipes/**/recipe.yaml
    python3 scripts/check_recipe.py --all

Exit 0 when every recipe passes; otherwise prints one error per line and exits 1.
Rules beyond the JSON schema (each is a test in tests/test_check_recipe.py):
  * id matches the recipe's directory path
  * README.md has the eight required sections
  * launch/rollback scripts, hardware profile, hardware snapshot exist
  * every metric has provenance and its raw file exists under results/<run_id>/
  * verified => every gate has last_pass_utc + run_id, and >=1 metric
  * patches[].sha256 matches the file on disk
  * no credential-looking strings anywhere in the recipe tree
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None

REQUIRED_README_SECTIONS = [
    "What this runs", "Hardware", "Software", "Launch", "Verify", "Results", "Known limits", "Rollback",
]
GATES = ("schema", "digest", "health", "quality", "performance")
CRED_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]
CRED_SKIP_SUFFIX = {".png", ".jpg", ".jpeg", ".svg", ".pt", ".safetensors", ".gz", ".zip"}


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "recipe.schema.json"


def _schema(root: Path) -> dict:
    p = root / "schemas" / "recipe.schema.json"
    return json.loads((p if p.exists() else SCHEMA_PATH).read_text())


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_recipe(path: Path, root: Path | None = None) -> list[str]:
    """Return a list of error strings for one recipe.yaml (empty list = pass)."""
    path = Path(path)
    rdir = path.parent
    root = Path(root) if root else _find_root(path)
    errs: list[str] = []
    pre = f"{rdir.relative_to(root) if rdir.is_relative_to(root) else rdir}: "

    try:
        r = yaml.safe_load(path.read_text())
    except Exception as e:  # noqa: BLE001
        return [pre + f"YAML parse error: {e}"]
    if not isinstance(r, dict):
        return [pre + "recipe.yaml is not a mapping"]

    # 1. JSON schema
    if jsonschema is not None:
        validator = jsonschema.Draft202012Validator(_schema(root), format_checker=jsonschema.FormatChecker())
        for ve in sorted(validator.iter_errors(r), key=lambda e: list(e.path)):
            loc = "/".join(str(x) for x in ve.path) or "<root>"
            errs.append(pre + f"schema {loc}: {ve.message}")
    else:
        errs.append(pre + "jsonschema not installed; schema not checked")

    # 2. id matches directory
    expected_id = "/".join(rdir.parts[-2:])
    if r.get("id") != expected_id:
        errs.append(pre + f"id '{r.get('id')}' does not match directory '{expected_id}'")

    # 3. README sections
    readme = rdir / "README.md"
    if not readme.exists():
        errs.append(pre + "README.md missing")
    else:
        text = readme.read_text()
        headings = {m.group(1).strip() for m in re.finditer(r"^##\s+(.+?)\s*$", text, re.M)}
        for s in REQUIRED_README_SECTIONS:
            if s not in headings:
                errs.append(pre + f"README missing section '## {s}'")

    # 4. referenced files exist
    def need(rel: str | None, what: str, base: Path = rdir) -> None:
        if rel and not _inside(base, rel):
            errs.append(pre + f"{what} '{rel}' escapes {base.name}/")
        elif rel and not (base / rel).exists():
            errs.append(pre + f"{what} '{rel}' not found")

    need((r.get("launch") or {}).get("script"), "launch.script")
    need((r.get("rollback") or {}).get("script"), "rollback.script")
    need((r.get("hardware") or {}).get("profile"), "hardware.profile", base=root)
    need(((r.get("hardware") or {}).get("observed") or {}).get("snapshot"), "hardware.observed.snapshot")

    # 5. metrics provenance
    metrics = r.get("metrics") or []
    for i, m in enumerate(metrics):
        prov = m.get("provenance") if isinstance(m, dict) else None
        if not prov:
            errs.append(pre + f"metrics[{i}] '{m.get('name') if isinstance(m, dict) else m}' has no provenance")
            continue
        run_dir = rdir / "results" / str(prov.get("run_id"))
        raw = run_dir / str(prov.get("raw"))
        if not _inside(run_dir, str(prov.get("raw"))):
            errs.append(pre + f"metrics[{i}] raw '{prov.get('raw')}' escapes results/{prov.get('run_id')}/")
        elif not raw.exists():
            errs.append(pre + f"metrics[{i}] raw file results/{prov.get('run_id')}/{prov.get('raw')} not found")

    # 6. verified gates
    if r.get("status") == "verified":
        gates = ((r.get("verification") or {}).get("gates") or {})
        for g in GATES:
            gd = gates.get(g) or {}
            if not gd.get("last_pass_utc") or not gd.get("run_id"):
                errs.append(pre + f"status verified but gate '{g}' has no last_pass_utc/run_id")
        if not metrics:
            errs.append(pre + "status verified but no metric with provenance is published")

    # 7. patches sha
    for p in (r.get("runtime") or {}).get("patches") or []:
        fp = rdir / p.get("path", "")
        if not _inside(rdir, p.get("path", "")):
            errs.append(pre + f"patch '{p.get('path')}' escapes the recipe dir")
        elif not fp.exists():
            errs.append(pre + f"patch '{p.get('path')}' not found")
        elif _sha256(fp) != p.get("sha256"):
            errs.append(pre + f"patch '{p.get('path')}' sha256 mismatch (on disk {_sha256(fp)[:12]}…)")

    # 7b. container dockerfile exists and matches its pinned sha256
    ctr = (r.get("runtime") or {}).get("container") or {}
    if ctr.get("dockerfile"):
        fp = rdir / ctr["dockerfile"]
        if not _inside(rdir, ctr["dockerfile"]):
            errs.append(pre + f"container dockerfile '{ctr['dockerfile']}' escapes the recipe dir")
        elif not fp.exists():
            errs.append(pre + f"container dockerfile '{ctr['dockerfile']}' not found")
        elif ctr.get("dockerfile_sha256") and _sha256(fp) != ctr["dockerfile_sha256"]:
            errs.append(pre + f"container dockerfile sha256 mismatch (on disk {_sha256(fp)[:12]}…)")
    elif ctr.get("dockerfile_sha256"):
        errs.append(pre + "container dockerfile_sha256 set without container dockerfile")

    # 8. credentials
    errs.extend(scan_credentials(rdir, pre))
    return errs


def scan_credentials(rdir: Path, pre: str = "") -> list[str]:
    """Credential-shaped strings anywhere under one recipe dir. Runs for drafts too: a draft without
    recipe.yaml used to get no scan at all, although SECURITY.md promises CI rejects credentials."""
    errs: list[str] = []
    for f in sorted(rdir.rglob("*")):
        if not f.is_file() or f.suffix.lower() in CRED_SKIP_SUFFIX or f.stat().st_size > 5_000_000:
            continue
        try:
            txt = f.read_text(errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        for pat in CRED_PATTERNS:
            if pat.search(txt):
                errs.append(pre + f"possible credential in {f.relative_to(rdir)} (pattern {pat.pattern[:30]}…)")
                break
    return errs


def _inside(base: Path, rel: str) -> bool:
    """True when base/rel stays inside base after resolving '..' and symlinks (absolute rel is refused)."""
    try:
        return (base / rel).resolve().is_relative_to(base.resolve()) and not Path(rel).is_absolute()
    except (OSError, ValueError):
        return False


def _find_root(path: Path) -> Path:
    for p in [path.parent, *path.parents]:
        if (p / "schemas" / "recipe.schema.json").exists():
            return p
    raise SystemExit(f"cannot find repo root (schemas/recipe.schema.json) above {path}")


def check_draft_contract(root: Path) -> list[str]:
    """Every recipes/<hw>/<slug>/ dir must be a real recipe: either a draft (README.md only) or
    a checked recipe (recipe.yaml + README.md). A non-draft status without recipe.yaml is an error."""
    errs = []
    for d in sorted(root.glob("recipes/*/*")):
        if not d.is_dir():
            continue
        has_readme = (d / "README.md").is_file()
        has_yaml = (d / "recipe.yaml").is_file()
        if not has_readme:
            errs.append(f"{d.relative_to(root)}: missing README.md (every recipe needs one)")
        if not has_yaml:
            errs.extend(scan_credentials(d, f"{d.relative_to(root)}: "))
        if not has_yaml:
            # draft without recipe.yaml is allowed — but it must not claim verified/experimental
            if has_readme:
                text = (d / "README.md").read_text()
                for bad in ("✅ verified", "🧪 experimental"):
                    if bad in text:
                        errs.append(
                            f"{d.relative_to(root)}: claims '{bad.strip()}' but has no recipe.yaml — "
                            "non-draft status requires recipe.yaml with provenance")
    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true", help="check every recipes/*/*/recipe.yaml under the repo root")
    a = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    paths = list(a.paths)
    if a.all or not paths:
        paths = sorted(root.glob("recipes/*/*/recipe.yaml"))
    total = 0
    # draft/contract check always runs — absence of recipe.yaml must never be a silent pass
    draft_errs = check_draft_contract(root)
    total += len(draft_errs)
    for e in draft_errs:
        print(e)
    if a.all or not paths:
        recipe_dirs = [d for d in root.glob("recipes/*/*") if d.is_dir()]
        if not paths and not recipe_dirs:
            print("no recipe dirs found at all — empty repo is not a valid state")
            return 1
        if not paths:
            print(f"checked draft contract: {len(draft_errs)} error(s) across "
                  f"{len(recipe_dirs)} recipe dir(s); no recipe.yaml to validate")
    for p in paths:
        errs = check_recipe(p, root=root)
        total += len(errs)
        for e in errs:
            print(e)
        if not errs:
            print(f"OK  {p.relative_to(root) if p.is_relative_to(root) else p}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
