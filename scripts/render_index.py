#!/usr/bin/env python3
"""render_index.py — regenerate the recipe index in README.md from every recipes/*/*/recipe.yaml.

Usage:
    python3 scripts/render_index.py           # rewrite README.md between the index markers
    python3 scripts/render_index.py --check   # exit 1 if README.md is out of date (CI)

Index contract:
  - every cell is Markdown-escaped (pipes, backslashes, newlines) so table structure cannot break;
  - Model shows a short name + quantization only — no revision history, no file sizes;
  - Headline shows at most one number + one short label — no test protocols;
  - the main metric is chosen by the explicit `metrics[].primary` flag (falls back to first);
  - dates are normalised to YYYY-MM-DD (or "—" for drafts);
  - drafts without recipe.yaml are indexed from their README H1;
  - any non-draft recipe without recipe.yaml is a contract error (see check_recipe.py).
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- recipe-index:start -->", "<!-- recipe-index:end -->"

# single source of truth for status vocabulary — schemas/recipe.schema.json uses the same enum
VALID_STATUSES = ("draft", "experimental", "verified", "deprecated")
STATUS_ICON = {
    "verified": "✅ verified",
    "experimental": "🧪 experimental",
    "draft": "📝 draft",
    "deprecated": "⛔ deprecated",
}


def md_escape(s) -> str:
    """Escape a value for a Markdown table cell so it can never split columns or rows."""
    t = str(s) if s is not None else ""
    t = t.replace("\\", "\\\\")
    t = t.replace("|", "\\|")
    t = t.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return t.strip()


def norm_date(v) -> str:
    """Return YYYY-MM-DD or '—'. Accepts date / datetime / string; never emits garbage."""
    if v is None:
        return "—"
    s = str(v).strip()
    if not s or s == "—":
        return "—"
    # datetime.date / datetime objects have isoformat
    iso = getattr(v, "isoformat", None)
    if callable(iso):
        s = iso()[:10]
    else:
        m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
        s = m.group(1) if m else "—"
    return s if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s) else "—"


def pick_metric(r: dict) -> dict | None:
    """Explicit primary first, then first metric. Never fabricate a number."""
    ms = r.get("metrics") or []
    if not ms:
        return None
    for m in ms:
        if isinstance(m, dict) and m.get("primary"):
            return m
    return ms[0] if isinstance(ms[0], dict) else None


def headline(r: dict) -> str:
    """One number + short label — the index is an overview, not a method card."""
    m = pick_metric(r)
    if not m:
        return "—"
    val = m.get("value")
    if val is None:
        return "—"
    label = f" *({md_escape(m['label'])})*" if m.get("label") else ""
    return f"{val:g} {md_escape(m.get('unit', ''))}{label}"


def model_cell(r: dict) -> str:
    """Short model name + quantization — no revision hashes or file inventories."""
    mod = r.get("model") or {}
    name = md_escape(mod.get("name", "?"))
    q = mod.get("quantization", "")
    # truncate quantization to the first clause; long strings stay in the recipe README
    q_short = md_escape(str(q).split("(")[0].split(";")[0].split(",")[0].strip())
    return f"{name} ({q_short})" if q_short else name


def render() -> str:
    rows = []
    for p in sorted(ROOT.glob("recipes/*/*/recipe.yaml")):
        r = yaml.safe_load(p.read_text())
        hw, slug = p.parent.parts[-2:]
        rows.append((hw, slug, r))
    # Draft recipes intentionally carry no recipe.yaml (no measured numbers to pin yet).
    # Index them anyway, from their README H1, so the table always lists every recipe.
    have = {hw + "/" + slug for hw, slug, _ in rows}
    for d in sorted(ROOT.glob("recipes/*/*")):
        if not d.is_dir():
            continue
        hw, slug = d.parts[-2:]
        if hw + "/" + slug in have:
            continue
        readme = d / "README.md"
        if not readme.is_file():
            continue
        title = slug
        model = slug
        for line in readme.read_text().splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                model = title.split(" — ")[0].split(" – ")[0].strip() or slug
                break
        rows.append((hw, slug, {
            "title": title,
            "status": "draft",
            "updated": None,
            "model": {"name": model, "quantization": ""},
        }))
    if not rows:
        return "\n_No recipes yet._\n"
    out = ["", "| Hardware | Recipe | Model | Status | Headline | Updated |",
           "|---|---|---|---|---|---|"]
    for hw, slug, r in rows:
        rel = f"recipes/{hw}/{slug}"
        status = r.get("status", "draft")
        icon = STATUS_ICON.get(status, md_escape(status))
        out.append(
            f"| `{hw}` | [{md_escape(r['title'])}]({rel}/) | {model_cell(r)} "
            f"| {icon} | {headline(r)} | {norm_date(r.get('updated'))} |"
        )
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    readme = ROOT / "README.md"
    text = readme.read_text()
    if START not in text or END not in text:
        print(f"README.md lacks {START} / {END} markers")
        return 1
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{START}\n{render()}\n{END}{tail}"
    if a.check:
        if new != text:
            print("README.md index is stale — run scripts/render_index.py")
            return 1
        print("README.md index up to date")
        return 0
    readme.write_text(new)
    print("README.md index rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())