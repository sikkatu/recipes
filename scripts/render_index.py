#!/usr/bin/env python3
"""render_index.py — regenerate the recipe index in README.md / README.ru.md / README.zh.md from every recipes/*/*/recipe.yaml.

Usage:
    python3 scripts/render_index.py           # rewrite README.md between the index markers
    python3 scripts/render_index.py --check   # exit 1 if README.md is out of date (CI)

Index contract:
  - every cell is Markdown-escaped (pipes, backslashes, newlines) so table structure cannot break;
  - Model shows a short name + quantization only — no revision history, no file sizes;
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
STATUS_ICON_RU = {
    "verified": "✅ проверен",
    "experimental": "🧪 эксперимент",
    "draft": "📝 черновик",
    "deprecated": "⛔ устарел",
}
STATUS_ICON_ZH = {
    "verified": "✅ 已验证",
    "experimental": "🧪 实验",
    "draft": "📝 草稿",
    "deprecated": "⛔ 已弃用",
}
TABLE_HEADER = {
    "en": lambda: ["", "| Hardware | Recipe | Model | Status |", "|---|---|---|---|"],
    "ru": lambda: ["", "| Железо | Рецепт | Модель | Статус |", "|---|---|---|---|"],
    "zh": lambda: ["", "| 硬件 | 配方 | 模型 | 状态 |", "|---|---|---|---|"],
}


def md_escape(s) -> str:
    """Escape a value for a Markdown table cell so it can never split columns or rows."""
    t = str(s) if s is not None else ""
    t = t.replace("\\", "\\\\")
    t = t.replace("|", "\\|")
    t = t.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return t.strip()


def model_cell(r: dict) -> str:
    """Short model name + quantization — no revision hashes or file inventories."""
    mod = r.get("model") or {}
    name = md_escape(mod.get("name", "?"))
    q = mod.get("quantization", "")
    # truncate quantization to the first clause; long strings stay in the recipe README
    q_short = md_escape(str(q).split("(")[0].split(";")[0].split(",")[0].strip())
    return f"{name} ({q_short})" if q_short else name


def render(lang: str = "en") -> str:
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
        loc = {"en": "README.md", "ru": "README.ru.md", "zh": "README.zh.md"}[lang]
        readme = d / loc
        if not readme.is_file():
            readme = d / "README.md"
        if not readme.is_file():
            continue
        title = slug
        model = slug
        status = "draft"
        rtext = readme.read_text()
        for line in rtext.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                model = title.split(" — ")[0].split(" – ")[0].strip() or slug
                break
        # honest status from the README's own Status line (all three languages)
        if re.search(r"Status: VERIFIED|Статус: ПРОВЕРЕНО|状态：已在目标硬件上验证", rtext):
            status = "verified"
        rows.append((hw, slug, {
            "title": title,
            "status": status,
            "updated": None,
            "model": {"name": model, "quantization": ""},
        }))
    if not rows:
        return "\n_No recipes yet._\n"
    out = [*TABLE_HEADER[lang]()]
    for hw, slug, r in rows:
        rel = f"recipes/{hw}/{slug}"
        status = r.get("status", "draft")
        icons = {"en": STATUS_ICON, "ru": STATUS_ICON_RU, "zh": STATUS_ICON_ZH}[lang]
        icon = icons.get(status, md_escape(status))
        out.append(
            f"| `{hw}` | [{md_escape(r['title'])}]({rel}/) | {model_cell(r)} "
            f"| {icon} |"
        )
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    stale = []
    for fname, lang in (("README.md", "en"), ("README.ru.md", "ru"), ("README.zh.md", "zh")):
        readme = ROOT / fname
        text = readme.read_text()
        if START not in text or END not in text:
            print(f"{fname} lacks {START} / {END} markers")
            return 1
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        new = f"{head}{START}\n{render(lang)}\n{END}{tail}"
        if a.check:
            if new != text:
                stale.append(fname)
        else:
            readme.write_text(new)
            print(f"{fname} index rewritten")
    if a.check:
        if stale:
            print("stale index: " + ", ".join(stale) + " — run scripts/render_index.py")
            return 1
        print("index up to date in all languages")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())