#!/usr/bin/env python3
"""render_card.py — render the at-a-glance recipe card (HTML, and PNG when Chrome is present).

Usage:
    python3 scripts/render_card.py recipes/<hw>/<slug>            # writes card.html (+ card.png)
    python3 scripts/render_card.py recipes/<hw>/<slug> --no-png

Inputs (both required):
    recipe.yaml   — schema-strict recipe. Supplies title/id/status/updated, model name/params/revision/
                    quantization, engine/version/commit, KV/draft method.
    card.yaml     — sidecar with the fields the strict schema does not carry (see templates/card.yaml
                    in the serving-recipe-publishing skill). Every number must cite a run_id or be
                    marked pending; the card renders pending items amber, never blank.

Rule: the card is generated, never hand-edited. Regenerate after every measurement round and commit
card.html/card.png alongside the results bundle.
"""
from __future__ import annotations

import argparse
import json
import html
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts" / "card_template.html"
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser",
]

PEND = '<span class="pend-inline">PENDING</span>'


def e(x) -> str:
    return html.escape(str(x)) if x is not None else ""


def val(x, unit: str = "") -> str:
    """Render a value or an amber PENDING marker."""
    if x in (None, "", "pending"):
        return PEND
    return f"{e(x)}{unit}"


def params_from_config(recipe_dir: Path) -> tuple[str, str] | None:
    """Derive total/active routed-expert counts from a sibling config.json (or card.model.config_json path), so the
    headline parameter count on the card cannot drift from the checkpoint. Returns None when no config is present.
    Counts routed experts + shared expert + dense-layer MLP + attention-free estimate is NOT attempted: this is the
    MoE-expert count only, which for DeepSeek-V4.x is >98% of the backbone. The card shows the model-card figure and
    this check refuses to render if the two disagree by more than 15%."""
    for cand in (recipe_dir / "config.json", recipe_dir / "model" / "config.json"):
        if cand.exists():
            c = json.loads(cand.read_text())
            c = c.get("text_config", c)
            try:
                L = int(c["num_hidden_layers"]); E = int(c["n_routed_experts"]); k = int(c["num_experts_per_tok"])
                h = int(c["hidden_size"]); m = int(c["moe_intermediate_size"]); sh = int(c.get("n_shared_experts", 0))
            except KeyError:
                return None
            per = 3 * h * m
            total = L * (E + sh) * per; active = L * (k + sh) * per
            return f"{total/1e9:.0f}B", f"{active/1e9:.1f}B"
    return None


def pills(recipe: dict, card: dict) -> str:
    m = recipe["model"]
    rt = recipe["runtime"]
    c = card.get("model", {})
    out = []
    out.append(f'<span class="pill type"><b>{e(c.get("type", "?"))}</b>{(" · " + e(c["type_detail"])) if c.get("type_detail") else ""}</span>')
    out.append(f'<span class="pill"><b>{e(c.get("total_params", m.get("params", "?")))}</b> total · <b>{val(c.get("active_params"))}</b> active</span>')
    ck = c.get("checkpoint_gb")
    out.append(f'<span class="pill">checkpoint <b>{val(ck, " GB")}</b> · rev <span class="mono">{e(str(m.get("revision", ""))[:8])}</span></span>')
    out.append(f'<span class="pill">weights <b>{e(c.get("weights_short", m.get("quantization", "")))}</b></span>')
    eng = f'{e(rt.get("engine", ""))} <span class="mono">{e(rt.get("engine_commit", ""))}</span>'
    if c.get("engine_extra"):
        eng += f' · {e(c["engine_extra"])}'
    out.append(f'<span class="pill">engine <b>{eng}</b></span>')
    if c.get("memory"):
        out.append(f'<span class="pill">memory <b>{e(c["memory"])}</b></span>')
    return "\n".join(out)


def hero(card: dict) -> str:
    h = card["hero"]
    boxes = []
    for key, label in (("c1", "Decode · C1"), ("c16", "Decode · C16 aggregate"), ("prefill", "Prefill · cold")):
        b = h.get(key, {})
        lbl = e(b["label"]) if b.get("label") else label  # optional override, e.g. a C8 aggregate on a slot-capped lane
        lbl = lbl + (f" ({e(b['class'])})" if b.get("class") else "")
        boxes.append(
            f'<div class="box"><div class="lbl">{lbl}</div>'
            f'<div class="hero">{val(b.get("value"))}<small>{e(b.get("unit", "tok/s"))}</small></div>'
            f'<div class="note">{e(b.get("note", ""))}</div></div>'
        )
    return "\n".join(boxes)


def row2(recipe: dict, card: dict) -> str:
    g = card.get("guesser", {})
    d = (recipe.get("model") or {}).get("draft") or {}
    acc = " ".join(f'<span>{e(k)} <b>{e(v)}</b></span>' for k, v in (g.get("accept") or {}).items()) or PEND
    guesser = (
        f'<div class="box"><div class="lbl">Guesser</div><div class="kv">'
        f'<b>{e(g.get("name", d.get("method", "none")))}</b> · {e(g.get("source", d.get("repo", "")))}<br>'
        f'{e(g.get("schedule", ""))}<br><span class="acc">accept: {acc}</span></div></div>'
    )
    extra = []
    for sec in card.get("secondary", [])[:2]:
        lines = "<br>".join(f'{e(k)}: <b>{val(v)}</b>' for k, v in sec.get("items", {}).items())
        extra.append(f'<div class="box"><div class="lbl">{e(sec["label"])}</div><div class="kv">{lines}</div></div>')
    return guesser + "\n" + "\n".join(extra)


def fidelity(card: dict) -> str:
    items = []
    for f in card.get("fidelity", []):
        cls = {"pass": "ok", "measured": "meas"}.get(f.get("status"), "pend")  # measured = a number, not a verdict
        items.append(f'<li class="{cls}">{e(f["text"])}</li>')
    return "\n".join(items) or f'<li class="pend">no fidelity evidence recorded</li>'


def harness(card: dict) -> str:
    """Claim card for agent results (per @getmeosu, 2026-09-20): fixed outer protocol + held-out sibling + cost per solved task.
    Every field is pending (amber) until measured; nothing here is inferred from tok/s."""
    h = card.get("harness") or {}
    if not h:
        return '<li class="pend">harness card pending — protocol hash, held-out sibling suite, cost per solved task</li>'
    items = []
    proto = h.get("protocol") or {}
    if proto:
        items.append(f'<li class="ok">protocol <span class="mono">{e(proto.get("file", "?"))}</span> sha256 <span class="mono">{e(str(proto.get("sha256", ""))[:12])}</span> · {e(proto.get("summary", ""))}</li>')
    else:
        items.append('<li class="pend">fixed outer protocol: pending</li>')
    dev = h.get("dev") or {}
    if dev.get("solved") is not None:
        items.append(f'<li class="ok">dev suite (decides candidacy): <b>{e(dev.get("name", ""))}</b> {e(dev["solved"])}/{e(dev.get("n", "?"))} solved ({e(dev.get("pct", ""))}) · {e(dev.get("note", ""))}</li>')
    else:
        items.append('<li class="pend">dev suite (decides candidacy): pending</li>')
    r = h.get("heldout") or {}; c = h.get("cost") or {}
    if r.get("solved") is not None and c.get("usd_per_1000_solved") is not None:
        # THE PAIR: Pass@1 next to $/solved on one line, so a cost claim cannot hide a quality drop (getmeosu, 2026-09-21)
        u = r.get("unsolved") or {}
        unsolved = " · ".join(f"{e(v)} {e(k)}" for k, v in u.items()) if u else "breakdown pending"
        items.append(
            f'<li class="pair"><span class="big">Pass@1 {e(r.get("pct", ""))}</span> <span class="dim">({e(r["solved"])}/{e(r.get("n", "?"))})</span>'
            f' <span class="sep">·</span> <span class="big">${e(c["usd_per_1000_solved"])} / 1000 solved</span>'
            f'<br><span class="dim">{e(r.get("name", "held-out sibling"))} · decides promotion · protocol sha {e(str(proto.get("sha256", ""))[:12])}'
            f' · priced {e(c.get("priced_on", "date pending"))}: {e(c.get("basis", ""))}</span>'
            f'<br><span class="dim">unsolved: {unsolved}</span></li>')
    else:
        if r.get("solved") is not None:
            items.append(f'<li class="ok">held-out sibling (decides promotion): <b>{e(r.get("name", ""))}</b> {e(r["solved"])}/{e(r.get("n", "?"))} solved ({e(r.get("pct", ""))})</li>')
        else:
            items.append('<li class="pend">held-out sibling (decides promotion; never run in a campaign): pending</li>')
        items.append('<li class="pend">Pass@1 · $ / solved pair: pending (needs held-out run + priced cost)</li>')
    return "\n".join(items)


def render(recipe_dir: Path) -> str:
    recipe = yaml.safe_load((recipe_dir / "recipe.yaml").read_text())
    derived = params_from_config(recipe_dir)
    if derived:
        card_probe = yaml.safe_load((recipe_dir / "card.yaml").read_text())
        claimed = str(card_probe.get("model", {}).get("total_params", ""))
        num = "".join(ch for ch in claimed if ch.isdigit() or ch == ".")
        if num:
            ratio = float(num) / float(derived[0].rstrip("B"))
            if not 0.85 <= ratio <= 1.15:
                raise SystemExit(f"card.yaml total_params {claimed} disagrees with config.json expert count {derived[0]} "
                                 f"(routed+shared experts only; ratio {ratio:.2f}). Fix the card or the config.")
        print(f"[render_card] config.json expert-count check: total ≈{derived[0]} active ≈{derived[1]}; card says {claimed} — ok")
    card = yaml.safe_load((recipe_dir / "card.yaml").read_text())
    tpl = TEMPLATE.read_text()
    rel = recipe_dir.relative_to(ROOT).as_posix()
    fields = {
        "TITLE": e(card.get("title", recipe["title"])),
        "SUBTITLE": e(card.get("subtitle", "")),
        "VERSION": e(card.get("version", "")),
        "UPDATED": e(recipe.get("updated", "")),
        "STATUS": e(recipe.get("status", "")),
        "PILLS": pills(recipe, card),
        "HERO": hero(card),
        "ROW2": row2(recipe, card),
        "FIDELITY": fidelity(card),
        "HARNESS": harness(card),
        "RECIPE_PATH": e(rel),
        "RUN_ID": e(card.get("run_id", "")),
        "HANDLES": e(card.get("handles", "github.com/sikkatu/recipes")),
    }
    for k, v in fields.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    leftover = [t for t in ("{{",) if t in tpl]
    if leftover:
        sys.exit("template placeholder left unfilled")
    return tpl


def to_png(html_path: Path, png_path: Path) -> bool:
    for c in CHROME_CANDIDATES:
        exe = c if Path(c).exists() else shutil.which(c)
        if not exe:
            continue
        # Chrome's --screenshot captures exactly the viewport; a fixed 900 px silently cropped the fidelity and
        # agent-claim strips once the card grew. Render tall, then trim background-only rows (Pillow; else stays tall).
        cmd = [exe, "--headless=new", "--disable-gpu", "--hide-scrollbars",
               f"--screenshot={png_path}", "--window-size=1200,2400", "--force-device-scale-factor=2",
               html_path.resolve().as_uri()]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and png_path.exists():
            _trim_bottom(png_path)
            return True
    return False


def _trim_bottom(png_path: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        return
    im = Image.open(png_path).convert("RGB")
    w, h = im.size
    bg = im.getpixel((w - 4, h - 4))  # page background sampled at the bottom-right corner
    px = im.load()
    last = h - 1
    while last > 0:
        row = [px[x, last] for x in range(0, w, 16)]
        if any(abs(p[0] - bg[0]) + abs(p[1] - bg[1]) + abs(p[2] - bg[2]) > 24 for p in row):
            break
        last -= 1
    cut = min(h, last + 1 + 48)  # keep 24 css px of bottom padding at 2x
    if cut < h:
        im.crop((0, 0, w, cut)).save(png_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe_dir")
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args()
    d = (ROOT / a.recipe_dir).resolve() if not Path(a.recipe_dir).is_absolute() else Path(a.recipe_dir)
    for req in ("recipe.yaml", "card.yaml"):
        if not (d / req).exists():
            sys.exit(f"missing {d / req}")
    out = d / "card.html"
    out.write_text(render(d))
    print(f"wrote {out.relative_to(ROOT)}")
    if not a.no_png:
        png = d / "card.png"
        if to_png(out, png):
            print(f"wrote {png.relative_to(ROOT)}")
        else:
            print("PNG skipped: no headless Chrome found (card.html is still valid)")


if __name__ == "__main__":
    main()
