"""Re-rank hotsplit hot list from live-lane counts merged with the offline corpus.

usage: python3 hotsplit_rerank.py LIVE OFFLINE OUT [live_weight=1.0]

  LIVE      one or several live snapshot files (glob/archive paths joined by
            whitespace, e.g. 'live/counts.rank*.json'). Each file must be the
            format written by patches/hotsplit.py::LiveCounter:
            {"live": {"decode": {layer: [E]}}, "meta": {...}}.
            Per-rank snapshots are merged (element-wise sum) before ranking.
  OFFLINE   offline corpus histogram, e.g. patches/expert_hist_mix.json
            (format {"train": {"decode": {layer: [E]}}}).
  OUT       result file; written atomically (tmp + os.replace).
  live_weight  blend factor between offline and live, default 1.0,
            linear mix (1-w)*offline + w*live per layer.

Refuses (exit 2) when the merged live snapshot has fewer than MIN_TOKENS
decode tokens per layer (default 50k): a small sample must not overrule the
900-turn corpus.

Applies on the NEXT lane boot; never restarts anything.

After running, put the OUT file somewhere the container can see (e.g. copy
it into $PWD/lives and point HOTSPLIT_COUNTS=/live/expert_hist_custom.json in
tp4.env), then restart the server so hotsplit.py picks up the new ranking.
"""

import glob
import json
import os
import sys


def load_live(paths):
    """Merges several rank snapshot files into one {layer: [E]} dict."""
    merged = {}
    files_found = []
    for pat in paths:
        for f in sorted(glob.glob(pat)):
            snap = json.load(open(f))["live"]["decode"]
            for li, vec in snap.items():
                li = str(li)
                if li not in merged:
                    merged[li] = list(vec)
                else:
                    merged[li] = [a + b for a, b in zip(merged[li], vec)]
            files_found.append(f)
    return merged, files_found


def norm(vec):
    s = float(sum(vec)) or 1.0
    return [x / s for x in vec]


def main(argv):
    if len(argv) < 4:
        print(__doc__)
        return 2
    live_p, off_p, out_p = argv[1], argv[2], argv[3]
    w = float(argv[4]) if len(argv) > 4 else 1.0
    MIN = int(os.environ.get("MIN_TOKENS", "50000"))

    live, files = load_live(live_p.split())
    if not files:
        print(f"No live snapshot files matched {live_p!r}", file=sys.stderr)
        return 2
    print(f"merged {len(files)} live snapshot(s)")

    off = json.load(open(off_p))["train"]["decode"]
    toks = min(sum(v) for v in live.values()) / 8  # top-8 routing
    if toks < MIN:
        print(f"live sample too small: {toks:.0f} tokens/layer < {MIN}; not writing")
        return 2

    out = {}
    for li, ov in off.items():
        o = norm(ov)
        l = norm(live.get(li, [0] * len(ov)))
        out[li] = [1e6 * ((1 - w) * a + w * b) for a, b in zip(o, l)]

    tmp = out_p + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"train": {"decode": out},
                   "meta": {"live_tokens_per_layer": toks, "live_weight": w,
                            "offline": off_p, "live_files": files}}, f)
    os.replace(tmp, out_p)
    print(f"wrote {out_p}: {len(out)} layers, live {toks:.0f} tokens/layer, weight {w}")
    print(f"Next: copy {out_p} to /live/expert_hist_custom.json inside the container")
    print("  and set HOTSPLIT_COUNTS=/live/expert_hist_custom.json in tp4.env, then restart.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
