"""Re-rank the hotsplit hot list from live-lane counts merged with the offline corpus.
usage: python3 hotsplit_rerank.py LIVE.json OFFLINE_MIX.json OUT.json [live_weight=1.0]
Writes {"train": {"decode": {layer: [E]}}} (normalised per layer, x1e6) - the format hotsplit.py reads.
Refuses (exit 2) if the live snapshot has fewer than MIN_TOKENS decode tokens per layer (default 50k):
a small live sample should not override the 900-turn corpus. Applies on the next lane boot; never restarts anything."""
import json, sys, os
live_p, off_p, out_p = sys.argv[1:4]
w = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
MIN = int(os.environ.get("MIN_TOKENS", "50000"))
live = json.load(open(live_p))["live"]["decode"]
off = json.load(open(off_p))["train"]["decode"]
def norm(v):
    s = float(sum(v)) or 1.0
    return [x / s for x in v]
toks = min(sum(v) for v in live.values()) / 8  # top-8 routing
if toks < MIN:
    print(f"live sample too small: {toks:.0f} tokens/layer < {MIN}; not writing"); sys.exit(2)
out = {}
for li, ov in off.items():
    o = norm(ov); l = norm(live.get(li, [0] * len(ov)))
    out[li] = [1e6 * (a + w * b) for a, b in zip(o, l)]
tmp = out_p + ".tmp"
json.dump({"train": {"decode": out}, "meta": {"live_tokens_per_layer": toks, "live_weight": w, "offline": off_p}}, open(tmp, "w"))
os.replace(tmp, out_p)
print(f"wrote {out_p}: {len(out)} layers, live {toks:.0f} tokens/layer, weight {w}")
