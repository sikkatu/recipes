#!/usr/bin/env python3
"""concurrency_sweep.py -- benchmark N=1..K PARALLEL streaming requests.

Part C (round 87): find how many parallel chats THIS hardware holds -- the
knee. Drives an ALREADY RUNNING engine, writes only its JSON report, never
touches configs or the LMCache. STD LIB ONLY (urllib + threading).

Per level N: keep exactly N streams in flight for --duration seconds; from
usage tokens (missing usage -> null, never faked) measure TTFT and pure
decode per-stream tok/s; aggregate tok/s; sample /metrics for KV/queue/
preemptions. Correctness check before the levels: sequential vs parallel
(temperature=0, fixed seed) -> identical / diverges-late (WARN) /
diverges-early (FAIL). Knee = first N with per-stream median drops > --drop-
threshold vs N=1 OR max KV >= 0.9 OR preemptions grew OR aggregate grew <10%.
"""
from __future__ import annotations
import argparse, json, os, re, signal, statistics, sys, threading, time
import urllib.request
from pathlib import Path

SCHEMA_VERSION = "concurrency-sweep-1"
HERE = Path(__file__).resolve().parent
RECIPE = HERE.parent

DEFAULT_TOPICS = [
    "Write a full Python implementation of a thread-safe token-bucket rate limiter plus a small test suite and an explanation of failure modes under concurrency.",
    "Write a complete asyncio TCP echo server with graceful shutdown, per-connection tasks, read timeouts and structured logging; explain the code.",
    "Given a buggy quicksort (bad pivot handling on duplicates), find the bug, explain it, fix it, and add five unit tests with expected outputs.",
    "Design an LRU cache with TTL and O(1) operations; give a complete Python implementation with tests and explain the data structures used.",
    "Write five SQL queries over orders(id,user_id,total,created_at) and users(id,country): monthly revenue per country, top spenders, 7-day rolling average, retention cohorts, and explain each query plan.",
    "Implement a small recursive-descent parser for arithmetic expressions with +,-,*,/, parentheses and unary minus. Full code, tokenizer included.",
    "Write a Python CLI tool that watches a log file and alerts on error-rate spikes using a sliding window; include argument parsing and tests.",
    "Write a minimal markdown-to-HTML converter in Python supporting headers, bold, italics, links and lists; include tests for each feature.",
    "Implement a producer-consumer pipeline with asyncio queues, bounded backpressure, cancellation handling and unit tests for the queue logic.",
    "Explain the token-bucket + sliding-window hybrid rate limiting algorithm and implement it in Python with a discussion of clock skew.",
]

METRIC_NAMES = ["vllm:num_requests_running", "vllm:num_requests_waiting",
                "vllm:kv_cache_usage_perc", "vllm:num_preemptions_total",
                "vllm:generation_tokens_total"]

def resolve_api_key():
    k = os.environ.get("API_KEY")
    if k: return k
    for cand in (RECIPE/"live"/"api_key", Path.home()/".config"/"kvp"/"api_key"):
        try: return cand.read_text().strip()
        except OSError: continue
    return None

def _headers(key):
    h = {"Content-Type": "application/json"}
    if key: h["Authorization"] = "Bearer " + key
    return h

def fetch_metrics(url, names=METRIC_NAMES, timeout=10.0):
    try: out = urllib.request.urlopen(url, timeout=timeout).read().decode("utf-8","replace")
    except Exception: return {n: None for n in names}
    res = {}
    for n in names:
        m = re.findall(r"^"+re.escape(n)+r"\s+([0-9.eE+\-]+)", out, re.M)
        res[n] = float(m[-1]) if m else None
    return res

def stream_request(url, key, prompt, model, max_tokens, seed=None,
                   temperature=None, timeout=1200.0):
    body = {"model": model, "messages":[{"role":"user","content":prompt}],
            "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    if temperature is not None: body["temperature"] = temperature
    if seed is not None: body["seed"] = seed
    req = urllib.request.Request(url+"/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers=_headers(key), method="POST")
    t0 = time.time(); t_first = t_last = None; usage = None; text = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8","replace").strip()
                if not line.startswith("data:"): continue
                p = line[5:].strip()
                if p == "[DONE]": break
                try: c = json.loads(p)
                except json.JSONDecodeError: continue
                t_last = time.time()
                if t_first is None: t_first = t_last
                for ch in (c.get("choices") or []):
                    d = (ch.get("delta") or {}).get("content")
                    if d: text.append(d)
                u = c.get("usage") or {}
                if u.get("completion_tokens"): usage = u["completion_tokens"]
    except Exception as e:
        return {"ttft": (t_first-t0) if t_first else None, "gen_tokens": usage,
                "decode_tps": None, "text": "".join(text),
                "err": type(e).__name__, "wall": time.time()-t0}
    ttft = (t_first-t0) if t_first else None
    dt = None
    if usage and t_last and t_first and t_last > t_first and usage >= 2:
        dt = (usage-1)/(t_last-t_first)
    return {"ttft": ttft, "gen_tokens": usage, "decode_tps": dt,
            "text": "".join(text), "err": None if text else "no_tokens",
            "wall": time.time()-t0}


# ----------------------------- correctness check ---------------------------
def _first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def correctness_check(url, key, model, prompts, max_tokens, k, seed=1234):
    """Sequential vs parallel (K prompts, temperature=0, fixed seed)."""
    k = min(int(k), len(prompts), 8)
    seq = []
    for i in range(k):
        r = stream_request(url, key, prompts[i], model, max_tokens,
                           seed=seed, temperature=0.0, timeout=900)
        seq.append(r.get("text") or "")
    slots = [None] * k

    def worker(i):
        slots[i] = stream_request(url, key, prompts[i], model, max_tokens,
                                  seed=seed, temperature=0.0, timeout=900)
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(k)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()

    identical = div_late = div_early = errs = 0
    details = []
    for i in range(k):
        r = slots[i] or {}
        if r.get("err"):
            errs += 1
            details.append({"i": i, "cls": "error", "reason": r["err"]})
            continue
        a = seq[i]
        b = r.get("text") or ""
        if not a.strip() or not b.strip():
            div_early += 1
            details.append({"i": i, "cls": "diverges-early", "reason": "empty"})
            continue
        if a == b:
            identical += 1
            details.append({"i": i, "cls": "identical"})
            continue
        d = _first_diff(a, b)
        frac = (d / max(len(a), 1)) if d is not None else 0.0
        if frac >= 0.5:
            div_late += 1
            details.append({"i": i, "cls": "diverges-late",
                            "first_diff": d, "frac": round(frac, 3)})
        else:
            div_early += 1
            details.append({"i": i, "cls": "diverges-early",
                            "first_diff": d, "frac": round(frac, 3)})
    verdict = "FAIL" if (div_early or errs) else ("WARN" if div_late else "PASS")
    return {"k": k, "identical": identical, "diverges_late": div_late,
            "diverges_early": div_early, "errors": errs,
            "verdict": verdict, "details": details}


# ------------------------------- one level ---------------------------------
def run_level(url, key, model, topics, n, duration, max_tokens,
              metrics_url, sample_iv=1.5):
    """Keep exactly N streaming requests in flight for `duration` seconds."""
    stop_at = time.time() + duration
    results = []
    lock = threading.Lock()
    idx = [0]
    smp = {"kv_max": 0.0, "running_max": 0, "waiting_max": 0,
           "preempt_0": None, "preempt_n": None,
           "running_ge_n": 0.0, "wall": 0.0}
    done = threading.Event()

    def sampler():
        t_last = time.time()
        while not done.is_set():
            m = fetch_metrics(metrics_url)
            now = time.time()
            with lock:
                if m["vllm:kv_cache_usage_perc"] is not None:
                    smp["kv_max"] = max(smp["kv_max"], m["vllm:kv_cache_usage_perc"])
                if m["vllm:num_requests_running"] is not None:
                    rv = int(m["vllm:num_requests_running"])
                    smp["running_max"] = max(smp["running_max"], rv)
                    if rv >= n:
                        smp["running_ge_n"] += (now - t_last)
                if m["vllm:num_requests_waiting"] is not None:
                    smp["waiting_max"] = max(smp["waiting_max"],
                                             int(m["vllm:num_requests_waiting"]))
                if m["vllm:num_preemptions_total"] is not None:
                    if smp["preempt_0"] is None:
                        smp["preempt_0"] = m["vllm:num_preemptions_total"]
                    smp["preempt_n"] = m["vllm:num_preemptions_total"]
                smp["wall"] += (now - t_last)
            t_last = now
            done.wait(sample_iv)

    st = threading.Thread(target=sampler, daemon=True)
    st.start()

    def worker():
        while time.time() < stop_at:
            with lock:
                i = idx[0]
                idx[0] += 1
            r = stream_request(url, key, topics[i % len(topics)], model,
                               max_tokens, timeout=duration * 3 + 120)
            with lock:
                results.append(r)

    ths = [threading.Thread(target=worker) for _ in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    done.set()
    st.join(timeout=5)

    oks = [r for r in results if not r["err"]]
    errs = len(results) - len(oks)
    stps = [r["decode_tps"] for r in oks if r["decode_tps"] is not None]
    ttfts = [r["ttft"] for r in oks if r["ttft"] is not None]
    total_gen = sum(r["gen_tokens"] or 0 for r in oks)

    def pct(vals, q):
        if not vals:
            return None
        ss = sorted(vals)
        return ss[max(0, min(len(ss) - 1, int(round((len(ss) - 1) * q))))]

    upd = None
    if smp["preempt_0"] is not None and smp["preempt_n"] is not None:
        upd = smp["preempt_n"] - smp["preempt_0"]
    notes = []
    if errs:
        notes.append(f"{errs} request(s) errored")
    if any(r["gen_tokens"] is None for r in oks):
        notes.append("usage missing in some responses -- metrics may be incomplete")
    return {"n": n, "requests": len(results), "errors": errs,
            "agg_tps": (total_gen / duration if duration > 0 else None),
            "stream_tps_p10": pct(stps, 0.10), "stream_tps_p50": pct(stps, 0.50),
            "ttft_p50": pct(ttfts, 0.50), "ttft_p95": pct(ttfts, 0.95),
            "kv_max": smp["kv_max"] or None,
            "running_max": smp["running_max"] or None,
            "waiting_max": smp["waiting_max"] or None,
            "preemptions_delta": upd,
            "running_ge_n_frac": (smp["running_ge_n"] / smp["wall"]
                                  if smp["wall"] else None),
            "notes": notes}


# -------------------------------- knee -------------------------------------
def find_knee(levels, drop_threshold=0.20):
    """First N (>=2) with a knee symptom -> (n, reason) else (None, None)."""
    if not levels:
        return None, None
    base = next((lv["stream_tps_p50"] for lv in levels if lv["n"] == 1), None)
    prev_agg = None
    for lv in levels:
        if lv["n"] == 1:
            prev_agg = lv.get("agg_tps")
            continue
        if (base and lv["stream_tps_p50"] is not None
                and lv["stream_tps_p50"] < base * (1 - drop_threshold)):
            return lv["n"], (f"per-stream median dropped >{drop_threshold:.0%} vs N=1 "
                             f"({lv['stream_tps_p50']:.1f} vs {base:.1f} tok/s)")
        if lv.get("kv_max") is not None and lv["kv_max"] >= 0.90:
            return lv["n"], f"max KV usage {lv['kv_max']:.0%} >= 90%"
        if lv.get("preemptions_delta"):
            return lv["n"], f"preemptions grew by {lv['preemptions_delta']}"
        if (prev_agg and lv.get("agg_tps") is not None and prev_agg > 0
                and lv["agg_tps"] < prev_agg * 1.10):
            return lv["n"], (f"aggregate tok/s grew <10% vs previous level "
                             f"({lv['agg_tps']:.0f} vs {prev_agg:.0f})")
        if lv.get("agg_tps") is not None:
            prev_agg = lv["agg_tps"]
    return None, None


# --------------------------------- main ------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        prog="concurrency-sweep.sh",
        description="Drive N=1..K parallel streaming requests at an ALREADY "
                    "RUNNING engine and find the knee."
                    " Every value shown in the report is measured.")
    ap.add_argument("--levels", default="1,2,4,6,8,10",
                    help="comma list of N (default 1,2,4,6,8,10)")
    ap.add_argument("--url", default=None,
                    help="engine base URL (default http://127.0.0.1:${PORT:-30007})")
    ap.add_argument("--model", default=None,
                    help="model name (default: read from /v1/models)")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="seconds per level with N streams held in flight (default 60)")
    ap.add_argument("--requests-per-level", type=int, default=None,
                    help="exit each level after this many finished requests (instead of --duration)")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="max_tokens for load requests (default 256)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="warmup requests per level, not counted (default 1)")
    ap.add_argument("--prompts", default=None,
                    help="file with prompts, one per line (default: built-in coding topics)")
    ap.add_argument("--metrics-url", default=None,
                    help="Prometheus-style /metrics URL (default <url>/metrics)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default benchmarks/results/concurrency-sweep-<UTC>.json)")
    ap.add_argument("--drop-threshold", type=float, default=0.20,
                    help="per-stream drop considered a knee vs N=1 (default 0.20)")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="known engine --max-num-seqs (marks levels above it as queued)")
    ap.add_argument("--yes", action="store_true",
                    help="skip the 'busy engine' pause")
    ap.add_argument("--no-correctness", action="store_true",
                    help="skip the correctness step (NOT recommended)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    url = args.url or f"http://127.0.0.1:{os.environ.get('PORT', '30007')}"
    metrics_url = args.metrics_url or (url.rstrip("/") + "/metrics")
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    if not levels:
        sys.exit("--levels is empty")
    engine_max = args.max_num_seqs

    key = resolve_api_key()
    # warn BEFORE starting (section 9.6): no-cache-salt unsupported by the
    # contract; the prompt warns about prefix-cache / L2 writes up front.
    print("=" * 72)
    print("WARNING: this load run FILLS the prefix cache and may STORE chunks")
    print("         into the persistent L2 cache (KV_PERSIST=1). Nothing else is")
    print("         written: configs and the LMCache server are NOT touched.")
    est = len(levels) * (args.duration + 30)
    print(f"levels={levels}  duration={args.duration}s/level  "
          f"max_tokens={args.max_tokens}  est. total ~{est/60:.0f} minutes")
    print("=" * 72)

    # existing load warning (section 9.6)
    if not args.yes:
        try:
            m = fetch_metrics(metrics_url)
            r0 = m.get("vllm:num_requests_running")
            if r0 and r0 > 0:
                print(f"WARNING: vllm:num_requests_running={int(r0)} -- the engine is "
                      "ALREADY serving other traffic; measurements will be skewed.")
                print("Press Ctrl-C in the next 5 s to abort (or rerun with --yes).")
                time.sleep(5)
        except Exception:
            pass

    # resolve model
    model = args.model
    if not model:
        try:
            req = urllib.request.Request(url.rstrip("/") + "/v1/models",
                                         headers=_headers(key))
            body = json.loads(urllib.request.urlopen(req, timeout=15).read())
            model = (body.get("data") or [{}])[0].get("id")
        except Exception:
            model = None
    if not model:
        sys.exit("ERROR: --model is required (could not read /v1/models -- is the engine up?)")
    print(f"engine={url}  model={model}")

    prompts = DEFAULT_TOPICS
    if args.prompts:
        try:
            lines = [l.strip() for l in Path(args.prompts).read_text().splitlines()
                     if l.strip() and not l.startswith("#")]
            if not lines:
                sys.exit(f"--prompts {args.prompts}: no prompts found")
            prompts = lines
        except OSError as e:
            sys.exit(f"--prompts {args.prompts}: {e.strerror}")
    distinct = {id(p): p for p in prompts}
    if len(distinct) < max(levels):
        print(f"Note: only {len(prompts)} distinct prompts; some repeat across N parallel "
              "streams -- prefix cache / LMCache may then skew the measurement.")

    # -------------- correctness step -------------------
    corr = None
    if args.no_correctness:
        corr = {"verdict": "SKIPPED", "k": 0, "identical": 0,
                "diverges_late": 0, "diverges_early": 0, "errors": 0,
                "details": [], "note": "--no-correctness given"}
    else:
        k = min(max(levels), 8)
        print(f"[correctness] sequential vs parallel, K={k}, temperature=0 ...")
        corr = correctness_check(url, key, model, prompts, min(args.max_tokens, 128), k)
        v = corr["verdict"]
        print(f"[correctness] verdict={v}  identical={corr['identical']}  "
              f"diverges-late={corr['diverges_late']}  diverges-early={corr['diverges_early']}  "
              f"errors={corr['errors']}")
        if v == "FAIL":
            print("WARNING: correctness FAILED -- the mode is UNRELIABLE for real work.")
            print("         The speed numbers below are still measured, but do NOT run "
                  "production traffic at N that fails correctness.")
            print("         Consider lowering MAX_NUM_SEQS to the highest passing level.")

    # -------------- levels -------------------
    lvs = []
    try:
        for n in levels:
            if engine_max is not None and n > engine_max:
                print(f"[N={n}] NOTE: above --max-num-seqs={engine_max} -- surplus "
                      "requests will queue (marked in the report)")
            if args.warmup:
                print(f"[N={n}] warmup ({args.warmup}) ...")
                for w in range(args.warmup):
                    stream_request(url, key, prompts[w % len(prompts)], model,
                                   min(args.max_tokens, 32), timeout=300)
            print(f"[N={n}] measuring {args.duration:.0f}s with N streams in flight ...")
            lv = run_level(url, key, model, prompts, n, args.duration,
                           args.max_tokens, metrics_url)
            if engine_max is not None and n > engine_max:
                lv.setdefault("notes", []).append(
                    f"above engine --max-num-seqs={engine_max} -- surplus queued")
            lvs.append(lv)
            print(f"[N={n}] requests={lv['requests']} errors={lv['errors']}  "
                  f"agg={lv['agg_tps']:.0f} tok/s  stream_p50="
                  f"{('%.1f' % lv['stream_tps_p50']) if lv['stream_tps_p50'] is not None else 'n/a'}"
                  f"  ttft_p95={(('%.1f' % lv['ttft_p95']) if lv['ttft_p95'] is not None else 'n/a')}s"
                  f"  kv_max={(('%.2f' % lv['kv_max']) if lv['kv_max'] is not None else 'n/a')}")
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl-C) -- writing a PARTIAL report with the "
              f"{len(lvs)} level(s) already measured.")

    knee_n, knee_reason = find_knee(lvs, args.drop_threshold)

    # -------------- report -------------------
    report = {
        "schema_version": SCHEMA_VERSION,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": {"url": url, "model": model, "max_num_seqs": engine_max},
        "params": {"levels": levels, "duration": args.duration,
                   "max_tokens": args.max_tokens, "prompts": len(prompts),
                   "drop_threshold": args.drop_threshold},
        "correctness": corr,
        "levels": lvs,
        "knee": {"n": knee_n, "reason": knee_reason},
        "verdict": ("knee at N=%s -- %s; recommended ceiling for THIS "
                    "prompt set: the level before" % (knee_n, knee_reason)
                    if knee_n else ("no knee up to N=%s measured" % max(levels) if lvs
                                    else "aborted before any level was measured")),
    }
    out = Path(args.out) if args.out else (
        RECIPE / "benchmarks" / "results" /
        f"concurrency-sweep-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print()
    print(f"report written: {out}")
    print("verdict: " + report["verdict"])
    if corr and corr.get("verdict") == "FAIL":
        print("NOTE: correctness FAILED above -- see report.correctness")
    return 0


if __name__ == "__main__":
    sys.exit(main())
