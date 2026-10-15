# Benchmarking recipes with llm-inference-bench

Every recipe's throughput and quality rows that come from a standard bench use one pinned tool:
[`local-inference-lab/llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench).
Our own instruments (knee/replay/teacher-forced divergence/BFCL gate) stay authoritative for
recipe headlines and promotion. The standard bench gives comparable, repeatable baselines.

## Pinning

- Clone at a release tag. Record `tag @ commit` in `results/<run-id>/bench-version.txt`. Current: **v0.7.7 @ `c71ec1f2`**.
- Run with `LLM_BENCH_NO_UPDATE_CHECK=1` (the script otherwise offers a self-upgrade on start) and `--display-mode plain`.
- Do not use `lil-bench` (the container wrapper) unless you mean to upload. We publish here, not on their board.
- The upstream repo has **no license file**. Do not copy its code or datasets into this repo. Publish only our output JSON and logs.

## Tiers

| Tier | When | Command shape | Time |
|---|---|---|---|
| T0 gate | every launch | `/v1/models` id, exact-word chat smoke, parsed `tool_calls[]` smoke | ~2 min |
| T1 perf | any recipe change | `--concurrency 1,4,8 --contexts 0,32768,131072 --duration 30` | ~10 min |
| T2 quality | engine / kernel / quant / speculation change | `--test-profile needle-checksum`; `--test-profile gsm8k --profile-runs 200 --compare-baseline <last promoted gsm8k json>`; optional `--tool-eval` subset | ~10–60 min |
| T3 promotion | before a new default route, or after a hardware/fabric change | full matrix, full GSM8K + MMLU-Pro, full `--tool-eval`; multi-node adds `--p2pmark-only` | hours |

**Promotion bar:** a paired McNemar p < 0.05 accuracy *loss* on T2 fails. Needle-checksum must stay EXACT N/N. T1 must stay within the run-to-run band of the last promoted run on the same workload.

## Labelling rules (the checker does not enforce these; reviewers do)

1. A T1 row's `workload` says **"llm-inference-bench synthetic filler"**, the context, and the concurrency. Speculative-decoding acceptance depends on content, so filler tok/s never replaces a recipe's own prompt-class headline.
2. Save the vLLM/SGLang `Running:`/`Waiting:` log lines from the bench window as `<run>.engine-load.txt`. A cell where other clients were running is relabelled or re-run, never averaged in.
3. Note any server-side cap the bench hit (e.g. `--max-num-seqs` below the bench's concurrency).
4. Scrub LAN IPs and local paths from the bench JSON before committing. The bench records `--host` and its own output path.
