# DeepSeek-V4.1-Flash — 4× RTX PRO 6000 (96 GB), TP4: expert balance for UVA expert offload

> **English** | [Русский](README.ru.md) | [中文](README.zh.md)


**Status: engineering balance + tooling, not a verified serving run.** What is real and checked
here is the *expert balance* — which routed experts belong in GPU memory and which in pinned host
memory, and the per-rank byte budgets — computed with the exact same unit-tested planner as
`recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit` (greedy share-per-byte hot set). The vLLM
runtime hook that *enforces* the split is model- and quant-specific and is **not** claimed to be
ported or validated on this hardware (see `patches/README.md`). Treat every launch as an
experiment and send the logs back.

## What this runs

DeepSeek-V4.1-Flash on 4× RTX PRO 6000 (96 GB GDDR7, PCIe 4.0 host link, no NVLink): vLLM TP4 + UVA expert offload with a
**UVA expert offload baseline plus the offline hot/cold balance planner** — the run itself is
stock vLLM TP4 with layer-granular UVA offload. The per-expert hot/cold split is planned and sized
here (`patches/hotsplit_balance.py`, `scripts/balance.sh`) with the same algorithm as the
MiMo-V2.6-Pro hotsplit recipe, but the runtime hook (`patches/hotsplit.py`) is **not mounted by
these launchers yet** — the recipe ships the planner and the patches, not a running hot/cold split.
Treat the hot/cold numbers below as a plan, not a measurement.

- model `deepseek-ai/DeepSeek-V4.1-Flash` @ `df42c109f1defefcbfcedbe7d905718a12266e40`
- quantization: MXFP4 (I8+E8M0 block scales) (as shipped, nothing re-quantized)
- draft: DSpark (mtp 0-2)

## Expert balance (the numbers this recipe is about)

Geometry from `deepseek-ai/DeepSeek-V4.1-Flash` config: **40 MoE layers × 384 routed experts** (+1 shared),
`num_experts_per_tok` = 6, hidden 5120, `moe_intermediate_size` 2304 → 35.39M params
per routed expert = **17.9 MiB** full. At TP4 a rank holds a ¼ shard → **4.47 MiB** per
expert slice. `balance.json` is the machine-readable form; `scripts/balance.sh` recomputes the table.

| quantity | value (per rank at TP4) |
|---|---|
| routed expert cells | 40 × 384 = 15,360 |
| one expert slice | 4.47 MiB (17.9 MiB full ÷ 4) |
| full routed bank | 67.12 GiB |
| **hot bank (GPU memory)** | **~52 GiB** (≈11,899 cells, 78%) — anchored on the pin-hot-experts split (295/384 of the cells) measured on a previous single-GPU lane |
| cold bank (pinned host) | 15.12 GiB/rank, 60.5 GiB across 4 ranks |
| decode expert traffic per layer | top-6 × 4.47 ≈ 26.8 MiB/layer/token/rank |
| decode expert traffic per token | 40 layers × 26.8 ≈ **1072 MiB/token/rank (~1.05 GiB)** |
| PCIe 4.0 ceiling (all cold) | 1072 MiB ÷ 25 GB/s ≈ **~23 tok/s** |

`HOTSPLIT_HOT_GIB` experiment ladder (per rank): **47, then 52, then 57 GiB** — raise only after a
first boot shows free GPU memory (96 GB GDDR7 × 0.90 ≈ 86 GiB usable; hot + KV + activations must fit together).
The real hot *set* is chosen from a routing histogram of your traffic (`--counts`), not from these
averages; `scripts/balance.sh --counts your.json` prints the exact coverage for your workload.

## Software (pinned image)

**No image has been verified on a 4-GPU target host yet.** These launchers run the stock vLLM model code (no `PATCH_MODEL_FILES`); the hot/cold
patches in `patches/` are applied only when you wire the runtime hook in yourself. Until a digest has been
validated end-to-end, treat the image choice as an experiment:

- Start from the vLLM release whose API the patches were written against (see
  `patches/mimo_v2_hotsplit.py` docstrings for the expected internals: `Mxfp4MoEMethod`,
  `make_mxfp4_moe_kernel`, `GPUModelRunner.execute_model`).
- Concrete image (from the vLLM nightly the patches were written against):
  `IMAGE="vllm/vllm-openai:nightly-8e2beaa3bff595577a069d49d96bde9074294ec2@sha256:167ccd8696bc09f1b63301185134e04b883623f8adcbd8e3a0ec3140a193abcf"`
  (tag carries the vLLM commit; digest pins the exact bytes).
- The launcher checks that some expected methods exist in the mounted vLLM sources before
  launching (SITE probe), but this is a smoke test, not a compatibility proof: signatures,
  sharding internals and Marlin kernel expectations are NOT verified automatically.
- Record the digest you validated in `results/<run-id>/` so the next run is reproducible.

Do **not** assume a random tag "just works" - verify on your hardware first.

## Files

| file | what |
|---|---|
| `balance.json` | expert geometry (layers, experts, top-k, per-expert bytes, TP) |
| `patches/hotsplit_balance.py` | the balance calculator (numpy only, no GPU) |
| `patches/hotsplit_plan.py` | the unit-tested planning core, byte-identical to MiMo's |
| `patches/README.md` | where the runtime enforcement hook comes from, and what is not ported |
| `scripts/balance.sh` | run the calculator for this model (with or without a histogram) |
| `scripts/preflight-tp4.sh` | host/GPU/NUMA diagnostics before a first boot |
| `scripts/manage.sh` | day-to-day control: status / logs / start / stop / restart / update / rollback / check |
| `scripts/run-tp4-uva.sh` | stock vLLM TP4 + UVA offload baseline (no hot/cold split) |
| `tp4.env.example` | every knob, with the default |

## Launch

First, compute and sanity-check the balance (no GPU needed):

```bash
scripts/balance.sh                 # uniform prior, prints the hot/cold split at the mid budget
scripts/balance.sh --counts my_traffic.json --gib 52
```

Then the runtime, preflight first, plan-only before moving any weight:

```bash
bash scripts/preflight-tp4.sh
MODEL_DIR=$HOME/models/df42c109f1defefcbfcedbe7d905718a12266e40 scripts/run-tp4-uva.sh          # baseline, no split
```

Start at 16K–128K context and raise in steps. The hot/cold enforcement hook is described in
`patches/README.md`; until it is ported and checked, `run-tp4-uva.sh` is the honest supported path.

## Verify

`scripts/balance.sh --json` must show `hot + cold == bank` and an integer cell count. On a real
boot, `numastat -p <worker>` shows pinned memory on each worker's GPU NUMA node. Quality bars are
teacher-forced logprobs, BFCL and a needle ladder; none has been run on TP4.

## Results

No measurement on this hardware. Arithmetic only: 26.8 MiB/layer/token/rank × 40 layers = ~1072 MiB/token/rank. The transfer-only decode ceiling over
PCIe 4.0 (~25 GB/s per GPU) is the cold share of that traffic; the plan line from `balance.sh`
gives the exact coverage for your histogram.

## Known limits

- Not run on target hardware; the MXFP4 (I8+E8M0 block scales) hot/cold enforcement hook is not ported (see `patches/README.md`).
- `hotsplit_plan.py` assumes global expert ids == local rows: expert parallelism is not supported.
- The anchor above is a byte budget from the pin-hot-experts split (295/384) measured on a previous single-GPU lane; the *set* must come from your traffic histogram.
- 4 workers pin ~60.5 GiB of host memory at once; keep the host floor generous (128 GiB).
- DSpark (mtp 0-2) speculation is untested with any hot/cold split on this hardware.

## Rollback

`scripts/rollback.sh` (stops+removes the hotsplit container; `--start-baseline` also starts the plain UVA baseline). `docker stop deepseek-flash-tp4-uva`, or simply do not enable the split — the baseline UVA launch never
moves a weight out of the stock placement. Nothing outside the container is modified.
