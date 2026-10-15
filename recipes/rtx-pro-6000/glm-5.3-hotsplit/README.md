# GLM-5.3 — 4× RTX PRO 6000 (96 GB), TP4: expert balance for UVA expert offload

**Status: engineering balance + tooling, not a verified serving run.** What is real and checked
here is the *expert balance* — which routed experts belong in GPU memory and which in pinned host
memory, and the per-rank byte budgets — computed with the exact same unit-tested planner as
`recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit` (greedy share-per-byte hot set). The vLLM .28 UVA
runtime hook that *enforces* the split is model- and quant-specific and is **not** claimed to be
ported or validated on this hardware (see `patches/README.md`). Treat every launch as an
experiment and send the logs back.

## What this runs

GLM-5.3 on 4× RTX PRO 6000 (96 GB GDDR7, PCIe 4.0 host link, no NVLink): vLLM .28 UVA TP4 + UVA expert offload with a
per-expert hot/cold split — hot routed experts in GPU memory, the rest read from pinned host memory
over PCIe. Same idea and same balance algorithm as the MiMo-V2.6-Pro hotsplit recipe, with this
model's geometry.

- model `incoai/GLM-5.3-NVFP4` @ `54e52520606f96b3d9fc84088ad22882a61648ac`
- quantization: NVFP4 (ModelOpt) (as shipped, nothing re-quantized)
- draft: MTP(2)

## Expert balance (the numbers this recipe is about)

Geometry from `incoai/GLM-5.3-NVFP4` config: **75 MoE layers × 256 routed experts** (+1 shared),
`num_experts_per_tok` = 8, hidden 6144, `moe_intermediate_size` 2048 → 37.75M params
per routed expert = **20.25 MiB** full. At TP4 a rank holds a ¼ shard → **5.06 MiB** per
expert slice. `balance.json` is the machine-readable form; `scripts/balance.sh` recomputes the table.

| quantity | value (per rank at TP4) |
|---|---|
| routed expert cells | 75 × 256 = 19,200 |
| one expert slice | 5.06 MiB (20.25 MiB full ÷ 4) |
| full routed bank | 94.92 GiB |
| **hot bank (GPU memory)** | **~36 GiB** (≈7,282 cells, 38%) — anchored on a measured 7360-slot GPU-memory cache from a previous single-GPU lane |
| cold bank (pinned host) | 58.92 GiB/rank, 235.7 GiB across 4 ranks |
| decode expert traffic | top-8 × 5.06 ≈ **40.5 MiB/token/rank** |

`HOTSPLIT_HOT_GIB` experiment ladder (per rank): **31, then 36, then 41 GiB** — raise only after a
first boot shows free GPU memory (96 GB GDDR7 × 0.90 ≈ 86 GiB usable; hot + KV + activations must fit together).
The real hot *set* is chosen from a routing histogram of your traffic (`--counts`), not from these
averages; `scripts/balance.sh --counts your.json` prints the exact coverage for your workload.

## Files

| file | what |
|---|---|
| `balance.json` | expert geometry (layers, experts, top-k, per-expert bytes, TP) |
| `patches/hotsplit_balance.py` | the balance calculator (numpy only, no GPU) |
| `patches/hotsplit_plan.py` | the unit-tested planning core, byte-identical to MiMo's |
| `patches/README.md` | where the runtime enforcement hook comes from, and what is not ported |
| `scripts/balance.sh` | run the calculator for this model (with or without a histogram) |
| `scripts/preflight-tp4.sh` | host/GPU/NUMA diagnostics before a first boot |
| `scripts/run-tp4-uva.sh` | stock vLLM .28 UVA TP4 + UVA offload baseline (no hot/cold split) |
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
MODEL_DIR=/models/54e52520606f96b3d9fc84088ad22882a61648ac scripts/run-tp4-uva.sh          # baseline, no split
```

Start at 16K–128K context and raise in steps. The hot/cold enforcement hook is described in
`patches/README.md`; until it is ported and checked, `run-tp4-uva.sh` is the honest supported path.

## Verify

`scripts/balance.sh --json` must show `hot + cold == bank` and an integer cell count. On a real
boot, `numastat -p <worker>` shows pinned memory on each worker's GPU NUMA node. Quality bars are
teacher-forced logprobs, BFCL and a needle ladder; none has been run on TP4.

## Results

No measurement on this hardware. Arithmetic only: 40.5. The transfer-only decode ceiling over
PCIe 4.0 (~25 GB/s per GPU) is the cold share of that traffic; the plan line from `balance.sh`
gives the exact coverage for your histogram.

## Known limits

- Not run on target hardware; the NVFP4 (ModelOpt) hot/cold enforcement hook is not ported (see `patches/README.md`).
- `hotsplit_plan.py` assumes global expert ids == local rows: expert parallelism is not supported.
- The anchor above is a byte budget from a measured 7360-slot GPU-memory cache; the *set* must come from your traffic histogram.
- 4 workers pin ~235.7 GiB of host memory at once; keep the host floor generous (128 GiB).
- MTP(2) speculation is untested with any hot/cold split on this hardware.

## Rollback

`scripts/rollback.sh` (stops+removes the hotsplit container; `--start-baseline` also starts the plain UVA baseline). `docker stop glm53-tp4-uva`, or simply do not enable the split — the baseline UVA launch never
moves a weight out of the stock placement. Nothing outside the container is modified.
