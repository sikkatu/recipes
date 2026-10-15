# GLM-5.3-Flash — 4× RTX PRO 6000 (96 GB), TP4: expert balance for UVA expert offload

**Status: engineering balance + tooling, not a verified serving run.** What is real and checked
here is the *expert balance* — which routed experts belong in GPU memory and which in pinned host
memory, and the per-rank byte budgets — computed with the exact same unit-tested planner as
`recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit` (greedy share-per-byte hot set). The **vLLM**
runtime hook that *enforces* the split is model- and quant-specific and is **not** claimed to be
ported or validated on this hardware (see `patches/README.md`). Treat every launch as an
experiment and send the logs back.

> **⚠️ Upstream blocker (SM120 / RTX PRO 6000).** Vanilla vLLM cannot serve `glm5_next`
> (GLM-5.3-Flash) on compute capability 12.0: rope-free sparse MLA (`qk_rope_head_dim=0`) has
> no working attention/KV path there — every boot dies at
> `concat_and_cache_mla: pe_dim must be 64 for fp8_ds_mla`. See
> [vllm#53963](https://github.com/vllm-project/vllm/issues/53963) and
> [vllm#55773](https://github.com/vllm-project/vllm/issues/55773). Candidate fixes
> (vllm#55277, vllm#55778, flashinfer#4802) are **not merged upstream**. This recipe therefore
> ships the balance + a vLLM UVA baseline launcher that expects a vLLM build with one of those
> fixes; the launcher prints the warning before every start. Details in `patches/README.md`.

## What this runs

GLM-5.3-Flash on 4× RTX PRO 6000 (96 GB GDDR7, PCIe 4.0 host link, no NVLink): **vLLM** TP4 + UVA expert offload with a
per-expert hot/cold split — hot routed experts in GPU memory, the rest read from pinned host memory
over PCIe. Same idea and same balance algorithm as the MiMo-V2.6-Pro hotsplit recipe, with this
model's geometry.

- model `nvidia/GLM-5.3-Flash-NVFP4` @ `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`
- quantization: NVFP4 (NVIDIA ModelOpt) (as shipped, nothing re-quantized)
- draft: MTP via vLLM `--speculative-config` (optional; untested with hot/cold split). The SGLang-only DFlash2 draft is not used — this recipe is vLLM throughout.

## Expert balance (the numbers this recipe is about)

Geometry from `nvidia/GLM-5.3-Flash-NVFP4` config: **42 MoE layers × 288 routed experts** (+1 shared),
`num_experts_per_tok` = 8, hidden 4096, `moe_intermediate_size` 2048 → 25.17M params
per routed expert = **13.5 MiB** full. At TP4 a rank holds a ¼ shard → **3.38 MiB** per
expert slice. `balance.json` is the machine-readable form; `scripts/balance.sh` recomputes the table.

| quantity | value (per rank at TP4) |
|---|---|
| routed expert cells | 42 × 288 = 12,096 |
| one expert slice | 3.38 MiB (13.5 MiB full ÷ 4) |
| full routed bank | 39.87 GiB |
| **hot bank (GPU memory)** | **~31 GiB** (≈9,406 cells, 78%) — anchored on the same 76.8% hot fraction as MiMo/DeepSeek |
| cold bank (pinned host) | 8.87 GiB/rank, 35.5 GiB across 4 ranks |
| decode expert traffic | top-8 × 3.38 ≈ **27.0 MiB/token/rank** |

`HOTSPLIT_HOT_GIB` experiment ladder (per rank): **26, then 31, then 36 GiB** — raise only after a
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
| `scripts/run-tp4-uva.sh` | stock **vLLM** TP4 + UVA offload baseline (no hot/cold split) |
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
MODEL_DIR=/models/09b04e5e74bca08ca8549fc736d4cdd8624bfde3 scripts/run-tp4-uva.sh          # baseline, no split
```

Start at 16K–128K context and raise in steps. The hot/cold enforcement hook is described in
`patches/README.md`; until it is ported and checked, `run-tp4-uva.sh` is the honest supported path.

## Verify

`scripts/balance.sh --json` must show `hot + cold == bank` and an integer cell count. On a real
boot, `numastat -p <worker>` shows pinned memory on each worker's GPU NUMA node. Quality bars are
teacher-forced logprobs, BFCL and a needle ladder; none has been run on TP4.

## Results

No measurement on this hardware. Arithmetic only: 27.0. The transfer-only decode ceiling over
PCIe 4.0 (~25 GB/s per GPU) is the cold share of that traffic; the plan line from `balance.sh`
gives the exact coverage for your histogram.

## Known limits

- Not run on target hardware; the NVFP4 (NVIDIA ModelOpt) hot/cold enforcement hook is not ported (see `patches/README.md`).
- The engine is **vLLM** throughout (no SGLang); the runtime hook targets vLLM's NVFP4 MoE path.
- `hotsplit_plan.py` assumes global expert ids == local rows: expert parallelism is not supported.
- The anchor above is a byte budget from the same 76.8% hot fraction as MiMo/DeepSeek; the *set* must come from your traffic histogram.
- 4 workers pin ~35.5 GiB of host memory at once; keep the host floor generous (128 GiB).
- MTP speculation is untested with any hot/cold split on this hardware.
- **BLOCKER:** vanilla vLLM cannot serve this model on SM120 (see the banner above and `patches/README.md`); the baseline launcher requires a vLLM build carrying the SM120 NoPE sparse-MLA fix.

## Rollback

`scripts/rollback.sh` (stops+removes the hotsplit container; `--start-baseline` also starts the plain UVA baseline). `docker stop glm-flash-tp4-uva`, or simply do not enable the split — the baseline UVA launch never
moves a weight out of the stock placement. Nothing outside the container is modified.
