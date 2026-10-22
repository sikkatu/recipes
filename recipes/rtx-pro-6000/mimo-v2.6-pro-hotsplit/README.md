# MiMo-V2.6-Pro — 4× RTX PRO 6000 (TP4): UVA expert offload + hotsplit

> **English** | [Русский](README.ru.md) | [中文](README.zh.md)


**Status: authored, not run on target hardware.** Nothing in `patches/` or `scripts/run-tp4-hotsplit.sh` has been executed on a 4-GPU machine. The pure planning logic has unit tests (`tests/test_hotsplit_tp4.py`); the vLLM/CUDA parts have only been syntax-checked. Treat the first boot as an experiment and send the logs back.

Target host: 4× RTX PRO 6000 Blackwell Workstation Edition — 96 GB GDDR7 with ECC each (1792 GB/s per card). The cards are PCIe Gen 5, but in this machine the host slots train at **PCIe 4.0**, so the GPU-to-host link is Gen 4 (~25 GB/s per GPU); host RAM is DDR4. No NVLink. The CPU is whatever the machine reports (`lscpu`, `/proc/meminfo`).

## Software (pinned image)

**No image has been verified on a 4-GPU target host yet.** The patches replace vLLM model files
wholesale (`PATCH_MODEL_FILES=1`), so the exact vLLM build matters. Until a digest has been
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
| `scripts/run-tp4-uva.sh` | stock vLLM TP4 + layer-granular UVA offload (the baseline, unchanged) |
| `scripts/run-tp4-hotsplit.sh` | same server + hotsplit, static or dynamic, NUMA pinning, live counters |
| `scripts/check-hotsplit-tp4.sh` | reads the container logs and says which stages happened |
| `scripts/manage.sh` | day-to-day control: status / logs / start / stop / restart / update / rollback / check |
| `scripts/preflight-tp4.sh` | diagnostics; now also prints GPU → NUMA node |
| `patches/hotsplit.py` | TP-aware hotsplit with optional dynamic hot set |
| `patches/hotsplit_plan.py` | numpy-only planning helpers (tested) |
| `patches/numa_bind.py`, `patches/sitecustomize.py` | pin each worker to its GPU's NUMA node; optional boot hook |
| `tp4.env.example` | every knob, with the default |

The launcher mounts `mimo_v2_hotsplit.py`, `mimo_v2_omni.py` and `expert_hist_mix.json` from this recipe's `patches/`.

## How it works

Stock `--cpu-offload-gb` decides residency per layer: a layer's experts are either all in GPU memory or all in pinned host memory. hotsplit re-homes them per expert instead: hot experts in GPU memory, the rest in pinned host memory read over UVA, and each MoE layer runs two Marlin launches (one per bank, `expert_map` selects the rows) whose partial outputs are summed. No kernel changes.

**Under TP4** every rank holds a slice of every expert, so every rank splits its own slice with the same hot list (router ids are replicated) and a per-rank byte budget (`HOTSPLIT_HOT_GIB`). Expert parallelism is *not* supported (the maps assume global id == row); the code refuses to split if EP is active.

**Safety gates before any weight is touched** (see `tests/test_hotsplit_safety.py`):

- *Layer collection is keyed by full module path.* A model can carry several `layers.<n>` subtrees (main decoder plus drafter blocks such as `model.mtp.layers.0`). MTP drafter modules are classified by `_is_mtp_module()` and left stock — the routing histogram covers the main decoder only. Two main-decoder modules claiming the same index abort the run instead of silently overwriting each other, and the histogram/model index sets are compared before the first split.
- *VRAM pre-check per layer.* The hot copy is made before the originals are freed, so the transient peak is hot weights **+ hot scales**. The size is measured from the real tensors (`_row_bytes`), compared against free VRAM with a reserve, and the run refuses with a diagnostic error *before* allocating if it cannot fit. Per-layer peak allocated/reserved memory and the bytes returned to the allocator after the originals are dropped are logged (`hotsplit mem`, `hotsplit release`), so a first controlled boot can see the real transient. Note: vLLM computes the KV cache after model load, so a large hot budget can still starve it later — the pre-check covers the split phase, not the whole boot.

**Dynamic mode (`HOTSPLIT_DYNAMIC=1`, default in the launcher).**

- The cold bank is every expert (the original UVA tensors, nothing copied). The hot bank is a fixed-address GDDR7 slot buffer. Two persistent int32 maps `[layers, experts]` say where each expert lives (`emap_hot`: slot or −1, `emap_cold`: row or −1).
- A graph-safe counter counts routed experts at decode.
- Every `HOTSPLIT_DYN_STEPS` engine steps (default 2000), between steps and on the compute stream, each layer swaps up to `HOTSPLIT_DYN_MAX_SWAPS` (default 2) of its weakest hot experts for the strongest cold ones, only if `score_in > margin × score_out` (margin 1.25). Scores are an exponentially decayed count seeded with the offline ranking. The rows are copied host→GPU, then both maps are flipped. Stream order makes the flip atomic with respect to forward passes; CUDA graphs only read the fixed-address maps, so nothing is re-captured.
- All ranks see identical counters at the same step, so they take identical decisions. A rank that diverged would still be correct (the split is internal to the rank) but TP waits for the slowest rank.
- Cost: one stall of roughly the swap bytes over PCIe per update (2 swaps × 69 layers × ~5 MiB per rank ≈ 0.7 GiB, tens of ms at PCIe 4.0 speeds; measure it, it is logged as `hotsplit dynamic update`). Extra GPU memory: one hot copy of the scales (`S` rows per layer) because the cold bank keeps all scales.
- If the step hook cannot be installed (different vLLM layout) the run **refuses** when `HOTSPLIT_DYNAMIC=1` (a silent fallback to static would make a dynamic-vs-static comparison unreliable). Every rank logs an explicit `dynamic_hook=ACTIVE` / `dynamic_hook=INACTIVE` status. To accept a static fallback deliberately, set `HOTSPLIT_ALLOW_STATIC_FALLBACK=1`; the log then warns that the run must not be used to compare dynamic vs static. A live proof that the dynamic counter updates during generation (not just import) is still an experiment step: watch `hotsplit dynamic update` lines in the container log while generating. Scheduler parameters are validated at startup (`HOTSPLIT_DYN_STEPS=0` and other bad values refuse to start instead of failing silently mid-run), repeated planner failures abort the run (`HOTSPLIT_DYN_MAX_PLAN_ERRORS`, default 3), each update logs a hot/cold hit-rate line for effectiveness measurement, and the weight swaps run on a dedicated CUDA stream published through an event so a decode-graph replay cannot observe a half-applied swap (see RUNNING.md for the race-check and measurement procedures).

What this does **not** do: per-token LRU miss filling. A slot-cache design (fused Triton row copy inside the layer, CUDA-graph capturable) reaches a higher LRU hit rate than the best static pin on the same trace, but it is written for the TRT-LLM NVFP4 kernel, and a copy-engine variant (per-miss DMA with a per-layer sync) did not beat the existing path. A miss over PCIe also costs several times more than over a coherent interconnect. Porting that design to Marlin is the obvious next step if dynamic mode shows that residency matters; it is not done here. Dynamic mode in this recipe is a slow adaptation to workload drift, and whether it beats static hotsplit on your traffic is unmeasured.

## Launch

Run the preflight first and check that the four GPUs report sensible NUMA nodes:

```bash
bash scripts/preflight-tp4.sh
```

First boot, plan only (no weights moved, the server runs as the stock UVA baseline; look for one `hotsplit plan [rank N/4]` line per rank):

```bash
MODEL_DIR=$HOME/models/MiMo-V2.6-Pro-MOPD HOTSPLIT_DRYRUN=1 MAX_MODEL_LEN=16384 scripts/run-tp4-hotsplit.sh
docker stop mimo-pro-tp4-hotsplit && docker rm mimo-pro-tp4-hotsplit
```

Then for real, static first, dynamic second:

```bash
MODEL_DIR=$HOME/models/MiMo-V2.6-Pro-MOPD HOTSPLIT_DYNAMIC=0 scripts/run-tp4-hotsplit.sh
scripts/check-hotsplit-tp4.sh
# stop, then:
MODEL_DIR=$HOME/models/MiMo-V2.6-Pro-MOPD HOTSPLIT_DYNAMIC=1 scripts/run-tp4-hotsplit.sh
```

Without `HOTSPLIT_HOT_GIB` the hot budget equals the GPU-memory bytes the stock placement used (KV size unchanged). Set it per rank (45, then 50, 55) only after the first boot shows how much GPU memory is free. Raise `MAX_MODEL_LEN` in steps (32K, 64K, 128K) as in `RUNNING.md`.

## Verify

1. `scripts/check-hotsplit-tp4.sh` — every stage should PASS (the dynamic-hook line is expected to MISS with `HOTSPLIT_DYNAMIC=0`).
2. `numastat -p <worker pid>` for each worker: pinned host memory should sit on the node of that worker's GPU. NUMA binding is best effort (needs `--cap-add SYS_NICE`, torch must expose the PCI bus id); a missing `HOTSPLIT_NUMA ... -> node` line means it did not happen.
3. Quality: teacher-forced logprobs, BFCL and a flood fixture are the right bars; none has been run on TP4. Do not use greedy byte-parity as a regression test (it is not stable on this stack either).
4. Dynamic mode: grep `hotsplit dynamic update`; swaps should decrease once the hot set matches the traffic, and decode speed should not dip visibly at each update.

## Expectations (arithmetic, not measurements)

MiMo-V2.6-Pro routes 8 of 384 experts in each of 69 MoE layers; at ~19 MiB per expert that is ~10 GiB of expert weights per generated token across all ranks, ~2.6 GiB per rank. Every byte not served from GPU memory crosses PCIe: at a 62% hot-hit rate (measured at 31% of cells resident) that is ~1 GiB per token per rank, and at ~25 GB/s per GPU (the PCIe 4.0 link this host runs at) roughly 40 ms per token, i.e. a ceiling in the 20s of tok/s from transfer alone. A bigger hot budget lowers it; the `hotsplit plan` line printed by `HOTSPLIT_DRYRUN=1` gives the offline coverage for your exact budget.

## Known limits and risks

- **OBSERVED on target hardware (2026-10-22): the UVA baseline answers with garbage.** The server starts, `/v1/models` and `/v1/chat/completions` respond, but generated text is meaningless (zero tokens, then random tokens across languages). The attention weights (resident on GPU) work; the routed-expert weights (116 GiB offloaded to pinned host memory via UVA) are read incorrectly — most likely the in-place MXFP4 read path (`MOE_BACKEND=humming`) does not handle UVA host pointers. **Do not use these outputs.** Under investigation: next step is the same run with `PATCH_MODEL_FILES=1 HOTSPLIT_DRYRUN=1` (patched model code, no weight moves) to separate stock-model-code errors from the UVA read path.
- **OBSERVED on target hardware (2026-10-22):** the Marlin repack (`prepare_moe_mxfp4_layer_for_marlin` -> `repack_weight`) creates new GPU tensors that `--cpu-offload-params` does NOT cover. With `--moe-backend marlin` + UVA offload the repacked weights accumulate in VRAM until `torch.empty(1.12 GiB)` dies with CUDA OOM (92.7 GiB allocated of 94.97). With `MOE_BACKEND=triton` the same run dies even earlier: `ValueError: Mxfp4 MoE backend 'TRITON' does not support ... MoEActivation.SILU activation` (MiMo uses gated SiLU). The UVA baseline therefore defaults to `MOE_BACKEND=humming` — the only backend in this vLLM build that both supports `MoEActivation.SILU` and avoids the GPU repack (Humming reads the MXFP4 weights in place). `run-tp4-hotsplit.sh` keeps marlin because the patches are written against Marlin kernels — expect this repack cost and plan VRAM for it.
- `PATCH_MODEL_FILES=1` mounts `mimo_v2_hotsplit.py` / `mimo_v2_omni.py` over vLLM's own model files — a whole-file replacement, not a small hook, so its compatibility with the exact vLLM build is critical. The launcher therefore verifies every mounted patch against `patches/MANIFEST.sha256` before launch and refuses on any mismatch. The remaining two checks (signature/API conformance vs the target vLLM, and a GPU integration smoke test of tensor shapes / routing / output) are experiments to run on the target host before trusting the patch; both are listed in RUNNING.md.
- `mimo_v2_hotsplit.py` is a full replacement of vLLM's `mimo_v2.py` for the pinned nightly image; its fused-fp8-qkv sharding path (`tp_rank`) was only exercised at TP1. If TP4 output is garbage, retry with `PATCH_MODEL_FILES=0` (image's own model file, hook via `sitecustomize`) to separate the two causes.
- The ranking `expert_hist_mix.json` comes from agent traffic on a previous single-GPU lane; re-rank for your traffic (`live/counts.rank0.json` + `patches/hotsplit_rerank.py`). Dynamic mode adapts within a layer's fixed slot count only; moving slots between layers needs a restart with a new ranking.
- Counters include padded slots of captured batches (negligible at `--max-num-seqs 1`, a small bias above that).
- 4 workers pin ~hundreds of GiB of host memory at once; keep `HOTSPLIT_HOST_FLOOR_GIB` generous (launcher default 128).
- EP is unsupported. Speculative decoding is untested here.

## Rollback

`scripts/rollback.sh` (stops+removes the hotsplit container; `--start-baseline` also starts the plain UVA baseline). `docker stop mimo-pro-tp4-hotsplit`, then run `scripts/run-tp4-uva.sh` (stock baseline) or relaunch with `HOTSPLIT_COUNTS` unset in the container (edit the launcher call) — nothing outside the container is modified.
