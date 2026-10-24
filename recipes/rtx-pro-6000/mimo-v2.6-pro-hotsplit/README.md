# MiMo-V2.6-Pro — 4× RTX PRO 6000 (TP4): UVA expert offload + hotsplit

> **English** | [Русский](README.ru.md) | [中文](README.zh.md)


**Status: VERIFIED on target hardware (2026-10-22).** The UVA baseline (`run-tp4-uva.sh`) runs end-to-end on the 4-GPU host: weights load, the API answers, and generation is coherent (A/B-confirmed against the stock model loader). `run-tp4-hotsplit.sh` on `MOE_BACKEND=humming` **works end-to-end on the target hardware (2026-10-23)**: all 69 layers split on 4 ranks (hot bank in GPU, cold bank in UVA), the server starts and the smoke test returns coherent output (`TP4_OK`, `finish_reason: stop`). Three bank-format bugs were fixed to get there — per-bank expert count (`c6dfb58`), per-bank `LayerConfig` (`dataclasses.replace(cfg, num_experts=n_rows)`) and per-bank scales (`32d3373`) — see `patches/hotsplit.py::_patch_humming_bank`. The Marlin repack path still OOMs (see Known limits). **Dynamic mode is verified on target hardware too (2026-10-23)**: with `HOTSPLIT_DYNAMIC=1` the hot set follows live routing — 3 planning windows over ~6,700 engine steps, 3 transactional updates (138 expert swaps per update across 69 layers, 43-74 ms each, `errors=0`), hot-bank hit rate 41.6->45.4% on a 50-request mixed-traffic run at 6.6 tok/s average (measured 2026-10-23), all answers coherent. The step hook fires on both GPUModelRunner implementations (V1 and V2) and the swap path handles MXFP4 `Float8_e8m0fnu` scales via a uint8 view. Output quality is smoke-verified; run longer quality checks before production use.

Target host: 4× RTX PRO 6000 Blackwell Workstation Edition — 96 GB GDDR7 with ECC each (1792 GB/s per card). The cards are PCIe Gen 5, but in this machine the host slots train at **PCIe 4.0**, so the GPU-to-host link is Gen 4 (~25 GB/s per GPU); host RAM is DDR4. No NVLink. The CPU is whatever the machine reports (`lscpu`, `/proc/meminfo`).

## Software (pinned image)

**The pinned image IS verified on the 4-GPU target host (2026-10-22).** The patches replace vLLM model files
wholesale (`PATCH_MODEL_FILES=1`), so the exact vLLM build matters — the digest below is the one
that was validated end-to-end. If you switch to a different build, treat that choice as an experiment:

- Start from the vLLM release whose API the patches were written against (see
  `patches/mimo_v2_hotsplit.py` docstrings for the expected internals: `Mxfp4MoEMethod`,
  `make_mxfp4_moe_kernel`, `GPUModelRunner.execute_model`).
- Concrete image (from the vLLM nightly the patches were written against):
  `IMAGE="vllm/vllm-openai:nightly-8e2beaa3bff595577a069d49d96bde9074294ec2@sha256:167ccd8696bc09f1b63301185134e04b883623f8adcbd8e3a0ec3140a193abcf"`
  (tag carries the vLLM commit; digest pins the exact bytes).
- The launcher checks that some expected methods exist in the mounted vLLM sources before
  launching (SITE probe), but this is a smoke test, not a compatibility proof: signatures,
  sharding internals and Marlin kernel expectations are NOT verified automatically.

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

All commands are copy-paste ready. Run them from `recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit/`.

### 1. Preflight

```bash
bash scripts/preflight-tp4.sh
```

### 2. UVA baseline (no hot/cold split)

```bash
export MODEL_DIR="$HOME/models/MiMo-V2.6-Pro-MOPD"
export IMAGE="vllm/vllm-openai:nightly-8e2beaa3bff595577a069d49d96bde9074294ec2@sha256:167ccd8696bc09f1b63301185134e04b883623f8adcbd8e3a0ec3140a193abcf"
export CPU_OFFLOAD_GB=128 GPU_UTIL=0.85   # MAX_MODEL_LEN defaults to 1048576 (1M context)
bash scripts/run-tp4-uva.sh
```

After startup (~13 min):
```bash
bash scripts/check-tp4-api.sh
```

### 3. Hotsplit (static, verified 2026-10-23)

Stop the baseline first (same port):
```bash
docker stop mimo-pro-tp4-uva && docker rm mimo-pro-tp4-uva
```

Then launch hotsplit. `HOTSPLIT_HOT_GIB` is **required** here: with full UVA offload the
default hot budget is 0 (empty hot set — a silent no-op). 30 GiB per rank is a safe start;
adjust after the first boot shows how much GPU memory is free.

```bash
export MODEL_DIR="$HOME/models/MiMo-V2.6-Pro-MOPD"
export IMAGE="vllm/vllm-openai:nightly-8e2beaa3bff595577a069d49d96bde9074294ec2@sha256:167ccd8696bc09f1b63301185134e04b883623f8adcbd8e3a0ec3140a193abcf"
export CPU_OFFLOAD_GB=128 GPU_UTIL=0.85   # MAX_MODEL_LEN defaults to 1048576 (1M context)
export HOTSPLIT_DYNAMIC=0 HOTSPLIT_HOT_GIB=30
bash scripts/run-tp4-hotsplit.sh
```

After startup (~13 min):
```bash
bash scripts/check-hotsplit-tp4.sh mimo-pro-tp4-hotsplit static
bash scripts/check-tp4-api.sh
```

### 4. Dynamic experiment (separate run, not yet quality-verified)

```bash
docker stop mimo-pro-tp4-hotsplit && docker rm mimo-pro-tp4-hotsplit
export HOTSPLIT_DYNAMIC=1
bash scripts/run-tp4-hotsplit.sh
```

### 5. Raw API queries (quality comparison)

```bash
KEY="$(cat live/api_key)"
URL="http://127.0.0.1:30007/v1/chat/completions"
curl -s "$URL" -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"mimo-pro","messages":[{"role":"user","content":"What is the capital of France? Answer in one sentence."}],"max_tokens":100,"temperature":0}'
```

For a longer quality comparison, run the same queries against the UVA baseline and compare
the `content` fields. `temperature=0` should give identical output for short factual answers.
## Verify

1. `scripts/check-hotsplit-tp4.sh` — every stage should PASS (the dynamic-hook line is expected to MISS with `HOTSPLIT_DYNAMIC=0`).
2. `numastat -p <worker pid>` for each worker: pinned host memory should sit on the node of that worker's GPU. NUMA binding is best effort (needs `--cap-add SYS_NICE`, torch must expose the PCI bus id); a missing `HOTSPLIT_NUMA ... -> node` line means it did not happen.
3. Quality: teacher-forced logprobs, BFCL and a flood fixture are the right bars; none has been run on TP4. Do not use greedy byte-parity as a regression test (it is not stable on this stack either).
4. Dynamic mode: grep `hotsplit dynamic update`; swaps should decrease once the hot set matches the traffic, and decode speed should not dip visibly at each update.

## Expectations (arithmetic, not measurements)

MiMo-V2.6-Pro routes 8 of 384 experts in each of 69 MoE layers; at ~19 MiB per expert that is ~10 GiB of expert weights per generated token across all ranks, ~2.6 GiB per rank. Every byte not served from GPU memory crosses PCIe: at a 62% hot-hit rate (measured at 31% of cells resident) that is ~1 GiB per token per rank, and at ~25 GB/s per GPU (the PCIe 4.0 link this host runs at) roughly 40 ms per token, i.e. a ceiling in the 20s of tok/s from transfer alone. A bigger hot budget lowers it; the `hotsplit plan` line printed by `HOTSPLIT_DRYRUN=1` gives the offline coverage for your exact budget.

## Known limits and risks

- **RESOLVED on target hardware (2026-10-22): the UVA baseline used to answer with garbage - root cause found and fixed.** A/B with identical weights and flags (`PATCH_MODEL_FILES=1` vs `0`, `HOTSPLIT_DRYRUN=1`, `MOE_BACKEND=humming`): the patched model files produce coherent text (" Paris. In the early Middle Ages, Paris was the largest city in the western world...") while the image's own `mimo_v2.py` produces training-corpus fragments ("            0     0,   0,  00 "). Root cause: the stock `load_weights` silently drops fused `qkv_proj` weight+`weight_scale_inv` pairs (they can straddle two `AutoWeightsLoader` calls and are then dropped); the patched loader keeps those pairs together. `run-tp4-uva.sh` now mounts the fixed model files by default (`PATCH_MODEL_FILES=1`). Attention backend `TRITON_ATTN_DIFFKV` and `MOE_BACKEND=humming` confirmed working; the checkpoint weights were never corrupt.
- **OBSERVED on target hardware (2026-10-22):** the Marlin repack (`prepare_moe_mxfp4_layer_for_marlin` -> `repack_weight`) creates new GPU tensors that `--cpu-offload-params` does NOT cover. With `--moe-backend marlin` + UVA offload the repacked weights accumulate in VRAM until `torch.empty(1.12 GiB)` dies with CUDA OOM (92.7 GiB allocated of 94.97). With `MOE_BACKEND=triton` the same run dies even earlier: `ValueError: Mxfp4 MoE backend 'TRITON' does not support ... MoEActivation.SILU activation` (MiMo uses gated SiLU). The UVA baseline therefore defaults to `MOE_BACKEND=humming` — the only backend in this vLLM build that both supports `MoEActivation.SILU` and avoids the GPU repack (Humming reads the MXFP4 weights in place). `run-tp4-hotsplit.sh` now defaults to `MOE_BACKEND=humming` too (marlin + UVA is a known OOM and is refused unless `HOTSPLIT_ALLOW_MARLIN_UVA=1`). Note the full split bank code is Marlin-specific: on humming the split is an EXPERIMENT — verify output before trusting it.
- `PATCH_MODEL_FILES=1` mounts `mimo_v2_hotsplit.py` / `mimo_v2_omni.py` over vLLM's own model files — a whole-file replacement, not a small hook, so its compatibility with the exact vLLM build is critical. The launcher therefore verifies every mounted patch against `patches/MANIFEST.sha256` before launch and refuses on any mismatch. The remaining two checks (signature/API conformance vs the target vLLM, and a GPU integration smoke test of tensor shapes / routing / output) are experiments to run on the target host before trusting the patch; both are listed in RUNNING.md.
- `mimo_v2_hotsplit.py` is a full replacement of vLLM's `mimo_v2.py` for the pinned nightly image. Its fused-fp8-qkv sharding path (`tp_rank`) is exercised at TP4 by the fixed loader (2026-10-22). `PATCH_MODEL_FILES=0` keeps the image's own model file - that is exactly the configuration that emits training-corpus fragments (see the RESOLVED item above); use it only to reproduce the upstream bug on purpose.
- The ranking `expert_hist_mix.json` comes from agent traffic on a previous single-GPU lane; re-rank for your traffic (`live/counts.rank0.json` + `patches/hotsplit_rerank.py`). Dynamic mode adapts within a layer's fixed slot count only; moving slots between layers needs a restart with a new ranking.
- Counters include padded slots of captured batches (negligible at `--max-num-seqs 1`, a small bias above that).
- 4 workers pin ~hundreds of GiB of host memory at once; keep `HOTSPLIT_HOST_FLOOR_GIB` generous (launcher default 128).
- EP is unsupported. Speculative decoding is untested here.
- **NUMA pinning: usually inert in VMs, optional `HOTSPLIT_NUMA_MAP` for experts.** On a dual-socket host emulated by Proxmox, `nvidia-smi topo -m` shows all GPUs on `PHB` (one PCIe host bridge) with `GPU NUMA ID: N/A` — the VM does not expose which socket each GPU is physically on. `numa_bind` therefore logs `numa_node unknown; not binding` and skips pinning. This is **correct and safe**: guessing the map wrong would pin workers to the wrong socket and make things worse. DDR4 locality (which stick is on which socket) is a host-side concern — check with `numactl --hardware` and `lstopo` on the Proxmox host (not inside the VM). If you know the physical topology for sure, `HOTSPLIT_NUMA_MAP="0:0,1:0,2:1,3:1"` (gpu:node pairs) enables the binding explicitly.
- **Prefix caching is ON by default** (`enable_prefix_caching=True` in the pinned vLLM build). Repeated prompt prefixes (system prompts, conversation history) are served from KV cache — the "cheap cached tokens" behaviour of cloud providers. `scripts/check-prefix-cache.sh` verifies caching two ways: Prometheus counters from `/metrics` when available, otherwise the engine log line `Prefix cache hit rate: X%` (printed periodically under load). It also prints decode speed (tok/s) per call.
- **1M-token context** (`MAX_MODEL_LEN=1048576`) is supported by the model config (`max_position_embeddings=1048576`, 9 full-attention + 61 sliding-window layers). KV cost on TP4 (2 KV-heads per rank, GQA 128:8): **~13.5 GiB/GPU at bf16** (recommended — full precision, fits the 81 GiB budget), ~6.8 GiB/GPU at fp8 (only if you need the headroom). Sliding-window layers add a constant ~11 MiB cap regardless of length. Verified arithmetic and config, not a live 1M run. 1M is the launcher default (`MAX_MODEL_LEN=1048576`, `MAX_NUM_BATCHED_TOKENS=8192` for long prefill); override with `MAX_MODEL_LEN=16384` if you want the smaller KV footprint.

## Rollback

`scripts/rollback.sh` (stops+removes the hotsplit container; `--start-baseline` also starts the plain UVA baseline). `docker stop mimo-pro-tp4-hotsplit`, then run `scripts/run-tp4-uva.sh` (stock baseline) or relaunch with `HOTSPLIT_COUNTS` unset in the container (edit the launcher call) — nothing outside the container is modified.
