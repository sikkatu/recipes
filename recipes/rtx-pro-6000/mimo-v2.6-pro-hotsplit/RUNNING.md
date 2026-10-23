# MiMo-V2.6-Pro on 4× RTX PRO 6000

> **English** | [Русский](RUNNING.ru.md) | [中文](RUNNING.zh.md)


This launches a **stock vLLM TP4 + UVA expert-offload baseline**. It does not claim that the custom Hotsplit patch has been ported or validated for TP4.

Requirements: Linux, NVIDIA driver, Docker + NVIDIA Container Toolkit, four visible GPUs, and hundreds of GB of free disk.

Run:
```bash
MAX_MODEL_LEN=16384 CPU_OFFLOAD_GB=96 scripts/run-tp4-uva.sh
```
For an existing local model:
```bash
MODEL_DIR=$HOME/models/MiMo-V2.6-Pro-MOPD MAX_MODEL_LEN=16384 scripts/run-tp4-uva.sh
```
Default model ID: `XiaomiMiMo/MiMo-V2.6-Pro-MOPD`. The cache is `~/.cache/huggingface`. The launcher starts at 16K context and sends a small API test. Increase context only after startup works (32K, 64K, 128K). It does not explicitly select a compressed KV-cache dtype.

`CPU_OFFLOAD_GB` is the per-rank offload cap, not RAM reserved at startup. Host DDR4 RAM over the PCIe 4.0 link is much slower than the GPU's own 1792 GB/s GDDR7, and slower than a coherent GPU-CPU interconnect.

The Hotsplit code hooks private vLLM internals. TP4 expert placement/ranking/counters must be validated before claiming Hotsplit works on this workstation. This baseline is the first real serving step, not a verified Hotsplit port.

Inspect/stop:
```bash
docker logs -f mimo-pro-tp4-uva
docker stats mimo-pro-tp4-uva
docker stop mimo-pro-tp4-uva
```
API: `http://SERVER_IP:30007/v1`. Do not expose publicly without authentication/firewalling.

## Hotsplit (static / dynamic hot experts)
`scripts/run-tp4-hotsplit.sh` launches the same server with per-expert GPU-residency, optional dynamic updates and NUMA pinning. It **runs end-to-end on target hardware (static mode verified 2026-10-23)**; dynamic mode is not yet verified (the step hook must log `step hook FIRED` on hardware). Start with `HOTSPLIT_DRYRUN=1` and read [README.md](README.md) and `tp4.env.example` first. Check a running container with `scripts/check-hotsplit-tp4.sh`.

## Before trusting PATCH_MODEL_FILES=1 (whole-file model replacement)

The launcher verifies `patches/` against `patches/MANIFEST.sha256` before every launch. Two more
checks are experiments to run on the target host before relying on the replacement:

1. **API/signature conformance with the target vLLM.** Diff the image's own
   `vllm/model_executor/models/mimo_v2.py` / `mimo_v2_omni.py` against the pinned vLLM version the
   patches were written for; confirm `load_weights`, `process_weights_after_loading` and the quant
   method hooks still match (the patches override these wholesale).
2. **Minimal GPU integration smoke test.** With `HOTSPLIT_DRYRUN=1` first: check the logged plan
   (per-layer hot counts, byte budget), then one short generation per rank and verify tensor shapes,
   expert routing counts (`live/counts.rank*.json`) and that the output is sane before real traffic.

## Dynamic-vs-static comparison

With `HOTSPLIT_DYNAMIC=1` the run refuses to start if the step hook cannot be installed (unless
`HOTSPLIT_ALLOW_STATIC_FALLBACK=1`). Every rank logs `dynamic_hook=ACTIVE` or `dynamic_hook=INACTIVE`.
To prove the dynamic counter updates during generation (not just import), generate while watching
`hotsplit dynamic update` lines in `docker logs -f`.

## Proving the dynamic mode works (not just imports)

`check-hotsplit-tp4.sh` reads logs; that is not proof the counter moves. To prove it:

1. Start with `HOTSPLIT_DYNAMIC=1 HOTSPLIT_STRICT_HOOK=1` and confirm every rank logs
   `dynamic_hook=ACTIVE` (a FAILED status with strict mode aborts the run).
2. Generate real text (a few dozen requests), then re-run `scripts/check-hotsplit-tp4.sh`:
   the `hotsplit dynamic update` count must grow and `live/counts.rank*.json` timestamps
   must advance.
3. Compare outputs before/after the swaps: the hot set changes at each update, so a
   mid-run divergence in greedy outputs is expected — compare with the same static run
   as the control, not byte-for-byte against itself.

## Proving NUMA locality (logging is not proof)

`HOTSPLIT_NUMA_POLICY=preferred` is advisory — the kernel may still place pinned pages on
another node under pressure. Only `bind` hard-guarantees locality (and may fail/OOM when the
node is full). To measure the real placement:

1. Run `scripts/numastat-report.sh <container>` while the server is up; for each worker PID
   it prints per-node page counts from `/proc/<pid>/numa_maps` and `numastat -p` when available.
2. Compare three configurations on the same workload (one at a time):
   `HOTSPLIT_NUMA_POLICY=preferred`, `=bind`, and `HOTSPLIT_NUMA=0` (no binding).
3. Measure host-to-device bandwidth on the same payload for each; the difference is the
   real cost/benefit of the binding, not the log line.

## Measuring real effectiveness (update count is not proof)
A moving update counter proves the planner runs; it does NOT prove the dynamic mode pays for
itself. To measure the actual effect, on the same workload and the same request set:
1. Run static (`HOTSPLIT_DYNAMIC=0`) and dynamic (`HOTSPLIT_DYNAMIC=1`) with identical
   `HOTSPLIT_HOT_GIB`, model revision and sampling params.
2. Record TTFT and inter-token latency per request (vLLM logs them per request; `nvidia-smi dmon`
   -s u shows PCIe RX/TX bytes for the card during the run). Cold-bank hits appear as host-link
   traffic; a working dynamic mode moves traffic from the cold path to the hot path over time.
3. Watch the `hotsplit dynamic hit-rate` log line after each update: it reports hot vs cold
   routed-row split and the swap cost. The hot share should trend up against the offline plan;
   if it does not, the live traffic differs from `HOTSPLIT_COUNTS` -- re-run `balance.sh`.
4. Note the live counter includes padded slots of captured batches (`live lane counts incl.
   padded slots` in the snapshot meta). The bias is small at decode sizes but compare static
   and dynamic runs at the same `MAX_NUM_SEQS` and batch shape to cancel it.

## Race check for dynamic swaps (P0-2)
The swap copies run on a dedicated CUDA stream and are published to the compute stream via a
CUDA event, so a decode-graph replay cannot observe a half-applied swap. To verify on hardware:
1. Start with `HOTSPLIT_DYNAMIC=1` and generate continuously while updates fire
   (`HOTSPLIT_DYN_LOG=1` shows each update).
2. Run the same traffic under `compute-sanitizer --tool racecheck` (or `memcheck`) and confirm
   no races are reported between the swap kernels and the MoE kernels.
3. Greedy outputs during a swap window must match a static run with the same inputs up to the
   documented hot-set change; weight-integrity errors (NaN, wrong expert routing) are a FAIL.
