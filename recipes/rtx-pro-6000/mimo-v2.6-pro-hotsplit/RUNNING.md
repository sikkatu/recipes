# MiMo-V2.6-Pro on 4× RTX PRO 6000

This launches a **stock vLLM TP4 + UVA expert-offload baseline**. It does not claim that the custom Hotsplit patch has been ported or validated for TP4.

Requirements: Linux, NVIDIA driver, Docker + NVIDIA Container Toolkit, four visible GPUs, and hundreds of GB of free disk.

Run:
```bash
MAX_MODEL_LEN=16384 CPU_OFFLOAD_GB=96 scripts/run-tp4-uva.sh
```
For an existing local model:
```bash
MODEL_DIR=/models/MiMo-V2.6-Pro-MOPD MAX_MODEL_LEN=16384 scripts/run-tp4-uva.sh
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
`scripts/run-tp4-hotsplit.sh` launches the same server with per-expert GPU-residency, optional dynamic updates and NUMA pinning. It is authored but **not run on target hardware**; start with `HOTSPLIT_DRYRUN=1` and read [README.md](README.md) and `tp4.env.example` first. Check a running container with `scripts/check-hotsplit-tp4.sh`.

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
