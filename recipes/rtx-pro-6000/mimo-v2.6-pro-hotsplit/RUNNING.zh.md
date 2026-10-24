# MiMo-V2.6-Pro on 4× RTX PRO 6000

> [English](RUNNING.md) | [Русский](RUNNING.ru.md) | **中文**

启动的是**原版 vLLM TP4 + UVA 专家卸载基线**。不声称自定义 Hotsplit 补丁已移植或
验证于 TP4。

要求：Linux、NVIDIA 驱动、Docker + NVIDIA Container Toolkit、四块可见 GPU，数百 GB 磁盘。

运行：
```bash
CPU_OFFLOAD_GB=128 scripts/run-tp4-uva.sh
```
本地已有模型：
```bash
MODEL_DIR=$HOME/models/MiMo-V2.6-Pro-MOPD scripts/run-tp4-uva.sh
```

默认模型：`XiaomiMiMo/MiMo-V2.6-Pro-MOPD`。缓存 `~/.cache/huggingface`。
16K 上下文 + 小型 API 测试。启动成功后再提高上下文（32K、64K、128K）。
`CPU_OFFLOAD_GB` 是每 rank 卸载上限，不是启动时的内存预留。

## Hotsplit（static / dynamic 热门专家）
见配方 README 与 [Wiki](https://github.com/sikkatu/recipes/wiki/Quick-Start.zh)。

## 信任 PATCH_MODEL_FILES=1 之前
启动脚本在启动前按 `patches/MANIFEST.sha256` 验证每个补丁。

## dynamic 与 static 对比
相同 `HOTSPLIT_HOT_GIB`、修订版与采样参数。

## 证明 dynamic 真的在工作
1. 以 `HOTSPLIT_DYNAMIC=1 HOTSPLIT_STRICT_HOOK=1` 启动，每 rank `dynamic_hook=ACTIVE`。
2. 生成真实文本，然后 `scripts/check-hotsplit-tp4.sh` —— 计数应增长。
3. 输出对比 —— 以 static 运行为对照。

## 证明 NUMA 局部性
`numastat -p <pid>` + `scripts/numastat-report.sh`；对比 preferred / bind / off + H2D 基准。

## 测量真实效果
更新计数不是证据。测量 TTFT/TPOT、`nvidia-smi dmon` 的 PCIe 流量，
观察每次更新的 `hotsplit dynamic hit-rate` 行。

## 动态换入换出的竞态检查（P0-2）
拷贝在专用 CUDA 流上执行，通过 event 发布。检查：
`compute-sanitizer --tool racecheck` 负载下运行；权重完整性必须保证。
