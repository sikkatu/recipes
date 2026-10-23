# DeepSeek-V4.1-Flash — 4× RTX PRO 6000（96 GB），TP4：UVA 专家卸载的专家均衡
> [English](README.md) | [Русский](README.ru.md) | **中文**

**状态：工程级均衡计算 + 工具，非经验证的部署运行。** 这里真实且经检查的是*专家均衡*：
哪些路由专家放在显存、哪些放在主机锁定内存，以及每 rank 字节预算 —— 用与
`recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit` 完全相同、有单元测试的规划器计算（按
"路由占比/字节数"贪心选热门集）。执行切分的 vLLM 运行时钩子依赖模型和量化方式，
**不声称**已移植或在本硬件上验证（见 `patches/README.md`）。每次启动都当实验对待，请回传日志。

目标主机：4× RTX PRO 6000 Blackwell Workstation Edition —— 每卡 96 GB GDDR7 带 ECC
（1792 GB/s）。卡是 PCIe Gen 5，但主机插槽协商为 **PCIe 4.0**，GPU↔主机链路为 Gen 4
（每 GPU 约 25 GB/s）；主机内存 DDR4。无 NVLink。

## 运行内容

DeepSeek-V4.1-Flash 在 4× RTX PRO 6000（96 GB GDDR7，主机链路 PCIe 4.0，无 NVLink）上：
vLLM TP4 + UVA 专家卸载 —— 即 **UVA 基线（逐层卸载）+ 离线 hot/cold 均衡规划器**。
实际运行是原版 vLLM TP4 逐层 UVA 卸载。逐专家 hot/cold 切分在此规划和估算
（`patches/hotsplit_balance.py`、`scripts/balance.sh`），算法与 MiMo-V2.6-Pro 配方相同，
但运行时钩子（`patches/hotsplit.py`）**尚未被这些启动脚本挂载** —— 配方提供规划器和补丁，
而非运行中的切分。下面的 hot/cold 数字是规划值，不是实测值。

- 模型 `deepseek-ai/DeepSeek-V4.1-Flash` @ `df42c109f1defefcbfcedbe7d905718a12266e40`
- 量化：MXFP4（I8+E8M0 块缩放）（按原样提供，未重新量化）
- 草稿模型：DSpark（mtp 0-2）

## 专家均衡（本配方的核心数字）

来自 `deepseek-ai/DeepSeek-V4.1-Flash` 配置的几何：**40 个 MoE 层 × 384 个路由专家**（+1 共享），
`num_experts_per_tok` = 6，hidden 5120，`moe_intermediate_size` 2304 → 每个路由专家 35.39M 参数
= **17.9 MiB** 完整。TP4 下每个 rank 持有 ¼ 分片 → 每专家切片 **4.47 MiB**。
`balance.json` 是机器可读形式；`scripts/balance.sh` 重新计算表格。

| 量 | 值（TP4 每 rank） |
|---|---|
| 路由专家单元数 | 40 × 384 = 15,360 |
| 单专家切片 | 4.47 MiB（17.9 MiB 完整 ÷ 4） |
| 完整路由专家库 | 67.12 GiB |
| **热门库（显存）** | **~52 GiB**（≈11,899 单元，78%）—— 锚点来自先前单 GPU 通道测得的 pin-hot-experts 切分（384 中的 295） |
| 冷门库（主机锁定） | 15.12 GiB/rank，4 rank 共 60.5 GiB |
| 解码专家流量每层 | top-6 × 4.47 ≈ 26.8 MiB/层/token/rank |
| 解码专家流量每 token | 40 层 × 26.8 ≈ **1072 MiB/token/rank（~1.05 GiB）** |
| PCIe 4.0 上限（全冷） | 1072 MiB ÷ 25 GB/s ≈ **~23 tok/s** |

`HOTSPLIT_HOT_GIB` 实验梯度（每 rank）：**47，然后 52，然后 57 GiB** —— 只在首次启动显示
剩余显存后再提高（96 GB GDDR7 × 0.90 ≈ 86 GiB 可用；hot + KV + 激活须共存）。

实际热门*集合*根据你的流量路由直方图（`--counts`）选择，不是这些平均值；
`scripts/balance.sh --counts your.json` 打印你的负载的精确覆盖率。

## 软件（固定镜像）

**尚无镜像在 4-GPU 目标主机上验证过。** 这些启动脚本运行原版 vLLM 模型代码（无
`PATCH_MODEL_FILES`）；`patches/` 中的 hot/cold 补丁只在你自己接入运行时钩子时才生效。
在 digest 端到端验证之前，请将镜像选择当实验对待：

- 从补丁编写所针对的 vLLM 版本开始（见 `patches/mimo_v2_hotsplit.py` 文档字符串中的
  期望内部件：`Mxfp4MoEMethod`、`make_mxfp4_moe_kernel`、`GPUModelRunner.execute_model`）。
- 具体镜像（来自补丁编写所针对的 vLLM nightly）：
  `IMAGE="vllm/vllm-openai:nightly-8e2beaa3bff595577a069d49d96bde9074294ec2@sha256:167ccd8696bc09f1b63301185134e04b883623f8adcbd8e3a0ec3140a193abcf"`
  （tag 携带 vLLM 提交；digest 固定精确字节）。
- 启动脚本在启动前检查挂载的 vLLM 源码中存在某些预期方法（SITE 探针），但这是 smoke 测试，
  不是兼容性证明：签名、分片内部实现和 Marlin 内核期望不会自动验证。

**不要**假设随机 tag "能用" —— 先在你的硬件上验证。

## 文件

| 文件 | 说明 |
|---|---|
| `balance.json` | 专家几何（层数、专家数、top-k、每专家字节、TP） |
| `patches/hotsplit_balance.py` | 均衡计算器（仅 numpy，无 GPU） |
| `patches/hotsplit_plan.py` | 有单元测试的规划核心，与 MiMo 逐字节一致 |
| `patches/README.md` | 运行时执行钩子的来源，以及未移植的部分 |
| `scripts/balance.sh` | 为本模型运行计算器（带或不带直方图） |
| `scripts/preflight-tp4.sh` | 首次启动前的主机/GPU/NUMA 诊断 |
| `scripts/manage.sh` | 日常管理：status / logs / start / stop / restart / update / rollback / check |
| `scripts/run-tp4-uva.sh` | 原版 vLLM TP4 + UVA 基线（无 hot/cold 切分） |
| `tp4.env.example` | 每个旋钮及其默认值 |

## 启动

先计算并检查均衡（不需要 GPU）：

```bash
scripts/balance.sh                 # 均匀先验，在中等预算打印 hot/cold 切分
scripts/balance.sh --counts my_traffic.json --gib 52
```

然后运行时，先 preflight，移动任何权重之前仅做规划：

```bash
bash scripts/preflight-tp4.sh
MODEL_DIR=$HOME/models/DeepSeek-V4.1-Flash scripts/run-tp4-uva.sh    # 基线，无切分
```

从 16K–128K 上下文开始，逐步提高。hot/cold 执行钩子在 `patches/README.md` 中描述；
在移植和检查之前，`run-tp4-uva.sh` 是诚实的支持路径。

## 验证

`scripts/balance.sh --json` 必须显示 `hot + cold == bank` 和整数单元计数。实际启动时
`numastat -p <worker>` 显示每个 worker 的 GPU NUMA 节点上的锁定内存。质量基线是
teacher-forced logprobs、BFCL 和针状阶梯；均未在 TP4 上运行过。

## 结果

本硬件上无测量值。仅算术：26.8 MiB/层/token/rank × 40 层 = ~1072 MiB/token/rank。
仅传输的解码上限（PCIe 4.0，每 GPU 约 25 GB/s）是该流量的冷部分；
`balance.sh` 的规划行给出你的直方图的精确覆盖率。

## 已知限制

- 未在目标硬件上运行；MXFP4（I8+E8M0 块缩放）hot/cold 执行钩子未移植（见 `patches/README.md`）。
- `hotsplit_plan.py` 假设全局专家 id == 本地行：不支持专家并行（EP）。
- 上述锚点是来自先前单 GPU 通道测得的 pin-hot-experts 切分（295/384）的字节预算；
  *集合*必须来自你的流量直方图。
- 4 个 worker 同时锁定约 60.5 GiB 主机内存；请保留充足的主机余量（128 GiB）。
- DSpark（mtp 0-2）投机解码在本硬件上与任何 hot/cold 切分组合均未测试。

## 回退

`scripts/rollback.sh`（停止并移除 hotsplit 容器；`--start-baseline` 也启动纯 UVA 基线）。
`docker stop deepseek-flash-tp4-uva`，或干脆不启用切分 —— UVA 基线启动从不移动权重离开
原版放置位置。容器外不做任何修改。