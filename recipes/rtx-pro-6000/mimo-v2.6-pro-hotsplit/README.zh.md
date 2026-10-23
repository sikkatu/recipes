# MiMo-V2.6-Pro — 4× RTX PRO 6000（TP4）：UVA 专家卸载 + hotsplit
> [English](README.md) | [Русский](README.ru.md) | **中文**

**状态：已在目标硬件上验证（2026-10-22）。** UVA 基线（`run-tp4-uva.sh`）在 4-GPU 主机上端到端运行：
权重加载正常，API 有响应，生成连贯（相对原版模型加载器的 A/B 已确认）。
`run-tp4-hotsplit.sh` 在 `MOE_BACKEND=humming` 下**已在目标硬件上端到端工作（2026-10-23）**：
全部 69 层在 4 个 rank 上拆分（hot 库在 GPU，cold 库在 UVA），服务器启动，smoke 测试返回
连贯输出（`TP4_OK`，`finish_reason: stop`）。为此修复了三个库格式 bug —— 每库专家计数
（`c6dfb58`）、每库 `LayerConfig`（`dataclasses.replace(cfg, num_experts=n_rows)`）和每库 scales
（`32d3373`）—— 见 `patches/hotsplit.py::_patch_humming_bank`。Marlin 重打包路径仍会 OOM
（见“已知限制”）。输出质量已通过 smoke 验证；投产前请做更长的质量检查。

目标主机：4× RTX PRO 6000 Blackwell Workstation Edition —— 每卡 96 GB GDDR7 带 ECC
（1792 GB/s）。卡是 PCIe Gen 5，但本机主机插槽协商为 **PCIe 4.0**，GPU↔主机链路为 Gen 4
（每 GPU 约 25 GB/s）；主机内存 DDR4。无 NVLink。CPU 以机器报告为准（`lscpu`、`/proc/meminfo`）。

## 软件（固定镜像）

**固定镜像已在 4-GPU 目标主机上验证（2026-10-22）。** 补丁整体替换 vLLM 的模型文件
（`PATCH_MODEL_FILES=1`），因此精确的 vLLM 构建至关重要 —— 下面的 digest 就是端到端验证过的那个。
若改用其他构建，请将该选择当实验对待：

- 从补丁编写所针对的 vLLM 版本开始（见 `patches/mimo_v2_hotsplit.py` 文档字符串中的
  期望内部件：`Mxfp4MoEMethod`、`make_mxfp4_moe_kernel`、`GPUModelRunner.execute_model`）。
- 具体镜像（来自补丁编写所针对的 vLLM nightly）：
  `IMAGE="vllm/vllm-openai:nightly-8e2beaa3bff595577a069d49d96bde9074294ec2@sha256:167ccd8696bc09f1b63301185134e04b883623f8adcbd8e3a0ec3140a193abcf"`
  （tag 携带 vLLM 提交；digest 固定精确字节）。
- 启动脚本在启动前检查挂载的 vLLM 源码中存在某些预期方法（SITE 探针），但这是 smoke 测试，
  不是兼容性证明：签名、分片内部实现和 Marlin 内核期望不会自动验证。
- 在 `results/<run-id>/` 记录你验证过的 digest，使下次运行可复现。

**不要**假设随机 tag "能用" —— 先在你的硬件上验证。

## 文件

| 文件 | 说明 |
|---|---|
| `scripts/run-tp4-uva.sh` | 原版 vLLM TP4 + 逐层 UVA 卸载（基线，未改动） |
| `scripts/run-tp4-hotsplit.sh` | 同一服务器 + hotsplit（static 或 dynamic）、NUMA 绑定、实时计数器 |
| `scripts/check-hotsplit-tp4.sh` | 读取容器日志并报告发生了哪些阶段 |
| `scripts/manage.sh` | 日常管理：status / logs / start / stop / restart / update / rollback / check |
| `scripts/preflight-tp4.sh` | 诊断；打印 GPU → NUMA 节点 |
| `patches/hotsplit.py` | TP 感知的 hotsplit，带可选的动态热门集 |
| `patches/hotsplit_plan.py` | numpy 规划辅助（已测试） |
| `patches/numa_bind.py`、`patches/sitecustomize.py` | 将 worker 绑定到其 GPU 的 NUMA 节点；可选启动钩子 |
| `tp4.env.example` | 每个旋钮及其默认值 |

启动脚本从此配方的 `patches/` 挂载 `mimo_v2_hotsplit.py`、`mimo_v2_omni.py` 和 `expert_hist_mix.json`。

## 工作原理

原版 `--cpu-offload-gb` 按层决定驻留：一个层的专家要么全在显存，要么全在主机锁定内存。
hotsplit 改为按专家放置：热门专家在显存，其余在主机锁定内存通过 UVA 读取，每个 MoE 层执行
两次 Marlin 调用（每个库一次，`expert_map` 选择行），部分输出求和。不修改内核。

**在 TP4 下**每个 rank 持有每个专家的切片，因此每个 rank 按相同的热门列表（路由器 id 复制）
和每 rank 字节预算（`HOTSPLIT_HOT_GIB`）划分自己的切片。*不*支持专家并行（EP）（映射假设
全局 id == 行）；如果 EP 处于活动状态，代码拒绝切分。

**在触及任何权重之前的安全门**（见 `tests/test_hotsplit_safety.py`）：

- *按完整模块路径收集层。* 模型可能携带多个 `layers.<n>` 子树（主解码器加草稿模块如
  `model.mtp.layers.0`）。MTP 草稿模块由 `_is_mtp_module()` 分类并保持原样 —— 路由直方图
  只覆盖主解码器。两个主解码器模块声明同一索引会中止运行，而不是悄悄互相覆盖，
  直方图/模型索引集在首次切分之前进行比较。
- *每层 VRAM 预检查。* 热门副本在释放原始权重之前创建，因此瞬态峰值是热门权重
  **+ 热门缩放因子**。大小从真实张量（`_row_bytes`）测量，与带保留的可用 VRAM 比较，
  如果放不下，运行在分配*之前*以诊断错误拒绝。每层的峰值分配/保留内存和释放原始权重后
  返回分配器的字节数会记录日志（`hotsplit mem`、`hotsplit release`），以便首次受控启动
  可以看到真实的瞬态。注意：vLLM 在模型加载之后才计算 KV cache，因此大的热门预算
  之后仍可能导致 KV cache 不足 —— 预检查覆盖切分阶段，而非整个启动过程。

**动态模式（`HOTSPLIT_DYNAMIC=1`，启动脚本默认）。**

- 冷库是每个专家（原始 UVA 张量，不复制任何东西）。热门库是固定地址的 GDDR7 槽缓冲区。
  两个持久 int32 映射 `[层, 专家]` 指示每个专家所在位置（`emap_hot`：槽或 −1，
  `emap_cold`：行或 −1）。
- 图安全计数器在解码时统计路由专家。
- 每 `HOTSPLIT_DYN_STEPS` 个引擎步（默认 2000），在步之间和计算流上，每层将其最弱的
  最多 `HOTSPLIT_DYN_MAX_SWAPS`（默认 2）个热门专家换成最强的冷库专家，仅当
  `score_in > margin × score_out`（margin 1.25）。分数是指数衰减计数，以离线排名播种。
  行从主机→GPU 复制，然后两个映射翻转。流顺序使翻转相对于前向传播是原子的；
  CUDA graphs 只读固定地址映射，因此无需重新捕获。
- 所有 rank 在同一步看到相同计数器，因此做出相同决策。分歧的 rank 仍然正确
  （切分是 rank 内部的），但 TP 等待最慢的 rank。
- 成本：每次更新一次 PCIe 传输停顿（2 次交换 × 69 层 × 每 rank 约 5 MiB ≈ 0.7 GiB，
  PCIe 4.0 速度下数十毫秒；请测量，它记录为 `hotsplit dynamic update`）。
  额外显存：缩放因子的一份热门副本（每层 S 行），因为冷库保留所有缩放因子。
- 如果 step 钩子无法安装（不同的 vLLM 布局），在 `HOTSPLIT_DYNAMIC=1` 时运行**拒绝**
  （静默回退到 static 会使 dynamic-vs-static 比较不可靠）。每个 rank 记录明确的
  `dynamic_hook=ACTIVE` / `dynamic_hook=INACTIVE` 状态。要刻意接受 static 回退，
  设置 `HOTSPLIT_ALLOW_STATIC_FALLBACK=1`；日志随后警告该运行不得用于比较 dynamic 与 static。
  动态计数器在生成期间（而非仅导入时）更新的实时证明仍是实验步骤：生成时观察容器日志中的
  `hotsplit dynamic update` 行。调度器参数在启动时验证（`HOTSPLIT_DYN_STEPS=0` 和其他
  错误值拒绝启动，而不是在运行中途静默失败），重复的规划器失败中止运行
  （`HOTSPLIT_DYN_MAX_PLAN_ERRORS`，默认 3），每次更新记录 hot/cold 命中率行以衡量效果，
  权重交换在专用 CUDA 流上运行并通过事件发布，使 decode-graph 重放无法观察到
  半应用的交换（见 RUNNING.md 中的竞态检查和测量流程）。

这**不做**的事：per-token LRU 未命中填充。Slot-cache 设计（层内融合 Triton 行复制，
可 CUDA-graph 捕获）在同一跟踪上达到比最佳静态 pin 更高的 LRU 命中率，但它是为
TRT-LLM NVFP4 内核编写的，而 copy-engine 变体（每未命中的 DMA 加每层同步）未超越
现有路径。PCIe 上的未命中也比一致性互连贵数倍。如果动态模式表明驻留很重要，
将该设计移植到 Marlin 是显而易见的下一步；此处未完成。本配方中的动态模式是
对工作负载漂移的缓慢适应，是否在你的流量上优于静态 hotsplit 尚未测量。

## 启动

先运行 preflight 并检查四张 GPU 报告合理的 NUMA 节点：

```bash
bash scripts/preflight-tp4.sh
```

首次启动，仅规划（不移动权重，服务器作为原版 UVA 基线运行；每个 rank 应看到一行
`hotsplit plan [rank N/4]`）：

```bash
MODEL_DIR=$HOME/models/MiMo-V2.6-Pro-MOPD HOTSPLIT_DRYRUN=1 MAX_MODEL_LEN=16384 scripts/run-tp4-hotsplit.sh
docker stop mimo-pro-tp4-hotsplit && docker rm mimo-pro-tp4-hotsplit
```

然后真正运行，先 static，后 dynamic：

```bash
MODEL_DIR=$HOME/models/MiMo-V2.6-Pro-MOPD HOTSPLIT_DYNAMIC=0 scripts/run-tp4-hotsplit.sh
scripts/check-hotsplit-tp4.sh
# 停止，然后：
MODEL_DIR=$HOME/models/MiMo-V2.6-Pro-MOPD HOTSPLIT_DYNAMIC=1 scripts/run-tp4-hotsplit.sh
```

不设 `HOTSPLIT_HOT_GIB` 时，热门预算等于原版放置所用的显存字节数（KV 大小不变）。
只有在首次启动显示有多少显存空闲之后，才按 rank 设置（45，然后 50、55）。
按步骤提高 `MAX_MODEL_LEN`（32K、64K、128K），如 `RUNNING.md` 所述。

## 验证

1. `scripts/check-hotsplit-tp4.sh` —— 每个阶段应为 PASS（在 `HOTSPLIT_DYNAMIC=0` 时
   dynamic-hook 行预期 MISS）。
2. 对每个 worker 执行 `numastat -p <worker pid>`：主机锁定内存应位于该 worker 的 GPU 节点上。
   NUMA 绑定是尽力而为（需要 `--cap-add SYS_NICE`，torch 必须暴露 PCI bus id）；
   缺少 `HOTSPLIT_NUMA ... -> node` 行意味着未生效。
3. 质量：teacher-forced logprobs、BFCL 和 flood 固件是正确的基线；均未在 TP4 上运行过。
   不要将贪心字节一致性用作回归测试（在此栈上它也不稳定）。
4. 动态模式：grep `hotsplit dynamic update`；当热门集与流量匹配后交换应减少，
   解码速度不应在每次更新时明显下降。

## 预期（算术，非测量）

MiMo-V2.6-Pro 在 69 个 MoE 层中的每一层从 384 个专家中路由 8 个；按每专家约 19 MiB，
即每个生成 token 跨所有 rank 约 10 GiB 专家权重，每 rank 约 2.6 GiB。每个未从显存
服务的字节都经过 PCIe：在 62% 热门命中率（测得 31% 单元驻留）下，即每 token 每 rank
约 1 GiB，而在每 GPU 约 25 GB/s（本主机运行的 PCIe 4.0 链路）下约每 token 40 毫秒，
即仅传输就带来 20 多 tok/s 的上限。更大的热门预算会降低它；`HOTSPLIT_DRYRUN=1` 打印的
`hotsplit plan` 行给出你的精确预算的离线覆盖率。

## 已知限制与风险

- **已在目标硬件上解决（2026-10-22）：UVA 基线此前输出乱码 - 根因已找到并修复。** 相同权重与参数下的 A/B 测试（`PATCH_MODEL_FILES=1` 对 `0`，`HOTSPLIT_DRYRUN=1`，`MOE_BACKEND=humming`）：补丁后的模型文件生成连贯文本（" Paris. In the early Middle Ages, Paris was the largest city in the western world..."），而镜像自带的 `mimo_v2.py` 生成训练语料片段（"            0     0,   0,  00 "）。根因：原版 `load_weights` 会静默丢失 fused `qkv_proj` 的 weight+`weight_scale_inv` 对（它们可能跨越两次 `AutoWeightsLoader` 调用而被丢弃）；补丁版加载器将这些对保持在一起。`run-tp4-uva.sh` 现在默认挂载修复后的模型文件（`PATCH_MODEL_FILES=1`）。Attention backend `TRITON_ATTN_DIFFKV` 与 `MOE_BACKEND=humming` 已确认可用；检查点权重从未损坏。
- **在目标硬件上已观察到（2026-10-22）：** Marlin 重打包（`prepare_moe_mxfp4_layer_for_marlin` -> `repack_weight`）会创建 `--cpu-offload-params` 不覆盖的新 GPU 张量。使用 `--moe-backend marlin` + UVA 卸载时，重打包的权重在显存中累积，直到 `torch.empty(1.12 GiB)` 因 CUDA OOM 失败（94.97 中已分配 92.7 GiB）。使用 `MOE_BACKEND=triton` 时同一运行更早失败：`ValueError: Mxfp4 MoE backend 'TRITON' does not support ... MoEActivation.SILU activation`（MiMo 使用 gated SiLU）。因此 UVA 基线默认使用 `MOE_BACKEND=humming` —— 此 vLLM 构建中唯一既支持 `MoEActivation.SILU` 又不做 GPU 重打包的后端（Humming 就地读取 MXFP4 权重）。`run-tp4-hotsplit.sh` 现在也默认 `MOE_BACKEND=humming`（marlin + UVA 是已知 OOM，除非 `HOTSPLIT_ALLOW_MARLIN_UVA=1` 否则拒绝启动）。注意：完整 split 的 bank 代码是针对 Marlin 内核写的 —— 在 humming 上 split 是实验，请先验证输出。
- `PATCH_MODEL_FILES=1` 将 `mimo_v2_hotsplit.py` / `mimo_v2_omni.py` 挂载到 vLLM 自己的
  模型文件之上 —— 是整文件替换而非小钩子，因此它与精确 vLLM 构建的兼容性至关重要。
  启动脚本因此在启动前对照 `patches/MANIFEST.sha256` 验证每个挂载的补丁，任何不匹配即拒绝。
  其余两项检查（签名/API 与目标 vLLM 的一致性，以及张量形状/路由/输出的 GPU 集成冒烟测试）
  是在信任补丁之前需在目标主机上运行的实验；两者均列在 RUNNING.md 中。
- `mimo_v2_hotsplit.py` 是针对固定 nightly 镜像的 vLLM `mimo_v2.py` 完整替换。其 fused-fp8-qkv 分片路径（`tp_rank`）已由修复后的加载器在 TP4 下验证（2026-10-22）。`PATCH_MODEL_FILES=0` 保留镜像自带的模型文件 - 这正是输出训练语料片段的配置（见上面“已解决”条目）；仅在需要刻意复现上游 bug 时使用。
  fused-fp8-qkv 分片路径（`tp_rank`）仅在 TP1 上试过。如果 TP4 输出是乱码，
  请用 `PATCH_MODEL_FILES=0` 重试（镜像自带的模型文件，通过 `sitecustomize` 挂钩）以区分两种原因。
- 排名 `expert_hist_mix.json` 来自先前单 GPU 通道上的代理流量；请为你的流量重新排名
  （`live/counts.rank0.json` + `patches/hotsplit_rerank.py`）。动态模式仅在层的固定槽位数内
  自适应；在层间移动槽位需要用新排名重启。
- 计数器包含捕获批次的填充槽位（在 `--max-num-seqs 1` 时可忽略，以上会略有偏差）。
- 4 个 worker 同时锁定数百 GiB 主机内存；请保持 `HOTSPLIT_HOST_FLOOR_GIB` 充足（启动脚本默认 128）。
- 不支持 EP。此处未测试投机解码。

## 回退

`scripts/rollback.sh`（停止并移除 hotsplit 容器；`--start-baseline` 也启动纯 UVA 基线）。
`docker stop mimo-pro-tp4-hotsplit`，然后运行 `scripts/run-tp4-uva.sh`（原版基线），
或在容器中不设置 `HOTSPLIT_COUNTS` 重新启动（编辑启动脚本调用）—— 容器外不做任何修改。
