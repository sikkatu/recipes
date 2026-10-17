# MiMo-V2.6-Pro — 4× RTX PRO 6000（TP4）：UVA 专家卸载 + hotsplit

> [English](README.md) | [Русский](README.ru.md) | **中文**

**状态：已编写，未在目标硬件上运行。** `patches/` 与 `scripts/run-tp4-hotsplit.sh` 中的
内容尚未在 4-GPU 机器上执行过。纯规划逻辑有单元测试（`tests/test_hotsplit_tp4.py`）；
vLLM/CUDA 部分只做了语法检查。请把首次启动当作实验，并把日志反馈回来。

目标主机：4× RTX PRO 6000 Blackwell Workstation Edition —— 每卡 96 GB GDDR7 带 ECC
（每卡 1792 GB/s）。卡本身是 PCIe Gen 5，但本机主机插槽以 **PCIe 4.0** 训练，
所以 GPU↔主机链路是 Gen 4（每 GPU 约 25 GB/s）；主机内存是 DDR4。无 NVLink。
CPU 以机器实际报告为准（`lscpu`、`/proc/meminfo`）。

## 文件
| 文件 | 说明 |
|---|---|
| `scripts/run-tp4-uva.sh` | 原版 vLLM TP4 + 层级 UVA 卸载（基线，未改动） |
| `scripts/run-tp4-hotsplit.sh` | 同一服务器 + hotsplit，static 或 dynamic，NUMA 绑定，实时计数 |
| `scripts/check-hotsplit-tp4.sh` | 读取容器日志，报告哪些阶段已发生 |
| `scripts/manage.sh` | 日常管理：status / logs / start / stop / restart / update / rollback / check |
| `scripts/preflight-tp4.sh` | 诊断；打印 GPU → NUMA 节点 |
| `patches/hotsplit.py` | TP 感知的 hotsplit，可选动态热门集 |
| `patches/hotsplit_plan.py` | 纯 numpy 规划工具（已测试） |
| `patches/numa_bind.py`、`patches/sitecustomize.py` | 把 worker 绑到其 GPU 的 NUMA 节点；可选启动钩子 |
| `tp4.env.example` | 每个旋钮及其默认值 |

启动脚本会从本配方的 `patches/` 挂载 `mimo_v2_hotsplit.py`、`mimo_v2_omni.py` 和 `expert_hist_mix.json`。

## 工作原理
原版 `--cpu-offload-gb` 按层决定驻留：一层的专家要么全在显存，要么全在锁定的主机内存。
hotsplit 改为按专家粒度重置：热门专家在显存，其余在主机锁定内存经 UVA 读取；
每个 MoE 层执行两次 Marlin 调用（每个库一次，`expert_map` 选行），部分输出求和。不改内核。

**TP4 下**每个 rank 持有每个专家的一个切片，所以每个 rank 按同一列表切分自己的切片
（路由 id 是复制的），带每 rank 字节预算（`HOTSPLIT_HOT_GIB`）。专家并行（EP）**不支持**。

**在动任何权重前的安全门**（见 `tests/test_hotsplit_safety.py`）：
- *按完整模块路径收集层。* 模型可能带多个 `layers.<n>` 子树。MTP 草稿网络由 `_is_mtp_module()` 分类并保持原样。
- *每层 VRAM 预检。* 热门副本在释放原始权重之前创建。

**Dynamic 模式（`HOTSPLIT_DYNAMIC=1`，启动脚本默认）。**
- 冷库是全部专家（原始 UVA 张量）。热库是固定地址的 GDDR7 缓冲。
- 每 `HOTSPLIT_DYN_STEPS` 步，把最弱的热门专家换成最强的冷门专家。
- 参数在启动时校验；规划器连续失败会中止运行。
- 换入换出在专用 CUDA 流上，通过 event 发布（与 replay 图无竞态）。
- 测量流程详情见 [RUNNING.md](RUNNING.md)。

## 启动
先跑 preflight：
```bash
bash scripts/preflight-tp4.sh
```
首次启动仅出计划（不动权重）：
```bash
MODEL_DIR=/models/MiMo-V2.6-Pro-MOPD HOTSPLIT_DRYRUN=1 MAX_MODEL_LEN=16384 scripts/run-tp4-hotsplit.sh
```
然后正式运行，先 static 后 dynamic：
```bash
MODEL_DIR=/models/MiMo-V2.6-Pro-MOPD HOTSPLIT_DYNAMIC=0 scripts/run-tp4-hotsplit.sh
```

## 验证
1. `scripts/check-hotsplit-tp4.sh` —— 每个阶段都应 PASS。
2. `numastat -p <worker pid>` —— 锁定内存应落在该 worker GPU 的节点上。
3. 质量：teacher-forced logprobs、BFCL、flood fixture。
4. Dynamic：grep `hotsplit dynamic update`；换入换出应随时间减少。

## 已知限制与风险
- 未在目标硬件上运行；Marlin MXFP4 与 TP4 切片未验证。
- `PATCH_MODEL_FILES=1` 把 `mimo_v2_hotsplit.py`/`mimo_v2_omni.py` 挂载到 vLLM 自己的模型文件之上 —— 整文件替换，兼容性关键。
- 排序 `expert_hist_mix.json` 来自先前流量；请按你的流量重新排序。
- 4 个 worker 会锁定数百 GiB 内存；`HOTSPLIT_HOST_FLOOR_GIB` 保持宽裕（128）。
- 不支持 EP。投机解码未测试。

## 回退
`scripts/rollback.sh`（停止并移除 hotsplit 容器；`--start-baseline` 也会启动原版 UVA 基线）。
