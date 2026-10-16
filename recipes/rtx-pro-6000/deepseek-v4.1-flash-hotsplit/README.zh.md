# DeepSeek-V4.1-Flash — 4× RTX PRO 6000（96 GB），TP4：UVA 专家卸载的专家均衡

> [English](README.md) | [Русский](README.ru.md) | **中文**

**状态：工程级均衡计算 + 工具，非经验证的部署运行。** 这里真实且经过检查的是*专家均衡*：
哪些路由专家该放在显存、哪些放在主机锁定内存，以及每 rank 字节预算 —— 用与 MiMo 参考
配方完全相同的、有单元测试的规划器计算。服务端部分（启动脚本、模型补丁）遵循参考契约，
但尚未在目标硬件上运行。

目标主机：4× RTX PRO 6000 Blackwell Workstation Edition —— 每卡 96 GB GDDR7 带 ECC。
卡是 PCIe Gen 5，但主机插槽是 **PCIe 4.0**，GPU↔主机链路约 25 GB/s；
主机内存 DDR4。无 NVLink。

## 模型几何
| 参数 | 值 |
|---|---|
| 层数 | 40 |
| 路由专家数 | 384 |
| Top-k | 6 |
| 单专家大小 | 17.9 MiB |
| 每 rank 专家库 | 67.1 GiB |
| 热门预算 | 52 GiB/rank（梯度 47 / 52 / 57） |
| 热门锚点 | 384 中的 295 |

## 均衡
`balance.json` 保存几何；`patches/hotsplit_plan.py`（numpy，已测试）按“路由占比 / 字节数”
贪心选热门集。校验：NaN/负数/不匹配计数报错，预算溢出报错，预言机确认最优。

## 启动
```bash
bash scripts/preflight-tp4.sh
bash scripts/balance.sh                 # 按你的流量直方图
MODEL_DIR=/models/DeepSeek-V4.1-Flash HOTSPLIT_DYNAMIC=1 scripts/run-tp4-uva.sh
```
热门预算通过 `HOTSPLIT_HOT_GIB`（每 rank）设置。所有旋钮见 `tp4.env.example`。

## 验证
`scripts/check-hotsplit-tp4.sh <container> <dynamic|static>` —— 每个 rank 必须报告
`dynamic_hook=ACTIVE`（dynamic）和 `hotsplit done`。然后冒烟测试 `/v1/chat/completions`。

## 已知限制
- 服务端未在目标硬件运行；请验证 Marlin MXFP4 与 TP4 切片。
- `expert_hist_mix.json` 是参考排序；请按你的流量重新排序。
- 不支持 EP。投机解码未测试。

## 回退
`scripts/rollback.sh`（停止并移除容器；`--start-baseline` 启动原版 UVA）。
