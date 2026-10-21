# GLM-5.3 — 4× RTX PRO 6000（96 GB），TP4：UVA 专家卸载的专家均衡

> [English](README.md) | [Русский](README.ru.md) | **中文**

**状态：工程级均衡计算 + 工具，非经验证的部署运行。** 这里真实的是*专家均衡*，
用与 MiMo 参考配方相同的、有测试的规划器计算。服务端遵循参考契约，但未在目标硬件运行。

目标主机：4× RTX PRO 6000（每卡 96 GB GDDR7 带 ECC）。卡是 PCIe Gen 5，
主机插槽是 **PCIe 4.0**（每 GPU 约 25 GB/s）；内存 DDR4。无 NVLink。

| `scripts/manage.sh` | 日常管理：status / logs / start / stop / restart / update / rollback / check |

## 几何与均衡
| 参数 | 值 |
|---|---|
| 层数 | 75 |
| 路由专家数 | 256 |
| Top-k | 8 |
| 单专家大小 | 20.25 MiB |
| **热门库（显存）** | **~36 GiB**（≈7282 格，38%）—— 锚点来自实测的 7360 槽缓存 |
| 冷库（锁定主机内存） | 58.92 GiB/rank，4 rank 共 235.7 GiB |
| 解码专家流量 | top-8 × 5.06 ≈ **40.5 MiB/每 token/每 rank** |

专家库几乎填满一张卡：热门 36 GiB 给 KV cache 留出显存。

## 启动
```bash
bash scripts/preflight-tp4.sh
bash scripts/balance.sh
MODEL_DIR=$HOME/models/GLM-5.3 HOTSPLIT_DYNAMIC=1 scripts/run-tp4-uva.sh
```

## 验证
`scripts/check-hotsplit-tp4.sh <container> <dynamic|static>` —— 每 rank：`dynamic_hook=ACTIVE`
和 `hotsplit done`。然后冒烟测试 `/v1/chat/completions`。

## 已知限制
- 锚点是实测缓存的字节预算；热门*集合*应来自你的流量直方图。
- 4 个 worker 锁定约 235.7 GiB 内存；主机下限保持宽裕（128 GiB）。
- MTP(2) 投机未在此硬件的任何 hot/cold 切分下测试。
- 服务端未在目标硬件运行。

## 回退
`scripts/rollback.sh`（`--start-baseline` —— 原版 UVA）。
