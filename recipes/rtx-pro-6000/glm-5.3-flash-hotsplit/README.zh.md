# GLM-5.3-Flash — 4× RTX PRO 6000（96 GB），TP4：UVA 专家卸载的专家均衡

> [English](README.md) | [Русский](README.ru.md) | **中文**

**状态：工程级均衡计算 + 工具。⚠ 启动在原版 vLLM 上被阻塞。**
这里真实且经过测试的是*专家均衡*，用与 MiMo 参考配方相同的规划器计算。服务端遵循参考契约。

**⚠ 启动阻塞项：** `glm5_next`（GLM-5.3-Flash）使用无 rope 的 MLA（`qk_rope_head_dim=0`），
原版 vLLM 在 SM120（RTX PRO 6000）上崩溃：`assert "pe_dim must be 64 for fp8_ds_mla"`
（vLLM issues #53963、#55773）。修复候选 PR #55277 / #55778 / FlashInfer #4802
尚未合入上游。均衡数学已就绪并经测试；上游修复一到即可启动。引擎是 **vLLM，不是 SGLang**
（项目决定）。

目标主机：4× RTX PRO 6000（每卡 96 GB GDDR7 带 ECC）。卡是 PCIe Gen 5，
主机插槽是 **PCIe 4.0**（每 GPU 约 25 GB/s）；内存 DDR4。无 NVLink。

| `scripts/manage.sh` | 日常管理：status / logs / start / stop / restart / update / rollback / check |

## 几何与均衡
| 参数 | 值 |
|---|---|
| 层数 | 42 |
| 路由专家数 | 288 |
| Top-k | 8 |
| 单专家大小 | 13.5 MiB |
| 每 rank 专家库 | 39.9 GiB |
| 热门预算 | 31 GiB（梯度 26 / 31 / 36） |

整个专家库能装进显存 —— 热门 31 GiB 给 KV cache 留出空间。

## 启动（解除阻塞后）
```bash
bash scripts/preflight-tp4.sh
bash scripts/balance.sh
MODEL_DIR=/models/GLM-5.3-Flash HOTSPLIT_DYNAMIC=1 scripts/run-tp4-uva.sh
```

## 验证
`scripts/check-hotsplit-tp4.sh <container> <dynamic|static>` —— 每 rank：`dynamic_hook=ACTIVE`
和 `hotsplit done`。

## 已知限制
- ⚠ SM120 阻塞项（见上）—— 如实声明，不隐瞒。
- 服务端未在目标硬件运行。
- 不支持 EP。MTP 投机未测试。

## 回退
`scripts/rollback.sh`（`--start-baseline` —— 原版 UVA）。
