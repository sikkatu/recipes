# 4× RTX PRO 6000 配方集

> [English](README.md) | [Русский](README.ru.md) | **中文**

面向 **4× RTX PRO 6000（96 GB GDDR7，TP4，主机 PCIe 4.0 插槽（卡本身是 Gen 5），无 NVLink）**
的硬件锁定、经验证的大型开源 LLM 部署配方。

这里的配方承诺：在*这台*硬件上（以机器实际报告为准，不是规格书），用*这个*容器 digest
和模型修订版，*这条*命令产生*这些*数字，用*这种方法*测量 —— 附上读取这些数字的原始输出。
配方还保留失败尝试的记录，那是一半的价值。

配方以 MIT 发布；模型和引擎保留各自的许可和原作者署名。

## 索引
<!-- recipe-index:start -->

| 硬件 | 配方 | 模型 | 状态 | 要点 | 更新 |
|---|---|---|---|---|---|
| `rtx-pro-6000` | [DeepSeek-V4.1-Flash — 4× RTX PRO 6000（96 GB），TP4：UVA 专家卸载的专家均衡](recipes/rtx-pro-6000/deepseek-v4.1-flash-hotsplit/) | DeepSeek-V4.1-Flash | 📝 草稿 | — | — |
| `rtx-pro-6000` | [GLM-5.3-Flash — 4× RTX PRO 6000（96 GB），TP4：UVA 专家卸载的专家均衡](recipes/rtx-pro-6000/glm-5.3-flash-hotsplit/) | GLM-5.3-Flash | 📝 草稿 | — | — |
| `rtx-pro-6000` | [GLM-5.3 — 4× RTX PRO 6000（96 GB），TP4：UVA 专家卸载的专家均衡](recipes/rtx-pro-6000/glm-5.3-hotsplit/) | GLM-5.3 | 📝 草稿 | — | — |
| `rtx-pro-6000` | [MiMo-V2.6-Pro — 4× RTX PRO 6000（TP4）：UVA 专家卸载 + hotsplit](recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit/) | MiMo-V2.6-Pro | ✅ 已验证 | — | — |

<!-- recipe-index:end -->

## 这个仓库是做什么的
下面四个配方平衡工作站上的**路由专家驻留**：哪些 MoE 专家留在显存，哪些通过 PCIe
从主机锁定内存读取。平衡由一个共享的、有单元测试的规划器计算（按“路由占比 / 字节数”
贪心选热门集），按模型几何结构定制。

**完整指南见 [Wiki](https://github.com/sikkatu/recipes/wiki)：**
[快速上手](https://github.com/sikkatu/recipes/wiki/Quick-Start.zh) ·
[专家均衡原理](https://github.com/sikkatu/recipes/wiki/How-Expert-Balance-Works.zh) ·
[模型与几何结构](https://github.com/sikkatu/recipes/wiki/Models-and-Geometry.zh) ·
[常见问题](https://github.com/sikkatu/recipes/wiki/FAQ.zh)

## 配方的结构
```
recipes/<hardware>/<recipe>/
  README.md          当前状态：运行什么、硬件、软件、启动、验证、结果、已知限制、回退
  balance.json       专家几何（层数、专家数、top-k、每专家字节数、TP）  [均衡配方]
  tp4.env.example    每个旋钮及其默认值
  scripts/           启动 / 预检 / 均衡 / 检查 —— 实际运行过的内容
  patches/           挂载进容器的代码；均衡计算器和规划器
```

## 契约

1. 不用 `:latest` —— 始终使用镜像 digest。
2. 模型修订版是 commit SHA。
3. 硬件以 `nvidia-smi` 报告为准 —— 不是规格表。
4. 状态标注诚实：`verified` 表示配方在记录的主机上端到端跑通；未运行过的配方如实说明。
5. 已知限制必须列出 —— 否则你没看。
6. 回退是一个脚本。
7. 仓库中没有任何凭据。

## 署名

这些配方是大量真实硬件实操的成果。如果你在自己的工作中使用本仓库 —— 配方、测量、分析或文档 —— **请注明出处**：在发布基于本仓库的结果时链接 `https://github.com/sikkatu/recipes`（以及具体配方页面）。Fork 和衍生文档必须保留该引用。欢迎提出问题和修正；不接受无声明的照搬。

## 本地验证
```bash
uv venv .venv && uv pip install -p .venv/bin/python jsonschema pyyaml pytest numpy
.venv/bin/python -m pytest tests -q && .venv/bin/python scripts/check_recipe.py --all
```
