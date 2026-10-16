# 运行时执行接缝

> [English](README.md) | [Русский](README.ru.md) | **中文**

本配方的*均衡*（哪些专家进显存、哪些进主机锁定内存）与模型无关且完全经过检查：
`hotsplit_plan.py` + `hotsplit_balance.py` + `balance.json`。与模型相关的是权重加载和
vLLM 绑定；见配方 README。
