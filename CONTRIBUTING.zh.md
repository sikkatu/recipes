# 如何贡献配方

> [English](CONTRIBUTING.md) | [Русский](CONTRIBUTING.ru.md) | **中文**

配方是一个承诺：*在这一套硬件上，配这一套软件，这条命令产生这些数字，用这种方法测量。*
这个文件里的一切，都是为了让拥有同样机器的陌生人能够验证这个承诺。

## 契约
每个配方位于 `recipes/<hardware-slug>/<recipe-slug>/`，包含：
| 文件 | 用途 |
|---|---|
| `recipe.yaml` | **`experimental`/`verified`/`deprecated` 必需；`draft` 可省略**（此时索引按 README 的 H1 显示草稿，`check_recipe.py` 禁止草稿 README 中出现非 draft 声明）。机器可读的钉定：模型修订版、镜像 digest、**实测**硬件、启动命令、带出处的指标、关卡、限制、回退。由 `schemas/recipe.schema.json` + `scripts/check_recipe.py` 在 CI 中校验。 |
| `README.md` | 给人看的页面。只写当前状态 —— 不写日记。必需章节：`What this runs`、`Hardware`、`Software`、`Launch`、`Verify`、`Results`、`Known limits`、`Rollback`。 |
| `scripts/` | README 里展示的启动脚本，加健康等待、预热、基准测试、回退。是你们实际运行的东西，不是美化版。 |
| `results/<run-id>/` | 每个已发布数字的出处包：`system.json`（来自 `scripts/collect_system_snapshot.sh`）、镜像 digest、模型修订版、精确启动命令、读取该数字的原始基准/质量输出、精简日志。 |
| `known-limits.md`（可选） | `limits:` 的展开形式，需要解释权衡时使用。 |
| `research/`（可选） | 记录、根因分析、失败实验。鼓励 —— 失败的经验是一半的价值。 |
| `patches/`（可选） | 配方挂载进容器的代码；每个文件在 `recipe.yaml` 中列出其 sha256。 |

## 检查器强制执行的规则
- **不用 `:latest`。** 容器镜像是 `name:tag` 加 `sha256:` digest。
- **模型修订版是 40 字符 commit SHA**，绝不用分支名。
- **硬件以机器报告为准**，不是规格书。`hardware.observed` 必须指向产出它的 `results/<run-id>/system.json`。
- **每个指标都有出处**：`run_id`、`method`，以及数字来源的 `raw` 文件 —— 该文件必须存在。
- **`status: verified` 要求**每个关卡（`schema`、`digest`、`health`、`quality`、`performance`）都带 `last_pass_utc` 和 `run_id`，且至少有一个已发布指标。
- **仓库中任何地方都不得有凭据。** API 密钥在启动时从仓库外的文件读取。
- **补丁与其 sha256 一致。**

## 人工执行的规则（PR 评审）
- 没有负载形状（并发、prompt/output 长度、prompt 类别）的数字不是数字。
- 修复前测量的数字要带 `label:`，README 里也要注明。
- 质量关**按配方声明**。一个配方可以接受 FP8 KV cache；另一个可以禁止。两者都不是“标准” —— 标准是配方声明并测量的东西。
- `Known limits` 在精神上不是可选的：没找到限制说明你没看。
- 回退必须是脚本，不是一句话。

## 状态含义
| 状态 | 含义 |
|---|---|
| `draft` | 编写中；可能无法启动 |
| `experimental` | 能启动且有数字，但某个关卡未测量或存在已知缺陷 |
| `verified` | 五个关卡都在一次记录的运行中通过；拥有同样机器的陌生人应在 ~5% 内复现主数字 |
| `deprecated` | 已被取代或软件已演进；保留存档 |

## 工作流程
```bash
uv venv .venv && uv pip install -p .venv/bin/python jsonschema pyyaml pytest numpy pillow
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/check_recipe.py --all
.venv/bin/python scripts/render_index.py
```
提交 PR；CI 运行同样的命令。评审关注出处，不是文笔。

## 署名
配方建立在他人的内核、引擎和检查点之上。请在配方 README 中以姓名和链接注明所用之物的作者。先测量，再与他们争论。
