# 用 llm-inference-bench 做配方基准测试

> [English](benchmarking.md) | [Русский](benchmarking.ru.md) | **中文**

配方中来自标准基准的吞吐与质量数据使用同一个固定工具：
[`local-inference-lab/llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench)。
我们自己的工具（knee/replay/teacher-forced divergence/BFCL gate）对配方标题和晋升仍具
权威性。标准基准提供可比、可复现的基线。

## 固定版本
- 按发布标签克隆。在 `results/<run-id>/bench-version.txt` 记录 `tag @ commit`。
- 用 `LLM_BENCH_NO_UPDATE_CHECK=1` 和 `--display-mode plain` 运行。
- 除非要上传到他们的榜单，否则不要用 `lil-bench`（容器包装器）。
- 上游仓库**没有许可文件**。不要把其代码或数据集复制进本仓库。

## 层级
| 层级 | 时机 | 命令形式 | 耗时 |
|---|---|---|---|
| T0 gate | 每次启动 | `/v1/models` id、精确词冒烟、`tool_calls[]` 冒烟 | ~2 分钟 |
| T1 perf | 任何配方变更 | `--concurrency 1,4,8 --contexts 0,32768,131072 --duration 30` | ~10 分钟 |
| T2 quality | 引擎/内核/量化/投机变更 | `--test-profile needle-checksum`；`--test-profile gsm8k --profile-runs 200` | ~10–60 分钟 |
| T3 promotion | 新默认路由前或硬件变更后 | 完整矩阵、完整 GSM8K + MMLU-Pro | 数小时 |

**晋升门槛：** T2 上配对 McNemar p < 0.05 的精度*下降*即失败。
Needle-checksum 必须保持 EXACT N/N。T1 必须在上次晋升运行的波动范围内。

## 标注规则（检查器不强制；评审者执行）
1. T1 行注明 **“llm-inference-bench synthetic filler”**、上下文和并发。
2. 把基准窗口内的 vLLM `Running:`/`Waiting:` 日志保存为 `<run>.engine-load.txt`。
3. 记录基准撞到的任何服务端上限（如 `--max-num-seqs`）。
4. 提交前清理基准 JSON 中的局域网 IP 和本地路径。
