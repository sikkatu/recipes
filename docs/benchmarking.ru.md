# Бенчмаркинг рецептов с llm-inference-bench

> [English](benchmarking.md) | **Русский** | [中文](benchmarking.zh.md)

Строки throughput и качества в рецептах, взятые из стандартного бенча, используют один
закреплённый инструмент: [`local-inference-lab/llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench).
Наши собственные инструменты (knee/replay/teacher-forced divergence/BFCL gate) остаются
авторитетными для заголовков рецептов и продвижения. Стандартный бенч даёт сопоставимые,
повторяемые базовые линии.

## Закрепление
- Клонируйте по тегу релиза. Записывайте `tag @ commit` в `results/<run-id>/bench-version.txt`.
- Запускайте с `LLM_BENCH_NO_UPDATE_CHECK=1` и `--display-mode plain`.
- Не используйте `lil-bench` (контейнерная обёртка), если не хотите загружать результаты к ним.
- В upstream-репозитории **нет файла лицензии**. Не копируйте его код или датасеты в наш репозиторий.

## Уровни
| Уровень | Когда | Форма команды | Время |
|---|---|---|---|
| T0 gate | каждый запуск | `/v1/models` id, smoke точных слов, smoke `tool_calls[]` | ~2 мин |
| T1 perf | любое изменение рецепта | `--concurrency 1,4,8 --contexts 0,32768,131072 --duration 30` | ~10 мин |
| T2 quality | изменение движка / ядра / квантизации / спекуляции | `--test-profile needle-checksum`; `--test-profile gsm8k --profile-runs 200` | ~10–60 мин |
| T3 promotion | перед новым маршрутом или после изменения железа | полная матрица, полный GSM8K + MMLU-Pro | часы |

**Порог продвижения:** парная точность McNemar p < 0.05 *потеря* на T2 — провал.
Needle-checksum должен оставаться EXACT N/N. T1 должен быть в пределах разброса запусков.

## Правила маркировки (чекер не проверяет; проверяют ревьюеры)
1. Строка T1 указывает **«llm-inference-bench synthetic filler»**, контекст и конкурентность.
2. Сохраняйте строки лога vLLM `Running:`/`Waiting:` из окна бенча как `<run>.engine-load.txt`.
3. Отмечайте любой серверный лимит, упёршийся в бенч (например `--max-num-seqs`).
4. Вычищайте LAN IP и локальные пути из JSON бенча перед коммитом.
