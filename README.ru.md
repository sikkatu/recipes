# Рецепты для 4× RTX PRO 6000

> **English** | **Русский** | [中文](README.zh.md)

Закреплённые под железо, проверенные рецепты для обслуживания больших open-weight LLM на
**4× RTX PRO 6000 (96 GB GDDR7, TP4, слоты PCIe 4.0 хоста (карты — Gen 5), без NVLink)**.

Рецепт здесь обещает: на *этом* железе (как его видит машина, а не спецификация), с *этим*
digest контейнера и ревизией модели *эта* команда даёт *эти* цифры, измеренные *так* —
и вот сырой вывод, из которого они считались. Рецепты также хранят журналы того, что
не заработало — это половина ценности.

Рецепты под MIT; модели и движки несут свои лицензии и кредит их авторов.

## Индекс
<!-- recipe-index:start -->

| Железо | Рецепт | Модель | Статус | Ключевое | Обновлено |
|---|---|---|---|---|---|
| `rtx-pro-6000` | [DeepSeek-V4.1-Flash — 4× RTX PRO 6000 (96 GB), TP4: баланс экспертов для UVA-выгрузки](recipes/rtx-pro-6000/deepseek-v4.1-flash-hotsplit/) | DeepSeek-V4.1-Flash | 📝 черновик | — | — |
| `rtx-pro-6000` | [GLM-5.3-Flash — 4× RTX PRO 6000 (96 GB), TP4: баланс экспертов для UVA-выгрузки](recipes/rtx-pro-6000/glm-5.3-flash-hotsplit/) | GLM-5.3-Flash | 📝 черновик | — | — |
| `rtx-pro-6000` | [GLM-5.3 — 4× RTX PRO 6000 (96 GB), TP4: баланс экспертов для UVA-выгрузки](recipes/rtx-pro-6000/glm-5.3-hotsplit/) | GLM-5.3 | 📝 черновик | — | — |
| `rtx-pro-6000` | [MiMo-V2.6-Pro — 4× RTX PRO 6000 (TP4): UVA-выгрузка экспертов + hotsplit](recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit/) | MiMo-V2.6-Pro | 📝 черновик | — | — |

<!-- recipe-index:end -->

## О чём этот репозиторий
Четыре рецепта ниже балансируют **резидентность маршрутизируемых экспертов** на рабочей
станции: какие MoE-эксперты сидят в памяти GPU, а какие читаются из закреплённой памяти
хоста по PCIe. Баланс считается одним общим планировщиком с юнит-тестами (жадный горячий
набор по доле маршрутизации на байт), размер подбирается под геометрию модели.

**Полное руководство — в [Wiki](https://github.com/sikkatu/recipes/wiki):** [Быстрый старт](https://github.com/sikkatu/recipes/wiki/Quick-Start.ru) ·
[Как работает балансировка](https://github.com/sikkatu/recipes/wiki/How-Expert-Balance-Works.ru) ·
[Модели и геометрия](https://github.com/sikkatu/recipes/wiki/Models-and-Geometry.ru) · [FAQ](https://github.com/sikkatu/recipes/wiki/FAQ.ru)

## Как устроен рецепт
```
recipes/<hardware>/<recipe>/
  README.md          текущее состояние: что запускает, железо, ПО, запуск, проверка, результаты, ограничения, откат
  RUNNING.md         короткая карточка запуска (черновики)
  balance.json       геометрия экспертов (слои, эксперты, top-k, байт на эксперта, TP)  [рецепты баланса]
  tp4.env.example    каждая ручка со значением по умолчанию
  scripts/           запуск / preflight / balance / check — что реально запускалось
  patches/           код, монтируемый в контейнер; калькулятор баланса и планировщик
  results/<run-id>/  пакет провенанса для каждой опубликованной цифры (после замеров)
```
`hardware/` хранит один профиль на машину с **наблюдаемыми** значениями;
`scripts/collect_system_snapshot.sh` их собирает.

## Контракт, в десяти строках
1. Никакого `:latest` — только digest образов.
2. Ревизии моделей — commit SHA.
3. Железо — то, что говорит `nvidia-smi`.
4. Каждая метрика: значение, единица, форма нагрузки, run-id, метод, сырой файл.
5. Пять ворот — schema, digest, health, quality, performance — с датами прохождения.
6. `verified` означает, что все пять прошли на записанном запуске.
7. Качественные ворота — на рецепт и заявлены; глобального порога нет.
8. Известные ограничения перечислены — иначе вы не смотрели.
9. Откат — это скрипт.
10. Никаких учётных данных в дереве.

См. [CONTRIBUTING.md](CONTRIBUTING.md) для полного контракта и
[schemas/recipe.schema.json](schemas/recipe.schema.json) для полей.

## Локальная проверка
```bash
uv venv .venv && uv pip install -p .venv/bin/python jsonschema pyyaml pytest numpy
.venv/bin/python -m pytest tests -q && .venv/bin/python scripts/check_recipe.py --all
```
