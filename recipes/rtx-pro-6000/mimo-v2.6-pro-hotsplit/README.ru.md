# MiMo-V2.6-Pro — 4× RTX PRO 6000 (TP4): UVA-выгрузка экспертов + hotsplit

> [English](README.md) | **Русский** | [中文](README.zh.md)

**Статус: написано, не запускалось на целевом железе.** Ничто в `patches/` или
`scripts/run-tp4-hotsplit.sh` не исполнялось на 4-GPU машине. Чистая логика планирования
покрыта юнит-тестами (`tests/test_hotsplit_tp4.py`); части vLLM/CUDA только проверены
синтаксически. Относитесь к первому запуску как к эксперименту и пришлите логи.

Целевой хост: 4× RTX PRO 6000 Blackwell Workstation Edition — по 96 GB GDDR7 с ECC
(1792 ГБ/с на карту). Карты PCIe Gen 5, но на этой машине слоты хоста обучаются как
**PCIe 4.0**, поэтому линия GPU↔хост — Gen 4 (~25 ГБ/с на GPU); RAM хоста — DDR4. Без NVLink.
CPU — то, что сообщает машина (`lscpu`, `/proc/meminfo`).

## Файлы
| файл | что |
|---|---|
| `scripts/run-tp4-uva.sh` | стоковый vLLM TP4 + UVA-выгрузка на уровне слоёв (базовый, без изменений) |
| `scripts/run-tp4-hotsplit.sh` | тот же сервер + hotsplit, static или dynamic, NUMA-закрепление, живые счётчики |
| `scripts/check-hotsplit-tp4.sh` | читает логи контейнера и говорит, какие стадии произошли |
| `scripts/manage.sh` | повседневное управление: status / logs / start / stop / restart / update / rollback / check |
| `scripts/preflight-tp4.sh` | диагностика; печатает GPU → NUMA-узел |
| `patches/hotsplit.py` | TP-осведомлённый hotsplit с опциональным dynamic hot-набором |
| `patches/hotsplit_plan.py` | планировочные хелперы на numpy (протестированы) |
| `patches/numa_bind.py`, `patches/sitecustomize.py` | закрепление воркера за NUMA-узлом его GPU; опциональный boot-хук |
| `tp4.env.example` | каждая ручка со значением по умолчанию |

Лаунчер монтирует `mimo_v2_hotsplit.py`, `mimo_v2_omni.py` и `expert_hist_mix.json` из `patches/` этого рецепта.

## Как это работает
Стоковый `--cpu-offload-gb` решает резидентность по слоям: эксперты слоя либо все в памяти GPU,
либо все в закреплённой памяти хоста. hotsplit переселяет их поэкспертно: горячие — в память GPU,
остальные — в закреплённую память хоста через UVA; каждый MoE-слой выполняет два запуска Marlin
(по одному на банк, `expert_map` выбирает строки), частичные выходы суммируются. Без правок ядер.

**Под TP4** каждый rank держит срез каждого эксперта, поэтому каждый rank делит свой срез по
одному и тому же списку (id роутера реплицирован) с бюджетом байт на rank (`HOTSPLIT_HOT_GIB`).
Экспертный параллелизм **не поддерживается** (карты предполагают глобальный id == строка);
код отказывается делить, если EP активен.

**Сторожевые ворота до любого трогания весов** (см. `tests/test_hotsplit_safety.py`):
- *Сбор слоёв по полному пути модуля.* Модель может нести несколько поддеревьев `layers.<n>`.
  MTP-черновики классифицируются `_is_mtp_module()` и остаются стоковыми.
- *VRAM pre-check на слой.* Горячая копия делается до освобождения оригиналов.

**Dynamic-режим (`HOTSPLIT_DYNAMIC=1`, по умолчанию в лаунчере).**
- Холодный банк — все эксперты (оригинальные UVA-тензоры). Горячий — фиксированный буфер GDDR7.
- Каждые `HOTSPLIT_DYN_STEPS` шагов слабейшие горячие меняются на сильнейшие холодные.
- Параметры валидируются на старте; повторные ошибки планировщика прерывают запуск.
- Перестановки на выделенном CUDA-потоке, публикуются через event (без гонок с replay-графом).
- Подробности процедур измерений — в [RUNNING.md](RUNNING.md).

## Запуск
Сначала preflight:
```bash
bash scripts/preflight-tp4.sh
```
Первый запуск — только план (веса не трогаются):
```bash
MODEL_DIR=$HOME/models/MiMo-V2.6-Pro-MOPD HOTSPLIT_DRYRUN=1 MAX_MODEL_LEN=16384 scripts/run-tp4-hotsplit.sh
```
Затем по-настоящему, сначала static, потом dynamic:
```bash
MODEL_DIR=$HOME/models/MiMo-V2.6-Pro-MOPD HOTSPLIT_DYNAMIC=0 scripts/run-tp4-hotsplit.sh
```

## Проверка
1. `scripts/check-hotsplit-tp4.sh` — каждая стадия должна PASS.
2. `numastat -p <worker pid>` — закреплённая память на узле GPU воркера.
3. Качество: teacher-forced logprobs, BFCL, flood fixture.
4. Dynamic: grep `hotsplit dynamic update`; перестановки должны уменьшаться.

## Известные ограничения и риски
- Не запускалось на целевом железе; Marlin MXFP4 с TP4-шардами не валидирован.
- `PATCH_MODEL_FILES=1` монтирует `mimo_v2_hotsplit.py`/`mimo_v2_omni.py` поверх файлов vLLM —
  замена целого файла, совместимость с конкретной сборкой vLLM критична.
- Ранжирование `expert_hist_mix.json` — из предыдущего трафика; переранжируйте под свой.
- 4 воркера закрепляют сотни GiB RAM; держите `HOTSPLIT_HOST_FLOOR_GIB` щедрым (128).
- EP не поддерживается. Спекулятивное декодирование не тестировалось.

## Откат
`scripts/rollback.sh` (останавливает+удаляет контейнер hotsplit; `--start-baseline` также
запускает стоковый UVA-базовый вариант).
