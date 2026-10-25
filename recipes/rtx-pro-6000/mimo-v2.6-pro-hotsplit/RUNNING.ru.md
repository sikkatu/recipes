# MiMo-V2.6-Pro на 4× RTX PRO 6000

> [English](RUNNING.md) | **Русский** | [中文](RUNNING.zh.md)

Запускается **vLLM TP4 + выгрузка экспертов UVA + hotsplit**. Проверено на целевом железе:
статика и динамика — 2026-10-23 (3 транзакционных обновления под нагрузкой, hit-rate
41.6->45.4%, 6.6 ток/с). Уровни проверки: запуск и распределение экспертов — ПРОВЕРЕНО;
динамические обновления — ПРОВЕРЕНО под нагрузкой; качество прошло 5-запросный A/B
с UVA-базой (только косметические различия; полная регрессионная оценка ещё впереди).

Требования: Linux, драйвер NVIDIA, Docker + NVIDIA Container Toolkit, четыре видимых GPU,
сотни ГБ свободного места.

Запуск:
```bash
CPU_OFFLOAD_GB=128 scripts/run-tp4-uva.sh
```
Для локальной модели:
```bash
MODEL_DIR=$HOME/models/MiMo-V2.6-Pro-MOPD scripts/run-tp4-uva.sh
```

Модель по умолчанию: `XiaomiMiMo/MiMo-V2.6-Pro-MOPD`. Кэш — `~/.cache/huggingface`.
Контекст 16K с маленьким API-тестом. Увеличивайте контекст (32K, 64K, 128K) только после
успешного старта. `CPU_OFFLOAD_GB` — предел выгрузки на rank, а не резерв RAM.

## Hotsplit (static / dynamic горячие эксперты)
См. README рецепта и [Wiki](https://github.com/sikkatu/recipes/wiki/Quick-Start.ru).

## До доверия PATCH_MODEL_FILES=1
Лаунчер верифицирует каждый патч по `patches/MANIFEST.sha256` до запуска.

## Сравнение dynamic-vs-static
Одинаковые `HOTSPLIT_HOT_GIB`, ревизия и параметры сэмплирования.

## Доказательство, что dynamic работает
1. Старт с `HOTSPLIT_DYNAMIC=1 HOTSPLIT_STRICT_HOOK=1`, каждый rank — `dynamic_hook=ACTIVE`.
2. Сгенерируйте реальный текст, затем `scripts/check-hotsplit-tp4.sh` — счётчик должен расти.
3. Сравнение вывода — со static-запуском как контролем.

## Доказательство NUMA-локальности
`numastat -p <pid>` + `scripts/numastat-report.sh`; сравни preferred / bind / off + H2D-бенч.

## Измерение реальной эффективности
Счётчик обновлений — не доказательство. Измеряйте TTFT/TPOT, PCIe через `nvidia-smi dmon`,
смотрите строку `hotsplit dynamic hit-rate` на каждом обновлении.

## Проверка гонок динамических перестановок (P0-2)
Копии идут на выделенном CUDA-потоке, публикуются через event. Проверка:
`compute-sanitizer --tool racecheck` под нагрузкой; целостность весов обязательна.
