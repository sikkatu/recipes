# GLM-5.3-Flash — 4× RTX PRO 6000 (96 GB), TP4: баланс экспертов для UVA-выгрузки

> [English](README.md) | **Русский** | [中文](README.zh.md)

**Статус: инженерный расчёт баланса + инструменты. ⚠ Запуск заблокирован на ванильном vLLM.**
Реальный и проверенный здесь — *баланс экспертов* тем же протестированным планировщиком,
что и в эталоне MiMo. Серверная часть следует контракту эталона.

**⚠ Блокер запуска:** `glm5_next` (GLM-5.3-Flash) использует rope-free MLA (`qk_rope_head_dim=0`)
и ванильный vLLM падает на SM120 (RTX PRO 6000): `assert "pe_dim must be 64 for fp8_ds_mla"`
(vLLM issues #53963, #55773). Кандидаты в фиксы — PR #55277 / #55778 / FlashInfer #4802 —
ещё не влиты в upstream. Математика баланса готова и протестирована; запуск заработает
сразу после фикса upstream. Движок — **vLLM, не SGLang** (решение проекта).

Целевой хост: 4× RTX PRO 6000 (96 GB GDDR7 с ECC). Карты PCIe Gen 5, слоты хоста —
**PCIe 4.0** (~25 ГБ/с на GPU); RAM — DDR4. Без NVLink.

| `scripts/manage.sh` | повседневное управление: status / logs / start / stop / restart / update / rollback / check |

## Геометрия и баланс
| Параметр | Значение |
|---|---|
| Слоёв | 42 |
| Маршрутизируемых экспертов | 288 |
| Top-k | 8 |
| Размер эксперта | 13.5 MiB |
| Банк на rank | 39.9 GiB |
| Hot-бюджет | 31 GiB (лестница 26 / 31 / 36) |

Весь банк целиком влезает в память GPU — hot 31 GiB оставляет VRAM под KV cache.

## Запуск (после снятия блокера)
```bash
bash scripts/preflight-tp4.sh
bash scripts/balance.sh
MODEL_DIR=/models/GLM-5.3-Flash HOTSPLIT_DYNAMIC=1 scripts/run-tp4-uva.sh
```

## Проверка
`scripts/check-hotsplit-tp4.sh <container> <dynamic|static>` — каждый rank: `dynamic_hook=ACTIVE`
и `hotsplit done`.

## Известные ограничения
- ⚠ Блокер SM120 (см. выше) — честно заявлен, не скрыт.
- Серверная часть не запускалась на целевом железе.
- EP не поддерживается. MTP-спекуляция не тестировалась.

## Откат
`scripts/rollback.sh` (`--start-baseline` — стоковый UVA).
